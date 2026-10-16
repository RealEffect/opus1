//! The FFT and the MDCT against the same functions inside C libopus.
//!
//! libopus is linked as a static library, so its internal functions and its
//! built-in mode can be reached directly.
//!
//! The tables of the FFT and the rotation table of the MDCT must be equal to
//! those of libopus. The FFT and the MDCT must give the bits of
//! `opus_fft_impl`, `clt_mdct_forward_c` and `clt_mdct_backward_c` of libopus
//! where the library is known to be the arm64 build of clang. Elsewhere the
//! tests print how many output values differ (run with `--nocapture`).

// The loops follow the C code, which steps through its arrays in chunks.
#![allow(clippy::chunks_exact_to_as_chunks)]

use opus1::kiss_fft::{
    FFT_TWIDDLES_480, KissCpx, KissFftState, opus_fft_impl, opus_fft_impl_fused,
    opus_fft_impl_scalar,
};
use opus1::modes::default_mode;
use std::os::raw::{c_int, c_void};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CCpx {
    r: f32,
    i: f32,
}

/// `kiss_fft_state` of the float build (celt/kiss_fft.h).
#[repr(C)]
struct CFftState {
    nfft: c_int,
    scale: f32,
    shift: c_int,
    factors: [i16; 16],
    bitrev: *const i16,
    twiddles: *const CCpx,
    arch_fft: *const c_void,
}

/// `mdct_lookup` (celt/mdct.h).
#[repr(C)]
struct CMdctLookup {
    n: c_int,
    maxshift: c_int,
    kfft: [*const CFftState; 4],
    trig: *const f32,
}

/// The start of `OpusCustomMode` of the float build (celt/modes.h), up to
/// and including the MDCT tables.
#[repr(C)]
struct CMode {
    fs: i32,
    overlap: c_int,
    nb_ebands: c_int,
    eff_ebands: c_int,
    preemph: [f32; 4],
    e_bands: *const i16,
    max_lm: c_int,
    nb_short_mdcts: c_int,
    short_mdct_size: c_int,
    nb_alloc_vectors: c_int,
    alloc_vectors: *const u8,
    log_n: *const i16,
    window: *const f32,
    mdct: CMdctLookup,
}

unsafe extern "C" {
    fn opus_custom_mode_create(fs: i32, frame_size: c_int, error: *mut c_int) -> *const CMode;
    #[link_name = "opus_fft_impl"]
    fn c_opus_fft_impl(st: *const CFftState, fout: *mut CCpx);
    fn clt_mdct_forward_c(
        l: *const CMdctLookup,
        input: *mut f32,
        out: *mut f32,
        window: *const f32,
        overlap: c_int,
        shift: c_int,
        stride: c_int,
        arch: c_int,
    );
    fn clt_mdct_backward_c(
        l: *const CMdctLookup,
        input: *mut f32,
        out: *mut f32,
        window: *const f32,
        overlap: c_int,
        shift: c_int,
        stride: c_int,
        arch: c_int,
    );
}

/// The built-in 48 kHz mode of libopus.
fn c_mode() -> &'static CMode {
    // Makes sure the test binary links libopus: the functions above are
    // found in the static library that `audiopus_sys` brings along.
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
    let mut error: c_int = -1;
    // SAFETY: for 48 kHz and 960 samples libopus returns a pointer to its
    // constant built-in mode.
    let mode = unsafe { &*opus_custom_mode_create(48000, 960, &mut error) };
    assert_eq!(error, 0);
    // These values show that the struct above has the layout of the library.
    assert_eq!(mode.fs, 48000);
    assert_eq!(mode.overlap, 120);
    assert_eq!(mode.max_lm, 3);
    assert_eq!(mode.short_mdct_size, 120);
    assert_eq!(mode.mdct.n, 1920);
    assert_eq!(mode.mdct.maxshift, 3);
    for (shift, nfft) in [480, 240, 120, 60].into_iter().enumerate() {
        let st = unsafe { &*mode.mdct.kfft[shift] };
        assert_eq!(st.nfft, nfft);
        assert_eq!(st.shift, if shift == 0 { -1 } else { shift as c_int });
    }
    mode
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

/// Number of values that differ in their bits, and the largest difference
/// relative to the largest value of `want`.
fn compare(got: &[f32], want: &[f32]) -> (usize, f32) {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut count = 0;
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(want) {
        if g.to_bits() != w.to_bits() {
            count += 1;
            worst = worst.max((g - w).abs() / peak);
        }
    }
    (count, worst)
}

/// The twiddle table, the bit-reversal tables and the scale factors of the
/// FFT are those of the built-in mode of libopus.
#[test]
fn fft_tables_are_those_of_libopus() {
    let mode = c_mode();
    let base = KissFftState::new(480).unwrap();
    for (shift, nfft) in [480usize, 240, 120, 60].into_iter().enumerate() {
        let st = unsafe { &*mode.mdct.kfft[shift] };
        let sub;
        let plan = if shift == 0 {
            &base
        } else {
            sub = KissFftState::new_sub(&base, nfft).unwrap();
            &sub
        };
        let c_bitrev = unsafe { core::slice::from_raw_parts(st.bitrev, nfft) };
        assert_eq!(plan.bitrev.as_slice(), c_bitrev, "bitrev {nfft}");
        assert_eq!(plan.scale().to_bits(), st.scale.to_bits(), "scale {nfft}");
        // All four sizes of libopus share one table of 480 entries.
        let c_tw = unsafe { core::slice::from_raw_parts(st.twiddles, 480) };
        for (i, (r, c)) in FFT_TWIDDLES_480.iter().zip(c_tw).enumerate() {
            assert_eq!(r.r.to_bits(), c.r.to_bits(), "twiddle {i}, real part");
            assert_eq!(r.i.to_bits(), c.i.to_bits(), "twiddle {i}, imaginary part");
        }
    }
}

/// The window and the rotation table of the MDCT are those of the built-in
/// mode of libopus.
///
/// The window had one other value until it was compared here: value 1 was
/// 0.0006055135 (0x3a1ebb53) where libopus celt/static_modes_float.h has
/// 0.00060551348f (0x3a1ebb52).
#[test]
fn mdct_tables_are_those_of_libopus() {
    let mode = c_mode();
    let rust = default_mode();

    let window = unsafe { core::slice::from_raw_parts(mode.window, 120) };
    assert_eq!(rust.window.len(), window.len());
    for (i, (r, c)) in rust.window.iter().zip(window).enumerate() {
        assert_eq!(r.to_bits(), c.to_bits(), "window, value {i}: opus1 {r:e}, libopus {c:e}");
    }

    let mut offset = 0;
    for shift in 0..=3usize {
        let n2 = (1920 >> shift) / 2;
        let c_trig = unsafe { core::slice::from_raw_parts(mode.mdct.trig.add(offset), n2) };
        let r_trig = rust.mdct.get_trig_debug(shift);
        assert_eq!(r_trig.len(), n2);
        for (i, (r, c)) in r_trig.iter().zip(c_trig).enumerate() {
            assert_eq!(r.to_bits(), c.to_bits(), "rotation table, shift {shift}, value {i}");
        }
        offset += n2;
    }
}

/// The radix-5 butterfly adds in the order of libopus (kiss_fft.c:289-302).
///
/// The test runs a 5-point FFT, which is one radix-5 butterfly with the
/// twiddle 1. The first input is any number. The components of the other
/// four are powers of two, chosen so that their sums and differences are
/// powers of two or zero as well. Then every product in the butterfly is
/// exact: it rounds the same whether or not the library fuses it with a sum,
/// and only the order of the sums decides the bits.
#[test]
fn radix5_sums_are_in_the_order_of_libopus() {
    c_mode();
    let plan = KissFftState::new(5).unwrap();

    // A 5-point plan of libopus, built by hand. The butterfly reads its
    // constants from entries 1 and 2 of the twiddle table.
    let ya = FFT_TWIDDLES_480[96];
    let yb = FFT_TWIDDLES_480[192];
    let twiddles = [
        CCpx { r: 1.0, i: -0.0 },
        CCpx { r: ya.r, i: ya.i },
        CCpx { r: yb.r, i: yb.i },
        CCpx { r: yb.r, i: -yb.i },
        CCpx { r: ya.r, i: -ya.i },
    ];
    let bitrev = [0i16, 1, 2, 3, 4];
    let mut factors = [0i16; 16];
    factors[0] = 5;
    factors[1] = 1;
    let c_plan = CFftState {
        nfft: 5,
        scale: 0.2,
        shift: -1,
        factors,
        bitrev: bitrev.as_ptr(),
        twiddles: twiddles.as_ptr(),
        arch_fft: core::ptr::null(),
    };

    let mut rng = Lcg(0x11b0_0005);
    let mut old_order_differs = 0;
    for round in 0..4000 {
        let mut x = [CCpx::default(); 5];
        let k0 = (rng.next_u32() % 17) as i32 - 8;
        let scale = 2f32.powi(k0);
        x[0] = CCpx { r: rng.unit() * scale, i: rng.unit() * scale };
        // In every second round the powers of two are as large as the last
        // bit of the first input or a little smaller. That is where the
        // order of the sums matters most. In the other rounds they have any
        // size.
        let near = round % 2 == 0;
        let power = |rng: &mut Lcg| {
            let e = if near {
                k0 - 24 - (rng.next_u32() % 4) as i32
            } else {
                (rng.next_u32() % 41) as i32 - 20
            };
            let v = 2f32.powi(e);
            if rng.next_u32() & 1 == 0 { v } else { -v }
        };
        for k in [1usize, 2] {
            x[k] = CCpx { r: power(&mut rng), i: power(&mut rng) };
            // The partner is 0, the same value or its negative, so the sum
            // and the difference of the pair are powers of two or zero.
            let partner = |rng: &mut Lcg, v: f32| match rng.next_u32() % 3 {
                0 => 0.0,
                1 => v,
                _ => -v,
            };
            x[5 - k] = CCpx { r: partner(&mut rng, x[k].r), i: partner(&mut rng, x[k].i) };
        }

        let mut c = x;
        unsafe { c_opus_fft_impl(&c_plan, c.as_mut_ptr()) };

        let mut portable: Vec<KissCpx> = x.iter().map(|v| KissCpx::new(v.r, v.i)).collect();
        let mut fast = portable.clone();
        opus_fft_impl_scalar(&plan, &mut portable);
        opus_fft_impl(&plan, &mut fast);
        for k in 0..5 {
            assert_eq!(
                (portable[k].r.to_bits(), portable[k].i.to_bits()),
                (c[k].r.to_bits(), c[k].i.to_bits()),
                "portable FFT, round {round}, output {k}, input {x:?}"
            );
            assert_eq!(
                (fast[k].r.to_bits(), fast[k].i.to_bits()),
                (c[k].r.to_bits(), c[k].i.to_bits()),
                "FFT, round {round}, output {k}, input {x:?}"
            );
        }

        // The first output with the sample added first, as opus1 had it.
        // It must differ from libopus for some inputs, or this test could
        // not tell the two orders apart.
        let old_r = (x[0].r + (x[1].r + x[4].r)) + (x[2].r + x[3].r);
        let old_i = (x[0].i + (x[1].i + x[4].i)) + (x[2].i + x[3].i);
        if old_r.to_bits() != c[0].r.to_bits() || old_i.to_bits() != c[0].i.to_bits() {
            old_order_differs += 1;
        }
    }
    println!("the old order of the sums gives another first output in {old_order_differs} of 4000 cases");
    assert!(old_order_differs > 200);
}

// ---------------------------------------------------------------------------
// The arithmetic of libopus as clang builds it for arm64.
//
// With its default setting (`-ffp-contract=on`) clang joins a product and a
// sum into one fused multiply-add when both are in the same C expression. Of
// two products it fuses the one on the left; the other one is rounded first.
// The functions below are celt/kiss_fft.c and celt/mdct.c (float build) with
// exactly those products written as `mul_add`.
// ---------------------------------------------------------------------------

mod model {
    use super::{CCpx, CFftState};

    /// `C_MUL` (celt/_kiss_fft_guts.h:112-114).
    fn c_mul(a: CCpx, b: CCpx) -> CCpx {
        CCpx {
            r: a.r.mul_add(b.r, -(a.i * b.i)),
            i: a.r.mul_add(b.i, a.i * b.r),
        }
    }

    fn c_add(a: CCpx, b: CCpx) -> CCpx {
        CCpx { r: a.r + b.r, i: a.i + b.i }
    }

    fn c_sub(a: CCpx, b: CCpx) -> CCpx {
        CCpx { r: a.r - b.r, i: a.i - b.i }
    }

    /// `kf_bfly2` (kiss_fft.c:76-105). No product shares an expression with
    /// a sum, so nothing is fused.
    fn bfly2(f: &mut [CCpx], m: usize, n: usize) {
        assert_eq!(m, 4);
        let tw = 0.707_106_77_f32;
        for g in f.chunks_exact_mut(8).take(n) {
            let t = g[4];
            g[4] = c_sub(g[0], t);
            g[0] = c_add(g[0], t);

            let t = CCpx { r: (g[5].r + g[5].i) * tw, i: (g[5].i - g[5].r) * tw };
            g[5] = c_sub(g[1], t);
            g[1] = c_add(g[1], t);

            let t = CCpx { r: g[6].i, i: -g[6].r };
            g[6] = c_sub(g[2], t);
            g[2] = c_add(g[2], t);

            let t = CCpx { r: (g[7].i - g[7].r) * tw, i: (-(g[7].i + g[7].r)) * tw };
            g[7] = c_sub(g[3], t);
            g[3] = c_add(g[3], t);
        }
    }

    /// `kf_bfly4` (kiss_fft.c:108-175).
    fn bfly4(f: &mut [CCpx], fstride: usize, tw: &[CCpx], m: usize, n: usize, mm: usize) {
        if m == 1 {
            for g in f.chunks_exact_mut(4).take(n) {
                let scratch0 = c_sub(g[0], g[2]);
                g[0] = c_add(g[0], g[2]);
                let scratch1 = c_add(g[1], g[3]);
                g[2] = c_sub(g[0], scratch1);
                g[0] = c_add(g[0], scratch1);
                let scratch1 = c_sub(g[1], g[3]);
                g[1] = CCpx { r: scratch0.r + scratch1.i, i: scratch0.i - scratch1.r };
                g[3] = CCpx { r: scratch0.r - scratch1.i, i: scratch0.i + scratch1.r };
            }
            return;
        }
        for i in 0..n {
            let g = &mut f[i * mm..];
            for j in 0..m {
                let s0 = c_mul(g[j + m], tw[j * fstride]);
                let s1 = c_mul(g[j + 2 * m], tw[j * fstride * 2]);
                let s2 = c_mul(g[j + 3 * m], tw[j * fstride * 3]);

                let s5 = c_sub(g[j], s1);
                g[j] = c_add(g[j], s1);
                let s3 = c_add(s0, s2);
                let s4 = c_sub(s0, s2);
                g[j + 2 * m] = c_sub(g[j], s3);
                g[j] = c_add(g[j], s3);

                g[j + m] = CCpx { r: s5.r + s4.i, i: s5.i - s4.r };
                g[j + 3 * m] = CCpx { r: s5.r - s4.i, i: s5.i + s4.r };
            }
        }
    }

    /// `kf_bfly3` (kiss_fft.c:180-235).
    fn bfly3(f: &mut [CCpx], fstride: usize, tw: &[CCpx], m: usize, n: usize, mm: usize) {
        let epi3 = tw[fstride * m];
        for i in 0..n {
            let g = &mut f[i * mm..];
            for j in 0..m {
                let s1 = c_mul(g[j + m], tw[j * fstride]);
                let s2 = c_mul(g[j + 2 * m], tw[j * fstride * 2]);

                let s3 = c_add(s1, s2);
                let mut s0 = c_sub(s1, s2);

                // Fout[m].r = Fout->r - HALF_OF(scratch[3].r): the product
                // is on the right of the subtraction and is fused.
                g[j + m] = CCpx {
                    r: (-s3.r).mul_add(0.5, g[j].r),
                    i: (-s3.i).mul_add(0.5, g[j].i),
                };

                s0.r *= epi3.i;
                s0.i *= epi3.i;

                g[j] = c_add(g[j], s3);

                g[j + 2 * m] = CCpx { r: g[j + m].r + s0.i, i: g[j + m].i - s0.r };
                g[j + m] = CCpx { r: g[j + m].r - s0.i, i: g[j + m].i + s0.r };
            }
        }
    }

    /// `kf_bfly5` (kiss_fft.c:239-312).
    fn bfly5(f: &mut [CCpx], fstride: usize, tw: &[CCpx], m: usize, n: usize, mm: usize) {
        let ya = tw[fstride * m];
        let yb = tw[fstride * 2 * m];
        for i in 0..n {
            let g = &mut f[i * mm..];
            for u in 0..m {
                let s0 = g[u];

                let s1 = c_mul(g[u + m], tw[u * fstride]);
                let s2 = c_mul(g[u + 2 * m], tw[2 * u * fstride]);
                let s3 = c_mul(g[u + 3 * m], tw[3 * u * fstride]);
                let s4 = c_mul(g[u + 4 * m], tw[4 * u * fstride]);

                let s7 = c_add(s1, s4);
                let s10 = c_sub(s1, s4);
                let s8 = c_add(s2, s3);
                let s9 = c_sub(s2, s3);

                g[u] = CCpx { r: s0.r + (s7.r + s8.r), i: s0.i + (s7.i + s8.i) };

                let s5 = CCpx {
                    r: s0.r + s7.r.mul_add(ya.r, s8.r * yb.r),
                    i: s0.i + s7.i.mul_add(ya.r, s8.i * yb.r),
                };
                let s6 = CCpx {
                    r: s10.i.mul_add(ya.i, s9.i * yb.i),
                    i: -(s10.r.mul_add(ya.i, s9.r * yb.i)),
                };

                g[u + m] = c_sub(s5, s6);
                g[u + 4 * m] = c_add(s5, s6);

                let s11 = CCpx {
                    r: s0.r + s7.r.mul_add(yb.r, s8.r * ya.r),
                    i: s0.i + s7.i.mul_add(yb.r, s8.i * ya.r),
                };
                let s12 = CCpx {
                    r: s9.i.mul_add(ya.i, -(s10.i * yb.i)),
                    i: s10.r.mul_add(yb.i, -(s9.r * ya.i)),
                };

                g[u + 2 * m] = c_add(s11, s12);
                g[u + 3 * m] = c_sub(s11, s12);
            }
        }
    }

    /// `opus_fft_impl` (kiss_fft.c:562-604).
    pub fn fft(st: &CFftState, fout: &mut [CCpx]) {
        let nfft = st.nfft as usize;
        let shift = st.shift.max(0) as usize;
        // SAFETY: a state of libopus has `nfft << shift` twiddles.
        let tw = unsafe { core::slice::from_raw_parts(st.twiddles, nfft << shift) };
        let mut fstride = [0usize; 9];
        fstride[0] = 1;
        let mut l = 0;
        loop {
            let p = st.factors[2 * l] as usize;
            let m = st.factors[2 * l + 1] as usize;
            fstride[l + 1] = fstride[l] * p;
            l += 1;
            if m == 1 {
                break;
            }
        }
        let mut m = st.factors[2 * l - 1] as usize;
        for i in (0..l).rev() {
            let m2 = if i != 0 { st.factors[2 * i - 1] as usize } else { 1 };
            match st.factors[2 * i] {
                2 => bfly2(fout, m, fstride[i]),
                4 => bfly4(fout, fstride[i] << shift, tw, m, fstride[i], m2),
                3 => bfly3(fout, fstride[i] << shift, tw, m, fstride[i], m2),
                5 => bfly5(fout, fstride[i] << shift, tw, m, fstride[i], m2),
                _ => unreachable!(),
            }
            m = m2;
        }
    }

    /// `clt_mdct_forward_c` (mdct.c:122-268) for a stride of 1. `trig` is
    /// the table of this transform size, `bitrev` that of its FFT.
    pub fn mdct_forward(
        st: &CFftState,
        trig: &[f32],
        input: &[f32],
        out: &mut [f32],
        window: &[f32],
        overlap: usize,
        stride: usize,
    ) {
        let n4 = st.nfft as usize;
        let n2 = 2 * n4;
        let bitrev = unsafe { core::slice::from_raw_parts(st.bitrev, n4) };
        let scale = st.scale;
        let mut f = vec![0f32; n2];
        let mut f2 = vec![CCpx::default(); n4];

        // Window, shuffle, fold.
        let mut xp1 = overlap / 2;
        let mut xp2 = n2 - 1 + overlap / 2;
        let mut wp1 = overlap / 2;
        let mut wp2 = (overlap / 2).wrapping_sub(1);
        let mut yp = 0;
        let limit = (overlap + 3) >> 2;
        let mut i = 0;
        while i < limit {
            f[yp] = input[xp1 + n2].mul_add(window[wp2], input[xp2] * window[wp1]);
            f[yp + 1] = input[xp1].mul_add(window[wp1], -(input[xp2 - n2] * window[wp2]));
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 = wp2.wrapping_sub(2);
            i += 1;
        }
        wp1 = 0;
        wp2 = overlap.wrapping_sub(1);
        while i < n4 - limit {
            f[yp] = input[xp2];
            f[yp + 1] = input[xp1];
            yp += 2;
            xp1 += 2;
            xp2 -= 2;
            i += 1;
        }
        while i < n4 {
            // -S_MUL(xp1[-N2], *wp1) + S_MUL(*xp2, *wp2): the negated
            // product on the left is the fused one.
            f[yp] = (-input[xp1 - n2]).mul_add(window[wp1], input[xp2] * window[wp2]);
            f[yp + 1] = input[xp1].mul_add(window[wp2], input[xp2 + n2] * window[wp1]);
            yp += 2;
            xp1 += 2;
            xp2 = xp2.wrapping_sub(2);
            wp1 += 2;
            wp2 = wp2.wrapping_sub(2);
            i += 1;
        }

        // Pre-rotation.
        for i in 0..n4 {
            let (t0, t1) = (trig[i], trig[n4 + i]);
            let (re, im) = (f[2 * i], f[2 * i + 1]);
            let yr = re.mul_add(t0, -(im * t1));
            let yi = im.mul_add(t0, re * t1);
            f2[bitrev[i] as usize] = CCpx { r: yr * scale, i: yi * scale };
        }

        fft(st, &mut f2);

        // Post-rotation.
        for i in 0..n4 {
            let (t0, t1) = (trig[i], trig[n4 + i]);
            let fp = f2[i];
            out[2 * i * stride] = fp.i.mul_add(t1, -(fp.r * t0));
            out[stride * (n2 - 1 - 2 * i)] = fp.r.mul_add(t1, fp.i * t0);
        }
    }

    /// `clt_mdct_backward_c` (mdct.c:272-386).
    pub fn mdct_backward(
        st: &CFftState,
        trig: &[f32],
        input: &[f32],
        out: &mut [f32],
        window: &[f32],
        overlap: usize,
        stride: usize,
    ) {
        let n4 = st.nfft as usize;
        let n2 = 2 * n4;
        let bitrev = unsafe { core::slice::from_raw_parts(st.bitrev, n4) };
        let o2 = overlap / 2;
        let mut f2 = vec![CCpx::default(); n4];

        // Pre-rotation. Real and imaginary part are swapped, because the
        // transform uses a forward FFT.
        for i in 0..n4 {
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let (t0, t1) = (trig[i], trig[n4 + i]);
            let yr = x2.mul_add(t0, x1 * t1);
            let yi = x1.mul_add(t0, -(x2 * t1));
            f2[bitrev[i] as usize] = CCpx { r: yi, i: yr };
        }

        fft(st, &mut f2);

        // Post-rotation, from both ends at once.
        for i in 0..n4.div_ceil(2) {
            let (re, im) = (f2[i].i, f2[i].r);
            let (t0, t1) = (trig[i], trig[n4 + i]);
            let yr = re.mul_add(t0, im * t1);
            let yi = re.mul_add(t1, -(im * t0));
            let j = n4 - 1 - i;
            let (re, im) = (f2[j].i, f2[j].r);
            out[o2 + 2 * i] = yr;
            out[o2 + n2 - 1 - 2 * i] = yi;
            let (t0, t1) = (trig[j], trig[n4 + j]);
            let yr = re.mul_add(t0, im * t1);
            let yi = re.mul_add(t1, -(im * t0));
            out[o2 + n2 - 2 - 2 * i] = yr;
            out[o2 + 2 * i + 1] = yi;
        }

        // Mirror on both sides (time-domain aliasing cancellation).
        for i in 0..o2 {
            let x1 = out[overlap - 1 - i];
            let x2 = out[i];
            let (wp1, wp2) = (window[i], window[overlap - 1 - i]);
            out[i] = x2.mul_add(wp2, -(x1 * wp1));
            out[overlap - 1 - i] = x2.mul_add(wp1, x1 * wp2);
        }
    }
}

/// The numbers a transform is tested with.
///
/// Class 0: audio-sized values. Class 1: zeros of both signs mixed with a
/// few values. Class 2: subnormal numbers and numbers just above them.
/// Class 3: values around the largest that stay finite through the
/// transform. Class 4: zeros of both signs, 1 and -1; many products and sums
/// are then exactly zero, and the sign of the zero is all that can differ.
const CLASSES: usize = 5;

fn fill(rng: &mut Lcg, out: &mut [f32], class: usize) {
    for v in out.iter_mut() {
        *v = match class {
            0 => rng.unit() * 32768.0,
            1 => match rng.next_u32() % 8 {
                0 => rng.unit() * 1000.0,
                1..=4 => 0.0,
                _ => -0.0,
            },
            2 => rng.unit() * 2e-38,
            4 => [0.0, -0.0, 1.0, -1.0][rng.next_u32() as usize % 4],
            _ => rng.unit() * 1e30,
        };
    }
}

/// Index of the first value whose bits differ.
fn first_difference(a: &[f32], b: &[f32]) -> Option<usize> {
    assert_eq!(a.len(), b.len());
    (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits())
}

/// True where the linked libopus is known to be built by clang for arm64
/// with fused multiply-add: the Homebrew library on Apple silicon. Only
/// there the tests require the bits of the library. Elsewhere they print how
/// many values differ.
const ARM64_CLANG_BUILD: bool = cfg!(all(target_arch = "aarch64", target_os = "macos"));

/// The transform tables of one FFT size of the built-in mode of libopus.
struct CSize {
    st: &'static CFftState,
    nfft: usize,
    /// Rotation table of the MDCT with `2 * nfft` coefficients.
    trig: &'static [f32],
}

fn c_sizes(mode: &'static CMode) -> Vec<CSize> {
    let mut offset = 0;
    (0..=3usize)
        .map(|shift| {
            let st = unsafe { &*mode.mdct.kfft[shift] };
            let nfft = st.nfft as usize;
            let trig = unsafe { core::slice::from_raw_parts(mode.mdct.trig.add(offset), 2 * nfft) };
            offset += 2 * nfft;
            CSize { st, nfft, trig }
        })
        .collect()
}

fn flat(v: &[CCpx]) -> Vec<f32> {
    v.iter().flat_map(|c| [c.r, c.i]).collect()
}

/// The model above gives the bits of the linked library: FFT, forward MDCT
/// and inverse MDCT of every size, for every class of input.
#[test]
fn model_of_the_arm64_build_matches_libopus() {
    let mode = c_mode();
    let window = unsafe { core::slice::from_raw_parts(mode.window, 120) };
    let overlap = 120usize;
    let mut rng = Lcg(0x11b0_0010);
    for (shift, size) in c_sizes(mode).iter().enumerate() {
        let (st, nfft, trig) = (size.st, size.nfft, size.trig);
        let n2 = 2 * nfft;
        let mut bad = [0usize; 3];
        for class in 0..CLASSES {
            for _ in 0..40 {
                let mut x = vec![0f32; 2 * nfft];
                fill(&mut rng, &mut x, class);
                let mut c: Vec<CCpx> = x.chunks_exact(2).map(|p| CCpx { r: p[0], i: p[1] }).collect();
                let mut m = c.clone();
                unsafe { c_opus_fft_impl(st, c.as_mut_ptr()) };
                model::fft(st, &mut m);
                if first_difference(&flat(&m), &flat(&c)).is_some() {
                    bad[0] += 1;
                }

                // Long blocks, and short blocks whose coefficients are 2, 4
                // or 8 values apart.
                for stride in [1usize, 2, 8] {
                    let mut input = vec![0f32; n2 + overlap];
                    fill(&mut rng, &mut input, class);
                    let mut c_in = input.clone();
                    let mut c_out = vec![0f32; stride * n2];
                    let mut m_out = vec![0f32; stride * n2];
                    unsafe {
                        clt_mdct_forward_c(
                            &mode.mdct,
                            c_in.as_mut_ptr(),
                            c_out.as_mut_ptr(),
                            mode.window,
                            overlap as c_int,
                            shift as c_int,
                            stride as c_int,
                            0,
                        )
                    };
                    model::mdct_forward(st, trig, &input, &mut m_out, window, overlap, stride);
                    if first_difference(&m_out, &c_out).is_some() {
                        bad[1] += 1;
                    }

                    let mut coeffs = vec![0f32; stride * n2];
                    fill(&mut rng, &mut coeffs, class);
                    let mut tail = vec![0f32; n2 + overlap];
                    fill(&mut rng, &mut tail, class);
                    let mut c_in = coeffs.clone();
                    let mut c_out = tail.clone();
                    let mut m_out = tail.clone();
                    unsafe {
                        clt_mdct_backward_c(
                            &mode.mdct,
                            c_in.as_mut_ptr(),
                            c_out.as_mut_ptr(),
                            mode.window,
                            overlap as c_int,
                            shift as c_int,
                            stride as c_int,
                            0,
                        )
                    };
                    model::mdct_backward(st, trig, &coeffs, &mut m_out, window, overlap, stride);
                    if first_difference(&m_out, &c_out).is_some() {
                        bad[2] += 1;
                    }
                }
            }
        }
        println!(
            "model against libopus, {nfft} points: {} of 200 FFTs differ, {} of 600 forward MDCTs, {} of 600 inverse MDCTs",
            bad[0], bad[1], bad[2]
        );
        if ARM64_CLANG_BUILD {
            assert_eq!(bad, [0, 0, 0], "{nfft} points");
        }
    }
}

/// The FFT gives the bits of `opus_fft_impl` of libopus.
///
/// `opus_fft_impl_fused` is portable code and must give the bits of the
/// model on every machine. Where the library is the arm64 build of clang,
/// `opus_fft_impl` and `opus_fft_impl_fused` must give the bits of the
/// library itself.
#[test]
fn fft_gives_the_bits_of_libopus() {
    let mode = c_mode();
    let base = KissFftState::new(480).unwrap();
    let mut rng = Lcg(0x11b0_0001);
    for (shift, size) in c_sizes(mode).iter().enumerate() {
        let nfft = size.nfft;
        let sub;
        let plan = if shift == 0 {
            &base
        } else {
            sub = KissFftState::new_sub(&base, nfft).unwrap();
            &sub
        };
        let mut differ = [0usize; 2];
        let total = CLASSES * 50 * 2 * nfft;
        for class in 0..CLASSES {
            for round in 0..50 {
                let mut x = vec![0f32; 2 * nfft];
                fill(&mut rng, &mut x, class);
                let mut c: Vec<CCpx> = x.chunks_exact(2).map(|p| CCpx { r: p[0], i: p[1] }).collect();
                let mut m = c.clone();
                let mut fast: Vec<KissCpx> = x.chunks_exact(2).map(|p| KissCpx::new(p[0], p[1])).collect();
                let mut fused = fast.clone();
                unsafe { c_opus_fft_impl(size.st, c.as_mut_ptr()) };
                model::fft(size.st, &mut m);
                opus_fft_impl(plan, &mut fast);
                opus_fft_impl_fused(plan, &mut fused);
                let c = flat(&c);
                let fast: Vec<f32> = fast.iter().flat_map(|v| [v.r, v.i]).collect();
                let fused: Vec<f32> = fused.iter().flat_map(|v| [v.r, v.i]).collect();
                assert_eq!(
                    first_difference(&fused, &flat(&m)),
                    None,
                    "opus_fft_impl_fused against the model, {nfft} points, class {class}, round {round}"
                );
                differ[0] += compare(&fast, &c).0;
                differ[1] += compare(&fused, &c).0;
                if ARM64_CLANG_BUILD {
                    assert_eq!(
                        first_difference(&fast, &c),
                        None,
                        "opus_fft_impl against libopus, {nfft} points, class {class}, round {round}"
                    );
                }
            }
        }
        println!(
            "fft {nfft}: {} of {total} values of opus_fft_impl differ from libopus, {} of opus_fft_impl_fused",
            differ[0], differ[1]
        );
        if ARM64_CLANG_BUILD {
            assert_eq!(differ, [0, 0], "{nfft} points");
        }
    }
}

/// The MDCT gives the bits of `clt_mdct_forward_c` and `clt_mdct_backward_c`
/// of libopus: the four transform sizes, long blocks and short blocks whose
/// coefficients are 2, 4 or 8 values apart.
///
/// `forward_fused` and `backward_fused` are portable code and must give the
/// bits of the model on every machine. Where the library is the arm64 build
/// of clang, `forward` and `backward` must give the bits of the library
/// itself.
///
/// opus1 runs with its own window, the one the codec uses. It is equal to
/// the window of libopus (see `mdct_tables_are_those_of_libopus`).
#[test]
fn mdct_gives_the_bits_of_libopus() {
    let mode = c_mode();
    let rust = default_mode();
    let window = rust.window;
    let overlap = 120usize;
    let mut rng = Lcg(0x11b0_0002);
    for (shift, size) in c_sizes(mode).iter().enumerate() {
        let n2 = 2 * size.nfft;
        // [forward, inverse] values that differ from libopus.
        let mut differ = [0usize; 2];
        let mut total = [0usize; 2];
        for stride in [1usize, 2, 4, 8] {
            for class in 0..CLASSES {
                for round in 0..10 {
                    let what = format!("{n2} coefficients, stride {stride}, class {class}, round {round}");

                    // Forward. The coefficients go to every `stride`-th
                    // place; the values between them must stay.
                    let mut input = vec![0f32; n2 + overlap];
                    fill(&mut rng, &mut input, class);
                    let mut before = vec![0f32; stride * (n2 - 1) + 1];
                    fill(&mut rng, &mut before, 0);
                    let mut c_in = input.clone();
                    let mut c_out = before.clone();
                    unsafe {
                        clt_mdct_forward_c(
                            &mode.mdct,
                            c_in.as_mut_ptr(),
                            c_out.as_mut_ptr(),
                            mode.window,
                            overlap as c_int,
                            shift as c_int,
                            stride as c_int,
                            0,
                        )
                    };
                    let mut m_out = before.clone();
                    model::mdct_forward(size.st, size.trig, &input, &mut m_out, window, overlap, stride);
                    let mut fused = before.clone();
                    rust.mdct.forward_fused(&input, &mut fused, window, overlap, shift, stride);
                    assert_eq!(first_difference(&fused, &m_out), None, "forward_fused against the model, {what}");
                    let mut fast = before.clone();
                    rust.mdct.forward(&input, &mut fast, window, overlap, shift, stride);
                    differ[0] += compare(&fast, &c_out).0;
                    total[0] += n2;
                    if ARM64_CLANG_BUILD {
                        assert_eq!(first_difference(&fast, &c_out), None, "forward against libopus, {what}");
                    }

                    // Inverse. The output holds the tail of the block before.
                    let mut coeffs = vec![0f32; stride * (n2 - 1) + 1];
                    fill(&mut rng, &mut coeffs, class);
                    let mut tail = vec![0f32; overlap / 2 + n2];
                    fill(&mut rng, &mut tail, class);
                    let mut c_in = coeffs.clone();
                    let mut c_out = tail.clone();
                    unsafe {
                        clt_mdct_backward_c(
                            &mode.mdct,
                            c_in.as_mut_ptr(),
                            c_out.as_mut_ptr(),
                            mode.window,
                            overlap as c_int,
                            shift as c_int,
                            stride as c_int,
                            0,
                        )
                    };
                    let mut m_out = tail.clone();
                    model::mdct_backward(size.st, size.trig, &coeffs, &mut m_out, window, overlap, stride);
                    let mut fused = tail.clone();
                    rust.mdct.backward_fused(&coeffs, &mut fused, window, overlap, shift, stride);
                    assert_eq!(first_difference(&fused, &m_out), None, "backward_fused against the model, {what}");
                    let mut fast = tail.clone();
                    rust.mdct.backward(&coeffs, &mut fast, window, overlap, shift, stride);
                    differ[1] += compare(&fast, &c_out).0;
                    total[1] += overlap / 2 + n2;
                    if ARM64_CLANG_BUILD {
                        assert_eq!(first_difference(&fast, &c_out), None, "backward against libopus, {what}");
                    }
                }
            }
        }
        println!(
            "mdct {n2}: {} of {} forward values differ from libopus, {} of {} inverse values",
            differ[0], total[0], differ[1], total[1]
        );
        if ARM64_CLANG_BUILD {
            assert_eq!(differ, [0, 0], "{n2} coefficients");
        }
    }
}
