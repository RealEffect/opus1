//! The frequency-domain helpers of libopus `dnn/freq.c` and `dnn/freq.h`:
//! band energies, the Burg cepstrum, the 18-band DCT and its inverse, the
//! 320-point forward and inverse transforms, and the LPC coefficients from
//! a cepstrum.
//!
//! Deep PLC uses all of them (feature analysis). OSCE uses `dct` and
//! `forward_transform` only. `lpc_weighting` is not ported: no 1.6.1 caller
//! uses it.
//!
//! # Rounding
//!
//! The functions give the bits of libopus 1.6.1 built for 64-bit ARM
//! without loop vectorization (`-O2 -fno-vectorize -fno-slp-vectorize`), on
//! every target. The rules come from the C source and were checked in the
//! assembly of that build:
//!
//! - `a * b + c` in one C expression, all in one type, is one fused
//!   multiply-add (`fmaf` and `fma` below). Of two products, `a * b + c * d`,
//!   the left one is fused and the right one rounded first.
//! - A `double` constant or a `double` function makes the expression
//!   `double`; the assignment to a `float` rounds. These parts are in `f64`
//!   here.
//! - Sums run in the order of the C loops.
//! - Constants that clang computes at compile time (`pow(.995, k)`,
//!   `1 - 6e-5 * i * i`, `sqrt(2./18)`) have the bits of the reference.
//!
//! # Math library
//!
//! `compute_burg_cepstrum` calls `log10` and `lpc_from_cepstrum` calls
//! `pow(10, x)`, both in double precision. (Apple clang turns the `pow`
//! call into `__exp10`. For every `f32` input in [-12, 12], both give the
//! same `f32` result after the multiplication by the band compensation;
//! this was checked when the port was designed.) With `std` these are the
//! functions of the platform's C library, like in libopus. Without `std`
//! they come from the `libm` crate, which can differ in the last bit of the
//! `f64` result for rare inputs. Each result is rounded to `f32` right
//! after, so a difference shows only when the exact value lies very close
//! to the middle between two `f32` values.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::compat::OnceCell;
use crate::kiss_fft::{KissCpx, KissFftState, opus_fft_fused};

use super::burg::silk_burg_analysis;
use super::lpcnet_tables::{DCT_TABLE, FFT_TWIDDLES, HALF_WINDOW};

/// `LPC_ORDER`: the order of the LPC filters.
pub const LPC_ORDER: usize = 16;
/// `PREEMPHASIS`: the pre-emphasis factor of the feature analysis.
pub const PREEMPHASIS: f32 = 0.85;
/// `FRAME_SIZE_5MS`: the frame size in units of 80 samples (5 ms at
/// 16 kHz).
pub const FRAME_SIZE_5MS: usize = 2;
/// `OVERLAP_SIZE_5MS`: the overlap in units of 80 samples.
pub const OVERLAP_SIZE_5MS: usize = 2;
/// `TRAINING_OFFSET_5MS`: the training offset in units of 80 samples.
pub const TRAINING_OFFSET_5MS: usize = 1;
/// `WINDOW_SIZE_5MS`: the window in units of 80 samples.
pub const WINDOW_SIZE_5MS: usize = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS;
/// `FRAME_SIZE`: 10 ms at 16 kHz.
pub const FRAME_SIZE: usize = 80 * FRAME_SIZE_5MS;
/// `OVERLAP_SIZE`: the overlap of two analysis windows.
pub const OVERLAP_SIZE: usize = 80 * OVERLAP_SIZE_5MS;
/// `TRAINING_OFFSET`.
pub const TRAINING_OFFSET: usize = 80 * TRAINING_OFFSET_5MS;
/// `WINDOW_SIZE`: the size of the forward transform.
pub const WINDOW_SIZE: usize = FRAME_SIZE + OVERLAP_SIZE;
/// `FREQ_SIZE`: the bins of the forward transform that are kept.
pub const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;
/// `NB_BANDS`: the bands of the band energies and of the cepstrum.
pub const NB_BANDS: usize = 18;
/// `NB_BANDS_1`.
pub const NB_BANDS_1: usize = NB_BANDS - 1;

/// `eband5ms` (`dnn/freq.c:45-48`): the band edges, in units of
/// `WINDOW_SIZE_5MS` bins (0 Hz to 8 kHz).
pub const EBAND5MS: [i16; NB_BANDS] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40];

/// `compensation` (`dnn/freq.c:50-52`): the band gains applied in
/// `lpc_from_cepstrum`.
pub const COMPENSATION: [f32; NB_BANDS] = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2,
    0.166667, 0.173913,
];

/// `sqrt(2./NB_BANDS)` of `dct` and `idct` (`dnn/freq.c:221`, `:233`), as
/// clang computes it at compile time: exactly 1/3 rounded to `f64`.
const DCT_SCALE: f64 = f64::from_bits(0x3fd5_5555_5555_5555);

/// `-pow(.995, k)` for k = 1 to 16 (`dnn/freq.c:169`), the bandwidth
/// expansion of the Burg LPC. clang computes these at compile time; the
/// bits are those of the reference build (`pow` of the C library gives the
/// same).
const NEG_POW_0995: [f64; LPC_ORDER] = [
    f64::from_bits(0xbfef_d70a_3d70_a3d7),
    f64::from_bits(0xbfef_ae48_e8a7_1de7),
    f64::from_bits(0xbfef_85bb_be87_8fac),
    f64::from_bits(0xbfef_5d62_7c4c_00df),
    f64::from_bits(0xbfef_353c_df83_f182),
    f64::from_bits(0xbfef_0d4a_a613_ec76),
    f64::from_bits(0xbfee_e58b_8e35_1aa3),
    f64::from_bits(0xbfee_bdff_5674_d6aa),
    f64::from_bits(0xbfee_96a5_bdb4_411c),
    f64::from_bits(0xbfee_6f7e_8327_d544),
    f64::from_bits(0xbfee_4889_6656_fe70),
    f64::from_bits(0xbfee_21c6_271b_adce),
    f64::from_bits(0xbfed_fb34_85a1_f0c7),
    f64::from_bits(0xbfed_d4d4_4267_87e5),
    f64::from_bits(0xbfed_aea5_1e3b_7e41),
    f64::from_bits(0xbfed_88a6_da3d_c176),
];

/// The lag window `1 - 6e-5*i*i` for i = 1 to 16 (`dnn/freq.c:289`), in
/// `f64`. Fused or not, the products give the same values for these `i`.
const LAG_WINDOW: [f64; LPC_ORDER] = {
    let mut w = [0.0; LPC_ORDER];
    let mut i = 0;
    while i < LPC_ORDER {
        let k = (i + 1) as f64;
        w[i] = 1.0 - 6e-5 * k * k;
        i += 1;
    }
    w
};

/// `1.f/((float)WINDOW_SIZE*WINDOW_SIZE*WINDOW_SIZE)` of
/// `compute_burg_cepstrum` (`dnn/freq.c:172`), in `f32`.
const INV_WINDOW_CUBE: f32 =
    1.0 / (WINDOW_SIZE as f32 * WINDOW_SIZE as f32 * WINDOW_SIZE as f32);

/// `a * b + c` with one rounding: a C expression that clang contracts to a
/// fused multiply-add. On 64-bit ARM it is the `fmadd` instruction, also
/// without `std`. Elsewhere it is `mul_add`, which is slow on a processor
/// without a fused multiply-add instruction.
#[inline(always)]
pub(crate) fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::mathops::mac(c, a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
}

/// `a * b + c` in `f64` with one rounding (see `fmaf`).
#[inline(always)]
pub(crate) fn fma(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::mathops::mac_f64(c, a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
}

/// The plan of the 320-point FFT (`kfft`, `dnn/lpcnet_tables.c:198-206`),
/// built once from the embedded twiddle table. Its factors are
/// {5, 64, 4, 16, 4, 4, 4, 1} and its scale is `0.003125f`, as in libopus.
fn kfft() -> &'static KissFftState {
    static KFFT: OnceCell<KissFftState> = OnceCell::new();
    KFFT.get(|| match KissFftState::with_twiddles(&FFT_TWIDDLES) {
        Some(st) => st,
        // 320 = 5 * 4 * 4 * 4 always has a plan (unit test
        // `the_plan_is_the_one_of_libopus`).
        None => unreachable!("no FFT plan for 320 points"),
    })
}

/// `opus_fft(&kfft, x, y, arch)` of `freq.c`: the scaled 320-point FFT.
fn fft320(x: &[KissCpx; WINDOW_SIZE], y: &mut [KissCpx; WINDOW_SIZE]) {
    opus_fft_fused(kfft(), x, y);
}

/// The first bin of band `i` and the width of the band in bins.
#[inline(always)]
fn band(i: usize) -> (usize, usize) {
    let start = EBAND5MS[i] as usize * WINDOW_SIZE_5MS;
    let size = (EBAND5MS[i + 1] - EBAND5MS[i]) as usize * WINDOW_SIZE_5MS;
    (start, size)
}

/// The energy of each band, with triangular band weights (libopus
/// `lpcn_compute_band_energy`, `dnn/freq.c:126-149`).
pub fn lpcn_compute_band_energy(band_e: &mut [f32; NB_BANDS], x: &[KissCpx; FREQ_SIZE]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let (start, band_size) = band(i);
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let bin = x[start + j];
            let tmp = fmaf(bin.i, bin.i, bin.r * bin.r);
            sum[i] = fmaf(1.0 - frac, tmp, sum[i]);
            sum[i + 1] = fmaf(frac, tmp, sum[i + 1]);
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    *band_e = sum;
}

/// The Burg cepstrum of 10 ms (libopus `burg_cepstral_analysis`,
/// `dnn/freq.c:183-194`): two 5 ms Burg analyses; `ceps[0..18]` gets their
/// mean and `ceps[18..36]` their difference.
pub fn burg_cepstral_analysis(ceps: &mut [f32; 2 * NB_BANDS], x: &[f32; FRAME_SIZE]) {
    let (first, second) = ceps.split_at_mut(NB_BANDS);
    let (first, second): (&mut [f32; NB_BANDS], &mut [f32; NB_BANDS]) =
        match (first.try_into(), second.try_into()) {
            (Ok(a), Ok(b)) => (a, b),
            _ => unreachable!("the halves of a 36-value array"),
        };
    compute_burg_cepstrum(&x[..FRAME_SIZE / 2], first, FRAME_SIZE / 2, LPC_ORDER);
    compute_burg_cepstrum(&x[FRAME_SIZE / 2..], second, FRAME_SIZE / 2, LPC_ORDER);
    for i in 0..NB_BANDS {
        let c0 = ceps[i];
        let c1 = ceps[NB_BANDS + i];
        // `.5*(c0+c1)`: a `double` product, but halving is exact, so the
        // `f32` product rounds the same.
        ceps[i] = 0.5 * (c0 + c1);
        ceps[NB_BANDS + i] = c0 - c1;
    }
}

/// Multiplies `x` by the analysis window (libopus `apply_window`,
/// `dnn/freq.c:317-323`): the first 160 values by `HALF_WINDOW`, the last
/// 160 by `HALF_WINDOW` in reverse order.
pub fn apply_window(x: &mut [f32; WINDOW_SIZE]) {
    for i in 0..OVERLAP_SIZE {
        x[i] *= HALF_WINDOW[i];
        x[WINDOW_SIZE - 1 - i] *= HALF_WINDOW[i];
    }
}

/// The 18-point DCT-II with `DCT_TABLE` (libopus `dct`,
/// `dnn/freq.c:213-223`): `out[i] = sum_j in[j]·DCT_TABLE[j·18 + i]`, times
/// `sqrt(2/18)`.
pub fn dct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS]) {
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum = fmaf(input[j], DCT_TABLE[j * NB_BANDS + i], sum);
        }
        out[i] = (sum as f64 * DCT_SCALE) as f32;
    }
}

/// The first `FREQ_SIZE` bins of the 320-point FFT of a real signal, scaled
/// by 1/320 (libopus `forward_transform`, `dnn/freq.c:237-249`, through
/// `opus_fft` with the plan `kfft`).
pub fn forward_transform(out: &mut [KissCpx; FREQ_SIZE], input: &[f32; WINDOW_SIZE]) {
    let mut x = [KissCpx::default(); WINDOW_SIZE];
    for (xi, &v) in x.iter_mut().zip(input.iter()) {
        xi.r = v;
    }
    let mut y = [KissCpx::default(); WINDOW_SIZE];
    fft320(&x, &mut y);
    out.copy_from_slice(&y[..FREQ_SIZE]);
}

/// LPC coefficients from an 18-band cepstrum (libopus `lpc_from_cepstrum`,
/// `dnn/freq.c:305-315`). Returns the prediction error.
pub fn lpc_from_cepstrum(lpc: &mut [f32; LPC_ORDER], cepstrum: &[f32; NB_BANDS]) -> f32 {
    let mut tmp = *cepstrum;
    tmp[0] += 4.0;
    let mut ex = [0.0f32; NB_BANDS];
    idct(&mut ex, &tmp);
    for (e, &c) in ex.iter_mut().zip(COMPENSATION.iter()) {
        *e = (10f64.powf(*e as f64) * c as f64) as f32;
    }
    lpc_from_bands(lpc, &ex)
}

/// `compute_band_energy_inverse` (`dnn/freq.c:55-79`): band sums of
/// `1/(|X|² + 1e-9)`, with triangular band weights.
fn compute_band_energy_inverse(band_e: &mut [f32; NB_BANDS], x: &[KissCpx; FREQ_SIZE]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let (start, band_size) = band(i);
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let bin = x[start + j];
            let tmp = fmaf(bin.i, bin.i, bin.r * bin.r);
            // `1.f/(tmp + 1e-9)`: a `double` division.
            let tmp = (1.0 / (tmp as f64 + 1e-9)) as f32;
            sum[i] = fmaf(1.0 - frac, tmp, sum[i]);
            sum[i + 1] = fmaf(frac, tmp, sum[i + 1]);
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    *band_e = sum;
}

/// `lpcn_lpc` (`dnn/freq.c:81-122`): the Levinson recursion of the feature
/// analysis. Writes `p` coefficients to `lpc` and the reflection
/// coefficients to `rc`, and returns the prediction error. It stops at
/// 30 dB of prediction gain. Not the same as `celt_lpc::lpc` (other stop
/// test and zero test).
fn lpcn_lpc(lpc: &mut [f32], rc: &mut [f32], ac: &[f32], p: usize) -> f32 {
    let mut error = ac[0];
    lpc[..p].fill(0.0);
    rc[..p].fill(0.0);
    if ac[0] != 0.0 {
        let stop = 0.001f32 * ac[0];
        for i in 0..p {
            // This iteration's reflection coefficient.
            let mut rr = 0.0f32;
            for j in 0..i {
                rr = fmaf(lpc[j], ac[i - j], rr);
            }
            rr += ac[i + 1];
            let r = -rr / error;
            rc[i] = r;
            // Update the LPC coefficients and the error.
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = fmaf(r, tmp2, tmp1);
                lpc[i - 1 - j] = fmaf(r, tmp1, tmp2);
            }
            error = fmaf(-(r * r), error, error);
            // Stop at 30 dB of prediction gain.
            if error < stop {
                break;
            }
        }
    }
    error
}

/// `compute_burg_cepstrum` (`dnn/freq.c:151-181`): the cepstrum of one Burg
/// LPC fit of the `len` samples of `pcm`, after pre-emphasis.
///
/// Requires `2 <= len <= FRAME_SIZE`, `1 <= order <= LPC_ORDER`, `order`
/// smaller than `len`, and `pcm.len() >= len`.
fn compute_burg_cepstrum(
    pcm: &[f32],
    burg_cepstrum: &mut [f32; NB_BANDS],
    len: usize,
    order: usize,
) {
    debug_assert!(order <= LPC_ORDER);
    debug_assert!(len <= FRAME_SIZE);
    let mut burg_in = [0.0f32; FRAME_SIZE];
    for i in 0..len - 1 {
        burg_in[i] = fmaf(-PREEMPHASIS, pcm[i], pcm[i + 1]);
    }
    let mut burg_lpc = [0.0f32; LPC_ORDER];
    let mut g = silk_burg_analysis(&mut burg_lpc, &burg_in, 1e-3, len - 1, 1, order);
    // `g /= len - 2*(order-1)`: an `int` divisor, converted to `float`.
    g /= (len as i32 - 2 * (order as i32 - 1)) as f32;

    let mut x = [0.0f32; WINDOW_SIZE];
    x[0] = 1.0;
    for i in 0..order {
        x[i + 1] = (burg_lpc[i] as f64 * NEG_POW_0995[i]) as f32;
    }
    let mut lpc_spec = [KissCpx::default(); FREQ_SIZE];
    forward_transform(&mut lpc_spec, &x);
    let mut eburg = [0.0f32; NB_BANDS];
    compute_band_energy_inverse(&mut eburg, &lpc_spec);
    // `.45*g*(1.f/(320.f*320*320))`, in `double`.
    let gain = 0.45 * g as f64 * INV_WINDOW_CUBE as f64;
    for e in eburg.iter_mut() {
        *e = (*e as f64 * gain) as f32;
    }

    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max = -2.0f32;
    let mut follow = -2.0f32;
    for i in 0..NB_BANDS {
        let l = (1e-2 + eburg[i] as f64).log10() as f32;
        // `MAX16(logMax-8, MAX16(follow-2.5, Ly[i]))`: `follow-2.5` is a
        // `double`, so both comparisons run in `double`.
        let follow_down = follow as f64 - 2.5;
        let inner = if follow_down > l as f64 { follow_down } else { l as f64 };
        let floor = (log_max - 8.0) as f64;
        ly[i] = (if floor > inner { floor } else { inner }) as f32;
        log_max = if log_max > ly[i] { log_max } else { ly[i] };
        follow = (if follow_down > ly[i] as f64 { follow_down } else { ly[i] as f64 }) as f32;
    }
    dct(burg_cepstrum, &ly);
    burg_cepstrum[0] += -4.0;
}

/// `interp_band_gain` (`dnn/freq.c:197-210`): band values interpolated to
/// `FREQ_SIZE` bins. The last bin is 0. (libopus leaves it as it was: its
/// `memset` clears 161 bytes, not 161 floats. Its only caller sets the bin
/// to 0 afterwards.)
fn interp_band_gain(g: &mut [f32; FREQ_SIZE], band_e: &[f32; NB_BANDS]) {
    g.fill(0.0);
    for i in 0..NB_BANDS - 1 {
        let (start, band_size) = band(i);
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            g[start + j] = fmaf(1.0 - frac, band_e[i], frac * band_e[i + 1]);
        }
    }
}

/// `idct` (`dnn/freq.c:225-235`): the inverse of `dct`,
/// `out[i] = sum_j in[j]·DCT_TABLE[i·18 + j]`, times `sqrt(2/18)`.
fn idct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS]) {
    for i in 0..NB_BANDS {
        let row = &DCT_TABLE[i * NB_BANDS..(i + 1) * NB_BANDS];
        let mut sum = 0.0f32;
        for (&v, &t) in input.iter().zip(row.iter()) {
            sum = fmaf(v, t, sum);
        }
        out[i] = (sum as f64 * DCT_SCALE) as f32;
    }
}

/// `inverse_transform` (`dnn/freq.c:251-268`): the real signal of a
/// one-sided spectrum, through the forward FFT of its conjugate-symmetric
/// extension, read in reverse order and multiplied by 320.
fn inverse_transform(out: &mut [f32; WINDOW_SIZE], input: &[KissCpx; FREQ_SIZE]) {
    let mut x = [KissCpx::default(); WINDOW_SIZE];
    x[..FREQ_SIZE].copy_from_slice(input);
    for i in FREQ_SIZE..WINDOW_SIZE {
        x[i].r = x[WINDOW_SIZE - i].r;
        x[i].i = -x[WINDOW_SIZE - i].i;
    }
    let mut y = [KissCpx::default(); WINDOW_SIZE];
    fft320(&x, &mut y);
    // Output in reverse order for the inverse FFT.
    out[0] = WINDOW_SIZE as f32 * y[0].r;
    for i in 1..WINDOW_SIZE {
        out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].r;
    }
}

/// `lpc_from_bands` (`dnn/freq.c:270-292`): LPC coefficients from band
/// energies, through the autocorrelation, with a -40 dB noise floor and lag
/// windowing. Returns the prediction error.
fn lpc_from_bands(lpc: &mut [f32; LPC_ORDER], ex: &[f32; NB_BANDS]) -> f32 {
    let mut xr = [0.0f32; FREQ_SIZE];
    interp_band_gain(&mut xr, ex);
    xr[FREQ_SIZE - 1] = 0.0;
    let mut x_auto_spec = [KissCpx::default(); FREQ_SIZE];
    for (bin, &v) in x_auto_spec.iter_mut().zip(xr.iter()) {
        bin.r = v;
    }
    let mut x_auto = [0.0f32; WINDOW_SIZE];
    inverse_transform(&mut x_auto, &x_auto_spec);
    let mut ac = [0.0f32; LPC_ORDER + 1];
    ac.copy_from_slice(&x_auto[..LPC_ORDER + 1]);

    // -40 dB noise floor: `ac[0] += ac[0]*1e-4 + 320/12/38.`, in `double`
    // (`320/12` is an integer division).
    let ac0 = ac[0] as f64;
    ac[0] = (ac0 + fma(ac0, 1e-4, (320 / 12) as f64 / 38.0)) as f32;
    // Lag windowing.
    for (a, &w) in ac[1..].iter_mut().zip(LAG_WINDOW.iter()) {
        *a = (*a as f64 * w) as f32;
    }
    let mut rc = [0.0f32; LPC_ORDER];
    lpcn_lpc(lpc, &mut rc, &ac, LPC_ORDER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiss_fft::opus_fft_impl_fused;
    use crate::dnn::lpcnet_tables::{FFT_BITREV, KFFT_SCALE};

    /// A small deterministic generator: values in [-1, 1).
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    #[test]
    fn the_plan_is_the_one_of_libopus() {
        let st = kfft();
        assert_eq!(st.nfft(), 320);
        assert_eq!(st.scale().to_bits(), KFFT_SCALE.to_bits());
        assert_eq!(&st.bitrev[..], &FFT_BITREV[..]);
    }

    #[test]
    fn folded_constants_have_the_reference_bits() {
        // From the assembly of the reference build (`dnn/freq.c:289`).
        let lag: [u64; 16] = [
            0x3fefff822bbecaac, 0x3feffe08aefb2aae, 0x3feffb9389b52008, 0x3feff822bbecaab9,
            0x3feff3b645a1cac1, 0x3fefee4e26d4801f, 0x3fefe7ea5f84cad5, 0x3fefe08aefb2aae3,
            0x3fefd82fd75e2047, 0x3fefced916872b02, 0x3fefc486ad2dcb14, 0x3fefb9389b52007e,
            0x3fefaceee0f3cb3e, 0x3fef9fa97e132b56, 0x3fef916872b020c5, 0x3fef822bbecaab8a,
        ];
        for (w, &bits) in LAG_WINDOW.iter().zip(lag.iter()) {
            assert_eq!(w.to_bits(), bits);
        }
        assert_eq!(((320 / 12) as f64 / 38.0).to_bits(), 0x3fe5e50d79435e51);
        assert_eq!(INV_WINDOW_CUBE.to_bits(), (1.0f32 / 32_768_000.0).to_bits());
        assert_eq!((2.0f64 / NB_BANDS as f64).sqrt().to_bits(), DCT_SCALE.to_bits());
        #[cfg(feature = "std")]
        for (k, p) in NEG_POW_0995.iter().enumerate() {
            assert_eq!(p.to_bits(), (-(0.995f64.powf((k + 1) as f64))).to_bits(), "k = {}", k + 1);
        }
    }

    #[test]
    fn the_fft_path_gives_the_fused_scalar_bits() {
        // On aarch64 `fft320` runs the vector butterflies; they must give
        // the bits of the portable fused code.
        let mut rng = Lcg(7);
        for _ in 0..50 {
            let mut x = [KissCpx::default(); WINDOW_SIZE];
            for v in x.iter_mut() {
                *v = KissCpx::new(rng.next(), rng.next());
            }
            let mut y = [KissCpx::default(); WINDOW_SIZE];
            fft320(&x, &mut y);
            let st = kfft();
            let mut z = [KissCpx::default(); WINDOW_SIZE];
            for (i, v) in x.iter().enumerate() {
                z[st.bitrev[i] as usize] = KissCpx::new(v.r * st.scale(), v.i * st.scale());
            }
            opus_fft_impl_fused(st, &mut z);
            for (a, b) in y.iter().zip(z.iter()) {
                assert_eq!((a.r.to_bits(), a.i.to_bits()), (b.r.to_bits(), b.i.to_bits()));
            }
        }
    }

    #[test]
    fn forward_transform_finds_a_cosine() {
        // A cosine at bin 20: X[20] = 1/2, the other bins near 0.
        let mut x = [0.0f32; WINDOW_SIZE];
        for (n, v) in x.iter_mut().enumerate() {
            *v = (2.0 * core::f64::consts::PI * 20.0 * n as f64 / 320.0).cos() as f32;
        }
        let mut spec = [KissCpx::default(); FREQ_SIZE];
        forward_transform(&mut spec, &x);
        for (k, b) in spec.iter().enumerate() {
            let want = if k == 20 { 0.5 } else { 0.0 };
            assert!((b.r - want).abs() < 1e-5 && b.i.abs() < 1e-5, "bin {k}: {b:?}");
        }
    }

    #[test]
    fn inverse_transform_undoes_forward_transform() {
        let mut rng = Lcg(3);
        let mut x = [0.0f32; WINDOW_SIZE];
        for v in x.iter_mut() {
            *v = rng.next();
        }
        let mut spec = [KissCpx::default(); FREQ_SIZE];
        forward_transform(&mut spec, &x);
        let mut back = [0.0f32; WINDOW_SIZE];
        inverse_transform(&mut back, &spec);
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} {b}");
        }
    }

    #[test]
    fn idct_undoes_dct() {
        let mut rng = Lcg(11);
        let mut v = [0.0f32; NB_BANDS];
        for x in v.iter_mut() {
            *x = 4.0 * rng.next();
        }
        let mut c = [0.0f32; NB_BANDS];
        dct(&mut c, &v);
        let mut back = [0.0f32; NB_BANDS];
        idct(&mut back, &c);
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} {b}");
        }
    }

    #[test]
    fn band_energy_of_a_flat_spectrum() {
        // |X|^2 = 1 in every bin. Band i gets the weights `1 - j/s` of its
        // own `s` bins, (s + 1)/2 in all, and the weights `j/s'` of the `s'`
        // bins of band i - 1, (s' - 1)/2 in all. The end bands are doubled.
        let spec = [KissCpx::new(0.6, 0.8); FREQ_SIZE];
        let mut e = [0.0f32; NB_BANDS];
        lpcn_compute_band_energy(&mut e, &spec);
        for i in 0..NB_BANDS {
            let left = if i > 0 { (band(i - 1).1 as f32 - 1.0) / 2.0 } else { 0.0 };
            let right = if i < NB_BANDS - 1 { (band(i).1 as f32 + 1.0) / 2.0 } else { 0.0 };
            let mut want = left + right;
            if i == 0 || i == NB_BANDS - 1 {
                want *= 2.0;
            }
            // The weights `frac` and `1 - frac` are rounded.
            assert!((e[i] - want).abs() < 1e-4 * want.max(1.0), "band {i}: {} {want}", e[i]);
        }
    }

    #[test]
    fn lpc_from_cepstrum_gives_finite_values() {
        let mut rng = Lcg(5);
        for _ in 0..100 {
            let mut ceps = [0.0f32; NB_BANDS];
            for c in ceps.iter_mut() {
                *c = 2.0 * rng.next();
            }
            ceps[0] -= 4.0;
            let mut lpc = [f32::NAN; LPC_ORDER];
            let err = lpc_from_cepstrum(&mut lpc, &ceps);
            assert!(err > 0.0 && err.is_finite(), "{err}");
            assert!(lpc.iter().all(|a| a.is_finite()), "{lpc:?}");
        }
    }

    #[test]
    fn lpcn_lpc_of_zero_autocorrelation_is_zero() {
        let mut lpc = [1.0f32; LPC_ORDER];
        let mut rc = [1.0f32; LPC_ORDER];
        let ac = [0.0f32; LPC_ORDER + 1];
        assert_eq!(lpcn_lpc(&mut lpc, &mut rc, &ac, LPC_ORDER), 0.0);
        assert!(lpc.iter().chain(rc.iter()).all(|&v| v == 0.0));
    }

    #[test]
    fn lpcn_lpc_finds_a_first_order_predictor() {
        // The autocorrelation of an AR(1) process with coefficient 0.9:
        // ac[k] = 0.9^k. The predictor is -0.9 (libopus's sign) and 0 after.
        let mut ac = [0.0f32; LPC_ORDER + 1];
        for (k, a) in ac.iter_mut().enumerate() {
            *a = 0.9f32.powi(k as i32);
        }
        let mut lpc = [0.0f32; LPC_ORDER];
        let mut rc = [0.0f32; LPC_ORDER];
        let err = lpcn_lpc(&mut lpc, &mut rc, &ac, LPC_ORDER);
        assert!((lpc[0] + 0.9).abs() < 1e-5, "{lpc:?}");
        assert!(lpc[1..].iter().all(|a| a.abs() < 1e-5), "{lpc:?}");
        assert!((err - 0.19).abs() < 1e-5, "{err}");
    }

    #[test]
    fn burg_cepstrum_is_finite_for_silence_and_full_scale() {
        let mut ceps = [0.0f32; 2 * NB_BANDS];
        burg_cepstral_analysis(&mut ceps, &[0.0; FRAME_SIZE]);
        assert!(ceps.iter().all(|c| c.is_finite()), "{ceps:?}");
        let mut square = [0.0f32; FRAME_SIZE];
        for (n, v) in square.iter_mut().enumerate() {
            *v = if (n / 8) % 2 == 0 { 32767.0 } else { -32767.0 };
        }
        burg_cepstral_analysis(&mut ceps, &square);
        assert!(ceps.iter().all(|c| c.is_finite()), "{ceps:?}");
    }

    #[test]
    fn apply_window_is_symmetric() {
        let mut x = [1.0f32; WINDOW_SIZE];
        apply_window(&mut x);
        for i in 0..OVERLAP_SIZE {
            assert_eq!(x[i], HALF_WINDOW[i]);
            assert_eq!(x[WINDOW_SIZE - 1 - i], HALF_WINDOW[i]);
        }
    }
}
