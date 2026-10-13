use crate::kiss_fft::{KissCpx, KissFftState, opus_fft_impl, opus_fft_impl_scalar};
use core::f32::consts::PI;
use core::mem::MaybeUninit;

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;

const MAX_N2: usize = 960;
const MAX_N4: usize = 480;
/// Maximum number of MDCT levels (`max_lm`). The default 48 kHz / 960 mode uses 3;
/// 4 leaves headroom for hybrid modes.
const MDCT_MAX_LM: usize = 4;
/// Bounded by the geometric sum of `n2` across all levels: `n/2 + n/4 + ... <= n`.
const MDCT_TRIG_MAX: usize = 2 * MAX_N2;

pub struct MdctLookup {
    pub n: usize,
    pub max_lm: usize,
    kfft: FixedVec<Option<KissFftState>, { MDCT_MAX_LM + 1 }>,
    trig: FixedVec<f32, MDCT_TRIG_MAX>,
}

/// What one transform size needs: the FFT plan, the rotation table and the
/// sizes. `n2` is the number of coefficients, `n4` the FFT size.
struct Plan<'a> {
    st: &'a KissFftState,
    /// `n2` cosines. The rotation of butterfly `i` uses `trig[i]` and
    /// `trig[n4 + i]`.
    trig: &'a [f32],
    n2: usize,
    n4: usize,
}

impl MdctLookup {
    pub fn new(n: usize, max_lm: usize) -> Self {
        debug_assert!(max_lm <= MDCT_MAX_LM);
        let mut kfft: FixedVec<Option<KissFftState>, { MDCT_MAX_LM + 1 }> = FixedVec::new();
        let mut trig: FixedVec<f32, MDCT_TRIG_MAX> = FixedVec::new();
        let mut curr_n = n;

        for shift in 0..=max_lm {
            let n4 = curr_n / 4;

            if shift == 0 {
                kfft.push(KissFftState::new(n4));
            } else if let Some(base) = kfft.first().unwrap().as_ref() {
                kfft.push(KissFftState::new_sub(base, n4));
            } else {
                kfft.push(None);
            }

            let n2 = curr_n / 2;
            for i in 0..n2 {
                let angle = 2.0 * PI * (i as f32 + 0.125) / curr_n as f32;
                trig.push(angle.cos());
            }

            curr_n >>= 1;
        }

        Self {
            n,
            max_lm,
            kfft,
            trig,
        }
    }

    fn get_trig(&self, shift: usize) -> (&[f32], usize) {
        let mut offset = 0;
        let mut curr_n = self.n;
        for _ in 0..shift {
            offset += curr_n / 2;
            curr_n >>= 1;
        }
        (&self.trig[offset..offset + curr_n / 2], curr_n / 4)
    }

    pub fn get_trig_debug(&self, shift: usize) -> &[f32] {
        let (trig, _) = self.get_trig(shift);
        trig
    }

    /// The plan of the transform size `self.n >> shift`.
    #[inline]
    fn plan(&self, shift: usize) -> Plan<'_> {
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let (trig, _) = self.get_trig(shift);
        // The steps below rely on these facts. A plan exists only for
        // 1 <= n4 <= MAX_N4.
        assert!(
            n2 == 2 * n4 && st.nfft() == n4 && st.bitrev.len() == n4 && n4 <= MAX_N4,
            "MDCT: size {} is not a multiple of 4",
            n
        );
        Plan { st, trig, n2, n4 }
    }

    /// Checks the arguments of `forward` and returns the plan.
    #[inline]
    fn forward_plan(
        &self,
        input: &[f32],
        output: &[f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) -> Plan<'_> {
        let p = self.plan(shift);
        let (n2, n4) = (p.n2, p.n4);
        // The window-and-fold step reads `input[..n2 + overlap]`: `overlap / 4`
        // butterflies at each end of the block use the window, the ones in
        // between do not. That needs `overlap` to be a multiple of 4 and at
        // most `n2`, as in libopus.
        assert!(
            overlap.is_multiple_of(4) && overlap / 2 <= n4,
            "MDCT forward: overlap {} does not fit a block of {} coefficients",
            overlap,
            n2
        );
        assert!(
            input.len() >= n2 + overlap,
            "MDCT forward: input too small (need {}, have {})",
            n2 + overlap,
            input.len()
        );
        assert!(
            window.len() >= overlap,
            "MDCT forward: window too small (need {}, have {})",
            overlap,
            window.len()
        );
        // The last coefficient goes to `output[stride * (n2 - 1)]`.
        assert!(
            stride >= 1 && output.len() > stride * (n2 - 1),
            "MDCT forward: output buffer too small (need {}, have {})",
            stride * (n2 - 1) + 1,
            output.len()
        );
        p
    }

    /// Checks the arguments of `backward` and returns the plan.
    #[inline]
    fn backward_plan(
        &self,
        input: &[f32],
        output: &[f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) -> Plan<'_> {
        let p = self.plan(shift);
        let n2 = p.n2;
        let overlap2 = overlap / 2;
        // Validate all buffer contracts up front: the pre-rotation reads
        // input[2*i*stride] and input[stride*(n2-1-2*i)], and the TDAC stage
        // reads window[..overlap] — failing after the FFT has run would waste
        // the whole transform (issue #27 deep scan).
        assert!(
            stride >= 1 && input.len() > stride * (n2 - 1),
            "MDCT backward: input too small (need {}, have {})",
            stride * (n2 - 1) + 1,
            input.len()
        );
        assert!(
            window.len() >= overlap,
            "MDCT backward: window too small (need {}, have {})",
            overlap,
            window.len()
        );
        // The rotation writes output[overlap2..overlap2 + n2], the TDAC stage
        // rewrites output[..overlap].
        assert!(
            output.len() >= overlap2 + n2 && output.len() >= overlap,
            "MDCT backward: output too small (need {}, have {})",
            (overlap2 + n2).max(overlap),
            output.len()
        );
        p
    }

    /// Forward MDCT of `n2 = (self.n >> shift) / 2` coefficients.
    ///
    /// Reads `input[..n2 + overlap]` and writes the coefficients to
    /// `output[0]`, `output[stride]`, ..., `output[stride * (n2 - 1)]`.
    #[inline]
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let p = self.forward_plan(input, output, window, overlap, shift, stride);

        // One choice of the implementation per transform.
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the arguments passed `forward_plan`.
            unsafe { forward_neon(&p, input, output, window, overlap, stride) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::compat::x86_has_avx() {
                // SAFETY: the processor has AVX.
                unsafe { forward_avx(&p, input, output, window, overlap, stride) }
            } else {
                forward_impl(&p, input, output, window, overlap, stride, opus_fft_impl);
            }
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64"
        )))]
        forward_impl(&p, input, output, window, overlap, stride, opus_fft_impl);
    }

    /// The same transform as `forward`, computed with the portable code on
    /// every target.
    ///
    /// It is the reference the vector versions are tested against.
    pub fn forward_scalar(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let p = self.forward_plan(input, output, window, overlap, shift, stride);
        forward_impl(&p, input, output, window, overlap, stride, opus_fft_impl_scalar);
    }

    /// Inverse MDCT of `n2 = (self.n >> shift) / 2` coefficients.
    ///
    /// Reads the coefficients from `input[0]`, `input[stride]`, ... and
    /// writes `output[overlap / 2..overlap / 2 + n2]`. Then it combines
    /// `output[..overlap]` with the window; the first `overlap / 2` values of
    /// that range are the tail the previous block left there.
    #[inline]
    pub fn backward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let p = self.backward_plan(input, output, window, overlap, shift, stride);

        // One choice of the implementation per transform.
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the arguments passed `backward_plan`.
            unsafe { backward_neon(&p, input, output, window, overlap, stride) }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::compat::x86_has_avx() {
                // SAFETY: the processor has AVX.
                unsafe { backward_avx(&p, input, output, window, overlap, stride) }
            } else {
                backward_impl(&p, input, output, window, overlap, stride, opus_fft_impl);
            }
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64"
        )))]
        backward_impl(&p, input, output, window, overlap, stride, opus_fft_impl);
    }

    /// The same transform as `backward`, computed with the portable code on
    /// every target.
    ///
    /// It is the reference the vector versions are tested against.
    pub fn backward_scalar(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let p = self.backward_plan(input, output, window, overlap, shift, stride);
        backward_impl(&p, input, output, window, overlap, stride, opus_fft_impl_scalar);
    }
}

// ---------------------------------------------------------------------------
// Portable transform.
//
// It runs on every target without a vector version. The vector versions are
// tested against it and must give the same bits. The operations and their
// order are those of libopus (mdct.c, float build); no product is fused with
// a sum.
// ---------------------------------------------------------------------------

/// Room for the `n4` complex values of one transform, not initialised.
type Work = [MaybeUninit<KissCpx>; MAX_N4];

/// The first `n4` entries of `buf` as a slice.
///
/// # Safety
/// The caller must write every entry before it reads it.
#[inline(always)]
unsafe fn work_slice(buf: &mut Work, n4: usize) -> &mut [KissCpx] {
    debug_assert!(n4 <= MAX_N4);
    core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut KissCpx, n4)
}

// The loops below walk several slices at once, some of them backwards. They
// are written as zips of slice and chunk iterators (`as_chunks`,
// `rchunks_exact`): the compiler turns such a zip into one counted loop
// without index checks. With `.rev()` it does not.

/// Every second value of `s`, upwards: `s[0]`, `s[2]`, ...
#[inline(always)]
fn up2(s: &[f32]) -> impl ExactSizeIterator<Item = f32> + '_ {
    s.as_chunks::<2>().0.iter().map(|c| c[0])
}

/// Every second value of `s`, downwards: `s[len - 1]`, `s[len - 3]`, ...
#[inline(always)]
fn down2(s: &[f32]) -> impl ExactSizeIterator<Item = f32> + '_ {
    s.rchunks_exact(2).map(|c| c[1])
}

/// The values of `s` from the last one to the first one.
#[inline(always)]
fn down1<T: Copy>(s: &[T]) -> impl ExactSizeIterator<Item = T> + '_ {
    s.rchunks_exact(1).map(|c| c[0])
}

/// Rotation in front of the forward FFT, with the FFT scale factor.
#[inline(always)]
fn forward_pre_rotate_one(re: f32, im: f32, t0: f32, t1: f32, scale: f32) -> KissCpx {
    let yr = re * t0 - im * t1;
    let yi = im * t0 + re * t1;
    KissCpx::new(yr * scale, yi * scale)
}

/// Rotation in front of the inverse FFT. `x1` and `x2` are the coefficients
/// `2 * i` and `n2 - 1 - 2 * i`.
#[inline(always)]
fn backward_pre_rotate_one(x1: f32, x2: f32, t0: f32, t1: f32) -> KissCpx {
    let yr = x2 * t0 + x1 * t1;
    let yi = x1 * t0 - x2 * t1;
    KissCpx::new(yi, yr)
}

/// Forward MDCT, first step: window, fold and rotate.
///
/// Writes the `n4` FFT inputs to `tmp` in natural order. `overlap` is
/// `4 * q`, `tmp.len()` is `n4 >= 2 * q`, `trig.len()` is `2 * n4`.
///
/// Butterfly `i` combines two samples from the front and two from the back
/// of the block. The first `q` and the last `q` butterflies use the window.
#[inline(always)]
fn forward_fold_rotate_scalar(
    input: &[f32],
    window: &[f32],
    trig: &[f32],
    tmp: &mut [KissCpx],
    q: usize,
    scale: f32,
) {
    let n4 = tmp.len();
    let n2 = 2 * n4;
    let (t0, t1) = trig.split_at(n4);
    let t1 = &t1[..n4];
    let (tmp_head, rest) = tmp.split_at_mut(q);
    let (tmp_mid, tmp_tail) = rest.split_at_mut(n4 - 2 * q);

    // The windowed butterflies read the first `4 * q` samples (`lo`), the
    // `4 * q` samples that start at `n2` (`hi`) and the window.
    let len = 4 * q;
    let lo = &input[..len];
    let hi = &input[n2..][..len];
    let w = &window[..len];

    // Butterflies 0..q: `u` walks up from the middle of the three slices,
    // `d` walks down from the middle.
    let rot = t0[..q].iter().zip(&t1[..q]);
    for (i, (o, (&t0, &t1))) in tmp_head.iter_mut().zip(rot).enumerate() {
        let u = 2 * q + 2 * i;
        let d = 2 * q - 1 - 2 * i;
        let re = hi[u] * w[d] + hi[d] * w[u];
        let im = lo[u] * w[u] - lo[d] * w[d];
        *o = forward_pre_rotate_one(re, im, t0, t1, scale);
    }

    // Butterflies q..n4 - q, no window: the real part walks down from sample
    // `n2 - 1`, the imaginary part walks up from sample `4 * q`.
    let mid = &input[len..n2];
    let rot = t0[q..n4 - q].iter().zip(&t1[q..n4 - q]);
    for (((o, re), im), (&t0, &t1)) in tmp_mid.iter_mut().zip(down2(mid)).zip(up2(mid)).zip(rot) {
        *o = forward_pre_rotate_one(re, im, t0, t1, scale);
    }

    // Butterflies n4 - q..n4: `u` walks up from the start of the three
    // slices, `d` walks down from their end.
    let rot = t0[n4 - q..].iter().zip(&t1[n4 - q..]);
    for (k, (o, (&t0, &t1))) in tmp_tail.iter_mut().zip(rot).enumerate() {
        let u = 2 * k;
        let d = len - 1 - 2 * k;
        let re = -lo[u] * w[u] + lo[d] * w[d];
        let im = hi[u] * w[d] + hi[d] * w[u];
        *o = forward_pre_rotate_one(re, im, t0, t1, scale);
    }
}

/// Copies the rotated values to the places the FFT expects them in.
/// `bitrev` is a permutation of `0..f2.len()`.
#[inline(always)]
fn scatter_scalar(tmp: &[KissCpx], bitrev: &[i16], f2: &mut [KissCpx]) {
    for (&v, &rev) in tmp.iter().zip(bitrev) {
        f2[rev as usize] = v;
    }
}

/// Forward MDCT, last step: rotate the FFT output and store it.
///
/// Butterfly `i` gives coefficient `2 * i` and coefficient `n2 - 1 - 2 * i`.
#[inline(always)]
fn forward_post_rotate_scalar(f2: &[KissCpx], trig: &[f32], output: &mut [f32], stride: usize) {
    let n4 = f2.len();
    let n2 = 2 * n4;
    let (t0, t1) = trig.split_at(n4);
    let t1 = &t1[..n4];
    if stride == 1 {
        // Coefficient `2 * k` comes from butterfly `k`, coefficient
        // `2 * k + 1` from butterfly `n4 - 1 - k`.
        let up = f2.iter().zip(t0).zip(t1);
        let down = down1(f2).zip(down1(t0)).zip(down1(t1));
        let pairs = output[..n2].as_chunks_mut::<2>().0.iter_mut();
        for ((o, ((fa, &a0), &a1)), ((fb, b0), b1)) in pairs.zip(up).zip(down) {
            o[0] = fa.i * a1 - fa.r * a0;
            o[1] = fb.r * b1 + fb.i * b0;
        }
    } else {
        for (i, ((fp, &t0), &t1)) in f2.iter().zip(t0).zip(t1).enumerate() {
            output[i * 2 * stride] = fp.i * t1 - fp.r * t0;
            output[stride * (n2 - 1 - 2 * i)] = fp.r * t1 + fp.i * t0;
        }
    }
}

/// Inverse MDCT, first step: rotate the coefficients.
///
/// Writes the `n4` FFT inputs to `tmp` in natural order. Butterfly `i` reads
/// coefficient `2 * i` and coefficient `n2 - 1 - 2 * i`.
#[inline(always)]
fn backward_pre_rotate_scalar(input: &[f32], trig: &[f32], tmp: &mut [KissCpx], stride: usize) {
    let n4 = tmp.len();
    let n2 = 2 * n4;
    let (t0, t1) = trig.split_at(n4);
    let t1 = &t1[..n4];
    let rot = t0.iter().zip(t1);
    if stride == 1 {
        let input = &input[..n2];
        for (((o, x1), x2), (&t0, &t1)) in tmp.iter_mut().zip(up2(input)).zip(down2(input)).zip(rot)
        {
            *o = backward_pre_rotate_one(x1, x2, t0, t1);
        }
    } else {
        for (i, (o, (&t0, &t1))) in tmp.iter_mut().zip(rot).enumerate() {
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            *o = backward_pre_rotate_one(x1, x2, t0, t1);
        }
    }
}

/// Inverse MDCT, step after the FFT: rotate and store `n2` samples.
///
/// Sample `2 * k` comes from butterfly `k`, sample `2 * k + 1` from
/// butterfly `n4 - 1 - k`. `out.len()` is `2 * f2.len()`.
#[inline(always)]
fn backward_post_rotate_scalar(f2: &[KissCpx], trig: &[f32], out: &mut [f32]) {
    let n4 = f2.len();
    let (t0, t1) = trig.split_at(n4);
    let t1 = &t1[..n4];
    let up = f2.iter().zip(t0).zip(t1);
    let down = down1(f2).zip(down1(t0)).zip(down1(t1));
    let pairs = out[..2 * n4].as_chunks_mut::<2>().0.iter_mut();
    for ((o, ((fa, &a0), &a1)), ((fb, b0), b1)) in pairs.zip(up).zip(down) {
        o[0] = fa.i * a0 + fa.r * a1;
        o[1] = fb.i * b1 - fb.r * b0;
    }
}

/// Inverse MDCT, last step: combine `out` with its mirror image through the
/// window (time-domain aliasing cancellation). `out` and `window` have the
/// length of the overlap.
#[inline(always)]
fn tdac_scalar(out: &mut [f32], window: &[f32]) {
    let overlap = out.len();
    let half = overlap / 2;
    let window = &window[..overlap];
    let (lo, hi) = out.split_at_mut(overlap - half);
    let (w_lo, w_hi) = window.split_at(overlap - half);
    let samples = lo[..half].iter_mut().zip(hi.rchunks_exact_mut(1));
    let weights = w_lo[..half].iter().zip(down1(w_hi));
    for ((front, back), (&wp1, wp2)) in samples.zip(weights) {
        let x2 = *front;
        let x1 = back[0];
        *front = x2 * wp2 - x1 * wp1;
        back[0] = x2 * wp1 + x1 * wp2;
    }
}

/// Forward MDCT with the portable steps. `fft` is the FFT to run.
#[inline(always)]
fn forward_impl(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
    fft: impl Fn(&KissFftState, &mut [KissCpx]),
) {
    let mut tmp_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    // SAFETY: the fold step writes all of `tmp`. The scatter step writes all
    // of `f2`, because the bit-reversal table is a permutation of `0..n4`.
    let tmp = unsafe { work_slice(&mut tmp_buf, p.n4) };
    let f2 = unsafe { work_slice(&mut f2_buf, p.n4) };

    forward_fold_rotate_scalar(input, window, p.trig, tmp, overlap / 4, p.st.scale());
    scatter_scalar(tmp, &p.st.bitrev, f2);
    fft(p.st, f2);
    forward_post_rotate_scalar(f2, p.trig, output, stride);
}

/// Inverse MDCT without its last step: rotation, FFT, rotation. Writes
/// `output[overlap / 2..overlap / 2 + n2]`. `fft` is the FFT to run.
#[inline(always)]
fn backward_rotations(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    overlap: usize,
    stride: usize,
    fft: impl Fn(&KissFftState, &mut [KissCpx]),
) {
    let mut tmp_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    // SAFETY: the rotation writes all of `tmp`. The scatter step writes all
    // of `f2`, because the bit-reversal table is a permutation of `0..n4`.
    let tmp = unsafe { work_slice(&mut tmp_buf, p.n4) };
    let f2 = unsafe { work_slice(&mut f2_buf, p.n4) };

    backward_pre_rotate_scalar(input, p.trig, tmp, stride);
    scatter_scalar(tmp, &p.st.bitrev, f2);
    fft(p.st, f2);
    let overlap2 = overlap / 2;
    backward_post_rotate_scalar(f2, p.trig, &mut output[overlap2..overlap2 + p.n2]);
}

/// Inverse MDCT with the portable steps. `fft` is the FFT to run.
#[inline(always)]
fn backward_impl(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
    fft: impl Fn(&KissFftState, &mut [KissCpx]),
) {
    backward_rotations(p, input, output, overlap, stride, fft);
    tdac_scalar(&mut output[..overlap], window);
}

/// The portable forward MDCT, compiled with AVX enabled.
///
/// AVX only changes how the compiler encodes and vectorizes the code. The
/// results are the same bits.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn forward_avx(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
) {
    forward_impl(p, input, output, window, overlap, stride, opus_fft_impl);
}

/// The portable inverse MDCT, compiled with AVX enabled, with the AVX
/// version of the last step.
///
/// AVX only changes how the compiler encodes and vectorizes the portable
/// steps. The results are the same bits.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn backward_avx(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
) {
    backward_rotations(p, input, output, overlap, stride, opus_fft_impl);
    mdct_tdac_avx(output, window, overlap);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_tdac_avx(output: &mut [f32], window: &[f32], overlap: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let overlap2 = overlap / 2;
    let mut i = 0usize;

    while i + 8 <= overlap2 {
        let x2 = _mm256_loadu_ps(output.as_ptr().add(i));

        let mut x1_tmp = [0f32; 8];
        let mut w2_tmp = [0f32; 8];
        for j in 0..8 {
            x1_tmp[j] = output[overlap - 1 - (i + j)];
            w2_tmp[j] = window[overlap - 1 - (i + j)];
        }
        let x1 = _mm256_loadu_ps(x1_tmp.as_ptr());

        let w1 = _mm256_loadu_ps(window.as_ptr().add(i));
        let w2 = _mm256_loadu_ps(w2_tmp.as_ptr());

        let out_fwd = _mm256_sub_ps(_mm256_mul_ps(x2, w2), _mm256_mul_ps(x1, w1));
        let out_rev = _mm256_add_ps(_mm256_mul_ps(x2, w1), _mm256_mul_ps(x1, w2));

        _mm256_storeu_ps(output.as_mut_ptr().add(i), out_fwd);

        let mut out_rev_tmp = [0f32; 8];
        _mm256_storeu_ps(out_rev_tmp.as_mut_ptr(), out_rev);
        for j in 0..8 {
            output[overlap - 1 - (i + j)] = out_rev_tmp[j];
        }

        i += 8;
    }

    for i in i..overlap2 {
        let x1 = output[overlap - 1 - i];
        let x2 = output[i];
        let wp1 = window[i];
        let wp2 = window[overlap - 1 - i];
        output[i] = x2 * wp2 - x1 * wp1;
        output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
    }
}

// ---------------------------------------------------------------------------
// NEON transform (aarch64).
//
// Each step gives the same bits as its portable version: every lane does the
// same multiplies, additions and subtractions in the same order, and no
// product is fused with a sum.
//
// The steps handle four butterflies at a time. When the number of butterflies
// is not a multiple of four, the last step is moved back so that it ends with
// the last butterfly. It then computes some butterflies a second time, which
// is harmless: no step reads what it writes.
// ---------------------------------------------------------------------------

/// Forward MDCT with the NEON steps.
///
/// # Safety
/// `p`, `input`, `output`, `window`, `overlap` and `stride` must have passed
/// `MdctLookup::forward_plan`.
#[cfg(target_arch = "aarch64")]
unsafe fn forward_neon(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
) {
    let n4 = p.n4;
    let q = overlap / 4;
    let mid = n4 - 2 * q;
    // The fold step has three groups of butterflies: `q` with the window,
    // `mid` without, `q` with the window. A group of one to three
    // butterflies is too short for a vector step.
    if n4 < 4 || (q != 0 && q < 4) || (mid != 0 && mid < 4) {
        forward_impl(p, input, output, window, overlap, stride, opus_fft_impl);
        return;
    }

    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    // SAFETY: the fold step writes all of `f2`, because the bit-reversal
    // table is a permutation of `0..n4`.
    let f2 = work_slice(&mut f2_buf, n4);

    // SAFETY: `forward_plan` checked that `overlap` is `4 * q`, that
    // `n4 >= 2 * q`, that `input` holds `2 * n4 + 4 * q` values, `window`
    // `4 * q` values and `output` `stride * (2 * n4 - 1) + 1` values. The plan
    // has `2 * n4` rotation values and `n4` bit-reversal entries below `n4`.
    // The group sizes were checked above.
    neon::forward_fold_rotate(input, window, p.trig, &p.st.bitrev, f2, q, p.st.scale());
    opus_fft_impl(p.st, f2);
    neon::forward_post_rotate(f2, p.trig, output, stride);
}

/// Inverse MDCT with the NEON steps.
///
/// # Safety
/// `p`, `input`, `output`, `window`, `overlap` and `stride` must have passed
/// `MdctLookup::backward_plan`.
#[cfg(target_arch = "aarch64")]
unsafe fn backward_neon(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
) {
    let n4 = p.n4;
    if n4 < 4 {
        backward_rotations(p, input, output, overlap, stride, opus_fft_impl);
        tdac_scalar(&mut output[..overlap], window);
        return;
    }

    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    // SAFETY: the rotation writes all of `f2`, because the bit-reversal table
    // is a permutation of `0..n4`.
    let f2 = work_slice(&mut f2_buf, n4);

    // SAFETY: `backward_plan` checked that `input` holds
    // `stride * (2 * n4 - 1) + 1` values and that `output` holds
    // `overlap / 2 + 2 * n4` values. The plan has `2 * n4` rotation values and
    // `n4` bit-reversal entries below `n4`. `n4 >= 4` was checked above.
    neon::backward_pre_rotate(input, p.trig, &p.st.bitrev, f2, stride);
    opus_fft_impl(p.st, f2);
    let overlap2 = overlap / 2;
    neon::backward_post_rotate(f2, p.trig, &mut output[overlap2..overlap2 + p.n2]);
    mdct_tdac_neon(output, window, overlap);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::KissCpx;
    use core::arch::aarch64::*;

    /// `[p[0], p[2], p[4], p[6]]`.
    ///
    /// # Safety
    /// `p[0..8]` must be readable.
    #[inline(always)]
    unsafe fn up4(p: *const f32) -> float32x4_t {
        vuzp1q_f32(vld1q_f32(p), vld1q_f32(p.add(4)))
    }

    /// `[p[0], p[-2], p[-4], p[-6]]`.
    ///
    /// # Safety
    /// `p[-7..=0]` must be readable.
    #[inline(always)]
    unsafe fn down4(p: *const f32) -> float32x4_t {
        let lo = vld1q_f32(p.sub(7));
        let hi = vld1q_f32(p.sub(3));
        // uzp2(hi, lo) is [p[-2], p[0], p[-6], p[-4]].
        vrev64q_f32(vuzp2q_f32(hi, lo))
    }

    /// The four lanes in reverse order.
    #[inline(always)]
    unsafe fn rev4(v: float32x4_t) -> float32x4_t {
        let r = vrev64q_f32(v);
        vextq_f32(r, r, 2)
    }

    /// Stores the complex values `(a[k], b[k])` at `f2[bitrev[k]]`, k = 0..4.
    ///
    /// # Safety
    /// `bitrev[0..4]` must be readable and each entry must be the index of a
    /// complex value inside the buffer `f2` points to.
    #[inline(always)]
    unsafe fn store_bitrev4(a: float32x4_t, b: float32x4_t, bitrev: *const i16, f2: *mut f32) {
        let lo = vzip1q_f32(a, b);
        let hi = vzip2q_f32(a, b);
        vst1_f32(f2.add(2 * *bitrev as usize), vget_low_f32(lo));
        vst1_f32(f2.add(2 * *bitrev.add(1) as usize), vget_high_f32(lo));
        vst1_f32(f2.add(2 * *bitrev.add(2) as usize), vget_low_f32(hi));
        vst1_f32(f2.add(2 * *bitrev.add(3) as usize), vget_high_f32(hi));
    }

    /// Forward MDCT, first step: window, fold, rotate and store in
    /// bit-reversed order, in one pass.
    ///
    /// # Safety
    /// With `n4 = f2.len()`:
    /// - `n4 >= 2 * q`, and each of `q` and `n4 - 2 * q` is 0 or at least 4
    /// - `input` holds at least `2 * n4 + 4 * q` values
    /// - `window` holds at least `4 * q` values
    /// - `trig` holds at least `2 * n4` values
    /// - `bitrev` holds `n4` entries, each of them below `n4`
    pub(super) unsafe fn forward_fold_rotate(
        input: &[f32],
        window: &[f32],
        trig: &[f32],
        bitrev: &[i16],
        f2: &mut [KissCpx],
        q: usize,
        scale: f32,
    ) {
        let n4 = f2.len();
        let n2 = 2 * n4;
        debug_assert!(n4 >= 2 * q);
        debug_assert!(q == 0 || q >= 4);
        debug_assert!(n4 == 2 * q || n4 - 2 * q >= 4);
        debug_assert!(input.len() >= n2 + 4 * q && window.len() >= 4 * q);
        debug_assert!(trig.len() >= n2 && bitrev.len() == n4);
        debug_assert!(bitrev.iter().all(|&r| (r as usize) < n4));

        let inp = input.as_ptr();
        let win = window.as_ptr();
        let tr = trig.as_ptr();
        let br = bitrev.as_ptr();
        let out = f2.as_mut_ptr() as *mut f32;
        let vscale = vdupq_n_f32(scale);

        // Rotates the butterflies `i..i + 4` and stores them.
        // SAFETY (for every call below): `i + 4 <= n4`.
        let rotate_store = |re: float32x4_t, im: float32x4_t, i: usize| {
            let t0 = vld1q_f32(tr.add(i));
            let t1 = vld1q_f32(tr.add(n4 + i));
            let yr = vsubq_f32(vmulq_f32(re, t0), vmulq_f32(im, t1));
            let yi = vaddq_f32(vmulq_f32(im, t0), vmulq_f32(re, t1));
            store_bitrev4(
                vmulq_f32(yr, vscale),
                vmulq_f32(yi, vscale),
                br.add(i),
                out,
            );
        };

        // Butterflies 0..q. `u` walks up and `d` walks down from the middle
        // of the first `4 * q` samples, of the `4 * q` samples at `n2`, and
        // of the window.
        // SAFETY: `i <= q - 4`, so `u + 8 <= 4 * q` and `d >= 7`.
        let mut start = 0;
        while start < q {
            let i = start.min(q - 4);
            let u = 2 * q + 2 * i;
            let d = 2 * q - 1 - 2 * i;
            let w1 = up4(win.add(u));
            let w2 = down4(win.add(d));
            let re = vaddq_f32(
                vmulq_f32(up4(inp.add(n2 + u)), w2),
                vmulq_f32(down4(inp.add(n2 + d)), w1),
            );
            let im = vsubq_f32(
                vmulq_f32(up4(inp.add(u)), w1),
                vmulq_f32(down4(inp.add(d)), w2),
            );
            rotate_store(re, im, i);
            start += 4;
        }

        // Butterflies q..n4 - q, no window. The real part walks down from
        // sample `n2 - 1`, the imaginary part walks up from sample `4 * q`.
        // SAFETY: `q <= i <= n4 - q - 4`, so the real parts are read from
        // `4 * q..` and the imaginary parts end before `n2`.
        let mut start = q;
        while start < n4 - q {
            let i = start.min(n4 - q - 4);
            let re = down4(inp.add(n2 - 1 + 2 * q - 2 * i));
            let im = up4(inp.add(2 * q + 2 * i));
            rotate_store(re, im, i);
            start += 4;
        }

        // Butterflies n4 - q..n4. `u` walks up from the start and `d` walks
        // down from the end of the same three ranges as in the first loop.
        // SAFETY: `k <= q - 4`, so `u + 8 <= 2 * q` and `d >= 2 * q + 7`.
        let mut start = 0;
        while start < q {
            let k = start.min(q - 4);
            let u = 2 * k;
            let d = 4 * q - 1 - 2 * k;
            let w1 = up4(win.add(u));
            let w2 = down4(win.add(d));
            let re = vaddq_f32(
                vmulq_f32(vnegq_f32(up4(inp.add(u))), w1),
                vmulq_f32(down4(inp.add(d)), w2),
            );
            let im = vaddq_f32(
                vmulq_f32(up4(inp.add(n2 + u)), w2),
                vmulq_f32(down4(inp.add(n2 + d)), w1),
            );
            rotate_store(re, im, n4 - q + k);
            start += 4;
        }
    }

    /// Forward MDCT, last step: rotate the FFT output and store it.
    ///
    /// # Safety
    /// With `n4 = f2.len()`: `n4 >= 4`, `stride >= 1`, `trig` holds at least
    /// `2 * n4` values and `output` at least `stride * (2 * n4 - 1) + 1`.
    pub(super) unsafe fn forward_post_rotate(
        f2: &[KissCpx],
        trig: &[f32],
        output: &mut [f32],
        stride: usize,
    ) {
        let n4 = f2.len();
        let n2 = 2 * n4;
        debug_assert!(n4 >= 4 && stride >= 1);
        debug_assert!(trig.len() >= n2 && output.len() > stride * (n2 - 1));

        let f = f2.as_ptr() as *const f32;
        let tr = trig.as_ptr();
        let out = output.as_mut_ptr();

        // The two outputs of the butterflies `i..i + 4`.
        // SAFETY (for every call below): `i + 4 <= n4`.
        let rotate = |i: usize| {
            let c = vld2q_f32(f.add(2 * i));
            let t0 = vld1q_f32(tr.add(i));
            let t1 = vld1q_f32(tr.add(n4 + i));
            (
                vsubq_f32(vmulq_f32(c.1, t1), vmulq_f32(c.0, t0)),
                vaddq_f32(vmulq_f32(c.0, t1), vmulq_f32(c.1, t0)),
            )
        };

        if stride == 1 {
            // Coefficient `2 * k` comes from butterfly `k`, coefficient
            // `2 * k + 1` from butterfly `n4 - 1 - k`. One step takes four
            // butterflies from the front and their four mirror images from
            // the back, and writes eight coefficients at each end.
            // SAFETY: `i + 4 <= n4` and `j + 4 <= n4`.
            let mut i = 0;
            loop {
                let j = n4 - 4 - i;
                let (yr_a, yi_a) = rotate(i);
                let (yr_b, yi_b) = rotate(j);
                vst2q_f32(out.add(2 * i), float32x4x2_t(yr_a, rev4(yi_b)));
                vst2q_f32(out.add(2 * j), float32x4x2_t(yr_b, rev4(yi_a)));
                // Done when the two blocks have met.
                if 2 * i + 8 >= n4 {
                    break;
                }
                i += 4;
            }
        } else {
            // Butterfly `i` gives coefficient `2 * i` and coefficient
            // `n2 - 1 - 2 * i`. They are `stride` values apart in `output`.
            // SAFETY: `i + 3 < n4`, so the largest index written is at most
            // `stride * (n2 - 1)`.
            let mut start = 0;
            while start < n4 {
                let i = start.min(n4 - 4);
                let (yr, yi) = rotate(i);
                let even = out.add(2 * i * stride);
                let odd = out.add(stride * (n2 - 1 - 2 * i));
                vst1q_lane_f32::<0>(even, yr);
                vst1q_lane_f32::<1>(even.add(2 * stride), yr);
                vst1q_lane_f32::<2>(even.add(4 * stride), yr);
                vst1q_lane_f32::<3>(even.add(6 * stride), yr);
                vst1q_lane_f32::<0>(odd, yi);
                vst1q_lane_f32::<1>(odd.sub(2 * stride), yi);
                vst1q_lane_f32::<2>(odd.sub(4 * stride), yi);
                vst1q_lane_f32::<3>(odd.sub(6 * stride), yi);
                start += 4;
            }
        }
    }

    /// Inverse MDCT, first step: rotate the coefficients and store the
    /// result in bit-reversed order.
    ///
    /// # Safety
    /// With `n4 = f2.len()`: `n4 >= 4`, `stride >= 1`, `input` holds at least
    /// `stride * (2 * n4 - 1) + 1` values, `trig` at least `2 * n4`, and
    /// `bitrev` holds `n4` entries, each of them below `n4`.
    pub(super) unsafe fn backward_pre_rotate(
        input: &[f32],
        trig: &[f32],
        bitrev: &[i16],
        f2: &mut [KissCpx],
        stride: usize,
    ) {
        let n4 = f2.len();
        let n2 = 2 * n4;
        debug_assert!(n4 >= 4 && stride >= 1);
        debug_assert!(input.len() > stride * (n2 - 1));
        debug_assert!(trig.len() >= n2 && bitrev.len() == n4);
        debug_assert!(bitrev.iter().all(|&r| (r as usize) < n4));

        let inp = input.as_ptr();
        let tr = trig.as_ptr();
        let br = bitrev.as_ptr();
        let out = f2.as_mut_ptr() as *mut f32;

        // Butterfly `i` reads coefficient `2 * i` (`x1`) and coefficient
        // `n2 - 1 - 2 * i` (`x2`).
        // SAFETY: `i + 4 <= n4`, so all coefficients read are below `n2`.
        let mut start = 0;
        while start < n4 {
            let i = start.min(n4 - 4);
            let (x1, x2) = if stride == 1 {
                (up4(inp.add(2 * i)), down4(inp.add(n2 - 1 - 2 * i)))
            } else {
                let even = inp.add(2 * i * stride);
                let odd = inp.add(stride * (n2 - 1 - 2 * i));
                let x1 = vld1q_dup_f32(even);
                let x1 = vld1q_lane_f32::<1>(even.add(2 * stride), x1);
                let x1 = vld1q_lane_f32::<2>(even.add(4 * stride), x1);
                let x1 = vld1q_lane_f32::<3>(even.add(6 * stride), x1);
                let x2 = vld1q_dup_f32(odd);
                let x2 = vld1q_lane_f32::<1>(odd.sub(2 * stride), x2);
                let x2 = vld1q_lane_f32::<2>(odd.sub(4 * stride), x2);
                let x2 = vld1q_lane_f32::<3>(odd.sub(6 * stride), x2);
                (x1, x2)
            };
            let t0 = vld1q_f32(tr.add(i));
            let t1 = vld1q_f32(tr.add(n4 + i));
            let yr = vaddq_f32(vmulq_f32(x2, t0), vmulq_f32(x1, t1));
            let yi = vsubq_f32(vmulq_f32(x1, t0), vmulq_f32(x2, t1));
            store_bitrev4(yi, yr, br.add(i), out);
            start += 4;
        }
    }

    /// Inverse MDCT, step after the FFT: rotate and store `n2` samples.
    ///
    /// # Safety
    /// With `n4 = f2.len()`: `n4 >= 4`, `trig` holds at least `2 * n4` values
    /// and `out` at least `2 * n4`.
    pub(super) unsafe fn backward_post_rotate(f2: &[KissCpx], trig: &[f32], out: &mut [f32]) {
        let n4 = f2.len();
        debug_assert!(n4 >= 4);
        debug_assert!(trig.len() >= 2 * n4 && out.len() >= 2 * n4);

        let f = f2.as_ptr() as *const f32;
        let tr = trig.as_ptr();
        let out = out.as_mut_ptr();

        // The two outputs of the butterflies `i..i + 4`. The FFT output has
        // the real part of the rotation input in `.i` and the imaginary part
        // in `.r`.
        // SAFETY (for every call below): `i + 4 <= n4`.
        let rotate = |i: usize| {
            let c = vld2q_f32(f.add(2 * i));
            let t0 = vld1q_f32(tr.add(i));
            let t1 = vld1q_f32(tr.add(n4 + i));
            (
                vaddq_f32(vmulq_f32(c.1, t0), vmulq_f32(c.0, t1)),
                vsubq_f32(vmulq_f32(c.1, t1), vmulq_f32(c.0, t0)),
            )
        };

        // Sample `2 * k` comes from butterfly `k`, sample `2 * k + 1` from
        // butterfly `n4 - 1 - k`. One step takes four butterflies from the
        // front and their four mirror images from the back, and writes eight
        // samples at each end.
        // SAFETY: `i + 4 <= n4` and `j + 4 <= n4`.
        let mut i = 0;
        loop {
            let j = n4 - 4 - i;
            let (yr_a, yi_a) = rotate(i);
            let (yr_b, yi_b) = rotate(j);
            vst2q_f32(out.add(2 * i), float32x4x2_t(yr_a, rev4(yi_b)));
            vst2q_f32(out.add(2 * j), float32x4x2_t(yr_b, rev4(yi_a)));
            // Done when the two blocks have met.
            if 2 * i + 8 >= n4 {
                break;
            }
            i += 4;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mdct_tdac_neon(output: &mut [f32], window: &[f32], overlap: usize) {
    use core::arch::aarch64::*;

    let overlap2 = overlap / 2;
    if overlap2 < 4 {
        for i in 0..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];
            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
        return;
    }

    unsafe {
        let out_ptr = output.as_mut_ptr();
        let win_ptr = window.as_ptr();
        let n4 = overlap2 & !3;
        let mut i = 0;

        while i < n4 {
            let x2_fwd = vld1q_f32(out_ptr.add(i));
            let x1_rev = vld1q_f32(out_ptr.add(overlap - 4 - i));

            let x1 = vrev64q_f32(x1_rev);
            let x1 = vextq_f32(x1, x1, 2);

            let wp1_fwd = vld1q_f32(win_ptr.add(i));
            let wp2_rev = vld1q_f32(win_ptr.add(overlap - 4 - i));
            let wp2 = vrev64q_f32(wp2_rev);
            let wp2 = vextq_f32(wp2, wp2, 2);
            let wp1 = wp1_fwd;

            let out_fwd = vsubq_f32(vmulq_f32(x2_fwd, wp2), vmulq_f32(x1, wp1));

            let out_rev = vaddq_f32(vmulq_f32(x2_fwd, wp1), vmulq_f32(x1, wp2));

            let out_rev = vrev64q_f32(out_rev);
            let out_rev = vextq_f32(out_rev, out_rev, 2);

            vst1q_f32(out_ptr.add(i), out_fwd);
            vst1q_f32(out_ptr.add(overlap - 4 - i), out_rev);

            i += 4;
        }

        for i in n4..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            output[i] = x2 * window[overlap - 1 - i] - x1 * window[i];
            output[overlap - 1 - i] = x2 * window[i] + x1 * window[overlap - 1 - i];
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod mdct_tests {
    #[test]
    fn test_mdct_backward_transient_no_blowup() {
        let mode = crate::modes::default_mode();
        let shift = 3;
        let n = mode.mdct.n >> shift; // 120
        let overlap = mode.overlap; // 120
        let stride = 8;

        let frame_size = 960usize;
        let mut freq = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            freq[i] = ((i as f32) * 0.01).sin() * 10.0;
        }

        let out_len = n + overlap; // 240
        let mut output0 = vec![0.0f32; out_len];
        let mut output1 = vec![0.0f32; out_len];

        mode.mdct.backward(
            &freq[0..],
            &mut output0,
            mode.window,
            overlap,
            shift,
            stride,
        );
        mode.mdct.backward(
            &freq[1..],
            &mut output1,
            mode.window,
            overlap,
            shift,
            stride,
        );

        let max0 = output0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max1 = output1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(max0.abs() < 500.0, "sub0 blowup: {}", max0);
        assert!(max1.abs() < 500.0, "sub1 blowup: {}", max1);
    }

    #[test]
    fn test_mdct_backward_stride1_neon_matches_scalar() {
        let mode = crate::modes::default_mode();
        let shift = 0; // non-transient full-size MDCT
        let n = mode.mdct.n >> shift; // 1920
        let n2 = n / 2; // 960
        let n4 = n / 4; // 480
        let overlap = mode.overlap; // 120
        let overlap2 = overlap / 2; // 60
        let stride = 1;

        let freq_len = n2;
        let mut freq = vec![0.0f32; freq_len + 4];
        for i in 0..freq_len {
            freq[i] = ((i as f32) * 0.01).sin() * 4577.0;
        }

        let out_len = overlap2 + n2; // 60 + 960 = 1020
        let mut output_hw = vec![0.0f32; out_len + 100];
        mode.mdct.backward(
            &freq[..],
            &mut output_hw,
            mode.window,
            overlap,
            shift,
            stride,
        );

        let st = mode.mdct.kfft[shift].as_ref().unwrap();
        let (trig, _) = mode.mdct.get_trig(shift);

        use crate::kiss_fft::KissCpx;
        let mut f2 = vec![KissCpx::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = freq[2 * i * stride];
            let x2 = freq[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        crate::kiss_fft::opus_fft_impl(st, &mut f2);

        let mut output_scalar = vec![0.0f32; out_len + 100];
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;
            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;
            output_scalar[overlap2 + 2 * i] = yr0;
            output_scalar[overlap2 + n2 - 1 - 2 * i] = yi0;
            output_scalar[overlap2 + n2 - 2 - 2 * i] = yr1;
            output_scalar[overlap2 + 2 * i + 1] = yi1;
        }
        // TDAC
        for i in 0..overlap2 {
            let x1 = output_scalar[overlap - 1 - i];
            let x2 = output_scalar[i];
            let wp1 = mode.window[i];
            let wp2 = mode.window[overlap - 1 - i];
            output_scalar[i] = x2 * wp2 - x1 * wp1;
            output_scalar[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }

        let max_diff = output_hw[..out_len]
            .iter()
            .zip(output_scalar[..out_len].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.5,
            "stride=1 NEON vs scalar mismatch: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_mdct_backward_neon_matches_scalar() {
        let mode = crate::modes::default_mode();
        let shift = 3;
        let n = mode.mdct.n >> shift; // 240
        let n2 = n / 2; // 120
        let n4 = n / 4; // 60
        let overlap = mode.overlap; // 120
        let overlap2 = overlap / 2; // 60
        let stride = 8;

        // Build a realistic freq vector (sine wave @ 440Hz)
        let frame_size = 960usize;
        let mut freq = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            freq[i] = ((i as f32) * 0.01).sin() * 200.0;
        }

        let out_len = n + overlap; // 360
        let mut output_hw = vec![0.0f32; out_len];
        mode.mdct.backward(
            &freq[0..],
            &mut output_hw,
            mode.window,
            overlap,
            shift,
            stride,
        );

        // Scalar reference
        let st = mode.mdct.kfft[shift].as_ref().unwrap();
        let (trig, _) = mode.mdct.get_trig(shift);

        use crate::kiss_fft::KissCpx;
        let mut f2 = vec![KissCpx::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = freq[2 * i * stride];
            let x2 = freq[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        crate::kiss_fft::opus_fft_impl(st, &mut f2);

        let mut output_scalar = vec![0.0f32; out_len];
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;
            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;
            output_scalar[overlap2 + 2 * i] = yr0;
            output_scalar[overlap2 + n2 - 1 - 2 * i] = yi0;
            output_scalar[overlap2 + n2 - 2 - 2 * i] = yr1;
            output_scalar[overlap2 + 2 * i + 1] = yi1;
        }
        // TDAC
        for i in 0..overlap2 {
            let x1 = output_scalar[overlap - 1 - i];
            let x2 = output_scalar[i];
            let wp1 = mode.window[i];
            let wp2 = mode.window[overlap - 1 - i];
            output_scalar[i] = x2 * wp2 - x1 * wp1;
            output_scalar[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }

        for i in 0..out_len {
            let diff = (output_hw[i] - output_scalar[i]).abs();
            if diff > 1e-3 {
                eprintln!(
                    "Mismatch at output[{}]: hw={} scalar={} diff={}",
                    i, output_hw[i], output_scalar[i], diff
                );
            }
        }
        let max_diff = output_hw
            .iter()
            .zip(output_scalar.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1,
            "NEON/HW vs scalar mismatch: max_diff={}",
            max_diff
        );
    }
}

#[cfg(all(test, feature = "std"))]
mod bit_exact_tests {
    use super::*;

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
    fn fill(rng: &mut Lcg, out: &mut [f32], class: usize) {
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
        for v in out.iter_mut() {
            *v = match class {
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
        }
    }

    /// Index of the first value that differs. Any NaN equals any other NaN:
    /// the sign and the payload of a NaN are not part of what the codec
    /// computes.
    fn first_difference(a: &[f32], b: &[f32]) -> Option<usize> {
        assert_eq!(a.len(), b.len());
        (0..a.len())
            .find(|&i| a[i].to_bits() != b[i].to_bits() && !(a[i].is_nan() && b[i].is_nan()))
    }

    /// The forward MDCT as it was before the rewrite: the loops of libopus
    /// (`clt_mdct_forward_c`, float build) with plain indexing.
    fn reference_forward(
        l: &MdctLookup,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let st = l.kfft[shift].as_ref().unwrap();
        let n = l.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let scale = st.scale();
        let (trig, _) = l.get_trig(shift);
        let overlap2 = overlap / 2;
        let mut f = vec![0f32; n2];
        let mut f2 = vec![KissCpx::default(); n4];

        let mut yp = 0usize;
        let mut xp1 = overlap2;
        let mut xp2 = n2 - 1 + overlap2;
        let mut wp1 = overlap2;
        let mut wp2 = overlap2.saturating_sub(1);
        let limit = overlap.div_ceil(4);
        let mid = n4.saturating_sub(limit);

        for _ in 0..limit.min(n4) {
            let w1 = window[wp1];
            let w2 = window[wp2];
            f[yp] = input[xp1 + n2] * w2 + input[xp2] * w1;
            yp += 1;
            f[yp] = input[xp1] * w1 - input[xp2 - n2] * w2;
            yp += 1;
            xp1 += 2;
            xp2 -= 2;
            wp1 += 2;
            wp2 = wp2.saturating_sub(2);
        }
        for _ in limit..mid {
            f[yp] = input[xp2];
            yp += 1;
            f[yp] = input[xp1];
            yp += 1;
            xp1 += 2;
            // Without a window this is the last use of `xp2`, and the step
            // would go below zero.
            xp2 = xp2.wrapping_sub(2);
        }
        let mut wp1 = 0usize;
        let mut wp2 = overlap.saturating_sub(1);
        for _ in 0..n4.saturating_sub(mid) {
            let w1 = window[wp1];
            let w2 = window[wp2];
            f[yp] = -input[xp1 - n2] * w1 + input[xp2] * w2;
            yp += 1;
            f[yp] = input[xp1] * w2 + input[xp2 + n2] * w1;
            yp += 1;
            xp1 += 2;
            xp2 = xp2.wrapping_sub(2);
            wp1 += 2;
            wp2 = wp2.wrapping_sub(2);
        }

        for i in 0..n4 {
            let re = f[2 * i];
            let im = f[2 * i + 1];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;
            f2[st.bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
        }

        opus_fft_impl_scalar(st, &mut f2);

        for i in 0..n4 {
            let fp = &f2[i];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;
            output[i * 2 * stride] = yr;
            output[stride * (n2 - 1 - 2 * i)] = yi;
        }
    }

    /// The inverse MDCT as it was before the rewrite: the loops of libopus
    /// (`clt_mdct_backward_c`, float build) with plain indexing.
    fn reference_backward(
        l: &MdctLookup,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let st = l.kfft[shift].as_ref().unwrap();
        let n = l.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let overlap2 = overlap / 2;
        let (trig, _) = l.get_trig(shift);
        let mut f2 = vec![KissCpx::default(); n4];

        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }

        opus_fft_impl_scalar(st, &mut f2);

        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            output[overlap2 + 2 * i] = yr0;
            output[overlap2 + n2 - 1 - 2 * i] = yi0;
            output[overlap2 + n2 - 2 - 2 * i] = yr1;
            output[overlap2 + 2 * i + 1] = yi1;
        }

        for i in 0..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];
            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
    }

    /// One transform shape to test.
    struct Case {
        /// Transform size given to `MdctLookup::new`.
        n: usize,
        max_lm: usize,
        shift: usize,
        stride: usize,
        overlap: usize,
    }

    /// The shapes the codec uses, and shapes that exercise every path of the
    /// vector code: group sizes that are not a multiple of four, groups too
    /// short for a vector step, no window at all.
    fn cases() -> Vec<Case> {
        let mut v = Vec::new();
        // The codec: long blocks of every frame size, and 2, 4 or 8 short
        // blocks whose coefficients are interleaved.
        for shift in 0..=3 {
            v.push(Case { n: 1920, max_lm: 3, shift, stride: 1, overlap: 120 });
        }
        for stride in [2, 4, 8] {
            v.push(Case { n: 1920, max_lm: 3, shift: 3, stride, overlap: 120 });
        }
        // Other strides and levels.
        v.push(Case { n: 1920, max_lm: 3, shift: 2, stride: 3, overlap: 120 });
        v.push(Case { n: 1920, max_lm: 3, shift: 1, stride: 5, overlap: 120 });
        // 64 butterflies with windowed groups of 0 to 32 butterflies.
        for overlap in [0, 4, 8, 12, 16, 20, 28, 32, 64, 116, 120, 124, 128] {
            v.push(Case { n: 256, max_lm: 0, shift: 0, stride: 1, overlap });
            v.push(Case { n: 256, max_lm: 0, shift: 0, stride: 3, overlap });
        }
        // Numbers of butterflies that are not a multiple of four, down to one.
        for (n, overlaps) in [
            (120usize, &[0usize, 16, 20, 24, 40, 60][..]),
            (60, &[0, 4, 16, 28][..]),
            (40, &[0, 16, 20][..]),
            (24, &[0, 4, 12][..]),
            (20, &[0, 8][..]),
            (16, &[0, 8][..]),
            (12, &[0, 4][..]),
            (8, &[0, 4][..]),
            (4, &[0][..]),
        ] {
            for &overlap in overlaps {
                v.push(Case { n, max_lm: 0, shift: 0, stride: 1, overlap });
                v.push(Case { n, max_lm: 0, shift: 0, stride: 2, overlap });
            }
        }
        v
    }

    /// A window of `overlap` values. The transforms accept any values.
    fn test_window(rng: &mut Lcg, overlap: usize) -> Vec<f32> {
        (0..overlap).map(|_| rng.unit()).collect()
    }

    /// `forward`, `forward_scalar` and the old loops give the same bits. The
    /// output slice has exactly the length the transform needs, and the
    /// values between the coefficients stay as they are.
    #[test]
    fn forward_matches_scalar_and_reference() {
        let mut rng = Lcg(0x5eed_1001);
        for c in cases() {
            let l = MdctLookup::new(c.n, c.max_lm);
            let n2 = (c.n >> c.shift) / 2;
            let window = test_window(&mut rng, c.overlap);
            for class in 0..CLASSES {
                let mut input = vec![0f32; n2 + c.overlap];
                fill(&mut rng, &mut input, class);
                let mut before = vec![0f32; c.stride * (n2 - 1) + 1];
                fill(&mut rng, &mut before, 0);

                let mut fast = before.clone();
                let mut scalar = before.clone();
                let mut reference = before.clone();
                l.forward(&input, &mut fast, &window, c.overlap, c.shift, c.stride);
                l.forward_scalar(&input, &mut scalar, &window, c.overlap, c.shift, c.stride);
                reference_forward(
                    &l,
                    &input,
                    &mut reference,
                    &window,
                    c.overlap,
                    c.shift,
                    c.stride,
                );

                let what = format!(
                    "n {} shift {} stride {} overlap {} class {}",
                    c.n, c.shift, c.stride, c.overlap, class
                );
                assert_eq!(first_difference(&scalar, &reference), None, "scalar, {what}");
                assert_eq!(first_difference(&fast, &scalar), None, "fast, {what}");
            }
        }
    }

    /// `backward`, `backward_scalar` and the old loops give the same bits.
    /// The slices have exactly the lengths the transform needs.
    #[test]
    fn backward_matches_scalar_and_reference() {
        let mut rng = Lcg(0x5eed_1002);
        for c in cases() {
            let l = MdctLookup::new(c.n, c.max_lm);
            let n2 = (c.n >> c.shift) / 2;
            let window = test_window(&mut rng, c.overlap);
            for class in 0..CLASSES {
                let mut input = vec![0f32; c.stride * (n2 - 1) + 1];
                fill(&mut rng, &mut input, class);
                // The first half of the overlap holds the tail of the block
                // before; the transform reads it.
                let mut before = vec![0f32; c.overlap / 2 + n2];
                fill(&mut rng, &mut before, class);

                let mut fast = before.clone();
                let mut scalar = before.clone();
                let mut reference = before.clone();
                l.backward(&input, &mut fast, &window, c.overlap, c.shift, c.stride);
                l.backward_scalar(&input, &mut scalar, &window, c.overlap, c.shift, c.stride);
                reference_backward(
                    &l,
                    &input,
                    &mut reference,
                    &window,
                    c.overlap,
                    c.shift,
                    c.stride,
                );

                let what = format!(
                    "n {} shift {} stride {} overlap {} class {}",
                    c.n, c.shift, c.stride, c.overlap, class
                );
                assert_eq!(first_difference(&scalar, &reference), None, "scalar, {what}");
                assert_eq!(first_difference(&fast, &scalar), None, "fast, {what}");
            }
        }
    }

    /// Samples behind the range a transform owns are not written.
    #[test]
    fn transforms_stay_inside_their_output() {
        let mut rng = Lcg(0x5eed_1003);
        let guard = 16;
        for c in cases() {
            let l = MdctLookup::new(c.n, c.max_lm);
            let n2 = (c.n >> c.shift) / 2;
            let window = test_window(&mut rng, c.overlap);

            let mut input = vec![0f32; n2 + c.overlap];
            fill(&mut rng, &mut input, 0);
            let used = c.stride * (n2 - 1) + 1;
            let mut out = vec![0f32; used + guard];
            fill(&mut rng, &mut out, 0);
            let before = out.clone();
            l.forward(&input, &mut out, &window, c.overlap, c.shift, c.stride);
            assert_eq!(
                first_difference(&out[used..], &before[used..]),
                None,
                "forward n {} stride {} overlap {}",
                c.n,
                c.stride,
                c.overlap
            );
            // With a stride, only every `stride`-th value is a coefficient.
            for (i, (a, b)) in out[..used].iter().zip(&before[..used]).enumerate() {
                if i % c.stride != 0 {
                    assert_eq!(a.to_bits(), b.to_bits(), "forward wrote between coefficients");
                }
            }

            let mut coeffs = vec![0f32; c.stride * (n2 - 1) + 1];
            fill(&mut rng, &mut coeffs, 0);
            let used = c.overlap / 2 + n2;
            let mut out = vec![0f32; used + guard];
            fill(&mut rng, &mut out, 0);
            let before = out.clone();
            l.backward(&coeffs, &mut out, &window, c.overlap, c.shift, c.stride);
            assert_eq!(
                first_difference(&out[used..], &before[used..]),
                None,
                "backward n {} stride {} overlap {}",
                c.n,
                c.stride,
                c.overlap
            );
        }
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn forward_rejects_an_overlap_that_is_not_a_multiple_of_four() {
        let l = MdctLookup::new(256, 0);
        let input = vec![0f32; 256];
        let mut output = vec![0f32; 128];
        let window = vec![0f32; 64];
        l.forward(&input, &mut output, &window, 30, 0, 1);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn forward_rejects_a_short_output_for_a_stride() {
        let l = MdctLookup::new(256, 0);
        let input = vec![0f32; 256];
        let mut output = vec![0f32; 128];
        let window = vec![0f32; 64];
        l.forward(&input, &mut output, &window, 64, 0, 2);
    }
}
