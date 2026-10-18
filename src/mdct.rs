use crate::kiss_fft::{
    Arith, Fused, KissCpx, KissFftState, Separate, opus_fft_impl, opus_fft_impl_fused,
    opus_fft_impl_scalar,
};
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

/// True when a slice of `len` values holds index `stride * (n2 - 1)`: the
/// last one of `n2` values that are `stride` apart. `n2` is at least 1.
///
/// The product is computed with an overflow check. The vector code computes
/// addresses from `stride`, so a product that wrapped around must not pass.
#[inline(always)]
fn holds_strided(len: usize, stride: usize, n2: usize) -> bool {
    stride.checked_mul(n2 - 1).is_some_and(|last| last < len)
}

impl MdctLookup {
    pub fn new(n: usize, max_lm: usize) -> Self {
        debug_assert!(max_lm <= MDCT_MAX_LM);
        let mut kfft: FixedVec<Option<KissFftState>, { MDCT_MAX_LM + 1 }> = FixedVec::new();
        let mut trig: FixedVec<f32, MDCT_TRIG_MAX> = FixedVec::new();
        let mut curr_n = n;
        // The transform sizes of the 48 kHz mode take their rotation factors
        // from the table of libopus. The table holds the sizes 1920, 960, 480
        // and 240 one after the other.
        let from_table = n == 1920 && max_lm <= 3;

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
            if from_table {
                let start = trig.len();
                for &t in &MDCT_TWIDDLES_960[start..start + n2] {
                    trig.push(t);
                }
            } else {
                // Any other size: as libopus computes the table of a custom
                // mode (`clt_mdct_init`, celt/mdct.c:100-101), with the
                // angle and the cosine in double precision. The constant of
                // celt/mathops.h:48, 3.1415926535897931, is this number.
                const PI: f64 = core::f64::consts::PI;
                for i in 0..n2 {
                    let angle = 2.0 * PI * (i as f64 + 0.125) / curr_n as f64;
                    trig.push(angle.cos() as f32);
                }
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

    /// The FFT plan of the transform size `self.n >> shift`, with
    /// `(self.n >> shift) / 4` points (libopus `mdct.kfft[shift]`). The speech and music analysis
    /// uses the 480-point plan of the 48 kHz mode.
    #[cfg_attr(not(feature = "analysis"), allow(dead_code))]
    pub(crate) fn kfft(&self, shift: usize) -> Option<&KissFftState> {
        self.kfft.get(shift).and_then(Option::as_ref)
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
            stride >= 1 && holds_strided(output.len(), stride, n2),
            "MDCT forward: output buffer too small (need {}, have {})",
            stride.saturating_mul(n2 - 1).saturating_add(1),
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
        // All buffer sizes are checked before the transform starts. The
        // pre-rotation reads input[2*i*stride] and input[stride*(n2-1-2*i)],
        // and the windowing of the overlap reads window[..overlap].
        assert!(
            stride >= 1 && holds_strided(input.len(), stride, n2),
            "MDCT backward: input too small (need {}, have {})",
            stride.saturating_mul(n2 - 1).saturating_add(1),
            input.len()
        );
        assert!(
            window.len() >= overlap,
            "MDCT backward: window too small (need {}, have {})",
            overlap,
            window.len()
        );
        // The rotation writes output[overlap2..overlap2 + n2]. The windowing
        // of the overlap rewrites output[..overlap].
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
    ///
    /// On aarch64 it gives the bits of `forward_fused`, which are the bits
    /// of the arm64 build of libopus. On every other target it gives the
    /// bits of `forward_scalar`.
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
                forward_impl::<Separate>(&p, input, output, window, overlap, stride, opus_fft_impl);
            }
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64"
        )))]
        forward_impl::<Separate>(&p, input, output, window, overlap, stride, opus_fft_impl);
    }

    /// The transform of `forward` with the portable code and no fused
    /// product, on every target.
    ///
    /// It is the code of the targets without a vector version, and what a
    /// build of libopus without fused multiply-add computes.
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
        forward_impl::<Separate>(
            &p,
            input,
            output,
            window,
            overlap,
            stride,
            opus_fft_impl_scalar,
        );
    }

    /// The transform of `forward` with the portable code and the products
    /// fused as clang fuses them in libopus for arm64, on every target.
    ///
    /// It is the reference the aarch64 steps are tested against. On a
    /// processor without a fused multiply-add instruction it is slow.
    pub fn forward_fused(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let p = self.forward_plan(input, output, window, overlap, shift, stride);
        forward_impl::<Fused>(&p, input, output, window, overlap, stride, opus_fft_impl_fused);
    }

    /// Inverse MDCT of `n2 = (self.n >> shift) / 2` coefficients.
    ///
    /// Reads the coefficients from `input[0]`, `input[stride]`, ... and
    /// writes `output[overlap / 2..overlap / 2 + n2]`. Then it combines
    /// `output[..overlap]` with the window; the first `overlap / 2` values of
    /// that range are the tail the previous block left there.
    ///
    /// On aarch64 it gives the bits of `backward_fused`, which are the bits
    /// of the arm64 build of libopus. On every other target it gives the
    /// bits of `backward_scalar`.
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
                backward_impl::<Separate>(&p, input, output, window, overlap, stride, opus_fft_impl);
            }
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "x86_64"
        )))]
        backward_impl::<Separate>(&p, input, output, window, overlap, stride, opus_fft_impl);
    }

    /// The transform of `backward` with the portable code and no fused
    /// product, on every target.
    ///
    /// It is the code of the targets without a vector version, and what a
    /// build of libopus without fused multiply-add computes.
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
        backward_impl::<Separate>(
            &p,
            input,
            output,
            window,
            overlap,
            stride,
            opus_fft_impl_scalar,
        );
    }

    /// The transform of `backward` with the portable code and the products
    /// fused as clang fuses them in libopus for arm64, on every target.
    ///
    /// It is the reference the aarch64 steps are tested against. On a
    /// processor without a fused multiply-add instruction it is slow.
    pub fn backward_fused(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let p = self.backward_plan(input, output, window, overlap, shift, stride);
        backward_impl::<Fused>(&p, input, output, window, overlap, stride, opus_fft_impl_fused);
    }
}

// ---------------------------------------------------------------------------
// Portable transform.
//
// The steps are the loops of libopus (mdct.c, float build). `A` says whether
// a product is fused with the sum it is written next to (see `Arith`):
// - `Separate` is the code of every target without a vector version.
// - `Fused` is what the vector steps for aarch64 compute. They are tested
//   against it and must give the same bits.
// ---------------------------------------------------------------------------

/// Room for the `n4` complex values of one transform, not initialised.
///
/// A step that fills such a buffer takes it as a slice of `MaybeUninit`.
/// The slice becomes a slice of values only after every entry was written.
type Work = [MaybeUninit<KissCpx>; MAX_N4];

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

/// Rotation in front of the forward FFT, with the FFT scale factor
/// (mdct.c:211-220).
#[inline(always)]
fn forward_pre_rotate_one<A: Arith>(re: f32, im: f32, t0: f32, t1: f32, scale: f32) -> KissCpx {
    let yr = A::mul_add(re, t0, -(im * t1));
    let yi = A::mul_add(im, t0, re * t1);
    KissCpx::new(yr * scale, yi * scale)
}

/// Rotation in front of the inverse FFT (mdct.c:325-326). `x1` and `x2` are
/// the coefficients `2 * i` and `n2 - 1 - 2 * i`.
#[inline(always)]
fn backward_pre_rotate_one<A: Arith>(x1: f32, x2: f32, t0: f32, t1: f32) -> KissCpx {
    let yr = A::mul_add(x2, t0, x1 * t1);
    let yi = A::mul_add(x1, t0, -(x2 * t1));
    KissCpx::new(yi, yr)
}

/// Forward MDCT, first step: window, fold and rotate.
///
/// Writes the `n4` FFT inputs to `tmp` in natural order. `overlap` is
/// `4 * q`, `tmp.len()` is `n4 >= 2 * q`, `trig.len()` is `2 * n4`.
///
/// Butterfly `i` combines two samples from the front and two from the back
/// of the block. The first `q` and the last `q` butterflies use the window.
///
/// Every entry of `tmp` is written: the three loops below cover `q`,
/// `n4 - 2 * q` and `q` butterflies, and each of the slices they walk has
/// exactly that many items.
#[inline(always)]
fn forward_fold_rotate_scalar<A: Arith>(
    input: &[f32],
    window: &[f32],
    trig: &[f32],
    tmp: &mut [MaybeUninit<KissCpx>],
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
        let re = A::mul_add(hi[u], w[d], hi[d] * w[u]);
        let im = A::mul_add(lo[u], w[u], -(lo[d] * w[d]));
        o.write(forward_pre_rotate_one::<A>(re, im, t0, t1, scale));
    }

    // Butterflies q..n4 - q, no window: the real part walks down from sample
    // `n2 - 1`, the imaginary part walks up from sample `4 * q`.
    let mid = &input[len..n2];
    let rot = t0[q..n4 - q].iter().zip(&t1[q..n4 - q]);
    for (((o, re), im), (&t0, &t1)) in tmp_mid.iter_mut().zip(down2(mid)).zip(up2(mid)).zip(rot) {
        o.write(forward_pre_rotate_one::<A>(re, im, t0, t1, scale));
    }

    // Butterflies n4 - q..n4: `u` walks up from the start of the three
    // slices, `d` walks down from their end.
    let rot = t0[n4 - q..].iter().zip(&t1[n4 - q..]);
    for (k, (o, (&t0, &t1))) in tmp_tail.iter_mut().zip(rot).enumerate() {
        let u = 2 * k;
        let d = len - 1 - 2 * k;
        // The C source has the negated product on the left
        // (mdct.c:185), so that is the one a fused build fuses.
        let re = A::mul_add(-lo[u], w[u], lo[d] * w[d]);
        let im = A::mul_add(hi[u], w[d], hi[d] * w[u]);
        o.write(forward_pre_rotate_one::<A>(re, im, t0, t1, scale));
    }
}

/// Copies the rotated values to the places the FFT expects them in.
///
/// Entry `bitrev[i]` of `f2` gets `tmp[i]`. Every entry of `f2` is written
/// when `bitrev` is a permutation of `0..f2.len()` and `tmp` has as many
/// values.
#[inline(always)]
fn scatter_scalar(tmp: &[KissCpx], bitrev: &[i16], f2: &mut [MaybeUninit<KissCpx>]) {
    debug_assert!(tmp.len() == f2.len() && bitrev.len() == f2.len());
    for (&v, &rev) in tmp.iter().zip(bitrev) {
        f2[rev as usize].write(v);
    }
}

/// Forward MDCT, last step: rotate the FFT output and store it.
///
/// Butterfly `i` gives coefficient `2 * i` and coefficient `n2 - 1 - 2 * i`.
#[inline(always)]
fn forward_post_rotate_scalar<A: Arith>(
    f2: &[KissCpx],
    trig: &[f32],
    output: &mut [f32],
    stride: usize,
) {
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
            o[0] = A::mul_add(fa.i, a1, -(fa.r * a0));
            o[1] = A::mul_add(fb.r, b1, fb.i * b0);
        }
    } else {
        for (i, ((fp, &t0), &t1)) in f2.iter().zip(t0).zip(t1).enumerate() {
            output[i * 2 * stride] = A::mul_add(fp.i, t1, -(fp.r * t0));
            output[stride * (n2 - 1 - 2 * i)] = A::mul_add(fp.r, t1, fp.i * t0);
        }
    }
}

/// Inverse MDCT, first step: rotate the coefficients.
///
/// Writes the `n4` FFT inputs to `tmp` in natural order. Butterfly `i` reads
/// coefficient `2 * i` and coefficient `n2 - 1 - 2 * i`.
///
/// Every entry of `tmp` is written: each of the slices the loops walk has
/// `n4` items.
#[inline(always)]
fn backward_pre_rotate_scalar<A: Arith>(
    input: &[f32],
    trig: &[f32],
    tmp: &mut [MaybeUninit<KissCpx>],
    stride: usize,
) {
    let n4 = tmp.len();
    let n2 = 2 * n4;
    let (t0, t1) = trig.split_at(n4);
    let t1 = &t1[..n4];
    let rot = t0.iter().zip(t1);
    if stride == 1 {
        let input = &input[..n2];
        for (((o, x1), x2), (&t0, &t1)) in tmp.iter_mut().zip(up2(input)).zip(down2(input)).zip(rot)
        {
            o.write(backward_pre_rotate_one::<A>(x1, x2, t0, t1));
        }
    } else {
        for (i, (o, (&t0, &t1))) in tmp.iter_mut().zip(rot).enumerate() {
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            o.write(backward_pre_rotate_one::<A>(x1, x2, t0, t1));
        }
    }
}

/// Inverse MDCT, step after the FFT: rotate and store `n2` samples.
///
/// Sample `2 * k` comes from butterfly `k`, sample `2 * k + 1` from
/// butterfly `n4 - 1 - k`. `out.len()` is `2 * f2.len()`.
#[inline(always)]
fn backward_post_rotate_scalar<A: Arith>(f2: &[KissCpx], trig: &[f32], out: &mut [f32]) {
    let n4 = f2.len();
    let (t0, t1) = trig.split_at(n4);
    let t1 = &t1[..n4];
    let up = f2.iter().zip(t0).zip(t1);
    let down = down1(f2).zip(down1(t0)).zip(down1(t1));
    let pairs = out[..2 * n4].as_chunks_mut::<2>().0.iter_mut();
    for ((o, ((fa, &a0), &a1)), ((fb, b0), b1)) in pairs.zip(up).zip(down) {
        o[0] = A::mul_add(fa.i, a0, fa.r * a1);
        o[1] = A::mul_add(fb.i, b1, -(fb.r * b0));
    }
}

/// Inverse MDCT, last step: combine `out` with its mirror image through the
/// window (time-domain aliasing cancellation). `out` and `window` have the
/// length of the overlap.
#[inline(always)]
fn tdac_scalar<A: Arith>(out: &mut [f32], window: &[f32]) {
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
        *front = A::mul_add(x2, wp2, -(x1 * wp1));
        back[0] = A::mul_add(x2, wp1, x1 * wp2);
    }
}

/// Forward MDCT with the portable steps. `fft` is the FFT to run; it must
/// round like `A`.
#[inline(always)]
fn forward_impl<A: Arith>(
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
    let tmp = &mut tmp_buf[..p.n4];
    let f2 = &mut f2_buf[..p.n4];

    forward_fold_rotate_scalar::<A>(input, window, p.trig, tmp, overlap / 4, p.st.scale());
    // SAFETY: the fold step wrote every entry of `tmp`.
    let tmp = unsafe { tmp.assume_init_ref() };
    scatter_scalar(tmp, &p.st.bitrev, f2);
    // SAFETY: the scatter step wrote every entry of `f2`. The bit-reversal
    // table of a plan is a permutation of `0..n4`: `KissFftState::build`
    // checks it, and `MdctLookup` never changes its plans.
    let f2 = unsafe { f2.assume_init_mut() };
    fft(p.st, f2);
    forward_post_rotate_scalar::<A>(f2, p.trig, output, stride);
}

/// Inverse MDCT without its last step: rotation, FFT, rotation. Writes
/// `output[overlap / 2..overlap / 2 + n2]`. `fft` is the FFT to run; it
/// must round like `A`.
#[inline(always)]
fn backward_rotations<A: Arith>(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    overlap: usize,
    stride: usize,
    fft: impl Fn(&KissFftState, &mut [KissCpx]),
) {
    let mut tmp_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    let tmp = &mut tmp_buf[..p.n4];
    let f2 = &mut f2_buf[..p.n4];

    backward_pre_rotate_scalar::<A>(input, p.trig, tmp, stride);
    // SAFETY: the rotation wrote every entry of `tmp`.
    let tmp = unsafe { tmp.assume_init_ref() };
    scatter_scalar(tmp, &p.st.bitrev, f2);
    // SAFETY: the scatter step wrote every entry of `f2`. The bit-reversal
    // table of a plan is a permutation of `0..n4`: `KissFftState::build`
    // checks it, and `MdctLookup` never changes its plans.
    let f2 = unsafe { f2.assume_init_mut() };
    fft(p.st, f2);
    let overlap2 = overlap / 2;
    backward_post_rotate_scalar::<A>(f2, p.trig, &mut output[overlap2..overlap2 + p.n2]);
}

/// Inverse MDCT with the portable steps. `fft` is the FFT to run; it must
/// round like `A`.
#[inline(always)]
fn backward_impl<A: Arith>(
    p: &Plan<'_>,
    input: &[f32],
    output: &mut [f32],
    window: &[f32],
    overlap: usize,
    stride: usize,
    fft: impl Fn(&KissFftState, &mut [KissCpx]),
) {
    backward_rotations::<A>(p, input, output, overlap, stride, fft);
    tdac_scalar::<A>(&mut output[..overlap], window);
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
    forward_impl::<Separate>(p, input, output, window, overlap, stride, opus_fft_impl);
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
    backward_rotations::<Separate>(p, input, output, overlap, stride, opus_fft_impl);
    // SAFETY: `backward_plan` checked that `output` and `window` hold
    // `overlap` values.
    mdct_tdac_avx(output, window, overlap);
}

/// Inverse MDCT, last step, with AVX.
///
/// # Safety
/// The processor must support AVX. `output` and `window` must hold at least
/// `overlap` values each.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_tdac_avx(output: &mut [f32], window: &[f32], overlap: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    debug_assert!(output.len() >= overlap && window.len() >= overlap);
    let overlap2 = overlap / 2;
    let mut i = 0usize;

    // SAFETY: the vector loads and the vector store touch `i..i + 8` with
    // `i + 8 <= overlap / 2`. Everything else is indexed with a check.
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
// Each step gives the bits of its portable version with `Fused`, which are
// the bits of the arm64 build of libopus. Where the portable step has
// `mul_add(a, b, c * d)`, the product `c * d` is computed first with `fmul`
// and `a * b` is fused onto it with `fmla`. For `mul_add(a, b, -(c * d))` the
// factor `c` is negated before the `fmul`; that gives the same bits as
// negating the product, also when the product is zero.
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
    debug_assert!(overlap == 4 * q && n4 >= 2 * q && n4 <= MAX_N4);
    debug_assert!(input.len() >= 2 * n4 + overlap && window.len() >= overlap);
    debug_assert!(stride >= 1 && holds_strided(output.len(), stride, 2 * n4));
    debug_assert!(p.trig.len() >= 2 * n4 && p.st.bitrev.len() == n4);
    let mid = n4 - 2 * q;
    // The fold step has three groups of butterflies: `q` with the window,
    // `mid` without, `q` with the window. A group of one to three
    // butterflies is too short for a vector step.
    if n4 < 4 || (q != 0 && q < 4) || (mid != 0 && mid < 4) {
        // On aarch64 `opus_fft_impl` rounds like `Fused`.
        forward_impl::<Fused>(p, input, output, window, overlap, stride, opus_fft_impl);
        return;
    }

    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    let f2 = &mut f2_buf[..n4];

    // SAFETY: `forward_plan` checked that `overlap` is `4 * q`, that
    // `n4 >= 2 * q`, that `input` holds `2 * n4 + 4 * q` values, `window`
    // `4 * q` values and `output` `stride * (2 * n4 - 1) + 1` values. The plan
    // has `2 * n4` rotation values and `n4` bit-reversal entries below `n4`.
    // The group sizes were checked above.
    neon::forward_fold_rotate(input, window, p.trig, &p.st.bitrev, f2, q, p.st.scale());
    // SAFETY: the fold step wrote entry `bitrev[i]` of `f2` for every
    // butterfly `i` in `0..n4`. The bit-reversal table of a plan is a
    // permutation of `0..n4`: `KissFftState::build` checks it, and
    // `MdctLookup` never changes its plans. So every entry was written.
    let f2 = f2.assume_init_mut();
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
    debug_assert!(n4 <= MAX_N4 && p.n2 == 2 * n4);
    debug_assert!(stride >= 1 && holds_strided(input.len(), stride, 2 * n4));
    debug_assert!(output.len() >= overlap / 2 + 2 * n4 && output.len() >= overlap);
    debug_assert!(window.len() >= overlap);
    debug_assert!(p.trig.len() >= 2 * n4 && p.st.bitrev.len() == n4);
    if n4 < 4 {
        // On aarch64 `opus_fft_impl` rounds like `Fused`.
        backward_impl::<Fused>(p, input, output, window, overlap, stride, opus_fft_impl);
        return;
    }

    let mut f2_buf: Work = [MaybeUninit::uninit(); MAX_N4];
    let f2 = &mut f2_buf[..n4];

    // SAFETY: `backward_plan` checked that `input` holds
    // `stride * (2 * n4 - 1) + 1` values and that `output` holds
    // `overlap / 2 + 2 * n4` values. The plan has `2 * n4` rotation values and
    // `n4` bit-reversal entries below `n4`. `n4 >= 4` was checked above.
    neon::backward_pre_rotate(input, p.trig, &p.st.bitrev, f2, stride);
    // SAFETY: the rotation wrote entry `bitrev[i]` of `f2` for every
    // butterfly `i` in `0..n4`. The bit-reversal table of a plan is a
    // permutation of `0..n4`: `KissFftState::build` checks it, and
    // `MdctLookup` never changes its plans. So every entry was written.
    let f2 = f2.assume_init_mut();
    opus_fft_impl(p.st, f2);
    let overlap2 = overlap / 2;
    neon::backward_post_rotate(f2, p.trig, &mut output[overlap2..overlap2 + p.n2]);
    // SAFETY: `backward_plan` checked that `window` holds `overlap` values.
    neon::tdac(&mut output[..overlap], window);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Fused, KissCpx, tdac_scalar};
    use core::arch::aarch64::*;
    use core::mem::MaybeUninit;

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
    /// Entry `bitrev[i]` of `f2` is written for every `i` in `0..f2.len()`.
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
        f2: &mut [MaybeUninit<KissCpx>],
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
            // yr = re * t0 - im * t1 and yi = im * t0 + re * t1, the product
            // on the left fused.
            let yr = vfmaq_f32(vmulq_f32(vnegq_f32(im), t1), re, t0);
            let yi = vfmaq_f32(vmulq_f32(re, t1), im, t0);
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
            // re = hi[u] * w2 + hi[d] * w1 and im = lo[u] * w1 - lo[d] * w2,
            // the product on the left fused.
            let re = vfmaq_f32(
                vmulq_f32(down4(inp.add(n2 + d)), w1),
                up4(inp.add(n2 + u)),
                w2,
            );
            let im = vfmaq_f32(
                vmulq_f32(vnegq_f32(down4(inp.add(d))), w2),
                up4(inp.add(u)),
                w1,
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
            // re = -lo[u] * w1 + lo[d] * w2: the negated product is the
            // fused one, so it is subtracted with `fmls`.
            // im = hi[u] * w2 + hi[d] * w1, the product on the left fused.
            let re = vfmsq_f32(vmulq_f32(down4(inp.add(d)), w2), up4(inp.add(u)), w1);
            let im = vfmaq_f32(
                vmulq_f32(down4(inp.add(n2 + d)), w1),
                up4(inp.add(n2 + u)),
                w2,
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
            // (c.im * t1 - c.re * t0, c.re * t1 + c.im * t0), the product on
            // the left fused.
            (
                vfmaq_f32(vmulq_f32(vnegq_f32(c.0), t0), c.1, t1),
                vfmaq_f32(vmulq_f32(c.1, t0), c.0, t1),
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
    /// Entry `bitrev[i]` of `f2` is written for every `i` in `0..f2.len()`.
    ///
    /// # Safety
    /// With `n4 = f2.len()`: `n4 >= 4`, `stride >= 1`, `input` holds at least
    /// `stride * (2 * n4 - 1) + 1` values, `trig` at least `2 * n4`, and
    /// `bitrev` holds `n4` entries, each of them below `n4`.
    pub(super) unsafe fn backward_pre_rotate(
        input: &[f32],
        trig: &[f32],
        bitrev: &[i16],
        f2: &mut [MaybeUninit<KissCpx>],
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
        // `n2 - 1 - 2 * i` (`x2`). One call rotates the butterflies
        // `i..i + 4` and stores them.
        // SAFETY (for every call below): `i + 4 <= n4`, so all coefficients
        // read are below `n2`.
        let block = |i: usize| {
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
            // yr = x2 * t0 + x1 * t1 and yi = x1 * t0 - x2 * t1, the product
            // on the left fused.
            let yr = vfmaq_f32(vmulq_f32(x1, t1), x2, t0);
            let yi = vfmaq_f32(vmulq_f32(vnegq_f32(x2), t1), x1, t0);
            store_bitrev4(yi, yr, br.add(i), out);
        };
        // Two blocks per step. They do not depend on each other, so a core
        // that runs the instructions in program order works on one while
        // the other waits for a result. The last block may overlap the one
        // before; it then stores some butterflies a second time, with the
        // same values.
        let mut start = 0;
        while start + 8 <= n4 {
            block(start);
            block(start + 4);
            start += 8;
        }
        while start < n4 {
            block(start.min(n4 - 4));
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
            // (c.im * t0 + c.re * t1, c.im * t1 - c.re * t0), the product on
            // the left fused.
            (
                vfmaq_f32(vmulq_f32(c.0, t1), c.1, t0),
                vfmaq_f32(vmulq_f32(vnegq_f32(c.0), t0), c.1, t1),
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

    /// Inverse MDCT, last step: combine `out` with its mirror image through
    /// the window (time-domain aliasing cancellation).
    ///
    /// Sample `i` and sample `out.len() - 1 - i` form a pair. One step takes
    /// four pairs. This step works in place, so no pair may be computed
    /// twice: the pairs left over in the middle go to the portable code.
    ///
    /// # Safety
    /// `window` must hold at least `out.len()` values.
    pub(super) unsafe fn tdac(out: &mut [f32], window: &[f32]) {
        let overlap = out.len();
        debug_assert!(window.len() >= overlap);
        // Number of pairs the vector steps take.
        let vec = (overlap / 2) & !3;
        let o = out.as_mut_ptr();
        let w = window.as_ptr();

        // SAFETY: `i + 4 <= vec <= overlap / 2`. The front vector covers the
        // samples `i..i + 4`. The back vector covers
        // `overlap - 4 - i..overlap - i`; it starts at or behind
        // `overlap - overlap / 2`. Both lie inside `out` and inside
        // `window`, and the two do not overlap. A step reads its samples
        // before it writes them.
        let mut i = 0;
        while i < vec {
            let x2 = vld1q_f32(o.add(i));
            let x1 = rev4(vld1q_f32(o.add(overlap - 4 - i)));
            let wp1 = vld1q_f32(w.add(i));
            let wp2 = rev4(vld1q_f32(w.add(overlap - 4 - i)));
            // front = x2 * wp2 - x1 * wp1 and back = x2 * wp1 + x1 * wp2,
            // the product on the left fused.
            let front = vfmaq_f32(vmulq_f32(vnegq_f32(x1), wp1), x2, wp2);
            let back = vfmaq_f32(vmulq_f32(x1, wp2), x2, wp1);
            vst1q_f32(o.add(i), front);
            vst1q_f32(o.add(overlap - 4 - i), rev4(back));
            i += 4;
        }

        tdac_scalar::<Fused>(&mut out[vec..overlap - vec], &window[vec..overlap - vec]);
    }
}


/// The rotation factors of the MDCT sizes of the 48 kHz mode: 960 values for
/// the transform of 1920 samples, then 480, 240 and 120 values for the three
/// shorter ones. Value `i` of a transform of `n` samples is
/// `cos(2 * pi * (i + 1/8) / n)`.
///
/// It is `mdct_twiddles960` of libopus 1.6
/// (celt/static_modes_float.h:530-890), digit for digit. The numbers have
/// eight significant digits. Some of them are not the `f32` nearest to the
/// exact cosine, so the table cannot be computed here; it has to be copied.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
static MDCT_TWIDDLES_960: [f32; 1800] = [
    0.99999992, 0.99999322, 0.99997582, 0.99994771, 0.99990889,
    0.99985936, 0.99979913, 0.99972818, 0.99964653, 0.99955418,
    0.99945112, 0.99933736, 0.99921289, 0.99907773, 0.99893186,
    0.99877530, 0.99860804, 0.99843009, 0.99824144, 0.99804211,
    0.99783209, 0.99761138, 0.99737998, 0.99713791, 0.99688516,
    0.99662173, 0.99634763, 0.99606285, 0.99576741, 0.99546131,
    0.99514455, 0.99481713, 0.99447905, 0.99413033, 0.99377096,
    0.99340095, 0.99302029, 0.99262901, 0.99222709, 0.99181455,
    0.99139139, 0.99095760, 0.99051321, 0.99005821, 0.98959261,
    0.98911641, 0.98862961, 0.98813223, 0.98762427, 0.98710573,
    0.98657662, 0.98603694, 0.98548670, 0.98492591, 0.98435457,
    0.98377269, 0.98318028, 0.98257734, 0.98196387, 0.98133989,
    0.98070539, 0.98006040, 0.97940491, 0.97873893, 0.97806247,
    0.97737554, 0.97667813, 0.97597027, 0.97525196, 0.97452320,
    0.97378401, 0.97303439, 0.97227435, 0.97150389, 0.97072303,
    0.96993178, 0.96913014, 0.96831812, 0.96749573, 0.96666298,
    0.96581987, 0.96496643, 0.96410265, 0.96322854, 0.96234412,
    0.96144939, 0.96054437, 0.95962906, 0.95870347, 0.95776762,
    0.95682151, 0.95586515, 0.95489856, 0.95392174, 0.95293471,
    0.95193746, 0.95093003, 0.94991241, 0.94888462, 0.94784667,
    0.94679856, 0.94574032, 0.94467195, 0.94359346, 0.94250486,
    0.94140618, 0.94029741, 0.93917857, 0.93804967, 0.93691073,
    0.93576176, 0.93460276, 0.93343375, 0.93225475, 0.93106577,
    0.92986681, 0.92865789, 0.92743903, 0.92621024, 0.92497153,
    0.92372291, 0.92246440, 0.92119602, 0.91991776, 0.91862966,
    0.91733172, 0.91602395, 0.91470637, 0.91337900, 0.91204185,
    0.91069493, 0.90933825, 0.90797184, 0.90659570, 0.90520986,
    0.90381432, 0.90240910, 0.90099422, 0.89956969, 0.89813553,
    0.89669174, 0.89523836, 0.89377538, 0.89230284, 0.89082074,
    0.88932910, 0.88782793, 0.88631726, 0.88479710, 0.88326746,
    0.88172836, 0.88017982, 0.87862185, 0.87705448, 0.87547771,
    0.87389156, 0.87229606, 0.87069121, 0.86907704, 0.86745357,
    0.86582080, 0.86417876, 0.86252747, 0.86086694, 0.85919719,
    0.85751824, 0.85583010, 0.85413281, 0.85242636, 0.85071078,
    0.84898610, 0.84725232, 0.84550947, 0.84375756, 0.84199662,
    0.84022666, 0.83844771, 0.83665977, 0.83486287, 0.83305704,
    0.83124228, 0.82941862, 0.82758608, 0.82574467, 0.82389442,
    0.82203535, 0.82016748, 0.81829082, 0.81640540, 0.81451123,
    0.81260835, 0.81069676, 0.80877649, 0.80684755, 0.80490998,
    0.80296379, 0.80100899, 0.79904562, 0.79707370, 0.79509323,
    0.79310425, 0.79110678, 0.78910084, 0.78708644, 0.78506362,
    0.78303239, 0.78099277, 0.77894479, 0.77688847, 0.77482382,
    0.77275088, 0.77066967, 0.76858020, 0.76648250, 0.76437659,
    0.76226250, 0.76014024, 0.75800984, 0.75587132, 0.75372471,
    0.75157003, 0.74940730, 0.74723654, 0.74505779, 0.74287105,
    0.74067635, 0.73847373, 0.73626320, 0.73404478, 0.73181850,
    0.72958438, 0.72734245, 0.72509273, 0.72283525, 0.72057002,
    0.71829708, 0.71601644, 0.71372814, 0.71143220, 0.70912863,
    0.70681747, 0.70449874, 0.70217247, 0.69983868, 0.69749739,
    0.69514863, 0.69279243, 0.69042881, 0.68805780, 0.68567941,
    0.68329369, 0.68090064, 0.67850031, 0.67609270, 0.67367786,
    0.67125580, 0.66882656, 0.66639015, 0.66394661, 0.66149595,
    0.65903821, 0.65657341, 0.65410159, 0.65162275, 0.64913694,
    0.64664418, 0.64414449, 0.64163790, 0.63912444, 0.63660414,
    0.63407702, 0.63154311, 0.62900244, 0.62645503, 0.62390091,
    0.62134011, 0.61877265, 0.61619857, 0.61361789, 0.61103064,
    0.60843685, 0.60583654, 0.60322974, 0.60061648, 0.59799679,
    0.59537069, 0.59273822, 0.59009940, 0.58745427, 0.58480284,
    0.58214514, 0.57948122, 0.57681109, 0.57413478, 0.57145232,
    0.56876374, 0.56606907, 0.56336834, 0.56066158, 0.55794881,
    0.55523006, 0.55250537, 0.54977477, 0.54703827, 0.54429592,
    0.54154774, 0.53879376, 0.53603401, 0.53326852, 0.53049731,
    0.52772043, 0.52493790, 0.52214974, 0.51935599, 0.51655668,
    0.51375184, 0.51094149, 0.50812568, 0.50530442, 0.50247775,
    0.49964570, 0.49680830, 0.49396558, 0.49111757, 0.48826430,
    0.48540580, 0.48254210, 0.47967323, 0.47679923, 0.47392012,
    0.47103594, 0.46814671, 0.46525247, 0.46235324, 0.45944907,
    0.45653997, 0.45362599, 0.45070714, 0.44778347, 0.44485501,
    0.44192178, 0.43898381, 0.43604115, 0.43309382, 0.43014185,
    0.42718527, 0.42422412, 0.42125842, 0.41828822, 0.41531353,
    0.41233440, 0.40935085, 0.40636291, 0.40337063, 0.40037402,
    0.39737313, 0.39436798, 0.39135861, 0.38834505, 0.38532733,
    0.38230548, 0.37927953, 0.37624953, 0.37321550, 0.37017747,
    0.36713547, 0.36408955, 0.36103972, 0.35798603, 0.35492850,
    0.35186718, 0.34880208, 0.34573325, 0.34266072, 0.33958451,
    0.33650468, 0.33342123, 0.33033422, 0.32724367, 0.32414961,
    0.32105209, 0.31795112, 0.31484675, 0.31173901, 0.30862793,
    0.30551354, 0.30239588, 0.29927499, 0.29615089, 0.29302362,
    0.28989321, 0.28675969, 0.28362310, 0.28048348, 0.27734085,
    0.27419526, 0.27104672, 0.26789529, 0.26474098, 0.26158384,
    0.25842390, 0.25526119, 0.25209575, 0.24892761, 0.24575680,
    0.24258336, 0.23940732, 0.23622872, 0.23304759, 0.22986396,
    0.22667787, 0.22348935, 0.22029844, 0.21710517, 0.21390958,
    0.21071169, 0.20751155, 0.20430919, 0.20110463, 0.19789793,
    0.19468910, 0.19147819, 0.18826523, 0.18505026, 0.18183330,
    0.17861439, 0.17539357, 0.17217088, 0.16894634, 0.16571999,
    0.16249186, 0.15926200, 0.15603043, 0.15279719, 0.14956231,
    0.14632583, 0.14308778, 0.13984820, 0.13660713, 0.13336459,
    0.13012062, 0.12687526, 0.12362854, 0.12038049, 0.11713116,
    0.11388057, 0.11062877, 0.10737578, 0.10412163, 0.10086638,
    0.097610040, 0.094352658, 0.091094266, 0.087834897, 0.084574589,
    0.081313374, 0.078051289, 0.074788367, 0.071524645, 0.068260157,
    0.064994938, 0.061729023, 0.058462447, 0.055195244, 0.051927451,
    0.048659101, 0.045390231, 0.042120874, 0.038851066, 0.035580842,
    0.032310238, 0.029039287, 0.025768025, 0.022496487, 0.019224708,
    0.015952723, 0.012680568, 0.0094082767, 0.0061358846, 0.0028634269,
    -0.00040906153, -0.0036815456, -0.0069539902, -0.010226360, -0.013498621,
    -0.016770737, -0.020042673, -0.023314395, -0.026585867, -0.029857055,
    -0.033127923, -0.036398436, -0.039668559, -0.042938257, -0.046207495,
    -0.049476239, -0.052744453, -0.056012102, -0.059279151, -0.062545565,
    -0.065811309, -0.069076349, -0.072340649, -0.075604174, -0.078866890,
    -0.082128761, -0.085389752, -0.088649829, -0.091908956, -0.095167100,
    -0.098424224, -0.10168029, -0.10493528, -0.10818913, -0.11144183,
    -0.11469334, -0.11794361, -0.12119263, -0.12444034, -0.12768673,
    -0.13093174, -0.13417536, -0.13741753, -0.14065824, -0.14389744,
    -0.14713510, -0.15037118, -0.15360565, -0.15683848, -0.16006962,
    -0.16329906, -0.16652674, -0.16975264, -0.17297673, -0.17619896,
    -0.17941930, -0.18263772, -0.18585419, -0.18906866, -0.19228112,
    -0.19549151, -0.19869981, -0.20190598, -0.20510998, -0.20831180,
    -0.21151138, -0.21470869, -0.21790371, -0.22109639, -0.22428671,
    -0.22747462, -0.23066010, -0.23384310, -0.23702361, -0.24020157,
    -0.24337696, -0.24654975, -0.24971989, -0.25288736, -0.25605213,
    -0.25921415, -0.26237339, -0.26552983, -0.26868342, -0.27183413,
    -0.27498193, -0.27812679, -0.28126867, -0.28440754, -0.28754336,
    -0.29067610, -0.29380573, -0.29693221, -0.30005551, -0.30317560,
    -0.30629245, -0.30940601, -0.31251626, -0.31562316, -0.31872668,
    -0.32182679, -0.32492345, -0.32801664, -0.33110631, -0.33419243,
    -0.33727497, -0.34035391, -0.34342920, -0.34650081, -0.34956871,
    -0.35263286, -0.35569324, -0.35874981, -0.36180254, -0.36485139,
    -0.36789634, -0.37093735, -0.37397438, -0.37700741, -0.38003640,
    -0.38306132, -0.38608214, -0.38909883, -0.39211135, -0.39511967,
    -0.39812375, -0.40112358, -0.40411910, -0.40711031, -0.41009715,
    -0.41307959, -0.41605762, -0.41903119, -0.42200027, -0.42496483,
    -0.42792484, -0.43088027, -0.43383109, -0.43677726, -0.43971875,
    -0.44265553, -0.44558757, -0.44851484, -0.45143731, -0.45435494,
    -0.45726770, -0.46017557, -0.46307851, -0.46597650, -0.46886949,
    -0.47175746, -0.47464038, -0.47751821, -0.48039093, -0.48325851,
    -0.48612091, -0.48897811, -0.49183006, -0.49467676, -0.49751815,
    -0.50035422, -0.50318492, -0.50601024, -0.50883014, -0.51164459,
    -0.51445356, -0.51725703, -0.52005495, -0.52284730, -0.52563406,
    -0.52841518, -0.53119065, -0.53396043, -0.53672449, -0.53948280,
    -0.54223533, -0.54498206, -0.54772295, -0.55045797, -0.55318710,
    -0.55591031, -0.55862756, -0.56133883, -0.56404409, -0.56674331,
    -0.56943646, -0.57212351, -0.57480443, -0.57747920, -0.58014778,
    -0.58281015, -0.58546628, -0.58811614, -0.59075970, -0.59339694,
    -0.59602782, -0.59865231, -0.60127040, -0.60388204, -0.60648722,
    -0.60908591, -0.61167807, -0.61426368, -0.61684271, -0.61941514,
    -0.62198093, -0.62454007, -0.62709251, -0.62963824, -0.63217722,
    -0.63470944, -0.63723486, -0.63975345, -0.64226519, -0.64477006,
    -0.64726802, -0.64975905, -0.65224312, -0.65472020, -0.65719027,
    -0.65965331, -0.66210928, -0.66455816, -0.66699992, -0.66943454,
    -0.67186199, -0.67428225, -0.67669528, -0.67910107, -0.68149959,
    -0.68389080, -0.68627470, -0.68865124, -0.69102041, -0.69338218,
    -0.69573652, -0.69808341, -0.70042283, -0.70275474, -0.70507913,
    -0.70739597, -0.70970524, -0.71200690, -0.71430093, -0.71658732,
    -0.71886604, -0.72113705, -0.72340034, -0.72565589, -0.72790366,
    -0.73014364, -0.73237580, -0.73460012, -0.73681657, -0.73902513,
    -0.74122577, -0.74341848, -0.74560322, -0.74777998, -0.74994874,
    -0.75210946, -0.75426212, -0.75640671, -0.75854320, -0.76067157,
    -0.76279178, -0.76490383, -0.76700769, -0.76910334, -0.77119075,
    -0.77326989, -0.77534076, -0.77740333, -0.77945757, -0.78150346,
    -0.78354098, -0.78557011, -0.78759083, -0.78960312, -0.79160694,
    -0.79360229, -0.79558915, -0.79756748, -0.79953727, -0.80149850,
    -0.80345114, -0.80539518, -0.80733060, -0.80925737, -0.81117547,
    -0.81308489, -0.81498559, -0.81687757, -0.81876081, -0.82063527,
    -0.82250095, -0.82435781, -0.82620585, -0.82804505, -0.82987537,
    -0.83169681, -0.83350933, -0.83531294, -0.83710760, -0.83889329,
    -0.84067000, -0.84243770, -0.84419639, -0.84594603, -0.84768662,
    -0.84941812, -0.85114053, -0.85285383, -0.85455799, -0.85625300,
    -0.85793884, -0.85961549, -0.86128294, -0.86294116, -0.86459014,
    -0.86622986, -0.86786031, -0.86948146, -0.87109330, -0.87269581,
    -0.87428898, -0.87587278, -0.87744720, -0.87901223, -0.88056784,
    -0.88211402, -0.88365076, -0.88517803, -0.88669582, -0.88820412,
    -0.88970290, -0.89119216, -0.89267187, -0.89414203, -0.89560260,
    -0.89705359, -0.89849497, -0.89992673, -0.90134885, -0.90276131,
    -0.90416411, -0.90555723, -0.90694065, -0.90831436, -0.90967833,
    -0.91103257, -0.91237705, -0.91371176, -0.91503669, -0.91635181,
    -0.91765712, -0.91895261, -0.92023825, -0.92151404, -0.92277996,
    -0.92403600, -0.92528214, -0.92651837, -0.92774468, -0.92896106,
    -0.93016748, -0.93136395, -0.93255044, -0.93372694, -0.93489345,
    -0.93604994, -0.93719641, -0.93833284, -0.93945922, -0.94057555,
    -0.94168180, -0.94277796, -0.94386403, -0.94493999, -0.94600583,
    -0.94706154, -0.94810711, -0.94914252, -0.95016777, -0.95118284,
    -0.95218773, -0.95318242, -0.95416690, -0.95514117, -0.95610520,
    -0.95705900, -0.95800255, -0.95893583, -0.95985885, -0.96077159,
    -0.96167404, -0.96256619, -0.96344803, -0.96431956, -0.96518076,
    -0.96603162, -0.96687213, -0.96770230, -0.96852209, -0.96933152,
    -0.97013057, -0.97091922, -0.97169748, -0.97246533, -0.97322277,
    -0.97396979, -0.97470637, -0.97543252, -0.97614822, -0.97685347,
    -0.97754825, -0.97823257, -0.97890641, -0.97956977, -0.98022263,
    -0.98086500, -0.98149687, -0.98211822, -0.98272906, -0.98332937,
    -0.98391915, -0.98449840, -0.98506710, -0.98562525, -0.98617285,
    -0.98670988, -0.98723635, -0.98775225, -0.98825757, -0.98875230,
    -0.98923645, -0.98971000, -0.99017295, -0.99062530, -0.99106704,
    -0.99149817, -0.99191868, -0.99232857, -0.99272783, -0.99311645,
    -0.99349445, -0.99386180, -0.99421851, -0.99456457, -0.99489998,
    -0.99522474, -0.99553884, -0.99584227, -0.99613505, -0.99641715,
    -0.99668858, -0.99694934, -0.99719943, -0.99743883, -0.99766756,
    -0.99788559, -0.99809295, -0.99828961, -0.99847558, -0.99865086,
    -0.99881544, -0.99896933, -0.99911252, -0.99924501, -0.99936680,
    -0.99947789, -0.99957827, -0.99966795, -0.99974692, -0.99981519,
    -0.99987275, -0.99991960, -0.99995574, -0.99998118, -0.99999590,
    0.99999967, 0.99997289, 0.99990328, 0.99979084, 0.99963557,
    0.99943748, 0.99919658, 0.99891288, 0.99858638, 0.99821711,
    0.99780508, 0.99735031, 0.99685281, 0.99631261, 0.99572973,
    0.99510420, 0.99443605, 0.99372529, 0.99297196, 0.99217610,
    0.99133774, 0.99045692, 0.98953366, 0.98856802, 0.98756003,
    0.98650973, 0.98541718, 0.98428242, 0.98310549, 0.98188645,
    0.98062534, 0.97932224, 0.97797718, 0.97659022, 0.97516144,
    0.97369088, 0.97217861, 0.97062469, 0.96902920, 0.96739220,
    0.96571376, 0.96399395, 0.96223284, 0.96043052, 0.95858705,
    0.95670253, 0.95477702, 0.95281061, 0.95080338, 0.94875543,
    0.94666684, 0.94453769, 0.94236808, 0.94015810, 0.93790786,
    0.93561743, 0.93328693, 0.93091644, 0.92850608, 0.92605595,
    0.92356614, 0.92103677, 0.91846795, 0.91585979, 0.91321239,
    0.91052587, 0.90780035, 0.90503595, 0.90223277, 0.89939095,
    0.89651059, 0.89359184, 0.89063481, 0.88763962, 0.88460641,
    0.88153531, 0.87842644, 0.87527995, 0.87209596, 0.86887462,
    0.86561605, 0.86232041, 0.85898782, 0.85561844, 0.85221241,
    0.84876987, 0.84529098, 0.84177587, 0.83822471, 0.83463763,
    0.83101481, 0.82735639, 0.82366252, 0.81993338, 0.81616911,
    0.81236987, 0.80853584, 0.80466718, 0.80076404, 0.79682660,
    0.79285503, 0.78884950, 0.78481017, 0.78073723, 0.77663084,
    0.77249118, 0.76831844, 0.76411277, 0.75987438, 0.75560344,
    0.75130013, 0.74696464, 0.74259715, 0.73819784, 0.73376692,
    0.72930457, 0.72481097, 0.72028632, 0.71573083, 0.71114467,
    0.70652804, 0.70188116, 0.69720420, 0.69249738, 0.68776090,
    0.68299495, 0.67819975, 0.67337550, 0.66852240, 0.66364066,
    0.65873050, 0.65379211, 0.64882573, 0.64383154, 0.63880978,
    0.63376065, 0.62868438, 0.62358117, 0.61845126, 0.61329485,
    0.60811216, 0.60290343, 0.59766888, 0.59240872, 0.58712318,
    0.58181249, 0.57647688, 0.57111658, 0.56573181, 0.56032281,
    0.55488980, 0.54943303, 0.54395272, 0.53844911, 0.53292243,
    0.52737292, 0.52180083, 0.51620638, 0.51058981, 0.50495138,
    0.49929132, 0.49360987, 0.48790727, 0.48218377, 0.47643962,
    0.47067506, 0.46489034, 0.45908570, 0.45326139, 0.44741768,
    0.44155479, 0.43567299, 0.42977253, 0.42385365, 0.41791662,
    0.41196169, 0.40598911, 0.39999914, 0.39399204, 0.38796806,
    0.38192746, 0.37587050, 0.36979743, 0.36370853, 0.35760405,
    0.35148424, 0.34534939, 0.33919973, 0.33303555, 0.32685710,
    0.32066465, 0.31445847, 0.30823881, 0.30200595, 0.29576015,
    0.28950169, 0.28323082, 0.27694782, 0.27065295, 0.26434649,
    0.25802871, 0.25169988, 0.24536026, 0.23901013, 0.23264977,
    0.22627944, 0.21989941, 0.21350997, 0.20711138, 0.20070391,
    0.19428785, 0.18786347, 0.18143104, 0.17499084, 0.16854314,
    0.16208822, 0.15562636, 0.14915783, 0.14268292, 0.13620189,
    0.12971502, 0.12322260, 0.11672491, 0.11022221, 0.10371479,
    0.097202924, 0.090686897, 0.084166986, 0.077643468, 0.071116625,
    0.064586736, 0.058054080, 0.051518937, 0.044981587, 0.038442310,
    0.031901387, 0.025359097, 0.018815721, 0.012271538, 0.0057268303,
    -0.00081812300, -0.0073630412, -0.013907644, -0.020451651, -0.026994782,
    -0.033536757, -0.040077295, -0.046616116, -0.053152941, -0.059687488,
    -0.066219479, -0.072748633, -0.079274670, -0.085797312, -0.092316279,
    -0.098831291, -0.10534207, -0.11184834, -0.11834981, -0.12484622,
    -0.13133727, -0.13782270, -0.14430223, -0.15077558, -0.15724246,
    -0.16370261, -0.17015575, -0.17660160, -0.18303989, -0.18947033,
    -0.19589266, -0.20230660, -0.20871187, -0.21510820, -0.22149531,
    -0.22787294, -0.23424080, -0.24059864, -0.24694616, -0.25328311,
    -0.25960920, -0.26592418, -0.27222777, -0.27851969, -0.28479968,
    -0.29106748, -0.29732280, -0.30356539, -0.30979497, -0.31601129,
    -0.32221406, -0.32840304, -0.33457794, -0.34073852, -0.34688450,
    -0.35301562, -0.35913161, -0.36523223, -0.37131719, -0.37738626,
    -0.38343915, -0.38947562, -0.39549541, -0.40149825, -0.40748390,
    -0.41345209, -0.41940257, -0.42533508, -0.43124938, -0.43714520,
    -0.44302229, -0.44888041, -0.45471930, -0.46053871, -0.46633839,
    -0.47211810, -0.47787758, -0.48361659, -0.48933489, -0.49503222,
    -0.50070835, -0.50636303, -0.51199602, -0.51760707, -0.52319595,
    -0.52876243, -0.53430625, -0.53982718, -0.54532499, -0.55079944,
    -0.55625029, -0.56167732, -0.56708028, -0.57245896, -0.57781311,
    -0.58314251, -0.58844693, -0.59372614, -0.59897992, -0.60420805,
    -0.60941029, -0.61458642, -0.61973623, -0.62485949, -0.62995598,
    -0.63502549, -0.64006780, -0.64508268, -0.65006994, -0.65502934,
    -0.65996069, -0.66486377, -0.66973837, -0.67458427, -0.67940128,
    -0.68418919, -0.68894779, -0.69367688, -0.69837625, -0.70304571,
    -0.70768504, -0.71229407, -0.71687258, -0.72142039, -0.72593729,
    -0.73042309, -0.73487761, -0.73930064, -0.74369201, -0.74805152,
    -0.75237898, -0.75667422, -0.76093704, -0.76516727, -0.76936471,
    -0.77352921, -0.77766056, -0.78175861, -0.78582316, -0.78985406,
    -0.79385112, -0.79781417, -0.80174305, -0.80563758, -0.80949760,
    -0.81332295, -0.81711346, -0.82086896, -0.82458930, -0.82827432,
    -0.83192386, -0.83553776, -0.83911587, -0.84265803, -0.84616410,
    -0.84963392, -0.85306735, -0.85646423, -0.85982442, -0.86314779,
    -0.86643418, -0.86968345, -0.87289547, -0.87607009, -0.87920719,
    -0.88230663, -0.88536827, -0.88839199, -0.89137765, -0.89432512,
    -0.89723429, -0.90010502, -0.90293719, -0.90573069, -0.90848539,
    -0.91120117, -0.91387791, -0.91651551, -0.91911385, -0.92167282,
    -0.92419231, -0.92667220, -0.92911240, -0.93151280, -0.93387330,
    -0.93619380, -0.93847419, -0.94071438, -0.94291427, -0.94507377,
    -0.94719279, -0.94927123, -0.95130901, -0.95330604, -0.95526223,
    -0.95717750, -0.95905177, -0.96088496, -0.96267699, -0.96442777,
    -0.96613725, -0.96780534, -0.96943197, -0.97101707, -0.97256058,
    -0.97406243, -0.97552255, -0.97694089, -0.97831737, -0.97965195,
    -0.98094456, -0.98219515, -0.98340367, -0.98457006, -0.98569428,
    -0.98677627, -0.98781599, -0.98881340, -0.98976845, -0.99068110,
    -0.99155132, -0.99237906, -0.99316428, -0.99390697, -0.99460708,
    -0.99526458, -0.99587945, -0.99645166, -0.99698119, -0.99746801,
    -0.99791210, -0.99831344, -0.99867202, -0.99898782, -0.99926082,
    -0.99949102, -0.99967841, -0.99982297, -0.99992470, -0.99998360,
    0.99999866, 0.99989157, 0.99961315, 0.99916346, 0.99854256,
    0.99775057, 0.99678762, 0.99565388, 0.99434953, 0.99287481,
    0.99122996, 0.98941527, 0.98743105, 0.98527764, 0.98295541,
    0.98046475, 0.97780610, 0.97497990, 0.97198664, 0.96882685,
    0.96550104, 0.96200980, 0.95835373, 0.95453345, 0.95054962,
    0.94640291, 0.94209404, 0.93762375, 0.93299280, 0.92820199,
    0.92325213, 0.91814408, 0.91287871, 0.90745693, 0.90187965,
    0.89614785, 0.89026249, 0.88422459, 0.87803519, 0.87169533,
    0.86520612, 0.85856866, 0.85178409, 0.84485357, 0.83777828,
    0.83055945, 0.82319831, 0.81569611, 0.80805415, 0.80027373,
    0.79235620, 0.78430289, 0.77611520, 0.76779452, 0.75934229,
    0.75075995, 0.74204897, 0.73321084, 0.72424708, 0.71515923,
    0.70594883, 0.69661748, 0.68716676, 0.67759830, 0.66791374,
    0.65811474, 0.64820297, 0.63818013, 0.62804795, 0.61780815,
    0.60746249, 0.59701275, 0.58646072, 0.57580819, 0.56505701,
    0.55420900, 0.54326604, 0.53222998, 0.52110274, 0.50988620,
    0.49858230, 0.48719297, 0.47572016, 0.46416584, 0.45253199,
    0.44082059, 0.42903367, 0.41717323, 0.40524131, 0.39323996,
    0.38117123, 0.36903718, 0.35683990, 0.34458148, 0.33226402,
    0.31988962, 0.30746042, 0.29497853, 0.28244610, 0.26986527,
    0.25723821, 0.24456706, 0.23185402, 0.21910124, 0.20631092,
    0.19348526, 0.18062644, 0.16773667, 0.15481816, 0.14187312,
    0.12890377, 0.11591234, 0.10290104, 0.089872115, 0.076827789,
    0.063770300, 0.050701883, 0.037624779, 0.024541229, 0.011453473,
    -0.0016362454, -0.014725683, -0.027812598, -0.040894747, -0.053969889,
    -0.067035784, -0.080090192, -0.093130877, -0.10615561, -0.11916214,
    -0.13214826, -0.14511174, -0.15805036, -0.17096189, -0.18384413,
    -0.19669487, -0.20951190, -0.22229304, -0.23503609, -0.24773886,
    -0.26039919, -0.27301490, -0.28558383, -0.29810383, -0.31057274,
    -0.32298845, -0.33534881, -0.34765171, -0.35989504, -0.37207670,
    -0.38419461, -0.39624670, -0.40823088, -0.42014512, -0.43198737,
    -0.44375560, -0.45544780, -0.46706195, -0.47859608, -0.49004821,
    -0.50141636, -0.51269860, -0.52389299, -0.53499762, -0.54601058,
    -0.55692998, -0.56775395, -0.57848064, -0.58910822, -0.59963485,
    -0.61005873, -0.62037809, -0.63059115, -0.64069616, -0.65069139,
    -0.66057513, -0.67034568, -0.68000137, -0.68954054, -0.69896157,
    -0.70826283, -0.71744274, -0.72649972, -0.73543221, -0.74423869,
    -0.75291765, -0.76146760, -0.76988708, -0.77817464, -0.78632887,
    -0.79434836, -0.80223175, -0.80997767, -0.81758481, -0.82505187,
    -0.83237755, -0.83956061, -0.84659981, -0.85349396, -0.86024186,
    -0.86684237, -0.87329434, -0.87959669, -0.88574831, -0.89174817,
    -0.89759523, -0.90328850, -0.90882699, -0.91420976, -0.91943588,
    -0.92450446, -0.92941463, -0.93416555, -0.93875641, -0.94318642,
    -0.94745482, -0.95156087, -0.95550388, -0.95928317, -0.96289809,
    -0.96634802, -0.96963238, -0.97275059, -0.97570213, -0.97848649,
    -0.98110318, -0.98355177, -0.98583184, -0.98794298, -0.98988485,
    -0.99165711, -0.99325945, -0.99469160, -0.99595331, -0.99704438,
    -0.99796460, -0.99871383, -0.99929194, -0.99969882, -0.99993441,
    0.99999465, 0.99956631, 0.99845292, 0.99665524, 0.99417450,
    0.99101241, 0.98717112, 0.98265328, 0.97746197, 0.97160077,
    0.96507367, 0.95788516, 0.95004017, 0.94154407, 0.93240267,
    0.92262226, 0.91220953, 0.90117161, 0.88951608, 0.87725091,
    0.86438452, 0.85092573, 0.83688375, 0.82226822, 0.80708914,
    0.79135693, 0.77508236, 0.75827658, 0.74095113, 0.72311786,
    0.70478900, 0.68597711, 0.66669509, 0.64695615, 0.62677382,
    0.60616193, 0.58513461, 0.56370626, 0.54189158, 0.51970551,
    0.49716327, 0.47428029, 0.45107226, 0.42755509, 0.40374490,
    0.37965800, 0.35531090, 0.33072029, 0.30590302, 0.28087610,
    0.25565668, 0.23026205, 0.20470960, 0.17901686, 0.15320143,
    0.12728100, 0.10127334, 0.075196277, 0.049067674, 0.022905443,
    -0.0032724865, -0.029448173, -0.055603678, -0.081721074, -0.10778246,
    -0.13376998, -0.15966582, -0.18545224, -0.21111155, -0.23662618,
    -0.26197864, -0.28715155, -0.31212766, -0.33688985, -0.36142116,
    -0.38570477, -0.40972403, -0.43346249, -0.45690388, -0.48003212,
    -0.50283138, -0.52528602, -0.54738066, -0.56910015, -0.59042960,
    -0.61135441, -0.63186022, -0.65193299, -0.67155895, -0.69072467,
    -0.70941699, -0.72762312, -0.74533057, -0.76252720, -0.77920124,
    -0.79534126, -0.81093618, -0.82597533, -0.84044840, -0.85434547,
    -0.86765701, -0.88037390, -0.89248743, -0.90398929, -0.91487161,
    -0.92512691, -0.93474818, -0.94372882, -0.95206268, -0.95974404,
    -0.96676764, -0.97312866, -0.97882275, -0.98384601, -0.98819498,
    -0.99186670, -0.99485864, -0.99716875, -0.99879546, -0.99973764,
];

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
}

#[cfg(all(test, feature = "std"))]
// The tests keep tables of kernel functions with long signatures.
#[allow(clippy::type_complexity)]
mod bit_exact_tests {
    use super::*;
    use crate::kiss_fft::test_guard::{PLACEMENTS, Region};

    /// Time of the inverse transforms of the codec and of their FFT alone.
    /// Not a check: run it by hand, pinned to one core, with
    /// `--ignored --nocapture transform_timing`.
    #[cfg(feature = "std")]
    #[test]
    #[ignore]
    fn transform_timing() {
        let l = MdctLookup::new(1920, 3);
        let mut rng = Lcg(0x5eed_7171);
        let window = test_window(&mut rng, 120);
        let reps = 3000;
        let time = |name: &str, f: &mut dyn FnMut()| {
            for _ in 0..200 {
                f();
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let start = std::time::Instant::now();
                for _ in 0..reps {
                    f();
                }
                best = best.min(start.elapsed().as_nanos() as f64 / reps as f64);
            }
            std::println!("{name:34} {best:9.1} ns");
        };
        for (shift, stride) in [(0usize, 1usize), (1, 1), (3, 1), (3, 8)] {
            let n2 = (1920 >> shift) / 2;
            let mut coeffs = vec![0f32; stride * (n2 - 1) + 1];
            fill(&mut rng, &mut coeffs, 0);
            let mut out = vec![0f32; 60 + n2];
            fill(&mut rng, &mut out, 0);
            time(&format!("backward n2 {n2} stride {stride}"), &mut || {
                l.backward(&coeffs, &mut out, &window, 120, shift, stride);
                std::hint::black_box(&mut out);
            });
        }
        // The steps around the FFT, alone.
        #[cfg(target_arch = "aarch64")]
        for shift in [0usize, 3] {
            let p = l.plan(shift);
            let n4 = p.n4;
            let mut coeffs = vec![0f32; 2 * n4];
            fill(&mut rng, &mut coeffs, 0);
            let mut f2 = unwritten(n4);
            time(&format!("pre-rotation {n4}"), &mut || {
                // SAFETY: `n4` is 480 or 60, `coeffs` holds `2 * n4`
                // values, and the plan has `2 * n4` rotation values and a
                // bit-reversal table of `n4` entries below `n4`.
                unsafe { neon::backward_pre_rotate(&coeffs, p.trig, &p.st.bitrev, &mut f2, 1) };
                std::hint::black_box(&mut f2);
            });
            let spectrum: Vec<KissCpx> =
                (0..n4).map(|_| KissCpx::new(rng.unit(), rng.unit())).collect();
            let mut out = vec![0f32; 2 * n4];
            time(&format!("post-rotation {n4}"), &mut || {
                // SAFETY: as above; `spectrum` holds `n4` values and `out`
                // `2 * n4`.
                unsafe { neon::backward_post_rotate(&spectrum, p.trig, &mut out) };
                std::hint::black_box(&mut out);
            });
            let mut out = vec![0f32; 120];
            fill(&mut rng, &mut out, 0);
            time("tdac 120", &mut || {
                // SAFETY: `window` holds 120 values.
                unsafe { neon::tdac(&mut out, &window) };
                std::hint::black_box(&mut out);
            });
        }
        for shift in [0usize, 3] {
            let p = l.plan(shift);
            let mut x = vec![KissCpx::new(0.0, 0.0); p.n4];
            for v in x.iter_mut() {
                *v = KissCpx::new(rng.unit(), rng.unit());
            }
            time(&format!("fft {}", p.n4), &mut || {
                opus_fft_impl(p.st, &mut x);
                std::hint::black_box(&mut x);
            });
        }
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

    /// Fills `out` with test data.
    ///
    /// Class 0: audio-sized values. Class 1: values whose products overflow
    /// to infinity and then give NaN. Class 2: values whose products are
    /// subnormal. Class 3: signed zeros, infinities, the largest and the
    /// smallest numbers, mixed with ordinary values. Class 4: zeros of both
    /// signs, 1 and -1; many products and sums are then exactly zero, and
    /// the sign of the zero is all that can go wrong.
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
                4 => [0.0, -0.0, 1.0, -1.0][rng.next_u32() as usize % 4],
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

    /// The reference forward MDCT: the loops of libopus
    /// (`clt_mdct_forward_c`, float build) with plain indexing. `fft` must
    /// round like `A`.
    fn reference_forward<A: Arith>(
        fft: fn(&KissFftState, &mut [KissCpx]),
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
            f[yp] = A::mul_add(input[xp1 + n2], w2, input[xp2] * w1);
            yp += 1;
            f[yp] = A::mul_add(input[xp1], w1, -(input[xp2 - n2] * w2));
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
            f[yp] = A::mul_add(-input[xp1 - n2], w1, input[xp2] * w2);
            yp += 1;
            f[yp] = A::mul_add(input[xp1], w2, input[xp2 + n2] * w1);
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
            let yr = A::mul_add(re, t0, -(im * t1));
            let yi = A::mul_add(im, t0, re * t1);
            f2[st.bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
        }

        fft(st, &mut f2);

        for i in 0..n4 {
            let fp = &f2[i];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = A::mul_add(fp.i, t1, -(fp.r * t0));
            let yi = A::mul_add(fp.r, t1, fp.i * t0);
            output[i * 2 * stride] = yr;
            output[stride * (n2 - 1 - 2 * i)] = yi;
        }
    }

    /// The reference inverse MDCT: the loops of libopus
    /// (`clt_mdct_backward_c`, float build) with plain indexing. `fft` must
    /// round like `A`.
    fn reference_backward<A: Arith>(
        fft: fn(&KissFftState, &mut [KissCpx]),
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
            let yr = A::mul_add(x2, t0, x1 * t1);
            let yi = A::mul_add(x1, t0, -(x2 * t1));
            f2[rev] = KissCpx::new(yi, yr);
        }

        fft(st, &mut f2);

        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = A::mul_add(re0, t0_0, im0 * t1_0);
            let yi0 = A::mul_add(re0, t1_0, -(im0 * t0_0));

            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = A::mul_add(re1, t0_1, im1 * t1_1);
            let yi1 = A::mul_add(re1, t1_1, -(im1 * t0_1));

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
            output[i] = A::mul_add(x2, wp2, -(x1 * wp1));
            output[overlap - 1 - i] = A::mul_add(x2, wp1, x1 * wp2);
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

    /// The transforms of one direction that one case compares.
    struct Transforms {
        fast: fn(&MdctLookup, &[f32], &mut [f32], &[f32], usize, usize, usize),
        scalar: fn(&MdctLookup, &[f32], &mut [f32], &[f32], usize, usize, usize),
        fused: fn(&MdctLookup, &[f32], &mut [f32], &[f32], usize, usize, usize),
        /// The copied libopus loops with `Separate` and with `Fused`.
        loops_separate: fn(&MdctLookup, &[f32], &mut [f32], &[f32], usize, usize, usize),
        loops_fused: fn(&MdctLookup, &[f32], &mut [f32], &[f32], usize, usize, usize),
    }

    /// Runs one direction over all cases and input classes.
    ///
    /// - `scalar` and `fused` give the bits of the copied libopus loops with
    ///   the same rounding.
    /// - `fast` gives the bits of `fused` on aarch64 and of `scalar`
    ///   elsewhere.
    /// - `fast` also runs with its three slices between blocks of memory
    ///   that must not be touched: it reads and writes nothing outside them.
    ///
    /// `sizes` gives the lengths of the input and of the output slice for a
    /// case; both are exactly what the transform needs.
    fn check_direction(seed: u64, t: &Transforms, sizes: fn(&Case, usize) -> (usize, usize)) {
        let mut rng = Lcg(seed);
        let mut regions = [Region::new(), Region::new(), Region::new()];
        for c in cases() {
            let l = MdctLookup::new(c.n, c.max_lm);
            let n2 = (c.n >> c.shift) / 2;
            let (in_len, out_len) = sizes(&c, n2);
            let window = test_window(&mut rng, c.overlap);
            for class in 0..CLASSES {
                let mut input = vec![0f32; in_len];
                fill(&mut rng, &mut input, class);
                // What the output slice holds before. The inverse transform
                // reads its first values; the forward transform must leave
                // the values between the coefficients as they are.
                let mut before = vec![0f32; out_len];
                fill(&mut rng, &mut before, class);

                let run = |f: fn(&MdctLookup, &[f32], &mut [f32], &[f32], usize, usize, usize)| {
                    let mut out = before.clone();
                    f(&l, &input, &mut out, &window, c.overlap, c.shift, c.stride);
                    out
                };
                let fast = run(t.fast);
                let scalar = run(t.scalar);
                let fused = run(t.fused);

                let what = format!(
                    "n {} shift {} stride {} overlap {} class {}",
                    c.n, c.shift, c.stride, c.overlap, class
                );
                assert_eq!(first_difference(&scalar, &run(t.loops_separate)), None, "scalar, {what}");
                assert_eq!(first_difference(&fused, &run(t.loops_fused)), None, "fused, {what}");
                let native = if cfg!(target_arch = "aarch64") { &fused } else { &scalar };
                assert_eq!(first_difference(&fast, native), None, "fast, {what}");

                for at_end in PLACEMENTS {
                    let [a, b, d] = &mut regions;
                    let input = a.place(&input, at_end);
                    let window = b.place(&window, at_end);
                    let out = d.place(&before, at_end);
                    (t.fast)(&l, input, out, window, c.overlap, c.shift, c.stride);
                    assert_eq!(first_difference(out, native), None, "guarded, {what}");
                }
            }
        }
    }

    #[test]
    fn forward_matches_portable_and_reference() {
        check_direction(
            0x5eed_1001,
            &Transforms {
                fast: MdctLookup::forward,
                scalar: MdctLookup::forward_scalar,
                fused: MdctLookup::forward_fused,
                loops_separate: |l, i, o, w, ov, sh, st| {
                    reference_forward::<Separate>(opus_fft_impl_scalar, l, i, o, w, ov, sh, st)
                },
                loops_fused: |l, i, o, w, ov, sh, st| {
                    reference_forward::<Fused>(opus_fft_impl_fused, l, i, o, w, ov, sh, st)
                },
            },
            |c, n2| (n2 + c.overlap, c.stride * (n2 - 1) + 1),
        );
    }

    #[test]
    fn backward_matches_portable_and_reference() {
        check_direction(
            0x5eed_1002,
            &Transforms {
                fast: MdctLookup::backward,
                scalar: MdctLookup::backward_scalar,
                fused: MdctLookup::backward_fused,
                loops_separate: |l, i, o, w, ov, sh, st| {
                    reference_backward::<Separate>(opus_fft_impl_scalar, l, i, o, w, ov, sh, st)
                },
                loops_fused: |l, i, o, w, ov, sh, st| {
                    reference_backward::<Fused>(opus_fft_impl_fused, l, i, o, w, ov, sh, st)
                },
            },
            |c, n2| (c.stride * (n2 - 1) + 1, c.overlap / 2 + n2),
        );
    }

    /// Fusing the products changes the last bits only: for audio-sized input
    /// the two portable transforms agree to 1e-5 of the largest value.
    #[test]
    fn fused_transform_is_close_to_the_separate_one() {
        let mut rng = Lcg(0x5eed_1004);
        let l = MdctLookup::new(1920, 3);
        let window = test_window(&mut rng, 120);
        let close = |a: &[f32], b: &[f32]| {
            let peak = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(diff <= 1e-5 * peak, "{diff} of {peak}");
        };
        for shift in 0..=3usize {
            let n2 = (1920 >> shift) / 2;
            let mut input = vec![0f32; n2 + 120];
            fill(&mut rng, &mut input, 0);
            let mut a = vec![0f32; n2];
            let mut b = vec![0f32; n2];
            l.forward_fused(&input, &mut a, &window, 120, shift, 1);
            l.forward_scalar(&input, &mut b, &window, 120, shift, 1);
            close(&a, &b);

            let mut coeffs = vec![0f32; n2];
            fill(&mut rng, &mut coeffs, 0);
            let mut a = vec![0f32; 60 + n2];
            fill(&mut rng, &mut a, 0);
            let mut b = a.clone();
            l.backward_fused(&coeffs, &mut a, &window, 120, shift, 1);
            l.backward_scalar(&coeffs, &mut b, &window, 120, shift, 1);
            close(&a, &b);
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

    /// A stride whose product with 127 wraps around to 1.
    fn wrapping_stride() -> usize {
        // Newton's iteration for the inverse of an odd number modulo a power
        // of two. Each round doubles the number of correct bits.
        let mut inv = 1usize;
        for _ in 0..7 {
            inv = inv.wrapping_mul(2usize.wrapping_sub(127usize.wrapping_mul(inv)));
        }
        assert_eq!(inv.wrapping_mul(127), 1);
        assert!(inv.checked_mul(127).is_none());
        inv
    }

    /// A stride whose product with the block length does not fit a `usize`
    /// is rejected. Without the overflow check the wrapped product would
    /// pass, and the vector code would compute addresses from the stride.
    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn forward_rejects_a_stride_that_overflows() {
        let l = MdctLookup::new(256, 0);
        let input = vec![0f32; 256];
        let mut output = vec![0f32; 128];
        let window = vec![0f32; 64];
        l.forward(&input, &mut output, &window, 64, 0, wrapping_stride());
    }

    #[test]
    #[should_panic(expected = "input too small")]
    fn backward_rejects_a_stride_that_overflows() {
        let l = MdctLookup::new(256, 0);
        let input = vec![0f32; 128];
        let mut output = vec![0f32; 256];
        let window = vec![0f32; 64];
        l.backward(&input, &mut output, &window, 64, 0, wrapping_stride());
    }

    #[cfg(target_arch = "aarch64")]
    /// `n` values of test data of one input class.
    fn data(rng: &mut Lcg, n: usize, class: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        fill(rng, &mut v, class);
        v
    }

    #[cfg(target_arch = "aarch64")]
    /// A permutation of `0..n`.
    fn permutation(rng: &mut Lcg, n: usize) -> Vec<i16> {
        let mut v: Vec<i16> = (0..n as i16).collect();
        for i in (1..n).rev() {
            v.swap(i, rng.next_u32() as usize % (i + 1));
        }
        v
    }

    #[cfg(target_arch = "aarch64")]
    fn complex_data(rng: &mut Lcg, n: usize, class: usize) -> Vec<KissCpx> {
        data(rng, 2 * n, class)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| KissCpx::new(c[0], c[1]))
            .collect()
    }

    #[cfg(target_arch = "aarch64")]
    fn flat(c: &[KissCpx]) -> Vec<f32> {
        c.iter().flat_map(|c| [c.r, c.i]).collect()
    }

    #[cfg(target_arch = "aarch64")]
    /// A buffer of `n` complex values with nothing written yet.
    fn unwritten(n: usize) -> Vec<MaybeUninit<KissCpx>> {
        vec![MaybeUninit::uninit(); n]
    }

    #[cfg(target_arch = "aarch64")]
    /// The numbers of butterflies the step tests run: every count from the
    /// smallest a vector step takes up to 41, so that every number of
    /// butterflies left over after the full steps occurs several times, and
    /// the sizes of the codec.
    fn butterfly_counts() -> Vec<usize> {
        (4..=41).chain([60, 120, 240, 480]).collect()
    }

    // The tests below call each NEON step and its portable version with
    // `Fused` directly.
    // Every slice has exactly the length the step needs, the rotation
    // factors are arbitrary numbers and the bit-reversal table is an
    // arbitrary permutation.
    //
    // The NEON step runs twice. All its slices end at a block of memory that
    // must not be touched, then all of them start right behind one. A load
    // or a store outside a slice stops the test with a memory fault.

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fold_matches_scalar() {
        let mut rng = Lcg(0x5eed_2001);
        let mut regions: [Region; 5] = core::array::from_fn(|_| Region::new());
        for n4 in butterfly_counts() {
            // Butterflies with the window at each end: none, or 4 and more,
            // and the group in the middle is empty or has 4 and more.
            let qs = (0..=n4 / 2)
                .filter(|&q| (q == 0 || q >= 4) && (n4 == 2 * q || n4 - 2 * q >= 4));
            for q in qs {
                for class in 0..CLASSES {
                    let input = data(&mut rng, 2 * n4 + 4 * q, class);
                    let window = data(&mut rng, 4 * q, 0);
                    let trig = data(&mut rng, 2 * n4, 0);
                    let bitrev = permutation(&mut rng, n4);
                    let scale = 1.0 / n4 as f32;

                    let mut tmp = unwritten(n4);
                    let mut want = unwritten(n4);
                    forward_fold_rotate_scalar::<Fused>(&input, &window, &trig, &mut tmp, q, scale);
                    let tmp = unsafe { tmp.assume_init_ref() };
                    scatter_scalar(tmp, &bitrev, &mut want);
                    let want = flat(unsafe { want.assume_init_ref() });

                    for at_end in PLACEMENTS {
                        let [a, b, c, d, e] = &mut regions;
                        let got = e.place(&unwritten(n4), at_end);
                        unsafe {
                            neon::forward_fold_rotate(
                                a.place(&input, at_end),
                                b.place(&window, at_end),
                                c.place(&trig, at_end),
                                d.place(&bitrev, at_end),
                                got,
                                q,
                                scale,
                            )
                        };
                        let got = unsafe { got.assume_init_ref() };
                        assert_eq!(
                            first_difference(&flat(got), &want),
                            None,
                            "n4 {n4} q {q} class {class}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_forward_post_rotate_matches_scalar() {
        let mut rng = Lcg(0x5eed_2002);
        let mut regions: [Region; 3] = core::array::from_fn(|_| Region::new());
        for n4 in butterfly_counts() {
            for stride in [1usize, 2, 3, 8] {
                for class in 0..CLASSES {
                    let f2 = complex_data(&mut rng, n4, class);
                    let trig = data(&mut rng, 2 * n4, 0);
                    let before = data(&mut rng, stride * (2 * n4 - 1) + 1, 0);
                    let mut want = before.clone();
                    forward_post_rotate_scalar::<Fused>(&f2, &trig, &mut want, stride);

                    for at_end in PLACEMENTS {
                        let [a, b, c] = &mut regions;
                        let got = c.place(&before, at_end);
                        unsafe {
                            neon::forward_post_rotate(
                                a.place(&f2, at_end),
                                b.place(&trig, at_end),
                                got,
                                stride,
                            )
                        };
                        assert_eq!(
                            first_difference(got, &want),
                            None,
                            "n4 {n4} stride {stride} class {class}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_backward_pre_rotate_matches_scalar() {
        let mut rng = Lcg(0x5eed_2003);
        let mut regions: [Region; 4] = core::array::from_fn(|_| Region::new());
        for n4 in butterfly_counts() {
            for stride in [1usize, 2, 3, 8] {
                for class in 0..CLASSES {
                    let input = data(&mut rng, stride * (2 * n4 - 1) + 1, class);
                    let trig = data(&mut rng, 2 * n4, 0);
                    let bitrev = permutation(&mut rng, n4);

                    let mut tmp = unwritten(n4);
                    let mut want = unwritten(n4);
                    backward_pre_rotate_scalar::<Fused>(&input, &trig, &mut tmp, stride);
                    let tmp = unsafe { tmp.assume_init_ref() };
                    scatter_scalar(tmp, &bitrev, &mut want);
                    let want = flat(unsafe { want.assume_init_ref() });

                    for at_end in PLACEMENTS {
                        let [a, b, c, d] = &mut regions;
                        let got = d.place(&unwritten(n4), at_end);
                        unsafe {
                            neon::backward_pre_rotate(
                                a.place(&input, at_end),
                                b.place(&trig, at_end),
                                c.place(&bitrev, at_end),
                                got,
                                stride,
                            )
                        };
                        let got = unsafe { got.assume_init_ref() };
                        assert_eq!(
                            first_difference(&flat(got), &want),
                            None,
                            "n4 {n4} stride {stride} class {class}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_backward_post_rotate_matches_scalar() {
        let mut rng = Lcg(0x5eed_2004);
        let mut regions: [Region; 3] = core::array::from_fn(|_| Region::new());
        for n4 in butterfly_counts() {
            for class in 0..CLASSES {
                let f2 = complex_data(&mut rng, n4, class);
                let trig = data(&mut rng, 2 * n4, 0);
                let mut want = vec![0f32; 2 * n4];
                backward_post_rotate_scalar::<Fused>(&f2, &trig, &mut want);

                for at_end in PLACEMENTS {
                    let [a, b, c] = &mut regions;
                    let got = c.place(&vec![0f32; 2 * n4], at_end);
                    unsafe {
                        neon::backward_post_rotate(a.place(&f2, at_end), b.place(&trig, at_end), got)
                    };
                    assert_eq!(first_difference(got, &want), None, "n4 {n4} class {class}");
                }
            }
        }
    }

    /// Every overlap from 0 to 70, odd ones too: 0 to 17 vector steps and 0
    /// to 3 pairs left over, with and without an unpaired sample in the
    /// middle.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_tdac_matches_scalar() {
        let mut rng = Lcg(0x5eed_2005);
        let mut regions: [Region; 2] = core::array::from_fn(|_| Region::new());
        for overlap in 0..=70usize {
            for class in 0..CLASSES {
                let window = data(&mut rng, overlap, 0);
                let before = data(&mut rng, overlap, class);
                let mut want = before.clone();
                tdac_scalar::<Fused>(&mut want, &window);

                for at_end in PLACEMENTS {
                    let [a, b] = &mut regions;
                    let got = a.place(&before, at_end);
                    unsafe { neon::tdac(got, b.place(&window, at_end)) };
                    assert_eq!(first_difference(got, &want), None, "overlap {overlap} class {class}");
                }
            }
        }
    }

    /// The copied table is a rotation table: every entry is within one unit
    /// in the last place of the exact cosine. A lookup for the 48 kHz mode
    /// holds the table; a shorter one holds its start.
    #[test]
    fn rotation_table_values() {
        let mut inexact = 0;
        let mut k = 0;
        for n in [1920usize, 960, 480, 240] {
            for i in 0..n / 2 {
                let exact = (2.0 * core::f64::consts::PI * (i as f64 + 0.125) / n as f64).cos();
                let t = MDCT_TWIDDLES_960[k];
                assert!((t as f64 - exact).abs() <= 6.0e-8, "size {n} entry {i}");
                if t != exact as f32 {
                    inexact += 1;
                }
                k += 1;
            }
        }
        assert_eq!(k, 1800);
        // Why the table is copied and not computed.
        assert!(inexact > 0);

        let l = MdctLookup::new(1920, 3);
        assert_eq!(l.trig.as_slice(), &MDCT_TWIDDLES_960[..]);
        let l = MdctLookup::new(1920, 1);
        assert_eq!(l.trig.as_slice(), &MDCT_TWIDDLES_960[..1440]);
        // Another size is computed.
        let l = MdctLookup::new(960, 0);
        assert_eq!(l.trig.len(), 480);
        assert!((l.trig[0] as f64 - (2.0 * core::f64::consts::PI * 0.125 / 960.0).cos()).abs() < 6e-8);
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
