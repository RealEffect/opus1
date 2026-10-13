use crate::fixedvec::FixedVec;
use crate::pitch::pitch_xcorr;

pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    let mut error = ac[0];
    if error <= 1e-10 {
        for x in lpc.iter_mut() {
            *x = 0.0;
        }
        return;
    }

    for i in 0..p {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;

        lpc[i] = r;
        for j in 0..i.div_ceil(2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;

        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

/// Largest `n` that `autocorr` accepts together with a window.
const AUTOCORR_WINDOWED_MAX: usize = 2880;

/// Autocorrelation of `x[..n]` for the lags 0 to `lag`, written to
/// `ac[..=lag]`.
///
/// With a window, the first and the last `overlap` samples are multiplied by
/// `window[..overlap]` first (the last ones in reverse order). That needs a
/// copy of the signal, which is limited to 2880 samples.
pub fn autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
) {
    match window {
        Some(win) => autocorr_windowed(x, ac, win, overlap, lag, n),
        None => autocorr_of(&x[..n], ac, lag),
    }
}

/// `autocorr` with a window. This is a separate function, never inlined, so
/// that the copy of the signal (11.5 KB) takes stack space only in calls that
/// use a window.
#[inline(never)]
fn autocorr_windowed(x: &[f32], ac: &mut [f32], win: &[f32], overlap: usize, lag: usize, n: usize) {
    if x.len() < n {
        return;
    }
    // Built in place, and only `n` samples of it are written.
    // (`FixedVec::from_slice` builds the buffer and then copies all 11.5 KB
    // to its final place; `extend_from_slice` stores one sample at a time.)
    let mut xx: FixedVec<f32, AUTOCORR_WINDOWED_MAX> = FixedVec::new();
    xx.resize(n, 0.0);
    xx.copy_from_slice(&x[..n]);
    for i in 0..overlap {
        xx[i] *= win[i];
        xx[n - 1 - i] *= win[i];
    }
    autocorr_of(&xx, ac, lag);
}

/// Autocorrelation of all of `xx` for the lags 0 to `lag`.
fn autocorr_of(xx: &[f32], ac: &mut [f32], lag: usize) {
    let n = xx.len();
    let fast_n = n - lag;

    // The first `fast_n` products of every lag, several lags at a time.
    pitch_xcorr(xx, xx, ac, fast_n, lag + 1);

    // Lag `k` has `lag - k` more products: xx[i] * xx[i - k] for i from
    // `k + fast_n` on.
    for (k, out) in ac[..=lag].iter_mut().enumerate() {
        let mut d = 0.0f32;
        for (a, b) in xx[k + fast_n..].iter().zip(&xx[fast_n..]) {
            d += a * b;
        }
        *out += d;
    }
}

/// Filter with feed-forward taps only:
/// `y[i] = x[i] + sum over j < ord of num[j] * x[i - j - 1]`.
///
/// Samples before `x[0]` count as zero. The products are added in rising
/// order of `j`, each with a separate multiply and add. Every target gives
/// the same result.
pub fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    let x = &x[..n];
    let num = &num[..ord];
    let y = &mut y[..n];

    // The first `ord` outputs have fewer than `ord` samples before them.
    let head = ord.min(n);
    for i in 0..head {
        let mut sum = x[i];
        for (&c, &past) in num.iter().zip(x[..i].iter().rev()) {
            sum += c * past;
        }
        y[i] = sum;
    }
    if head == n {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    celt_fir_full_neon(x, num, y, head);
    #[cfg(not(target_arch = "aarch64"))]
    celt_fir_full_scalar(x, num, y, head);
}

/// `celt_fir` for the outputs from `start` on, which all have `num.len()`
/// samples before them. Portable code; `start >= num.len()`.
///
/// Four outputs are computed per pass. Each has its own running sum, so the
/// result is the same as one output after the other, and the compiler can
/// use vector instructions for the four sums.
#[allow(dead_code)] // unused on targets that have a vector kernel
fn celt_fir_full_scalar(x: &[f32], num: &[f32], y: &mut [f32], start: usize) {
    let n = y.len();
    let ord = num.len();
    assert!(x.len() == n && start >= ord);

    let mut i = start;
    while i + 4 <= n {
        let mut acc = [x[i], x[i + 1], x[i + 2], x[i + 3]];
        // Tap `j` reads x[i - j - 1..i - j + 3]: windows of four samples that
        // move back by one sample per tap.
        for (&c, w) in num.iter().zip(x[i - ord..i + 3].windows(4).rev()) {
            for k in 0..4 {
                acc[k] += c * w[k];
            }
        }
        y[i..i + 4].copy_from_slice(&acc);
        i += 4;
    }
    for i in i..n {
        let mut sum = x[i];
        for (&c, &past) in num.iter().zip(x[i - ord..i].iter().rev()) {
            sum += c * past;
        }
        y[i] = sum;
    }
}

/// NEON version of `celt_fir_full_scalar`, with the same operations per
/// output (no fused multiply-add), so it gives the same bits.
#[cfg(target_arch = "aarch64")]
fn celt_fir_full_neon(x: &[f32], num: &[f32], y: &mut [f32], start: usize) {
    use core::arch::aarch64::*;

    let n = y.len();
    let ord = num.len();
    assert!(x.len() == n && start >= ord);

    let mut i = start;
    // 16 outputs per pass in four registers. One register has to wait three
    // cycles for each addition of a tap; with four of them the processor
    // works on the other three in the meantime.
    while i + 16 <= n {
        // SAFETY: i >= ord and i + 16 <= n == x.len(). Tap `j < ord` reads
        // x[i - j - 1 + 4g..][..4] for g < 4, which starts at i - ord at the
        // earliest and ends before i + 15. The first loads read
        // x[i..i + 16], and the stores write y[i..i + 16].
        unsafe {
            let xp = x.as_ptr().add(i);
            let mut acc = [
                vld1q_f32(xp),
                vld1q_f32(xp.add(4)),
                vld1q_f32(xp.add(8)),
                vld1q_f32(xp.add(12)),
            ];
            let mut past = xp;
            for &c in num {
                past = past.sub(1);
                for (g, a) in acc.iter_mut().enumerate() {
                    *a = vaddq_f32(*a, vmulq_n_f32(vld1q_f32(past.add(4 * g)), c));
                }
            }
            for (g, a) in acc.iter().enumerate() {
                vst1q_f32(y.as_mut_ptr().add(i + 4 * g), *a);
            }
        }
        i += 16;
    }
    while i + 4 <= n {
        // SAFETY: i >= ord and i + 4 <= n == x.len(). Tap `j < ord` reads
        // x[i - j - 1..i - j + 3], which starts at i - ord at the earliest
        // and ends before i + 3. The first load reads x[i..i + 4], and the
        // store writes y[i..i + 4].
        unsafe {
            let mut acc = vld1q_f32(x.as_ptr().add(i));
            let mut past = x.as_ptr().add(i);
            for &c in num {
                past = past.sub(1);
                acc = vaddq_f32(acc, vmulq_n_f32(vld1q_f32(past), c));
            }
            vst1q_f32(y.as_mut_ptr().add(i), acc);
        }
        i += 4;
    }
    for i in i..n {
        let mut sum = x[i];
        for (&c, &past) in num.iter().zip(x[i - ord..i].iter().rev()) {
            sum += c * past;
        }
        y[i] = sum;
    }
}

/// Filter with feedback taps only:
/// `y[i] = x[i] - sum over j < ord of den[j] * y[i - j - 1]`.
///
/// `mem` holds the last `ord` outputs, newest first: on entry those of the
/// call before, on return those of this call.
///
/// The order of the additions differs per target, and each target keeps the
/// order it has always had:
/// - portable code, and aarch64 with fewer than four taps: starts with `x[i]`
///   and subtracts the products in rising order of `j`.
/// - aarch64 with four taps or more: see `celt_iir_step_neon`.
///
/// In both orders the first product of an output uses the output before it,
/// and all later additions wait for that product. So the outputs cannot be
/// computed side by side, as libopus does with a different order.
pub fn celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if ord >= 4 {
        celt_iir_neon(&x[..n], &den[..ord], &mut y[..n], &mut mem[..ord]);
        return;
    }
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        y[i] = sum;
    }
}

/// `celt_iir` on aarch64 for four taps or more. `x` and `y` have the same
/// length, and so have `den` and `mem`.
#[cfg(target_arch = "aarch64")]
fn celt_iir_neon(x: &[f32], den: &[f32], y: &mut [f32], mem: &mut [f32]) {
    let n = x.len();
    let ord = den.len();
    assert!(y.len() == n && mem.len() == ord && ord >= 4);

    // The first `ord` outputs read from `mem`, which moves by one sample per
    // output.
    let head = ord.min(n);
    for i in 0..head {
        let out = celt_iir_step_neon(x[i], den, mem);
        mem.copy_within(..ord - 1, 1);
        mem[0] = out;
        y[i] = out;
    }
    if head == n {
        return;
    }

    // From here on the last `ord` outputs are in `y` itself, so nothing has
    // to be moved per output.
    celt_iir_full_neon(x, den, y, head);
    for (m, &out) in mem.iter_mut().zip(y.iter().rev()) {
        *m = out;
    }
}

/// One output of `celt_iir` on aarch64 for four taps or more, with the
/// earlier outputs in `past`, newest first.
///
/// The products are added with fused multiply-add into four running sums
/// that take the taps 4m, 4m+1, 4m+2 and 4m+3. Those are added as
/// (s0 + s2) + (s1 + s3). Then the last `ord % 4` products are added one by
/// one with a separate multiply and add, and the total is subtracted from
/// `x`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn celt_iir_step_neon(x: f32, den: &[f32], past: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let (den4, den_rest) = den.as_chunks::<4>();
    let (past4, past_rest) = past[..den.len()].as_chunks::<4>();
    // SAFETY: NEON is always present on aarch64. Each load reads one
    // `[f32; 4]`.
    let mut fb = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for (c, p) in den4.iter().zip(past4) {
            acc = vfmaq_f32(acc, vld1q_f32(c.as_ptr()), vld1q_f32(p.as_ptr()));
        }
        let pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair)
    };
    for (&c, &p) in den_rest.iter().zip(past_rest) {
        fb += c * p;
    }
    x - fb
}

/// `celt_iir` on aarch64 for four taps or more, for the outputs from `start`
/// on, where the last `den.len()` outputs are in `y`; `start >= den.len()`.
/// Same bits as `celt_iir_step_neon` for every output.
///
/// `y` holds the earlier outputs oldest first, the other way round than
/// `celt_iir_step_neon` wants them. So each group of four taps is reversed
/// instead. That leaves every product and every running sum as it was and
/// only swaps the lanes: the running sums are then added as
/// (s2 + s0) + (s3 + s1), which is the same value.
///
/// The four newest outputs are kept in a register. Loading them from `y`
/// would have to wait for the store of the newest output every time.
#[cfg(target_arch = "aarch64")]
fn celt_iir_full_neon(x: &[f32], den: &[f32], y: &mut [f32], start: usize) {
    use core::arch::aarch64::*;

    let n = y.len();
    let ord = den.len();
    // `start <= n` is needed too: the first load below reads
    // y[start - 4..start] even when the loop does not run.
    assert!(x.len() == n && ord >= 4 && start >= ord && start <= n);
    let (den4, den_rest) = den.as_chunks::<4>();
    let blocks = den4.len();

    // SAFETY: 4 <= ord <= start <= n, and i runs from `start` to n - 1. The
    // load before the loop reads y[start - 4..start]. Group
    // `m < blocks` reads y[i - 4m - 4..i - 4m], and tap `j < ord` of the rest
    // reads y[i - 1 - j]; both start at y[i - ord] at the earliest. x[i] and
    // y[i] are inside the slices because i < n. Each load of taps reads one
    // `[f32; 4]` of `den`.
    unsafe {
        let reversed = |v: float32x4_t| {
            let r = vrev64q_f32(v);
            vextq_f32::<2>(r, r)
        };
        let yp = y.as_mut_ptr();
        // The outputs i - 4 to i - 1, oldest in lane 0.
        let mut newest = vld1q_f32(yp.add(start - 4));
        for i in start..n {
            let taps = reversed(vld1q_f32(den4[0].as_ptr()));
            let mut acc = vfmaq_f32(vdupq_n_f32(0.0), taps, newest);
            for (m, group) in den4.iter().enumerate().skip(1) {
                let taps = reversed(vld1q_f32(group.as_ptr()));
                acc = vfmaq_f32(acc, taps, vld1q_f32(yp.add(i - 4 * m - 4)));
            }
            // The lanes are (s3, s2, s1, s0).
            let pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let mut fb = vget_lane_f32::<1>(pair) + vget_lane_f32::<0>(pair);
            for (k, &c) in den_rest.iter().enumerate() {
                fb += c * *yp.add(i - 1 - (4 * blocks + k));
            }
            let out = *x.get_unchecked(i) - fb;
            *yp.add(i) = out;
            // Shift the register by one output. Only the insertion of `out`
            // has to wait for it.
            newest = vsetq_lane_f32::<3>(out, vextq_f32::<1>(newest, newest));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::tests::{KINDS, Kind, Rng, assert_same, samples};

    /// Filter taps: small enough for a stable feedback filter with ordinary
    /// audio, and as wild as the samples for the other kinds.
    fn taps(rng: &mut Rng, ord: usize, kind: Kind) -> Vec<f32> {
        match kind {
            Kind::Audio => (0..ord).map(|_| rng.unit() / (ord as f32 + 1.0)).collect(),
            _ => samples(rng, ord, kind),
        }
    }

    /// `celt_fir` as the portable code computed it before: one output at a
    /// time, with a test per tap.
    fn celt_fir_plain(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
        for i in 0..n {
            let mut sum = x[i];
            for j in 0..ord {
                if i > j {
                    sum += num[j] * x[i - j - 1];
                }
            }
            y[i] = sum;
        }
    }

    /// `celt_iir` as the portable code computed it before: `mem` is moved by
    /// one sample per output.
    fn celt_iir_plain(
        x: &[f32],
        den: &[f32],
        y: &mut [f32],
        n: usize,
        ord: usize,
        mem: &mut [f32],
    ) {
        for i in 0..n {
            let mut sum = x[i];
            for j in 0..ord {
                sum -= den[j] * mem[j];
            }
            for j in (1..ord).rev() {
                mem[j] = mem[j - 1];
            }
            if ord > 0 {
                mem[0] = sum;
            }
            y[i] = sum;
        }
    }

    /// `autocorr` as it was before: the windowed signal is built with
    /// `FixedVec::from_slice`, and the last products use indexing.
    fn autocorr_plain(
        x: &[f32],
        ac: &mut [f32],
        window: Option<&[f32]>,
        overlap: usize,
        lag: usize,
        n: usize,
    ) {
        let mut xx = x[..n].to_vec();
        if let Some(win) = window {
            for i in 0..overlap {
                xx[i] *= win[i];
                xx[n - 1 - i] *= win[i];
            }
        }
        let fast_n = n - lag;
        pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);
        for k in 0..=lag {
            let mut d = 0.0f32;
            for i in (k + fast_n)..n {
                d += xx[i] * xx[i - k];
            }
            ac[k] += d;
        }
    }

    #[test]
    fn celt_fir_matches_the_plain_version() {
        let mut rng = Rng::new(21);
        for kind in KINDS {
            // Lengths 0 and 1, every tail length, and shorter than the filter.
            for n in (0..=40).chain([100, 241]) {
                for ord in (0..=13).chain([16, 24, 25, 26, 50]) {
                    let x = samples(&mut rng, n, kind);
                    let num = taps(&mut rng, ord, kind);
                    let mut got = vec![f32::NAN; n];
                    celt_fir(&x, &num, &mut got, n, ord);
                    let mut want = vec![f32::NAN; n];
                    celt_fir_plain(&x, &num, &mut want, n, ord);
                    assert_same(&got, &want, &format!("{kind:?} n {n} ord {ord}"));
                }
            }
        }
    }

    #[test]
    fn celt_fir_full_scalar_matches_the_plain_version() {
        let mut rng = Rng::new(22);
        for kind in KINDS {
            for n in (0..=40).chain([100, 241]) {
                for ord in (0..=13).chain([16, 24, 25]) {
                    if ord > n {
                        continue;
                    }
                    let x = samples(&mut rng, n, kind);
                    let num = taps(&mut rng, ord, kind);
                    let mut want = vec![f32::NAN; n];
                    celt_fir_plain(&x, &num, &mut want, n, ord);
                    // The outputs before `ord` come from the plain version.
                    let mut got = want.clone();
                    for v in got.iter_mut().skip(ord) {
                        *v = f32::NAN;
                    }
                    celt_fir_full_scalar(&x, &num, &mut got, ord);
                    assert_same(&got, &want, &format!("{kind:?} n {n} ord {ord}"));
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn celt_fir_full_neon_matches_scalar() {
        let mut rng = Rng::new(23);
        for kind in KINDS {
            for n in (0..=40).chain([100, 241]) {
                for ord in (0..=13).chain([16, 24, 25]) {
                    if ord > n {
                        continue;
                    }
                    let x = samples(&mut rng, n, kind);
                    let num = taps(&mut rng, ord, kind);
                    let head = samples(&mut rng, n, kind);
                    let mut got = head.clone();
                    celt_fir_full_neon(&x, &num, &mut got, ord);
                    let mut want = head.clone();
                    celt_fir_full_scalar(&x, &num, &mut want, ord);
                    assert_same(&got, &want, &format!("{kind:?} n {n} ord {ord}"));
                }
            }
        }
    }

    /// The public function against the plain version, where the target uses
    /// the portable order: everywhere but aarch64 with four taps or more.
    #[test]
    fn celt_iir_matches_the_plain_version() {
        let mut rng = Rng::new(25);
        for kind in KINDS {
            for n in (0..=40).chain([100, 241]) {
                // (With no tap at all the function still writes mem[0], so
                // that case needs a memory of one sample and is left out.)
                for ord in (1..=13).chain([16, 24, 25]) {
                    if cfg!(target_arch = "aarch64") && ord >= 4 {
                        continue;
                    }
                    let x = samples(&mut rng, n, kind);
                    let den = taps(&mut rng, ord, kind);
                    let mem0 = samples(&mut rng, ord, kind);
                    let what = format!("{kind:?} n {n} ord {ord}");

                    let mut got = vec![f32::NAN; n];
                    let mut got_mem = mem0.clone();
                    celt_iir(&x, &den, &mut got, n, ord, &mut got_mem);
                    let mut want = vec![f32::NAN; n];
                    let mut want_mem = mem0.clone();
                    celt_iir_plain(&x, &den, &mut want, n, ord, &mut want_mem);
                    assert_same(&got, &want, &what);
                    assert_same(&got_mem, &want_mem, &what);
                }
            }
        }
    }

    /// `celt_iir` for aarch64 as it was before, for four taps or more: one
    /// output per pass, `mem` moved by one sample per output.
    #[cfg(target_arch = "aarch64")]
    fn celt_iir_neon_old(
        x: &[f32],
        den: &[f32],
        y: &mut [f32],
        n: usize,
        ord: usize,
        mem: &mut [f32],
    ) {
        use core::arch::aarch64::*;
        assert!(ord >= 4 && den.len() >= ord && mem.len() >= ord);
        for i in 0..n {
            let mut fb_sum = unsafe {
                let mut feedback = vdupq_n_f32(0.0);
                let mut j = 0;
                while j + 4 <= ord {
                    let coeff = vld1q_f32(den.as_ptr().add(j));
                    let mem_vals = vld1q_f32(mem.as_ptr().add(j));
                    feedback = vfmaq_f32(feedback, coeff, mem_vals);
                    j += 4;
                }
                let fb_pair = vadd_f32(vget_low_f32(feedback), vget_high_f32(feedback));
                vget_lane_f32(fb_pair, 0) + vget_lane_f32(fb_pair, 1)
            };
            for j in ord / 4 * 4..ord {
                fb_sum += den[j] * mem[j];
            }
            let sum = x[i] - fb_sum;
            for j in (1..ord).rev() {
                mem[j] = mem[j - 1];
            }
            mem[0] = sum;
            y[i] = sum;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn celt_iir_matches_the_old_neon_version() {
        let mut rng = Rng::new(26);
        for kind in KINDS {
            // Shorter than the filter, every tail length, and long runs.
            for n in (0..=40).chain([100, 241]) {
                for ord in (4..=13).chain([16, 23, 24, 25, 26, 27, 50]) {
                    let x = samples(&mut rng, n, kind);
                    let den = taps(&mut rng, ord, kind);
                    let mem0 = samples(&mut rng, ord, kind);
                    let what = format!("{kind:?} n {n} ord {ord}");

                    let mut got = vec![f32::NAN; n];
                    let mut got_mem = mem0.clone();
                    celt_iir(&x, &den, &mut got, n, ord, &mut got_mem);
                    let mut want = vec![f32::NAN; n];
                    let mut want_mem = mem0.clone();
                    celt_iir_neon_old(&x, &den, &mut want, n, ord, &mut want_mem);
                    assert_same(&got, &want, &what);
                    assert_same(&got_mem, &want_mem, &what);
                }
            }
        }
    }

    #[test]
    fn celt_iir_continues_from_its_memory() {
        // Two calls give the same outputs as one call over both parts.
        let mut rng = Rng::new(27);
        for ord in [1usize, 3, 4, 10, 24] {
            for split in [0usize, 1, 5, 24, 60] {
                let n = 120;
                let x = samples(&mut rng, n, Kind::Audio);
                let den = taps(&mut rng, ord, Kind::Audio);
                let mut whole = vec![0.0f32; n];
                let mut mem = vec![0.0f32; ord];
                celt_iir(&x, &den, &mut whole, n, ord, &mut mem);

                let mut parts = vec![0.0f32; n];
                let mut mem2 = vec![0.0f32; ord];
                let (y1, y2) = parts.split_at_mut(split);
                celt_iir(&x[..split], &den, y1, split, ord, &mut mem2);
                celt_iir(&x[split..], &den, y2, n - split, ord, &mut mem2);
                assert_same(&parts, &whole, &format!("ord {ord} split {split}"));
                assert_same(&mem2, &mem, &format!("ord {ord} split {split}"));
            }
        }
    }

    #[test]
    fn autocorr_matches_the_plain_version() {
        let mut rng = Rng::new(28);
        let win: Vec<f32> = (0..120).map(|i| ((i as f32 + 0.5) / 120.0).sin()).collect();
        for kind in KINDS {
            for (n, lag, overlap) in [
                (5usize, 4usize, 0usize),
                (24, 4, 0),
                (25, 24, 0),
                (64, 8, 20),
                (240, 24, 120),
                (988, 4, 0),
                (1024, 24, 120),
                (2880, 24, 120),
            ] {
                let x = samples(&mut rng, n, kind);
                for window in [None, Some(&win[..])] {
                    if window.is_none() && overlap > 0 {
                        continue;
                    }
                    let mut got = vec![f32::NAN; lag + 1];
                    autocorr(&x, &mut got, window, overlap, lag, n);
                    let mut want = vec![f32::NAN; lag + 1];
                    autocorr_plain(&x, &mut want, window, overlap, lag, n);
                    assert_same(
                        &got,
                        &want,
                        &format!("{kind:?} n {n} lag {lag} window {}", window.is_some()),
                    );
                }
            }
        }
    }
}
