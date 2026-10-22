use crate::analysis::{AnalysisInfo, LEAK_BANDS};
use crate::bands::{
    SPREAD_NONE, SPREAD_NORMAL, compute_band_energies, haar1, log2amp,
    normalise_bands, quant_all_bands, spreading_decision,
};
use crate::celt_lpc::{self, autocorr, celt_fir, celt_iir};
use crate::mathops::{celt_exp2, celt_log2, mac, mac_f64};
use crate::pitch::{pitch_downsample, pitch_search};
use crate::modes::{CeltMode, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF};
use crate::quant_bands::{
    quant_coarse_energy_advanced, quant_energy_finalise, quant_fine_energy, unquant_coarse_energy,
    unquant_energy_finalise, unquant_fine_energy,
};
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, clt_compute_allocation};

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::deep_plc::DeepPlcArgs;
use crate::fixedvec::FixedVec;
use crate::in_place::write_fields;
use crate::{OPUS_BITRATE_MAX, bitrate_to_bits};

// --- Heap-free buffer capacity constants (all sized for the worst case:
//     2 channels, 48 kHz, max frame). Runtime construction uses smaller logical
//     lengths when fewer channels / bands are active; the FixedVec tracks that. ---
/// Maximum channel count supported by the API.
const CELT_MAX_CHANNELS: usize = 2;
/// Number of energy bands (`nb_ebands`) for the 48 kHz / 120-overlap mode.
const CELT_NB_EBANDS: usize = 21;
/// `nb_ebands * max_channels` = worst-case per-band-per-channel element count.
const CELT_NB_X_CH: usize = CELT_NB_EBANDS * CELT_MAX_CHANNELS;
/// MDCT overlap for the default mode.
const CELT_OVERLAP: usize = 120;
/// Encoder `in_mem`: `channels * overlap`.
const CELT_IN_MEM: usize = CELT_MAX_CHANNELS * CELT_OVERLAP;
/// Longest frame the encoder codes, in samples at 48 kHz (20 ms).
/// `CeltEncoder::encode` panics on a longer one.
const CELT_ENC_MAX_FRAME: usize = 960;
/// Encoder input buffer stride: `(CELT_ENC_MAX_FRAME + overlap) * channels`.
const CELT_BUFSTRIDE: usize = (CELT_ENC_MAX_FRAME + CELT_OVERLAP) * CELT_MAX_CHANNELS;
/// `CELT_ENC_MAX_FRAME * channels`.
const CELT_FRAME_X_CH: usize = CELT_ENC_MAX_FRAME * CELT_MAX_CHANNELS;
/// Frequency scratch (+4 padding for NEON pre-rotation over-read).
const CELT_W_FREQ: usize = CELT_FRAME_X_CH + 4;
/// PVQ stride-access scratch: `frame_x_ch + STRIDE_ACCESS_PAD`.
const CELT_W_X_ENC: usize = CELT_FRAME_X_CH + STRIDE_ACCESS_PAD;
/// Room per channel in `CeltEncoder::prefilter_buf`: the history of the
/// pre-filter and one frame of input after it.
const PREFILTER_STRIDE: usize = COMBFILTER_MAXPERIOD + CELT_ENC_MAX_FRAME;
/// `CeltEncoder::prefilter_buf`: `channels * PREFILTER_STRIDE`.
const CELT_PREFILTER_BUF: usize = CELT_MAX_CHANNELS * PREFILTER_STRIDE;
/// Prefilter pitch buffer: `(COMBFILTER_MAXPERIOD + CELT_ENC_MAX_FRAME) >> 1`.
const CELT_PREFILTER_PITCH: usize = (COMBFILTER_MAXPERIOD + CELT_ENC_MAX_FRAME) >> 1;
/// Longest frame the decoder produces, in samples at 48 kHz (20 ms).
const CELT_DEC_MAX_FRAME: usize = 960;
/// Decoder `w_x`: the normalised spectrum of two channels and the padding
/// that the band kernels may read past its end. The synthesis turns it into
/// the spectrum of the frame in place. The de-emphasis and the loss
/// concealment use the same array as working memory afterwards.
const CELT_W_X_DEC: usize = {
    let spectrum = CELT_DEC_MAX_FRAME * CELT_MAX_CHANNELS + STRIDE_ACCESS_PAD;
    let conceal = 2 * PLC_MAX_PERIOD + PLC_LPC_ORDER;
    if spectrum > conceal {
        spectrum
    } else {
        conceal
    }
};
/// Samples that each channel's output history can slide forward in
/// `decode_mem` before it is moved back to the start (see
/// `CeltDecoder::advance_history`).
const CELT_HIST_SLACK: usize = CELT_DEC_MAX_FRAME;
/// Samples per channel in the decoder's `decode_mem`: the history
/// (`DECODE_BUFFER_SIZE + overlap`) and the room it slides through.
const CELT_HIST_STRIDE: usize = DECODE_BUFFER_SIZE + CELT_OVERLAP + CELT_HIST_SLACK;
/// Decoder `decode_mem`: `channels * CELT_HIST_STRIDE`.
const CELT_DECODE_MEM: usize = CELT_MAX_CHANNELS * CELT_HIST_STRIDE;

/// libopus `resampling_factor`: how many samples at 48 kHz, the internal rate
/// of CELT, one sample at `sampling_rate` stands for. `None` for a rate CELT
/// does not take.
pub(crate) fn resampling_factor(sampling_rate: i32) -> Option<usize> {
    match sampling_rate {
        48000 => Some(1),
        24000 => Some(2),
        16000 => Some(3),
        12000 => Some(4),
        8000 => Some(6),
        _ => None,
    }
}

/// libopus `compute_mdcts`: the MDCT of every input channel, for one long
/// block or for `1 << lm` short blocks (`short_blocks`), interleaved.
///
/// `in_buf` holds `overlap + frame_size` samples per input channel
/// (`cc` channels, `buf_stride` apart). `freq` gets `frame_size` values
/// per channel. When the packet codes fewer channels than the input has
/// (`channels < cc`), the spectrum of the one coded channel is half the
/// sum of the two. With `upsample != 1` the spectrum is then scaled and
/// limited to the band of the input (`upsample_scale_zero`).
#[allow(clippy::too_many_arguments)]
fn compute_mdcts(
    mode: &CeltMode,
    short_blocks: bool,
    in_buf: &[f32],
    buf_stride: usize,
    freq: &mut [f32],
    frame_size: usize,
    channels: usize,
    cc: usize,
    lm: usize,
    upsample: usize,
) {
    let overlap = mode.overlap;
    for c in 0..cc {
        let input = &in_buf[c * buf_stride..(c + 1) * buf_stride];
        let out = &mut freq[c * frame_size..];
        if short_blocks {
            let b = 1usize << lm;
            let n = frame_size / b;
            for i in 0..b {
                mode.mdct
                    .forward(&input[i * n..], &mut out[i..], mode.window, overlap, mode.max_lm, b);
            }
        } else {
            mode.mdct
                .forward(input, out, mode.window, overlap, mode.max_lm - lm, 1);
        }
    }
    if cc == 2 && channels == 1 {
        let (mono, second) = freq[..2 * frame_size].split_at_mut(frame_size);
        for (m, &s) in mono.iter_mut().zip(second.iter()) {
            *m = 0.5 * *m + 0.5 * s;
        }
    }
    upsample_scale_zero(freq, frame_size, upsample, channels);
}

/// libopus `compute_mdcts` with `upsample != 1`: keep only the first
/// `N/upsample` coefficients (the API Nyquist band), scale them by `upsample`
/// to preserve energy after zero-stuffing, and zero the rest.
pub(crate) fn upsample_scale_zero(freq: &mut [f32], frame_size: usize, upsample: usize, channels: usize) {
    if upsample <= 1 {
        return;
    }
    let bound = frame_size / upsample;
    for c in 0..channels {
        let base = c * frame_size;
        for v in freq[base..base + bound].iter_mut() {
            *v *= upsample as f32;
        }
        freq[base + bound..base + frame_size].fill(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Sum of `|x[i]|`, four partial sums side by side.
///
/// Partial sum `k` adds samples `k`, `k + 4`, `k + 8` and so on, in that
/// order. The four are then added as `(s0 + s1) + (s2 + s3)`, and the last
/// `len % 4` samples are added one by one. `sum_abs_lanes_reference` in the
/// tests is this order written as a plain loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn sum_abs_neon(x: &[f32]) -> f32 {
    let n = x.len();
    let p = x.as_ptr();
    let mut i = 0;
    // SAFETY: every load reads the 4 values at `p + i .. p + i + 4`, and the
    // loop condition keeps `i + 4 <= n == x.len()`.
    let mut sum = unsafe {
        let mut sum_vec = vdupq_n_f32(0.0);
        while i + 4 <= n {
            // A plain add. A fused multiply-add with a factor of 1.0 gives
            // the same bits one cycle later.
            sum_vec = vaddq_f32(sum_vec, vabsq_f32(vld1q_f32(p.add(i))));
            i += 4;
        }
        vaddvq_f32(sum_vec)
    };

    for &v in &x[i..] {
        sum += v.abs();
    }

    sum
}

#[inline(always)]
fn sum_abs(x: &[f32]) -> f32 {
    // SAFETY: the AVX version runs only when the processor has AVX, and it
    // reads `x.len()` values of `x`.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if crate::compat::x86_has_avx() {
            return sum_abs_avx(x, x.len());
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        sum_abs_neon(x)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().map(|&v| v.abs()).sum()
    }
}

/// Largest of `init` and every `|x[i]|` (libopus `celt_maxabs_res`).
///
/// A sample that is not a number never becomes the maximum, because the
/// test `v > max` is false for it. This is the portable version, and the
/// reference the vector version is tested against.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn max_abs_scalar(x: &[f32], init: f32) -> f32 {
    let mut m = init;
    for &v in x {
        let v = v.abs();
        if v > m {
            m = v;
        }
    }
    m
}

/// `max_abs_scalar` with 16 samples per pass.
///
/// Each of the 16 positions keeps its own maximum with the same test
/// `v > max`, so a sample that is not a number is skipped here too. A
/// maximum does not depend on the order of the comparisons, so the result
/// has the same bits as the scalar one.
#[cfg(target_arch = "aarch64")]
#[inline]
fn max_abs_neon(x: &[f32], init: f32) -> f32 {
    let n = x.len();
    let p = x.as_ptr();
    let mut i = 0;
    // SAFETY: every load reads the 4 values at `p + i .. p + i + 4`, and the
    // loop conditions keep `i + 4 <= n` (`i + 16 <= n` for the four loads of
    // the first loop).
    let lanes: [f32; 4] = unsafe {
        let mut m0 = vdupq_n_f32(init);
        let mut m1 = m0;
        let mut m2 = m0;
        let mut m3 = m0;
        while i + 16 <= n {
            let v0 = vabsq_f32(vld1q_f32(p.add(i)));
            let v1 = vabsq_f32(vld1q_f32(p.add(i + 4)));
            let v2 = vabsq_f32(vld1q_f32(p.add(i + 8)));
            let v3 = vabsq_f32(vld1q_f32(p.add(i + 12)));
            m0 = vbslq_f32(vcgtq_f32(v0, m0), v0, m0);
            m1 = vbslq_f32(vcgtq_f32(v1, m1), v1, m1);
            m2 = vbslq_f32(vcgtq_f32(v2, m2), v2, m2);
            m3 = vbslq_f32(vcgtq_f32(v3, m3), v3, m3);
            i += 16;
        }
        while i + 4 <= n {
            let v0 = vabsq_f32(vld1q_f32(p.add(i)));
            m0 = vbslq_f32(vcgtq_f32(v0, m0), v0, m0);
            i += 4;
        }
        m0 = vbslq_f32(vcgtq_f32(m1, m0), m1, m0);
        m2 = vbslq_f32(vcgtq_f32(m3, m2), m3, m2);
        m0 = vbslq_f32(vcgtq_f32(m2, m0), m2, m0);
        core::mem::transmute::<float32x4_t, [f32; 4]>(m0)
    };
    let mut m = lanes[0];
    for &v in &lanes[1..] {
        if v > m {
            m = v;
        }
    }
    for &v in &x[i..] {
        let v = v.abs();
        if v > m {
            m = v;
        }
    }
    m
}

/// Largest of `init` and every `|x[i]|`; see `max_abs_scalar`.
#[inline]
pub(crate) fn max_abs(x: &[f32], init: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        max_abs_neon(x, init)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        max_abs_scalar(x, init)
    }
}

/// Pre-emphasis of one channel (libopus `celt_preemphasis`, float build).
///
/// `pcm` is the input at the API rate, in the range -1 to 1. `out` gets
/// `pcm.len() * upsample` samples at 48 kHz. Each input sample is scaled to
/// the 16-bit range and followed by `upsample - 1` zeros. That signal `x`
/// goes through the filter `out[i] = x[i] - coef * x[i - 1]`.
///
/// `mem` is `coef * x[-1]` from the previous frame. The function returns the
/// value to pass for the next frame.
#[inline]
pub(crate) fn celt_preemphasis(pcm: &[f32], out: &mut [f32], upsample: usize, coef: f32, mem: f32) -> f32 {
    debug_assert_eq!(out.len(), pcm.len() * upsample);
    let mut m = mem;
    match upsample {
        1 => {
            #[cfg(target_arch = "aarch64")]
            {
                m = preemphasis_neon(pcm, out, coef, m);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                m = preemphasis_scalar(pcm, out, coef, m);
            }
        }
        // 24 kHz input. The same loop as below, with a block length the
        // compiler knows.
        2 => m = preemphasis_stuffed::<2>(pcm, out, coef, m),
        _ => {
            for (block, &p) in out.chunks_exact_mut(upsample).zip(pcm) {
                m = preemphasis_block(block, p, coef, m);
            }
        }
    }
    m
}

/// `celt_preemphasis` for 48 kHz input (no stuffed zeros). This is the
/// portable version, and the reference the vector version is tested against.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn preemphasis_scalar(pcm: &[f32], out: &mut [f32], coef: f32, mem: f32) -> f32 {
    let mut m = mem;
    for (d, &p) in out.iter_mut().zip(pcm) {
        let x = p * 32768.0;
        *d = x - m;
        m = x * coef;
    }
    m
}

/// `preemphasis_scalar` with 8 samples per pass.
///
/// `out[i] = x[i] - coef * x[i - 1]` does not depend on `out[i - 1]`, so the
/// samples can be computed side by side. The product and the difference
/// stay two operations with a rounding each, as in the scalar loop, so the
/// result has the same bits.
#[cfg(target_arch = "aarch64")]
#[inline]
fn preemphasis_neon(pcm: &[f32], out: &mut [f32], coef: f32, mem: f32) -> f32 {
    const SCALE: f32 = 32768.0;
    let n = pcm.len();
    // The stores below rely on this. A caller that passes a shorter `out`
    // gets a panic, not a write past its end.
    assert!(out.len() >= n, "celt_preemphasis: output shorter than input");
    if n == 0 {
        return mem;
    }
    out[0] = pcm[0] * SCALE - mem;
    let mut i = 1;
    // SAFETY: `1 <= i` and `i + 4 <= n` at every group of four (`i + 8 <= n`
    // in the first loop, which handles two groups). A group loads
    // `pcm[i - 1 .. i + 3]` and `pcm[i .. i + 4]` and stores
    // `out[i .. i + 4]`. All of these are inside the slices, because
    // `pcm.len() == n` and `out.len() >= n`.
    unsafe {
        let p = pcm.as_ptr();
        let o = out.as_mut_ptr();
        let scale = vdupq_n_f32(SCALE);
        let coefv = vdupq_n_f32(coef);
        while i + 8 <= n {
            let x_a = vmulq_f32(vld1q_f32(p.add(i)), scale);
            let x_b = vmulq_f32(vld1q_f32(p.add(i + 4)), scale);
            let prev_a = vmulq_f32(vld1q_f32(p.add(i - 1)), scale);
            let prev_b = vmulq_f32(vld1q_f32(p.add(i + 3)), scale);
            vst1q_f32(o.add(i), vsubq_f32(x_a, vmulq_f32(prev_a, coefv)));
            vst1q_f32(o.add(i + 4), vsubq_f32(x_b, vmulq_f32(prev_b, coefv)));
            i += 8;
        }
        while i + 4 <= n {
            let x_a = vmulq_f32(vld1q_f32(p.add(i)), scale);
            let prev_a = vmulq_f32(vld1q_f32(p.add(i - 1)), scale);
            vst1q_f32(o.add(i), vsubq_f32(x_a, vmulq_f32(prev_a, coefv)));
            i += 4;
        }
    }
    // The filter memory after sample `i - 1`, then the last samples.
    let mut m = pcm[i - 1] * SCALE * coef;
    for (d, &p) in out[i..n].iter_mut().zip(&pcm[i..]) {
        let x = p * SCALE;
        *d = x - m;
        m = x * coef;
    }
    m
}

/// `celt_preemphasis` for an input with `UP - 1` zeros after every sample.
#[inline(always)]
fn preemphasis_stuffed<const UP: usize>(pcm: &[f32], out: &mut [f32], coef: f32, mem: f32) -> f32 {
    let mut m = mem;
    for (block, &p) in out.as_chunks_mut::<UP>().0.iter_mut().zip(pcm) {
        m = preemphasis_block(block, p, coef, m);
    }
    m
}

/// One input sample `p` and the zeros that follow it. `block` gets the
/// filter output for these `block.len()` samples.
#[inline(always)]
fn preemphasis_block(block: &mut [f32], p: f32, coef: f32, mem: f32) -> f32 {
    let x = p * 32768.0;
    block[0] = x - mem;
    let mut m = x * coef;
    for d in block[1..].iter_mut() {
        // The filter step for an input of zero.
        *d = 0.0 - m;
        m = 0.0 * coef;
    }
    m
}

/// Pre-emphasis of the `cc` channels of the interleaved input `pcm` (libopus
/// `celt_preemphasis` with a channel stride of `CC`), `n` samples per
/// channel at the API rate.
///
/// Channel `c` goes to `in_buf[c * buf_stride + overlap..]`: `n * upsample`
/// samples at 48 kHz. `mem[c]` is its filter memory (see
/// `celt_preemphasis`), and it is updated for the next frame.
#[allow(clippy::too_many_arguments)]
#[inline]
fn preemphasis_input(
    pcm: &[f32],
    cc: usize,
    in_buf: &mut [f32],
    buf_stride: usize,
    overlap: usize,
    upsample: usize,
    coef: f32,
    mem: &mut [f32],
) {
    let n = pcm.len() / cc;
    let frame = overlap..overlap + n * upsample;
    if cc == 1 {
        mem[0] = celt_preemphasis(pcm, &mut in_buf[frame], upsample, coef, mem[0]);
        return;
    }
    let (left, right) = in_buf.split_at_mut(buf_stride);
    let pairs = pcm.as_chunks::<2>().0;
    let (left, right) = (&mut left[frame.clone()], &mut right[frame]);
    let m = [mem[0], mem[1]];
    let m = match upsample {
        #[cfg(target_arch = "aarch64")]
        1 => preemphasis_stereo_neon(pairs, left, right, coef, m),
        #[cfg(not(target_arch = "aarch64"))]
        1 => preemphasis_stereo_scalar(pairs, left, right, coef, m),
        // 24 kHz input, with a block length the compiler knows.
        2 => preemphasis_stereo_blocks(
            pairs,
            left.as_chunks_mut::<2>().0,
            right.as_chunks_mut::<2>().0,
            coef,
            m,
        ),
        _ => preemphasis_stereo_blocks(
            pairs,
            left.chunks_exact_mut(upsample),
            right.chunks_exact_mut(upsample),
            coef,
            m,
        ),
    };
    mem[..2].copy_from_slice(&m);
}

/// `preemphasis_scalar` for interleaved stereo: the left channel of each
/// pair goes to `left`, the right channel to `right`. `mem` and the result
/// hold the memories of the two channels. This is the portable version,
/// and the reference the vector version is tested against.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn preemphasis_stereo_scalar(
    pairs: &[[f32; 2]],
    left: &mut [f32],
    right: &mut [f32],
    coef: f32,
    mem: [f32; 2],
) -> [f32; 2] {
    let mut m = mem;
    for ((&[l, r], dl), dr) in pairs.iter().zip(left.iter_mut()).zip(right.iter_mut()) {
        let x = l * 32768.0;
        *dl = x - m[0];
        m[0] = x * coef;
        let x = r * 32768.0;
        *dr = x - m[1];
        m[1] = x * coef;
    }
    m
}

/// `preemphasis_stereo_scalar` with 4 samples of each channel per pass.
///
/// `vld2q_f32` loads 4 pairs and splits them into the 4 left and the 4
/// right samples. Each output is `x[i] - coef * x[i - 1]` with the product
/// and the difference rounded on their own, as in the scalar loop, so the
/// result has the same bits.
#[cfg(target_arch = "aarch64")]
#[inline]
fn preemphasis_stereo_neon(
    pairs: &[[f32; 2]],
    left: &mut [f32],
    right: &mut [f32],
    coef: f32,
    mem: [f32; 2],
) -> [f32; 2] {
    const SCALE: f32 = 32768.0;
    let n = pairs.len();
    // The stores below rely on this. A caller that passes a shorter output
    // gets a panic, not a write past its end.
    assert!(
        left.len() >= n && right.len() >= n,
        "celt_preemphasis: output shorter than input"
    );
    if n == 0 {
        return mem;
    }
    left[0] = pairs[0][0] * SCALE - mem[0];
    right[0] = pairs[0][1] * SCALE - mem[1];
    let mut i = 1;
    // SAFETY: `1 <= i` and `i + 4 <= n` at every pass. A pass loads the
    // pairs `i - 1 .. i + 3` and `i .. i + 4`, 8 values each, and stores
    // `left[i .. i + 4]` and `right[i .. i + 4]`. All of these are inside
    // the slices, because `pairs.len() == n` and both outputs hold at
    // least `n` values.
    unsafe {
        let p = pairs.as_ptr() as *const f32;
        let (l, r) = (left.as_mut_ptr(), right.as_mut_ptr());
        let scale = vdupq_n_f32(SCALE);
        let coefv = vdupq_n_f32(coef);
        while i + 4 <= n {
            let x = vld2q_f32(p.add(2 * i));
            let prev = vld2q_f32(p.add(2 * (i - 1)));
            let (x_l, x_r) = (vmulq_f32(x.0, scale), vmulq_f32(x.1, scale));
            let (prev_l, prev_r) = (vmulq_f32(prev.0, scale), vmulq_f32(prev.1, scale));
            vst1q_f32(l.add(i), vsubq_f32(x_l, vmulq_f32(prev_l, coefv)));
            vst1q_f32(r.add(i), vsubq_f32(x_r, vmulq_f32(prev_r, coefv)));
            i += 4;
        }
    }
    // The filter memories after pair `i - 1`, then the last pairs.
    let [l, r] = pairs[i - 1];
    let mut m = [l * SCALE * coef, r * SCALE * coef];
    for ((&[l, r], dl), dr) in pairs[i..].iter().zip(&mut left[i..n]).zip(&mut right[i..n]) {
        let x = l * SCALE;
        *dl = x - m[0];
        m[0] = x * coef;
        let x = r * SCALE;
        *dr = x - m[1];
        m[1] = x * coef;
    }
    m
}

/// `preemphasis_block` for interleaved stereo with zeros stuffed after every
/// sample: pair `k` and its zeros give the `k`-th block of `left` and of
/// `right`.
#[inline(always)]
fn preemphasis_stereo_blocks<B: AsMut<[f32]>>(
    pairs: &[[f32; 2]],
    left: impl IntoIterator<Item = B>,
    right: impl IntoIterator<Item = B>,
    coef: f32,
    mem: [f32; 2],
) -> [f32; 2] {
    let mut m = mem;
    for ((&[l, r], mut bl), mut br) in pairs.iter().zip(left).zip(right) {
        m[0] = preemphasis_block(bl.as_mut(), l, coef, m[0]);
        m[1] = preemphasis_block(br.as_mut(), r, coef, m[1]);
    }
    m
}

/// Largest CELT-internal synthesis buffer. The CELT decoder writes `n + overlap`
/// MDCT samples per channel where `n ≤ frame_size ≤ 960` (20 ms @ 48 kHz); 2048
/// is the libopus reference value (`DECODE_BUFFER_SIZE`) and gives ample margin.
const DECODE_BUFFER_SIZE: usize = 2048;

/// libopus `inv_table` of `transient_analysis` (celt_encoder.c 288-297):
/// "Table of 6*64/x, trained on real data to minimize the average error".
/// Laid out 16 values per row, as in libopus.
#[rustfmt::skip]
const INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25,
     23,  22,  21,  20, 19, 18, 17, 16, 16, 15, 15, 14, 13, 13, 12, 12,
     12,  12,  11,  11, 11, 10, 10, 10,  9,  9,  9,  9,  9,  9,  8,  8,
      8,   8,   8,   7,  7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,  6,
      6,   6,   6,   6,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,
      5,   5,   5,   5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
      4,   4,   4,   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,
      3,   3,   3,   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,
];

/// Scratch of the tone and transient analyses, in two halves. Each half has
/// room for the longest frame (20 ms) and the overlap. The first half holds
/// the mono mix `tone_detect` analyses for stereo input. The second half
/// holds the envelopes of `transient_passes`: half as many values per
/// channel.
const CELT_ANALYSIS_TMP: usize = 2 * (CELT_ENC_MAX_FRAME + CELT_OVERLAP);

/// Filter state of one channel in the forward pass of `transient_passes`.
#[derive(Clone, Copy, Default)]
struct TransientForward {
    /// The two memories of the high-pass filter.
    mem0: f32,
    mem1: f32,
    /// Memory of the forward smoothing filter.
    smooth: f32,
    /// Sum of the energies so far.
    mean: f32,
}

impl TransientForward {
    /// One step of the forward pass: two input samples in, one value of the
    /// smoothed energy envelope out.
    ///
    /// The samples go through the high-pass filter. Their energy is added to
    /// `mean` and goes through the forward smoothing filter with the decay
    /// `forward_decay`; `keep` is `1 - forward_decay`. `blank` is set for
    /// the first 6 steps: libopus clears the first 12 filtered samples.
    ///
    /// In the float build of libopus, `PSHR32` is no shift at all, so the
    /// energy is not scaled. clang fuses the energy of the two samples and
    /// the smoothing into fused multiply-adds on 64-bit ARM (`mac`). The
    /// half of `mem1` in the high-pass filter is exact, so fusing it changes
    /// nothing.
    #[inline(always)]
    fn step(&mut self, xa: f32, xb: f32, blank: bool, forward_decay: f32, keep: f32) -> f32 {
        let mut ya = self.mem0 + xa;
        let mem00 = self.mem0;
        self.mem0 = self.mem0 - xa + 0.5 * self.mem1;
        self.mem1 = xa - mem00;

        let mut yb = self.mem0 + xb;
        let mem00 = self.mem0;
        self.mem0 = self.mem0 - xb + 0.5 * self.mem1;
        self.mem1 = xb - mem00;

        if blank {
            ya = 0.0;
            yb = 0.0;
        }
        // C: x2 = PSHR32(MULT16_16(a,a) + MULT16_16(b,b), 4), and
        // mean += PSHR32(x2, 12). Both shifts are identities in float.
        let x2 = mac(yb * yb, ya, ya);
        self.mean += x2;
        self.smooth = mac(x2, keep, self.smooth);
        forward_decay * self.smooth
    }
}

/// Number of forward steps whose filtered samples count as zero.
const TRANSIENT_BLANK_STEPS: usize = 6;

/// One step of the backward smoothing of `transient_passes`, in place.
/// `mem` is the filter memory and `max_e` the largest output so far.
#[inline(always)]
fn transient_backward_step(e: &mut f32, mem: &mut f32, max_e: &mut f32) {
    *mem = mac(*e, 0.875, *mem);
    *e = 0.125 * *mem;
    if *e > *max_e {
        *max_e = *e;
    }
}

/// What the two passes of the transient analysis leave for
/// `transient_decision`, per channel, next to the envelope.
#[derive(Clone, Copy, Default)]
struct TransientPasses {
    /// Sum of the energies of the high-passed samples.
    mean: [f32; 2],
    /// Largest value of the smoothed envelope.
    max_e: [f32; 2],
}

/// First part of libopus `transient_analysis` (float build): the smoothed
/// energy envelope of every channel. `transient_decision` is the second
/// part.
///
/// `input` holds `len` samples per channel, one channel after the other.
/// `envelope` gets `len / 2` values per channel. For stereo, `mix` gets the
/// `len` samples of `tone_mix_stereo`; mono leaves it alone. Neither buffer
/// needs to be initialized.
///
/// libopus runs three loops per channel: the high-pass filter, the energy
/// with forward smoothing, and the backward smoothing. Each is a recurrence,
/// so its speed is set by the delay of one add and one multiply per sample.
/// Here the first two loops are one loop, and both channels of a stereo
/// frame go through each loop side by side. The same loop also adds up the
/// sums `tone_detect` needs for a delay of 1 (see `ToneSums`), which are
/// three more chains of this kind. The processor then works on up to seven
/// chains at once. Every value is computed with the same operations in the
/// same order as in the separate loops.
fn transient_passes(
    input: &[f32],
    len: usize,
    channels: usize,
    allow_weak_transients: bool,
    envelope: &mut [f32],
    mix: &mut [f32],
) -> (TransientPasses, ToneSums) {
    let forward_decay = if allow_weak_transients {
        0.03125f32
    } else {
        0.0625f32
    };
    let keep = 1.0 - forward_decay;

    let len2 = len / 2;
    debug_assert!((1..=2).contains(&channels));
    let input = &input[..channels * len];
    let envelope = &mut envelope[..channels * len2];

    // Forward pass, then backward pass. `fwd[c].mean` and `max_e[c]` are
    // the results for channel `c`, next to its part of `envelope`.
    //
    // A step of the forward loop takes samples `2 * i` and `2 * i + 1`. The
    // sums for the tone detector need two samples more, so the loop runs on
    // windows of four samples, two samples apart. These end one step
    // early. `done` counts the steps of the loop; the code after it does
    // what is left.
    let mut fwd = [TransientForward::default(); 2];
    let mut max_e = [0.0f32; 2];
    let mut tone = ToneSums::default();
    let mut done = 0usize;
    if channels == 2 {
        let (in0, in1) = input.split_at(len);
        let (env0, env1) = envelope.split_at_mut(len2);
        let mix = &mut mix[..len];
        let [f0, f1] = &mut fwd;
        let windows = in0.windows(4).step_by(2).zip(in1.windows(4).step_by(2));
        let mix_pairs = mix.as_chunks_mut::<2>().0.iter_mut();
        let outputs = env0.iter_mut().zip(env1.iter_mut()).zip(mix_pairs);
        for (i, ((a, b), ((e0, e1), m))) in windows.zip(outputs).enumerate() {
            let x0 = 0.5 * (a[0] + b[0]);
            let x1 = 0.5 * (a[1] + b[1]);
            let x2 = 0.5 * (a[2] + b[2]);
            let x3 = 0.5 * (a[3] + b[3]);
            tone.add(x0, x1, x2);
            tone.add(x1, x2, x3);
            m[0] = x0;
            m[1] = x1;
            let blank = i < TRANSIENT_BLANK_STEPS;
            *e0 = f0.step(a[0], a[1], blank, forward_decay, keep);
            *e1 = f1.step(b[0], b[1], blank, forward_decay, keep);
            done = i + 1;
        }
        for i in done..len2 {
            let blank = i < TRANSIENT_BLANK_STEPS;
            env0[i] = f0.step(in0[2 * i], in0[2 * i + 1], blank, forward_decay, keep);
            env1[i] = f1.step(in1[2 * i], in1[2 * i + 1], blank, forward_decay, keep);
        }
        for i in 2 * done..len {
            mix[i] = 0.5 * (in0[i] + in1[i]);
        }
        for i in 2 * done..len.saturating_sub(2) {
            tone.add(mix[i], mix[i + 1], mix[i + 2]);
        }

        let mut mem = [0.0f32; 2];
        let [mem0, mem1] = &mut mem;
        let [max0, max1] = &mut max_e;
        for (e0, e1) in env0.iter_mut().rev().zip(env1.iter_mut().rev()) {
            transient_backward_step(e0, mem0, max0);
            transient_backward_step(e1, mem1, max1);
        }
    } else {
        let f0 = &mut fwd[0];
        for (i, (w, e0)) in input.windows(4).step_by(2).zip(envelope.iter_mut()).enumerate() {
            tone.add(w[0], w[1], w[2]);
            tone.add(w[1], w[2], w[3]);
            *e0 = f0.step(w[0], w[1], i < TRANSIENT_BLANK_STEPS, forward_decay, keep);
            done = i + 1;
        }
        for i in done..len2 {
            let blank = i < TRANSIENT_BLANK_STEPS;
            envelope[i] = f0.step(input[2 * i], input[2 * i + 1], blank, forward_decay, keep);
        }
        for i in 2 * done..len.saturating_sub(2) {
            tone.add(input[i], input[i + 1], input[i + 2]);
        }

        let mut mem = 0.0f32;
        for e0 in envelope.iter_mut().rev() {
            transient_backward_step(e0, &mut mem, &mut max_e[0]);
        }
    }

    let passes = TransientPasses {
        mean: [fwd[0].mean, fwd[1].mean],
        max_e,
    };
    (passes, tone)
}

/// Second part of libopus `transient_analysis` (float build): the decision.
///
/// `passes` and `envelope` come from `transient_passes`, called with the
/// same `len`, `channels` and `allow_weak_transients`.
#[allow(clippy::too_many_arguments)]
fn transient_decision(
    passes: &TransientPasses,
    envelope: &[f32],
    len: usize,
    channels: usize,
    tf_estimate: &mut f32,
    tf_chan: &mut usize,
    allow_weak_transients: bool,
    weak_transient: &mut bool,
    _tone_freq: f32,
    toneishness: f32,
) -> bool {
    let mut mask_metric = 0.0f32;
    *weak_transient = false;
    let len2 = len / 2;
    let envelope = &envelope[..channels * len2];

    for c in 0..channels {
        let tmp2 = &envelope[c * len2..(c + 1) * len2];
        let max_e = passes.max_e[c];
        let mut mean = passes.mean[c];

        // C: celt_sqrt(mean * maxE*.5*len2). The product of the two floats
        // is rounded to float. The constant .5 is a double, so the rest and
        // the square root are computed in double precision.
        mean = (f64::from(mean * max_e) * 0.5 * len2 as f64).sqrt() as f32;
        // C (float build): norm = len2 / (EPSILON + mean)  [EPSILON = 1e-15f]
        let norm = (len2 as f32) / (1e-15 + mean);

        // C computes the harmonic mean with INTEGER arithmetic (mask_metric and
        // unmask are opus_int32; the 64/6 scaling is an integer division).
        let mut unmask: i32 = 0;
        for i in (12..(len2 - 5)).step_by(4) {
            let id = (64.0 * norm * (tmp2[i] + 1e-15)).floor() as i32;
            let id = id.clamp(0, 127) as usize;
            unmask += INV_TABLE[id] as i32;
        }

        unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
        if (unmask as f32) > mask_metric {
            *tf_chan = c;
            mask_metric = unmask as f32;
        }
    }

    let mut is_transient = mask_metric > 200.0;

    if toneishness > 0.98 && _tone_freq < 0.026 {
        is_transient = false;
        mask_metric = 0.0;
    }
    // C 454-459: at low bitrates, a "weak transient" is coded with long
    // blocks and a finer time resolution instead, to avoid a partial collapse.
    if allow_weak_transients && is_transient && mask_metric < 600.0 {
        is_transient = false;
        *weak_transient = true;
    }

    // C (celt_encoder.c:460-461): tf_max = max(0, sqrt(27*mask_metric) - 42);
    // tf_estimate = sqrt(max(0, 0.0069*min(163, tf_max) - 0.139))  [float build]
    //
    // `MULT16_16` rounds the product to float. The constant 0.139
    // (`QCONST32` is no conversion in float) is a double, so the
    // subtraction and the square root are computed in double precision.
    let tf_max = ((27.0 * mask_metric).sqrt() - 42.0).max(0.0);
    let x = f64::from(0.0069f32 * tf_max.min(163.0)) - 0.139;
    *tf_estimate = (if x > 0.0 { x } else { 0.0 }).sqrt() as f32;

    is_transient
}

/// libopus `patch_transient_decision` (celt_encoder.c 470-509): whether
/// the energy went up so much since the frame before that the frame should
/// be coded as a transient after all.
///
/// The energies of the frame before (`old_e`) are first spread over
/// neighbouring bands with a slope of 6 dB per band. The decision is
/// whether the energies of this frame (`new_e`) are on average more than
/// 6 dB above that.
fn patch_transient_decision(
    new_e: &[f32],
    old_e: &[f32],
    nb_ebands: usize,
    start: usize,
    end: usize,
    channels: usize,
) -> bool {
    let first = start.max(2);
    if end < first + 2 {
        // libopus never sees so few bands here; the mean below would
        // divide by zero.
        return false;
    }
    let mut spread_old = [0.0f32; CELT_NB_EBANDS];
    // "Apply an aggressive (-6 dB/Bark) spreading function to the old frame
    // to avoid false detection caused by irrelevant bands"
    if channels == 1 {
        spread_old[start] = old_e[start];
        for i in start + 1..end {
            spread_old[i] = (spread_old[i - 1] - 1.0).max(old_e[i]);
        }
    } else {
        spread_old[start] = old_e[start].max(old_e[start + nb_ebands]);
        for i in start + 1..end {
            spread_old[i] = (spread_old[i - 1] - 1.0).max(old_e[i].max(old_e[i + nb_ebands]));
        }
    }
    for i in (start..end - 1).rev() {
        spread_old[i] = spread_old[i].max(spread_old[i + 1] - 1.0);
    }
    // "Compute mean increase"
    let mut mean_diff = 0.0f32;
    for c in 0..channels {
        for i in first..end - 1 {
            let x1 = new_e[i + c * nb_ebands].max(0.0);
            let x2 = spread_old[i].max(0.0);
            mean_diff += (x1 - x2).max(0.0);
        }
    }
    mean_diff /= (channels * (end - 1 - first)) as f32;
    mean_diff > 1.0
}

fn l1_metric(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    // The vector versions read `n` values through a pointer. This check
    // makes a call with `n > tmp.len()` panic instead.
    let tmp = &tmp[..n];
    // SAFETY: the AVX version runs only when the processor has AVX, and
    // `tmp` holds exactly `n` values (sliced above).
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if n >= 16 && crate::compat::x86_has_avx() {
            return l1_metric_avx(tmp, n, lm, bias);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= 16 {
            // SAFETY: `tmp` holds exactly `n` values (sliced above), and
            // NEON is part of every aarch64 target this crate builds for.
            return unsafe { l1_metric_neon(tmp, n, lm, bias) };
        }
    }

    let mut l1 = 0.0f32;
    for &tv in tmp.iter() {
        l1 += tv.abs();
    }
    l1 + (lm as f32) * bias * l1
}

/// Sum of the first `n` values of `|x[i]|`, in two groups of 8 partial sums.
///
/// # Safety
///
/// The processor must have AVX. Reads `x[..n]` through a pointer without
/// checks, so the caller must make sure that `n <= x.len()`. Every load
/// reads 8 values at `i .. i + 8`, and the loop conditions keep
/// `i + 8 <= n` (`i + 16 <= n` for the two loads of the first loop).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_abs_avx(x: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    // The loads below cover `x[..n]`.
    assert!(n <= x.len());

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let sign_mask = _mm256_set1_ps(-0.0);

    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        sum0 = _mm256_add_ps(sum0, _mm256_andnot_ps(sign_mask, v0));
        sum1 = _mm256_add_ps(sum1, _mm256_andnot_ps(sign_mask, v1));
        i += 16;
    }

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        sum0 = _mm256_add_ps(sum0, _mm256_andnot_ps(sign_mask, v));
        i += 8;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let s4 = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, t1);
    let t2 = _mm_shuffle_ps(s2, s2, 0x55);
    let mut out = _mm_cvtss_f32(_mm_add_ss(s2, t2));

    for j in i..n {
        out += x[j].abs();
    }

    out
}

/// # Safety
///
/// The processor must have AVX, and `n <= tmp.len()` (see `sum_abs_avx`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn l1_metric_avx(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    debug_assert!(n <= tmp.len());
    // SAFETY: the caller guarantees AVX and `n <= tmp.len()`.
    let l1 = sum_abs_avx(tmp, n);
    l1 + (lm as f32) * bias * l1
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_metric_neon(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    debug_assert!(n <= tmp.len());
    // SAFETY: the caller guarantees `n <= tmp.len()`. Every load reads the
    // 4 values at `i .. i + 4`, and the loop conditions keep `i + 4 <= n`.
    unsafe {
        let mut sum4 = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 15 < n {
            let v0 = vld1q_f32(tmp.as_ptr().add(i));
            let v1 = vld1q_f32(tmp.as_ptr().add(i + 4));
            let v2 = vld1q_f32(tmp.as_ptr().add(i + 8));
            let v3 = vld1q_f32(tmp.as_ptr().add(i + 12));

            sum4 = vaddq_f32(sum4, vabsq_f32(v0));
            sum4 = vaddq_f32(sum4, vabsq_f32(v1));
            sum4 = vaddq_f32(sum4, vabsq_f32(v2));
            sum4 = vaddq_f32(sum4, vabsq_f32(v3));

            i += 16;
        }

        while i + 3 < n {
            let v = vld1q_f32(tmp.as_ptr().add(i));
            sum4 = vaddq_f32(sum4, vabsq_f32(v));
            i += 4;
        }

        let sum2 = vpaddq_f32(sum4, sum4);
        let sum1 = vpaddq_f32(sum2, sum2);
        let mut l1 = vgetq_lane_f32(sum1, 0);

        while i < n {
            l1 += tmp[i].abs();
            i += 1;
        }

        l1 + (lm as f32) * bias * l1
    }
}

const MAX_NB_EBANDS: usize = 21;

const MAX_TF_TMP: usize = 176;

#[allow(clippy::too_many_arguments)]
fn tf_analysis(
    mode: &CeltMode,
    len: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[f32],
    n0: usize,
    lm: i32,
    tf_estimate: f32,
    tf_chan: usize,
    importance: &[i32],
) -> i32 {
    debug_assert!(len <= MAX_NB_EBANDS);
    let mut metric = [0i32; MAX_NB_EBANDS];
    let mut tmp = [0.0f32; MAX_TF_TMP];
    let mut tmp_1 = [0.0f32; MAX_TF_TMP];

    let bias = 0.04 * (-0.25f32).max(0.5 - tf_estimate);

    for (i, metric_i) in metric[..len].iter_mut().enumerate() {
        let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
        let narrow = (mode.e_bands[i + 1] - mode.e_bands[i]) == 1;
        let offset = tf_chan * n0 + ((mode.e_bands[i] as usize) << lm);
        tmp[..n].copy_from_slice(&x[offset..offset + n]);

        let mut l1 = l1_metric(&tmp[..n], n, if is_transient { lm } else { 0 }, bias);
        let mut best_l1 = l1;
        let mut best_level = 0;

        if is_transient && !narrow {
            tmp_1[..n].copy_from_slice(&tmp[..n]);
            haar1(&mut tmp_1[..n], n >> lm, 1 << lm);
            l1 = l1_metric(&tmp_1[..n], n, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }

        for k in 0..(lm + if is_transient || narrow { 0 } else { 1 }) {
            let b = if is_transient { lm - k - 1 } else { k + 1 };

            haar1(&mut tmp[..n], n >> k, 1 << k);
            l1 = l1_metric(&tmp[..n], n, b, bias);

            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k + 1;
            }
        }

        if is_transient {
            *metric_i = 2 * best_level;
        } else {
            *metric_i = -2 * best_level;
        }

        if narrow && (*metric_i == 0 || *metric_i == -2 * lm) {
            *metric_i -= 1;
        }
    }

    let mut tf_select = 0;
    let mut selcost = [0i32; 2];

    for sel in 0..2 {
        let mut cost0 = importance[0]
            * (metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel] as i32)
                .abs();
        let mut cost1 = importance[0]
            * (metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1]
                    as i32)
                .abs()
            + if is_transient { 0 } else { lambda };

        for i in 1..len {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0
                + importance[i]
                    * (metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel]
                            as i32)
                        .abs();
            cost1 = curr1
                + importance[i]
                    * (metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize]
                            [4 * (is_transient as usize) + 2 * sel + 1]
                            as i32)
                        .abs();
        }
        selcost[sel] = cost0.min(cost1);
    }

    // C: only allow tf_select=1 for transients (celt_encoder.c:787)
    if selcost[1] < selcost[0] && is_transient {
        tf_select = 1;
    }

    // Viterbi forward pass with path storage, then backward pass to
    // reconstruct the optimal tf_res (C celt_encoder.c:789-822).
    let sel = tf_select as usize;
    let mut cost0 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel] as i32)
            .abs();
    let mut cost1 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1] as i32)
            .abs()
        + if is_transient { 0 } else { lambda };

    let mut path0 = [0i32; MAX_NB_EBANDS];
    let mut path1 = [0i32; MAX_NB_EBANDS];
    for i in 1..len {
        let from0 = cost0;
        let from1 = cost1 + lambda;
        let curr0 = if from0 < from1 {
            path0[i] = 0;
            from0
        } else {
            path0[i] = 1;
            from1
        };

        let from0 = cost0 + lambda;
        let from1 = cost1;
        let curr1 = if from0 < from1 {
            path1[i] = 0;
            from0
        } else {
            path1[i] = 1;
            from1
        };

        cost0 = curr0
            + importance[i]
                * (metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel]
                        as i32)
                    .abs();
        cost1 = curr1
            + importance[i]
                * (metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1]
                        as i32)
                    .abs();
    }
    tf_res[len - 1] = if cost0 < cost1 { 0 } else { 1 };
    for i in (0..len - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }

    tf_select
}

fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    mut tf_select: i32,
    rc: &mut RangeCoder,
) -> i32 {
    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let mut budget = rc.storage as i32 * 8;
    let mut tell = rc.tell();

    let tf_select_rsv = if lm > 0 && tell + logp < budget { 1 } else { 0 };
    budget -= tf_select_rsv;

    for tf_res_i in tf_res[start..end].iter_mut() {
        if tell + logp <= budget {
            rc.encode_bit_logp(*tf_res_i ^ curr != 0, logp as u32);
            tell = rc.tell();
            curr = *tf_res_i;
            tf_changed |= curr;
        } else {
            *tf_res_i = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    if tf_select_rsv != 0
        && TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + (tf_changed as usize)]
            != TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 + (tf_changed as usize)]
    {
        rc.encode_bit_logp(tf_select != 0, 1);
    } else {
        tf_select = 0;
    }

    for tf_res_i in tf_res[start..end].iter_mut() {
        *tf_res_i = TF_SELECT_TABLE[lm as usize]
            [4 * (is_transient as usize) + 2 * (tf_select as usize) + (*tf_res_i as usize)]
            as i32;
    }

    tf_changed
}

fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    rc: &mut RangeCoder,
) {
    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let budget = rc.storage as i32 * 8;
    let mut tell = rc.tell();

    let tf_select_rsv = if lm > 0 && tell + logp < budget { 1 } else { 0 };
    let budget = budget - tf_select_rsv;

    for tf_res_i in tf_res[start..end].iter_mut() {
        if tell + logp <= budget {
            curr ^= if rc.decode_bit_logp(logp as u32) {
                1
            } else {
                0
            };
            tell = rc.tell();
            tf_changed |= curr;
        }
        *tf_res_i = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let mut tf_select = 0;
    let _budget = budget + tf_select_rsv;
    if tf_select_rsv > 0
        && TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + (tf_changed as usize)]
            != TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 + (tf_changed as usize)]
    {
        tf_select = if rc.decode_bit_logp(1) { 1 } else { 0 };
    }

    for tf_res_i in tf_res[start..end].iter_mut() {
        *tf_res_i = TF_SELECT_TABLE[lm as usize]
            [4 * (is_transient as usize) + 2 * (tf_select as usize) + (*tf_res_i as usize)]
            as i32;
    }
}

/// libopus `stereo_analysis`: whether coding left and right separately
/// takes fewer bits than coding mid and side. The L1 norm of each pair
/// stands for its entropy.
fn stereo_analysis(m: &CeltMode, x: &[f32], lm: i32, n0: usize) -> bool {
    // C `EPSILON` of the float build.
    let mut sum_lr = 1e-15f32;
    let mut sum_ms = 1e-15f32;

    for i in 0..13 {
        let start = (m.e_bands[i] as usize) << lm;
        let end = (m.e_bands[i + 1] as usize) << lm;
        for j in start..end {
            let l = x[j];
            let r = x[n0 + j];
            let m_val = l + r;
            let s_val = l - r;
            sum_lr += l.abs() + r.abs();
            sum_ms += m_val.abs() + s_val.abs();
        }
    }

    // The constant as libopus writes it.
    #[allow(clippy::approx_constant)]
    {
        sum_ms *= 0.707107;
    }
    let mut thetas = 13;
    if lm <= 1 {
        thetas -= 8;
    }

    let left = (((m.e_bands[13] as usize) << (lm + 1)) + thetas) as f32 * sum_ms;
    let right = ((m.e_bands[13] as usize) << (lm + 1)) as f32 * sum_lr;

    left > right
}

const COMBFILTER_MINPERIOD: usize = 15;
const COMBFILTER_MAXPERIOD: usize = 1024;

/// The taps of the three pitch filters (libopus `comb_filter`, `gains`).
///
/// libopus writes them as `QCONST16(0.3066406250f, 15)` and so on, which
/// in the float build is the float nearest to that decimal. Each one is a
/// multiple of 2^-12 and so an exact `f32`. Shorter decimals such as
/// 0.3066406 round to another `f32` in four of the seven cases.
// The digits are libopus's: each literal is its exact value.
#[allow(clippy::excessive_precision)]
const PREFILTER_GAINS: [[f32; 3]; 3] = [
    [0.306_640_625, 0.217_041_015_625, 0.129_638_671_875],
    [0.463_867_187_5, 0.268_066_406_25, 0.0],
    [0.799_804_687_5, 0.100_097_656_25, 0.0],
];

/// One output of libopus `comb_filter_const_c` (celt/celt.c):
/// `x + g[0] * centre + g[1] * near + g[2] * far`, where `near` and `far` are
/// the sums of the two samples one and two places from the centre tap. The
/// products are added in that order, each with `mac`: clang fuses each of
/// them into its addition on 64-bit ARM.
#[inline(always)]
fn comb_tap3(x: f32, g: [f32; 3], centre: f32, near: f32, far: f32) -> f32 {
    mac(mac(mac(x, g[0], centre), g[1], near), g[2], far)
}

/// One output of the cross-fade at the start of libopus `comb_filter`:
/// `x` plus the three taps of the old filter, each weighted with `1 - f`,
/// plus the three taps of the new one, each weighted with `f`. The weights
/// are multiplied into the gains first (C `MULT_COEF((COEF_ONE-f),g00)`),
/// and the six products are added in that order, each with `mac`.
#[inline(always)]
fn comb_fade_tap6(x: f32, f: f32, old: [f32; 3], old_taps: [f32; 3], new: [f32; 3], new_taps: [f32; 3]) -> f32 {
    let one_minus_f = 1.0 - f;
    let mut s = x;
    for k in 0..3 {
        s = mac(s, one_minus_f * old[k], old_taps[k]);
    }
    for k in 0..3 {
        s = mac(s, f * new[k], new_taps[k]);
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn comb_filter_const(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    // The vector versions below read and write through pointers without
    // further checks. They read `x[x_idx - t - 2 .. x_idx + n]` and write
    // `y[y_idx .. y_idx + n]`. A caller with other values gets a panic here.
    assert!(
        t >= 2 && x_idx >= t + 2 && x_idx + n <= x.len() && y_idx + n <= y.len(),
        "comb_filter_const: range outside its buffers"
    );
    #[cfg(target_arch = "aarch64")]
    {
        comb_filter_const_neon(y, x, y_idx, x_idx, t, n, g10, g11, g12);
    }
    // SAFETY: the range was checked above, and this version runs only when
    // the processor has AVX and FMA.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if crate::compat::x86_has_avx_fma() {
            comb_filter_const_avx(y, x, y_idx, x_idx, t, n, g10, g11, g12);
            return;
        }
    }
    // SAFETY: the range was checked above, and SSE is part of this target.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        comb_filter_const_sse(y, x, y_idx, x_idx, t, n, g10, g11, g12);
        #[allow(clippy::needless_return)]
        return;
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    {
        comb_filter_const_scalar(y, x, y_idx, x_idx, t, n, g10, g11, g12);
    }
}

#[inline]
#[allow(dead_code)]
fn comb_filter_const_scalar(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    let mut x1;
    let mut x2;
    let mut x3;
    let mut x4;
    let mut x0;

    x4 = x[x_idx - t - 2];
    x3 = x[x_idx - t - 1];
    x2 = x[x_idx - t];
    x1 = x[x_idx - t + 1];

    for i in 0..n {
        x0 = x[x_idx + i - t + 2];
        y[y_idx + i] = comb_tap3(x[x_idx + i], [g10, g11, g12], x2, x1 + x3, x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

#[cfg(target_arch = "aarch64")]
fn comb_filter_const_neon(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    // The only caller is `comb_filter_const`, which has checked this range.
    // The check is repeated here so that this safe function stays safe if
    // it gets another caller.
    assert!(t >= 2 && x_idx >= t + 2 && x_idx + n <= x.len() && y_idx + n <= y.len());
    // SAFETY: the range the kernel reads and writes is inside `x` and `y`
    // (checked above).
    unsafe { comb_filter_const_neon_impl(y, x, y_idx, x_idx, t, n, g10, g11, g12) }
}

/// `comb_filter_const` for 64-bit ARM, four samples per pass.
///
/// Every sample adds each product with a fused multiply-add, in the order
/// of `comb_tap3`, as libopus's arm64 build does. The result has the bits
/// of `comb_filter_const_scalar`.
///
/// # Safety
///
/// Reads `x[x_idx - t - 2 .. x_idx + n]` and writes `y[y_idx .. y_idx + n]`
/// without checks. The caller must make sure that `t >= 2` and that both
/// ranges are inside their slices. With `t >= 2`, the load of the four
/// values at `x_idx + i - t + 2` ends at or before `x_idx + i + 4`, which
/// the loop condition keeps at or below `x_idx + n`. The first load, at
/// `x_idx - t - 2`, also happens when `n` is 0; it ends at or before
/// `x_idx`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_neon_impl(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    use core::arch::aarch64::*;
    debug_assert!(t >= 2 && x_idx >= t + 2 && x_idx + n <= x.len() && y_idx + n <= y.len());

    let g10v = vdupq_n_f32(g10);
    let g11v = vdupq_n_f32(g11);
    let g12v = vdupq_n_f32(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);

    let mut x0v = vld1q_f32(xbase.sub(t + 2));

    let mut i = 0;
    while i + 4 <= n {
        let x4v = vld1q_f32(xbase.add(i).sub(t - 2));

        let x2v = vextq_f32(x0v, x4v, 2);

        let x1v = vextq_f32(x0v, x4v, 1);

        let x3v = vextq_f32(x0v, x4v, 3);

        let xi = vld1q_f32(xbase.add(i));

        let mut yi = xi;
        yi = vfmaq_f32(yi, g10v, x2v);
        yi = vfmaq_f32(yi, g11v, vaddq_f32(x1v, x3v));
        yi = vfmaq_f32(yi, g12v, vaddq_f32(x4v, x0v));
        vst1q_f32(ybase.add(i), yi);

        x0v = x4v;
        i += 4;
    }

    let x0v_arr: [f32; 4] = core::mem::transmute(x0v);
    let mut sx4 = x0v_arr[0];
    let mut sx3 = x0v_arr[1];
    let mut sx2 = x0v_arr[2];
    let mut sx1 = x0v_arr[3];

    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = comb_tap3(x[x_idx + i], [g10, g11, g12], sx2, sx1 + sx3, sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

/// # Safety
///
/// Reads `x[x_idx - t - 2 .. x_idx + n]` and writes `y[y_idx .. y_idx + n]`
/// without checks. The caller must make sure that `t >= 2` and that both
/// ranges are inside their slices. With `t >= 2`, the load of the four
/// values at `x_idx + i - t + 2` ends at or before `x_idx + i + 4`, which
/// the loop condition keeps at or below `x_idx + n`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_sse(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    assert!(t >= 2 && x_idx >= t + 2 && x_idx + n <= x.len() && y_idx + n <= y.len());

    let g10v = _mm_set1_ps(g10);
    let g11v = _mm_set1_ps(g11);
    let g12v = _mm_set1_ps(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);
    let mut x0v = _mm_loadu_ps(xbase.sub(t + 2));

    let mut i = 0;
    while i + 4 <= n {
        let x4v = _mm_loadu_ps(xbase.add(i).sub(t - 2));

        let x2v = _mm_shuffle_ps(x0v, x4v, 0x4e);

        let x1v = _mm_shuffle_ps(x0v, x2v, 0x99);

        let x3v = _mm_shuffle_ps(x2v, x4v, 0x99);

        let xi = _mm_loadu_ps(xbase.add(i));

        let mut yi = xi;
        yi = _mm_add_ps(yi, _mm_mul_ps(g10v, x2v));
        let yi2 = _mm_add_ps(
            _mm_mul_ps(g11v, _mm_add_ps(x3v, x1v)),
            _mm_mul_ps(g12v, _mm_add_ps(x4v, x0v)),
        );
        yi = _mm_add_ps(yi, yi2);
        _mm_storeu_ps(ybase.add(i), yi);

        x0v = x4v;
        i += 4;
    }

    let x0v_arr: [f32; 4] = core::mem::transmute(x0v);
    let mut sx4 = x0v_arr[0];
    let mut sx3 = x0v_arr[1];
    let mut sx2 = x0v_arr[2];
    let mut sx1 = x0v_arr[3];

    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

/// # Safety
///
/// The processor must have AVX and FMA. Reads
/// `x[x_idx - t - 2 .. x_idx + n]` and writes `y[y_idx .. y_idx + n]`
/// without checks. The caller must make sure that `t >= 2` and that both
/// ranges are inside their slices. With `t >= 2`, the load of the eight
/// values at `x_idx + i - t + 2` ends at or before `x_idx + i + 8`, which
/// the loop conditions keep at or below `x_idx + n`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_avx(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    assert!(t >= 2 && x_idx >= t + 2 && x_idx + n <= x.len() && y_idx + n <= y.len());

    let g10v = _mm256_set1_ps(g10);
    let g11v = _mm256_set1_ps(g11);
    let g12v = _mm256_set1_ps(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);

    let mut i = 0;

    while i + 16 <= n {
        let xi_a = _mm256_loadu_ps(xbase.add(i));
        let x0_a = _mm256_loadu_ps(xbase.add(i).sub(t + 2));
        let x4_a = _mm256_loadu_ps(xbase.add(i).sub(t - 2));

        let x2_a = _mm256_loadu_ps(xbase.add(i).sub(t));
        let x1x3_a = _mm256_add_ps(
            _mm256_loadu_ps(xbase.add(i).sub(t + 1)),
            _mm256_loadu_ps(xbase.add(i).sub(t - 1)),
        );
        let x0x4_a = _mm256_add_ps(x0_a, x4_a);

        let mut yi_a = xi_a;
        yi_a = _mm256_fmadd_ps(g10v, x2_a, yi_a);
        yi_a = _mm256_fmadd_ps(g11v, x1x3_a, yi_a);
        yi_a = _mm256_fmadd_ps(g12v, x0x4_a, yi_a);
        _mm256_storeu_ps(ybase.add(i), yi_a);

        let j = i + 8;
        let xi_b = _mm256_loadu_ps(xbase.add(j));
        let x0_b = _mm256_loadu_ps(xbase.add(j).sub(t + 2));
        let x4_b = _mm256_loadu_ps(xbase.add(j).sub(t - 2));
        let x2_b = _mm256_loadu_ps(xbase.add(j).sub(t));
        let x1x3_b = _mm256_add_ps(
            _mm256_loadu_ps(xbase.add(j).sub(t + 1)),
            _mm256_loadu_ps(xbase.add(j).sub(t - 1)),
        );
        let x0x4_b = _mm256_add_ps(x0_b, x4_b);

        let mut yi_b = xi_b;
        yi_b = _mm256_fmadd_ps(g10v, x2_b, yi_b);
        yi_b = _mm256_fmadd_ps(g11v, x1x3_b, yi_b);
        yi_b = _mm256_fmadd_ps(g12v, x0x4_b, yi_b);
        _mm256_storeu_ps(ybase.add(j), yi_b);

        i += 16;
    }

    while i + 8 <= n {
        let xi = _mm256_loadu_ps(xbase.add(i));
        let x0 = _mm256_loadu_ps(xbase.add(i).sub(t + 2));
        let x4 = _mm256_loadu_ps(xbase.add(i).sub(t - 2));
        let x2 = _mm256_loadu_ps(xbase.add(i).sub(t));
        let x1x3 = _mm256_add_ps(
            _mm256_loadu_ps(xbase.add(i).sub(t + 1)),
            _mm256_loadu_ps(xbase.add(i).sub(t - 1)),
        );
        let x0x4 = _mm256_add_ps(x0, x4);

        let mut yi = xi;
        yi = _mm256_fmadd_ps(g10v, x2, yi);
        yi = _mm256_fmadd_ps(g11v, x1x3, yi);
        yi = _mm256_fmadd_ps(g12v, x0x4, yi);
        _mm256_storeu_ps(ybase.add(i), yi);

        i += 8;
    }

    if i + 4 <= n {
        // SAFETY: the same ranges as this call's, without the first `i`
        // samples: `x_idx + i >= t + 2`, and both ends stay where they were.
        comb_filter_const_sse_fma(y, x, y_idx + i, x_idx + i, t, n - i, g10, g11, g12);
        return;
    }

    let mut sx4 = x[x_idx + i - t - 2];
    let mut sx3 = x[x_idx + i - t - 1];
    let mut sx2 = x[x_idx + i - t];
    let mut sx1 = x[x_idx + i - t + 1];
    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

/// # Safety
///
/// The processor must have AVX and FMA. The ranges are those of
/// `comb_filter_const_sse`, with the same conditions for the caller.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_sse_fma(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    assert!(t >= 2 && x_idx >= t + 2 && x_idx + n <= x.len() && y_idx + n <= y.len());

    let g10v = _mm_set1_ps(g10);
    let g11v = _mm_set1_ps(g11);
    let g12v = _mm_set1_ps(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);
    let mut x0v = _mm_loadu_ps(xbase.sub(t + 2));

    let mut i = 0;
    while i + 4 <= n {
        let x4v = _mm_loadu_ps(xbase.add(i).sub(t - 2));
        let x2v = _mm_shuffle_ps(x0v, x4v, 0x4e);
        let x1v = _mm_shuffle_ps(x0v, x2v, 0x99);
        let x3v = _mm_shuffle_ps(x2v, x4v, 0x99);
        let xi = _mm_loadu_ps(xbase.add(i));

        let mut yi = xi;
        yi = _mm_fmadd_ps(g10v, x2v, yi);
        yi = _mm_fmadd_ps(g11v, _mm_add_ps(x1v, x3v), yi);
        yi = _mm_fmadd_ps(g12v, _mm_add_ps(x0v, x4v), yi);
        _mm_storeu_ps(ybase.add(i), yi);

        x0v = x4v;
        i += 4;
    }

    let x0v_arr: [f32; 4] = core::mem::transmute(x0v);
    let mut sx4 = x0v_arr[0];
    let mut sx3 = x0v_arr[1];
    let mut sx2 = x0v_arr[2];
    let mut sx1 = x0v_arr[3];
    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn comb_filter(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        if x_idx != y_idx || !core::ptr::eq(x.as_ptr(), y.as_ptr()) {
            y[y_idx..y_idx + n].copy_from_slice(&x[x_idx..x_idx + n]);
        }
        return;
    }

    let t0 = t0.clamp(
        COMBFILTER_MINPERIOD,
        x_idx.saturating_sub(2).max(COMBFILTER_MINPERIOD),
    );
    let t1 = t1.clamp(
        COMBFILTER_MINPERIOD,
        x_idx.saturating_sub(2).max(COMBFILTER_MINPERIOD),
    );

    let g00 = g0 * PREFILTER_GAINS[tapset0 as usize][0];
    let g01 = g0 * PREFILTER_GAINS[tapset0 as usize][1];
    let g02 = g0 * PREFILTER_GAINS[tapset0 as usize][2];

    let g10 = g1 * PREFILTER_GAINS[tapset1 as usize][0];
    let g11 = g1 * PREFILTER_GAINS[tapset1 as usize][1];
    let g12 = g1 * PREFILTER_GAINS[tapset1 as usize][2];

    let mut x1 = x[x_idx - t1 + 1];
    let mut x2 = x[x_idx - t1];
    let mut x3 = x[x_idx - t1 - 1];
    let mut x4 = x[x_idx - t1 - 2];

    let mut inner_overlap = overlap;
    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        inner_overlap = 0;
    }

    let mut i = 0;
    while i < inner_overlap && i < n {
        let x0 = x[x_idx + i - t1 + 2];
        let f = window[i] * window[i];
        let at = x_idx + i - t0;
        let old_taps = [x[at], x[at + 1] + x[at - 1], x[at + 2] + x[at - 2]];
        y[y_idx + i] = comb_fade_tap6(
            x[x_idx + i],
            f,
            [g00, g01, g02],
            old_taps,
            [g10, g11, g12],
            [x2, x1 + x3, x0 + x4],
        );

        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }

    if i < n {
        if g1 == 0.0 {
            y[y_idx + i..y_idx + n].copy_from_slice(&x[x_idx + i..x_idx + n]);
        } else {
            comb_filter_const(y, x, y_idx + i, x_idx + i, t1, n - i, g10, g11, g12);
        }
    }
}

/// In-place comb filter over the frames of `C` channels that share one
/// filter: `bufs[c][y_idx..y_idx + n]` is both the input and the output of
/// channel `c`. The samples the filter reads at `y_idx + i - t` and around
/// it may already be filtered, as in libopus, which calls
/// `comb_filter(out, out, ...)` once per channel.
#[allow(clippy::too_many_arguments)]
fn comb_filter_inplace<const C: usize>(
    bufs: &mut [&mut [f32]; C],
    y_idx: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        // nothing to do; the buffers already hold the input
        return;
    }

    let t0 = t0.clamp(COMBFILTER_MINPERIOD, y_idx - 2);
    let t1 = t1.clamp(COMBFILTER_MINPERIOD, y_idx - 2);

    let g00 = g0 * PREFILTER_GAINS[tapset0 as usize][0];
    let g01 = g0 * PREFILTER_GAINS[tapset0 as usize][1];
    let g02 = g0 * PREFILTER_GAINS[tapset0 as usize][2];

    let g10 = g1 * PREFILTER_GAINS[tapset1 as usize][0];
    let g11 = g1 * PREFILTER_GAINS[tapset1 as usize][1];
    let g12 = g1 * PREFILTER_GAINS[tapset1 as usize][2];

    // The first `overlap` samples cross-fade from the old filter to the new
    // one, unless the two are the same filter.
    let fade = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap.min(n)
    };
    if fade > 0 {
        let (old, new) = ([g00, g01, g02], [g10, g11, g12]);
        comb_inplace_fade(bufs, y_idx, t0, t1, fade, old, new, window);
    }
    // The rest of the frame uses the new filter alone. libopus leaves it
    // as it is when the new gain is 0.
    if fade < n && g1 != 0.0 {
        comb_inplace_const(bufs, y_idx + fade, t1, n - fade, [g10, g11, g12]);
    }
}

/// libopus `run_prefilter` (celt/celt_encoder.c): finds the pitch period
/// and gain of the comb filter for this frame and filters `in_buf` with it.
///
/// `pre` is `CeltEncoder::prefilter_buf`: per channel, `PREFILTER_STRIDE`
/// values, of which the first `COMBFILTER_MAXPERIOD` hold the history of
/// the unfiltered input. The frame is put after the history, and at the end
/// the history moves up by `frame_size`.
// 3.1416/3.141593 literals mirror celt_encoder.c; keep them verbatim.
#[allow(clippy::approx_constant)]
fn run_prefilter(
    in_buf: &mut [f32],
    pre: &mut [f32],
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: i32,
    tapset_decision: i32,
    window: &[f32],
    channels: usize,
    frame_size: usize,
    overlap: usize,

    pitch_buf: &mut [f32],
    before: &mut [f32],
    after: &mut [f32],

    loss_rate: i32,
    tf_estimate: f32,
    nb_available_bytes: usize,
    tone_freq: f32,
    toneishness: f32,
    complexity: i32,
    // C `enabled`: when false, no new filter is searched for. The gain is
    // zero and only the previous frame's filter is faded out.
    enabled: bool,
    // C `analysis->max_pitch_ratio` when the analysis is valid: the share
    // of the energy below 6 kHz, which scales the gain.
    max_pitch_ratio: Option<f32>,
) -> (bool, f32, usize) {
    let max_period = COMBFILTER_MAXPERIOD;
    let min_period = COMBFILTER_MINPERIOD;
    let buf_stride = frame_size + overlap;
    // C `pre[c]`: the history and then the frame, at `c * PREFILTER_STRIDE`.
    let pre_size = max_period + frame_size;
    assert!(pre_size <= PREFILTER_STRIDE && pre.len() >= channels * PREFILTER_STRIDE);

    for c in 0..channels {
        pre[c * PREFILTER_STRIDE + max_period..c * PREFILTER_STRIDE + pre_size].copy_from_slice(
            &in_buf[c * buf_stride + overlap..c * buf_stride + overlap + frame_size],
        );
    }

    let mut pitch_index: usize;
    let mut gain1: f32;
    if enabled && toneishness > 0.99 {
        // Tone path (C 1449-1473): bypass the pitch search for pure tones.
        let mut multiple = 1.0f32;
        let mut tone_freq = tone_freq;
        if tone_freq >= 3.1416 {
            tone_freq = 3.141593 - tone_freq;
        }
        while tone_freq >= multiple * 0.39 {
            multiple += 1.0;
        }
        if tone_freq > 0.006148 {
            // C `floor(.5+2.f*M_PI*multiple/tone_freq)`: in double precision.
            let period = 0.5 + 2.0 * core::f64::consts::PI * multiple as f64 / tone_freq as f64;
            pitch_index = (period as usize).min(COMBFILTER_MAXPERIOD - 2);
        } else {
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = 0.75;
    } else if enabled && complexity >= 5 {
        let pitch_buf_len = (max_period + frame_size) >> 1;
        {
            let mut pre_slices: FixedVec<&[f32], 2> = FixedVec::new();
            for c in 0..channels {
                pre_slices.push(&pre[c * PREFILTER_STRIDE..c * PREFILTER_STRIDE + pre_size]);
            }
            crate::pitch::pitch_downsample(&pre_slices, pitch_buf, pitch_buf_len, channels, 2);
        }

        // C `pitch_search` over `max_period - 3 * min_period` lags, then
        // `remove_doubling`.
        (pitch_index, gain1) = crate::pitch::prefilter_pitch(
            pitch_buf,
            max_period,
            min_period,
            frame_size,
            prefilter_period,
            prefilter_gain,
        );
        if pitch_index > max_period - 2 {
            pitch_index = max_period - 2;
        }
        gain1 *= 0.7;
        // C: halve at >2% loss, halve again at >4%, zero at >8%
        if loss_rate > 2 {
            gain1 *= 0.5;
        }
        if loss_rate > 4 {
            gain1 *= 0.5;
        }
        if loss_rate > 8 {
            gain1 = 0.0;
        }
    } else {
        gain1 = 0.0;
        pitch_index = COMBFILTER_MINPERIOD;
    }
    // C 1492-1495: the analysis lowers the gain of a signal with much
    // energy above 6 kHz, where the comb filter does more harm than good.
    if let Some(ratio) = max_pitch_ratio {
        gain1 *= ratio;
    }

    let mut pf_threshold = 0.2f32;
    if (pitch_index as i32 - prefilter_period as i32).unsigned_abs() as usize * 10 > pitch_index {
        pf_threshold += 0.2;
        // Completely disable the prefilter on strong transients without continuity.
        if tf_estimate > 0.98 {
            gain1 = 0.0;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += 0.1;
    }
    if nb_available_bytes < 35 {
        pf_threshold += 0.1;
    }
    if prefilter_gain > 0.4 {
        pf_threshold -= 0.1;
    }
    if prefilter_gain > 0.55 {
        pf_threshold -= 0.1;
    }
    pf_threshold = pf_threshold.max(0.2);

    let pf_on;
    if gain1 < pf_threshold {
        gain1 = 0.0;
        pf_on = false;
    } else {
        if (gain1 - prefilter_gain).abs() < 0.1 {
            gain1 = prefilter_gain;
        }
        let qg = ((gain1 * 32.0 / 3.0 + 0.5).floor() as i32 - 1).clamp(0, 7);
        gain1 = 0.09375 * (qg + 1) as f32;
        pf_on = true;
    }

    let before = &mut before[..channels];
    for c in 0..channels {
        let start = c * buf_stride + overlap;
        before[c] = sum_abs(&in_buf[start..start + frame_size]);
    }

    let offset = 0usize;
    let prev_period = prefilter_period.clamp(COMBFILTER_MINPERIOD, max_period - 2);

    for c in 0..channels {
        if offset > 0 {
            let pre_c = &pre[c * PREFILTER_STRIDE..];
            comb_filter(
                in_buf,
                pre_c,
                c * buf_stride + overlap,
                max_period,
                prev_period,
                prev_period,
                offset,
                -prefilter_gain,
                -prefilter_gain,
                prefilter_tapset,
                prefilter_tapset,
                window,
                0,
            );
        }

        {
            let pre_c = &pre[c * PREFILTER_STRIDE..];
            comb_filter(
                in_buf,
                pre_c,
                c * buf_stride + overlap + offset,
                max_period + offset,
                prev_period,
                pitch_index,
                frame_size - offset,
                -prefilter_gain,
                -gain1,
                prefilter_tapset,
                tapset_decision,
                window,
                overlap,
            );
        }
    }

    let after = &mut after[..channels];
    for c in 0..channels {
        let start = c * buf_stride + overlap;
        after[c] = sum_abs(&in_buf[start..start + frame_size]);
    }

    // C 1557-1569: whether the filter is worth it.
    let cancel_pitch = if channels == 2 {
        // C `MULT16_32_Q15(MULT16_16_Q15(.25, gain1), before[0]) +
        // MULT16_32_Q15(.01, before[1])`: clang fuses the first product into
        // the addition.
        let quarter_gain = 0.25 * gain1;
        let thresh = [
            mac(0.01 * before[1], quarter_gain, before[0]),
            mac(0.01 * before[0], quarter_gain, before[1]),
        ];
        // "Don't use the filter if one channel gets significantly worse."
        let worse = after[0] - before[0] > thresh[0] || after[1] - before[1] > thresh[1];
        // "Use the filter only if at least one channel gets significantly
        // better."
        let no_gain = before[0] - after[0] < thresh[0] && before[1] - after[1] < thresh[1];
        worse || no_gain
    } else {
        // "Check that the mono channel actually got better."
        after[0] > before[0]
    };

    // C 1572-1584: revert to a gain of zero, but still fade the previous
    // frame's filter out over the overlap. The decoder's post-filter always
    // cross-fades from the old gain, so dropping the fade here leaves 120
    // samples it post-filters that were never pre-filtered.
    if cancel_pitch {
        for c in 0..channels {
            let body = c * buf_stride + overlap;
            in_buf[body..body + frame_size].copy_from_slice(
                &pre[c * PREFILTER_STRIDE + max_period..c * PREFILTER_STRIDE + pre_size],
            );
            comb_filter(
                in_buf,
                &pre[c * PREFILTER_STRIDE..],
                body + offset,
                max_period + offset,
                prev_period,
                pitch_index,
                overlap,
                -prefilter_gain,
                0.0,
                prefilter_tapset,
                tapset_decision,
                window,
                overlap,
            );
        }
    }

    // C: `OPUS_COPY(st->prefilter_mem+c*COMBFILTER_MAXPERIOD, pre[c]+N,
    // COMBFILTER_MAXPERIOD)`: the last `max_period` values of history and
    // frame become the history.
    for c in 0..channels {
        pre[c * PREFILTER_STRIDE..c * PREFILTER_STRIDE + pre_size]
            .copy_within(frame_size..pre_size, 0);
    }

    if cancel_pitch {
        (false, 0.0, pitch_index)
    } else {
        (pf_on, gain1, pitch_index)
    }
}

/// Append one frame of pre-emphasized, unfiltered input to the history of
/// one channel in `CeltEncoder::prefilter_buf` (C `prefilter_mem`). libopus
/// `run_prefilter` does this every frame, whether or not the comb filter is
/// enabled.
fn push_prefilter_history(mem: &mut [f32], frame: &[f32]) {
    let (n, max_period) = (frame.len(), mem.len());
    if n >= max_period {
        mem.copy_from_slice(&frame[n - max_period..]);
    } else {
        mem.copy_within(n.., 0);
        mem[max_period - n..].copy_from_slice(frame);
    }
}

/// Padding appended to `w_x` to absorb any SIMD over-shoot past the last band.
/// Sized to `MAX_PVQ_N` (352) — far larger than the widest SIMD access (16),
/// which is always loop-guarded (`while i + width <= n`). (Previously
/// `MAX_PVQ_N * 8`, which had no access pattern justifying it.)
const STRIDE_ACCESS_PAD: usize = crate::pvq::MAX_PVQ_N;

pub struct CeltEncoder {
    mode: &'static CeltMode,
    channels: usize,
    /// libopus `CELTEncoder.upsample`: the API input is zero-stuffed by this
    /// factor so the encoder always works at the mode's 48 kHz geometry.
    upsample: usize,
    /// End band the packet is coded to, from the Opus bandwidth (libopus
    /// `CELT_SET_END_BAND`): NB 13, MB/WB 17, SWB 19, FB 21. The decoder
    /// derives the same value from the TOC, so coding even one band more
    /// desynchronises the range coder.
    end_band: usize,
    /// First band the packet codes (libopus `CELT_SET_START_BAND`): 0, or
    /// 17 for Hybrid, where SILK codes the bands below.
    start_band: usize,
    pub complexity: i32,
    /// libopus `st->in_mem`: per channel, the last `overlap` samples of the
    /// previous frame after the pre-filter. The next frame's transform
    /// starts with them.
    in_mem: FixedVec<f32, CELT_IN_MEM>,
    old_band_e: FixedVec<f32, CELT_NB_X_CH>,
    preemph_mem: FixedVec<f32, CELT_MAX_CHANNELS>,
    tonal_average: i32,
    hf_average: i32,
    tapset_decision: i32,
    spread_decision: i32,
    intensity: i32,
    last_coded_bands: i32,
    /// Per channel, `PREFILTER_STRIDE` values. The first
    /// `COMBFILTER_MAXPERIOD` are libopus `st->prefilter_mem`: the
    /// pre-emphasized input before the pre-filter, up to the end of the
    /// last frame. During a frame, the frame itself follows them (libopus
    /// `pre` in `run_prefilter`). At the end of the frame the history moves
    /// up by the frame size. libopus copies the history into `pre` and back
    /// every frame; keeping the two together needs one move instead.
    prefilter_buf: FixedVec<f32, CELT_PREFILTER_BUF>,
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: i32,
    old_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    old_band_e3: FixedVec<f32, CELT_NB_X_CH>,
    last_band_log_e: FixedVec<f32, CELT_NB_X_CH>,
    delayed_intra: f32,
    lsb_depth: i32,
    overlap_max: f32,
    bitrate: i32,
    vbr: bool,
    constrained_vbr: bool,
    vbr_state: VbrState,

    w_in_buf: FixedVec<f32, CELT_BUFSTRIDE>,
    w_freq: FixedVec<f32, CELT_W_FREQ>,
    w_band_e: FixedVec<f32, CELT_NB_X_CH>,
    w_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    w_x: FixedVec<f32, CELT_W_X_ENC>,
    w_band_log_e: FixedVec<f32, CELT_NB_X_CH>,
    w_band_log_e2: FixedVec<f32, CELT_NB_X_CH>,
    w_error: FixedVec<f32, CELT_NB_X_CH>,
    w_tf_res: FixedVec<i32, CELT_NB_EBANDS>,
    w_cap: FixedVec<i32, CELT_NB_EBANDS>,
    w_offsets: FixedVec<i32, CELT_NB_EBANDS>,
    w_pulses: FixedVec<i32, CELT_NB_EBANDS>,
    w_ebits: FixedVec<i32, CELT_NB_X_CH>,
    w_fine_priority: FixedVec<i32, CELT_NB_X_CH>,
    w_collapse_masks: FixedVec<u32, CELT_NB_X_CH>,
    consec_transient: i32,
    /// libopus `st->rng`: the range of the coder at the end of the last
    /// frame (C 2813), the noise seed of `quant_all_bands` in the next one.
    rng: u32,

    w_prefilter_pitch_buf: FixedVec<f32, CELT_PREFILTER_PITCH>,
    w_prefilter_before: FixedVec<f32, CELT_MAX_CHANNELS>,
    w_prefilter_after: FixedVec<f32, CELT_MAX_CHANNELS>,

    w_analysis_tmp: FixedVec<f32, CELT_ANALYSIS_TMP>,

    loss_rate: i32,
    /// Hybrid only (libopus `silk_info.offset`): the quantization offset of
    /// the SILK frame coded before this CELT frame, in Q10. 100 is neutral.
    silk_offset: i32,
    /// Hybrid only (libopus `silk_info.signalType`): 0 no voice activity,
    /// 1 unvoiced, 2 voiced.
    silk_signal_type: i32,
    /// libopus `st->analysis` (`CELT_SET_ANALYSIS`): the speech and music
    /// analysis of the Opus layer for the next frame. Invalid unless the
    /// Opus layer sets one; a reset makes it invalid again.
    analysis: AnalysisInfo,
    /// libopus `st->lfe` (`OPUS_SET_LFE`): this encoder codes the
    /// low-frequency effects channel of a surround stream.
    lfe: bool,
    /// libopus `st->energy_mask` (`OPUS_SET_ENERGY_MASK`): 21 values per
    /// channel, in log2 units, or `None` without a mask.
    energy_mask: Option<[f32; CELT_NB_X_CH]>,
    /// libopus `st->stream_channels` (`CELT_SET_CHANNELS`): the number of
    /// channels coded in the packets. With 1 for a 2-channel encoder, the
    /// two input channels are coded as their mean.
    stream_channels: usize,
    /// libopus `st->disable_pf` (`CELT_SET_PREDICTION` 0 or 1): no pitch
    /// pre-filter.
    disable_pf: bool,
    /// libopus `st->disable_inv` (`OPUS_SET_PHASE_INVERSION_DISABLED`):
    /// intensity stereo never inverts the sign of the second channel. A
    /// reset keeps it.
    disable_inv: bool,
    /// libopus `st->force_intra` (`CELT_SET_PREDICTION` 0): the coarse
    /// energy of every frame is coded without prediction from the frame
    /// before.
    force_intra: bool,
    /// libopus `energyError`: per band and channel, what the energy
    /// quantizer of the last frame left over, limited to +-0.5. The next
    /// frame leans towards it when its energy is about the same.
    energy_error: FixedVec<f32, CELT_NB_X_CH>,
    /// libopus `st->spec_avg`: a slow average of the spectral level, for
    /// the temporal VBR. A frame above it gets more bits.
    spec_avg: f32,
    /// libopus `st->stereo_saving`: how much mid/side coding saves on the
    /// signal so far. `alloc_trim_analysis` raises it by at most 0.25 per
    /// frame; the VBR target of the next frames goes down with it.
    stereo_saving: f32,
    /// The cosines of the spreading rotations of recent frames (see
    /// `RotationCache`). It only saves time, so a reset keeps it.
    rotations: crate::pvq::RotationCache,
}

/// libopus `delayedIntra` after `OPUS_RESET_STATE`.
const DELAYED_INTRA_RESET: f32 = 1.0;

const INTEN_THRESHOLDS: [i32; 21] = [
    1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
];
const INTEN_HYSTERESIS: [i32; 21] = [
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
];

fn hysteresis_decision(val: i32, thresholds: &[i32], hysteresis: &[i32], prev: i32) -> i32 {
    let mut i = 0;
    while i < thresholds.len() {
        if val < thresholds[i] {
            break;
        }
        i += 1;
    }
    let mut res = i as i32;
    if res > prev && val < thresholds[prev as usize] + hysteresis[prev as usize] {
        res = prev;
    }
    if res < prev && res > 0 && val > thresholds[prev as usize - 1] - hysteresis[prev as usize - 1]
    {
        res = prev;
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn alloc_trim_analysis(
    mode: &CeltMode,
    x: &[f32],
    band_log_e: &[f32],
    end: usize,
    lm: i32,
    channels: usize,
    n0: usize,
    stereo_saving: &mut f32,
    tf_estimate: f32,
    intensity: i32,
    surround_trim: f32,
    equiv_rate: i32,
    // C `analysis->tonality_slope` when the analysis is valid.
    tonality_slope: Option<f32>,
) -> i32 {
    let mut trim = 5.0f32;
    // "At low bitrate, reducing the trim seems to help."
    if equiv_rate < 64000 {
        trim = 4.0;
    } else if equiv_rate < 80000 {
        // libopus steps in units of 1024 b/s.
        let frac = (equiv_rate - 64000) >> 10;
        trim = 4.0 + (1.0 / 16.0) * frac as f32;
    }

    if channels == 2 {
        // "Compute inter-channel correlation for low frequencies", with the
        // inner product of libopus (celt_inner_prod).
        let corr = |i: usize| {
            let offset = (mode.e_bands[i] as usize) << lm;
            let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
            crate::pitch::inner_prod(&x[offset..], &x[n0 + offset..], n)
        };
        let mut sum = 0.0f32;
        for i in 0..8 {
            sum += corr(i);
        }
        sum = (sum / 8.0).abs().min(1.0);
        let mut min_xc = sum;
        for i in 8..intensity.max(0) as usize {
            min_xc = min_xc.min(corr(i).abs());
        }
        min_xc = min_xc.min(1.0);

        // "mid-side savings estimations based on the LF average"
        let log_xc = celt_log2(1.001 - sum * sum);
        // "mid-side savings estimations based on min correlation"
        let log_xc2 = (log_xc * 0.5).max(celt_log2(1.001 - min_xc * min_xc));

        trim += (-4.0f32).max(0.75 * log_xc);
        *stereo_saving = (*stereo_saving + 0.25).min(-0.5 * log_xc2);
    }

    let mut diff = 0.0f32;
    for c in 0..channels {
        for i in 0..end - 1 {
            diff += band_log_e[c * mode.nb_ebands + i] * (2 + 2 * i as i32 - end as i32) as f32;
        }
    }
    diff /= (channels * (end - 1)) as f32;
    trim -= (-2.0f32).max(2.0f32.min((diff + 1.0) / 6.0));
    trim -= surround_trim;
    trim -= 2.0 * tf_estimate;
    // C 935-939: a signal whose tonality rises towards the high bands gets
    // a lower trim (more bits up there), within 2 steps either way. The
    // limits are the ternaries `MIN16` and `MAX16`.
    if let Some(slope) = tonality_slope {
        let t = 2.0 * (slope + 0.05);
        let t = if 2.0 < t { 2.0 } else { t };
        let t = if -2.0 > t { -2.0 } else { t };
        trim -= t;
    }

    let trim_index = (trim + 0.5).floor() as i32;
    trim_index.clamp(0, 10)
}

#[inline(always)]
fn median3(a: f32, b: f32, c: f32) -> f32 {
    let mut v = [a, b, c];
    crate::compat::sort_by(&mut v[..], |x, y| {
        x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal)
    });
    v[1]
}

#[inline(always)]
fn median5(v: &[f32]) -> f32 {
    let mut x = [v[0], v[1], v[2], v[3], v[4]];
    crate::compat::sort_by(&mut x[..], |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    x[2]
}

// Port of celt_encoder.c tone_lpc / tone_detect (float path, no FIXED_POINT shifts)
fn tone_lpc(x: &[f32], len: usize, delay: usize, lpc: &mut [f32; 2]) -> bool {
    // returns true on fail
    if len <= 2 * delay {
        return true;
    }
    let mut sums = ToneSums::default();
    for i in 0..len - 2 * delay {
        sums.add(x[i], x[i + delay], x[i + 2 * delay]);
    }
    tone_lpc_solve(x, len, delay, sums, lpc)
}

/// The three sums that `tone_lpc` starts with, over `i < len - 2 * delay`:
/// `x[i] * x[i]`, `x[i] * x[i + delay]` and `x[i] * x[i + 2 * delay]`.
///
/// Each is a long chain of additions, so the loop takes three cycles per
/// sample whatever else the processor could do. For a delay of 1, which
/// every frame needs, `transient_passes` computes them inside its own loop.
#[derive(Clone, Copy, Default)]
struct ToneSums {
    r00: f32,
    r01: f32,
    r02: f32,
}

impl ToneSums {
    /// Adds the three products of sample `x0`. `x1` and `x2` are the
    /// samples one and two delays after it.
    #[inline(always)]
    fn add(&mut self, x0: f32, x1: f32, x2: f32) {
        // `r00 += x[i] * x[i]` and so on: one fused multiply-add each in
        // libopus on 64-bit ARM (`mac`).
        self.r00 = mac(self.r00, x0, x0);
        self.r01 = mac(self.r01, x0, x1);
        self.r02 = mac(self.r02, x0, x2);
    }
}

/// `a * b - c * d` as libopus computes it on this processor: on 64-bit ARM
/// clang fuses the first product into the subtraction (`mac`).
#[inline(always)]
fn mul_sub_mul(a: f32, b: f32, c: f32, d: f32) -> f32 {
    mac(-(c * d), a, b)
}

/// The rest of `tone_lpc`, from the sums on. Requires `len > 2 * delay`.
/// Returns true on fail.
///
/// The steps round as in libopus on the same processor (`mul_sub_mul`).
/// The system is badly conditioned for a low tone, so one rounding step
/// can move the frequency enough to change the pre-filter period.
fn tone_lpc_solve(x: &[f32], len: usize, delay: usize, sums: ToneSums, lpc: &mut [f32; 2]) -> bool {
    let mut r00 = sums.r00;
    let mut r01 = sums.r01;
    let mut r02 = sums.r02;
    let mut r11: f32;
    let mut r12: f32;

    let mut edges = 0.0f32;
    for i in 0..delay {
        let (a, b) = (x[len + i - 2 * delay], x[i]);
        edges += mul_sub_mul(a, a, b, b);
    }
    r11 = r00 + edges;
    edges = 0.0;
    for i in 0..delay {
        let (a, b) = (x[len + i - delay], x[i + delay]);
        edges += mul_sub_mul(a, a, b, b);
    }
    let r22: f32 = r11 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += mul_sub_mul(x[len + i - 2 * delay], x[len + i - delay], x[i], x[i + delay]);
    }
    r12 = r01 + edges;
    // Reverse and sum to get backward contribution (float: simple sums)
    let (rr00, rr01, rr11, rr02, rr12) = (
        r00 + r22,
        r01 + r12,
        2.0 * r11,
        2.0 * r02,
        r12 + r01,
    );
    r00 = rr00;
    r01 = rr01;
    r11 = rr11;
    r02 = rr02;
    r12 = rr12;
    // "Solve A*x=b, where A=[r00, r01; r01, r11] and b=[r02; r12]."
    let den = mul_sub_mul(r00, r11, r01, r01);
    if den < 0.001 * (r00 * r11) {
        return true;
    }
    let num1 = mul_sub_mul(r02, r11, r01, r12);
    lpc[1] = if num1 >= den {
        1.0
    } else if num1 <= -den {
        -1.0
    } else {
        num1 / den
    };
    let num0 = mul_sub_mul(r00, r12, r02, r01);
    lpc[0] = if 0.5 * num0 >= den {
        1.999999
    } else if 0.5 * num0 <= -den {
        -1.999999
    } else {
        num0 / den
    };
    false
}

/// The signal `tone_detect` analyses for stereo input: the average of the
/// two channels. `in_buf` holds `mix.len()` samples per channel, one channel
/// after the other.
fn tone_mix_stereo(in_buf: &[f32], mix: &mut [f32]) {
    let (left, right) = in_buf[..2 * mix.len()].split_at(mix.len());
    for ((a, &l), &r) in mix.iter_mut().zip(left).zip(right) {
        *a = 0.5 * (l + r);
    }
}

/// Port of libopus `tone_detect`. Returns `(tone_freq, toneishness)`.
///
/// `x` is one channel of overlap and frame: the input for mono, the average
/// of the two channels for stereo (`tone_mix_stereo`). `first` holds the
/// sums of `x` for a delay of 1 when the caller has them already.
fn tone_detect(x: &[f32], fs: i32, first: Option<ToneSums>) -> (f32, f32) {
    let n = x.len();
    let mut delay = 1usize;
    let mut lpc = [0.0f32; 2];
    let mut fail = match first {
        Some(sums) if n > 2 => tone_lpc_solve(x, n, delay, sums, &mut lpc),
        _ => tone_lpc(x, n, delay, &mut lpc),
    };
    while delay <= (fs as usize / 3000) && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
        delay *= 2;
        fail = tone_lpc(x, n, delay, &mut lpc);
    }
    // "Check that our filter has complex roots." In the float build
    // 3.999999 is a double, so the sum is in double precision: one fused
    // multiply-add in libopus on 64-bit ARM.
    if !fail && mac_f64((lpc[0] * lpc[0]) as f64, lpc[1] as f64, 3.999999) < 0.0 {
        let toneishness = -lpc[1];
        // C `acos(.5f*lpc[0])/delay`: the double-precision `acos`, divided
        // in double precision, then rounded to float.
        let half = (0.5 * lpc[0]) as f64;
        #[cfg(feature = "std")]
        let angle = half.acos();
        #[cfg(not(feature = "std"))]
        let angle = crate::compat::Math::acos(half);
        let freq = (angle / delay as f64) as f32;
        (freq, toneishness)
    } else {
        (-1.0, 0.0)
    }
}

/// libopus `amp2Log2` (celt/quant_bands.c, float build) for the encoder:
/// the base-2 logarithm of the energy of every band below `end`, less the
/// mean energy of the band.
///
/// The energy goes in as it is. An empty band, whose energy is the square
/// root of the 1e-27 that `compute_band_energies` starts from, gets about
/// -44.8. (`bands::amp2log2` raises the energy to 1e-10 first, -33.2, and
/// puts -14 in the bands below its `start`.)
pub(crate) fn amp2log2_enc(
    mode: &CeltMode,
    end: usize,
    band_e: &[f32],
    band_log_e: &mut [f32],
    channels: usize,
) {
    let nb = mode.nb_ebands;
    for c in 0..channels {
        for i in 0..end {
            band_log_e[c * nb + i] = celt_log2(band_e[c * nb + i]) - mode.e_means[i];
        }
    }
}

/// What `dynalloc_analysis` finds besides the boosts it writes to
/// `offsets`.
struct Dynalloc {
    /// libopus `maxDepth`: how far the loudest band is above its noise
    /// floor. The VBR target goes by it (`compute_vbr`).
    max_depth: f32,
    /// libopus `tot_boost`: the bits the boosts are expected to take, in
    /// units of 1/8 bit.
    tot_boost: i32,
}

/// What the VBR target of a frame goes by (`VbrState::frame_bytes`): the
/// encoder settings and the results of the frame analysis.
struct VbrFrame {
    mode: &'static CeltMode,
    lm: usize,
    /// C `C`: the channels the packet codes.
    channels: usize,
    /// The frame codes bands from 17 up only (Hybrid).
    hybrid: bool,
    constrained_vbr: bool,
    lfe: bool,
    /// The encoder has an energy mask (C `st->energy_mask != NULL`).
    has_surround_mask: bool,
    /// Hybrid only: the quantization offset of the SILK frame, in Q10.
    silk_offset: i32,
    /// C `st->lastCodedBands`.
    last_coded_bands: i32,
    /// C `st->intensity`: the first band coded as intensity stereo.
    intensity: i32,
    /// C `st->stereo_saving`.
    stereo_saving: f32,
    /// C `equiv_rate`: the bitrate the allocation decisions go by.
    equiv_rate: i32,
    /// C `tell0_frac`: where the range coder stood when the frame started,
    /// in 1/8 bit.
    tell0_frac: i32,
    /// C `total_boost`: the bits the coded band boosts take, in 1/8 bit.
    total_boost: i32,
    dynalloc: Dynalloc,
    tf_estimate: f32,
    /// How much the other channels of a surround stream mask this one.
    surround_masking: f32,
    /// How far this frame's spectrum is above its slow average.
    temporal_vbr: f32,
    silence: bool,
    /// C `st->analysis`: the speech and music analysis of the Opus layer,
    /// or an invalid one.
    analysis: AnalysisInfo,
    /// C `pitch_change`: the pitch period moved by more than a quarter
    /// while the pre-filter was strong (see `CeltEncoder::encode`).
    pitch_change: bool,
}

/// The state of the VBR rate control, which keeps the average size of the
/// frames on the bitrate (libopus `vbr_reservoir`, `vbr_drift`,
/// `vbr_offset` and `vbr_count`). `OPUS_RESET_STATE` sets it to 0.
#[derive(Clone, Copy, Default)]
struct VbrState {
    /// What the frames so far took more than their target rate, in 1/8 bit
    /// (constrained VBR only).
    reservoir: i32,
    /// The average of what the targets missed the rate by, in 1/8 bit.
    drift: i32,
    /// The correction the next targets get, in 1/8 bit (`-drift`).
    offset: i32,
    /// Frames coded under VBR, up to 970.
    count: i32,
}

impl VbrState {
    /// C 2427-2534, under VBR: the bytes this frame takes, at most
    /// `nb_compressed_bytes`.
    ///
    /// The target starts from the bitrate (`vbr_rate`, in 1/8 bit per
    /// frame) and follows what the frame analysis found (`compute_vbr`,
    /// or the Hybrid rules). It never goes below what the coder has
    /// written so far (`tell_frac`, in 1/8 bit) and the boosts. Under
    /// constrained VBR the state is updated so that the average stays on
    /// the bitrate.
    fn frame_bytes(
        &mut self,
        frame: &VbrFrame,
        vbr_rate: i32,
        tell_frac: i32,
        nb_compressed_bytes: usize,
    ) -> usize {
        let lm = frame.lm;
        let channels = frame.channels as i32;
        // C 2427-2429: "Don't attempt to use more than 510 kb/s, even for
        // frames smaller than 20 ms."
        let nb_compressed_bytes =
            nb_compressed_bytes.min(crate::packet::MAX_FRAME_BYTES >> (3 - lm));
        let lm_diff = frame.mode.max_lm as i32 - lm as i32;
        let mut base_target = if frame.hybrid {
            0.max(vbr_rate - ((9 * channels + 4) << BITRES))
        } else {
            vbr_rate - ((40 * channels + 20) << BITRES)
        };
        if frame.constrained_vbr {
            base_target += self.offset >> lm_diff;
        }
        // The frame keeps at least what is coded so far and the boosts. A
        // Hybrid frame also keeps 37 bits more than it started with.
        let round_up = (1 << (BITRES + 3)) - 1;
        let mut min_allowed = ((tell_frac + frame.total_boost + round_up) >> (BITRES + 3)) + 2;
        if frame.hybrid {
            let hybrid_min =
                (frame.tell0_frac + (37 << BITRES) + frame.total_boost + round_up) >> (BITRES + 3);
            min_allowed = min_allowed.max(hybrid_min);
        }
        let mut target = if frame.hybrid {
            let mut t = base_target;
            // C 2469-2470: tonal frames, which are harder to code, get
            // extra bits; SILK's quantization offset tells them apart.
            if frame.silk_offset < 100 {
                t += (12 << BITRES) >> (3 - lm);
            }
            if frame.silk_offset > 100 {
                t -= (18 << BITRES) >> (3 - lm);
            }
            t += ((frame.tf_estimate - 0.25) * 50.0 * (1 << BITRES) as f32) as i32;
            if frame.tf_estimate > 0.7 {
                t = t.max(50 << BITRES);
            }
            t
        } else {
            compute_vbr(frame, base_target)
        };
        target += tell_frac;
        // C keeps this byte count signed. A target below zero then gives
        // `min_allowed`. As an unsigned count it wraps around and gives
        // the largest packet.
        let mut nb_available =
            ((target + (1 << (BITRES + 2))) >> (BITRES + 3)).max(min_allowed) as usize;
        nb_available = nb_available.min(nb_compressed_bytes);
        let mut delta = target - vbr_rate;
        target = (nb_available as i32) << (BITRES + 3);
        if frame.silence {
            nb_available = 2;
            target = (2 * 8) << BITRES;
            delta = 0;
        }
        // C 2502-2520, float build: `alpha` is 1 / (count + 20) for the
        // first 970 frames, then 0.001.
        let alpha = if self.count < 970 {
            self.count += 1;
            1.0 / (self.count + 20) as f32
        } else {
            0.001
        };
        if frame.constrained_vbr {
            self.reservoir += target - vbr_rate;
            // "Compute the offset we need to apply in order to reach the
            // target"
            let miss = delta * (1 << lm_diff) - self.offset - self.drift;
            self.drift += (alpha * miss as f32) as i32;
            self.offset = -self.drift;
        }
        if frame.constrained_vbr && self.reservoir < 0 {
            let adjust = (-self.reservoir) / (8 << BITRES);
            if !frame.silence {
                nb_available = (nb_available as i32 + adjust) as usize;
            }
            self.reservoir = 0;
        }
        nb_compressed_bytes.min(nb_available)
    }
}

/// libopus `compute_vbr`: the bit target of a CELT-only frame under VBR, in
/// 1/8 bit, from `base_target`. With a valid analysis from the Opus layer,
/// the target goes down for a frame of little activity, and up for a tonal
/// one and for a change of pitch.
fn compute_vbr(frame: &VbrFrame, base_target: i32) -> i32 {
    let mode = frame.mode;
    let lm = frame.lm as i32;
    let c = frame.channels as i32;
    let intensity = frame.intensity;
    let tf_estimate = frame.tf_estimate;
    let lfe = frame.lfe;
    let has_surround_mask = frame.has_surround_mask;
    let nb_ebands = mode.nb_ebands as i32;
    let mut coded_bands = if frame.last_coded_bands != 0 {
        frame.last_coded_bands
    } else {
        nb_ebands
    };
    coded_bands = coded_bands.clamp(0, nb_ebands);
    let mut coded_bins = mode.e_bands[coded_bands as usize] as i32 * (1 << lm);
    if c == 2 {
        let idx = intensity.min(coded_bands) as usize;
        coded_bins += mode.e_bands[idx] as i32 * (1 << lm);
    }
    let mut target = base_target;
    let analysis = &frame.analysis;
    // C 1630-1633. `activity < .4` compares in double precision.
    if analysis.valid && (analysis.activity as f64) < 0.4 {
        target -= ((coded_bins << BITRES) as f32 * (0.4 - analysis.activity)) as i32;
    }
    if c == 2 {
        // "Stereo savings". A saving below 0.1 raises the target, as in
        // libopus.
        let coded_stereo_bands = intensity.min(coded_bands);
        let coded_stereo_dof =
            mode.e_bands[coded_stereo_bands as usize] as i32 * (1 << lm) - coded_stereo_bands;
        // "Maximum fraction of the bits we can save if the signal is mono."
        let max_frac = (0.8 * coded_stereo_dof as f32) / coded_bins as f32;
        let ss = frame.stereo_saving.min(1.0);
        let save = max_frac * target as f32;
        let adjust = (ss - 0.1) * (coded_stereo_dof << BITRES) as f32;
        target -= save.min(adjust) as i32;
    }
    target += frame.dynalloc.tot_boost - (19 << lm);
    // libopus `SHL32(MULT16_32_Q15(tf_estimate-0.044, target), 1)`, which
    // is `(tf_estimate - 0.044) * target` in the float build.
    target += ((tf_estimate - 0.044) * target as f32) as i32;
    // C 1655-1673: "Apply tonality boost", compensating for the average
    // tonality. The limit is the ternary `MAX16`.
    if analysis.valid && !lfe {
        let above = analysis.tonality - 0.15;
        let tonal = (if 0.0 > above { 0.0 } else { above }) - 0.12;
        let bins = (coded_bins << BITRES) as f32;
        let mut tonal_target = target + ((bins * 1.2) * tonal) as i32;
        if frame.pitch_change {
            tonal_target += (bins * 0.8) as i32;
        }
        target = tonal_target;
    }
    if has_surround_mask && !lfe {
        let surround_target =
            target + (frame.surround_masking * coded_bins as f32 * (1 << BITRES) as f32) as i32;
        target = (target / 4).max(surround_target);
    }
    // The target goes down for a signal that is not far above its noise
    // floor (`max_depth`), down to a quarter.
    {
        let bins = mode.e_bands[(nb_ebands - 2) as usize] as i32 * (1 << lm);
        let floor_depth = ((c * bins * (1 << BITRES)) as f32 * frame.dynalloc.max_depth) as i32;
        let floor_depth = floor_depth.max(target >> 2);
        target = target.min(floor_depth);
    }
    if (!has_surround_mask || lfe) && frame.constrained_vbr {
        target = base_target + ((target - base_target) as f32 * 0.67) as i32;
    }
    if !has_surround_mask && tf_estimate < 0.2 {
        // C 2448 passes `equiv_rate` as the bitrate.
        let amount = 0.0000031 * (32000.min((96000 - frame.equiv_rate).max(0)) as f32);
        let tvbr_factor = frame.temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }
    target.min(2 * base_target)
}

/// libopus `dynalloc_analysis` (celt_encoder.c 1051-1295, float build).
///
/// Finds the bands that get more bits than the static allocation gives
/// them ("dynamic allocation"). `offsets[i]` gets the number of boost
/// steps of band `i`. The coded count can be smaller: each step is coded
/// while the band's cap and the packet allow it.
///
/// Also computes, for every band below `end`:
/// - `importance`: the weight of the band in the time-frequency decision
///   (`tf_analysis`). 13 when the analysis is skipped.
/// - `spread_weight`: the weight of the band in the spreading decision.
///   A band far below the masking of its neighbours counts less.
///
/// `band_log_e` holds the energies of this frame. `band_log_e2` holds the
/// same energies from long blocks, which `band_log_e` is not on a
/// transient. `old_band_e` holds the quantized energies of the frame
/// before.
#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis(
    mode: &CeltMode,
    band_log_e: &[f32],
    band_log_e2: &[f32],
    old_band_e: &[f32],
    start: usize,
    end: usize,
    channels: usize,
    offsets: &mut [i32],
    lsb_depth: i32,
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    lm: usize,
    effective_bytes: i32,
    lfe: bool,
    surround_dynalloc: &[f32],
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: f32,
    toneishness: f32,
    // C `analysis->leak_boost` when the analysis is valid.
    leak_boost: Option<&[u8; LEAK_BANDS]>,
) -> Dynalloc {
    let nb = mode.nb_ebands;
    let e_bands = mode.e_bands;
    offsets[..nb].fill(0);

    // "Noise floor must take into account eMeans, the depth, the width of
    // the bands and the preemphasis filter (approx. square of bark band
    // ID)". The terms are added in the order of libopus.
    let mut noise_floor = [0.0f32; CELT_NB_EBANDS];
    for (i, floor) in noise_floor.iter_mut().enumerate().take(end) {
        *floor = 0.0625 * mode.log_n[i] as f32 + 0.5 + (9 - lsb_depth) as f32 - mode.e_means[i]
            + 0.0062 * (i + 5) as f32 * (i + 5) as f32;
    }
    let mut max_depth = -31.9f32;
    for c in 0..channels {
        for i in 0..end {
            max_depth = max_depth.max(band_log_e[c * nb + i] - noise_floor[i]);
        }
    }

    // "Compute a really simple masking model to avoid taking into account
    // completely masked bands when computing the spreading decision."
    {
        let mut mask = [0.0f32; CELT_NB_EBANDS];
        for i in 0..end {
            mask[i] = band_log_e[i] - noise_floor[i];
        }
        if channels == 2 {
            for i in 0..end {
                mask[i] = mask[i].max(band_log_e[nb + i] - noise_floor[i]);
            }
        }
        let sig = mask;
        for i in 1..end {
            mask[i] = mask[i].max(mask[i - 1] - 2.0);
        }
        for i in (0..end.saturating_sub(1)).rev() {
            mask[i] = mask[i].max(mask[i + 1] - 3.0);
        }
        for i in 0..end {
            // "Compute SMR: Mask is never more than 72 dB below the peak and
            // never below the noise floor."
            let smr = sig[i] - (max_depth - 12.0).max(0.0).max(mask[i]);
            let shift = (-((0.5 + smr).floor() as i32)).clamp(0, 5);
            spread_weight[i] = 32 >> shift;
        }
    }

    let mut tot_boost = 0i32;
    // "Make sure that dynamic allocation can't make us bust the budget. We
    // enable the feature starting at 24 kb/s for 20-ms frames and 96 kb/s
    // for 2.5 ms frames."
    if effective_bytes >= 30 + 5 * lm as i32 && !lfe {
        let mut follower = [0.0f32; CELT_NB_X_CH];
        // libopus keeps `last` from the first channel for the second.
        let mut last = 0usize;
        for c in 0..channels {
            let mut e3 = [0.0f32; CELT_NB_EBANDS];
            e3[..end].copy_from_slice(&band_log_e2[c * nb..c * nb + end]);
            if lm == 0 {
                // "For 2.5 ms frames, the first 8 bands have just one bin,
                // so the energy is highly unreliable (high variance). For
                // that reason, we take the max with the previous energy so
                // that at least 2 bins are getting used."
                for i in 0..end.min(8) {
                    e3[i] = band_log_e2[c * nb + i].max(old_band_e[c * nb + i]);
                }
            }
            let f = &mut follower[c * nb..(c + 1) * nb];
            f[0] = e3[0];
            for i in 1..end {
                // "The last band to be at least 3 dB higher than the
                // previous one is the last we'll consider."
                if e3[i] > e3[i - 1] + 0.5 {
                    last = i;
                }
                f[i] = (f[i - 1] + 1.5).min(e3[i]);
            }
            for i in (0..last).rev() {
                f[i] = f[i].min((f[i + 1] + 2.0).min(e3[i]));
            }
            // "Combine with a median filter to avoid dynalloc triggering
            // unnecessarily."
            let offset = 1.0f32;
            for i in 2..end.saturating_sub(2) {
                f[i] = f[i].max(median5(&e3[i - 2..i + 3]) - offset);
            }
            if end >= 3 {
                let tmp = median3(e3[0], e3[1], e3[2]) - offset;
                f[0] = f[0].max(tmp);
                f[1] = f[1].max(tmp);
                let tmp = median3(e3[end - 3], e3[end - 2], e3[end - 1]) - offset;
                f[end - 2] = f[end - 2].max(tmp);
                f[end - 1] = f[end - 1].max(tmp);
            }
            for i in 0..end {
                f[i] = f[i].max(noise_floor[i]);
            }
        }
        if channels == 2 {
            for i in start..end {
                // "Consider 24 dB "cross-talk""
                follower[nb + i] = follower[nb + i].max(follower[i] - 4.0);
                follower[i] = follower[i].max(follower[nb + i] - 4.0);
                follower[i] = 0.5
                    * ((band_log_e[i] - follower[i]).max(0.0)
                        + (band_log_e[nb + i] - follower[nb + i]).max(0.0));
            }
        } else {
            for i in start..end {
                follower[i] = (band_log_e[i] - follower[i]).max(0.0);
            }
        }
        for i in start..end {
            follower[i] = follower[i].max(surround_dynalloc[i]);
        }
        for i in start..end {
            importance[i] = (0.5 + 13.0 * celt_exp2(follower[i].min(4.0))).floor() as i32;
        }
        // "For non-transient CBR/CVBR frames, halve the dynalloc
        // contribution"
        if (!vbr || constrained_vbr) && !is_transient {
            for f in follower[start..end].iter_mut() {
                *f *= 0.5;
            }
        }
        for (i, f) in follower.iter_mut().enumerate().take(end).skip(start) {
            if i < 8 {
                *f *= 2.0;
            }
            if i >= 12 {
                *f *= 0.5;
            }
        }
        // "Compensate for Opus' under-allocation on tones."
        if toneishness > 0.98 {
            let freq_bin =
                (0.5 + (tone_freq * 120.0) as f64 / core::f64::consts::PI).floor() as i32;
            for i in start..end {
                let (lo, hi) = (e_bands[i] as i32, e_bands[i + 1] as i32);
                if freq_bin >= lo && freq_bin <= hi {
                    follower[i] += 2.0;
                }
                if freq_bin >= lo - 1 && freq_bin <= hi + 1 {
                    follower[i] += 1.0;
                }
                if freq_bin >= lo - 2 && freq_bin <= hi + 2 {
                    follower[i] += 1.0;
                }
                if freq_bin >= lo - 3 && freq_bin <= hi + 3 {
                    follower[i] += 0.5;
                }
            }
            if freq_bin >= e_bands[end] as i32 && end >= 2 {
                follower[end - 1] += 2.0;
                follower[end - 2] += 1.0;
            }
        }
        // C 1223-1231: the analysis asks for more precision in bands that
        // leak into their neighbours, in steps of 1/64.
        if let Some(leak) = leak_boost {
            for i in start..end.min(LEAK_BANDS) {
                follower[i] += (1.0 / 64.0) * leak[i] as f32;
            }
        }
        for i in start..end {
            follower[i] = follower[i].min(4.0);
            let width = (channels as i32 * (e_bands[i + 1] - e_bands[i]) as i32) << lm;
            let (boost, boost_bits) = if width < 6 {
                let boost = follower[i] as i32;
                (boost, (boost * width) << BITRES)
            } else if width > 48 {
                let boost = (follower[i] * 8.0) as i32;
                (boost, ((boost * width) << BITRES) / 8)
            } else {
                let boost = (follower[i] * width as f32 / 6.0) as i32;
                (boost, (boost * 6) << BITRES)
            };
            // "For CBR and non-transient CVBR frames, limit dynalloc to 2/3
            // of the bits"
            if (!vbr || (constrained_vbr && !is_transient))
                && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes / 3
            {
                let cap = (2 * effective_bytes / 3) << BITRES << 3;
                // libopus stores the bits left, not a count of steps: the
                // band is boosted for as long as the packet allows.
                offsets[i] = cap - tot_boost;
                tot_boost = cap;
                break;
            }
            offsets[i] = boost;
            tot_boost += boost_bits;
        }
    } else {
        importance[start..end].fill(13);
    }
    Dynalloc {
        max_depth,
        tot_boost,
    }
}

/// How much the other surround channels mask this stream (libopus
/// celt_encode_with_ec, celt_encoder.c 2114-2188).
///
/// `energy_mask` is the mask given to `CeltEncoder::set_energy_mask`.
/// `surround_dynalloc` must be all zero on entry.
///
/// Returns `(surround_masking, surround_trim)`:
/// - `surround_masking` is the average mask over the coded bands. A
///   negative value lowers the VBR target.
/// - `surround_trim` is the slope of the mask over frequency. It moves the
///   allocation trim towards the less masked end of the spectrum.
///
/// `surround_dynalloc[i]` is set for a band that is masked much less than
/// that average and slope predict. The dynamic allocation boosts it.
fn surround_masking_analysis(
    mode: &CeltMode,
    energy_mask: &[f32],
    channels: usize,
    last_coded_bands: i32,
    surround_dynalloc: &mut [f32; CELT_NB_EBANDS],
) -> (f32, f32) {
    let nb_ebands = mode.nb_ebands;
    let e_bands = mode.e_bands;
    let mask_end = (last_coded_bands.max(2) as usize).min(nb_ebands);
    let mut mask_avg = 0.0f32;
    let mut diff = 0.0f32;
    let mut count = 0i32;
    for c in 0..channels {
        for i in 0..mask_end {
            // `min` then `max`, as the C macros: a NaN in the mask becomes
            // 0.25 here instead of spreading.
            #[allow(clippy::manual_clamp)]
            let mut mask = energy_mask[nb_ebands * c + i].min(0.25).max(-2.0);
            if mask > 0.0 {
                mask *= 0.5;
            }
            let width = (e_bands[i + 1] - e_bands[i]) as i32;
            mask_avg += mask * width as f32;
            count += width;
            diff += mask * (1 + 2 * i as i32 - mask_end as i32) as f32;
        }
    }
    mask_avg = mask_avg / count as f32 + 0.2;
    diff = diff * 6.0 / (channels * (mask_end - 1) * (mask_end + 1) * mask_end) as f32;
    // C: "Again, being conservative".
    diff = (0.5 * diff).clamp(-0.031, 0.031);
    // The band in the middle of the coded spectrum.
    let mut midband = 0usize;
    while (e_bands[midband + 1] as i32) < e_bands[mask_end] as i32 / 2 {
        midband += 1;
    }
    let mut count_dynalloc = 0;
    for i in 0..mask_end {
        let lin = mask_avg + diff * (i as i32 - midband as i32) as f32;
        // The raw mask values here, not the limited ones.
        let unmask = if channels == 2 {
            energy_mask[i].max(energy_mask[nb_ebands + i])
        } else {
            energy_mask[i]
        };
        let unmask = unmask.min(0.0) - lin;
        if unmask > 0.25 {
            surround_dynalloc[i] = unmask - 0.25;
            count_dynalloc += 1;
        }
    }
    if count_dynalloc >= 3 {
        // C: "If we need dynalloc in many bands, it's probably because our
        // initial masking rate was too low."
        mask_avg += 0.25;
        if mask_avg > 0.0 {
            // The first estimate was far off: no masking at all.
            mask_avg = 0.0;
            diff = 0.0;
            surround_dynalloc[..mask_end].fill(0.0);
        } else {
            for v in surround_dynalloc[..mask_end].iter_mut() {
                *v = (*v - 0.25).max(0.0);
            }
        }
    }
    mask_avg += 0.2;
    // C scales the slope by 64 for the allocation trim: the limit of 0.031
    // above lets the mask move the trim by at most 2 steps.
    (mask_avg, 64.0 * diff)
}

/// How many of the `nb_available` bytes its packet has left a frame may use
/// (libopus celt_encode_with_ec, celt_encoder.c 1936-1961).
///
/// `vbr_rate` is the target rate in 1/8 bit per frame under constrained VBR,
/// and `None` when there is no such bound. A frame may take twice that
/// rate, less what earlier frames took too much (`vbr_reservoir`). `tell`
/// is the range coder position at the start of the frame. A packet that is
/// still empty always gets 2 bytes.
fn constrained_vbr_bytes(
    vbr_rate: Option<i32>,
    vbr_reservoir: i32,
    tell: i32,
    nb_available: usize,
) -> usize {
    let Some(vbr_rate) = vbr_rate else {
        return nb_available;
    };
    let vbr_bound = vbr_rate;
    let max_allowed = core::cmp::max(
        if tell == 1 { 2 } else { 0 },
        (vbr_rate + vbr_bound - vbr_reservoir) >> (BITRES + 3),
    );
    nb_available.min(max_allowed as usize)
}

/// Tests of the encoder's own kernels: each fast version against the plain
/// loop it replaces, bit for bit.
#[cfg(test)]
mod enc_kernel_tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;

    /// Small deterministic generator, so that a failure can be repeated.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / (1u32 << 30) as f32 - 1.0
        }
    }

    /// Kinds of test input, from ordinary samples to values that overflow.
    #[derive(Clone, Copy, Debug)]
    enum Kind {
        /// Samples in the range the API takes: -1 to 1.
        Audio,
        /// Random signs and magnitudes between 1e-30 and 1e15.
        WideRange,
        /// Zeros of both signs, subnormal numbers and the smallest normal one.
        Tiny,
        /// The largest finite values: products overflow to infinity.
        Huge,
        /// Infinities and both kinds of not-a-number among ordinary samples.
        NonFinite,
    }

    const KINDS: [Kind; 5] = [
        Kind::Audio,
        Kind::WideRange,
        Kind::Tiny,
        Kind::Huge,
        Kind::NonFinite,
    ];

    fn samples(rng: &mut Rng, n: usize, kind: Kind) -> Vec<f32> {
        (0..n)
            .map(|_| match kind {
                Kind::Audio => rng.unit(),
                Kind::WideRange => {
                    let exp = (rng.next_u32() % 46) as i32 - 30;
                    rng.unit() * 10f32.powi(exp)
                }
                Kind::Tiny => match rng.next_u32() % 5 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::MIN_POSITIVE,
                    3 => -f32::from_bits(1 + rng.next_u32() % 1000),
                    _ => f32::from_bits(1 + rng.next_u32() % 0x0080_0000),
                },
                Kind::Huge => match rng.next_u32() % 4 {
                    0 => f32::MAX,
                    1 => f32::MIN,
                    _ => 1e30 * rng.unit(),
                },
                Kind::NonFinite => match rng.next_u32() % 16 {
                    0 => f32::INFINITY,
                    1 => f32::NEG_INFINITY,
                    2 => f32::NAN,
                    // A signaling not-a-number, of either sign.
                    3 => f32::from_bits(0x7F80_0001 | (rng.next_u32() << 31)),
                    _ => 100.0 * rng.unit(),
                },
            })
            .collect()
    }

    /// Bit-for-bit comparison. Two not-a-number values count as equal: which
    /// payload an operation returns is not something the codec relies on.
    fn same_bits(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[track_caller]
    fn assert_same(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lengths differ");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(same_bits(x, y), "{what}: index {i}: {x:e} against {y:e}");
        }
    }

    /// Lengths that cover the empty slice, one sample, every tail length of
    /// the 4- and 16-sample loops, and the sizes the encoder uses.
    fn lengths() -> impl Iterator<Item = usize> {
        (0..=67).chain([120, 240, 360, 479, 840, 841, 960, 1080])
    }

    #[test]
    fn max_abs_matches_the_scalar_loop() {
        let mut rng = Rng::new(1);
        let inits = [0.0f32, -0.0, 1e-9, 0.25, 1.0, 3e38, f32::INFINITY, -1.0];
        for kind in KINDS {
            for n in lengths() {
                for round in 0..4 {
                    let x = samples(&mut rng, n, kind);
                    let init = inits[(n + round) % inits.len()];
                    let want = max_abs_scalar(&x, init);
                    let got = max_abs(&x, init);
                    assert!(
                        same_bits(want, got),
                        "{kind:?} n={n} init={init:e}: {want:e} against {got:e}"
                    );
                }
            }
        }
    }

    /// The largest value is found wherever it is, and a sample that is not
    /// a number at any position is skipped.
    #[test]
    fn max_abs_finds_a_peak_at_every_position() {
        let not_numbers = [
            f32::NAN,
            f32::from_bits(0x7F80_0001),
            f32::from_bits(0xFFC0_0000),
        ];
        for n in [1usize, 3, 4, 5, 16, 17, 31, 32, 33, 67, 120] {
            for pos in 0..n {
                for bad in not_numbers {
                    let mut x = vec![0.125f32; n];
                    x[pos] = -0.75;
                    assert_eq!(max_abs(&x, 0.0), 0.75, "n={n} pos={pos}");
                    assert_eq!(max_abs(&x, 0.875), 0.875, "n={n} pos={pos}");
                    // A not-a-number next to the peak, then one in its place.
                    x[(pos + 1) % n] = bad;
                    let want = max_abs_scalar(&x, 0.0);
                    assert!(same_bits(max_abs(&x, 0.0), want), "n={n} pos={pos}");
                    x[pos] = bad;
                    let want = max_abs_scalar(&x, 0.0);
                    assert!(same_bits(max_abs(&x, 0.0), want), "n={n} pos={pos}");
                    assert!(!max_abs(&x, 0.0).is_nan());
                }
            }
        }
        // An empty slice and a slice of only not-a-number values give `init`.
        assert_eq!(max_abs(&[], 0.5), 0.5);
        assert_eq!(max_abs(&[f32::NAN; 40], 0.5), 0.5);
    }

    /// The pre-emphasis loop as it was written before `celt_preemphasis`:
    /// one pass over the 48 kHz samples, with a division to find the input
    /// sample and a remainder to find the stuffed zeros.
    fn preemphasis_reference(
        pcm: &[f32],
        out: &mut [f32],
        upsample: usize,
        coef: f32,
        mem: f32,
    ) -> f32 {
        let mut m = mem;
        for i in 0..pcm.len() * upsample {
            let x = if i % upsample == 0 {
                pcm[i / upsample] * 32768.0
            } else {
                0.0
            };
            out[i] = x - m;
            m = x * coef;
        }
        m
    }

    #[test]
    fn preemphasis_matches_the_reference_loop() {
        let mut rng = Rng::new(2);
        // 0.85 is the coefficient of the 48 kHz mode.
        for coef in [0.85f32, 0.0, -0.5, 1.0] {
            for upsample in [1usize, 2, 3, 4, 6] {
                for kind in KINDS {
                    for n in (0..=19).chain([20, 40, 80, 120, 160, 240, 480, 960]) {
                        let pcm = samples(&mut rng, n, kind);
                        let mem = samples(&mut rng, 1, kind)[0];
                        let mut want = vec![0.0f32; n * upsample];
                        let mut got = vec![f32::NAN; n * upsample];
                        let want_mem = preemphasis_reference(&pcm, &mut want, upsample, coef, mem);
                        let got_mem = celt_preemphasis(&pcm, &mut got, upsample, coef, mem);
                        let what = format!("{kind:?} n={n} upsample={upsample} coef={coef}");
                        assert_same(&want, &got, &what);
                        assert!(same_bits(want_mem, got_mem), "{what}: filter memory");
                    }
                }
            }
        }
    }

    /// The order in which `sum_abs_neon` adds, written as a plain loop.
    #[cfg(target_arch = "aarch64")]
    fn sum_abs_lanes_reference(x: &[f32]) -> f32 {
        let (groups, rest) = x.as_chunks::<4>();
        let mut s = [0.0f32; 4];
        for group in groups {
            for (acc, &v) in s.iter_mut().zip(group) {
                *acc += v.abs();
            }
        }
        let mut sum = (s[0] + s[1]) + (s[2] + s[3]);
        for &v in rest {
            sum += v.abs();
        }
        sum
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sum_abs_neon_matches_its_reference_order() {
        let mut rng = Rng::new(8);
        for kind in KINDS {
            for n in lengths() {
                let x = samples(&mut rng, n, kind);
                let want = sum_abs_lanes_reference(&x);
                let got = sum_abs(&x);
                assert!(same_bits(want, got), "{kind:?} n={n}: {want:e} against {got:e}");
            }
        }
    }

    /// `l1_metric` checks its length argument before a vector kernel reads.
    #[test]
    #[should_panic]
    fn l1_metric_rejects_a_length_past_the_slice() {
        let x = [1.0f32; 24];
        let _ = l1_metric(&x, 32, 0, 0.0);
    }

    /// `comb_filter_const` checks its ranges before a vector kernel reads
    /// in front of `x` or writes past `y`.
    #[test]
    #[should_panic(expected = "range outside its buffers")]
    fn comb_filter_const_rejects_a_period_before_the_buffer() {
        let x = [0.0f32; 64];
        let mut y = [0.0f32; 64];
        comb_filter_const(&mut y, &x, 0, 16, 15, 8, 0.1, 0.1, 0.1);
    }

    #[test]
    #[should_panic(expected = "range outside its buffers")]
    fn comb_filter_const_rejects_an_output_range_past_the_buffer() {
        let x = [0.0f32; 64];
        let mut y = [0.0f32; 8];
        comb_filter_const(&mut y, &x, 0, 32, 15, 16, 0.1, 0.1, 0.1);
    }

    #[test]
    #[should_panic(expected = "range outside its buffers")]
    fn comb_filter_const_rejects_an_input_range_past_the_buffer() {
        let x = [0.0f32; 40];
        let mut y = [0.0f32; 64];
        comb_filter_const(&mut y, &x, 0, 32, 15, 9, 0.1, 0.1, 0.1);
    }

    /// The kernels read `x[x_idx + i - t + 2]` with an unsigned `t - 2`, so
    /// a period below 2 is refused.
    #[test]
    #[should_panic(expected = "range outside its buffers")]
    fn comb_filter_const_rejects_a_period_below_two() {
        let x = [0.0f32; 64];
        let mut y = [0.0f32; 64];
        comb_filter_const(&mut y, &x, 0, 32, 1, 8, 0.1, 0.1, 0.1);
    }

    /// `l1_metric` below 16 values, where every target runs the plain loop:
    /// one sum in index order, then the bias term. Only the first `n`
    /// values of the slice count.
    #[test]
    fn l1_metric_adds_short_input_in_index_order() {
        let mut rng = Rng::new(10);
        for kind in KINDS {
            for n in 0..16usize {
                for lm in 0..=4i32 {
                    let x = samples(&mut rng, n + 3, kind);
                    let bias = 0.04 * rng.unit();
                    let mut l1 = 0.0f32;
                    for &v in &x[..n] {
                        l1 += v.abs();
                    }
                    let want = l1 + (lm as f32) * bias * l1;
                    let got = l1_metric(&x, n, lm, bias);
                    assert!(same_bits(want, got), "{kind:?} n={n} lm={lm}: {want:e} against {got:e}");
                }
            }
        }
    }

    /// `l1_metric` from 16 values on, where the NEON kernel runs. It adds in
    /// the order of `sum_abs_lanes_reference`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn l1_metric_neon_matches_its_reference_order() {
        let mut rng = Rng::new(11);
        for kind in KINDS {
            for n in lengths().filter(|&n| n >= 16) {
                for lm in 0..=4i32 {
                    let x = samples(&mut rng, n + 5, kind);
                    let bias = 0.04 * rng.unit();
                    let l1 = sum_abs_lanes_reference(&x[..n]);
                    let want = l1 + (lm as f32) * bias * l1;
                    let got = l1_metric(&x, n, lm, bias);
                    assert!(same_bits(want, got), "{kind:?} n={n} lm={lm}: {want:e} against {got:e}");
                }
            }
        }
    }

    /// `comb_filter_const` with one index expression per tap, as libopus
    /// writes the filter:
    /// `y[i] = x[i] + g10 * x[i - T] + g11 * (x[i - T + 1] + x[i - T - 1])
    /// + g12 * (x[i - T + 2] + x[i - T - 2])`, each product added with
    /// `mac` (one fused multiply-add on 64-bit ARM).
    fn comb_filter_const_reference(x: &[f32], x_idx: usize, t: usize, n: usize, g: [f32; 3]) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let at = x_idx + i;
                let centre = x[at - t];
                let near = x[at - t + 1] + x[at - t - 1];
                let far = x[at - t + 2] + x[at - t - 2];
                mac(mac(mac(x[at], g[0], centre), g[1], near), g[2], far)
            })
            .collect()
    }

    /// The arguments of `comb_filter_const`: `y`, `x`, `y_idx`, `x_idx`,
    /// `t`, `n` and the three gains.
    type CombFilterConst = fn(&mut [f32], &[f32], usize, usize, usize, usize, f32, f32, f32);

    /// Runs `kernel` on every combination of period, length and position
    /// worth testing and compares its output with the reference.
    ///
    /// `x` is exactly as long as the kernel may read: it starts `lead`
    /// samples before the first tap and ends with the last input sample.
    /// `y` has four more values on each side of the output range, which
    /// must keep the pattern they start with.
    fn check_comb_filter_const(kernel: CombFilterConst) {
        let mut rng = Rng::new(12);
        // 2 is the smallest period the kernels accept, 15 and 1022 are the
        // limits of the pitch pre-filter and of the post-filter. The
        // lengths cover every tail of the 4-, 8- and 16-sample loops and
        // what the codec passes: 0 to 960 samples.
        for t in [2usize, 3, 4, 5, 15, 16, 17, 100, 1022] {
            for n in (0..=21).chain([40, 119, 120, 360, 840, 960, 961]) {
                for (lead, y_idx) in [(0usize, 4usize), (1, 5), (7, 4)] {
                    for kind in KINDS {
                        let x_idx = lead + t + 2;
                        let x = samples(&mut rng, x_idx + n, kind);
                        let g = [0.3 * rng.unit(), 0.2 * rng.unit(), 0.1 * rng.unit()];
                        let want = comb_filter_const_reference(&x, x_idx, t, n, g);

                        let mut y = vec![-7.5f32; y_idx + n + 4];
                        kernel(&mut y, &x, y_idx, x_idx, t, n, g[0], g[1], g[2]);
                        let what = format!("{kind:?} t={t} n={n} lead={lead}");
                        assert_same(&want, &y[y_idx..y_idx + n], &what);
                        let kept = y[..y_idx].iter().chain(&y[y_idx + n..]).all(|&v| v == -7.5);
                        assert!(kept, "{what}: wrote outside the output range");
                    }
                }
            }
        }
    }

    /// The portable version. This machine does not run it inside the codec.
    #[test]
    fn comb_filter_const_scalar_matches_the_formula() {
        check_comb_filter_const(comb_filter_const_scalar);
    }

    /// The version the codec runs on 64-bit ARM.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn comb_filter_const_neon_matches_its_reference_order() {
        check_comb_filter_const(comb_filter_const);
    }

    /// `transient_analysis` as it was written before its loops were fused:
    /// three loops per channel, one channel after the other, as in libopus.
    /// Returns `(is_transient, tf_estimate, tf_chan, weak_transient)`.
    fn transient_reference(
        input: &[f32],
        len: usize,
        channels: usize,
        allow_weak_transients: bool,
        tone_freq: f32,
        toneishness: f32,
    ) -> (bool, f32, usize, bool) {
        let mut tmp = vec![0.0f32; len];
        let mut tmp2 = vec![0.0f32; len / 2];
        let mut tf_chan = 0;
        let mut weak_transient = false;
        let mut mask_metric = 0.0f32;
        let forward_decay = if allow_weak_transients {
            0.03125f32
        } else {
            0.0625f32
        };
        let len2 = len / 2;

        for c in 0..channels {
            let mut mem0 = 0.0f32;
            let mut mem1 = 0.0f32;
            for i in 0..len {
                let x = input[c * len + i];
                let y = mem0 + x;
                let mem00 = mem0;
                mem0 = mem0 - x + 0.5 * mem1;
                mem1 = x - mem00;
                tmp[i] = y;
            }
            tmp[..12].fill(0.0);

            let mut mean = 0.0f32;
            mem0 = 0.0f32;
            for i in 0..len2 {
                let x2 = mac(tmp[2 * i + 1] * tmp[2 * i + 1], tmp[2 * i], tmp[2 * i]);
                mean += x2;
                mem0 = mac(x2, 1.0 - forward_decay, mem0);
                tmp2[i] = forward_decay * mem0;
            }

            mem0 = 0.0f32;
            let mut max_e = 0.0f32;
            for i in (0..len2).rev() {
                mem0 = mac(tmp2[i], 0.875, mem0);
                tmp2[i] = 0.125 * mem0;
                if tmp2[i] > max_e {
                    max_e = tmp2[i];
                }
            }

            mean = (f64::from(mean * max_e) * 0.5 * len2 as f64).sqrt() as f32;
            let norm = (len2 as f32) / (1e-15 + mean);
            let mut unmask: i32 = 0;
            for i in (12..(len2 - 5)).step_by(4) {
                let id = (64.0 * norm * (tmp2[i] + 1e-15)).floor() as i32;
                unmask += INV_TABLE[id.clamp(0, 127) as usize] as i32;
            }
            unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
            if (unmask as f32) > mask_metric {
                tf_chan = c;
                mask_metric = unmask as f32;
            }
        }

        let mut is_transient = mask_metric > 200.0;
        if toneishness > 0.98 && tone_freq < 0.026 {
            is_transient = false;
            mask_metric = 0.0;
        }
        if allow_weak_transients && is_transient && mask_metric < 600.0 {
            is_transient = false;
            weak_transient = true;
        }
        let tf_max = ((27.0 * mask_metric).sqrt() - 42.0).max(0.0);
        let x = (0.0069f32 * tf_max.min(163.0)) as f64 - 0.139;
        let tf_estimate = x.max(0.0).sqrt() as f32;
        (is_transient, tf_estimate, tf_chan, weak_transient)
    }

    /// The two parts of the transient analysis in a row, as the encoder
    /// calls them. Also checks what the first part computes for the tone
    /// detector: the mono mix and the sums for a delay of 1.
    #[allow(clippy::too_many_arguments)]
    fn transient_analysis(
        input: &[f32],
        len: usize,
        channels: usize,
        tf_estimate: &mut f32,
        tf_chan: &mut usize,
        allow_weak_transients: bool,
        weak_transient: &mut bool,
        tone_freq: f32,
        toneishness: f32,
        envelope: &mut [f32],
    ) -> bool {
        let mut mix = vec![f32::NAN; len + 2];
        let (passes, sums) =
            transient_passes(input, len, channels, allow_weak_transients, envelope, &mut mix);

        let x: Vec<f32> = if channels == 2 {
            let mut want_mix = vec![0.0f32; len];
            tone_mix_stereo(input, &mut want_mix);
            assert_same(&want_mix, &mix[..len], "mono mix");
            want_mix
        } else {
            input[..len].to_vec()
        };
        // The mix must stay inside its `len` samples.
        assert!(mix[len..].iter().all(|v| v.is_nan()));
        let mut want = ToneSums::default();
        for i in 0..len.saturating_sub(2) {
            want.add(x[i], x[i + 1], x[i + 2]);
        }
        assert!(same_bits(want.r00, sums.r00), "r00 len={len} channels={channels}");
        assert!(same_bits(want.r01, sums.r01), "r01 len={len} channels={channels}");
        assert!(same_bits(want.r02, sums.r02), "r02 len={len} channels={channels}");
        // With these sums the tone detector gives what it gives without.
        let with = tone_detect(&x, 48000, Some(sums));
        let without = tone_detect(&x, 48000, None);
        assert!(same_bits(with.0, without.0) && same_bits(with.1, without.1));

        transient_decision(
            &passes,
            envelope,
            len,
            channels,
            tf_estimate,
            tf_chan,
            allow_weak_transients,
            weak_transient,
            tone_freq,
            toneishness,
        )
    }

    /// `tone_lpc` as libopus writes it (celt_encoder.c), in one function,
    /// with the rounding of libopus on this processor (`mac`): the
    /// version without `ToneSums` and `tone_lpc_solve`.
    fn tone_lpc_reference(x: &[f32], len: usize, delay: usize, lpc: &mut [f32; 2]) -> bool {
        if len <= 2 * delay {
            return true;
        }
        let mut r00 = 0.0f32;
        let mut r01 = 0.0f32;
        let mut r02 = 0.0f32;
        for i in 0..len - 2 * delay {
            r00 = mac(r00, x[i], x[i]);
            r01 = mac(r01, x[i], x[i + delay]);
            r02 = mac(r02, x[i], x[i + 2 * delay]);
        }
        let mut edges = 0.0f32;
        for i in 0..delay {
            let (a, b) = (x[len + i - 2 * delay], x[i]);
            edges += mac(-(b * b), a, a);
        }
        let r11 = r00 + edges;
        edges = 0.0;
        for i in 0..delay {
            let (a, b) = (x[len + i - delay], x[i + delay]);
            edges += mac(-(b * b), a, a);
        }
        let r22 = r11 + edges;
        edges = 0.0;
        for i in 0..delay {
            let (a, b) = (x[len + i - 2 * delay], x[len + i - delay]);
            edges += mac(-(x[i] * x[i + delay]), a, b);
        }
        let r12 = r01 + edges;
        let (r00, r01, r11, r02, r12) = (r00 + r22, r01 + r12, 2.0 * r11, 2.0 * r02, r12 + r01);
        let den = mac(-(r01 * r01), r00, r11);
        if den < 0.001 * (r00 * r11) {
            return true;
        }
        let num1 = mac(-(r01 * r12), r02, r11);
        lpc[1] = if num1 >= den {
            1.0
        } else if num1 <= -den {
            -1.0
        } else {
            num1 / den
        };
        let num0 = mac(-(r02 * r01), r00, r12);
        lpc[0] = if 0.5 * num0 >= den {
            1.999999
        } else if 0.5 * num0 <= -den {
            -1.999999
        } else {
            num0 / den
        };
        false
    }

    /// `tone_detect` as libopus writes it: every delay through
    /// `tone_lpc_reference`.
    fn tone_detect_reference(x: &[f32], fs: i32) -> (f32, f32) {
        let n = x.len();
        let mut delay = 1usize;
        let mut lpc = [0.0f32; 2];
        let mut fail = tone_lpc_reference(x, n, delay, &mut lpc);
        while delay <= (fs as usize / 3000) && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
            delay *= 2;
            fail = tone_lpc_reference(x, n, delay, &mut lpc);
        }
        if !fail && mac_f64((lpc[0] * lpc[0]) as f64, lpc[1] as f64, 3.999999) < 0.0 {
            let freq = (((0.5 * lpc[0]) as f64).acos() / delay as f64) as f32;
            (freq, -lpc[1])
        } else {
            (-1.0, 0.0)
        }
    }

    #[test]
    fn tone_detect_matches_the_version_before_the_split() {
        let mut rng = Rng::new(9);
        let mut tones = 0usize;
        let mut later_delays = 0usize;
        for len in [2usize, 3, 4, 5, 40, 41, 240, 360, 600, 1080] {
            for round in 0..200u32 {
                // Sine tones over the whole range of frequencies (where the
                // detector reports a tone), some with noise, and the signals
                // of the transient tests.
                let freq = 0.002 + 3.0 * rng.unit().abs();
                let noise = if round % 3 == 0 { 0.0 } else { 50.0 * rng.unit().abs() };
                let x: Vec<f32> = if round % 2 == 0 {
                    (0..len)
                        .map(|i| 9000.0 * (freq * i as f32).sin() + noise * rng.unit())
                        .collect()
                } else {
                    transient_signal(&mut rng, len, round)
                };
                let want = tone_detect_reference(&x, 48000);
                let got = tone_detect(&x, 48000, None);
                let what = format!("len={len} round={round}");
                assert!(same_bits(want.0, got.0), "{what}: tone_freq");
                assert!(same_bits(want.1, got.1), "{what}: toneishness");
                tones += (got.1 > 0.9) as usize;

                let mut lpc = [0.0f32; 2];
                if !tone_lpc(&x, len, 1, &mut lpc) && lpc[0] > 1.0 && lpc[1] < 0.0 {
                    later_delays += 1;
                }
                for delay in [1usize, 2, 4, 8, 16, 32] {
                    let mut a = [7.0f32; 2];
                    let mut b = [7.0f32; 2];
                    let fa = tone_lpc_reference(&x, len, delay, &mut a);
                    let fb = tone_lpc(&x, len, delay, &mut b);
                    assert_eq!(fa, fb, "{what} delay={delay}: fail flag");
                    assert!(same_bits(a[0], b[0]) && same_bits(a[1], b[1]), "{what} delay={delay}");
                }
            }
        }
        // The signals must reach both outcomes and the longer delays.
        assert!(tones > 100, "{tones}");
        assert!(later_delays > 100, "{later_delays}");
    }

    /// Test signals for the transient analysis, `len` samples at the level
    /// of 16-bit audio. `shape` picks one of: noise, noise with a burst at
    /// a random place, a quiet start and a loud end, a loud start and a
    /// quiet end, a sine tone, silence, a single click.
    fn transient_signal(rng: &mut Rng, len: usize, shape: u32) -> Vec<f32> {
        let at = (rng.next_u32() as usize) % len;
        let step = 0.01 + 0.5 * rng.unit().abs();
        (0..len)
            .map(|i| match shape % 7 {
                0 => 3000.0 * rng.unit(),
                1 => {
                    let gain = if i >= at && i < at + 40 { 20000.0 } else { 200.0 };
                    gain * rng.unit()
                }
                2 => (if i >= at { 15000.0 } else { 30.0 }) * rng.unit(),
                3 => (if i < at { 15000.0 } else { 30.0 }) * rng.unit(),
                4 => 8000.0 * (step * i as f32).sin(),
                5 => 0.0,
                _ => {
                    if i == at {
                        30000.0
                    } else {
                        0.0
                    }
                }
            })
            .collect()
    }

    #[test]
    fn transient_analysis_matches_the_three_loop_version() {
        let mut rng = Rng::new(4);
        let mut seen_transient = [0usize; 2];
        let mut seen_weak = 0usize;
        let mut seen_chan1 = 0usize;
        // Frame sizes 2.5, 5, 10 and 20 ms plus the overlap, and four short
        // lengths, two of them odd. The scratch starts with other values.
        for len in [40usize, 41, 62, 63, 240, 360, 600, 1080] {
            for channels in 1..=2usize {
                for round in 0..120u32 {
                    let mut input = transient_signal(&mut rng, len, round);
                    if channels == 2 {
                        input.extend(transient_signal(&mut rng, len, round / 7));
                    }
                    let allow_weak = round % 3 == 0;
                    let (tone_freq, toneishness) = match round % 5 {
                        0 => (0.01, 0.99),
                        1 => (0.5, 0.99),
                        _ => (-1.0, 0.0),
                    };
                    let want =
                        transient_reference(&input, len, channels, allow_weak, tone_freq, toneishness);

                    let mut envelope = vec![f32::NAN; channels * (len / 2) + 3];
                    let mut tf_estimate = -1.0f32;
                    let mut tf_chan = 0usize;
                    let mut weak = true;
                    let is_transient = transient_analysis(
                        &input,
                        len,
                        channels,
                        &mut tf_estimate,
                        &mut tf_chan,
                        allow_weak,
                        &mut weak,
                        tone_freq,
                        toneishness,
                        &mut envelope,
                    );
                    let what = format!("len={len} channels={channels} round={round}");
                    assert_eq!(is_transient, want.0, "{what}: is_transient");
                    assert!(same_bits(tf_estimate, want.1), "{what}: tf_estimate");
                    assert_eq!(tf_chan, want.2, "{what}: tf_chan");
                    assert_eq!(weak, want.3, "{what}: weak_transient");
                    seen_transient[is_transient as usize] += 1;
                    seen_weak += weak as usize;
                    seen_chan1 += tf_chan;
                }
            }
        }
        // The signals must reach every outcome, or the test shows little.
        assert!(seen_transient[0] > 100 && seen_transient[1] > 100, "{seen_transient:?}");
        assert!(seen_weak > 10, "{seen_weak}");
        assert!(seen_chan1 > 50, "{seen_chan1}");
    }

    /// Values that overflow or are not numbers take the same path through
    /// both versions.
    #[test]
    fn transient_analysis_matches_on_extreme_input() {
        let mut rng = Rng::new(5);
        for kind in KINDS {
            for channels in 1..=2usize {
                for len in [240usize, 1080] {
                    let input = samples(&mut rng, channels * len, kind);
                    let want = transient_reference(&input, len, channels, false, -1.0, 0.0);
                    let mut envelope = vec![0.0f32; channels * (len / 2)];
                    let mut tf_estimate = 0.0f32;
                    let mut tf_chan = 0usize;
                    let mut weak = false;
                    let is_transient = transient_analysis(
                        &input,
                        len,
                        channels,
                        &mut tf_estimate,
                        &mut tf_chan,
                        false,
                        &mut weak,
                        -1.0,
                        0.0,
                        &mut envelope,
                    );
                    let what = format!("{kind:?} len={len} channels={channels}");
                    assert_eq!(is_transient, want.0, "{what}: is_transient");
                    assert!(same_bits(tf_estimate, want.1), "{what}: tf_estimate");
                    assert_eq!(tf_chan, want.2, "{what}: tf_chan");
                    assert_eq!(weak, want.3, "{what}: weak_transient");
                }
            }
        }
    }

    /// The 48 kHz path (the vector kernel where there is one) against the
    /// portable loop: every length up to 40, so every tail of the 8- and
    /// 4-sample loops, and the frame sizes.
    #[test]
    fn preemphasis_kernel_matches_the_scalar_loop() {
        let mut rng = Rng::new(7);
        for kind in KINDS {
            for n in (0..=40).chain([120, 240, 480, 960]) {
                for coef in [0.85f32, -0.25] {
                    let pcm = samples(&mut rng, n, kind);
                    let mem = samples(&mut rng, 1, kind)[0];
                    let mut want = vec![0.0f32; n];
                    let mut got = vec![f32::NAN; n];
                    let want_mem = preemphasis_scalar(&pcm, &mut want, coef, mem);
                    let got_mem = celt_preemphasis(&pcm, &mut got, 1, coef, mem);
                    let what = format!("{kind:?} n={n} coef={coef}");
                    assert_same(&want, &got, &what);
                    assert!(same_bits(want_mem, got_mem), "{what}: filter memory");
                }
            }
        }
    }

    /// Frame after frame with the memory carried over gives the same output
    /// as one long frame.
    #[test]
    fn preemphasis_carries_its_memory_across_frames() {
        let mut rng = Rng::new(3);
        for upsample in [1usize, 2, 3, 4, 6] {
            let pcm = samples(&mut rng, 480, Kind::Audio);
            let mut whole = vec![0.0f32; 480 * upsample];
            let whole_mem = celt_preemphasis(&pcm, &mut whole, upsample, 0.85, 0.25);
            let mut parts = vec![0.0f32; 480 * upsample];
            let mut mem = 0.25;
            for (src, dst) in pcm.chunks(60).zip(parts.chunks_mut(60 * upsample)) {
                mem = celt_preemphasis(src, dst, upsample, 0.85, mem);
            }
            assert_same(&whole, &parts, "frames against one block");
            assert!(same_bits(whole_mem, mem));
        }
    }

    /// `preemphasis_input` on interleaved stereo gives each channel what the
    /// reference loop gives that channel alone, at every rate. The portable
    /// stereo loop is checked against the reference as well.
    #[test]
    fn interleaved_preemphasis_matches_each_channel_alone() {
        let mut rng = Rng::new(11);
        let overlap = 5;
        for upsample in [1usize, 2, 3, 4, 6] {
            for kind in KINDS {
                for n in (0..=19).chain([20, 40, 120, 160, 240, 480, 960]) {
                    let channels = [samples(&mut rng, n, kind), samples(&mut rng, n, kind)];
                    let mem = [samples(&mut rng, 1, kind)[0], samples(&mut rng, 1, kind)[0]];
                    let pcm: Vec<f32> = (0..2 * n).map(|i| channels[i % 2][i / 2]).collect();
                    let buf_stride = overlap + n * upsample;
                    let mut in_buf = vec![f32::NAN; 2 * buf_stride];
                    let mut got_mem = mem;
                    preemphasis_input(
                        &pcm,
                        2,
                        &mut in_buf,
                        buf_stride,
                        overlap,
                        upsample,
                        0.85,
                        &mut got_mem,
                    );
                    let mut want = [vec![0.0f32; n * upsample], vec![0.0f32; n * upsample]];
                    for c in 0..2 {
                        let what = format!("{kind:?} n={n} upsample={upsample} channel {c}");
                        let (input, out) = (&channels[c], &mut want[c]);
                        let want_mem = preemphasis_reference(input, out, upsample, 0.85, mem[c]);
                        let (head, got) =
                            in_buf[c * buf_stride..(c + 1) * buf_stride].split_at(overlap);
                        assert_same(out, got, &what);
                        assert!(same_bits(want_mem, got_mem[c]), "{what}: filter memory");
                        assert!(head.iter().all(|v| v.is_nan()), "{what}: history");
                    }
                    if upsample == 1 {
                        let (mut left, mut right) = (vec![0.0f32; n], vec![0.0f32; n]);
                        let pairs = pcm.as_chunks::<2>().0;
                        let m = preemphasis_stereo_scalar(pairs, &mut left, &mut right, 0.85, mem);
                        let what = format!("{kind:?} n={n} portable stereo loop");
                        assert_same(&want[0], &left, &what);
                        assert_same(&want[1], &right, &what);
                        assert!(same_bits(m[0], got_mem[0]) && same_bits(m[1], got_mem[1]));
                    }
                }
            }
        }
    }
}

impl CeltEncoder {
    pub fn new(mode: &'static CeltMode, channels: usize) -> Self {
        Self::with_sampling_rate(mode, channels, 48000)
    }

    /// CELT encoder for `sampling_rate`, which need not be 48 kHz: the input
    /// is zero-stuffed by `resampling_factor(sampling_rate)` so the mode's
    /// 48 kHz geometry is used internally, exactly like libopus
    /// (`celt_encoder.c` `st->upsample`).
    pub fn with_sampling_rate(
        mode: &'static CeltMode,
        channels: usize,
        sampling_rate: i32,
    ) -> Self {
        // All internal buffers are sized for at most 2 channels. A larger
        // value would panic later inside `FixedVec::from_value` with an
        // unhelpful message.
        assert!(
            (1..=2).contains(&channels),
            "CeltEncoder::new: channels must be 1 or 2 (got {channels})"
        );
        let upsample = resampling_factor(sampling_rate).unwrap_or(1);
        let overlap = mode.overlap;
        let nb_ebands = mode.nb_ebands;
        let nb_x_ch = nb_ebands * channels;
        let frame_x_ch = CELT_ENC_MAX_FRAME * channels;
        let bufstride_x_ch = (CELT_ENC_MAX_FRAME + overlap) * channels;
        Self {
            mode,
            channels,
            upsample,
            end_band: mode.nb_ebands,
            start_band: 0,
            complexity: 9,
            in_mem: FixedVec::from_value(0.0, channels * overlap),
            old_band_e: FixedVec::from_value(0.0, nb_x_ch),
            preemph_mem: FixedVec::from_value(0.0, channels),
            tonal_average: 256,
            hf_average: 0,
            tapset_decision: 0,
            spread_decision: SPREAD_NORMAL,
            intensity: 0,
            last_coded_bands: 0,
            prefilter_buf: FixedVec::from_value(0.0, channels * PREFILTER_STRIDE),
            // libopus clears its state, so the first frame has no pitch
            // period to continue.
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            old_band_e2: FixedVec::from_value(OLD_LOG_E_RESET, nb_x_ch),
            old_band_e3: FixedVec::from_value(OLD_LOG_E_RESET, nb_x_ch),
            last_band_log_e: FixedVec::from_value(0.0, nb_x_ch),
            delayed_intra: DELAYED_INTRA_RESET,
            lsb_depth: 24,
            overlap_max: 0.0,
            bitrate: OPUS_BITRATE_MAX,
            vbr: false,
            constrained_vbr: true,
            vbr_state: VbrState::default(),

            w_in_buf: FixedVec::from_value(0.0, bufstride_x_ch),
            w_freq: FixedVec::from_value(0.0, frame_x_ch + 4),
            w_band_e: FixedVec::from_value(0.0, nb_x_ch),
            w_band_e2: FixedVec::from_value(0.0, nb_x_ch),

            w_x: FixedVec::from_value(0.0, frame_x_ch + STRIDE_ACCESS_PAD),
            w_band_log_e: FixedVec::from_value(0.0, nb_x_ch),
            w_band_log_e2: FixedVec::from_value(0.0, nb_x_ch),
            w_error: FixedVec::from_value(0.0, nb_x_ch),
            w_tf_res: FixedVec::from_value(0, nb_ebands),
            w_cap: FixedVec::from_value(0, nb_ebands),
            w_offsets: FixedVec::from_value(0, nb_ebands),
            w_pulses: FixedVec::from_value(0, nb_ebands),
            w_ebits: FixedVec::from_value(0, nb_x_ch),
            w_fine_priority: FixedVec::from_value(0, nb_x_ch),
            w_collapse_masks: FixedVec::from_value(0, nb_x_ch),

            w_prefilter_pitch_buf: FixedVec::from_value(
                0.0,
                (COMBFILTER_MAXPERIOD + CELT_ENC_MAX_FRAME) >> 1,
            ),
            w_prefilter_before: FixedVec::from_value(0.0, channels),
            w_prefilter_after: FixedVec::from_value(0.0, channels),
            w_analysis_tmp: FixedVec::from_value(0.0, CELT_ANALYSIS_TMP),
            consec_transient: 0,
            rng: 0,

            loss_rate: 0,
            silk_offset: 0,
            silk_signal_type: 0,
            analysis: AnalysisInfo::default(),
            lfe: false,
            energy_mask: None,
            stream_channels: channels,
            disable_pf: false,
            disable_inv: false,
            force_intra: false,
            energy_error: FixedVec::from_value(0.0, nb_x_ch),
            rotations: crate::pvq::RotationCache::new(),
            spec_avg: 0.0,
            stereo_saving: 0.0,
        }
    }

    /// libopus `OPUS_RESET_STATE`: forget the signal coded so far and keep
    /// the settings.
    ///
    /// libopus clears everything from its field `rng` on
    /// (celt_encoder.c, `opus_custom_encoder_ctl`). That is:
    /// - the input history: pre-emphasis memories, the overlap of the last
    ///   frame, the pre-filter history, and the largest sample of the
    ///   last overlap (`overlap_max`);
    /// - the pre-filter: period 0, gain 0, tap set 0;
    /// - the band energies and the energy quantization errors to 0, and the
    ///   two older energy histories to -28;
    /// - the VBR reservoir, drift, offset and frame count to 0;
    /// - `lastCodedBands`, the intensity band, the stereo saving, the
    ///   spectral average of the temporal VBR and the count of transient
    ///   frames to 0;
    /// - the SILK information to 0;
    /// - the analysis of the Opus layer becomes invalid;
    /// - the energy mask is removed.
    ///
    /// It then sets the decision states to their start values: spreading
    /// normal, tap set 0, `tonal_average` 256, `hf_average` 0, and
    /// `delayedIntra` 1.
    ///
    /// The settings stay: complexity, start and end band, bitrate, VBR and its
    /// constraint, LSB depth, LFE, packet loss, prediction mode and the
    /// number of coded channels (`set_stream_channels`).
    pub fn reset_state(&mut self) {
        let nb_x_ch = self.mode.nb_ebands * self.channels;
        self.spread_decision = SPREAD_NORMAL;
        self.delayed_intra = DELAYED_INTRA_RESET;
        self.tonal_average = 256;
        self.last_coded_bands = 0;
        self.hf_average = 0;
        self.tapset_decision = 0;
        self.prefilter_period = 0;
        self.prefilter_gain = 0.0;
        self.prefilter_tapset = 0;
        self.consec_transient = 0;
        self.rng = 0;
        self.silk_signal_type = 0;
        self.silk_offset = 0;
        self.analysis = AnalysisInfo::default();
        self.preemph_mem.fill(0.0);
        self.vbr_state = VbrState::default();
        self.overlap_max = 0.0;
        self.intensity = 0;
        self.energy_mask = None;
        self.in_mem.fill(0.0);
        self.prefilter_buf.fill(0.0);
        self.old_band_e[..nb_x_ch].fill(0.0);
        self.old_band_e2[..nb_x_ch].fill(OLD_LOG_E_RESET);
        self.old_band_e3[..nb_x_ch].fill(OLD_LOG_E_RESET);
        self.last_band_log_e[..nb_x_ch].fill(0.0);
        self.energy_error[..nb_x_ch].fill(0.0);
        self.spec_avg = 0.0;
        self.stereo_saving = 0.0;
    }

    /// Set the highest coded band (`CELT_SET_END_BAND`). Must match the
    /// bandwidth the packet's TOC signals.
    pub fn set_end_band(&mut self, end: usize) {
        self.end_band = end;
    }

    /// libopus `CELT_SET_START_BAND`: the first band the packets code. 0
    /// codes all bands. Hybrid frames start at band 17, because SILK codes
    /// the signal below it. A value of 21 or more is an error, and the
    /// setting stays.
    pub fn set_start_band(&mut self, start: usize) -> Result<(), &'static str> {
        if start >= self.mode.nb_ebands {
            return Err("CELT start band must be below the number of bands");
        }
        self.start_band = start;
        Ok(())
    }

    /// libopus `CELT_SET_PREDICTION`: how much a frame may depend on the
    /// frames before it.
    /// - 2 (the default): no limit.
    /// - 1: no pitch pre-filter.
    /// - 0: no pitch pre-filter, and the coarse energy of every frame is
    ///   coded without prediction from the frame before (intra).
    ///
    /// The Opus encoder sets 0 when `OPUS_SET_PREDICTION_DISABLED` is on,
    /// and 2 otherwise. Any other value is an error, and the setting stays.
    pub fn set_prediction(&mut self, value: i32) -> Result<(), &'static str> {
        if !(0..=2).contains(&value) {
            return Err("CELT prediction must be 0, 1 or 2");
        }
        self.disable_pf = value <= 1;
        self.force_intra = value == 0;
        Ok(())
    }

    /// libopus `CELT_SET_CHANNELS`: the number of channels coded in the
    /// packets, 1 or at most the channels of the encoder.
    ///
    /// A 2-channel encoder set to 1 codes the mean of its two input
    /// channels as a mono stream (libopus `compute_mdcts`: half the sum of
    /// the two spectra). The input analyses (transient, tone, pitch
    /// pre-filter) still see both channels. The energy of the mono stream
    /// also becomes the history of the second channel, so that the
    /// stream can switch back to stereo. The Opus encoder sets this before
    /// every frame from its stereo decision.
    pub fn set_stream_channels(&mut self, channels: usize) -> Result<(), &'static str> {
        if channels == 0 || channels > self.channels {
            return Err("CELT stream channels must be 1 or the encoder's channel count");
        }
        self.stream_channels = channels;
        Ok(())
    }

    /// The number of channels coded in the packets (`set_stream_channels`).
    pub fn stream_channels(&self) -> usize {
        self.stream_channels
    }

    /// libopus `OPUS_SET_PACKET_LOSS_PERC` for the CELT encoder: the
    /// expected packet loss in percent, 0 to 100. The Opus encoder passes
    /// its own setting on.
    ///
    /// With loss, the pitch pre-filter is weaker (half its gain from 3 %,
    /// a quarter from 5 %, off from 9 %), and the coarse energy is coded
    /// without prediction from the frame before more often. Any other
    /// value is an error, and the setting stays.
    pub fn set_packet_loss_perc(&mut self, percent: i32) -> Result<(), &'static str> {
        if !(0..=100).contains(&percent) {
            return Err("packet loss must be 0 to 100 percent");
        }
        self.loss_rate = percent;
        Ok(())
    }

    pub fn set_lsb_depth(&mut self, depth: i32) {
        self.lsb_depth = depth.clamp(8, 24);
    }

    /// libopus `OPUS_SET_BITRATE` for the CELT encoder: the bitrate VBR
    /// aims at, in bits per second. With `OPUS_BITRATE_MAX` (-1, the
    /// default) there is no target, and a frame takes the bytes it is given.
    /// A value of 500 or less is an error, and the setting stays. A value
    /// above 750 kb/s per channel is lowered to that, as libopus does.
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), &'static str> {
        if bitrate <= 500 && bitrate != OPUS_BITRATE_MAX {
            return Err("CELT bitrate must be above 500 b/s");
        }
        self.bitrate = bitrate.min(750_000 * self.channels as i32);
        Ok(())
    }
    pub fn set_vbr(&mut self, vbr: bool) {
        self.vbr = vbr;
    }
    pub fn set_constrained_vbr(&mut self, cvbr: bool) {
        self.constrained_vbr = cvbr;
    }
    /// Hybrid only (libopus `CELT_SET_SILK_INFO`): the signal type (0 no
    /// voice activity, 1 unvoiced, 2 voiced) and the quantization offset
    /// (Q10) of the SILK frame coded before this CELT frame.
    pub fn set_silk_info(&mut self, signal_type: i32, offset_q10: i32) {
        self.silk_signal_type = signal_type;
        self.silk_offset = offset_q10;
    }
    /// libopus `CELT_SET_ANALYSIS`: the result of the speech and music
    /// analysis for the next frames. The Opus layer sets it before every
    /// CELT or Hybrid frame. When `info.valid` is set, the encoder reads:
    ///
    /// - `tonality_slope` for the allocation trim,
    /// - `leak_boost` for the dynamic allocation,
    /// - `max_pitch_ratio` for the gain of the pitch pre-filter,
    /// - `tonality` and `activity` for the VBR target,
    /// - `bandwidth` for the highest band the allocation may skip to.
    ///
    /// The value stays until the next call or [`reset_state`](Self::reset_state).
    pub fn set_analysis(&mut self, info: &AnalysisInfo) {
        self.analysis = *info;
    }
    /// Low-frequency effects mode (libopus `OPUS_SET_LFE`, celt_encoder.c).
    /// Use it for the LFE channel of a surround stream, which is mono.
    ///
    /// In this mode the encoder spends its bits on the two lowest bands. It
    /// skips the transient, time-frequency and dynamic allocation analyses,
    /// and it keeps every band from band 2 up at least 80 dB below band 0.
    pub fn set_lfe(&mut self, lfe: bool) {
        self.lfe = lfe;
    }
    /// Phase inversion disabling (libopus `OPUS_SET_PHASE_INVERSION_DISABLED`,
    /// celt_encoder.c). With `true`, intensity stereo codes both channels
    /// of a band with the same sign. The default is `false`.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.disable_inv = disabled;
    }
    /// The value set with `set_phase_inversion_disabled`.
    pub fn phase_inversion_disabled(&self) -> bool {
        self.disable_inv
    }
    /// Surround masking input (libopus `OPUS_SET_ENERGY_MASK`).
    ///
    /// `mask` holds 21 values per channel, one channel after the other, in
    /// log2 units (libopus `celt_glog`). Each value says how far a band of
    /// this stream is above the level at which the other surround channels
    /// mask it. A negative value means the band is masked. The encoder then
    /// spends fewer bits on masked bands and, under VBR, on the whole frame.
    ///
    /// `None` removes the mask. A slice shorter than `21 * channels` is not
    /// a mask and removes it too. The mask has no effect in LFE mode and on
    /// Hybrid frames.
    ///
    /// libopus keeps the caller's pointer and reads it on every frame. This
    /// function copies the values, so call it again when they change. The
    /// multistream encoder computes a new mask for every frame.
    pub fn set_energy_mask(&mut self, mask: Option<&[f32]>) {
        let n = self.mode.nb_ebands * self.channels;
        self.energy_mask = mask.filter(|m| m.len() >= n).map(|m| {
            let mut values = [0.0; CELT_NB_X_CH];
            values[..n].copy_from_slice(&m[..n]);
            values
        });
    }

    /// libopus `celt_encode_with_ec`: code one frame of `api_frame_size`
    /// samples per channel into `rc`. `pcm` holds the channels interleaved,
    /// as the Opus encoder gets them.
    ///
    /// The frame may take every byte of `rc` (`rc.storage`). Under VBR it
    /// takes fewer, and `rc` is shrunk to the bytes it took. The packet
    /// codes the bands from `set_start_band` to `set_end_band`.
    pub fn encode(&mut self, pcm: &[f32], api_frame_size: usize, rc: &mut RangeCoder) {
        let mode = self.mode;
        let start_band = self.start_band;
        // C `CC`: the channels of the input. `channels` is C `C`, the
        // channels coded in the packet. The input analyses and the
        // pre-filter work on `cc` channels, everything from the band
        // energies on on `channels`.
        let cc = self.channels;
        let pcm = &pcm[..api_frame_size * cc];
        let channels = self.stream_channels.min(cc);
        let upsample = self.upsample;
        // Internal (48 kHz) frame size; `pcm` is at the API rate.
        let frame_size = api_frame_size * upsample;
        let nb_ebands = mode.nb_ebands;
        let end = self.end_band.clamp(1, nb_ebands).max(start_band);
        let overlap = mode.overlap;

        // The encoder geometry requires frame_size == short_mdct_size << lm
        // (120/240/480/960). Any other size would mis-size the MDCT buffers.
        // OpusEncoder::encode() validates this before reaching CELT.
        let Some(lm) = mode.lm_for(frame_size) else {
            panic!(
                "CeltEncoder: invalid frame_size {} (valid: {}..={} kHz multiples of {})",
                frame_size,
                mode.short_mdct_size,
                mode.short_mdct_size << mode.max_lm,
                mode.short_mdct_size
            );
        };

        // C `vbr_rate`: under VBR, the bits the frame should take, in 1/8
        // bit. `None` under CBR.
        let vbr_rate = (self.vbr && self.bitrate != OPUS_BITRATE_MAX)
            .then(|| bitrate_to_bits(self.bitrate, mode.fs, frame_size) << BITRES);
        let hybrid = start_band != 0;
        // C `nbCompressedBytes` on entry: the room the packet has.
        let packet_bytes = rc.storage as usize;
        let tell_initial = rc.tell();
        let tell_initial_frac = rc.tell_frac();
        let nb_filled_bytes_initial = ((tell_initial + 4) >> 3).max(0) as usize;
        let (mut nb_compressed_bytes, mut effective_bytes) =
            self.frame_bytes(rc, vbr_rate, packet_bytes, tell_initial);
        let mut total_bits = (nb_compressed_bytes * 8) as i32;

        // `in_buf` holds, per channel, `overlap` samples of history and then
        // the pre-emphasized frame (C `in`). The frame is written straight
        // into it.
        //
        // C 2017: tone and transient analysis see the *unfiltered* history in
        // the overlap head: the tail of the history in `prefilter_buf` (C
        // `prefilter_mem`). The prefiltered overlap of the previous frame
        // (`in_mem`) is put there afterwards. Analysing that one puts a step
        // between head and body wherever the comb filter was active, which
        // reads as a transient on steady pitched input.
        let buf_stride = frame_size + overlap;
        let max_period = COMBFILTER_MAXPERIOD;
        let in_buf = &mut self.w_in_buf[..buf_stride * cc];
        for (c, channel) in in_buf.chunks_exact_mut(buf_stride).enumerate() {
            let history_end = c * PREFILTER_STRIDE + max_period;
            channel[..overlap]
                .copy_from_slice(&self.prefilter_buf[history_end - overlap..history_end]);
        }
        preemphasis_input(
            pcm,
            cc,
            in_buf,
            buf_stride,
            overlap,
            upsample,
            mode.preemph[0],
            &mut self.preemph_mem,
        );

        let mut tf_estimate = 0.0f32;
        let mut tf_chan = 0;
        let mut weak_transient = false;

        // C 1969-2005: silence detection (sample_max, overlap_max and
        // lsb_depth). libopus measures the maxima on the API-rate `pcm`:
        // the first `C*(N-overlap)/upsample` samples, then the next
        // `C*overlap/upsample` ones. The order of the samples does not
        // change a maximum, so all channels are measured together.
        //
        // `C` is the number of coded channels (`channels`), but `pcm` holds
        // the `cc` input channels interleaved. So when a stereo input is
        // coded as mono, libopus measures only the first half of the frame,
        // with both input channels in it. The rest of the frame takes no
        // part in the silence flag. opus1 does the same.
        let api_overlap = overlap / upsample;
        let n_nonoverlap = api_frame_size.saturating_sub(api_overlap);
        let (body, rest) = pcm.split_at(n_nonoverlap * channels);
        let tail = &rest[..api_overlap * channels];
        let mut sample_max = max_abs(body, self.overlap_max);
        let new_overlap_max = max_abs(tail, 0.0);
        self.overlap_max = new_overlap_max;
        if new_overlap_max > sample_max {
            sample_max = new_overlap_max;
        }
        let threshold = 1.0 / ((1 << self.lsb_depth) as f32);
        let mut silence = sample_max <= threshold;
        if tell_initial == 1 {
            rc.encode_bit_logp(silence, 15);
        } else {
            silence = false;
        }
        // C `tell`: where the range coder stood on entry. A silent frame
        // counts as full, so nothing else is coded in it.
        let mut tell_c = tell_initial;
        // C `nbAvailableBytes = 2` of a silent frame under VBR.
        let mut silent_vbr = false;
        if silence {
            // "In VBR mode there is no need to send more than the minimum."
            if vbr_rate.is_some() {
                let target_bytes = nb_filled_bytes_initial + 2;
                if target_bytes < nb_compressed_bytes {
                    nb_compressed_bytes = target_bytes;
                    total_bits = (nb_compressed_bytes * 8) as i32;
                    rc.shrink(nb_compressed_bytes as u32);
                }
                // C `effectiveBytes=nbCompressedBytes=IMIN(..)`: the
                // analyses that still run go by the two bytes.
                effective_bytes = nb_compressed_bytes;
                silent_vbr = true;
            }
            let cur_tell = rc.tell();
            tell_c = (nb_compressed_bytes * 8) as i32;
            rc.nbits_total += tell_c - cur_tell;
        }
        // C 2032: Hybrid at a low CELT rate, on anything but voiced speech.
        let hybrid_low_rate = hybrid && effective_bytes < 15 && self.silk_signal_type != 2;
        // C `nbAvailableBytes`: what the packet has left, after the VBR bound.
        let nb_available_bytes = if silent_vbr {
            2
        } else {
            nb_compressed_bytes.saturating_sub(nb_filled_bytes_initial)
        };
        // C 1928-1930: the rate the allocation decisions go by.
        let frame_overhead = (40 * channels as i32 + 20) * ((400 >> lm) - 50);
        let mut equiv_rate = ((packet_bytes as i32 * 8 * 50) << (3 - lm)) - frame_overhead;
        if self.bitrate != OPUS_BITRATE_MAX {
            equiv_rate = equiv_rate.min(self.bitrate - frame_overhead);
        }

        // Tone detection (C 2022) and transient analysis (C 2027), both on
        // `in_buf` with its N + overlap samples per channel. libopus detects
        // the tone first. Here the two passes of the transient analysis come
        // first, because their loop also adds up the sums the tone detector
        // starts with. The decision about the transient comes last: it
        // needs the tone.
        //
        // The scratch buffer holds the mono mix for the tone detector in its
        // first half and the envelopes of the transient analysis in its
        // second half.
        let (mix, envelope) = self.w_analysis_tmp.split_at_mut(CELT_ANALYSIS_TMP / 2);
        // C 2027: an LFE frame is never analysed for transients.
        let passes = if self.complexity >= 1 && !self.lfe {
            Some(transient_passes(
                in_buf,
                buf_stride,
                cc,
                hybrid_low_rate,
                envelope,
                mix,
            ))
        } else {
            if cc == 2 {
                tone_mix_stereo(in_buf, &mut mix[..buf_stride]);
            }
            None
        };
        let tone_input: &[f32] = if cc == 2 {
            &mix[..buf_stride]
        } else {
            &in_buf[..buf_stride]
        };
        let (tone_freq, mut toneishness) =
            tone_detect(tone_input, mode.fs, passes.as_ref().map(|p| p.1));
        let mut is_transient = match &passes {
            Some((passes, _)) => transient_decision(
                passes,
                envelope,
                buf_stride,
                cc,
                &mut tf_estimate,
                &mut tf_chan,
                hybrid_low_rate,
                &mut weak_transient,
                tone_freq,
                toneishness,
            ),
            None => false,
        };
        // C 2093: a mono stream has only the one channel to analyse.
        if cc == 2 && channels == 1 {
            tf_chan = 0;
        }
        // Put the prefiltered overlap back for the prefilter and the MDCT
        // (C run_prefilter: `OPUS_COPY(in, st->in_mem, overlap)`).
        for (channel, saved) in in_buf
            .chunks_exact_mut(buf_stride)
            .zip(self.in_mem.chunks_exact(overlap))
        {
            channel[..overlap].copy_from_slice(saved);
        }
        // Clamp toneishness as in C 2034: toneishness = min(toneishness, 1 - tf_estimate)
        // For float, tf_estimate is log-domain; clamp to [0,1]
        toneishness = toneishness.min((1.0 - tf_estimate).clamp(0.0, 1.0));

        // C 2037-2043: find the pitch period and gain. A new filter is only
        // looked for when the packet has room for more than its parameters:
        // above 12 bytes per channel (3 bytes for LFE), outside Hybrid, not
        // in a silent frame, and with 16 bits left for the flag and the
        // period.
        let pf_enabled = ((self.lfe && nb_available_bytes > 3)
            || nb_available_bytes > 12 * channels)
            && !hybrid
            && !silence
            && tell_c + 16 <= total_bits
            && !self.disable_pf;
        // C 2043: the tapset this frame's comb filter uses and signals. The
        // spreading decision further down picks `tapset_decision` for the
        // *next* frame, so the value is taken here.
        let prefilter_tapset = self.tapset_decision;
        // What `run_prefilter` does with the frame: search for a new filter
        // (a tone at any complexity, a pitch from complexity 5 up), or fade
        // out the filter of the frame before.
        let pf_search = pf_enabled && (toneishness > 0.99 || self.complexity >= 5);
        let (pf_on, gain1, pitch_index) = if pf_search || self.prefilter_gain != 0.0 {
            run_prefilter(
                in_buf,
                &mut self.prefilter_buf,
                self.prefilter_period,
                self.prefilter_gain,
                self.prefilter_tapset,
                prefilter_tapset,
                mode.window,
                cc,
                frame_size,
                overlap,
                &mut self.w_prefilter_pitch_buf,
                &mut self.w_prefilter_before,
                &mut self.w_prefilter_after,
                self.loss_rate,
                tf_estimate,
                nb_available_bytes,
                tone_freq,
                toneishness,
                self.complexity,
                pf_enabled,
                self.analysis.valid.then_some(self.analysis.max_pitch_ratio),
            )
        } else {
            // Nothing to search for and nothing to fade out: `run_prefilter`
            // would leave the frame as it is. Its history still has to stay
            // current for the analysis head above and for the pitch search
            // when the comb filter turns back on.
            for c in 0..cc {
                let body = c * buf_stride + overlap;
                push_prefilter_history(
                    &mut self.prefilter_buf[c * PREFILTER_STRIDE..][..max_period],
                    &in_buf[body..body + frame_size],
                );
            }
            (false, 0.0f32, COMBFILTER_MINPERIOD)
        };
        // C 2042-2044: the pitch period jumped by more than a quarter while
        // the filter of this or the last frame was strong. Only the VBR
        // target reads it, and only with a valid analysis. libopus compares
        // with the old period after `run_prefilter` raised it to at least
        // `COMBFILTER_MINPERIOD`, and in double precision.
        let pitch_change = {
            let old_period = self.prefilter_period.max(COMBFILTER_MINPERIOD) as f64;
            let pitch = pitch_index as f64;
            (gain1 > 0.4 || self.prefilter_gain > 0.4)
                && (!self.analysis.valid || self.analysis.tonality as f64 > 0.3)
                && (pitch > 1.26 * old_period || pitch < 0.79 * old_period)
        };

        // Save the prefiltered overlap for the next frame (C run_prefilter:
        // `OPUS_COPY(st->in_mem, in + N, overlap)`). These are the last
        // `overlap` samples of `in_buf`, which the prefilter changed in place.
        for (channel, saved) in in_buf
            .chunks_exact(buf_stride)
            .zip(self.in_mem.chunks_exact_mut(overlap))
        {
            saved.copy_from_slice(&channel[frame_size..]);
        }
        // Room for the spectra of all input channels: a mono stream of a
        // stereo input is mixed from both (`compute_mdcts`).
        let freq = &mut self.w_freq[..frame_size * cc];
        let band_e = &mut self.w_band_e[..nb_ebands * channels];
        let band_log_e = &mut self.w_band_log_e[..nb_ebands * channels];
        let x_pad_end = (frame_size * channels + STRIDE_ACCESS_PAD).min(self.w_x.len());
        let x = &mut self.w_x[..x_pad_end];

        self.w_error[..nb_ebands * channels].fill(0.0);
        let error = &mut self.w_error[..nb_ebands * channels];

        if start_band == 0 && !silence && rc.tell() + 16 <= total_bits {
            rc.encode_bit_logp(pf_on, 1);
            if pf_on {
                let qg = (gain1 / 0.09375 - 1.0 + 0.5).floor() as i32;
                let qg = qg.clamp(0, 7);
                let pi = (pitch_index + 1) as u32;
                let octave = 32 - pi.leading_zeros();
                let octave = (octave as i32 - 5).max(0) as u32;
                rc.enc_uint(octave, 6);
                rc.enc_bits(pi - (16 << octave), 4 + octave);
                rc.enc_bits(qg as u32, 3);
                rc.encode_icdf(prefilter_tapset, &TAPSET_ICDF, 2);
            }
        }

        // C 2068-2075: the frame is coded with short blocks if it is a
        // transient and the packet has room for the flag. The flag itself
        // is written after the last check below (C 2234).
        let mut short_blocks = false;
        // C `transient_got_disabled`: the frame could not signal a
        // transient. It still counts as one for `consec_transient`.
        let mut transient_got_disabled = false;
        if lm > 0 && rc.tell() + 3 <= total_bits {
            short_blocks = is_transient;
        } else {
            // A frame that cannot signal a transient is not one. 2.5 ms
            // frames have no transient flag, and a packet that is already
            // full has no room for it. The decoder reads every later symbol
            // of such a frame as non-transient.
            is_transient = false;
            transient_got_disabled = true;
        }
        let second_mdct = short_blocks && self.complexity >= 8;
        // C `bandLogE2`: the energies from long blocks, which the dynamic
        // allocation compares with those of the frame before.
        let band_log_e2 = &mut self.w_band_log_e2[..nb_ebands * channels];
        if second_mdct {
            compute_mdcts(
                mode, false, in_buf, buf_stride, freq, frame_size, channels, cc, lm, upsample,
            );
            let band_e2 = &mut self.w_band_e2[..nb_ebands * channels];
            compute_band_energies(mode, freq, band_e2, end, channels, lm);
            amp2log2_enc(mode, end, band_e2, band_log_e2, channels);
            for c in 0..channels {
                for i in 0..end {
                    band_log_e2[c * nb_ebands + i] += lm as f32 * 0.5;
                }
            }
        }
        compute_mdcts(
            mode, short_blocks, in_buf, buf_stride, freq, frame_size, channels, cc, lm, upsample,
        );
        compute_band_energies(mode, freq, band_e, end, channels, lm);
        if self.lfe {
            // C 2101-2108: everything of interest in an LFE channel is in
            // bands 0 and 1. The other bands are held 80 dB below band 0,
            // so they cost next to nothing. An LFE frame is never
            // transient.
            let limit = 1e-4 * band_e[0];
            for e in band_e.iter_mut().take(end).skip(2) {
                *e = e.min(limit).max(1e-15);
            }
        }
        // C 2109: the energies of all bands, also of those below the first
        // coded band of a Hybrid frame. The dynamic allocation looks at
        // them.
        amp2log2_enc(mode, end, band_e, band_log_e, channels);

        // C 2111-2188: with an energy mask, find out how much the other
        // surround channels mask this stream. Not for Hybrid and not for LFE.
        let mut surround_dynalloc = [0.0f32; CELT_NB_EBANDS];
        let (surround_masking, surround_trim) = match &self.energy_mask {
            Some(energy_mask) if !hybrid && !self.lfe => surround_masking_analysis(
                mode,
                energy_mask,
                channels,
                self.last_coded_bands,
                &mut surround_dynalloc,
            ),
            _ => (0.0, 0.0),
        };

        // C 2189-2206: temporal VBR (not for LFE). A frame whose spectrum
        // is above the slow average `spec_avg` gets more bits, a quieter
        // one fewer.
        let mut temporal_vbr = 0.0f32;
        if !self.lfe && end > start_band {
            let mut follow = -10.0f32;
            let mut frame_avg = 0.0f32;
            let offset = if short_blocks { 0.5 * lm as f32 } else { 0.0 };
            for i in start_band..end {
                follow = (follow - 1.0).max(band_log_e[i] - offset);
                if channels == 2 {
                    follow = follow.max(band_log_e[i + nb_ebands] - offset);
                }
                frame_avg += follow;
            }
            frame_avg /= (end - start_band) as f32;
            temporal_vbr = (frame_avg - self.spec_avg).clamp(-1.5, 3.0);
            self.spec_avg += 0.02 * temporal_vbr;
        }

        if !second_mdct {
            band_log_e2.copy_from_slice(band_log_e);
        }

        // C 2215-2232: "Last chance to catch any transient we might have
        // missed in the time-domain analysis". The frame is coded again
        // with short blocks when its energy jumped up from the frame
        // before.
        if lm > 0
            && rc.tell() + 3 <= total_bits
            && !is_transient
            && self.complexity >= 5
            && !self.lfe
            && !hybrid
            && patch_transient_decision(
                band_log_e,
                &self.old_band_e,
                nb_ebands,
                start_band,
                end,
                channels,
            )
        {
            is_transient = true;
            short_blocks = true;
            compute_mdcts(
                mode, true, in_buf, buf_stride, freq, frame_size, channels, cc, lm, upsample,
            );
            compute_band_energies(mode, freq, band_e, end, channels, lm);
            amp2log2_enc(mode, end, band_e, band_log_e, channels);
            // "Compensate for the scaling of short vs long mdcts"
            for c in 0..channels {
                for i in 0..end {
                    band_log_e2[c * nb_ebands + i] += lm as f32 * 0.5;
                }
            }
            tf_estimate = 0.2;
        }

        if lm > 0 && rc.tell() + 3 <= total_bits {
            rc.encode_bit_logp(is_transient, 3);
        }

        normalise_bands(
            mode,
            freq,
            x,
            band_e,
            end,
            channels,
            (1 << lm) as usize,
        );

        // C 2244: the time-frequency resolution of each band is analysed
        // except for Hybrid, LFE, tones and very low bitrates.
        let enable_tf_analysis = effective_bytes >= 15 * channels
            && !hybrid
            && self.complexity >= 2
            && !self.lfe
            && toneishness < 0.98;

        // C 2250: the dynamic allocation looks at the energies of the frame
        // before, so it runs before the coarse energy is quantized.
        self.w_offsets[..nb_ebands].fill(0);
        let offsets = &mut self.w_offsets[..nb_ebands];
        let mut importance = [13i32; MAX_NB_EBANDS];
        let mut spread_weight = [0i32; MAX_NB_EBANDS];
        let dynalloc = dynalloc_analysis(
            mode,
            band_log_e,
            band_log_e2,
            &self.old_band_e,
            start_band,
            end,
            channels,
            offsets,
            self.lsb_depth,
            is_transient,
            self.vbr,
            self.constrained_vbr,
            lm,
            effective_bytes as i32,
            self.lfe,
            &surround_dynalloc,
            &mut importance,
            &mut spread_weight,
            tone_freq,
            toneishness,
            self.analysis.valid.then_some(&self.analysis.leak_boost),
        );

        self.w_tf_res[..nb_ebands].fill(0);
        let tf_res = &mut self.w_tf_res[..nb_ebands];
        // C 2254-2281: "Disable variable tf resolution for hybrid and at
        // very low bitrate".
        let tf_select = if enable_tf_analysis {
            let lambda = 80.max(20480 / effective_bytes.max(1) + 2) as i32;
            tf_analysis(
                mode,
                end,
                is_transient,
                tf_res,
                lambda,
                x,
                frame_size,
                lm as i32,
                tf_estimate,
                tf_chan,
                &importance,
            )
        } else if hybrid && weak_transient {
            // A weak transient: long blocks with the better time resolution.
            tf_res[..end].fill(1);
            0
        } else if hybrid_low_rate {
            // Low-bitrate Hybrid: 5 ms time resolution rather than 2.5 ms.
            tf_res[..end].fill(0);
            is_transient as i32
        } else {
            tf_res[..end].fill(is_transient as i32);
            0
        };

        // C 2283-2297: "When the energy is stable, slightly bias energy
        // quantization towards the previous error to make the gain more
        // stable (a constant offset is better than fluctuations)."
        for c in 0..channels {
            for i in start_band..end {
                let k = i + c * nb_ebands;
                if (band_log_e[k] - self.old_band_e[k]).abs() < 2.0 {
                    band_log_e[k] -= 0.25 * self.energy_error[k];
                }
            }
        }
        quant_coarse_energy_advanced(
            mode,
            start_band,
            end,
            end,
            band_log_e,
            &mut self.old_band_e,
            total_bits as u32,
            error,
            rc,
            channels,
            lm,
            nb_available_bytes,
            self.force_intra,
            &mut self.delayed_intra,
            self.complexity >= 4,
            self.loss_rate,
            self.lfe,
        );

        // C 2360: the caps of the bands (init_caps).
        self.w_cap[..nb_ebands].fill(0);
        let cap = &mut self.w_cap[..nb_ebands];
        for (i, cap_i) in cap.iter_mut().enumerate() {
            let n = (mode.e_bands[i + 1] - mode.e_bands[i]) << lm;
            *cap_i = ((mode.cache.caps[nb_ebands * (2 * lm + channels - 1) + i] as i32 + 64)
                * channels as i32
                * n as i32)
                >> 2;
        }

        tf_encode(
            start_band,
            end,
            is_transient,
            tf_res,
            lm as i32,
            tf_select,
            rc,
        );

        // C 2401-2403: "Always use MS for 2.5 ms frames until we can do a
        // better analysis".
        let mut dual_stereo_val = if channels == 2 && lm != 0 {
            stereo_analysis(mode, x, lm as i32, frame_size) as i32
        } else {
            0
        };

        if channels == 2 {
            self.intensity = hysteresis_decision(
                equiv_rate / 1000,
                &INTEN_THRESHOLDS,
                &INTEN_HYSTERESIS,
                self.intensity,
            );
            // C 2408: `st->intensity = IMIN(end, IMAX(start, st->intensity))`.
            // The lower limit is needed for Hybrid, where the first coded
            // band is 17. The allocator writes the intensity band as its
            // distance from `start`. A band below `start` would be written
            // as a negative distance, and the decoder would read the rest
            // of the frame as noise.
            self.intensity = self.intensity.max(start_band as i32).min(end as i32);
        }

        // C 2305-2351: the spreading decision, coded when the packet has
        // room for it. Without room it is the normal one, at every
        // complexity. The next frame's decision starts from the value kept
        // here.
        if rc.tell() + 4 <= total_bits {
            if self.lfe {
                // C 2307-2310: LFE always uses normal spreading and the
                // first comb-filter tap shape.
                self.tapset_decision = 0;
                self.spread_decision = SPREAD_NORMAL;
            } else if hybrid {
                // C 2311-2318: Hybrid spreads its two or four bands hard,
                // except on transients.
                self.spread_decision = if self.complexity == 0 {
                    SPREAD_NONE
                } else if short_blocks {
                    SPREAD_NORMAL
                } else {
                    crate::bands::SPREAD_AGGRESSIVE
                };
            } else if short_blocks || self.complexity < 3 || nb_available_bytes < 10 * channels {
                self.spread_decision = if self.complexity == 0 {
                    SPREAD_NONE
                } else {
                    SPREAD_NORMAL
                };
            } else {
                // C 2343: the high-frequency average behind the tapset
                // decision is only updated on long blocks that use the
                // comb filter.
                let update_hf = pf_on && !short_blocks;
                self.spread_decision = spreading_decision(
                    mode,
                    x,
                    &mut self.tonal_average,
                    self.spread_decision,
                    &mut self.hf_average,
                    &mut self.tapset_decision,
                    update_hf,
                    end,
                    channels,
                    (1 << lm) as usize,
                    &spread_weight,
                );
            }
            rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
        } else {
            self.spread_decision = SPREAD_NORMAL;
        }

        // C 2353-2355: for LFE, everything interesting is in the first band.
        if self.lfe {
            offsets[0] = (effective_bytes / 3).min(8) as i32;
        }

        let mut dynalloc_logp = 6i32;
        let total_bits_bitres = total_bits << BITRES;
        let mut total_boost = 0i32;
        let mut tell_frac = rc.tell_frac();

        for i in start_band..end {
            let width =
                channels as i32 * (mode.e_bands[i + 1] - mode.e_bands[i]) as i32 * (1 << lm);
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            let mut j = 0i32;

            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_bitres - total_boost
                && boost < cap[i]
            {
                let flag = j < offsets[i];
                rc.encode_bit_logp(flag, dynalloc_loop_logp as u32);
                tell_frac = rc.tell_frac();
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
                j += 1;
            }

            if j > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
            offsets[i] = boost;
        }

        // C 2385-2395: the allocation trim is 5 unless the packet has room to
        // signal another value. A decoder that finds no room assumes 5, so
        // an encoder that goes on with its own value allocates the bands
        // differently and the rest of the frame decodes as noise.
        let mut alloc_trim = 5;
        if rc.tell_frac() + (6 << BITRES) <= total_bits_bitres - total_boost {
            if hybrid || self.lfe {
                // C 2414-2418: Hybrid and LFE keep the neutral trim of 5,
                // and have no stereo saving.
                self.stereo_saving = 0.0;
            } else {
                alloc_trim = alloc_trim_analysis(
                    mode,
                    x,
                    band_log_e,
                    end,
                    lm as i32,
                    channels,
                    frame_size,
                    &mut self.stereo_saving,
                    tf_estimate,
                    self.intensity,
                    surround_trim,
                    equiv_rate,
                    self.analysis.valid.then_some(self.analysis.tonality_slope),
                );
            }
            rc.encode_icdf(alloc_trim, &TRIM_ICDF, 7);
        }
        // C 2427-2534: under VBR, the frame takes the bytes its target
        // calls for.
        if let Some(vbr_rate) = vbr_rate {
            let frame = VbrFrame {
                mode,
                lm,
                channels,
                hybrid,
                constrained_vbr: self.constrained_vbr,
                lfe: self.lfe,
                has_surround_mask: self.energy_mask.is_some(),
                silk_offset: self.silk_offset,
                last_coded_bands: self.last_coded_bands,
                intensity: self.intensity,
                stereo_saving: self.stereo_saving,
                equiv_rate,
                tell0_frac: tell_initial_frac,
                total_boost,
                dynalloc,
                tf_estimate,
                surround_masking,
                temporal_vbr,
                silence,
                analysis: self.analysis,
                pitch_change,
            };
            nb_compressed_bytes =
                self.vbr_state
                    .frame_bytes(&frame, vbr_rate, rc.tell_frac(), nb_compressed_bytes);
            total_bits = (nb_compressed_bytes * 8) as i32;
            rc.shrink(nb_compressed_bytes as u32);
        }

        // C 2632: the allocator lowers `st->intensity` to the coded bands,
        // or sets it to 0 when the packet has no room to signal it. The next
        // frame's hysteresis starts from that value.
        self.w_pulses[..nb_ebands].fill(0);
        let pulses = &mut self.w_pulses[..nb_ebands];

        let stereo = channels > 1;
        let ebands_stereo = if stereo {
            nb_ebands * channels
        } else {
            nb_ebands
        };
        self.w_fine_priority[..ebands_stereo].fill(0);
        let fine_priority = &mut self.w_fine_priority[..ebands_stereo];
        self.w_ebits[..ebands_stereo].fill(0);
        let ebits = &mut self.w_ebits[..ebands_stereo];
        let mut balance = 0;

        // C: bits = packet_bits - tell - 1; then subtract anti_collapse_rsv
        // (celt_encoder.c:2614-2618) BEFORE the allocation.
        let mut alloc_bits = (total_bits << BITRES) - rc.tell_frac() - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 {
            if alloc_bits >= ((lm as i32 + 2) << BITRES) {
                1i32 << BITRES
            } else {
                0
            }
        } else {
            0
        };
        alloc_bits -= anti_collapse_rsv;

        // C 2603-2624: `signalBandwidth`, the last band the allocator may
        // keep when it skips bands from the top: `end - 1`. With a valid
        // analysis, the bandwidth the analysis found, but at least a band
        // that grows with the rate per coded channel. LFE keeps bands 0 and
        // 1 only.
        let mut signal_bandwidth = end as i32 - 1;
        if self.analysis.valid {
            let c = channels as i32;
            let min_bandwidth = if equiv_rate < 32000 * c {
                13
            } else if equiv_rate < 48000 * c {
                16
            } else if equiv_rate < 60000 * c {
                18
            } else if equiv_rate < 80000 * c {
                19
            } else {
                20
            };
            signal_bandwidth = self.analysis.bandwidth.max(min_bandwidth);
        }
        if self.lfe {
            signal_bandwidth = 1;
        }
        let coded_bands = clt_compute_allocation(
            mode,
            start_band,
            end,
            offsets,
            cap,
            alloc_trim,
            &mut self.intensity,
            &mut dual_stereo_val,
            alloc_bits,
            &mut balance,
            pulses,
            ebits,
            fine_priority,
            channels as i32,
            lm as i32,
            rc,
            true,
            // The bands coded in the frames before: bands below it are
            // kept with a lower threshold, so that they do not go in and
            // out from frame to frame.
            self.last_coded_bands,
            signal_bandwidth,
        );
        // C 2630-2633: the number of coded bands moves by at most one band
        // per frame.
        self.last_coded_bands = if self.last_coded_bands != 0 {
            (self.last_coded_bands + 1).min((self.last_coded_bands - 1).max(coded_bands))
        } else {
            coded_bands
        };

        quant_fine_energy(
            mode,
            start_band,
            end,
            &mut self.old_band_e,
            error,
            ebits,
            rc,
            channels,
        );
        self.energy_error.fill(0.0);

        self.w_collapse_masks[..nb_ebands * channels].fill(0);
        let collapse_masks = &mut self.w_collapse_masks[..nb_ebands * channels];
        let (x_split, y_split) = x.split_at_mut(frame_size);
        let y_opt = if channels == 2 { Some(y_split) } else { None };

        let mut dual_stereo = dual_stereo_val != 0;

        // C 2674-2679. At complexity 8 and above `quant_all_bands_enc`
        // resynthesises the bands for its second pass per stereo band
        // (`theta_rdo`). The seed is `st->rng`.
        let mut seed = self.rng;
        crate::bands::quant_all_bands_enc_with(
            mode,
            start_band,
            end,
            x_split,
            y_opt,
            collapse_masks,
            band_e,
            pulses,
            short_blocks,
            self.spread_decision,
            &mut dual_stereo,
            self.intensity as usize,
            tf_res,
            (total_bits << BITRES) - anti_collapse_rsv,
            &mut balance,
            rc,
            lm as i32,
            coded_bands,
            &mut seed,
            self.complexity,
            self.disable_inv,
            &mut self.rotations,
        );

        if anti_collapse_rsv > 0 {
            let anti_collapse_on = if self.consec_transient < 2 {
                1u32
            } else {
                0u32
            };
            rc.enc_bits(anti_collapse_on, 1);
        }

        quant_energy_finalise(
            mode,
            start_band,
            end,
            &mut self.old_band_e,
            error,
            ebits,
            fine_priority,
            total_bits - rc.tell(),
            rc,
            channels,
        );
        // C 2729-2735: keep what the quantizer left over for the next frame.
        for c in 0..channels {
            for i in start_band..end {
                let k = i + c * nb_ebands;
                self.energy_error[k] = error[k].clamp(-0.5, 0.5);
            }
        }

        if silence {
            for v in self.old_band_e[..channels * nb_ebands].iter_mut() {
                *v = -28.0;
            }
        }

        // C 2776-2778: a mono stream of a stereo input leaves its energies
        // as the history of both channels.
        if cc == 2 && channels == 1 {
            self.old_band_e.copy_within(0..nb_ebands, nb_ebands);
        }

        self.last_band_log_e.copy_from_slice(&self.old_band_e);

        if !is_transient {
            self.old_band_e3.copy_from_slice(&self.old_band_e2);
            self.old_band_e2.copy_from_slice(&self.old_band_e);
        } else {
            for i in 0..cc * nb_ebands {
                self.old_band_e2[i] = self.old_band_e2[i].min(self.old_band_e[i]);
            }
        }
        // C 2789-2802: "In case start or end were to change". The bands
        // this frame did not code have no energy the next frame could
        // predict from.
        for c in 0..cc {
            let base = c * nb_ebands;
            for i in (0..start_band).chain(end..nb_ebands) {
                self.old_band_e[base + i] = 0.0;
                self.old_band_e2[base + i] = OLD_LOG_E_RESET;
                self.old_band_e3[base + i] = OLD_LOG_E_RESET;
            }
        }

        // C 2815-2817: room left in the packet is not written to;
        // `RangeCoder::done` clears it. Do not pad it with zero bits here:
        // that moves raw bits that are not written out yet away from the
        // end of the packet, and the decoder reads zeros in their place.

        // C 2772: remember the tapset this frame was filtered with. The
        // decoder cross-fades from it at the start of the next frame, so it
        // must not be the `tapset_decision` just updated for the next frame.
        // The period is kept even when the filter is off (C 2770). The next
        // frame then takes up a pitch that continues it more readily
        // (run_prefilter: `pf_threshold`, and `remove_doubling`).
        self.prefilter_period = pitch_index;
        self.prefilter_gain = gain1;
        self.prefilter_tapset = prefilter_tapset;

        // C 2809-2812: every 2.5 ms frame, and every frame without room for
        // the transient flag, counts as a transient here. After two such
        // frames the next transient frame has no anti-collapse.
        // A long stream of 2.5 ms frames would overflow a plain count.
        if is_transient || transient_got_disabled {
            self.consec_transient = self.consec_transient.saturating_add(1);
        } else {
            self.consec_transient = 0;
        }
        // C 2813: `st->rng = enc->rng`.
        self.rng = rc.rng;
    }

    /// C 1904-1961: the bytes the frame may take (C `nbCompressedBytes`)
    /// and the bytes it is expected to take (C `effectiveBytes`).
    ///
    /// `packet_bytes` is the room in `rc`, and `tell` where the coder
    /// stands. Under VBR the frame is expected to take its target rate
    /// (`vbr_rate`, in 1/8 bit), not the largest packet the caller allows.
    /// Under CBR it is expected to take what the packet has left.
    ///
    /// Under constrained VBR a frame may take at most twice the target
    /// rate, less what earlier frames took too much, and `rc` is shrunk to
    /// that. libopus applies this bound first, "so that bust-prevention
    /// logic triggers correctly if we don't have enough bits".
    fn frame_bytes(
        &self,
        rc: &mut RangeCoder,
        vbr_rate: Option<i32>,
        packet_bytes: usize,
        tell: i32,
    ) -> (usize, usize) {
        let nb_filled_bytes = ((tell + 4) >> 3).max(0) as usize;
        let nb_available = packet_bytes.saturating_sub(nb_filled_bytes);
        let effective_bytes = match vbr_rate {
            Some(rate) => (rate >> (BITRES + 3)).max(0) as usize,
            None => nb_available,
        };
        let constrained_rate = vbr_rate.filter(|_| self.constrained_vbr);
        let max_allowed = constrained_vbr_bytes(
            constrained_rate,
            self.vbr_state.reservoir,
            tell,
            nb_available,
        );
        if max_allowed < nb_available {
            let nb_compressed_bytes = nb_filled_bytes + max_allowed;
            rc.shrink(nb_compressed_bytes as u32);
            (nb_compressed_bytes, effective_bytes)
        } else {
            (packet_bytes, effective_bytes)
        }
    }
}

/// Log-energy a band's history starts from and returns to when the band is
/// not coded (libopus celt_decoder.c: `-GCONST(28.f)`).
const OLD_LOG_E_RESET: f32 = -28.0;

/// Longest pitch period the pitch-based concealment searches for, in samples
/// at 48 kHz (libopus `PLC_PITCH_LAG_MAX`, a pitch of 66.67 Hz).
const PLC_PITCH_LAG_MAX: usize = 720;
/// Shortest pitch period the pitch-based concealment searches for (libopus
/// `PLC_PITCH_LAG_MIN`, a pitch of 480 Hz).
const PLC_PITCH_LAG_MIN: usize = 100;
/// Order of the linear predictor the pitch-based concealment uses (libopus
/// `CELT_LPC_ORDER`).
const PLC_LPC_ORDER: usize = 24;
/// Number of past samples the pitch-based concealment analyses (libopus
/// `MAX_PERIOD`).
const PLC_MAX_PERIOD: usize = 1024;
/// Longest CELT frame in bits that the decoder accepts: what a range coder
/// of this crate can hold. libopus stops at 1275 bytes
/// (`celt_decode_with_ec`); the Opus packet parser enforces that limit.
/// `CeltEncoder` and its tests also work with longer frames, so the CELT
/// decoder keeps accepting them.
const MAX_CELT_FRAME_BITS: i32 = (crate::range_coder::RANGE_BUF_MAX * 8) as i32;

/// What the previous call to the decoder produced (libopus `FRAME_NONE`,
/// `FRAME_NORMAL`, `FRAME_PLC_NOISE`, `FRAME_PLC_PERIODIC` and
/// `FRAME_PLC_NEURAL`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameType {
    /// Nothing yet since the decoder was created or reset.
    None,
    /// A decoded frame.
    Normal,
    /// A lost frame, concealed with shaped noise.
    PlcNoise,
    /// A lost frame, concealed by repeating the last pitch period.
    PlcPeriodic,
    /// A lost frame, concealed with deep PLC (feature `deep-plc`).
    #[cfg(feature = "deep-plc")]
    PlcNeural,
    /// A lost frame, concealed with deep PLC from DRED features (feature
    /// `dred`).
    #[cfg(feature = "dred")]
    Dred,
}

/// The samples of `plc_pcm` (libopus `PLC_UPDATE_SAMPLES`): four frames of
/// 10 ms at 16 kHz.
#[cfg(feature = "deep-plc")]
const PLC_UPDATE_SAMPLES: usize = 4 * crate::dnn::plc::LPCNET_FRAME_SIZE;

/// Constant part of the in-place comb filter, for `n` samples from `at` of
/// each channel in `bufs`:
/// `buf[i] += g[0] * buf[i - t] + g[1] * (buf[i - t + 1] + buf[i - t - 1])
/// + g[2] * (buf[i - t + 2] + buf[i - t - 2])`.
///
/// The samples it reads at `i - t` and around it may be ones it has just
/// written. `t` must be at least `COMBFILTER_MINPERIOD` and at most
/// `at - 2`. All channels use the same period and taps.
#[inline]
fn comb_inplace_const<const C: usize>(
    bufs: &mut [&mut [f32]; C],
    at: usize,
    t: usize,
    n: usize,
    g: [f32; 3],
) {
    // Two channels take the version that loads each sample once. See
    // `comb_inplace_const_window_neon` for why one channel does not.
    #[cfg(target_arch = "aarch64")]
    if C == 2 {
        comb_inplace_const_window_neon::<C, 2>(bufs, at, t, n, g);
    } else {
        comb_inplace_const_neon::<C, COMB_VECTORS>(bufs, at, t, n, g);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for buf in bufs.iter_mut() {
        comb_inplace_const_scalar(buf, at, t, n, g);
    }
}

/// Portable version of `comb_inplace_const` for one channel, one sample at
/// a time.
fn comb_inplace_const_scalar(buf: &mut [f32], at: usize, t: usize, n: usize, g: [f32; 3]) {
    for idx in at..at + n {
        let s = buf[idx];
        let r = buf[idx - t];
        let rp1 = buf[idx - t + 1];
        let rm1 = buf[idx - t - 1];
        let rp2 = buf[idx - t + 2];
        let rm2 = buf[idx - t - 2];
        buf[idx] = comb_tap3(s, g, r, rp1 + rm1, rp2 + rm2);
    }
}

/// Vectors of four samples that `comb_inplace_const_neon` and
/// `comb_inplace_fade_neon` compute per channel in one step. On the
/// Cortex-A55 and the Cortex-A76, 3 took less time than 1 or 2.
#[cfg(target_arch = "aarch64")]
const COMB_VECTORS: usize = 3;

/// `comb_inplace_const` with vector instructions. One step filters `V`
/// vectors of four samples of every channel.
///
/// The newest sample a step reads is `buf[i + 4 * V - 1 - t + 2]`. With
/// `4 * V + 2 <= t` that is before `buf[i]`, the first sample the step
/// writes. So the samples of a step do not depend on each other, and the
/// step loads all its input before it stores. The vectors of a step go
/// through each operation one after the other. A processor that runs
/// instructions in program order can then start the next vector's
/// operation while the previous one is still running.
///
/// Each sample takes the steps of `comb_tap3` in its order, each product
/// added with a fused multiply-add, so the result is the same as the
/// portable version's.
#[cfg(target_arch = "aarch64")]
fn comb_inplace_const_neon<const C: usize, const V: usize>(
    bufs: &mut [&mut [f32]; C],
    at: usize,
    t: usize,
    n: usize,
    g: [f32; 3],
) {
    const { assert!(V >= 1 && 4 * V + 2 <= COMBFILTER_MINPERIOD) };
    // `t` comes from the packet, so this is checked in every build.
    assert!(t >= COMBFILTER_MINPERIOD && at >= t + 2);
    for buf in bufs.iter() {
        assert!(at + n <= buf.len());
    }
    let step = 4 * V;
    let main = n - n % step;
    let ptr: [*mut f32; C] = core::array::from_fn(|c| bufs[c].as_mut_ptr());
    // SAFETY: for channel `c`, every step reads `buf[i - t - 2..i - t + 4 * V
    // + 2]` and reads and writes `buf[i..i + 4 * V]`, for `i` from `at` in
    // steps of `4 * V` while `i + 4 * V <= at + main`. The checks above give
    // `i - t - 2 >= 0` and `i + 4 * V <= buf.len()`, and `4 * V + 2 <= t`
    // puts the range that is read below the one that is written. Each
    // channel is accessed only through the pointer taken from its own
    // `&mut` slice, and two `&mut` slices do not overlap.
    unsafe {
        let gv = [vdupq_n_f32(g[0]), vdupq_n_f32(g[1]), vdupq_n_f32(g[2])];
        let zero = vdupq_n_f32(0.0);
        let mut i = at;
        while i < at + main {
            let mut acc = [[zero; V]; C];
            let mut near = [[zero; V]; C];
            let mut far = [[zero; V]; C];
            for c in 0..C {
                for v in 0..V {
                    let y = ptr[c].add(i + 4 * v);
                    let r = y.sub(t);
                    near[c][v] = vaddq_f32(vld1q_f32(r.add(1)), vld1q_f32(r.sub(1)));
                    far[c][v] = vaddq_f32(vld1q_f32(r.add(2)), vld1q_f32(r.sub(2)));
                    acc[c][v] = vfmaq_f32(vld1q_f32(y), gv[0], vld1q_f32(r));
                }
            }
            for c in 0..C {
                for v in 0..V {
                    acc[c][v] = vfmaq_f32(acc[c][v], gv[1], near[c][v]);
                }
            }
            for c in 0..C {
                for v in 0..V {
                    acc[c][v] = vfmaq_f32(acc[c][v], gv[2], far[c][v]);
                }
            }
            for c in 0..C {
                for v in 0..V {
                    vst1q_f32(ptr[c].add(i + 4 * v), acc[c][v]);
                }
            }
            i += step;
        }
    }
    for buf in bufs.iter_mut() {
        comb_inplace_const_scalar(buf, at + main, t, n - main, g);
    }
}

/// `comb_inplace_const` with vector instructions that load each sample of
/// the taps once. One step filters `V` vectors of four samples of every
/// channel.
///
/// The taps of the samples `i..i + 4` are `buf[i - t - 2..i - t + 6]`.
/// With `r = i - t`, the kernel holds the vectors `buf[r - 4..r]`,
/// `buf[r..r + 4]` and `buf[r + 4..r + 8]` in registers. It takes the taps
/// one and two places from the centre out of them with `vextq_f32`. A step
/// loads the vectors that follow the last one it has, up to
/// `buf[r + 4 * V + 4]` with `r = i - t`, exclusive. With `4 * V + 4 <= t`
/// that is before `buf[i]`, the first sample the step writes. Every sample
/// before `buf[i]` has its final value: it is history or it was written by
/// an earlier step, and it is not written again. So the vectors that a
/// step keeps for the next one stay valid.
///
/// Per vector of output this takes 2 loads and 9 vector operations, where
/// `comb_inplace_const_neon` takes 6 loads and 5 vector operations. The
/// Cortex-A55 loads slowly: for the two channels of a 20 ms frame this
/// version took 0.67 of the time of the other one there, and 1.38 times
/// its time on the Cortex-A76. For one channel it was slower than the
/// version before both kernels on the Cortex-A76, so one channel takes
/// `comb_inplace_const_neon`.
///
/// Each sample takes the steps of `comb_tap3` in its order, each product
/// added with a fused multiply-add, so the result is the same as the
/// portable version's.
#[cfg(target_arch = "aarch64")]
fn comb_inplace_const_window_neon<const C: usize, const V: usize>(
    bufs: &mut [&mut [f32]; C],
    at: usize,
    t: usize,
    n: usize,
    g: [f32; 3],
) {
    const { assert!(V >= 1 && 4 * V + 4 <= COMBFILTER_MINPERIOD) };
    // `t` comes from the packet, so this is checked in every build.
    assert!(t >= COMBFILTER_MINPERIOD && at >= t + 2);
    for buf in bufs.iter() {
        assert!(at + n <= buf.len());
    }
    let step = 4 * V;
    let main = n - n % step;
    let ptr: [*mut f32; C] = core::array::from_fn(|c| bufs[c].as_mut_ptr());
    // SAFETY: for channel `c`, the first loads read `buf[at - t - 2..at - t
    // + 4]`, and `at >= t + 2`. A step at `i` reads `buf[i - t + 4..i - t +
    // 4 * V + 4]` and reads and writes `buf[i..i + 4 * V]`, for `i` from
    // `at` in steps of `4 * V` while `i + 4 * V <= at + main`. So every
    // access lies in `buf[0..at + n]`, which the checks above put inside
    // the slice. `4 * V + 4 <= t` puts each read below the samples the
    // step writes. Each channel is accessed only through the pointer taken
    // from its own `&mut` slice, and two `&mut` slices do not overlap.
    unsafe {
        let gv = [vdupq_n_f32(g[0]), vdupq_n_f32(g[1]), vdupq_n_f32(g[2])];
        let zero = vdupq_n_f32(0.0);
        // `prev` is `buf[r - 4..r]` and `cur` is `buf[r..r + 4]`. Only the
        // last two lanes of `prev` are used, so only they are loaded.
        let mut prev: [float32x4_t; C] = core::array::from_fn(|c| {
            vcombine_f32(vdup_n_f32(0.0), vld1_f32(ptr[c].add(at - t - 2)))
        });
        let mut cur: [float32x4_t; C] = core::array::from_fn(|c| vld1q_f32(ptr[c].add(at - t)));
        let mut i = at;
        while i < at + main {
            // `next[c][v]` is `buf[r + 4 * v + 4..r + 4 * v + 8]`.
            let mut next = [[zero; V]; C];
            let mut acc = [[zero; V]; C];
            for c in 0..C {
                for v in 0..V {
                    next[c][v] = vld1q_f32(ptr[c].add(i - t + 4 + 4 * v));
                    acc[c][v] = vld1q_f32(ptr[c].add(i + 4 * v));
                }
            }
            let mut centre = [[zero; V]; C];
            let mut near = [[zero; V]; C];
            let mut far = [[zero; V]; C];
            for c in 0..C {
                for v in 0..V {
                    let before = match v {
                        0 => prev[c],
                        1 => cur[c],
                        _ => next[c][v - 2],
                    };
                    let mid = if v == 0 { cur[c] } else { next[c][v - 1] };
                    let after = next[c][v];
                    centre[c][v] = mid;
                    // `buf[k + 1] + buf[k - 1]` and `buf[k + 2] + buf[k - 2]`
                    // for the four samples `k` of `mid`.
                    let plus1 = vextq_f32::<1>(mid, after);
                    let minus1 = vextq_f32::<3>(before, mid);
                    near[c][v] = vaddq_f32(plus1, minus1);
                    let plus2 = vextq_f32::<2>(mid, after);
                    let minus2 = vextq_f32::<2>(before, mid);
                    far[c][v] = vaddq_f32(plus2, minus2);
                }
            }
            for c in 0..C {
                for v in 0..V {
                    acc[c][v] = vfmaq_f32(acc[c][v], gv[0], centre[c][v]);
                }
            }
            for c in 0..C {
                for v in 0..V {
                    acc[c][v] = vfmaq_f32(acc[c][v], gv[1], near[c][v]);
                }
            }
            for c in 0..C {
                for v in 0..V {
                    acc[c][v] = vfmaq_f32(acc[c][v], gv[2], far[c][v]);
                }
            }
            for c in 0..C {
                for v in 0..V {
                    vst1q_f32(ptr[c].add(i + 4 * v), acc[c][v]);
                }
                prev[c] = if V == 1 { cur[c] } else { next[c][V - 2] };
                cur[c] = next[c][V - 1];
            }
            i += step;
        }
    }
    for buf in bufs.iter_mut() {
        comb_inplace_const_scalar(buf, at + main, t, n - main, g);
    }
}

/// Cross-fading part of the in-place comb filter, for `n` samples from
/// `at` of each channel in `bufs`. The old filter (period `t0`, taps `g0`)
/// is weighted with `1 - window[i]^2` and the new one (period `t1`, taps
/// `g1`) with `window[i]^2`.
///
/// Both periods must be at least `COMBFILTER_MINPERIOD` and at most
/// `at - 2`. `window` must hold `n` values. All channels use the same
/// filters.
#[allow(clippy::too_many_arguments)]
#[inline]
fn comb_inplace_fade<const C: usize>(
    bufs: &mut [&mut [f32]; C],
    at: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: [f32; 3],
    g1: [f32; 3],
    window: &[f32],
) {
    #[cfg(target_arch = "aarch64")]
    comb_inplace_fade_neon::<C, COMB_VECTORS>(bufs, at, t0, t1, n, g0, g1, window);
    #[cfg(not(target_arch = "aarch64"))]
    for buf in bufs.iter_mut() {
        comb_inplace_fade_scalar(buf, at, t0, t1, n, g0, g1, window);
    }
}

/// Portable version of `comb_inplace_fade` for one channel, one sample at a
/// time.
#[allow(clippy::too_many_arguments)]
fn comb_inplace_fade_scalar(
    buf: &mut [f32],
    at: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: [f32; 3],
    g1: [f32; 3],
    window: &[f32],
) {
    for (i, &w) in window[..n].iter().enumerate() {
        let idx = at + i;
        let f = w * w;
        let s = buf[idx];
        let r0 = buf[idx - t0];
        let r0p1 = buf[idx - t0 + 1];
        let r0m1 = buf[idx - t0 - 1];
        let r0p2 = buf[idx - t0 + 2];
        let r0m2 = buf[idx - t0 - 2];
        let r1 = buf[idx - t1];
        let r1p1 = buf[idx - t1 + 1];
        let r1m1 = buf[idx - t1 - 1];
        let r1p2 = buf[idx - t1 + 2];
        let r1m2 = buf[idx - t1 - 2];
        buf[idx] = comb_fade_tap6(
            s,
            f,
            g0,
            [r0, r0p1 + r0m1, r0p2 + r0m2],
            g1,
            [r1, r1p1 + r1m1, r1p2 + r1m2],
        );
    }
}

/// `comb_inplace_fade` with vector instructions. One step filters `V`
/// vectors of four samples of every channel. See
/// `comb_inplace_const_neon` for why the samples of a step are
/// independent. The weights of the two filters depend on the position in
/// the window only, so the channels share them.
///
/// Each sample takes the steps of `comb_fade_tap6` in its order, so the
/// result is the same as the portable version's.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn comb_inplace_fade_neon<const C: usize, const V: usize>(
    bufs: &mut [&mut [f32]; C],
    at: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: [f32; 3],
    g1: [f32; 3],
    window: &[f32],
) {
    const { assert!(V >= 1 && 4 * V + 2 <= COMBFILTER_MINPERIOD) };
    // The periods come from the packet, so this is checked in every build.
    assert!(t0 >= COMBFILTER_MINPERIOD && t1 >= COMBFILTER_MINPERIOD);
    assert!(at >= t0 + 2 && at >= t1 + 2 && n <= window.len());
    for buf in bufs.iter() {
        assert!(at + n <= buf.len());
    }
    let step = 4 * V;
    let main = n - n % step;
    let ptr: [*mut f32; C] = core::array::from_fn(|c| bufs[c].as_mut_ptr());

    /// The three taps of one filter around `r`: `r[0]`, `r[1] + r[-1]` and
    /// `r[2] + r[-2]`.
    ///
    /// SAFETY: the caller guarantees that `r.sub(2)..r.add(6)` is readable.
    #[inline(always)]
    unsafe fn taps(r: *const f32) -> [float32x4_t; 3] {
        // SAFETY: see the function's contract.
        unsafe {
            [
                vld1q_f32(r),
                vaddq_f32(vld1q_f32(r.add(1)), vld1q_f32(r.sub(1))),
                vaddq_f32(vld1q_f32(r.add(2)), vld1q_f32(r.sub(2))),
            ]
        }
    }

    // SAFETY: every step reads `window[k..k + 4 * V]` with
    // `k + 4 * V <= main <= window.len()`. For channel `c` it reads
    // `buf[i - t - 2..i - t + 4 * V + 2]` for both periods and reads and
    // writes `buf[i..i + 4 * V]`, with `i = at + k`, while
    // `k + 4 * V <= main`. The checks above give `i - t - 2 >= 0` and
    // `i + 4 * V <= buf.len()`, and `4 * V + 2 <= t` puts the ranges that
    // are read below the one that is written. Each channel is accessed only
    // through the pointer taken from its own `&mut` slice.
    unsafe {
        let w = window.as_ptr();
        let old = [vdupq_n_f32(g0[0]), vdupq_n_f32(g0[1]), vdupq_n_f32(g0[2])];
        let new = [vdupq_n_f32(g1[0]), vdupq_n_f32(g1[1]), vdupq_n_f32(g1[2])];
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let mut k = 0;
        while k < main {
            // The weights of `comb_fade_tap6`: `(1 - f) * g0[j]` and
            // `f * g1[j]`, with `f = w * w`.
            let mut w_old = [[zero; 3]; V];
            let mut w_new = [[zero; 3]; V];
            for v in 0..V {
                let wk = vld1q_f32(w.add(k + 4 * v));
                let f = vmulq_f32(wk, wk);
                let one_minus_f = vsubq_f32(one, f);
                for j in 0..3 {
                    w_old[v][j] = vmulq_f32(one_minus_f, old[j]);
                    w_new[v][j] = vmulq_f32(f, new[j]);
                }
            }
            let mut acc = [[zero; V]; C];
            let mut old_taps = [[[zero; 3]; V]; C];
            let mut new_taps = [[[zero; 3]; V]; C];
            for c in 0..C {
                for v in 0..V {
                    let y = ptr[c].add(at + k + 4 * v);
                    acc[c][v] = vld1q_f32(y);
                    old_taps[c][v] = taps(y.sub(t0));
                    new_taps[c][v] = taps(y.sub(t1));
                }
            }
            // The order of `comb_fade_tap6`.
            for j in 0..3 {
                for c in 0..C {
                    for v in 0..V {
                        acc[c][v] = vfmaq_f32(acc[c][v], w_old[v][j], old_taps[c][v][j]);
                    }
                }
            }
            for j in 0..3 {
                for c in 0..C {
                    for v in 0..V {
                        acc[c][v] = vfmaq_f32(acc[c][v], w_new[v][j], new_taps[c][v][j]);
                    }
                }
            }
            for c in 0..C {
                for v in 0..V {
                    vst1q_f32(ptr[c].add(at + k + 4 * v), acc[c][v]);
                }
            }
            k += step;
        }
    }
    for buf in bufs.iter_mut() {
        comb_inplace_fade_scalar(buf, at + main, t0, t1, n - main, g0, g1, &window[main..]);
    }
}

#[cfg(all(test, feature = "std"))]
mod comb_inplace_tests {
    use super::*;

    pub(super) fn next(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed >> 8
    }

    /// `len` samples in the decoder's signal range. With `extreme`, about
    /// one sample in eight is a value at the edge of what `f32` can hold.
    pub(super) fn signal(len: usize, seed: &mut u32, extreme: bool) -> Vec<f32> {
        const EDGE: [f32; 10] = [
            0.0,
            -0.0,
            1e-30,
            -1e-38,
            1e30,
            -1e30,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NAN,
        ];
        (0..len)
            .map(|_| {
                let r = next(seed);
                if extreme && r.is_multiple_of(8) {
                    EDGE[(next(seed) % EDGE.len() as u32) as usize]
                } else {
                    (r as f32 / 8_388_608.0 - 1.0) * 32768.0
                }
            })
            .collect()
    }

    /// Same bit patterns. Two NaN count as equal whatever their payload.
    pub(super) fn assert_same(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "{what}: sample {i} is {x:e} ({:#x}), expected {y:e} ({:#x})",
                x.to_bits(),
                y.to_bits()
            );
        }
    }

    const LENGTHS: [usize; 20] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 119, 120, 121, 240, 720, 840,
    ];
    const PERIODS: [usize; 12] = [15, 16, 17, 18, 19, 20, 21, 22, 33, 100, 700, 1022];

    /// libopus `comb_filter` (celt/celt.c) with `x == y`, written as one loop
    /// over the frame: the cross-fade for the first `overlap` samples, then
    /// the new filter alone, which is skipped when its gain is 0. Every
    /// product is added with `mac`, in the order of the C expressions.
    #[allow(clippy::too_many_arguments)]
    fn one_loop_filter(
        buf: &mut [f32],
        y_idx: usize,
        t0: usize,
        t1: usize,
        n: usize,
        g0: f32,
        g1: f32,
        tapset0: i32,
        tapset1: i32,
        window: &[f32],
        overlap: usize,
    ) {
        if g0 == 0.0 && g1 == 0.0 {
            return;
        }
        let t0 = t0.clamp(COMBFILTER_MINPERIOD, y_idx - 2);
        let t1 = t1.clamp(COMBFILTER_MINPERIOD, y_idx - 2);
        let [g00, g01, g02] = PREFILTER_GAINS[tapset0 as usize].map(|g| g0 * g);
        let [g10, g11, g12] = PREFILTER_GAINS[tapset1 as usize].map(|g| g1 * g);
        let same = g0 == g1 && t0 == t1 && tapset0 == tapset1;
        let fade = if same { 0 } else { overlap.min(n) };
        for i in 0..n {
            let idx = y_idx + i;
            if i >= fade && g1 == 0.0 {
                break;
            }
            let x = buf[idx];
            let new_taps = [
                buf[idx - t1],
                buf[idx - t1 + 1] + buf[idx - t1 - 1],
                buf[idx - t1 + 2] + buf[idx - t1 - 2],
            ];
            buf[idx] = if i < fade {
                let f = window[i] * window[i];
                let old_taps = [
                    buf[idx - t0],
                    buf[idx - t0 + 1] + buf[idx - t0 - 1],
                    buf[idx - t0 + 2] + buf[idx - t0 - 2],
                ];
                let mut s = x;
                s = mac(s, (1.0 - f) * g00, old_taps[0]);
                s = mac(s, (1.0 - f) * g01, old_taps[1]);
                s = mac(s, (1.0 - f) * g02, old_taps[2]);
                s = mac(s, f * g10, new_taps[0]);
                s = mac(s, f * g11, new_taps[1]);
                mac(s, f * g12, new_taps[2])
            } else {
                mac(mac(mac(x, g10, new_taps[0]), g11, new_taps[1]), g12, new_taps[2])
            };
        }
    }

    #[test]
    fn whole_filter_matches_the_one_loop_form() {
        let window = crate::modes::default_mode().window;
        let overlap = CELT_OVERLAP;
        let mut seed = 1u32;
        for round in 0..400 {
            let n = LENGTHS[(next(&mut seed) % LENGTHS.len() as u32) as usize];
            let y_idx = 1030 + (next(&mut seed) % 60) as usize;
            let t0 = PERIODS[(next(&mut seed) % PERIODS.len() as u32) as usize];
            let t1 = if round % 5 == 0 {
                t0
            } else {
                PERIODS[(next(&mut seed) % PERIODS.len() as u32) as usize]
            };
            let gain = |s: &mut u32| 0.09375 * (next(s) % 9) as f32;
            let g0 = gain(&mut seed);
            let g1 = if round % 5 == 0 { g0 } else { gain(&mut seed) };
            let tapset0 = (next(&mut seed) % 3) as i32;
            let tapset1 = if round % 5 == 0 {
                tapset0
            } else {
                (next(&mut seed) % 3) as i32
            };
            let extreme = round % 4 == 3;
            let mut got = signal(y_idx + n + 7, &mut seed, extreme);
            let mut want = got.clone();
            comb_filter_inplace(
                &mut [&mut got[..]], y_idx, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap,
            );
            one_loop_filter(
                &mut want, y_idx, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap,
            );
            let what = format!("n {n} t0 {t0} t1 {t1} g0 {g0} g1 {g1}");
            assert_same(&got, &want, &what);

            // Two channels with the same filter. Each must come out as if
            // it were filtered alone.
            let mut got0 = signal(y_idx + n + 7, &mut seed, extreme);
            let mut got1 = signal(y_idx + n + 7, &mut seed, extreme);
            let (mut want0, mut want1) = (got0.clone(), got1.clone());
            comb_filter_inplace(
                &mut [&mut got0[..], &mut got1[..]],
                y_idx, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap,
            );
            for want in [&mut want0, &mut want1] {
                one_loop_filter(want, y_idx, t0, t1, n, g0, g1, tapset0, tapset1, window, overlap);
            }
            assert_same(&got0, &want0, &format!("channel 0 of 2, {what}"));
            assert_same(&got1, &want1, &format!("channel 1 of 2, {what}"));
        }
    }

    /// Runs `kernel` on `C` channels of random signal and compares each
    /// channel with `portable` on that channel alone.
    #[cfg(target_arch = "aarch64")]
    fn check_channels<const C: usize>(
        len: usize,
        seed: &mut u32,
        extreme: bool,
        what: &str,
        kernel: impl Fn(&mut [&mut [f32]; C]),
        portable: impl Fn(&mut [f32]),
    ) {
        let mut got: Vec<Vec<f32>> = (0..C).map(|_| signal(len, seed, extreme)).collect();
        let mut want = got.clone();
        let mut bufs: [&mut [f32]; C] = {
            let mut it = got.iter_mut();
            core::array::from_fn(|_| &mut it.next().unwrap()[..])
        };
        kernel(&mut bufs);
        for (c, (got, want)) in got.iter().zip(want.iter_mut()).enumerate() {
            portable(want);
            assert_same(got, want, &format!("channel {c} of {C}, {what}"));
        }
    }

    /// A vector version of `comb_inplace_const`.
    #[cfg(target_arch = "aarch64")]
    type ConstKernel<const C: usize> = fn(&mut [&mut [f32]; C], usize, usize, usize, [f32; 3]);

    #[cfg(target_arch = "aarch64")]
    fn check_const_part<const C: usize>(seed: &mut u32, name: &str, kernel: ConstKernel<C>) {
        for extreme in [false, true] {
            for &n in &LENGTHS {
                for &t in &PERIODS {
                    for tapset in 0..3 {
                        let g = PREFILTER_GAINS[tapset].map(|g| 0.75 * g);
                        // Start right after the history the filter reads,
                        // and at a few other places.
                        for at in [t + 2, t + 3, t + 9, 1100] {
                            check_channels::<C>(
                                at + n + 5,
                                seed,
                                extreme,
                                &format!("{name}, n {n} t {t} at {at}"),
                                |bufs| kernel(bufs, at, t, n, g),
                                |buf| comb_inplace_const_scalar(buf, at, t, n, g),
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn const_part_neon_matches_portable_version() {
        let mut seed = 7u32;
        check_const_part::<1>(&mut seed, "1 vector", comb_inplace_const_neon::<1, 1>);
        check_const_part::<1>(&mut seed, "2 vectors", comb_inplace_const_neon::<1, 2>);
        check_const_part::<1>(&mut seed, "3 vectors", comb_inplace_const_neon::<1, 3>);
        check_const_part::<2>(&mut seed, "1 vector", comb_inplace_const_neon::<2, 1>);
        check_const_part::<2>(&mut seed, "2 vectors", comb_inplace_const_neon::<2, 2>);
        check_const_part::<2>(&mut seed, "3 vectors", comb_inplace_const_neon::<2, 3>);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn const_part_window_neon_matches_portable_version() {
        let mut seed = 13u32;
        check_const_part::<1>(&mut seed, "window, 1 vector", comb_inplace_const_window_neon::<1, 1>);
        check_const_part::<1>(&mut seed, "window, 2 vectors", comb_inplace_const_window_neon::<1, 2>);
        check_const_part::<2>(&mut seed, "window, 1 vector", comb_inplace_const_window_neon::<2, 1>);
        check_const_part::<2>(&mut seed, "window, 2 vectors", comb_inplace_const_window_neon::<2, 2>);
    }

    #[cfg(target_arch = "aarch64")]
    fn check_fade_part<const C: usize, const V: usize>(seed: &mut u32) {
        let window = crate::modes::default_mode().window;
        for extreme in [false, true] {
            for &n in LENGTHS.iter().filter(|&&n| n <= window.len()) {
                for &t0 in &PERIODS {
                    for &t1 in &[15, 16, 19, 100, 1022] {
                        let g0 = PREFILTER_GAINS[t0 % 3].map(|g| 0.5625 * g);
                        let g1 = PREFILTER_GAINS[t1 % 3].map(|g| 0.75 * g);
                        for at in [t0.max(t1) + 2, 1100] {
                            check_channels::<C>(
                                at + n + 5,
                                seed,
                                extreme,
                                &format!("{V} vectors, n {n} t0 {t0} t1 {t1} at {at}"),
                                |bufs| {
                                    comb_inplace_fade_neon::<C, V>(bufs, at, t0, t1, n, g0, g1, window)
                                },
                                |buf| comb_inplace_fade_scalar(buf, at, t0, t1, n, g0, g1, window),
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fade_part_neon_matches_portable_version() {
        let mut seed = 11u32;
        check_fade_part::<1, 1>(&mut seed);
        check_fade_part::<1, 2>(&mut seed);
        check_fade_part::<1, 3>(&mut seed);
        check_fade_part::<2, 1>(&mut seed);
        check_fade_part::<2, 2>(&mut seed);
        check_fade_part::<2, 3>(&mut seed);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic]
    fn const_part_neon_rejects_a_period_longer_than_the_history() {
        let mut buf = vec![0.0f32; 64];
        comb_inplace_const_neon::<1, COMB_VECTORS>(&mut [&mut buf[..]], 20, 19, 8, [0.1, 0.1, 0.1]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic]
    fn const_part_neon_rejects_a_frame_past_the_end() {
        let mut buf = vec![0.0f32; 64];
        comb_inplace_const_neon::<1, COMB_VECTORS>(&mut [&mut buf[..]], 40, 15, 28, [0.1, 0.1, 0.1]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic]
    fn const_part_neon_rejects_a_second_channel_past_the_end() {
        let (mut buf0, mut buf1) = (vec![0.0f32; 64], vec![0.0f32; 60]);
        let bufs = &mut [&mut buf0[..], &mut buf1[..]];
        comb_inplace_const_neon::<2, COMB_VECTORS>(bufs, 40, 15, 24, [0.1, 0.1, 0.1]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic]
    fn const_part_window_neon_rejects_a_period_longer_than_the_history() {
        let mut buf = vec![0.0f32; 64];
        comb_inplace_const_window_neon::<1, 2>(&mut [&mut buf[..]], 20, 19, 8, [0.1, 0.1, 0.1]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic]
    fn const_part_window_neon_rejects_a_second_channel_past_the_end() {
        let (mut buf0, mut buf1) = (vec![0.0f32; 64], vec![0.0f32; 60]);
        let bufs = &mut [&mut buf0[..], &mut buf1[..]];
        comb_inplace_const_window_neon::<2, 2>(bufs, 40, 15, 24, [0.1, 0.1, 0.1]);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[should_panic]
    fn fade_part_neon_rejects_a_second_channel_past_the_end() {
        let window = crate::modes::default_mode().window;
        let (mut buf0, mut buf1) = (vec![0.0f32; 64], vec![0.0f32; 60]);
        let (g0, g1) = ([0.1; 3], [0.2; 3]);
        let bufs = &mut [&mut buf0[..], &mut buf1[..]];
        comb_inplace_fade_neon::<2, COMB_VECTORS>(bufs, 40, 15, 16, 24, g0, g1, window);
    }
}

#[cfg(all(test, feature = "std", target_arch = "aarch64"))]
mod deemph_segments_tests {
    use super::comb_inplace_tests::{assert_same, next, signal};
    use super::*;

    /// Time of the de-emphasis of one 20 ms frame, by each way of computing
    /// it. Not a check: run it by hand, pinned to one core, with
    /// `--ignored --nocapture deemphasis_timing`.
    #[test]
    #[ignore]
    fn deemphasis_timing() {
        let n = 960;
        let warmup = DEEMPH_WARMUP;
        let coef = 0.85f32;
        let mut seed = 0x5eed_d00du32;
        let hist: [Vec<f32>; 2] = core::array::from_fn(|_| shaped_signal(0, warmup + n + 4, &mut seed));
        let at = warmup;
        let mut out = vec![0.0f32; 2 * n];
        let reps = 4000;
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
            std::println!("{name:28} {best:9.1} ns per frame");
        };
        let mut m1 = [0.0f32];
        let mut m2 = [0.0f32; 2];
        time("segments mono", &mut || {
            m1 = deemph_segments_neon::<1, false>([&hist[0]], at, &mut out, n, warmup, coef, m1);
        });
        time("segments stereo interleaved", &mut || {
            m2 = deemph_segments_neon::<2, true>([&hist[0], &hist[1]], at, &mut out, n, warmup, coef, m2);
        });
        time("one sample mono", &mut || {
            let x = &hist[0][at..at + n];
            m1[0] = deemph_mono(x, &mut out[..n], 1, coef, m1[0], |o, v| *o = v);
        });
        time("one sample stereo", &mut || {
            let (x0, x1) = (&hist[0][at..at + n], &hist[1][at..at + n]);
            m2 = deemph_stereo(x0, x1, stereo_pairs(&mut out), 1, coef, m2, |o, v| *o = v);
        });
        std::hint::black_box(&out);
    }

    /// Test signals of `len` samples. Several of them end in silence or
    /// change level sharply: there the filters of the segments do not reach
    /// the exact memory during their run-in, and the segments must be
    /// computed again.
    fn shaped_signal(kind: usize, len: usize, seed: &mut u32) -> Vec<f32> {
        let mut x = signal(len, seed, kind == 7);
        let cut = (next(seed) as usize) % len.max(1);
        match kind {
            // Noise.
            0 | 7 => {}
            // Silence.
            1 => x.fill(0.0),
            // Noise, then silence.
            2 => x[cut..].fill(0.0),
            // Silence, then noise.
            3 => x[..cut].fill(0.0),
            // Noise that falls by 90 dB at one point.
            4 => x[cut..].iter_mut().for_each(|v| *v *= 3e-5),
            // Values near the smallest ones `f32` holds.
            5 => x.iter_mut().for_each(|v| *v *= 1e-36),
            // A constant.
            _ => x.fill(12345.0),
        }
        x
    }

    /// Run the segment filter on `C` channels, in both layouts, and compare
    /// it with the filter that takes one sample at a time.
    fn check<const C: usize>(kind: usize, n: usize, warmup: usize, coef: f32, seed: &mut u32) {
        let at = warmup + [0, 1, 6][(next(seed) % 3) as usize];
        let what = format!("{C} channels, signal {kind}, n {n}, warmup {warmup}, coef {coef}");
        let hist: [Vec<f32>; C] = core::array::from_fn(|_| shaped_signal(kind, at + n + 2, seed));
        let mem: [f32; C] = core::array::from_fn(|c| [0.0, 1234.5, -3e-20][(c + kind) % 3]);
        let h: [&[f32]; C] = core::array::from_fn(|c| &hist[c][..]);

        // Three more values than the frame: they must stay as they are.
        let mut planar = vec![7.0f32; C * n + 3];
        let mut interleaved = vec![7.0f32; C * n + 3];
        let planar_mem = deemph_segments_neon::<C, false>(h, at, &mut planar, n, warmup, coef, mem);
        let inter_mem =
            deemph_segments_neon::<C, true>(h, at, &mut interleaved, n, warmup, coef, mem);
        for c in 0..C {
            let mut want = vec![7.0f32; n];
            let want_mem = deemph_mono(
                &hist[c][at..at + n],
                &mut want[..],
                1,
                coef,
                mem[c],
                |o, v| *o = v,
            );
            let got: Vec<f32> = interleaved[..C * n].iter().skip(c).step_by(C).copied().collect();
            assert_same(&planar[c * n..(c + 1) * n], &want, &format!("{what}: planar"));
            assert_same(&got, &want, &format!("{what}: interleaved"));
            assert_same(&[planar_mem[c]], &[want_mem], &format!("{what}: memory"));
            assert_same(&[inter_mem[c]], &[want_mem], &format!("{what}: memory, interleaved"));
        }
        assert_same(&planar[C * n..], &[7.0; 3], &format!("{what}: past the end"));
        assert_same(&interleaved[C * n..], &[7.0; 3], &format!("{what}: past the end"));
    }

    #[test]
    fn segments_match_one_sample_at_a_time() {
        let coef = crate::modes::default_mode().preemph[0];
        let mut seed = 21u32;
        for kind in 0..8 {
            for n in [0usize, 32, 64, 96, 160, 480, 960] {
                // Without a run-in, and with a short one, most segments
                // are computed again. With the long one, few are.
                for warmup in [0usize, 4, 32, 96, 160] {
                    for coef in [coef, 0.0, 0.5, 0.99] {
                        check::<1>(kind, n, warmup, coef, &mut seed);
                        check::<2>(kind, n, warmup, coef, &mut seed);
                    }
                }
            }
        }
    }

    /// Frames of digital silence after a signal must be computed once.
    ///
    /// In silence the filter memory stops at a value that depends on the
    /// sign it had when the signal ended. The filters of the segments must
    /// reach that same value during their run-in. They did not when they
    /// started from zero and the memory came down from a positive value:
    /// every segment of every silent frame was then computed twice.
    #[test]
    fn silence_after_a_signal_is_computed_once() {
        let coef = crate::modes::default_mode().preemph[0];
        let mut seed = 5u32;
        let mut settled = [0u32; 2];
        for (k, last) in [1000.0f32, -1000.0].into_iter().enumerate() {
            for n in [480usize, 960] {
                // The decoder's layout: the frame at the end of the history.
                let at = 2048 - n;
                let mut hist = [
                    signal(2048, &mut seed, false),
                    signal(2048, &mut seed, false),
                ];
                // One run with two channels and one with the first channel
                // alone.
                let mut mem2 = [0.0f32; 2];
                let mut mem1 = [0.0f32; 1];
                let mut y = vec![0.0f32; 2 * n];
                let warmup = DEEMPH_WARMUP;
                for frame in 0..12 {
                    for h in hist.iter_mut() {
                        h.copy_within(n.., 0);
                        // The signal ends with a constant of one sign.
                        h[at..].fill(if frame == 0 { last } else { 0.0 });
                    }
                    DEEMPH_RECOMPUTED.with(|count| count.set(0));
                    let h = [&hist[0][..], &hist[1][..]];
                    mem2 = deemph_segments_neon::<2, false>(h, at, &mut y, n, warmup, coef, mem2);
                    mem1 = deemph_segments_neon::<1, false>([h[0]], at, &mut y, n, warmup, coef, mem1);
                    // The memory takes about 570 samples to stop. Later
                    // frames are all silence, their run-in included.
                    if frame >= 4 {
                        let recomputed = DEEMPH_RECOMPUTED.with(|count| count.get());
                        assert_eq!(recomputed, 0, "last {last}, n {n}, frame {frame}");
                    }
                }
                assert_eq!(mem2[0].to_bits(), mem2[1].to_bits());
                assert_eq!(mem2[0].to_bits(), mem1[0].to_bits());
                settled[k] = mem1[0].to_bits();
            }
        }
        // The premise of this test: the two signs stop at different values.
        assert_ne!(settled[0], settled[1]);
    }

    #[test]
    #[should_panic]
    fn segments_reject_a_frame_that_is_not_a_multiple_of_32() {
        let hist = [0.0f32; 400];
        let mut y = [0.0f32; 48];
        deemph_segments_neon::<1, false>([&hist[..]], 200, &mut y, 48, 160, 0.85, [0.0]);
    }

    #[test]
    #[should_panic]
    fn segments_reject_a_history_shorter_than_the_run_in() {
        let hist = [0.0f32; 400];
        let mut y = [0.0f32; 64];
        deemph_segments_neon::<1, false>([&hist[..]], 100, &mut y, 64, 160, 0.85, [0.0]);
    }

    #[test]
    #[should_panic]
    fn segments_reject_a_frame_past_the_end_of_the_history() {
        let hist = [0.0f32; 220];
        let mut y = [0.0f32; 64];
        deemph_segments_neon::<1, false>([&hist[..]], 160, &mut y, 64, 160, 0.85, [0.0]);
    }

    #[test]
    #[should_panic]
    fn segments_reject_an_output_shorter_than_the_frames() {
        let hist = [0.0f32; 400];
        let mut y = [0.0f32; 127];
        deemph_segments_neon::<2, true>([&hist[..], &hist[..]], 200, &mut y, 64, 160, 0.85, [0.0; 2]);
    }
}

#[cfg(all(test, feature = "std"))]
mod decoder_scratch_tests {
    use super::*;

    /// Decode all of `packet` into `pcm`, one channel after the other.
    fn decode_planar(dec: &mut CeltDecoder, packet: &[u8], n: usize, pcm: &mut [f32]) -> usize {
        let mut rc = RangeCoder::new_decoder(packet);
        dec.decode(&mut rc, (packet.len() * 8) as i32, n, pcm, PcmLayout::Planar)
    }

    /// A test signal with a steady tone, noise bursts and pauses, so that
    /// the frames use the post-filter, short blocks and bands without bits.
    fn test_signal(channels: usize, samples: usize) -> Vec<f32> {
        let mut seed = 3u32;
        let mut pcm = vec![0.0f32; samples * channels];
        for i in 0..samples {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / 8_388_608.0 - 1.0;
            let t = i as f32 / 48000.0;
            let tone = (2.0 * core::f32::consts::PI * 220.0 * t).sin()
                + 0.5 * (2.0 * core::f32::consts::PI * 440.0 * t).sin();
            let burst = (i / 2000) % 5 == 3;
            let pause = (i / 7000) % 4 == 2;
            for ch in 0..channels {
                let side = if ch == 0 { 1.0 } else { 0.6 };
                pcm[i * channels + ch] = if pause {
                    0.0
                } else if burst {
                    0.4 * noise * side
                } else {
                    0.25 * tone * side + 0.01 * noise
                };
            }
        }
        pcm
    }

    /// The decoder does not clear `w_x` between frames. Its old contents
    /// must never reach the output. Decoder `b` gets the array overwritten
    /// with NaN before every frame. A NaN that was read
    /// would show in its output. Lost frames are part of the stream, so the
    /// concealment is covered as well.
    #[test]
    fn old_scratch_contents_do_not_reach_the_output() {
        let mode = crate::modes::default_mode();
        for channels in [1usize, 2] {
            for frame_size in [120usize, 240, 480, 960] {
                for (start_band, bytes) in
                    [(0usize, 12usize), (0, 40), (0, 160), (17, 10), (17, 30)]
                {
                    let frames = 60;
                    let pcm = test_signal(channels, frames * frame_size);
                    let mut enc = CeltEncoder::new(mode, channels);
                    enc.set_start_band(start_band).unwrap();
                    let mut a = CeltDecoder::new(mode, channels, 48000);
                    let mut b = CeltDecoder::new(mode, channels, 48000);
                    a.set_start_band(start_band).unwrap();
                    b.set_start_band(start_band).unwrap();
                    let mut out_a = vec![0.0f32; frame_size * channels];
                    let mut out_b = vec![0.0f32; frame_size * channels];
                    for k in 0..frames {
                        let frame =
                            &pcm[k * frame_size * channels..(k + 1) * frame_size * channels];
                        let mut rc = RangeCoder::new_encoder(bytes as u32);
                        enc.encode(frame, frame_size, &mut rc);
                        rc.done();
                        let packet = rc.buf[..bytes].to_vec();

                        b.w_x.fill(f32::NAN);
                        // Two lost frames in every ten, and a longer loss
                        // that takes the concealment from pitch to noise.
                        let lost = k % 10 >= 8 || (30..42).contains(&k);
                        if lost {
                            a.decode_lost(frame_size, &mut out_a, PcmLayout::Planar).unwrap();
                            b.decode_lost(frame_size, &mut out_b, PcmLayout::Planar).unwrap();
                        } else {
                            let n = decode_planar(&mut a, &packet, frame_size, &mut out_a);
                            assert_eq!(n, frame_size);
                            decode_planar(&mut b, &packet, frame_size, &mut out_b);
                        }
                        let same = out_a
                            .iter()
                            .zip(&out_b)
                            .all(|(x, y)| x.to_bits() == y.to_bits());
                        assert!(
                            same && out_a.iter().all(|v| v.is_finite()),
                            "channels {channels} frame {frame_size} start {start_band} bytes {bytes}: frame {k} differs"
                        );
                    }
                }
            }
        }
    }

    /// The output of every frame must be the de-emphasis filter, taken one
    /// sample at a time, of the frame in the decoder's history. This holds
    /// the faster ways of computing it (two channels in one loop, segments)
    /// and every output layout to the plain definition.
    #[test]
    fn output_is_the_deemphasis_of_the_history() {
        let mode = crate::modes::default_mode();
        let coef = mode.preemph[0];
        for channels in [1usize, 2] {
            for n in [120usize, 240, 480, 960] {
                for rate in [48000, 24000, 16000, 12000, 8000] {
                    for layout in [
                        PcmLayout::Planar,
                        PcmLayout::Interleaved,
                        PcmLayout::InterleavedAdd,
                    ] {
                        let ds = 48000 / rate as usize;
                        let frame_size = n / ds;
                        let frames = 24;
                        let pcm = test_signal(channels, frames * n);
                        let mut enc = CeltEncoder::new(mode, channels);
                        let mut dec = CeltDecoder::new(mode, channels, rate);
                        let what = format!(
                            "{channels} channels, n {n}, {rate} Hz, layout {}",
                            layout as u8
                        );
                        for k in 0..frames {
                            let frame = &pcm[k * n * channels..(k + 1) * n * channels];
                            let mut rc = RangeCoder::new_encoder(80);
                            enc.encode(frame, n, &mut rc);
                            rc.done();
                            let packet = rc.buf[..80].to_vec();

                            let mem_before = [dec.preemph_mem[0], dec.preemph_mem[channels - 1]];
                            let base = |i: usize| (i % 13) as f32 * 0.125 - 0.75;
                            let mut out: Vec<f32> = (0..frame_size * channels).map(base).collect();
                            let mut rc = RangeCoder::new_decoder(&packet);
                            let got = if k % 8 == 6 {
                                dec.decode_lost(frame_size, &mut out, layout).unwrap()
                            } else {
                                dec.decode(&mut rc, 640, frame_size, &mut out, layout)
                            };
                            assert_eq!(got, frame_size, "{what}");

                            for ch in 0..channels {
                                let at = dec.hist_start(ch) + DECODE_BUFFER_SIZE - n;
                                let x = dec.decode_mem[at..at + n].to_vec();
                                let mut want = vec![0.0f32; frame_size];
                                let m =
                                    deemph_mono(&x, &mut want, ds, coef, mem_before[ch], |o, v| {
                                        *o = v
                                    });
                                assert_eq!(
                                    m.to_bits(),
                                    dec.preemph_mem[ch].to_bits(),
                                    "{what}: memory"
                                );
                                for (i, w) in want.iter().enumerate() {
                                    let (idx, w) = match layout {
                                        PcmLayout::Planar => (ch * frame_size + i, *w),
                                        PcmLayout::Interleaved => (i * channels + ch, *w),
                                        PcmLayout::InterleavedAdd => {
                                            (i * channels + ch, base(i * channels + ch) + *w)
                                        }
                                    };
                                    assert_eq!(
                                        out[idx].to_bits(),
                                        w.to_bits(),
                                        "{what}: frame {k}, sample {i}, channel {ch}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The reference for `deemph_mono` and `deemph_stereo`: the
    /// de-emphasis as one loop over the frame that tests the position of
    /// every sample. `out[i / ds]` receives every `ds`-th sample, stored or
    /// added.
    fn plain_deemphasis(
        x: &[f32],
        out: &mut [f32],
        ds: usize,
        coef: f32,
        mem: f32,
        add: bool,
    ) -> f32 {
        let mut m = mem;
        for (i, &x) in x.iter().enumerate() {
            let val = x + 1e-30f32 + m;
            if i % ds == 0 {
                let v = val * (1.0 / 32768.0);
                if add {
                    out[i / ds] += v;
                } else {
                    out[i / ds] = v;
                }
            }
            m = val * coef;
        }
        m
    }

    /// `deemph_mono` and `deemph_stereo` are the portable de-emphasis. On
    /// 64-bit ARM the decoder uses them only for frames of 2.5 and 5 ms, on
    /// other processors for every frame. This test holds them to the plain
    /// loop for every output rate, for both ways of writing the result and
    /// for both stereo layouts.
    #[test]
    fn portable_deemphasis_matches_the_plain_loop() {
        use super::comb_inplace_tests::{assert_same, next, signal};
        let coef = crate::modes::default_mode().preemph[0];
        let store = |o: &mut f32, v: f32| *o = v;
        let add = |o: &mut f32, v: f32| *o += v;
        let mut seed = 31u32;
        // Three kinds of input: the decoder's signal range, the same with
        // values at the edge of what `f32` holds, and values as small as
        // the offset that the filter adds to every sample. With the last
        // kind the order of the two additions shows in the result.
        for kind in 0..3 {
            for ds in [1usize, 2, 3, 4, 6] {
                for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 20, 40, 120, 160] {
                    for adding in [false, true] {
                        let what = format!("kind {kind}, ds {ds}, {len} samples, adding {adding}");
                        let mut x = [
                            signal(len * ds, &mut seed, kind == 1),
                            signal(len * ds, &mut seed, kind == 1),
                        ];
                        let mut mem = [1234.5f32, -0.25];
                        if kind == 2 {
                            x.iter_mut().flatten().for_each(|v| *v *= 1e-35);
                            mem = [3e-30, -2e-31];
                        }
                        // What the output buffers hold before. One value
                        // more than the frame: it must stay as it is.
                        let before: [Vec<f32>; 2] = core::array::from_fn(|_| {
                            (0..len + 1)
                                .map(|_| next(&mut seed) as f32 / 16_777_216.0 - 0.5)
                                .collect()
                        });
                        let mut want = before.clone();
                        let want_mem: [f32; 2] = core::array::from_fn(|c| {
                            plain_deemphasis(&x[c], &mut want[c][..len], ds, coef, mem[c], adding)
                        });

                        // One channel.
                        let mut got = before[0].clone();
                        let out = &mut got[..len];
                        let m = if adding {
                            deemph_mono(&x[0], out, ds, coef, mem[0], add)
                        } else {
                            deemph_mono(&x[0], out, ds, coef, mem[0], store)
                        };
                        assert_same(&got, &want[0], &format!("mono, {what}"));
                        assert_same(&[m], &want_mem[..1], &format!("mono memory, {what}"));

                        // Two channels, one after the other.
                        let mut got = before.clone();
                        let [got0, got1] = &mut got;
                        let places = got0[..len].iter_mut().zip(got1[..len].iter_mut());
                        let m = if adding {
                            deemph_stereo(&x[0], &x[1], places, ds, coef, mem, add)
                        } else {
                            deemph_stereo(&x[0], &x[1], places, ds, coef, mem, store)
                        };
                        assert_same(&got[0], &want[0], &format!("planar left, {what}"));
                        assert_same(&got[1], &want[1], &format!("planar right, {what}"));
                        assert_same(&m, &want_mem, &format!("planar memory, {what}"));

                        // Two channels, the samples of a pair together.
                        let interleave = |v: &[Vec<f32>; 2]| -> Vec<f32> {
                            (0..2 * (len + 1)).map(|i| v[i % 2][i / 2]).collect()
                        };
                        let mut got = interleave(&before);
                        let places = stereo_pairs(&mut got[..2 * len]);
                        let m = if adding {
                            deemph_stereo(&x[0], &x[1], places, ds, coef, mem, add)
                        } else {
                            deemph_stereo(&x[0], &x[1], places, ds, coef, mem, store)
                        };
                        assert_same(&got, &interleave(&want), &format!("interleaved, {what}"));
                        assert_same(&m, &want_mem, &format!("interleaved memory, {what}"));
                    }
                }
            }
        }
    }

    /// The output must not depend on where the history sits in
    /// `decode_mem`, nor on what `decode_mem` holds outside the history.
    ///
    /// The history slides through `decode_mem` and is moved back when it
    /// reaches the end (`advance_history`). The samples that enter it at
    /// its end are old values until the frame is written over them. Decoder
    /// `b` starts with its history at another place, so it moves it back at
    /// other frames than decoder `a`. Before every frame, everything in its
    /// `decode_mem` outside the history is overwritten with NaN. An old
    /// value that was read would show as a difference or as a NaN. The
    /// stream changes its frame length and has lost frames.
    #[test]
    fn output_does_not_depend_on_the_place_of_the_history() {
        let mode = crate::modes::default_mode();
        let chan_mem = DECODE_BUFFER_SIZE + CELT_OVERLAP;
        let sizes = [120usize, 960, 240, 480, 480, 120, 120, 960, 960, 240, 120];
        let frames = 90;
        let total: usize = (0..frames).map(|k| sizes[k % sizes.len()]).sum();
        for channels in [1usize, 2] {
            let pcm = test_signal(channels, total);
            for start_pos in [0usize, 120, 600, 840, CELT_HIST_SLACK] {
                let mut enc = CeltEncoder::new(mode, channels);
                let mut a = CeltDecoder::new(mode, channels, 48000);
                let mut b = CeltDecoder::new(mode, channels, 48000);
                // A new decoder's history is all zero, so it can start
                // anywhere.
                b.hist_pos = start_pos;
                let mut moved_back = 0;
                let mut done = 0;
                for k in 0..frames {
                    let n = sizes[k % sizes.len()];
                    let frame = &pcm[done * channels..(done + n) * channels];
                    done += n;
                    let mut rc = RangeCoder::new_encoder(60);
                    enc.encode(frame, n, &mut rc);
                    rc.done();
                    let packet = rc.buf[..60].to_vec();

                    for ch in 0..channels {
                        let part = ch * CELT_HIST_STRIDE;
                        let hist = b.hist_start(ch);
                        b.decode_mem[part..hist].fill(f32::NAN);
                        b.decode_mem[hist + chan_mem..part + CELT_HIST_STRIDE].fill(f32::NAN);
                    }
                    let pos_before = b.hist_pos;
                    let mut out_a = vec![0.0f32; n * channels];
                    let mut out_b = vec![0.0f32; n * channels];
                    // Single losses, and one long enough to go from the
                    // pitch concealment to the noise concealment.
                    let lost = k % 10 >= 8 || (40..58).contains(&k);
                    if lost {
                        a.decode_lost(n, &mut out_a, PcmLayout::Planar).unwrap();
                        b.decode_lost(n, &mut out_b, PcmLayout::Planar).unwrap();
                    } else {
                        assert_eq!(decode_planar(&mut a, &packet, n, &mut out_a), n);
                        assert_eq!(decode_planar(&mut b, &packet, n, &mut out_b), n);
                    }
                    if b.hist_pos < pos_before {
                        moved_back += 1;
                    }
                    assert!(b.hist_pos <= CELT_HIST_SLACK);
                    let same = out_a
                        .iter()
                        .zip(&out_b)
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                    assert!(
                        same && out_a.iter().all(|v| v.is_finite()),
                        "{channels} channels, history from {start_pos}: frame {k} of {n} samples differs"
                    );
                }
                // The history reached the end of its part many times.
                assert!(moved_back >= 20, "moved back {moved_back} times");
            }
        }
    }
}

/// Offset that libopus adds to every sample in the de-emphasis
/// (`VERY_SMALL`, celt/arch.h). It keeps the filter memory away from the
/// very small numbers that are slow to compute with.
const DEEMPH_VERY_SMALL: f32 = 1e-30;
/// Factor from the decoder's signal range to the output range of -1.0 to
/// 1.0 (libopus `SIG2RES`, float build).
const DEEMPH_SCALE: f32 = 1.0 / 32768.0;

/// One sample of the de-emphasis filter `y[i] = x[i] + coef * y[i - 1]`
/// (libopus `deemphasis`). `m` holds `coef * y[i - 1]`. Returns `y[i]`.
///
/// libopus writes `SATURATE(x + VERY_SMALL + m, SIG_SAT)`. In its float
/// build `SATURATE` does nothing (celt/arch.h).
#[inline(always)]
fn deemph_step(x: f32, m: &mut f32, coef: f32) -> f32 {
    let y = x + DEEMPH_VERY_SMALL + *m;
    *m = y * coef;
    y
}

/// How `CeltDecoder` lays out the samples of a frame in the caller's
/// buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmLayout {
    /// One channel after the other: `pcm[ch * frame_size + i]`.
    Planar,
    /// The channels of each sample together: `pcm[i * channels + ch]`.
    Interleaved,
    /// As `Interleaved`, and the frame is added to what `pcm` holds
    /// (libopus `accum`).
    InterleavedAdd,
}

/// The places of the left and the right sample of each pair in an
/// interleaved stereo buffer.
#[inline(always)]
fn stereo_pairs(out: &mut [f32]) -> impl Iterator<Item = (&mut f32, &mut f32)> {
    out.as_chunks_mut::<2>().0.iter_mut().map(|pair| {
        let [left, right] = pair;
        (left, right)
    })
}

/// De-emphasis of one channel. Returns the filter memory for the next
/// frame.
///
/// `x` holds `ds` samples for every sample of `out`. The filter runs over
/// all of them. The first sample of each group of `ds`, scaled to the
/// output range, goes to `put` together with its place in `out`. `put`
/// either stores it or adds it.
#[inline(always)]
fn deemph_mono(
    x: &[f32],
    out: &mut [f32],
    ds: usize,
    coef: f32,
    mem: f32,
    put: impl Fn(&mut f32, f32),
) -> f32 {
    debug_assert!(ds >= 1 && x.len() == out.len() * ds);
    let mut m = mem;
    if ds == 1 {
        for (o, &x) in out.iter_mut().zip(x) {
            put(o, deemph_step(x, &mut m, coef) * DEEMPH_SCALE);
        }
    } else {
        for (o, group) in out.iter_mut().zip(x.chunks_exact(ds)) {
            put(o, deemph_step(group[0], &mut m, coef) * DEEMPH_SCALE);
            for &x in &group[1..] {
                deemph_step(x, &mut m, coef);
            }
        }
    }
    m
}

/// De-emphasis of two channels in one loop (libopus
/// `deemphasis_stereo_simple`). Returns the two filter memories.
///
/// Each output sample needs the one before it, so one channel alone cannot
/// go faster than one addition and one multiplication after the other per
/// sample. The two channels do not depend on each other. In one loop the
/// processor works on both at once, and two channels take the time of one.
///
/// `out` yields the place of each output sample of the two channels. `x0`
/// and `x1` hold `ds` samples for every item of `out`, and `put` stores or
/// adds a sample, as in `deemph_mono`.
#[inline(always)]
fn deemph_stereo<'a>(
    x0: &[f32],
    x1: &[f32],
    out: impl Iterator<Item = (&'a mut f32, &'a mut f32)>,
    ds: usize,
    coef: f32,
    mem: [f32; 2],
    put: impl Fn(&mut f32, f32),
) -> [f32; 2] {
    debug_assert!(ds >= 1 && x0.len() == x1.len());
    let [mut m0, mut m1] = mem;
    if ds == 1 {
        for ((&a, &b), (o0, o1)) in x0.iter().zip(x1).zip(out) {
            let y0 = deemph_step(a, &mut m0, coef);
            let y1 = deemph_step(b, &mut m1, coef);
            put(o0, y0 * DEEMPH_SCALE);
            put(o1, y1 * DEEMPH_SCALE);
        }
    } else {
        let groups = x0.chunks_exact(ds).zip(x1.chunks_exact(ds));
        for ((g0, g1), (o0, o1)) in groups.zip(out) {
            let y0 = deemph_step(g0[0], &mut m0, coef);
            let y1 = deemph_step(g1[0], &mut m1, coef);
            put(o0, y0 * DEEMPH_SCALE);
            put(o1, y1 * DEEMPH_SCALE);
            for (&a, &b) in g0[1..].iter().zip(&g1[1..]) {
                deemph_step(a, &mut m0, coef);
                deemph_step(b, &mut m1, coef);
            }
        }
    }
    [m0, m1]
}

/// Shortest frame, in samples at 48 kHz, whose de-emphasis is computed in
/// segments (`deemph_segments_neon`). Below it the segments are too short
/// for the run-in to pay off.
#[cfg(target_arch = "aarch64")]
const DEEMPH_SEGMENTS_MIN: usize = 480;
/// Samples that the de-emphasis filter of a segment runs over before its
/// segment starts (`deemph_segments_neon`).
///
/// A filter that starts from a wrong memory comes closer to the right
/// values by the factor `coef` (0.85) per sample. After about 100 samples
/// the difference is one unit in the last place of a `f32`. From there it
/// disappears with a probability of about 0.15 per sample.
///
/// Measured on 13 decode scenarios of the test harness (music and speech,
/// 190 000 to 1 100 000 segments each): with a run-in of 160 samples, 9
/// scenarios computed no segment again, 2 computed one, one computed 1 in
/// 3600 and one 1 in 2100. With 128 samples it was 1 segment in 50 to 230
/// (6 scenarios), and with 96 samples about 3 in 4.
#[cfg(target_arch = "aarch64")]
const DEEMPH_WARMUP: usize = 160;

#[cfg(all(test, feature = "std", target_arch = "aarch64"))]
std::thread_local! {
    /// Segments that `deemph_segments_neon` computed a second time on this
    /// thread. Only tests count them.
    static DEEMPH_RECOMPUTED: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Rows to columns: lane `j` of the result's vector `k` is lane `k` of
/// `rows[j]`. Applying it twice gives the input back.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn transpose4(rows: [float32x4_t; 4]) -> [float32x4_t; 4] {
    // SAFETY: these instructions are part of every 64-bit ARM processor,
    // and they only rearrange values held in registers.
    unsafe {
        let t0 = vreinterpretq_f64_f32(vtrn1q_f32(rows[0], rows[1]));
        let t1 = vreinterpretq_f64_f32(vtrn2q_f32(rows[0], rows[1]));
        let t2 = vreinterpretq_f64_f32(vtrn1q_f32(rows[2], rows[3]));
        let t3 = vreinterpretq_f64_f32(vtrn2q_f32(rows[2], rows[3]));
        [
            vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)),
            vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)),
            vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)),
            vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)),
        ]
    }
}

/// De-emphasis of one frame of `C` channels (1 or 2), computed in eight
/// segments at the same time. The result is the same, bit for bit, as that
/// of `deemph_mono` on each channel.
///
/// `hist[c][at..at + n]` is the frame of channel `c`. `out` receives the
/// `n` output samples of each channel, scaled to the output range: one
/// channel after the other, or with `INTERLEAVED` the channels of each
/// sample together. `mem` holds the filter memories. The function returns
/// them for the next frame.
///
/// # Why segments
///
/// Each output sample needs the one before it. A channel therefore takes
/// one addition and one multiplication, one after the other, per sample,
/// however fast the processor is otherwise. Eight filters that do not
/// depend on each other take the same time as one.
///
/// # How the result stays exact
///
/// Each channel's frame is cut into equal segments: eight for one channel,
/// four each for two. The filter of the first segment starts from `mem`.
/// The filter of every other segment does not know its starting memory,
/// because that is the result of the segment before. It starts `warmup`
/// samples early with a guess. A wrong memory fades by the factor `coef`
/// per sample, so at the start of its segment the filter has usually
/// reached exactly the memory it would have had.
///
/// The guess is `mem`, the memory at the start of the frame, and not zero.
/// For most signals the two do equally well. Digital silence is the
/// exception. There the memory stops at one of several neighbouring `f32`
/// values: a memory that comes down from a positive value stops at a
/// higher one than a memory that rises from zero or from a negative value.
/// A filter started from zero would then never reach the exact memory, and
/// every segment of every silent frame would be computed twice. Started
/// from `mem`, it already holds the value at which the memory stopped.
///
/// "Usually" is not enough, so every segment is checked afterwards, in
/// order. The memory a filter had at the start of its segment is compared
/// with the memory the filter before it ended with, which is exact. If the
/// two are the same bits, the segment was computed from the right memory
/// with the same operations in the same order, so its samples are exact,
/// and so is the memory it ends with. If they differ, the segment is
/// computed again, one sample at a time, from the exact memory. That
/// happens when the signal drops to silence, where a wrong memory fades no
/// faster than the signal itself.
///
/// `n` must be a multiple of 32, `warmup` a multiple of 4, and `hist`
/// must hold `warmup` samples before `at`. Their values do not matter for
/// the result.
#[cfg(target_arch = "aarch64")]
fn deemph_segments_neon<const C: usize, const INTERLEAVED: bool>(
    hist: [&[f32]; C],
    at: usize,
    out: &mut [f32],
    n: usize,
    warmup: usize,
    coef: f32,
    mem: [f32; C],
) -> [f32; C] {
    /// Filters that run at the same time: two vectors of four.
    const LANES: usize = 8;
    assert!(C == 1 || C == 2);
    let per_ch = LANES / C;
    let seg = n / per_ch;
    assert!(n.is_multiple_of(32) && warmup.is_multiple_of(4) && at >= warmup);
    assert!(out.len() >= C * n);
    for c in 0..C {
        assert!(at + n <= hist[c].len());
    }
    // An empty frame leaves the memories as they are. The steps below need
    // segments of at least 4 samples.
    if n == 0 {
        return mem;
    }
    // The place in `out` of sample `i` of channel `c`. With one channel
    // the two layouts are the same.
    let place = |c: usize, i: usize| if INTERLEAVED { i * C + c } else { c * n + i };

    // Filter `s` handles segment `s % per_ch` of channel `s / per_ch`.
    // `x_ptr[s]` is where its run-in starts.
    let mut x_ptr = [core::ptr::null::<f32>(); LANES];
    for c in 0..C {
        let x_base = hist[c].as_ptr();
        for q in 0..per_ch {
            // SAFETY: `at - warmup + q * seg <= at + n <= hist[c].len()`
            // (checked above), so the pointer stays inside the slice.
            unsafe {
                x_ptr[c * per_ch + q] = x_base.add(at - warmup + q * seg);
            }
        }
    }
    let o = out.as_mut_ptr();

    // SAFETY: `n > 0`, so `seg` is a positive multiple of 4. Filter `s`
    // reads `x_ptr[s][t..t + 4]` for `t` in steps of 4 below
    // `warmup + seg`. The last value read is at
    // `at + (q + 1) * seg - 1 < at + n` in `hist[c]`. At step `t` it writes
    // samples `k..k + 4` of its segment, with `k = t - warmup` below `seg`:
    // the places `place(c, q * seg + k)` to `place(c, q * seg + k + 3)`.
    // They are below `C * n <= out.len()`. With `INTERLEAVED` and two
    // channels the two channels' samples `k..k + 4` of segment `q` are the
    // eight values from `2 * (q * seg + k)`. The filters write different
    // places. `hist` and `out` cannot overlap: one is borrowed shared and
    // the other mutably. `warmup` and `seg` are multiples of 4.
    let (start_mem, end_mem) = unsafe {
        let very_small = vdupq_n_f32(DEEMPH_VERY_SMALL);
        let coef_v = vdupq_n_f32(coef);
        let scale = vdupq_n_f32(DEEMPH_SCALE);
        // Four samples of each of the eight filters at step `t`: per group
        // `g`, one vector per sample, the four filters `4 * g..4 * g + 4`
        // in its lanes.
        let load = |t: usize| {
            let group = |g: usize| {
                transpose4([
                    vld1q_f32(x_ptr[4 * g].add(t)),
                    vld1q_f32(x_ptr[4 * g + 1].add(t)),
                    vld1q_f32(x_ptr[4 * g + 2].add(t)),
                    vld1q_f32(x_ptr[4 * g + 3].add(t)),
                ])
            };
            [group(0), group(1)]
        };

        // The filters of both groups for four samples: one sample of group
        // 0, then the same sample of group 1. On a core that runs the
        // instructions in program order, the step of one group runs while
        // the other waits for the result of its last step. Returns the
        // outputs scaled to the output range, one vector per sample.
        let filter = |m: &mut [float32x4_t; 2], x: [[float32x4_t; 4]; 2]| {
            let mut scaled = [[very_small; 4]; 2];
            for s in 0..4 {
                let a0 = vaddq_f32(x[0][s], very_small);
                let a1 = vaddq_f32(x[1][s], very_small);
                let y0 = vaddq_f32(a0, m[0]);
                let y1 = vaddq_f32(a1, m[1]);
                m[0] = vmulq_f32(y0, coef_v);
                m[1] = vmulq_f32(y1, coef_v);
                scaled[0][s] = vmulq_f32(y0, scale);
                scaled[1][s] = vmulq_f32(y1, scale);
            }
            scaled
        };

        // Every loop below loads and rearranges the input of the next step
        // while the filters run on the input of this one, and stores the
        // output of the step before. Without branches between them, an
        // in-order core can issue these independent instructions in the
        // cycles in which the filters wait for their results.

        // Run-in: memories only, no output. Every filter starts from the
        // memory of its channel at the start of the frame. With two
        // channels the second group holds the filters of channel 1. The
        // next step's input lies below `warmup + seg`, because `seg >= 4`.
        let end = warmup + seg;
        let mut m = [vdupq_n_f32(mem[0]), vdupq_n_f32(mem[C - 1])];
        let mut x = load(0);
        let mut t = 0;
        while t < warmup {
            let now = x;
            x = load(t + 4);
            filter(&mut m, now);
            t += 4;
        }

        // The first segment of each channel continues from the frame
        // before, with the exact memory.
        let mut start_mem = [0.0f32; LANES];
        vst1q_f32(start_mem.as_mut_ptr(), m[0]);
        vst1q_f32(start_mem.as_mut_ptr().add(4), m[1]);
        for c in 0..C {
            start_mem[c * per_ch] = mem[c];
        }
        m[0] = vld1q_f32(start_mem.as_ptr());
        m[1] = vld1q_f32(start_mem.as_ptr().add(4));

        // Store samples `k..k + 4` of every filter, from `scaled` (one
        // vector per sample).
        let store = |scaled: [[float32x4_t; 4]; 2], k: usize| {
            if INTERLEAVED && C == 2 {
                // Group `c` holds the four segments of channel `c`. Lane
                // pair `q` of `pairs[h][s]` (h = q / 2) is (left, right) of
                // sample `k + s` of segment `q`.
                let pairs: [[float64x2_t; 4]; 2] = [
                    core::array::from_fn(|s| {
                        vreinterpretq_f64_f32(vzip1q_f32(scaled[0][s], scaled[1][s]))
                    }),
                    core::array::from_fn(|s| {
                        vreinterpretq_f64_f32(vzip2q_f32(scaled[0][s], scaled[1][s]))
                    }),
                ];
                for q in 0..4 {
                    let h = &pairs[q / 2];
                    let (lo, hi) = if q % 2 == 0 {
                        (vtrn1q_f64(h[0], h[1]), vtrn1q_f64(h[2], h[3]))
                    } else {
                        (vtrn2q_f64(h[0], h[1]), vtrn2q_f64(h[2], h[3]))
                    };
                    let p = o.add(2 * (q * seg + k));
                    vst1q_f32(p, vreinterpretq_f32_f64(lo));
                    vst1q_f32(p.add(4), vreinterpretq_f32_f64(hi));
                }
            } else {
                for (g, group) in scaled.into_iter().enumerate() {
                    // Back to one vector of four samples per filter.
                    for (j, row) in transpose4(group).into_iter().enumerate() {
                        let s = 4 * g + j;
                        vst1q_f32(o.add(place(s / per_ch, (s % per_ch) * seg + k)), row);
                    }
                }
            }
        };

        // The segments. The first step is taken before the loop, so that
        // the loop always has the output of a step to store. The last step
        // loads its own input again instead of reading past the segments.
        let next = |t: usize| if t + 4 < end { t + 4 } else { t };
        let now = x;
        x = load(next(t));
        let mut done = filter(&mut m, now);
        let mut k = 0;
        t += 4;
        while t < end {
            let now = x;
            x = load(next(t));
            store(done, k);
            done = filter(&mut m, now);
            k = t - warmup;
            t += 4;
        }
        store(done, k);

        let mut end_mem = [0.0f32; LANES];
        vst1q_f32(end_mem.as_mut_ptr(), m[0]);
        vst1q_f32(end_mem.as_mut_ptr().add(4), m[1]);
        (start_mem, end_mem)
    };

    // Check the segments of each channel in order. `exact` is the memory
    // that the one-sample-at-a-time filter has at the start of segment `q`.
    let mut next_mem = mem;
    for c in 0..C {
        let mut exact = end_mem[c * per_ch];
        for q in 1..per_ch {
            let s = c * per_ch + q;
            exact = if start_mem[s].to_bits() == exact.to_bits() {
                end_mem[s]
            } else {
                #[cfg(all(test, feature = "std"))]
                DEEMPH_RECOMPUTED.with(|count| count.set(count.get() + 1));
                let x = &hist[c][at + q * seg..at + (q + 1) * seg];
                let mut m = exact;
                for (i, &x) in x.iter().enumerate() {
                    out[place(c, q * seg + i)] = deemph_step(x, &mut m, coef) * DEEMPH_SCALE;
                }
                m
            };
        }
        next_mem[c] = exact;
    }
    next_mem
}

/// Store every `ds`-th value of `y` in `out`, or add it, as `put` decides.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn emit_mono(y: &[f32], out: &mut [f32], ds: usize, put: impl Fn(&mut f32, f32)) {
    if ds == 1 {
        for (o, &v) in out.iter_mut().zip(y) {
            put(o, v);
        }
    } else {
        for (o, group) in out.iter_mut().zip(y.chunks_exact(ds)) {
            put(o, group[0]);
        }
    }
}

/// Store every `ds`-th value of `y0` and `y1` at the places `out` yields,
/// or add it, as `put` decides.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn emit_stereo<'a>(
    y0: &[f32],
    y1: &[f32],
    out: impl Iterator<Item = (&'a mut f32, &'a mut f32)>,
    ds: usize,
    put: impl Fn(&mut f32, f32),
) {
    if ds == 1 {
        for ((&a, &b), (o0, o1)) in y0.iter().zip(y1).zip(out) {
            put(o0, a);
            put(o1, b);
        }
    } else {
        let groups = y0.chunks_exact(ds).zip(y1.chunks_exact(ds));
        for ((g0, g1), (o0, o1)) in groups.zip(out) {
            put(o0, g0[0]);
            put(o1, g1[0]);
        }
    }
}

/// The pitch post-filter of a frame of `n` samples, in place in the
/// histories `bufs` of all channels (see `CeltDecoder::postfilter`).
/// `filters` holds the period, gain and tapset of the filter before last,
/// of the last one and of this frame's.
#[inline(always)]
fn postfilter_frame<const C: usize>(
    mode: &CeltMode,
    bufs: &mut [&mut [f32]; C],
    n: usize,
    lm: usize,
    filters: [(usize, f32, i32); 3],
) {
    let short_n = mode.short_mdct_size;
    let at = DECODE_BUFFER_SIZE - n;
    let [old, last, new] = filters;
    comb_filter_inplace(
        bufs,
        at,
        old.0,
        last.0,
        short_n,
        old.1,
        last.1,
        old.2,
        last.2,
        mode.window,
        mode.overlap,
    );
    if lm != 0 {
        comb_filter_inplace(
            bufs,
            at + short_n,
            last.0,
            new.0,
            n - short_n,
            last.1,
            new.1,
            last.2,
            new.2,
            mode.window,
            mode.overlap,
        );
    }
}

/// The spectrum of one channel from its normalised spectrum, in place
/// (libopus `denormalise_bands` and the zeroing after it in
/// `celt_synthesis`). `x` holds the `n` values of the channel, `amp` the
/// band log-amplitudes of the channel (see `log2amp`).
///
/// Every band from `start` to `end` is multiplied by its amplitude, the
/// polynomial `celt_exp2` of its log-amplitude. Then the values below band
/// `start` and from `bound` upwards are set to 0. `bound` is at most the
/// end of band `end`, so every value of `x` is written. The values are
/// those of `bands::denormalise_bands` into another buffer, followed by
/// the same zeroing.
fn denormalise_in_place(
    mode: &CeltMode,
    x: &mut [f32],
    amp: &[f32],
    start: usize,
    end: usize,
    lm: usize,
    bound: usize,
) {
    for (i, &lg) in amp.iter().enumerate().take(end).skip(start) {
        let lo = (mode.e_bands[i] as usize) << lm;
        let hi = (mode.e_bands[i + 1] as usize) << lm;
        // C: `g = celt_exp2_db(MIN32(32.f, lg))`. The cap keeps the gain
        // finite. The comparison is written as in `denormalise_bands`, so
        // that a NaN takes the same path.
        let g = celt_exp2(if 32.0 < lg { 32.0 } else { lg });
        for v in &mut x[lo..hi] {
            *v *= g;
        }
    }
    let low = (mode.e_bands[start] as usize) << lm;
    x[..low].fill(0.0);
    x[bound..].fill(0.0);
}

pub struct CeltDecoder {
    mode: &'static CeltMode,
    /// Number of output channels (libopus `channels`).
    channels: usize,
    /// Number of channels coded in the packets (libopus `stream_channels`).
    /// It may differ from `channels`: a mono packet is copied to both output
    /// channels, a stereo packet is mixed down to one.
    stream_channels: usize,
    /// Decimation factor for non-48kHz API rates (1=48k, 2=24k, 3=16k, 4=12k, 6=8k).
    /// Mirrors libopus `CELTDecoder.downsample`.
    downsample: usize,
    /// First band of the range that frames code (libopus `start`, set with
    /// `set_start_band`). A reset does not change it.
    start_band: usize,
    /// End of the band range (libopus `end`, set with `set_end_band`).
    /// Frames are decoded and concealed within this range. A reset does
    /// not change it.
    end_band: usize,
    /// The decoder complexity, 0 to 10 (libopus `complexity`, set with
    /// `OPUS_SET_COMPLEXITY`). From 5, a lost CELT-only frame is concealed
    /// with deep PLC when the decoder has it. A reset does not change it.
    complexity: i32,
    /// libopus `disable_inv` (`OPUS_SET_PHASE_INVERSION_DISABLED`): decode
    /// intensity stereo bands without the sign inversion of the second
    /// channel. It starts `true` for a one-channel decoder, which mixes a
    /// stereo stream down and where the inversion would cancel in the mix,
    /// and `false` for two channels (celt_decoder.c:261). A reset does not
    /// change it.
    disable_inv: bool,

    // Everything below is cleared by `reset_state`.
    rng: u32,
    /// Pitch period found for the pitch-based concealment (libopus
    /// `last_pitch_index`).
    last_pitch_index: usize,
    /// Time since the last decoded frame, in units of 2.5 ms (libopus
    /// `loss_duration`).
    loss_duration: i32,
    /// Length of the current run of concealed frames, in units of 2.5 ms
    /// (libopus `plc_duration`).
    plc_duration: i32,
    last_frame_type: FrameType,
    /// While set, lost frames are concealed with noise. It is set at the
    /// start and after a noise-concealed frame, and cleared once two frames
    /// in a row have been decoded (libopus `skip_plc`).
    skip_plc: bool,
    /// The post-filter period of the last frame (libopus
    /// `postfilter_period`): 0 when the frame switched the filter off, as
    /// in libopus. The comb filter treats any period below 15 as 15
    /// (`COMBFILTER_MINPERIOD`). `OPUS_GET_PITCH` returns this value.
    prefilter_period: usize,
    /// The post-filter period of the frame before (libopus
    /// `postfilter_period_old`).
    prefilter_period_old: usize,
    prefilter_gain: f32,
    prefilter_gain_old: f32,
    prefilter_tapset: i32,
    prefilter_tapset_old: i32,
    /// Set after a pitch-concealed frame. The next frame that goes through
    /// the inverse transform must first prepare the overlap region (libopus
    /// `prefilter_and_fold`).
    prefilter_and_fold: bool,
    preemph_mem: FixedVec<f32, CELT_MAX_CHANNELS>,
    /// Where each channel's history starts inside its part of `decode_mem`
    /// (see `advance_history`).
    hist_pos: usize,
    /// Output history per channel: `DECODE_BUFFER_SIZE` samples after the
    /// post-filter, then `overlap` samples of the inverse transform that the
    /// next frame completes (libopus `_decode_mem`).
    ///
    /// Channel `ch` owns `CELT_HIST_STRIDE` samples from
    /// `ch * CELT_HIST_STRIDE`. Its history is the
    /// `DECODE_BUFFER_SIZE + overlap` samples that start `hist_pos` samples
    /// into that part.
    decode_mem: FixedVec<f32, CELT_DECODE_MEM>,
    /// Band log-energies of the last frame, for two channels (libopus
    /// `oldEBands`). The three arrays below have the same layout.
    old_band_e: FixedVec<f32, CELT_NB_X_CH>,
    /// libopus `oldLogE`.
    old_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    /// libopus `oldLogE2`.
    old_band_e3: FixedVec<f32, CELT_NB_X_CH>,
    /// Estimate of the background noise level in each band (libopus
    /// `backgroundLogE`). Noise concealment decays towards it.
    background_log_e: FixedVec<f32, CELT_NB_X_CH>,
    /// Predictor of each channel for the pitch-based concealment (libopus
    /// `lpc`).
    plc_lpc: [f32; CELT_MAX_CHANNELS * PLC_LPC_ORDER],
    /// Deep PLC output at 16 kHz that is not used yet (libopus `plc_pcm`).
    /// The first `plc_fill` samples are valid.
    #[cfg(feature = "deep-plc")]
    plc_pcm: [i16; PLC_UPDATE_SAMPLES],
    /// libopus `plc_fill`.
    #[cfg(feature = "deep-plc")]
    plc_fill: usize,
    /// The memory of the pre-emphasis of deep PLC's output (libopus
    /// `plc_preemphasis_mem`): the last sample of the history after
    /// de-emphasis when deep PLC starts, then the last sample of each
    /// frame it conceals, before pre-emphasis.
    #[cfg(feature = "deep-plc")]
    plc_preemphasis_mem: f32,

    w_tf_res: FixedVec<i32, CELT_NB_EBANDS>,
    w_cap: FixedVec<i32, CELT_NB_EBANDS>,
    w_offsets: FixedVec<i32, CELT_NB_EBANDS>,
    w_pulses: FixedVec<i32, CELT_NB_EBANDS>,
    w_ebits: FixedVec<i32, CELT_NB_X_CH>,
    w_fine_priority: FixedVec<i32, CELT_NB_X_CH>,
    w_x: FixedVec<f32, CELT_W_X_DEC>,
    w_collapse_masks: FixedVec<u32, CELT_NB_X_CH>,
    w_band_amp: FixedVec<f32, CELT_NB_X_CH>,
}

impl CeltDecoder {
    /// Create a CELT decoder. `sampling_rate` is the API sampling rate
    /// (8000–48000); the decoder always operates at 48 kHz internally and
    /// decimates by `resampling_factor(sampling_rate)` on output.
    pub fn new(mode: &'static CeltMode, channels: usize, sampling_rate: i32) -> Self {
        let mut dec = core::mem::MaybeUninit::<Self>::uninit();
        // SAFETY: `write_new` initializes every field of `dec`.
        unsafe {
            Self::write_new(dec.as_mut_ptr(), mode, channels, sampling_rate);
            dec.assume_init()
        }
    }

    /// Writes the decoder of [`new`](Self::new) to `this`, which may be
    /// uninitialized memory, without building it on the stack first (see
    /// `crate::in_place`).
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    pub(crate) unsafe fn write_new(
        this: *mut Self,
        mode: &'static CeltMode,
        channels: usize,
        sampling_rate: i32,
    ) {
        // All internal buffers are sized for at most 2 channels.
        assert!(
            (1..=2).contains(&channels),
            "CeltDecoder::new: channels must be 1 or 2 (got {channels})"
        );
        // The buffers below are sized for the standard mode.
        assert!(
            mode.overlap == CELT_OVERLAP
                && mode.short_mdct_size << mode.max_lm <= CELT_DEC_MAX_FRAME,
            "CeltDecoder::new: mode with another overlap or a longer frame"
        );
        let nb_ebands = mode.nb_ebands;
        // The energy arrays and the scratch buffers always have room for two
        // channels: a stereo packet can arrive in a mono decoder.
        let nb_x_ch = nb_ebands * CELT_MAX_CHANNELS;
        // SAFETY: the caller guarantees that `this` is valid for writes.
        // Each closure below fills the whole buffer it is given.
        unsafe {
            write_fields!(this, {
                mode: mode,
                channels: channels,
                stream_channels: channels,
                downsample: resampling_factor(sampling_rate).unwrap_or(1),
                start_band: 0,
                end_band: mode.eff_ebands,
                complexity: 0,
                disable_inv: channels == 1,

                rng: 0,
                last_pitch_index: 0,
                loss_duration: 0,
                plc_duration: 0,
                last_frame_type: FrameType::None,
                skip_plc: true,
                prefilter_period: 0,
                prefilter_period_old: 0,
                prefilter_gain: 0.0,
                prefilter_gain_old: 0.0,
                prefilter_tapset: 0,
                prefilter_tapset_old: 0,
                prefilter_and_fold: false,
                preemph_mem = |p| FixedVec::write_from_value(p, 0.0, channels),
                hist_pos: 0,
                decode_mem = |p| FixedVec::write_from_value(p, 0.0, channels * CELT_HIST_STRIDE),
                old_band_e = |p| FixedVec::write_from_value(p, 0.0, nb_x_ch),
                // libopus `oldLogE` / `oldLogE2`: the band energies of the two
                // previous frames. They start at -28 (silence), not 0: the
                // anti-collapse step compares a band against them, and 0 makes
                // it inject noise into the first transient frame.
                old_band_e2 = |p| FixedVec::write_from_value(p, OLD_LOG_E_RESET, nb_x_ch),
                old_band_e3 = |p| FixedVec::write_from_value(p, OLD_LOG_E_RESET, nb_x_ch),
                background_log_e = |p| FixedVec::write_from_value(p, 0.0, nb_x_ch),
                plc_lpc: [0.0; CELT_MAX_CHANNELS * PLC_LPC_ORDER],
                #[cfg(feature = "deep-plc")]
                plc_pcm: [0; PLC_UPDATE_SAMPLES],
                #[cfg(feature = "deep-plc")]
                plc_fill: 0,
                #[cfg(feature = "deep-plc")]
                plc_preemphasis_mem: 0.0,

                w_tf_res = |p| FixedVec::write_from_value(p, 0, nb_ebands),
                w_cap = |p| FixedVec::write_from_value(p, 0, nb_ebands),
                w_offsets = |p| FixedVec::write_from_value(p, 0, nb_ebands),
                w_pulses = |p| FixedVec::write_from_value(p, 0, nb_ebands),
                w_ebits = |p| FixedVec::write_from_value(p, 0, nb_x_ch),
                w_fine_priority = |p| FixedVec::write_from_value(p, 0, nb_x_ch),

                w_x = |p| FixedVec::write_from_value(p, 0.0, CELT_W_X_DEC),
                w_collapse_masks = |p| FixedVec::write_from_value(p, 0, nb_x_ch),
                w_band_amp = |p| FixedVec::write_from_value(p, 0.0, nb_x_ch),
            });
        }
    }

    /// Reset all decoder state (equivalent to libopus `OPUS_RESET_STATE`).
    /// Used at SILK↔CELT mode transitions to avoid cross-mode artifacts.
    /// The stream channel count and the band range are kept, as in libopus.
    pub fn reset_state(&mut self) {
        self.rng = 0;
        self.last_pitch_index = 0;
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FrameType::None;
        self.skip_plc = true;
        self.prefilter_period = 0;
        self.prefilter_period_old = 0;
        self.prefilter_gain = 0.0;
        self.prefilter_gain_old = 0.0;
        self.prefilter_tapset = 0;
        self.prefilter_tapset_old = 0;
        self.prefilter_and_fold = false;
        self.preemph_mem.fill(0.0);
        self.hist_pos = 0;
        self.decode_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.old_band_e2.fill(OLD_LOG_E_RESET);
        self.old_band_e3.fill(OLD_LOG_E_RESET);
        self.background_log_e.fill(0.0);
        self.plc_lpc.fill(0.0);
        #[cfg(feature = "deep-plc")]
        {
            self.plc_pcm.fill(0);
            self.plc_fill = 0;
            self.plc_preemphasis_mem = 0.0;
        }
    }

    /// Set the decoder complexity, 0 to 10 (libopus `OPUS_SET_COMPLEXITY`
    /// on the CELT decoder, which the Opus decoder forwards). Other values
    /// are refused and change nothing. From 5, a lost CELT-only frame is
    /// concealed with deep PLC, when the caller passes the deep PLC state
    /// (only `OpusDecoder` does). A reset keeps the value.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        if !(0..=10).contains(&complexity) {
            return Err("Invalid complexity");
        }
        self.complexity = complexity;
        Ok(())
    }

    /// The complexity set with `set_complexity` (0 at the start).
    pub fn complexity(&self) -> i32 {
        self.complexity
    }

    /// libopus `OPUS_GET_PITCH` on the CELT decoder: the post-filter period
    /// of the last frame (`postfilter_period`), 0 when it had none.
    pub fn pitch(&self) -> i32 {
        self.prefilter_period as i32
    }

    /// Phase inversion disabling (libopus `OPUS_SET_PHASE_INVERSION_DISABLED`
    /// on the CELT decoder). With `true`, the second channel of an
    /// intensity stereo band is not inverted, even when the packet codes
    /// the inversion. A reset keeps the value.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.disable_inv = disabled;
    }

    /// The value of `set_phase_inversion_disabled`: `true` at the start
    /// for one output channel, `false` for two.
    pub fn phase_inversion_disabled(&self) -> bool {
        self.disable_inv
    }

    /// Set how many channels the next packets carry (libopus
    /// `CELT_SET_CHANNELS`). It starts equal to the decoder's own channel
    /// count and stays until it is set again. A CELT frame does not say how
    /// many channels it codes: the caller reads that from the Opus packet's
    /// first byte and sets it here before decoding the frame.
    ///
    /// When the two counts differ the decoder converts, as libopus does. A
    /// mono packet is written to both channels of a two-channel decoder. A
    /// stereo packet is mixed down by a one-channel decoder.
    pub fn set_stream_channels(&mut self, channels: usize) -> Result<(), &'static str> {
        if !(1..=2).contains(&channels) {
            return Err("CELT: stream channels must be 1 or 2");
        }
        self.stream_channels = channels;
        Ok(())
    }

    /// Number of channels the decoder expects in the next packet.
    pub fn stream_channels(&self) -> usize {
        self.stream_channels
    }

    /// The range coder state after the last decoded frame (libopus
    /// `OPUS_GET_FINAL_RANGE` on the CELT decoder). An encoder and a decoder
    /// that agree on a packet end with the same value.
    ///
    /// A frame concealed with noise changes this value, because the noise is
    /// drawn from it. libopus's Opus decoder reports 0 for a lost frame
    /// instead of asking the CELT decoder.
    pub fn final_range(&self) -> u32 {
        self.rng
    }

    /// Set the first band that the next frames code (libopus
    /// `CELT_SET_START_BAND`): 0 for a CELT-only stream, 17 for the CELT
    /// layer of a Hybrid stream. It must lie below the number of bands
    /// (21). It starts at 0 and stays until it is set again.
    pub fn set_start_band(&mut self, start: usize) -> Result<(), &'static str> {
        if start >= self.mode.nb_ebands {
            return Err("CELT: start band out of range");
        }
        self.start_band = start;
        Ok(())
    }

    /// Set the end of the band range that the next frames code (libopus
    /// `CELT_SET_END_BAND`): 13, 17, 19 or 21 for a narrow, wide, super-wide
    /// or full band stream. It must lie between 1 and the number of bands
    /// (21). It starts at 21 and stays until it is set again. Lost frames
    /// are concealed within the same range.
    pub fn set_end_band(&mut self, end: usize) -> Result<(), &'static str> {
        if !(1..=self.mode.nb_ebands).contains(&end) {
            return Err("CELT: end band out of range");
        }
        self.end_band = end;
        Ok(())
    }

    /// Conceal one lost frame of `frame_size` samples per channel, laid out
    /// in `pcm` as `layout` says (libopus `celt_decode_with_ec` with
    /// `data == NULL`). The bands concealed are those that
    /// `set_start_band` and `set_end_band` set.
    ///
    /// A CELT-only frame is concealed by repeating the last pitch period of
    /// the output, for up to 100 ms. After that, after a reset, and for the
    /// CELT layer of a Hybrid stream (start band above 0), the frame is
    /// noise shaped like the background noise.
    pub fn decode_lost(
        &mut self,
        frame_size: usize,
        pcm: &mut [f32],
        layout: PcmLayout,
    ) -> Result<usize, &'static str> {
        self.decode_lost_with_deep_plc(frame_size, pcm, layout, None)
    }

    /// `decode_lost` with the deep PLC state of the decoder (libopus
    /// `celt_decode_with_ec_dred` with `lpcnet` and no data). With the
    /// models loaded, complexity 5 or more and start band 0, a lost frame
    /// is concealed with deep PLC for up to 200 ms of loss.
    pub(crate) fn decode_lost_with_deep_plc(
        &mut self,
        frame_size: usize,
        pcm: &mut [f32],
        layout: PcmLayout,
        deep_plc: Option<&mut DeepPlcArgs>,
    ) -> Result<usize, &'static str> {
        let Some((n, lm)) = self.frame_geometry(frame_size) else {
            return Err("CELT: frame size is not 2.5, 5, 10 or 20 ms");
        };
        if pcm.len() < frame_size * self.channels {
            return Err("CELT: output buffer too small");
        }
        self.conceal(n, lm, deep_plc);
        self.deemphasis(pcm, n, frame_size, layout);
        Ok(frame_size)
    }

    /// The frame length at 48 kHz and its size exponent (`120 << lm`) for an
    /// API frame of `frame_size` samples per channel. `None` if that is not
    /// a CELT frame size.
    fn frame_geometry(&self, frame_size: usize) -> Option<(usize, usize)> {
        // The API frame_size is in output samples. Internally CELT always
        // decodes at 48 kHz, so upscale by the downsample factor (libopus
        // celt_decoder.c: `frame_size *= st->downsample`).
        let n = frame_size.checked_mul(self.downsample)?;
        self.mode.lm_for(n).map(|lm| (n, lm))
    }

    /// Index in `decode_mem` of the first sample of channel `ch`'s history.
    /// The history is the `DECODE_BUFFER_SIZE + overlap` samples from there.
    #[inline(always)]
    fn hist_start(&self, ch: usize) -> usize {
        ch * CELT_HIST_STRIDE + self.hist_pos
    }

    /// Make room for a frame of `n` samples at the end of every channel's
    /// history: the oldest `n` samples leave it (libopus
    /// `OPUS_MOVE(decode_mem, decode_mem + N, ...)`).
    ///
    /// libopus moves the whole history down for every frame. Here the
    /// history slides forward through its part of `decode_mem` instead,
    /// which costs nothing. Only when it reaches the end of that part is it
    /// moved back to the start. With 2.5 ms frames that is one move in nine
    /// frames.
    ///
    /// The `n` samples that enter the history at its end, and the `overlap`
    /// samples after them, hold old values until the caller writes them.
    fn advance_history(&mut self, n: usize) {
        let chan_mem = DECODE_BUFFER_SIZE + CELT_OVERLAP;
        debug_assert!(n <= CELT_DEC_MAX_FRAME && self.hist_pos <= CELT_HIST_SLACK);
        if self.hist_pos + n <= CELT_HIST_SLACK {
            self.hist_pos += n;
        } else {
            for ch in 0..self.channels {
                let base = ch * CELT_HIST_STRIDE;
                let old = base + self.hist_pos;
                self.decode_mem.copy_within(old + n..old + chan_mem, base);
            }
            self.hist_pos = 0;
        }
    }

    /// Decode one CELT frame of `total_bits` bits from `rc` into
    /// `frame_size` samples per channel, laid out in `pcm` as `layout`
    /// says (libopus `celt_decode_with_ec`). The bands decoded are those
    /// that `set_start_band` and `set_end_band` set.
    ///
    /// Returns `frame_size`, or 0 if an argument is not valid: the frame
    /// size is not 2.5, 5, 10 or 20 ms, the output buffer is too small, the
    /// frame is longer than a range coder can hold (2048 bytes), or the band
    /// range holds no band. Nothing is written then.
    ///
    /// A frame of 0 or 1 bytes (`total_bits <= 8`) counts as lost and is
    /// concealed, as in libopus (`data == NULL || len <= 1`).
    pub fn decode(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        frame_size: usize,
        pcm: &mut [f32],
        layout: PcmLayout,
    ) -> usize {
        self.decode_with_deep_plc(rc, total_bits, frame_size, pcm, layout, None)
    }

    /// `decode` with the deep PLC state of the decoder, as the main CELT
    /// call of an Opus frame passes it (libopus `celt_decode_with_ec_dred`
    /// with `lpcnet`). A frame with data ends a deep PLC loss
    /// (`lpcnet->blend = 0`); a lost frame may be concealed with deep PLC
    /// (see `decode_lost_with_deep_plc`).
    pub(crate) fn decode_with_deep_plc(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        frame_size: usize,
        pcm: &mut [f32],
        layout: PcmLayout,
        deep_plc: Option<&mut DeepPlcArgs>,
    ) -> usize {
        let mode = self.mode;
        // `channels` is the number of output channels and `c` the number of
        // channels coded in the packet (libopus `CC` and `C`).
        let channels = self.channels;
        let c = self.stream_channels;
        let nb_ebands = mode.nb_ebands;
        let (start_band, end_band) = (self.start_band, self.end_band);

        // The decode geometry requires frame_size == short_mdct_size << lm.
        // Any other size would decode garbage or index out of bounds, so it
        // is rejected.
        let api_frame_size = frame_size;
        let Some((frame_size, lm)) = self.frame_geometry(frame_size) else {
            return 0;
        };

        // An output buffer that is too small is rejected before any of the
        // frame is decoded.
        if pcm.len() < api_frame_size * channels {
            return 0;
        }
        // A band range without a band is a bad argument. It would divide by
        // zero in the bit allocation.
        if start_band >= end_band {
            return 0;
        }
        // A negative length or one beyond what a range coder can hold is a
        // bad argument (libopus celt_decode_with_ec: `len < 0 || len > 1275`).
        if !(0..=MAX_CELT_FRAME_BITS).contains(&total_bits) {
            return 0;
        }

        // libopus celt_decode_with_ec: `data == NULL || len <= 1` is a lost
        // frame.
        if total_bits <= 8 {
            self.conceal(frame_size, lm, deep_plc);
            self.deemphasis(pcm, frame_size, api_frame_size, layout);
            return api_frame_size;
        }
        // libopus: "This is a bit of a hack just to make sure
        // opus_decode_native() knows we're no longer in PLC."
        #[cfg(feature = "deep-plc")]
        if let Some(p) = deep_plc {
            p.state.clear_blend();
        }
        #[cfg(not(feature = "deep-plc"))]
        let _ = deep_plc;

        // The pitch-based concealment is allowed again once two packets in a
        // row have arrived.
        if self.loss_duration == 0 {
            self.skip_plc = false;
        }

        // Make room for the new frame at the end of each channel's history.
        // Nothing reads the history before the frame is synthesised.
        self.advance_history(frame_size);

        // A mono frame continues from the louder of the two channels of the
        // previous frame.
        if c == 1 {
            for i in 0..nb_ebands {
                self.old_band_e[i] = self.old_band_e[i].max(self.old_band_e[nb_ebands + i]);
            }
        }

        let mut tell = rc.tell();
        let mut silence = false;
        if tell >= total_bits {
            silence = true;
        } else if tell == 1 {
            silence = rc.decode_bit_logp(15);
        }
        if silence {
            // Pretend we've read all remaining bits (libopus celt_decoder.c:1324-1328)
            // so that every subsequent `tell+...<=total_bits` guard correctly
            // sees no budget left and decoding proceeds with defaults.
            tell = total_bits;
            rc.nbits_total += tell - rc.tell();
        }

        // C `postfilter_pitch`: 0 when the frame has no post-filter.
        let mut pitch_index = 0;
        let mut gain1 = 0.0f32;
        let mut prefilter_tapset = 0;

        if start_band == 0 && !silence && rc.tell() + 16 <= total_bits {
            let pf_on = rc.decode_bit_logp(1);
            if pf_on {
                let octave = rc.dec_uint(6);
                pitch_index = ((16 << octave) + rc.dec_bits(4 + octave)) as usize - 1;
                let qg = rc.dec_bits(3);
                if rc.tell() + 2 <= total_bits {
                    prefilter_tapset = rc.decode_icdf(&TAPSET_ICDF, 2);
                }
                gain1 = 0.09375 * (qg as f32 + 1.0);
            }
        }

        let mut is_transient = false;
        if lm > 0 && rc.tell() + 3 <= total_bits {
            is_transient = rc.decode_bit_logp(3);
        }
        let short_blocks = is_transient;

        let intra_ener = if rc.tell() + 3 <= total_bits {
            rc.decode_bit_logp(3)
        } else {
            false
        };

        // The first frame after a loss predicts its band energies from the
        // frames before the loss. Lower that starting point so that a wrong
        // prediction cannot produce a loud artifact (libopus
        // celt_decode_with_ec, "If recovering from packet loss").
        if !intra_ener && self.loss_duration != 0 {
            let missing = (self.loss_duration >> lm).min(10);
            // Shorter frames fluctuate more, so they get a larger margin.
            let safety = match lm {
                0 => 1.5f32,
                1 => 0.5,
                _ => 0.0,
            };
            for ch in 0..CELT_MAX_CHANNELS {
                for i in start_band..end_band {
                    let idx = ch * nb_ebands + i;
                    let e0 = self.old_band_e[idx];
                    let e1 = self.old_band_e2[idx];
                    let e2 = self.old_band_e3[idx];
                    let predicted = if e0 < e1.max(e2) {
                        // The energy was already falling: continue the trend.
                        let slope = (e1 - e0).max(0.5 * (e2 - e0)).min(2.0);
                        (e0 - ((1 + missing) as f32 * slope).max(0.0)).max(-20.0)
                    } else {
                        // Otherwise take the lowest of the last frames.
                        e0.min(e1).min(e2)
                    };
                    self.old_band_e[idx] = predicted - safety;
                }
            }
        }

        unquant_coarse_energy(
            mode,
            start_band,
            end_band,
            &mut self.old_band_e,
            intra_ener,
            rc,
            c,
            lm,
        );
        self.w_tf_res[..nb_ebands].fill(0);
        let tf_res = &mut self.w_tf_res[..nb_ebands];
        tf_decode(start_band, end_band, is_transient, tf_res, lm as i32, rc);

        let spread_decision = if rc.tell() + 4 <= total_bits {
            rc.decode_icdf(&SPREAD_ICDF, 5)
        } else {
            SPREAD_NORMAL
        };

        self.w_cap[..nb_ebands].fill(0);
        let cap = &mut self.w_cap[..nb_ebands];
        for (i, cap_i) in cap.iter_mut().enumerate() {
            let n = (mode.e_bands[i + 1] - mode.e_bands[i]) << lm;
            *cap_i = ((mode.cache.caps[nb_ebands * (2 * lm + c - 1) + i] as i32 + 64)
                * c as i32
                * n as i32)
                >> 2;
        }

        self.w_offsets[..nb_ebands].fill(0);
        let offsets = &mut self.w_offsets[..nb_ebands];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_bitres = total_bits << BITRES;
        let mut tell_frac = rc.tell_frac();
        for i in start_band..end_band {
            let width = c as i32 * (mode.e_bands[i + 1] - mode.e_bands[i]) as i32 * (1 << lm);
            let quanta = (width << BITRES).min((6i32 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_bitres && boost < cap[i] {
                let flag = rc.decode_bit_logp(dynalloc_loop_logp as u32);
                tell_frac = rc.tell_frac();
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_bitres -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = dynalloc_logp.max(2) - 1;
                dynalloc_logp = dynalloc_logp.max(2);
            }
        }

        let alloc_trim = if rc.tell_frac() + (6 << BITRES) <= total_bits_bitres {
            rc.decode_icdf(&TRIM_ICDF, 7)
        } else {
            5
        };
        let anti_collapse_rsv = if is_transient && lm >= 2 {
            let remaining = (total_bits << BITRES) - rc.tell_frac() - 1;
            if remaining >= ((lm as i32 + 2) << BITRES) {
                1i32 << BITRES
            } else {
                0
            }
        } else {
            0
        };

        let mut intensity = 0;
        let mut dual_stereo_val = if c == 2 { 1 } else { 0 };
        let mut balance = 0;
        self.w_pulses[..nb_ebands].fill(0);
        let pulses = &mut self.w_pulses[..nb_ebands];

        let ebands_stereo = nb_ebands * c;
        self.w_fine_priority[..ebands_stereo].fill(0);
        let fine_priority = &mut self.w_fine_priority[..ebands_stereo];
        self.w_ebits[..ebands_stereo].fill(0);
        let ebits = &mut self.w_ebits[..ebands_stereo];

        let alloc_bits = (total_bits << BITRES) - rc.tell_frac() - 1 - anti_collapse_rsv;
        let coded_bands = clt_compute_allocation(
            mode,
            start_band,
            end_band,
            offsets,
            cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo_val,
            alloc_bits,
            &mut balance,
            pulses,
            ebits,
            fine_priority,
            c as i32,
            lm as i32,
            rc,
            false,
            0,
            end_band as i32 - 1,
        );

        unquant_fine_energy(
            mode,
            start_band,
            end_band,
            &mut self.old_band_e,
            ebits,
            rc,
            c,
        );

        // `w_x` is not cleared. `quant_all_bands` writes every band from
        // `start_band` to `end_band` before anything reads it, and nothing
        // reads the other bands.
        let x_pad_end = (frame_size * c + STRIDE_ACCESS_PAD).min(self.w_x.len());
        let x = &mut self.w_x[..x_pad_end];
        self.w_collapse_masks[..nb_ebands * c].fill(0);
        let collapse_masks = &mut self.w_collapse_masks[..nb_ebands * c];

        let (x_split, y_split) = x.split_at_mut(frame_size);
        let y_opt = if c == 2 { Some(y_split) } else { None };

        let mut dual_stereo = dual_stereo_val != 0;
        self.w_band_amp[..nb_ebands * c].fill(0.0);
        let band_amp = &mut self.w_band_amp[..nb_ebands * c];
        log2amp(mode, nb_ebands, band_amp, &self.old_band_e, c);
        quant_all_bands(
            false,
            mode,
            start_band,
            end_band,
            x_split,
            y_opt,
            collapse_masks,
            band_amp,
            pulses,
            short_blocks,
            spread_decision,
            &mut dual_stereo,
            intensity as usize,
            tf_res,
            (total_bits << BITRES) - anti_collapse_rsv,
            &mut balance,
            rc,
            lm as i32,
            coded_bands,
            true,
            self.disable_inv,
            &mut self.rng,
        );
        let mut anti_collapse_on = false;
        if anti_collapse_rsv > 0 {
            anti_collapse_on = rc.dec_bits(1) != 0;
        }
        unquant_energy_finalise(
            mode,
            start_band,
            end_band,
            &mut self.old_band_e,
            ebits,
            fine_priority,
            total_bits - rc.tell(),
            rc,
            c,
        );
        if anti_collapse_on {
            self.rng = crate::bands::anti_collapse(
                mode,
                x,
                collapse_masks,
                lm as i32,
                c,
                frame_size,
                start_band,
                end_band,
                &self.old_band_e,
                &self.old_band_e2,
                &self.old_band_e3,
                pulses,
                self.rng,
            );
        }
        if silence {
            // libopus celt_decoder.c:1530-1534 — silence frames carry no energy;
            // force the long-term predictor to a noise floor so the next packet
            // decodes cleanly. Without this the decoder's `oldBandE` retains
            // the previous frame's energies and the next decode diverges, which
            // is the reported "一時的に破壊" (temporarily destroyed) symptom.
            for v in self.old_band_e[..c * nb_ebands].iter_mut() {
                *v = -28.0;
            }
        }

        if self.prefilter_and_fold {
            self.fold_concealed_overlap(frame_size);
        }
        self.synthesis(c, lm, is_transient, start_band, end_band, silence);

        // C 1538-1539. The comb filter limits the periods itself; these
        // lines keep the stored values those of libopus.
        self.prefilter_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
        self.prefilter_period_old = self.prefilter_period_old.max(COMBFILTER_MINPERIOD);
        self.postfilter(frame_size, lm, pitch_index, gain1, prefilter_tapset);
        self.prefilter_period_old = self.prefilter_period;
        self.prefilter_gain_old = self.prefilter_gain;
        self.prefilter_tapset_old = self.prefilter_tapset;
        self.prefilter_period = pitch_index;
        self.prefilter_gain = gain1;
        self.prefilter_tapset = prefilter_tapset;
        if lm != 0 {
            self.prefilter_period_old = self.prefilter_period;
            self.prefilter_gain_old = self.prefilter_gain;
            self.prefilter_tapset_old = self.prefilter_tapset;
        }

        // A mono frame leaves the same energies in both channels.
        if c == 1 {
            self.old_band_e.copy_within(0..nb_ebands, nb_ebands);
        }

        if !is_transient {
            self.old_band_e3.copy_from_slice(&self.old_band_e2);
            self.old_band_e2.copy_from_slice(&self.old_band_e);
        } else {
            for i in 0..CELT_MAX_CHANNELS * nb_ebands {
                self.old_band_e2[i] = self.old_band_e2[i].min(self.old_band_e[i]);
            }
        }
        // The background noise estimate may rise by 2.4 dB per second. After
        // a pause in transmission the first frame carries the weight of all
        // the frames that were not sent.
        let max_background_increase = (self.loss_duration + (1 << lm)).min(160) as f32 * 0.001;
        for i in 0..CELT_MAX_CHANNELS * nb_ebands {
            self.background_log_e[i] =
                (self.background_log_e[i] + max_background_increase).min(self.old_band_e[i]);
        }
        // libopus celt_decode_with_ec, "In case start or end were to
        // change": bands this frame did not code carry no energy history.
        // Without this, the first frame after a switch to a wider bandwidth
        // (or from Hybrid to CELT-only) predicts its new bands from stale
        // energies and decodes them at the wrong level.
        for ch in 0..CELT_MAX_CHANNELS {
            for i in (0..start_band).chain(end_band..nb_ebands) {
                self.old_band_e[ch * nb_ebands + i] = 0.0;
                self.old_band_e2[ch * nb_ebands + i] = OLD_LOG_E_RESET;
                self.old_band_e3[ch * nb_ebands + i] = OLD_LOG_E_RESET;
            }
        }

        self.rng = rc.rng;

        self.deemphasis(pcm, frame_size, api_frame_size, layout);
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FrameType::Normal;
        self.prefilter_and_fold = false;

        api_frame_size
    }

    /// Turn the normalised spectrum in `w_x` and the band energies into time
    /// samples at the end of each channel's history. Port of libopus
    /// `celt_synthesis`. `c` is the number of channels in `w_x`.
    ///
    /// The history must already be shifted by one frame. The first
    /// `overlap` samples of the frame are completed from what the previous
    /// frame left there, and `overlap` unfinished samples follow the frame.
    fn synthesis(
        &mut self,
        c: usize,
        lm: usize,
        is_transient: bool,
        start: usize,
        eff_end: usize,
        silence: bool,
    ) {
        let mode = self.mode;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let n = mode.short_mdct_size << lm;
        let m = 1usize << lm;
        // Number of transforms, their length, and the transform size index.
        let (b, nb, shift) = if is_transient {
            (m, mode.short_mdct_size, mode.max_lm)
        } else {
            (1, n, mode.max_lm - lm)
        };
        let chan_mem = DECODE_BUFFER_SIZE + overlap;
        let out_syn = DECODE_BUFFER_SIZE - n;
        let hist = [self.hist_start(0), self.hist_start(1)];

        // Spectrum of every coded channel (libopus `denormalise_bands`), in
        // place of the normalised spectrum: zero below the first band, and
        // from the last band or the output Nyquist frequency upwards.
        // libopus writes it to another buffer. Here nothing reads the
        // normalised spectrum any more, and one buffer less is touched in
        // every frame.
        let freq = &mut self.w_x[..];
        if silence || start >= eff_end {
            freq[..c * n].fill(0.0);
        } else {
            let band_amp = &mut self.w_band_amp[..nb_ebands * c];
            log2amp(mode, nb_ebands, band_amp, &self.old_band_e, c);
            let bound = (m * mode.e_bands[eff_end] as usize).min(n / self.downsample);
            for ch in 0..c {
                let amp = &band_amp[ch * nb_ebands..(ch + 1) * nb_ebands];
                let spectrum = &mut freq[ch * n..(ch + 1) * n];
                denormalise_in_place(mode, spectrum, amp, start, eff_end, lm, bound);
            }
        }

        if channels == 1 && c == 2 {
            // A stereo stream in a mono decoder: mix the two spectra down.
            let (left, right) = freq.split_at_mut(n);
            for (l, r) in left.iter_mut().zip(&right[..n]) {
                *l = 0.5 * *l + 0.5 * *r;
            }
        }

        let mdct = &mode.mdct;
        for ch in 0..channels {
            // A mono stream in a stereo decoder gives the same spectrum to
            // both channels. So does the mix above to its one channel.
            let spectrum = if c == channels { ch * n } else { 0 };
            let mem = &mut self.decode_mem[hist[ch]..hist[ch] + chan_mem];
            for blk in 0..b {
                let out = out_syn + nb * blk;
                mdct.backward(
                    &freq[spectrum + blk..],
                    &mut mem[out..],
                    mode.window,
                    overlap,
                    shift,
                    b,
                );
            }
        }
        // libopus saturates the output here with `SATURATE(x, SIG_SAT)`.
        // In its float build that macro does nothing (celt/arch.h), so
        // there is no limit here either. A damaged frame can decode to
        // samples far above full scale, and the decoder state must then
        // stay the same as in libopus.
    }

    /// Run the pitch post-filter over the new frame of every channel, in
    /// place at the end of the history.
    ///
    /// The first 2.5 ms cross-fade from the filter before last to the last
    /// one. The rest of the frame cross-fades from the last filter to the
    /// one given here.
    fn postfilter(&mut self, n: usize, lm: usize, pitch: usize, gain: f32, tapset: i32) {
        let filters = [
            (self.prefilter_period_old, self.prefilter_gain_old, self.prefilter_tapset_old),
            (self.prefilter_period, self.prefilter_gain, self.prefilter_tapset),
            (pitch, gain, tapset),
        ];
        let chan_mem = DECODE_BUFFER_SIZE + CELT_OVERLAP;
        let pos = self.hist_pos;
        let (mem0, mem1) = self.decode_mem.split_at_mut(CELT_HIST_STRIDE);
        let ch0 = &mut mem0[pos..pos + chan_mem];
        if self.channels == 2 {
            let ch1 = &mut mem1[pos..pos + chan_mem];
            postfilter_frame(self.mode, &mut [ch0, ch1], n, lm, filters);
        } else {
            postfilter_frame(self.mode, &mut [ch0], n, lm, filters);
        }
    }

    /// De-emphasis of the new frame of every channel into `pcm`, keeping
    /// one sample in `downsample` (libopus `deemphasis`).
    fn deemphasis(&mut self, pcm: &mut [f32], n: usize, api_frame_size: usize, layout: PcmLayout) {
        // One choice of the implementation per frame. The segment filter
        // needs a frame length that is a multiple of 32. The standard mode
        // has one for 10 and 20 ms (480 and 960). A mode with another frame
        // length takes the portable filter below.
        #[cfg(target_arch = "aarch64")]
        if n >= DEEMPH_SEGMENTS_MIN && n.is_multiple_of(32) {
            self.deemphasis_in_segments(pcm, n, api_frame_size, layout);
            return;
        }
        let coef = self.mode.preemph[0];
        let ds = self.downsample;
        let store = |o: &mut f32, v: f32| *o = v;
        let add = |o: &mut f32, v: f32| *o += v;
        // The new frame of each channel in `decode_mem`.
        let at0 = self.hist_start(0) + DECODE_BUFFER_SIZE - n;
        if self.channels == 1 {
            // With one channel the three layouts are the same places.
            let x = &self.decode_mem[at0..at0 + n];
            let out = &mut pcm[..api_frame_size];
            let m = self.preemph_mem[0];
            self.preemph_mem[0] = if layout == PcmLayout::InterleavedAdd {
                deemph_mono(x, out, ds, coef, m, add)
            } else {
                deemph_mono(x, out, ds, coef, m, store)
            };
        } else {
            let at1 = at0 + CELT_HIST_STRIDE;
            let x0 = &self.decode_mem[at0..at0 + n];
            let x1 = &self.decode_mem[at1..at1 + n];
            let m = [self.preemph_mem[0], self.preemph_mem[1]];
            let out = &mut pcm[..2 * api_frame_size];
            let m = match layout {
                PcmLayout::Planar => {
                    let (out0, out1) = out.split_at_mut(api_frame_size);
                    let places = out0.iter_mut().zip(out1.iter_mut());
                    deemph_stereo(x0, x1, places, ds, coef, m, store)
                }
                PcmLayout::Interleaved => {
                    deemph_stereo(x0, x1, stereo_pairs(out), ds, coef, m, store)
                }
                PcmLayout::InterleavedAdd => {
                    deemph_stereo(x0, x1, stereo_pairs(out), ds, coef, m, add)
                }
            };
            self.preemph_mem[..2].copy_from_slice(&m);
        }
    }

    /// `deemphasis` for frames of 10 and 20 ms, with the filter computed in
    /// segments (`deemph_segments_neon`). The output is the same.
    ///
    /// The filter writes straight into `pcm` when the output rate is 48 kHz
    /// and the frame replaces what `pcm` holds. When the frame is added to
    /// `pcm`, or the output rate is lower, it writes into `w_x`, which is
    /// free once the frame is synthesised, and a second pass puts the
    /// samples in their places.
    #[cfg(target_arch = "aarch64")]
    fn deemphasis_in_segments(
        &mut self,
        pcm: &mut [f32],
        n: usize,
        api_frame_size: usize,
        layout: PcmLayout,
    ) {
        let coef = self.mode.preemph[0];
        let ds = self.downsample;
        let chan_mem = DECODE_BUFFER_SIZE + CELT_OVERLAP;
        let at = DECODE_BUFFER_SIZE - n;
        let store = |o: &mut f32, v: f32| *o = v;
        let add = |o: &mut f32, v: f32| *o += v;
        let (h0, h1) = (self.hist_start(0), self.hist_start(1));
        let hist0 = &self.decode_mem[h0..h0 + chan_mem];
        let warmup = DEEMPH_WARMUP;
        if self.channels == 1 {
            let m = [self.preemph_mem[0]];
            let m = if ds == 1 && layout != PcmLayout::InterleavedAdd {
                deemph_segments_neon::<1, false>([hist0], at, pcm, n, warmup, coef, m)
            } else {
                let y = &mut self.w_x[..n];
                let m = deemph_segments_neon::<1, false>([hist0], at, y, n, warmup, coef, m);
                let out = &mut pcm[..api_frame_size];
                if layout == PcmLayout::InterleavedAdd {
                    emit_mono(y, out, ds, add);
                } else {
                    emit_mono(y, out, ds, store);
                }
                m
            };
            self.preemph_mem[0] = m[0];
        } else {
            let hist = [hist0, &self.decode_mem[h1..h1 + chan_mem]];
            let m = [self.preemph_mem[0], self.preemph_mem[1]];
            let m = if ds == 1 && layout == PcmLayout::Planar {
                deemph_segments_neon::<2, false>(hist, at, pcm, n, warmup, coef, m)
            } else if ds == 1 && layout == PcmLayout::Interleaved {
                deemph_segments_neon::<2, true>(hist, at, pcm, n, warmup, coef, m)
            } else {
                let y = &mut self.w_x[..2 * n];
                let m = deemph_segments_neon::<2, false>(hist, at, y, n, warmup, coef, m);
                let (y0, y1) = y.split_at(n);
                let out = &mut pcm[..2 * api_frame_size];
                match layout {
                    PcmLayout::Planar => {
                        let (out0, out1) = out.split_at_mut(api_frame_size);
                        emit_stereo(y0, y1, out0.iter_mut().zip(out1.iter_mut()), ds, store);
                    }
                    PcmLayout::Interleaved => emit_stereo(y0, y1, stereo_pairs(out), ds, store),
                    PcmLayout::InterleavedAdd => emit_stereo(y0, y1, stereo_pairs(out), ds, add),
                }
                m
            };
            self.preemph_mem[..2].copy_from_slice(&m);
        }
    }

    /// Prepare the overlap region after a pitch-concealed frame, so that the
    /// next inverse transform blends with it. Port of libopus
    /// `prefilter_and_fold`. `n` is the length of the frame about to be
    /// synthesised, and the history is already shifted by it.
    ///
    /// The concealment left `overlap` extra samples after its frame. They
    /// are finished output, but the next frame's post-filter will run over
    /// them again, so the inverse of that filter is applied first. They are
    /// then folded with the window, the way the inverse transform leaves
    /// its own unfinished samples.
    fn fold_concealed_overlap(&mut self, n: usize) {
        let mode = self.mode;
        let overlap = mode.overlap;
        let chan_mem = DECODE_BUFFER_SIZE + overlap;
        let at = DECODE_BUFFER_SIZE - n;
        let mut etmp = [0.0f32; CELT_OVERLAP];
        let etmp = &mut etmp[..overlap];
        for ch in 0..self.channels {
            let hist = self.hist_start(ch);
            let buf = &mut self.decode_mem[hist..hist + chan_mem];
            comb_filter(
                etmp,
                buf,
                0,
                at,
                self.prefilter_period_old,
                self.prefilter_period,
                overlap,
                -self.prefilter_gain_old,
                -self.prefilter_gain,
                self.prefilter_tapset_old,
                self.prefilter_tapset,
                &[],
                0,
            );
            for i in 0..overlap / 2 {
                buf[at + i] = mac(
                    mode.window[overlap - i - 1] * etmp[i],
                    mode.window[i],
                    etmp[overlap - 1 - i],
                );
            }
        }
    }

    /// Find the pitch period of the output history for the concealment
    /// (libopus `celt_plc_pitch_search`).
    fn plc_pitch_search(&mut self) -> usize {
        let (hist0, hist1) = (self.hist_start(0), self.hist_start(1));
        // `w_x` is free: the concealment uses it only after the search.
        let lp = &mut self.w_x[..DECODE_BUFFER_SIZE >> 1];
        let ch0 = &self.decode_mem[hist0..hist0 + DECODE_BUFFER_SIZE];
        let ch1 = if self.channels == 2 {
            Some(&self.decode_mem[hist1..hist1 + DECODE_BUFFER_SIZE])
        } else {
            None
        };
        match ch1 {
            Some(ch1) => pitch_downsample(&[ch0, ch1], lp, DECODE_BUFFER_SIZE >> 1, 2, 2),
            None => pitch_downsample(&[ch0], lp, DECODE_BUFFER_SIZE >> 1, 1, 2),
        }
        let lag = pitch_search(
            &lp[PLC_PITCH_LAG_MAX >> 1..],
            lp,
            DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
            PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
        );
        // The lag is below PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN, so the
        // period is within the two limits.
        PLC_PITCH_LAG_MAX - lag
    }

    /// Whether the last frame was concealed with deep PLC.
    #[inline]
    fn last_frame_neural(&self) -> bool {
        #[cfg(feature = "dred")]
        {
            self.last_frame_type == FrameType::PlcNeural || self.last_frame_type == FrameType::Dred
        }
        #[cfg(all(feature = "deep-plc", not(feature = "dred")))]
        {
            self.last_frame_type == FrameType::PlcNeural
        }
        #[cfg(not(feature = "deep-plc"))]
        {
            false
        }
    }

    /// Produce one concealed frame of `n` samples (at 48 kHz) at the end of
    /// each channel's history. Port of libopus `celt_decode_lost`.
    ///
    /// There are three methods. Shaped noise conceals the CELT layer of a
    /// Hybrid stream, the first frames after a reset, and losses longer
    /// than 100 ms. Otherwise the last pitch period is repeated. With deep
    /// PLC (`deep_plc` given, its models loaded, complexity 5 or more),
    /// deep PLC conceals instead, for up to 200 ms of loss.
    fn conceal(&mut self, n: usize, lm: usize, deep_plc: Option<&mut DeepPlcArgs>) {
        let mode = self.mode;
        let start = self.start_band;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let chan_mem = DECODE_BUFFER_SIZE + overlap;
        let at = DECODE_BUFFER_SIZE - n;
        let loss_duration = self.loss_duration;

        #[allow(unused_mut)]
        let mut noise_based = self.plc_duration >= 40 || start != 0 || self.skip_plc;
        // libopus celt_decoder.c:726-731: deep PLC takes over from the
        // other two methods, also from noise between 100 and 200 ms of
        // loss.
        //
        // With the feature `dred`, queued DRED features take over whatever
        // the complexity and the length of the loss (libopus
        // celt_decoder.c:731-734, `FRAME_DRED`).
        #[cfg(feature = "deep-plc")]
        let (mut neural, dred_frame) = match deep_plc {
            Some(p) if start == 0 && p.models.is_some() => {
                #[cfg(feature = "dred")]
                let dred = p.state.fec_fill_pos() > p.state.fec_cursor().0;
                #[cfg(not(feature = "dred"))]
                let dred = false;
                if dred || (self.complexity >= 5 && self.plc_duration < 80 && !self.skip_plc) {
                    (Some(p), dred)
                } else {
                    (None, false)
                }
            }
            _ => (None, false),
        };
        #[cfg(feature = "deep-plc")]
        let curr_neural = neural.is_some();
        #[cfg(not(feature = "deep-plc"))]
        let (curr_neural, _) = (false, deep_plc);
        if curr_neural {
            noise_based = false;
        }
        let last_neural = self.last_frame_neural();

        if noise_based {
            // Noise with the spectral shape of the last frame, decaying
            // towards the background noise level.
            let end = self.end_band;
            let eff_end = start.max(end.min(mode.eff_ebands));
            self.advance_history(n);
            if self.prefilter_and_fold {
                self.fold_concealed_overlap(n);
            }

            let decay = if loss_duration == 0 { 1.5f32 } else { 0.5 };
            for ch in 0..channels {
                for i in start..end {
                    let idx = ch * nb_ebands + i;
                    self.old_band_e[idx] =
                        self.background_log_e[idx].max(self.old_band_e[idx] - decay);
                }
            }
            let mut seed = self.rng;
            for ch in 0..channels {
                for i in start..eff_end {
                    let boffs = n * ch + ((mode.e_bands[i] as usize) << lm);
                    let blen = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
                    let band = &mut self.w_x[boffs..boffs + blen];
                    for v in band.iter_mut() {
                        seed = crate::bands::celt_lcg_rand(seed);
                        *v = ((seed as i32) >> 20) as f32;
                    }
                    crate::bands::renormalise_vector(band, blen, 1.0);
                }
            }
            self.rng = seed;

            self.synthesis(channels, lm, false, start, eff_end, false);

            // Run the post-filter with the last parameters. C 790-791 first
            // raise the periods to at least 15.
            self.prefilter_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
            self.prefilter_period_old = self.prefilter_period_old.max(COMBFILTER_MINPERIOD);
            self.postfilter(
                n,
                lm,
                self.prefilter_period,
                self.prefilter_gain,
                self.prefilter_tapset,
            );
            self.prefilter_period_old = self.prefilter_period;
            self.prefilter_gain_old = self.prefilter_gain;
            self.prefilter_tapset_old = self.prefilter_tapset;

            self.prefilter_and_fold = false;
            // Skip the pitch-based concealment until two packets in a row
            // have arrived.
            self.skip_plc = true;
        } else {
            // Repeat the last pitch period of the output.
            let first_loss =
                self.last_frame_type != FrameType::PlcPeriodic && !(last_neural && curr_neural);
            let mut fade = 1.0f32;
            let pitch_index = if first_loss {
                self.last_pitch_index = self.plc_pitch_search();
                self.last_pitch_index
            } else {
                fade = 0.8;
                self.last_pitch_index
            };
            // libopus celt_decoder.c:829-831: the first frame of deep PLC
            // feeds it the last 40 ms of the history, which must be read
            // before the history moves.
            #[cfg(feature = "deep-plc")]
            if let Some(p) = neural.as_deref_mut()
                && !last_neural
            {
                let h0 = self.hist_start(0);
                let h1 = self.hist_start(1);
                let mem = &self.decode_mem;
                let ch0 = &mem[h0..h0 + DECODE_BUFFER_SIZE];
                if channels == 2 {
                    let ch1 = &mem[h1..h1 + DECODE_BUFFER_SIZE];
                    p.state.update_from_celt(&[ch0, ch1], &mut self.plc_preemphasis_mem);
                } else {
                    p.state.update_from_celt(&[ch0], &mut self.plc_preemphasis_mem);
                }
            }

            // The excitation of two pitch periods shows whether the signal
            // is decaying. No more than PLC_MAX_PERIOD samples are kept.
            let exc_length = (2 * pitch_index).min(PLC_MAX_PERIOD);
            let window = mode.window;
            // Shift the history by one frame. The last decoded sample is
            // now just before `at`. The frame and the `overlap` samples
            // after it are written below.
            self.advance_history(n);
            // After a frame of deep PLC, deep PLC replaces all that the
            // pitch period repetition would write, and libopus uses none of
            // it (its `buf_copy` serves only the first frame of deep PLC),
            // so the repetition is skipped. It changes no other state then.
            let channels_to_extrapolate = if last_neural && curr_neural { 0 } else { channels };
            for ch in 0..channels_to_extrapolate {
                let hist = self.hist_start(ch);
                let buf = &mut self.decode_mem[hist..hist + chan_mem];
                // `exc_all[PLC_LPC_ORDER + i]` is libopus `exc[i]`. The
                // samples before it are the history of the analysis filter.
                let (exc_all, tmp) = self.w_x[..2 * PLC_MAX_PERIOD + PLC_LPC_ORDER]
                    .split_at_mut(PLC_MAX_PERIOD + PLC_LPC_ORDER);
                exc_all.copy_from_slice(&buf[at - PLC_MAX_PERIOD - PLC_LPC_ORDER..at]);
                let lpc = &mut self.plc_lpc[ch * PLC_LPC_ORDER..(ch + 1) * PLC_LPC_ORDER];

                if first_loss {
                    // Predictor for the last PLC_MAX_PERIOD samples before
                    // the loss (libopus `_celt_autocorr` with the window,
                    // then `_celt_lpc`).
                    // The window is applied to a copy here, which saves
                    // the stack copy that `autocorr` makes for a window.
                    let xx = &mut tmp[..PLC_MAX_PERIOD];
                    xx.copy_from_slice(&exc_all[PLC_LPC_ORDER..]);
                    for i in 0..overlap {
                        xx[i] *= window[i];
                        xx[PLC_MAX_PERIOD - 1 - i] *= window[i];
                    }
                    let mut ac = [0.0f32; PLC_LPC_ORDER + 1];
                    autocorr(xx, &mut ac, None, 0, PLC_LPC_ORDER, PLC_MAX_PERIOD);
                    // Noise floor at -40 dB.
                    ac[0] *= 1.0001;
                    // Lag windowing keeps the recursion stable.
                    for i in 1..=PLC_LPC_ORDER {
                        let lag = i as f32;
                        ac[i] = mac(ac[i], -(ac[i] * (0.008f32 * 0.008f32) * lag), lag);
                    }
                    celt_lpc::lpc(lpc, &ac, PLC_LPC_ORDER);
                }

                // Excitation of the last `exc_length` samples before the
                // loss.
                let exc_start = PLC_LPC_ORDER + PLC_MAX_PERIOD - exc_length;
                celt_fir(exc_all, exc_start, lpc, &mut tmp[..exc_length]);
                exc_all[exc_start..].copy_from_slice(&tmp[..exc_length]);
                let exc = &exc_all[PLC_LPC_ORDER..];

                // If the excitation is decaying, keep decaying at the same
                // rate, so that the concealment adds no energy.
                let decay = {
                    let (mut e1, mut e2) = (1.0f32, 1.0f32);
                    let decay_length = exc_length >> 1;
                    for i in 0..decay_length {
                        let e = exc[PLC_MAX_PERIOD - decay_length + i];
                        e1 = mac(e1, e, e);
                        let e = exc[PLC_MAX_PERIOD - 2 * decay_length + i];
                        e2 = mac(e2, e, e);
                    }
                    (e1.min(e2) / e2).sqrt()
                };

                // Repeat the excitation with a period of `pitch_index`,
                // scaled down by `decay` on every repetition. A full
                // transform window is needed: the frame and `overlap`
                // samples more.
                let extrapolation_offset = PLC_MAX_PERIOD - pitch_index;
                let extrapolation_len = n + overlap;
                // Losses after the first are faded as well.
                let mut attenuation = fade * decay;
                // Energy of the decoded signal whose excitation is copied.
                let mut s1 = 0.0f32;
                let mut j = 0;
                for i in 0..extrapolation_len {
                    if j >= pitch_index {
                        j -= pitch_index;
                        attenuation *= decay;
                    }
                    buf[at + i] = attenuation * exc[extrapolation_offset + j];
                    let past = buf[at - PLC_MAX_PERIOD + extrapolation_offset + j];
                    s1 = mac(s1, past, past);
                    j += 1;
                }

                // Back to the signal domain. The filter continues from the
                // last decoded samples, so the signal stays continuous.
                celt_iir(buf, at, extrapolation_len, lpc);

                // The synthesis can come out louder than the signal it
                // copies, when the signal changed inside the analysed
                // window. Attenuate it then.
                let mut s2 = 0.0f32;
                for &v in buf[at..at + extrapolation_len].iter() {
                    s2 = mac(s2, v, v);
                }
                // The test is written this way so that a NaN from the filter
                // counts as too loud.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                let too_loud = !(s1 > 0.2 * s2);
                if too_loud {
                    buf[at..at + extrapolation_len].fill(0.0);
                } else if s1 < s2 {
                    let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                    for i in 0..overlap {
                        let tmp_g = mac(1.0, -window[i], 1.0 - ratio);
                        buf[at + i] *= tmp_g;
                    }
                    for v in buf[at + overlap..at + extrapolation_len].iter_mut() {
                        *v *= ratio;
                    }
                }
            }
            #[cfg(feature = "deep-plc")]
            if let Some(p) = neural {
                self.conceal_neural(n, last_neural, p);
            }
            self.prefilter_and_fold = true;
        }

        // Saturate the counters so they cannot wrap around.
        self.loss_duration = (loss_duration + (1 << lm)).min(10000);
        self.plc_duration = (self.plc_duration + (1 << lm)).min(10000);
        self.last_frame_type = if noise_based {
            FrameType::PlcNoise
        } else {
            FrameType::PlcPeriodic
        };
        #[cfg(feature = "deep-plc")]
        if curr_neural {
            self.last_frame_type = FrameType::PlcNeural;
        }
        // A frame made from DRED features starts the counting of the loss
        // anew.
        #[cfg(feature = "deep-plc")]
        if dred_frame {
            self.plc_duration = 0;
            self.skip_plc = false;
            #[cfg(feature = "dred")]
            {
                self.last_frame_type = FrameType::Dred;
            }
        }
    }

    /// The deep PLC part of libopus `celt_decode_lost`
    /// (celt_decoder.c:1020-1075), after the pitch period repetition has
    /// written the frame: deep PLC conceals 16 kHz audio, which replaces
    /// the frame and the `overlap` samples after it.
    ///
    /// 1. Deep PLC fills `plc_pcm` 10 ms at a time until it holds what the
    ///    frame, the resampling filter and the overlap need.
    /// 2. The 16 kHz audio is upsampled to 48 kHz with `SINC_FILTER`
    ///    (three phases, the samples times 3) into channel 0.
    /// 3. Pre-emphasis, continued from `plc_preemphasis_mem` for the frame
    ///    and from the frame's last sample for the overlap.
    /// 4. The whole history of channel 0 is copied to channel 1: deep PLC
    ///    is mono.
    /// 5. On the first frame of deep PLC, its first `overlap` samples are
    ///    cross-faded with those of the pitch period repetition, per
    ///    channel.
    ///
    /// The roundings are those of libopus built without loop
    /// vectorization: each tap of the filter is a fused multiply-add, the
    /// pre-emphasis is one fused multiply-add, and the cross-fade fuses the
    /// product of the pitch-based sample (`(1 - w) * old + (w * new)`).
    #[cfg(feature = "deep-plc")]
    fn conceal_neural(&mut self, n: usize, last_neural: bool, p: &mut DeepPlcArgs) {
        use crate::dnn::freq::fmaf;
        use crate::dnn::plc::LPCNET_FRAME_SIZE;
        use crate::dnn::plc::lpcnet_plc::{SINC_FILTER, SINC_ORDER};
        // libopus `PREEMPHASIS` of `dnn/freq.h`.
        const PREEMPHASIS: f32 = 0.85;
        let Some(models) = p.models else {
            unreachable!("deep PLC runs only with its models");
        };
        let channels = self.channels;
        let overlap = CELT_OVERLAP;
        let chan_mem = DECODE_BUFFER_SIZE + overlap;
        let at = DECODE_BUFFER_SIZE - n;
        let window = self.mode.window;

        // The pitch-based start of each channel, for the cross-fade.
        let mut buf_copy = [0.0f32; CELT_MAX_CHANNELS * CELT_OVERLAP];
        for ch in 0..channels {
            let h = self.hist_start(ch);
            buf_copy[ch * overlap..(ch + 1) * overlap]
                .copy_from_slice(&self.decode_mem[h + at..h + at + overlap]);
        }

        // Enough 16 kHz samples for the frame, the delay of the filter and
        // the overlap.
        let needed = (n + SINC_ORDER + overlap) / 3;
        if !last_neural {
            self.plc_fill = 0;
        }
        while self.plc_fill < needed {
            let fill = self.plc_fill;
            let chunk: &mut [i16; LPCNET_FRAME_SIZE] = (&mut self.plc_pcm
                [fill..fill + LPCNET_FRAME_SIZE])
                .try_into()
                .unwrap_or_else(|_| unreachable!("a slice of LPCNET_FRAME_SIZE samples"));
            p.state.conceal(models, p.arch, chunk);
            self.plc_fill += LPCNET_FRAME_SIZE;
        }

        // Upsample to 48 kHz. C: `sum += 3*plc_pcm[k]*sinc_filter[m]`, the
        // integer product converted to float.
        let h0 = self.hist_start(0);
        let pcm = &self.plc_pcm;
        let buf = &mut self.decode_mem[h0..h0 + chan_mem];
        let x = |k: usize| (3 * pcm[k] as i32) as f32;
        for i in 0..(n + overlap) / 3 {
            let mut sum = 0.0f32;
            for j in 0..17 {
                sum = fmaf(x(i + j), SINC_FILTER[3 * j], sum);
            }
            buf[at + 3 * i] = sum;
            let mut sum = 0.0f32;
            for j in 0..16 {
                sum = fmaf(x(i + j + 1), SINC_FILTER[3 * j + 2], sum);
            }
            buf[at + 3 * i + 1] = sum;
            let mut sum = 0.0f32;
            for j in 0..16 {
                sum = fmaf(x(i + j + 1), SINC_FILTER[3 * j + 1], sum);
            }
            buf[at + 3 * i + 2] = sum;
        }
        self.plc_pcm.copy_within(n / 3..self.plc_fill, 0);
        self.plc_fill -= n / 3;

        // Pre-emphasis of the frame, then of the overlap. The overlap
        // continues from the frame without changing the memory, which the
        // next frame continues from.
        let mut mem = self.plc_preemphasis_mem;
        for v in buf[at..DECODE_BUFFER_SIZE].iter_mut() {
            let tmp = *v;
            *v = fmaf(mem, -PREEMPHASIS, tmp);
            mem = tmp;
        }
        self.plc_preemphasis_mem = mem;
        for v in buf[DECODE_BUFFER_SIZE..chan_mem].iter_mut() {
            let tmp = *v;
            *v = fmaf(mem, -PREEMPHASIS, tmp);
            mem = tmp;
        }

        // Deep PLC is mono: channel 1 gets channel 0's whole history.
        if channels == 2 {
            let h1 = self.hist_start(1);
            self.decode_mem.copy_within(h0..h0 + chan_mem, h1);
        }
        // Cross-fade from the pitch period repetition on the first frame.
        if !last_neural {
            for ch in 0..channels {
                let h = self.hist_start(ch);
                let start = &mut self.decode_mem[h + at..h + at + overlap];
                let old = &buf_copy[ch * overlap..(ch + 1) * overlap];
                for ((v, &o), &w) in start.iter_mut().zip(old).zip(window) {
                    *v = fmaf(1.0 - w, o, w * *v);
                }
            }
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::bands::denormalise_bands;
    use crate::{modes, range_coder::RangeCoder};

    /// `denormalise_in_place` gives the bits of `bands::denormalise_bands`
    /// into another buffer followed by the zeroing of libopus
    /// `celt_synthesis`: every frame size, several band ranges and output
    /// rates, log-amplitudes from -30 to 34 (above the cap of 32).
    #[test]
    fn spectrum_in_place_is_that_of_denormalise_bands() {
        let mode = modes::default_mode();
        let nb = mode.nb_ebands;
        let mut seed = 0x5eed_7001u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for lm in 0..=3usize {
            let n = mode.short_mdct_size << lm;
            for (start, end) in [(0usize, 21usize), (0, 13), (0, 17), (17, 21), (17, 19), (5, 6)] {
                for ds in [1usize, 2, 3, 4, 6] {
                    let bound = ((mode.e_bands[end] as usize) << lm).min(n / ds);
                    let x: Vec<f32> =
                        (0..n).map(|_| (next() as i32 >> 8) as f32 / 8_388_608.0).collect();
                    let amp: Vec<f32> =
                        (0..nb).map(|_| (next() >> 16) as f32 / 1024.0 - 30.0).collect();

                    let mut want = vec![f32::NAN; n];
                    denormalise_bands(mode, &x, &mut want, &amp, start, end, 1, 1 << lm);
                    let low = (mode.e_bands[start] as usize) << lm;
                    want[..low].fill(0.0);
                    want[bound..].fill(0.0);

                    let mut got = x.clone();
                    denormalise_in_place(mode, &mut got, &amp, start, end, lm, bound);
                    let same = got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(same, "lm {lm} bands {start}..{end} downsample {ds}");
                }
            }
        }
    }

    // A frame of 48 samples is not a CELT frame size (120, 240, 480 or
    // 960 at 48 kHz), and the encoder panics on it. `OpusEncoder::encode`
    // returns an error before it reaches the CELT encoder. Such frames come
    // from 8 kHz audio fed to a 48 kHz encoder without resampling.
    #[test]
    #[should_panic]
    fn test_celt_frame_size_48_panics_confirms_crash_path() {
        let mode = modes::default_mode();
        let mut enc = CeltEncoder::new(mode, 1);
        let pcm = vec![0.0f32; 48 + mode.overlap];
        let mut rc = RangeCoder::new_encoder(100);
        enc.encode(&pcm, 48, &mut rc);
    }

    // A frame whose comb filter is cancelled must still fade the
    // previous frame's filter out over the overlap, because the decoder's
    // post-filter always does. Pitched history into a silent frame: any
    // filtering adds energy, so the filter is cancelled after an "on" frame.
    #[test]
    fn cancelled_prefilter_still_fades_out_previous_filter() {
        let mode = modes::default_mode();
        let (n, overlap, max_period) = (960, mode.overlap, COMBFILTER_MAXPERIOD);
        // The history of the pre-filter, then room for the frame.
        let mut pre = vec![0.0f32; PREFILTER_STRIDE];
        for (i, v) in pre[..max_period].iter_mut().enumerate() {
            *v = (i as f32 * 0.3).sin() * 1000.0;
        }
        let mut in_buf = vec![0.0f32; n + overlap];
        let mut pitch_buf = vec![0.0f32; (max_period + n) >> 1];
        let (mut before, mut after) = ([0.0f32], [0.0f32]);
        let (pf_on, gain, _) = run_prefilter(
            &mut in_buf,
            &mut pre,
            100,
            0.5,
            0,
            0,
            mode.window,
            1,
            n,
            overlap,
            &mut pitch_buf,
            &mut before,
            &mut after,
            0,
            0.0,
            100,
            -1.0,
            0.0,
            9,
            true,
            None,
        );
        assert!(!pf_on && gain == 0.0, "expected a cancelled filter");
        let body = &in_buf[overlap..];
        assert!(
            body[..overlap].iter().any(|&v| v.abs() > 1.0),
            "previous filter was not faded out over the overlap"
        );
        assert!(
            body[overlap..].iter().all(|&v| v == 0.0),
            "frame past the overlap must be the unfiltered input"
        );
    }

    // libopus `OPUS_SET_BITRATE` of the CELT encoder lowers the bitrate to
    // 750 kb/s per channel. No stream tried so far has packets that depend
    // on that limit: a frame never takes more than 1275 bytes, and above
    // the limit the bitrate only moves the VBR target further past that.
    // So the stored value is checked here.
    #[test]
    fn bitrate_is_limited_to_750_kbps_per_channel() {
        let mode = modes::default_mode();
        for channels in [1usize, 2] {
            let mut enc = CeltEncoder::new(mode, channels);
            let limit = 750_000 * channels as i32;
            for (set, kept) in [
                (64_000, 64_000),
                (limit, limit),
                (limit + 1, limit),
                (3_000_000, limit),
                (OPUS_BITRATE_MAX, OPUS_BITRATE_MAX),
            ] {
                enc.set_bitrate(set).unwrap();
                assert_eq!(enc.bitrate, kept, "{channels} channels, set {set}");
            }
        }
    }
}

/// The comb filter against `comb_filter` of the libopus these tests link
/// (celt/celt.c, exported by the static library), in place as the decoder
/// calls it and from one buffer into another as the encoder does.
#[cfg(all(test, feature = "std", target_arch = "aarch64"))]
mod comb_filter_libopus_tests {
    use super::*;
    use audiopus_sys as _;

    unsafe extern "C" {
        /// `void comb_filter(opus_val32 *y, opus_val32 *x, int T0, int T1,
        /// int N, opus_val16 g0, opus_val16 g1, int tapset0, int tapset1,
        /// const celt_coef *window, int overlap, int arch)`, float build.
        #[link_name = "comb_filter"]
        fn c_comb_filter(
            y: *mut f32,
            x: *mut f32,
            t0: i32,
            t1: i32,
            n: i32,
            g0: f32,
            g1: f32,
            tapset0: i32,
            tapset1: i32,
            window: *const f32,
            overlap: i32,
            arch: i32,
        );
    }

    /// Periods, gains (multiples of 0.09375, as the codec quantizes them, 0
    /// included), tapsets and frame lengths of the codec, in many
    /// combinations. Old and new filter are the same in one case of five,
    /// so the cross-fade is skipped there.
    fn cases() -> impl Iterator<Item = (usize, usize, usize, f32, f32, i32, i32)> {
        (0..600usize).map(|round| {
            let n = [960usize, 120, 480, 240, 840, 360][round % 6];
            let t0 = 15 + (round * 37) % 1008;
            let mut t1 = 15 + (round * 53) % 1008;
            let g0 = 0.09375 * (round % 9) as f32;
            let mut g1 = 0.09375 * ((round / 9) % 9) as f32;
            let tapset0 = (round % 3) as i32;
            let mut tapset1 = ((round / 3) % 3) as i32;
            if round % 5 == 0 {
                (t1, g1, tapset1) = (t0, g0, tapset0);
            }
            (n, t0, t1, g0, g1, tapset0, tapset1)
        })
    }

    fn signal(len: usize, seed: &mut u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((*seed >> 8) as f32 / 8_388_608.0 - 1.0) * 3000.0
            })
            .collect()
    }

    #[test]
    fn in_place_filter_is_the_one_of_libopus() {
        let window = crate::modes::default_mode().window;
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let mut seed = 3u32;
        for (n, t0, t1, g0, g1, tapset0, tapset1) in cases() {
            let at = COMBFILTER_MAXPERIOD + 4;
            let input = [signal(at + n, &mut seed), signal(at + n, &mut seed)];
            let mut want = input.clone();
            // SAFETY: the filter reads `t + 2` samples before `at` and
            // writes `n` samples from `at`, all inside `buf`.
            unsafe {
                for buf in want.iter_mut() {
                    let p = buf.as_mut_ptr().add(at);
                    c_comb_filter(p, p, t0 as i32, t1 as i32, n as i32, g0, g1, tapset0, tapset1, window.as_ptr(), CELT_OVERLAP as i32, 0);
                }
            }
            let what = format!("n {n} t0 {t0} t1 {t1} g0 {g0} g1 {g1} tapsets {tapset0} {tapset1}");

            let mut got = input[0].clone();
            comb_filter_inplace(&mut [&mut got[..]], at, t0, t1, n, g0, g1, tapset0, tapset1, window, CELT_OVERLAP);
            assert!(same(&got, &want[0]), "one channel: {what}");

            // The two channels of a stereo frame, filtered together.
            let [mut got0, mut got1] = input;
            let bufs = &mut [&mut got0[..], &mut got1[..]];
            comb_filter_inplace(bufs, at, t0, t1, n, g0, g1, tapset0, tapset1, window, CELT_OVERLAP);
            assert!(same(&got0, &want[0]), "channel 0 of 2: {what}");
            assert!(same(&got1, &want[1]), "channel 1 of 2: {what}");
        }
    }

    #[test]
    fn filter_into_another_buffer_is_the_one_of_libopus() {
        let window = crate::modes::default_mode().window;
        let mut seed = 5u32;
        for (n, t0, t1, g0, g1, tapset0, tapset1) in cases() {
            let at = COMBFILTER_MAXPERIOD + 4;
            let mut x = signal(at + n, &mut seed);
            let mut got = vec![0.0f32; n];
            let mut want = vec![0.0f32; n];
            comb_filter(&mut got, &x, 0, at, t0, t1, n, g0, g1, tapset0, tapset1, window, CELT_OVERLAP);
            // SAFETY: the filter reads `x[at - t - 2..at + n]` and writes
            // `want[..n]`.
            unsafe {
                c_comb_filter(want.as_mut_ptr(), x.as_mut_ptr().add(at), t0 as i32, t1 as i32, n as i32, g0, g1, tapset0, tapset1, window.as_ptr(), CELT_OVERLAP as i32, 0);
            }
            let what = format!("n {n} t0 {t0} t1 {t1} g0 {g0} g1 {g1} tapsets {tapset0} {tapset1}");
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{what}");
        }
    }
}
