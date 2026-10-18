//! NEON versions of the busiest loops of the analysis (64-bit ARM only).
//!
//! Each kernel does, lane by lane, exactly the operations of the scalar
//! code in `tonality.rs`: the same products, sums and divisions, in the same
//! order, with the same fused multiply-adds. A lane only takes work that the
//! scalar code does in a chain of its own (one sample, one FFT bin). So the
//! results are the same bits as the scalar code, which matches libopus. The
//! tests at the end of this file compare the two on random and real input.

use core::arch::aarch64::*;

use super::mlp::{GruLayer, TANSIG_D, TANSIG_N, WEIGHTS_SCALE};
use super::tables::ANALYSIS_WINDOW;
use super::tonality::{
    ANALYSIS_BUF_SIZE, ATAN_A, ATAN_B, ATAN_C, ATAN_E, BinScratch, INV_TWO_PI, N, N2, Phase,
    TONALITY_SCALE,
};
use super::{AnalysisPcm, PcmRef, SecondChannel};
use crate::kiss_fft::{KissCpx, KissFftState};

/// `window_for_fft_scalar` of `tonality.rs`, four samples of each of the
/// four quarters of the window at a time.
pub(super) fn window_for_fft(inmem: &[f32; ANALYSIS_BUF_SIZE], st: &KissFftState, fout: &mut [KissCpx; N]) {
    let scale = st.scale();
    let bitrev: &[i16; N] = st.bitrev[..N].try_into().expect("a plan of 480 points");
    // Stores one complex value at its bit-reversed place. The index is
    // checked: the plan holds a permutation of 0..480.
    let mut put = |index: i16, v: float32x2_t| {
        let c = &mut fout[index as usize];
        // SAFETY: `c` is one complex value, two f32 in a `repr(C)` struct.
        unsafe { vst1_f32((c as *mut KissCpx).cast::<f32>(), v) };
    };
    // SAFETY: NEON is part of every 64-bit ARM processor. For i from 0 to
    // 236 in steps of 4, the loads read ANALYSIS_WINDOW[i..i+4] and inmem
    // at i, N2+i, N-i-4 and N+N2-i-4, four values each, all inside the
    // 240 and 720 values.
    unsafe {
        for i in (0..N2).step_by(4) {
            let w = vld1q_f32(ANALYSIS_WINDOW.as_ptr().add(i));
            let x = inmem.as_ptr();
            // (w * x) * scale, the same two roundings as the scalar code.
            let re = vmulq_n_f32(vmulq_f32(w, vld1q_f32(x.add(i))), scale);
            let im = vmulq_n_f32(vmulq_f32(w, vld1q_f32(x.add(N2 + i))), scale);
            let re_end = vmulq_n_f32(vmulq_f32(w, reverse(vld1q_f32(x.add(N - i - 4)))), scale);
            let im_end = vmulq_n_f32(vmulq_f32(w, reverse(vld1q_f32(x.add(N + N2 - i - 4)))), scale);
            let (lo, hi) = (vzip1q_f32(re, im), vzip2q_f32(re, im));
            put(bitrev[i], vget_low_f32(lo));
            put(bitrev[i + 1], vget_high_f32(lo));
            put(bitrev[i + 2], vget_low_f32(hi));
            put(bitrev[i + 3], vget_high_f32(hi));
            let (lo, hi) = (vzip1q_f32(re_end, im_end), vzip2q_f32(re_end, im_end));
            put(bitrev[N - 1 - i], vget_low_f32(lo));
            put(bitrev[N - 2 - i], vget_high_f32(lo));
            put(bitrev[N - 3 - i], vget_low_f32(hi));
            put(bitrev[N - 4 - i], vget_high_f32(hi));
        }
    }
}

/// The phase-based tonality and the energy of FFT bins 1 to 236, four bins
/// at a time: `tonality_bins_scalar` of `tonality.rs` on four lanes. Returns
/// the first bin that is left to the scalar code (237).
///
/// The two branches of `fast_atan2f` differ only in which of `x*x` and
/// `y*y` takes which place, and in the last two additions. So each lane
/// first picks the larger and the smaller square, and one division serves
/// both branches; the selects at the end give the result of the branch the
/// scalar code takes. Negation is exact, so these are the scalar bits.
pub(super) fn tonality_bins(out: &[KissCpx; N], phase: &mut Phase<'_>, r: &mut BinScratch) -> usize {
    let mut i = 1;
    // SAFETY: `bins4` reads and writes bins i..i+4 and their mirrors, which
    // lie inside the arrays for i from 1 to 236 (see there).
    unsafe {
        // Two groups per pass give a processor that runs instructions in
        // order independent work while the divisions of one group run.
        while i + 8 <= N2 {
            bins4(out, i, phase, r);
            bins4(out, i + 4, phase, r);
            i += 8;
        }
        while i + 4 <= N2 {
            bins4(out, i, phase, r);
            i += 4;
        }
    }
    i
}

/// Bins `i..i+4` of `tonality_bins`.
///
/// # Safety
///
/// `i` must be at least 1 and `i + 4` at most 240. The loads then read the
/// complex values i..i+4 and N-i-3..N-i+1 of `out` (2*i to 2*(N-i)+1 as
/// f32, inside its 960 values), and the loads and stores of the 240-value
/// arrays touch i..i+4.
#[inline(always)]
unsafe fn bins4(out: &[KissCpx; N], i: usize, phase: &mut Phase<'_>, r: &mut BinScratch) {
    debug_assert!((1..=N2 - 4).contains(&i));
    // SAFETY: NEON is part of every 64-bit ARM processor; the caller keeps
    // `i` in range (see above).
    unsafe {
        let p = out.as_ptr().cast::<f32>();
        let one = vdupq_n_f32(1.0);
        let offset = vdupq_n_f32(0.015);
        let a = vld2q_f32(p.add(2 * i));
        let b = vld2q_f32(p.add(2 * (N - i - 3)));
        let (ar, ai) = (a.0, a.1);
        let (br, bi) = (reverse(b.0), reverse(b.1));
        let x1r = vaddq_f32(ar, br);
        let x1i = vsubq_f32(ai, bi);
        let x2r = vaddq_f32(ai, bi);
        let x2i = vsubq_f32(br, ar);

        let old_angle = vld1q_f32(phase.angle.as_ptr().add(i));
        let old_d_angle = vld1q_f32(phase.d_angle.as_ptr().add(i));
        let old_d2_angle = vld1q_f32(phase.d2_angle.as_ptr().add(i));

        let angle = vmulq_n_f32(atan2(x1i, x1r), INV_TWO_PI);
        let d_angle = vsubq_f32(angle, old_angle);
        let d2_angle = vsubq_f32(d_angle, old_d_angle);

        let angle2 = vmulq_n_f32(atan2(x2i, x2r), INV_TWO_PI);
        let d_angle2 = vsubq_f32(angle2, angle);
        let d2_angle2 = vsubq_f32(d_angle2, d_angle);

        // float2int: round to nearest, ties to even (FCVTNS).
        let mut mod1 = vsubq_f32(d2_angle, vcvtq_f32_s32(vcvtnq_s32_f32(d2_angle)));
        let noisiness = vabsq_f32(mod1);
        mod1 = vmulq_f32(mod1, mod1);
        mod1 = vmulq_f32(mod1, mod1);

        let mut mod2 = vsubq_f32(d2_angle2, vcvtq_f32_s32(vcvtnq_s32_f32(d2_angle2)));
        let noisiness = vaddq_f32(noisiness, vabsq_f32(mod2));
        mod2 = vmulq_f32(mod2, mod2);
        mod2 = vmulq_f32(mod2, mod2);

        // mod2 + mod2 is 2*mod2 exactly.
        let avg_mod = vmulq_n_f32(vaddq_f32(vaddq_f32(old_d2_angle, mod1), vaddq_f32(mod2, mod2)), 0.25);
        let tonality = vsubq_f32(vdivq_f32(one, vfmaq_n_f32(one, avg_mod, TONALITY_SCALE)), offset);
        let tonality2 = vsubq_f32(vdivq_f32(one, vfmaq_n_f32(one, mod2, TONALITY_SCALE)), offset);

        vst1q_f32(r.noisiness.as_mut_ptr().add(i), noisiness);
        vst1q_f32(r.tonality.as_mut_ptr().add(i), tonality);
        vst1q_f32(r.tonality2.as_mut_ptr().add(i), tonality2);
        vst1q_f32(phase.angle.as_mut_ptr().add(i), angle2);
        vst1q_f32(phase.d_angle.as_mut_ptr().add(i), d_angle2);
        vst1q_f32(phase.d2_angle.as_mut_ptr().add(i), mod2);

        let energy = vfmaq_f32(vfmaq_f32(vfmaq_f32(vmulq_f32(br, br), ar, ar), ai, ai), bi, bi);
        vst1q_f32(r.energy.as_mut_ptr().add(i), energy);
    }
}

/// The lanes of `v` in reverse order.
#[inline(always)]
unsafe fn reverse(v: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of every 64-bit ARM processor.
    unsafe {
        let v = vrev64q_f32(v);
        vextq_f32::<2>(v, v)
    }
}

/// `fast_atan2f(y, x)` of `tonality.rs` on four lanes, with one division.
#[inline(always)]
unsafe fn atan2(y: float32x4_t, x: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of every 64-bit ARM processor.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let x2 = vmulq_f32(x, x);
        let y2 = vmulq_f32(y, y);
        // The scalar code takes its first branch where x2 < y2.
        let first = vcltq_f32(x2, y2);
        let big = vbslq_f32(first, y2, x2);
        let small = vbslq_f32(first, x2, y2);
        let den = vmulq_f32(vfmaq_n_f32(big, small, ATAN_B), vfmaq_n_f32(big, small, ATAN_C));
        let xy = vmulq_f32(x, y);
        let q = vdivq_f32(vmulq_f32(xy, vfmaq_n_f32(big, small, ATAN_A)), den);
        let pos = vdupq_n_f32(ATAN_E);
        let neg = vdupq_n_f32(-ATAN_E);
        let c1 = vbslq_f32(vcltq_f32(y, zero), neg, pos);
        let c2 = vbslq_f32(vcltq_f32(xy, zero), neg, pos);
        let r = vbslq_f32(first, vsubq_f32(c1, q), vsubq_f32(vaddq_f32(q, c1), c2));
        vbslq_f32(vcltq_f32(vaddq_f32(x2, y2), vdupq_n_f32(1e-18)), zero, r)
    }
}

/// libopus `downmix_float` and `downmix_int` for the layouts that
/// `OpusEncoder` and the stream encoders of a stereo multistream encoder
/// pass: one channel, and two channels summed. Writes `y.len()` samples from
/// sample `offset` on. Returns false, and writes nothing, for any other
/// layout; the scalar code then does the work.
///
/// Float input: `x * 32768`, plus `x2 * 32768` fused into the sum, then the
/// limits -65536 and 65536 and NaN to 0, as the scalar code. 16-bit input:
/// the conversion to `f32` and a plain sum.
pub(super) fn downmix(pcm: &AnalysisPcm<'_>, y: &mut [f32], offset: usize) -> bool {
    let n = y.len();
    let two = match (pcm.channels, pcm.c1, pcm.c2) {
        (1, 0, SecondChannel::None | SecondChannel::AllOthers) => false,
        (2, 0, SecondChannel::AllOthers | SecondChannel::Channel(1)) => true,
        _ => return false,
    };
    let ch = pcm.channels;
    match pcm.pcm {
        PcmRef::F32(x) => {
            let x = &x[offset * ch..offset * ch + n * ch];
            let (blocks, tail) = (n / 4, n % 4);
            // SAFETY: NEON is part of every 64-bit ARM processor. Block `k`
            // reads samples `4 * k * ch` to `4 * (k + 1) * ch - 1` of `x`
            // and writes `y[4 * k..4 * k + 4]`, all inside the slices,
            // whose lengths were checked above.
            unsafe {
                let lo = vdupq_n_f32(-65536.0);
                let hi = vdupq_n_f32(65536.0);
                for k in 0..blocks {
                    let v = if two {
                        let lr = vld2q_f32(x.as_ptr().add(8 * k));
                        vfmaq_n_f32(vmulq_n_f32(lr.0, 32768.0), lr.1, 32768.0)
                    } else {
                        vmulq_n_f32(vld1q_f32(x.as_ptr().add(4 * k)), 32768.0)
                    };
                    vst1q_f32(y.as_mut_ptr().add(4 * k), limit(v, lo, hi));
                }
            }
            for j in n - tail..n {
                let mut v = x[j * ch] * 32768.0;
                if two {
                    v = crate::mathops::mac(v, x[j * ch + 1], 32768.0);
                }
                y[j] = limit_scalar(v);
            }
        }
        PcmRef::I16(x) => {
            let x = &x[offset * ch..offset * ch + n * ch];
            let (blocks, tail) = (n / 8, n % 8);
            // SAFETY: as above, with blocks of 8 samples.
            unsafe {
                for k in 0..blocks {
                    let (a, b) = if two {
                        let lr = vld2q_s16(x.as_ptr().add(16 * k));
                        (
                            vaddq_f32(to_f32(vget_low_s16(lr.0)), to_f32(vget_low_s16(lr.1))),
                            vaddq_f32(to_f32(vget_high_s16(lr.0)), to_f32(vget_high_s16(lr.1))),
                        )
                    } else {
                        let v = vld1q_s16(x.as_ptr().add(8 * k));
                        (to_f32(vget_low_s16(v)), to_f32(vget_high_s16(v)))
                    };
                    vst1q_f32(y.as_mut_ptr().add(8 * k), a);
                    vst1q_f32(y.as_mut_ptr().add(8 * k + 4), b);
                }
            }
            for j in n - tail..n {
                let mut v = x[j * ch] as f32;
                if two {
                    v += x[j * ch + 1] as f32;
                }
                y[j] = v;
            }
        }
    }
    true
}

/// Four 16-bit samples as `f32`.
#[inline(always)]
unsafe fn to_f32(v: int16x4_t) -> float32x4_t {
    // SAFETY: NEON is part of every 64-bit ARM processor.
    unsafe { vcvtq_f32_s32(vmovl_s16(v)) }
}

/// The limits of `downmix_float`, lane by lane: below -65536 becomes
/// -65536, above 65536 becomes 65536, and a NaN becomes 0.
#[inline(always)]
unsafe fn limit(v: float32x4_t, lo: float32x4_t, hi: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of every 64-bit ARM processor.
    unsafe {
        let v = vbslq_f32(vcltq_f32(v, lo), lo, v);
        let v = vbslq_f32(vcgtq_f32(v, hi), hi, v);
        // A NaN is the only value not equal to itself.
        vbslq_f32(vceqq_f32(v, v), v, vdupq_n_f32(0.0))
    }
}

/// The scalar form of `limit`.
// Written as in C, followed by the NaN test.
#[allow(clippy::manual_clamp)]
#[inline(always)]
fn limit_scalar(mut v: f32) -> f32 {
    if v < -65536.0 {
        v = -65536.0;
    }
    if v > 65536.0 {
        v = 65536.0;
    }
    if v.is_nan() { 0.0 } else { v }
}

/// `tansig_approx` of `mlp.rs` on four lanes.
#[inline(always)]
unsafe fn tansig(x: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of every 64-bit ARM processor.
    unsafe {
        let x2 = vmulq_f32(x, x);
        let n = vfmaq_f32(vdupq_n_f32(TANSIG_N[0]), vfmaq_f32(vdupq_n_f32(TANSIG_N[1]), vdupq_n_f32(TANSIG_N[2]), x2), x2);
        let d = vfmaq_f32(vdupq_n_f32(TANSIG_D[0]), vfmaq_f32(vdupq_n_f32(TANSIG_D[1]), vdupq_n_f32(TANSIG_D[2]), x2), x2);
        let r = vdivq_f32(vmulq_f32(n, x), d);
        // The ternaries of C: 1 < r ? 1 : r, then -1 > r ? -1 : r. A NaN
        // stays a NaN.
        let one = vdupq_n_f32(1.0);
        let r = vbslq_f32(vcltq_f32(one, r), one, r);
        let minus_one = vdupq_n_f32(-1.0);
        vbslq_f32(vcgtq_f32(minus_one, r), minus_one, r)
    }
}

/// `sigmoid_approx` of `mlp.rs` on four lanes.
#[inline(always)]
unsafe fn sigmoid(x: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of every 64-bit ARM processor.
    unsafe {
        let half = vdupq_n_f32(0.5);
        vfmaq_f32(half, half, tansig(vmulq_n_f32(x, 0.5)))
    }
}

/// The end of `compute_dense` of `mlp.rs`, four outputs at a time: the
/// scaling by `WEIGHTS_SCALE` and the activation. Returns the outputs left
/// over (fewer than four), which the scalar code does.
pub(super) fn scale_and_activate(output: &mut [f32], sigmoid_output: bool) -> &mut [f32] {
    let (blocks, rest) = output.as_chunks_mut::<4>();
    for b in blocks {
        // SAFETY: NEON is part of every 64-bit ARM processor; `b` holds 4
        // values.
        unsafe {
            let v = vmulq_n_f32(vld1q_f32(b.as_ptr()), WEIGHTS_SCALE);
            let v = if sigmoid_output { sigmoid(v) } else { tansig(v) };
            vst1q_f32(b.as_mut_ptr(), v);
        }
    }
    rest
}

/// The inputs of the recurrent layer of libopus's network.
pub(super) const GRU_INPUTS: usize = 32;
/// Its outputs, which are also its state.
pub(super) const GRU_NEURONS: usize = 24;
/// Vectors of 4 outputs of one gate.
const GATE_VECTORS: usize = GRU_NEURONS / 4;

/// `compute_gru_scalar` of `mlp.rs` for the layer of libopus (32 inputs,
/// 24 outputs), four outputs of a gate per vector.
///
/// The update gate, the reset gate and the input part of the output share
/// one pass over the inputs, and the two gates one pass over the state.
/// Each output still adds its bias, then its input terms in the order of
/// the inputs, then its state terms in the order of the state, each with
/// one fused multiply-add, as in C.
pub(super) fn gru(layer: &GruLayer, state: &mut [f32; GRU_NEURONS], input: &[f32]) {
    const G: usize = 3 * GRU_NEURONS;
    let input: &[f32; GRU_INPUTS] = input[..GRU_INPUTS].try_into().expect("32 inputs");
    let bias: &[f32; G] = layer.bias[..G].try_into().expect("72 biases");
    let w_in: &[f32; GRU_INPUTS * G] = layer.input_weights[..GRU_INPUTS * G].try_into().expect("input weights");
    let w_rec: &[f32; GRU_NEURONS * G] =
        layer.recurrent_weights[..GRU_NEURONS * G].try_into().expect("recurrent weights");
    // SAFETY: NEON is part of every 64-bit ARM processor. Every load and
    // store reads or writes four values inside the arrays above, whose sizes
    // the types fix: gate `g`, vector `k` of row `j` starts at
    // `j * 72 + g * 24 + 4 * k`, at most 71 - 3 within the row.
    unsafe {
        let load = |p: *const f32, k: usize| vld1q_f32(p.add(4 * k));
        let mut z: [float32x4_t; GATE_VECTORS] = core::array::from_fn(|k| load(bias.as_ptr(), k));
        let mut r: [float32x4_t; GATE_VECTORS] = core::array::from_fn(|k| load(bias.as_ptr().add(GRU_NEURONS), k));
        let mut h: [float32x4_t; GATE_VECTORS] =
            core::array::from_fn(|k| load(bias.as_ptr().add(2 * GRU_NEURONS), k));
        for (j, &x) in input.iter().enumerate() {
            let row = w_in.as_ptr().add(j * G);
            for k in 0..GATE_VECTORS {
                z[k] = vfmaq_n_f32(z[k], load(row, k), x);
                r[k] = vfmaq_n_f32(r[k], load(row.add(GRU_NEURONS), k), x);
                h[k] = vfmaq_n_f32(h[k], load(row.add(2 * GRU_NEURONS), k), x);
            }
        }
        for (j, &x) in state.iter().enumerate() {
            let row = w_rec.as_ptr().add(j * G);
            for k in 0..GATE_VECTORS {
                z[k] = vfmaq_n_f32(z[k], load(row, k), x);
                r[k] = vfmaq_n_f32(r[k], load(row.add(GRU_NEURONS), k), x);
            }
        }
        let old: [float32x4_t; GATE_VECTORS] = core::array::from_fn(|k| load(state.as_ptr(), k));
        let mut reset_state = [0.0f32; GRU_NEURONS];
        for k in 0..GATE_VECTORS {
            z[k] = sigmoid(vmulq_n_f32(z[k], WEIGHTS_SCALE));
            r[k] = sigmoid(vmulq_n_f32(r[k], WEIGHTS_SCALE));
            vst1q_f32(reset_state.as_mut_ptr().add(4 * k), vmulq_f32(old[k], r[k]));
        }
        for (j, &x) in reset_state.iter().enumerate() {
            let row = w_rec.as_ptr().add(j * G + 2 * GRU_NEURONS);
            for k in 0..GATE_VECTORS {
                h[k] = vfmaq_n_f32(h[k], load(row, k), x);
            }
        }
        let one = vdupq_n_f32(1.0);
        for k in 0..GATE_VECTORS {
            // z*state + (1-z)*tanh(h): the left product fused, the right one
            // rounded, as in the scalar code.
            let t = vmulq_f32(vsubq_f32(one, z[k]), tansig(vmulq_n_f32(h[k], WEIGHTS_SCALE)));
            vst1q_f32(state.as_mut_ptr().add(4 * k), vfmaq_f32(t, z[k], old[k]));
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::analysis::tonality::{
        downmix_scalar, fft_state, tonality_bins_scalar, window_for_fft_scalar,
    };

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Random values with the specials that the limits and the fused sum
    /// treat in their own way.
    fn samples(n: usize, seed: u64) -> (Vec<f32>, Vec<i16>) {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let specials = [
            0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 1.999_999_9, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            f32::MAX, -f32::MAX, 1e-40, 3.0e34, -3.0e34,
        ];
        let f = (0..n)
            .map(|_| match next() % 8 {
                0 => specials[(next() % specials.len() as u64) as usize],
                _ => ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0,
            })
            .collect();
        let i = (0..n).map(|_| next() as i16).collect();
        (f, i)
    }

    /// The NEON downmix gives the bits of the scalar one for every layout
    /// it covers, every length and offset, and declines the others.
    #[test]
    fn downmix_matches_the_scalar_code() {
        let layouts = [
            (1, 0, SecondChannel::AllOthers, true),
            (1, 0, SecondChannel::None, true),
            (2, 0, SecondChannel::AllOthers, true),
            (2, 0, SecondChannel::Channel(1), true),
            (2, 1, SecondChannel::None, false),
            (2, 0, SecondChannel::None, false),
            (3, 0, SecondChannel::AllOthers, false),
            (6, 2, SecondChannel::Channel(3), false),
        ];
        for (k, &(channels, c1, c2, covered)) in layouts.iter().enumerate() {
            let (f, i) = samples(1000 * channels, k as u64 + 1);
            for n in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 240, 479, 480, 960] {
                for offset in [0usize, 1, 3, 20] {
                    if (offset + n) * channels > f.len() {
                        continue;
                    }
                    for pcm in [PcmRef::F32(&f), PcmRef::I16(&i)] {
                        let p = AnalysisPcm { pcm, channels, c1, c2, frame_size: n };
                        let mut a = vec![7.0f32; n];
                        let mut b = vec![7.0f32; n];
                        assert_eq!(downmix(&p, &mut a, offset), covered);
                        downmix_scalar(&p, &mut b, offset);
                        if covered {
                            let nan = |v: &[f32]| v.iter().any(|x| x.is_nan());
                            assert!(!nan(&a) && !nan(&b));
                            assert_eq!(bits(&a), bits(&b), "layout {k} n {n} offset {offset}");
                        }
                    }
                }
            }
        }
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        let canon = |v: &[f32]| v.iter().map(|x| if x.is_nan() { u32::MAX } else { x.to_bits() }).collect::<Vec<_>>();
        canon(a) == canon(b)
    }

    /// Runs the NEON and the scalar bins on the same input and state and
    /// checks every output and the new state.
    fn check_bins(out: &[KissCpx; N], state: &[[f32; N2]; 3], what: &str) {
        let (mut a, mut b) = (*state, *state);
        let (mut ra, mut rb) = (BinScratch::new(), BinScratch::new());
        {
            let [angle, d_angle, d2_angle] = &mut a;
            let mut phase = Phase { angle, d_angle, d2_angle };
            let first = tonality_bins(out, &mut phase, &mut ra);
            assert_eq!(first, 237);
            tonality_bins_scalar(out, first, &mut phase, &mut ra);
        }
        {
            let [angle, d_angle, d2_angle] = &mut b;
            let mut phase = Phase { angle, d_angle, d2_angle };
            tonality_bins_scalar(out, 1, &mut phase, &mut rb);
        }
        for k in 0..3 {
            assert!(same(&a[k], &b[k]), "{what}: state array {k}");
        }
        assert!(same(&ra.tonality, &rb.tonality), "{what}: tonality");
        assert!(same(&ra.tonality2, &rb.tonality2), "{what}: tonality2");
        assert!(same(&ra.noisiness, &rb.noisiness), "{what}: noisiness");
        assert!(same(&ra.energy, &rb.energy), "{what}: energy");
    }

    /// The NEON bins give the bits of the scalar code: on random spectra
    /// with values of every size, with the special values where
    /// `fast_atan2f` changes branch or returns 0, and on the spectra of
    /// real blocks.
    #[test]
    fn tonality_bins_match_the_scalar_code() {
        let mut s = 0x0bad_cafe_1234_5678u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let specials = [0.0f32, -0.0, 1e-10, -1e-10, 7e-10, 1e-20, 1.0, -1.0, 3.0e18, f32::MIN_POSITIVE];
        for case in 0..400 {
            let mut value = || -> f32 {
                let r = next();
                match r % 10 {
                    0 => specials[((r >> 8) % specials.len() as u64) as usize],
                    1 => 1.0,
                    2 => -1.0,
                    _ => {
                        let m = ((r >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0;
                        let e = ((r >> 16) % 40) as i32 - 20;
                        m * 10f32.powi(e)
                    }
                }
            };
            let out: [KissCpx; N] = core::array::from_fn(|_| KissCpx::new(value(), value()));
            let state: [[f32; N2]; 3] = core::array::from_fn(|k| {
                core::array::from_fn(|_| {
                    let v = ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0;
                    if k == 2 { v * v * 0.01 } else { v }
                })
            });
            check_bins(&out, &state, &format!("random case {case}"));
        }
        // Real blocks: the FFT of a windowed mix of tones and noise, one
        // block after the other, so the phase state is that of real input.
        let fft = crate::modes::default_mode().mdct.kfft(0).unwrap();
        let mut state = [[0.0f32; N2]; 3];
        for block in 0..50 {
            let fin: [KissCpx; N] = core::array::from_fn(|n| {
                let t = (block * 240 + n) as f32;
                let tone = (t * 0.031).sin() * 3000.0 + (t * 0.2093).sin() * 800.0;
                let noise = ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 50.0;
                KissCpx::new(tone + noise, (t * 0.0517).cos() * 1500.0 + noise)
            });
            let mut out = [KissCpx::new(0.0, 0.0); N];
            crate::kiss_fft::opus_fft(fft, &fin, &mut out);
            check_bins(&out, &state, &format!("block {block}"));
            let [angle, d_angle, d2_angle] = &mut state;
            let mut phase = Phase { angle, d_angle, d2_angle };
            tonality_bins_scalar(&out, 1, &mut phase, &mut BinScratch::new());
        }
    }

    /// The NEON window gives the bits and the places of the scalar one.
    #[test]
    fn window_for_fft_matches_the_scalar_code() {
        let mut s = 0x5eed_0000_1111_2222u64;
        for case in 0..20 {
            let inmem: [f32; ANALYSIS_BUF_SIZE] = core::array::from_fn(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 65536.0;
                if case == 0 { f32::from_bits(s as u32) } else { v }
            });
            let mut a = [KissCpx::new(7.0, 7.0); N];
            let mut b = a;
            window_for_fft(&inmem, fft_state(), &mut a);
            window_for_fft_scalar(&inmem, fft_state(), &mut b);
            let flat = |v: &[KissCpx; N]| v.iter().flat_map(|c| [c.r, c.i]).collect::<Vec<_>>();
            assert!(same(&flat(&a), &flat(&b)), "case {case}");
        }
    }

    /// The NEON network gives the bits of the scalar one: the dense
    /// layers' activations and the recurrent layer, on random inputs of
    /// every size, near the limits of the activation and with NaN.
    #[test]
    fn network_matches_the_scalar_code() {
        use crate::analysis::mlp::{LAYER0, LAYER1, LAYER2, compute_gru_scalar, sigmoid_approx, tansig_approx};
        let mut s = 0x0123_4567_89ab_cdefu64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        };
        for case in 0..3000 {
            let scale = [0.01f32, 1.0, 10.0, 300.0, 1e4][case % 5];
            let mut x: [f32; 32] = core::array::from_fn(|_| next() * scale);
            if case % 97 == 0 {
                x[case % 32] = f32::NAN;
            }
            // Activations after the scaling by 1/128, on 4 lanes.
            for sig in [false, true] {
                let mut a = x;
                let rest = scale_and_activate(&mut a[..30], sig).len();
                assert_eq!(rest, 2);
                for (k, (&got, &inp)) in a[..28].iter().zip(&x[..28]).enumerate() {
                    let v = inp * WEIGHTS_SCALE;
                    let want = if sig { sigmoid_approx(v) } else { tansig_approx(v) };
                    assert!(same(&[got], &[want]), "case {case} lane {k} sigmoid {sig}");
                }
            }
            // The recurrent layer, from a random state.
            let state0: [f32; GRU_NEURONS] = core::array::from_fn(|_| next());
            let (mut a, mut b) = (state0, state0);
            gru(&LAYER1, &mut a, &x);
            compute_gru_scalar(&LAYER1, &mut b, &x);
            assert!(same(&a, &b), "case {case}");
        }
        // A run of the whole network, where the state carries over.
        let (mut a, mut b) = ([0.0f32; 32], [0.0f32; 32]);
        for step in 0..2000 {
            let feat: [f32; 25] = core::array::from_fn(|_| next() * 3.0);
            let mut hidden = [0.0f32; 32];
            crate::analysis::mlp::compute_dense(&LAYER0, &mut hidden, &feat);
            gru(&LAYER1, (&mut a[..24]).try_into().unwrap(), &hidden);
            compute_gru_scalar(&LAYER1, &mut b, &hidden);
            assert!(same(&a, &b), "step {step}");
            let mut out = [0.0f32; 2];
            crate::analysis::mlp::compute_dense(&LAYER2, &mut out, &a);
            assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }
}
