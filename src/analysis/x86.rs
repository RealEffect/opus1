//! SSE2 versions of the busiest loops of the analysis (x86-64).
//!
//! Each kernel follows the NEON kernel of the same name in `neon.rs`, lane
//! by lane: a lane only takes work that the scalar code of `tonality.rs`
//! and `mlp.rs` does in a chain of its own (one sample, one FFT bin, one
//! output of a layer). Every x86-64 processor has SSE2, so the kernels run
//! without a check.
//!
//! On x86-64 the scalar code rounds each product before its addition (see
//! [`crate::mathops::mac`]). The kernels do the same, with the same
//! operations in the same order, so their results are the bits of the
//! scalar code. Two operations need a remark:
//!
//! - A select is `_mm_and_ps`, `_mm_andnot_ps` and `_mm_or_ps`. SSE4.1's
//!   `_mm_blendv_ps` would save no time: it takes 3 micro-operations on
//!   recent Intel cores too.
//! - Rounding to the nearest integer is `_mm_cvtps_epi32` and back. It
//!   rounds half-way values to the even integer, as `float2int` of
//!   `tonality.rs` does, because Rust code runs in the default rounding
//!   mode. The two differ only at magnitudes of 2^31 and more. The phase
//!   differences that are rounded stay below 3: each is the difference of
//!   two phase steps, and a phase step is the difference of two angles of
//!   at most about 1/2 turn. A NaN stays a NaN in both.
//!
//! The tests at the end of this file compare the kernels with the scalar
//! code on random and real input.

use core::arch::x86_64::*;

use super::mlp::{GruLayer, TANSIG_D, TANSIG_N, WEIGHTS_SCALE};
use super::tables::ANALYSIS_WINDOW;
use super::tonality::{
    ANALYSIS_BUF_SIZE, ATAN_A, ATAN_B, ATAN_C, ATAN_E, BinScratch, INV_TWO_PI, N, N2, Phase,
    TONALITY_SCALE, downmix_scalar,
};
use super::{AnalysisPcm, PcmRef, SecondChannel};
use crate::kiss_fft::{KissCpx, KissFftState};

/// `_MM_SHUFFLE(2, 0, 2, 0)`: lanes 0 and 2 of each operand, the real parts
/// of two complex values each (or the first channel of two pairs).
const EVEN: i32 = 0x88;
/// `_MM_SHUFFLE(3, 1, 3, 1)`: lanes 1 and 3 of each operand.
const ODD: i32 = 0xDD;

/// `a` where `mask` is set, `b` elsewhere.
#[inline(always)]
fn select(mask: __m128, a: __m128, b: __m128) -> __m128 {
    // SAFETY: SSE is part of this target (see the `cfg` of the module), and
    // these instructions only compute on values held in registers.
    unsafe { _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b)) }
}

/// `acc + a * k`, the product rounded first, as `mac` on x86-64.
#[inline(always)]
fn mul_add(acc: __m128, a: __m128, k: f32) -> __m128 {
    // SAFETY: as in `select`.
    unsafe { _mm_add_ps(acc, _mm_mul_ps(a, _mm_set1_ps(k))) }
}

/// The magnitude of each lane: the sign bit cleared, as `f32::abs`.
#[inline(always)]
fn abs(v: __m128) -> __m128 {
    // SAFETY: as in `select`.
    unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), v) }
}

/// The lanes of `v` in reverse order (`_MM_SHUFFLE(0, 1, 2, 3)`).
#[inline(always)]
fn reverse(v: __m128) -> __m128 {
    // SAFETY: as in `select`.
    unsafe { _mm_shuffle_ps::<0x1B>(v, v) }
}

/// `window_for_fft_scalar` of `tonality.rs`, four samples of each of the
/// four quarters of the window at a time.
pub(super) fn window_for_fft(inmem: &[f32; ANALYSIS_BUF_SIZE], st: &KissFftState, fout: &mut [KissCpx; N]) {
    let scale = st.scale();
    let bitrev: &[i16; N] = st.bitrev[..N].try_into().expect("a plan of 480 points");
    // Stores the two lower lanes of `v` as one complex value at its
    // bit-reversed place. The index is checked: the plan holds a
    // permutation of 0..480.
    let mut put = |index: i16, v: __m128| {
        let c = &mut fout[index as usize];
        // SAFETY: SSE2 is part of this target (see the `cfg` of the
        // module). `c` is one complex value, two f32 in a `repr(C)` struct:
        // the 8 bytes that `_mm_storel_epi64` writes, which need no
        // alignment.
        unsafe { _mm_storel_epi64((c as *mut KissCpx).cast::<__m128i>(), _mm_castps_si128(v)) };
    };
    // SAFETY: SSE is part of this target. For i from 0 to 236 in steps of
    // 4, the loads read ANALYSIS_WINDOW[i..i+4] and inmem at i, N2+i,
    // N-i-4 and N+N2-i-4, four values each, all inside the 240 and 720
    // values.
    unsafe {
        let s = _mm_set1_ps(scale);
        for i in (0..N2).step_by(4) {
            let w = _mm_loadu_ps(ANALYSIS_WINDOW.as_ptr().add(i));
            let x = inmem.as_ptr();
            // (w * x) * scale, the same two roundings as the scalar code.
            let re = _mm_mul_ps(_mm_mul_ps(w, _mm_loadu_ps(x.add(i))), s);
            let im = _mm_mul_ps(_mm_mul_ps(w, _mm_loadu_ps(x.add(N2 + i))), s);
            let re_end = _mm_mul_ps(_mm_mul_ps(w, reverse(_mm_loadu_ps(x.add(N - i - 4)))), s);
            let im_end = _mm_mul_ps(_mm_mul_ps(w, reverse(_mm_loadu_ps(x.add(N + N2 - i - 4)))), s);
            let (lo, hi) = (_mm_unpacklo_ps(re, im), _mm_unpackhi_ps(re, im));
            put(bitrev[i], lo);
            put(bitrev[i + 1], _mm_movehl_ps(lo, lo));
            put(bitrev[i + 2], hi);
            put(bitrev[i + 3], _mm_movehl_ps(hi, hi));
            let (lo, hi) = (_mm_unpacklo_ps(re_end, im_end), _mm_unpackhi_ps(re_end, im_end));
            put(bitrev[N - 1 - i], lo);
            put(bitrev[N - 2 - i], _mm_movehl_ps(lo, lo));
            put(bitrev[N - 3 - i], hi);
            put(bitrev[N - 4 - i], _mm_movehl_ps(hi, hi));
        }
    }
}

/// The phase-based tonality and the energy of FFT bins 1 to 236, four bins
/// at a time: `tonality_bins_scalar` of `tonality.rs` on four lanes. Returns
/// the first bin that is left to the scalar code (237).
///
/// As in the NEON kernel, `atan2` picks the larger and the smaller square
/// in each lane, so that one division serves both branches of
/// `fast_atan2f`, and selects the result of the branch the scalar code
/// takes.
pub(super) fn tonality_bins(out: &[KissCpx; N], phase: &mut Phase<'_>, r: &mut BinScratch) -> usize {
    let mut i = 1;
    // SAFETY: `bins4` reads and writes bins i..i+4 and their mirrors, which
    // lie inside the arrays for i from 1 to 236 (see there).
    unsafe {
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
    // SAFETY: SSE2 is part of this target; the caller keeps `i` in range
    // (see above).
    unsafe {
        let p = out.as_ptr().cast::<f32>();
        let one = _mm_set1_ps(1.0);
        let offset = _mm_set1_ps(0.015);
        // Bins i..i+4, as (re, im) pairs.
        let a0 = _mm_loadu_ps(p.add(2 * i));
        let a1 = _mm_loadu_ps(p.add(2 * i + 4));
        let (ar, ai) = (_mm_shuffle_ps::<EVEN>(a0, a1), _mm_shuffle_ps::<ODD>(a0, a1));
        // Bins N-i-3..N-i+1, taken in reverse order: N-i, N-i-1, N-i-2,
        // N-i-3. `_MM_SHUFFLE(0, 2, 0, 2)` takes lanes 2 and 0 of each
        // operand, `_MM_SHUFFLE(1, 3, 1, 3)` lanes 3 and 1.
        let b0 = _mm_loadu_ps(p.add(2 * (N - i - 3)));
        let b1 = _mm_loadu_ps(p.add(2 * (N - i - 1)));
        let (br, bi) = (_mm_shuffle_ps::<0x22>(b1, b0), _mm_shuffle_ps::<0x77>(b1, b0));
        let x1r = _mm_add_ps(ar, br);
        let x1i = _mm_sub_ps(ai, bi);
        let x2r = _mm_add_ps(ai, bi);
        let x2i = _mm_sub_ps(br, ar);

        let old_angle = _mm_loadu_ps(phase.angle.as_ptr().add(i));
        let old_d_angle = _mm_loadu_ps(phase.d_angle.as_ptr().add(i));
        let old_d2_angle = _mm_loadu_ps(phase.d2_angle.as_ptr().add(i));

        let inv_two_pi = _mm_set1_ps(INV_TWO_PI);
        let angle = _mm_mul_ps(atan2(x1i, x1r), inv_two_pi);
        let d_angle = _mm_sub_ps(angle, old_angle);
        let d2_angle = _mm_sub_ps(d_angle, old_d_angle);

        let angle2 = _mm_mul_ps(atan2(x2i, x2r), inv_two_pi);
        let d_angle2 = _mm_sub_ps(angle2, angle);
        let d2_angle2 = _mm_sub_ps(d_angle2, d_angle);

        // float2int: round to nearest, ties to even (see the top of the
        // file).
        let mut mod1 = _mm_sub_ps(d2_angle, _mm_cvtepi32_ps(_mm_cvtps_epi32(d2_angle)));
        let noisiness = abs(mod1);
        mod1 = _mm_mul_ps(mod1, mod1);
        mod1 = _mm_mul_ps(mod1, mod1);

        let mut mod2 = _mm_sub_ps(d2_angle2, _mm_cvtepi32_ps(_mm_cvtps_epi32(d2_angle2)));
        let noisiness = _mm_add_ps(noisiness, abs(mod2));
        mod2 = _mm_mul_ps(mod2, mod2);
        mod2 = _mm_mul_ps(mod2, mod2);

        // mod2 + mod2 is 2*mod2 exactly.
        let sum = _mm_add_ps(_mm_add_ps(old_d2_angle, mod1), _mm_add_ps(mod2, mod2));
        let avg_mod = _mm_mul_ps(sum, _mm_set1_ps(0.25));
        let tonality = _mm_sub_ps(_mm_div_ps(one, mul_add(one, avg_mod, TONALITY_SCALE)), offset);
        let tonality2 = _mm_sub_ps(_mm_div_ps(one, mul_add(one, mod2, TONALITY_SCALE)), offset);

        _mm_storeu_ps(r.noisiness.as_mut_ptr().add(i), noisiness);
        _mm_storeu_ps(r.tonality.as_mut_ptr().add(i), tonality);
        _mm_storeu_ps(r.tonality2.as_mut_ptr().add(i), tonality2);
        _mm_storeu_ps(phase.angle.as_mut_ptr().add(i), angle2);
        _mm_storeu_ps(phase.d_angle.as_mut_ptr().add(i), d_angle2);
        _mm_storeu_ps(phase.d2_angle.as_mut_ptr().add(i), mod2);

        // ((b*b + a*a) + c*c) + d*d, each product rounded, as the scalar
        // code on x86-64.
        let sq = |v: __m128| _mm_mul_ps(v, v);
        let energy = _mm_add_ps(_mm_add_ps(_mm_add_ps(sq(br), sq(ar)), sq(ai)), sq(bi));
        _mm_storeu_ps(r.energy.as_mut_ptr().add(i), energy);
    }
}

/// `fast_atan2f(y, x)` of `tonality.rs` on four lanes, with one division.
#[inline(always)]
fn atan2(y: __m128, x: __m128) -> __m128 {
    // SAFETY: as in `select`.
    unsafe {
        let zero = _mm_setzero_ps();
        let x2 = _mm_mul_ps(x, x);
        let y2 = _mm_mul_ps(y, y);
        // The scalar code takes its first branch where x2 < y2.
        let first = _mm_cmplt_ps(x2, y2);
        let big = select(first, y2, x2);
        let small = select(first, x2, y2);
        let den = _mm_mul_ps(mul_add(big, small, ATAN_B), mul_add(big, small, ATAN_C));
        let xy = _mm_mul_ps(x, y);
        let q = _mm_div_ps(_mm_mul_ps(xy, mul_add(big, small, ATAN_A)), den);
        let pos = _mm_set1_ps(ATAN_E);
        let neg = _mm_set1_ps(-ATAN_E);
        let c1 = select(_mm_cmplt_ps(y, zero), neg, pos);
        let c2 = select(_mm_cmplt_ps(xy, zero), neg, pos);
        let r = select(first, _mm_sub_ps(c1, q), _mm_sub_ps(_mm_add_ps(q, c1), c2));
        // 0 where x2 + y2 < 1e-18.
        _mm_andnot_ps(_mm_cmplt_ps(_mm_add_ps(x2, y2), _mm_set1_ps(1e-18)), r)
    }
}

/// libopus `downmix_float` and `downmix_int` for the layouts that
/// `OpusEncoder` and the stream encoders of a stereo multistream encoder
/// pass: one channel, and two channels summed. Writes `y.len()` samples from
/// sample `offset` on. Returns false, and writes nothing, for any other
/// layout; the scalar code then does the work.
///
/// Float input: `x * 32768`, plus `x2 * 32768` with the product rounded
/// first, then the limits -65536 and 65536 and NaN to 0, as the scalar
/// code. 16-bit input: the conversion to `f32` and a plain sum. The last
/// samples that do not fill a block are left to `downmix_scalar`.
pub(super) fn downmix(pcm: &AnalysisPcm<'_>, y: &mut [f32], offset: usize) -> bool {
    let n = y.len();
    let two = match (pcm.channels, pcm.c1, pcm.c2) {
        (1, 0, SecondChannel::None | SecondChannel::AllOthers) => false,
        (2, 0, SecondChannel::AllOthers | SecondChannel::Channel(1)) => true,
        _ => return false,
    };
    let ch = pcm.channels;
    // The samples the blocks below compute.
    let done = match pcm.pcm {
        // 24-bit input is rare enough for the scalar code.
        PcmRef::I32(_) => return false,
        PcmRef::F32(x) => {
            let x = &x[offset * ch..offset * ch + n * ch];
            let blocks = n / 4;
            // SAFETY: SSE is part of this target. Block `k` reads samples
            // `4 * k * ch` to `4 * (k + 1) * ch - 1` of `x` and writes
            // `y[4 * k..4 * k + 4]`, all inside the slices, whose lengths
            // were checked above.
            unsafe {
                let scale = _mm_set1_ps(32768.0);
                let lo = _mm_set1_ps(-65536.0);
                let hi = _mm_set1_ps(65536.0);
                for k in 0..blocks {
                    let v = if two {
                        let a = _mm_loadu_ps(x.as_ptr().add(8 * k));
                        let b = _mm_loadu_ps(x.as_ptr().add(8 * k + 4));
                        let (l, r) = (_mm_shuffle_ps::<EVEN>(a, b), _mm_shuffle_ps::<ODD>(a, b));
                        _mm_add_ps(_mm_mul_ps(l, scale), _mm_mul_ps(r, scale))
                    } else {
                        _mm_mul_ps(_mm_loadu_ps(x.as_ptr().add(4 * k)), scale)
                    };
                    _mm_storeu_ps(y.as_mut_ptr().add(4 * k), limit(v, lo, hi));
                }
            }
            4 * blocks
        }
        PcmRef::I16(x) => {
            let x = &x[offset * ch..offset * ch + n * ch];
            let blocks = n / 8;
            // SAFETY: SSE2 is part of this target. Block `k` reads samples
            // `8 * k * ch` to `8 * (k + 1) * ch - 1` of `x` (16 or 32
            // bytes) and writes `y[8 * k..8 * k + 8]`, all inside the
            // slices, whose lengths were checked above.
            unsafe {
                let p = x.as_ptr();
                let o = y.as_mut_ptr();
                for k in 0..blocks {
                    if two {
                        // Each 32-bit lane holds one pair: the first channel
                        // in its lower half, the second in its upper half.
                        for h in 0..2 {
                            let v = _mm_loadu_si128(p.add(16 * k + 8 * h).cast::<__m128i>());
                            let l = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v));
                            let r = _mm_srai_epi32::<16>(v);
                            let sum = _mm_add_ps(_mm_cvtepi32_ps(l), _mm_cvtepi32_ps(r));
                            _mm_storeu_ps(o.add(8 * k + 4 * h), sum);
                        }
                    } else {
                        let v = _mm_loadu_si128(p.add(8 * k).cast::<__m128i>());
                        // Each sample twice, then the arithmetic shift
                        // extends its sign to 32 bits.
                        let a = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(v, v));
                        let b = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(v, v));
                        _mm_storeu_ps(o.add(8 * k), _mm_cvtepi32_ps(a));
                        _mm_storeu_ps(o.add(8 * k + 4), _mm_cvtepi32_ps(b));
                    }
                }
            }
            8 * blocks
        }
    };
    downmix_scalar(pcm, &mut y[done..], offset + done);
    true
}

/// The limits of `downmix_float`, lane by lane: below -65536 becomes
/// -65536, above 65536 becomes 65536, and a NaN becomes 0.
///
/// `_mm_max_ps(a, b)` is `a > b ? a : b` and `_mm_min_ps(a, b)` is
/// `a < b ? a : b`, so both keep a NaN in `b`, as the tests of C do.
#[inline(always)]
fn limit(v: __m128, lo: __m128, hi: __m128) -> __m128 {
    // SAFETY: as in `select`.
    unsafe {
        let v = _mm_max_ps(lo, v);
        let v = _mm_min_ps(hi, v);
        // A NaN is the only value that is unordered with itself.
        _mm_and_ps(_mm_cmpord_ps(v, v), v)
    }
}

/// `tansig_approx` of `mlp.rs` on four lanes.
#[inline(always)]
fn tansig(x: __m128) -> __m128 {
    // SAFETY: as in `select`.
    unsafe {
        let x2 = _mm_mul_ps(x, x);
        // c0 + (c1 + c2 * x2) * x2, each product rounded first.
        let poly = |c: [f32; 3]| _mm_add_ps(_mm_set1_ps(c[0]), _mm_mul_ps(mul_add(_mm_set1_ps(c[1]), x2, c[2]), x2));
        let n = poly(TANSIG_N);
        let d = poly(TANSIG_D);
        let r = _mm_div_ps(_mm_mul_ps(n, x), d);
        // The ternaries of C: 1 < r ? 1 : r, then -1 > r ? -1 : r, which
        // are `_mm_min_ps` and `_mm_max_ps` with r second. A NaN stays a
        // NaN.
        let r = _mm_min_ps(_mm_set1_ps(1.0), r);
        _mm_max_ps(_mm_set1_ps(-1.0), r)
    }
}

/// `sigmoid_approx` of `mlp.rs` on four lanes.
#[inline(always)]
fn sigmoid(x: __m128) -> __m128 {
    // SAFETY: as in `select`.
    unsafe {
        let half = _mm_set1_ps(0.5);
        _mm_add_ps(half, _mm_mul_ps(half, tansig(_mm_mul_ps(x, half))))
    }
}

/// The end of `compute_dense` of `mlp.rs`, four outputs at a time: the
/// scaling by `WEIGHTS_SCALE` and the activation. Returns the outputs left
/// over (fewer than four), which the scalar code does.
pub(super) fn scale_and_activate(output: &mut [f32], sigmoid_output: bool) -> &mut [f32] {
    let (blocks, rest) = output.as_chunks_mut::<4>();
    for b in blocks {
        // SAFETY: SSE is part of this target; `b` holds 4 values.
        unsafe {
            let v = _mm_mul_ps(_mm_loadu_ps(b.as_ptr()), _mm_set1_ps(WEIGHTS_SCALE));
            let v = if sigmoid_output { sigmoid(v) } else { tansig(v) };
            _mm_storeu_ps(b.as_mut_ptr(), v);
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
/// The update and the reset gate share one pass over the inputs and one
/// over the state. Then the two gates go to memory, and the output gate
/// takes its own passes over the inputs and the reset state. The NEON
/// kernel keeps all three gates in registers, but x86-64 has only 16
/// vector registers, too few for 18 sums. Each output still adds its bias,
/// then its input terms in the order of the inputs, then its state terms in
/// the order of the state, each product rounded first, as the scalar code.
pub(super) fn gru(layer: &GruLayer, state: &mut [f32; GRU_NEURONS], input: &[f32]) {
    const G: usize = 3 * GRU_NEURONS;
    let input: &[f32; GRU_INPUTS] = input[..GRU_INPUTS].try_into().expect("32 inputs");
    let bias: &[f32; G] = layer.bias[..G].try_into().expect("72 biases");
    let w_in: &[f32; GRU_INPUTS * G] = layer.input_weights[..GRU_INPUTS * G].try_into().expect("input weights");
    let w_rec: &[f32; GRU_NEURONS * G] =
        layer.recurrent_weights[..GRU_NEURONS * G].try_into().expect("recurrent weights");
    // The update gate and the reset state (the old state times the reset
    // gate).
    let mut update = [0.0f32; GRU_NEURONS];
    let mut reset_state = [0.0f32; GRU_NEURONS];
    // SAFETY: SSE is part of this target (see the `cfg` of the module).
    // Every load and store reads or writes four values inside the arrays
    // above, whose sizes the types fix: gate `g`, vector `k` of row `j`
    // starts at `j * 72 + g * 24 + 4 * k`, at most 71 - 3 within the row,
    // and vector `k` of a 24-value array at `4 * k`, at most 23 - 3.
    unsafe {
        let load = |p: *const f32, k: usize| _mm_loadu_ps(p.add(4 * k));
        // acc + w * x, the product rounded first.
        let term = |acc: __m128, w: __m128, x: __m128| _mm_add_ps(acc, _mm_mul_ps(w, x));
        let scale = _mm_set1_ps(WEIGHTS_SCALE);
        let mut z: [__m128; GATE_VECTORS] = core::array::from_fn(|k| load(bias.as_ptr(), k));
        let mut r: [__m128; GATE_VECTORS] = core::array::from_fn(|k| load(bias.as_ptr().add(GRU_NEURONS), k));
        for (j, &x) in input.iter().enumerate() {
            let row = w_in.as_ptr().add(j * G);
            let x = _mm_set1_ps(x);
            for k in 0..GATE_VECTORS {
                z[k] = term(z[k], load(row, k), x);
                r[k] = term(r[k], load(row.add(GRU_NEURONS), k), x);
            }
        }
        for (j, &x) in state.iter().enumerate() {
            let row = w_rec.as_ptr().add(j * G);
            let x = _mm_set1_ps(x);
            for k in 0..GATE_VECTORS {
                z[k] = term(z[k], load(row, k), x);
                r[k] = term(r[k], load(row.add(GRU_NEURONS), k), x);
            }
        }
        for k in 0..GATE_VECTORS {
            _mm_storeu_ps(update.as_mut_ptr().add(4 * k), sigmoid(_mm_mul_ps(z[k], scale)));
            let r = sigmoid(_mm_mul_ps(r[k], scale));
            _mm_storeu_ps(reset_state.as_mut_ptr().add(4 * k), _mm_mul_ps(load(state.as_ptr(), k), r));
        }
        let mut h: [__m128; GATE_VECTORS] =
            core::array::from_fn(|k| load(bias.as_ptr().add(2 * GRU_NEURONS), k));
        for (j, &x) in input.iter().enumerate() {
            let row = w_in.as_ptr().add(j * G + 2 * GRU_NEURONS);
            let x = _mm_set1_ps(x);
            for k in 0..GATE_VECTORS {
                h[k] = term(h[k], load(row, k), x);
            }
        }
        for (j, &x) in reset_state.iter().enumerate() {
            let row = w_rec.as_ptr().add(j * G + 2 * GRU_NEURONS);
            let x = _mm_set1_ps(x);
            for k in 0..GATE_VECTORS {
                h[k] = term(h[k], load(row, k), x);
            }
        }
        let one = _mm_set1_ps(1.0);
        for k in 0..GATE_VECTORS {
            // (1-z)*tanh(h) + z*state, each product rounded first.
            let z = load(update.as_ptr(), k);
            let t = _mm_mul_ps(_mm_sub_ps(one, z), tansig(_mm_mul_ps(h[k], scale)));
            _mm_storeu_ps(state.as_mut_ptr().add(4 * k), term(t, z, load(state.as_ptr(), k)));
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::analysis::tonality::{
        downmix_scalar, fft_state, tonality_bins_scalar, window_for_fft_scalar,
    };

    /// Small deterministic generator, so that a failure can be repeated.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// Uniform in [-0.5, 0.5).
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    /// Bit-for-bit equality; any two NaN count as equal.
    fn same(a: &[f32], b: &[f32]) -> bool {
        let canon = |v: &[f32]| v.iter().map(|x| if x.is_nan() { u32::MAX } else { x.to_bits() }).collect::<Vec<_>>();
        canon(a) == canon(b)
    }

    /// Random samples with the specials that the limits and the sum treat
    /// in their own way, and 16-bit samples with the limits of `i16`.
    fn samples(n: usize, seed: u64) -> (Vec<f32>, Vec<i16>) {
        let mut rng = Rng(seed);
        let specials = [
            0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 1.999_999_9, f32::INFINITY, f32::NEG_INFINITY, f32::NAN,
            f32::MAX, -f32::MAX, 1e-40, 3.0e34, -3.0e34,
        ];
        let f = (0..n)
            .map(|_| match rng.next() % 8 {
                0 => specials[(rng.next() % specials.len() as u64) as usize],
                _ => rng.unit() * 4.0,
            })
            .collect();
        let i = (0..n)
            .map(|_| match rng.next() % 8 {
                0 => [i16::MIN, i16::MAX, -1, 0][(rng.next() % 4) as usize],
                _ => rng.next() as i16,
            })
            .collect();
        (f, i)
    }

    /// The SSE downmix gives the bits of the scalar one for every layout it
    /// covers, every length (every tail of the blocks of 4 and 8) and
    /// offset, and declines the others.
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
        let lengths = (0..=33).chain([240, 479, 480, 960]);
        for (k, &(channels, c1, c2, covered)) in layouts.iter().enumerate() {
            let (f, i) = samples(1000 * channels, k as u64 + 1);
            for n in lengths.clone() {
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
                            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                            assert_eq!(bits(&a), bits(&b), "layout {k} n {n} offset {offset}");
                        } else {
                            assert!(a.iter().all(|&v| v == 7.0), "layout {k}: written");
                        }
                    }
                }
            }
        }
        // 24-bit input is left to the scalar code.
        let x = [0i32; 8];
        let p = AnalysisPcm { pcm: PcmRef::I32(&x), channels: 1, c1: 0, c2: SecondChannel::None, frame_size: 8 };
        assert!(!downmix(&p, &mut [0.0; 8], 0));
    }

    /// Runs the SSE and the scalar bins on the same input and state and
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

    /// The SSE bins give the bits of the scalar code: on random spectra
    /// with values of every size, with the special values where
    /// `fast_atan2f` changes branch or returns 0, with phase differences
    /// half-way between two integers, and on the spectra of real blocks.
    #[test]
    fn tonality_bins_match_the_scalar_code() {
        let mut rng = Rng(0x0bad_cafe_1234_5678);
        let specials = [0.0f32, -0.0, 1e-10, -1e-10, 7e-10, 1e-20, 1.0, -1.0, 3.0e18, f32::MIN_POSITIVE];
        for case in 0..400 {
            let mut value = || -> f32 {
                let r = rng.next();
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
                    let v = rng.unit() * 4.0;
                    if k == 2 { v * v * 0.01 } else { v }
                })
            });
            check_bins(&out, &state, &format!("random case {case}"));
        }
        // A spectrum of zeros: every angle is 0, so the second differences
        // are minus the old state. Old states of k + 0.5 make them half-way
        // between two integers, where the rounding goes to the even one.
        let zeros = [KissCpx::new(0.0, 0.0); N];
        let halves: [f32; N2] = core::array::from_fn(|i| (i % 7) as f32 - 3.5);
        check_bins(&zeros, &[[0.0; N2], halves, [0.0; N2]], "half-way differences");
        // Real blocks: the FFT of a windowed mix of tones and noise, one
        // block after the other, so the phase state is that of real input.
        let fft = fft_state();
        let mut state = [[0.0f32; N2]; 3];
        for block in 0..50 {
            let fin: [KissCpx; N] = core::array::from_fn(|n| {
                let t = (block * 240 + n) as f32;
                let tone = (t * 0.031).sin() * 3000.0 + (t * 0.2093).sin() * 800.0;
                let noise = rng.unit() * 50.0;
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

    /// The SSE window gives the bits and the places of the scalar one, on
    /// random bit patterns (NaN and infinities among them) and on samples
    /// of the 16-bit range.
    #[test]
    fn window_for_fft_matches_the_scalar_code() {
        let mut rng = Rng(0x5eed_0000_1111_2222);
        for case in 0..20 {
            let inmem: [f32; ANALYSIS_BUF_SIZE] = core::array::from_fn(|_| {
                let r = rng.next();
                let unit = (r >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                if case == 0 { f32::from_bits(r as u32) } else { unit * 65536.0 }
            });
            let mut a = [KissCpx::new(7.0, 7.0); N];
            let mut b = a;
            window_for_fft(&inmem, fft_state(), &mut a);
            window_for_fft_scalar(&inmem, fft_state(), &mut b);
            let flat = |v: &[KissCpx; N]| v.iter().flat_map(|c| [c.r, c.i]).collect::<Vec<_>>();
            assert!(same(&flat(&a), &flat(&b)), "case {case}");
        }
    }

    /// The SSE network gives the bits of the scalar one: the activations of
    /// the dense layers for every length (every tail of the blocks of 4),
    /// and the recurrent layer, on random inputs of every size, near the
    /// limits of the activation and with NaN and infinities.
    #[test]
    fn network_matches_the_scalar_code() {
        use crate::analysis::mlp::{LAYER0, LAYER1, LAYER2, compute_gru_scalar, sigmoid_approx, tansig_approx};
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for case in 0..3000 {
            let scale = [0.01f32, 1.0, 10.0, 300.0, 1e4][case % 5];
            let mut x: [f32; 32] = core::array::from_fn(|_| rng.unit() * 2.0 * scale);
            match case % 97 {
                0 => x[case % 32] = f32::NAN,
                1 => x[case % 32] = f32::INFINITY,
                2 => x[case % 32] = -f32::MAX,
                _ => {}
            }
            // Activations after the scaling by 1/128, on 4 lanes.
            let len = case % 33;
            for sig in [false, true] {
                let mut a = x;
                let rest = scale_and_activate(&mut a[..len], sig).len();
                assert_eq!(rest, len % 4);
                for (k, (&got, &inp)) in a[..len - rest].iter().zip(&x).enumerate() {
                    let v = inp * WEIGHTS_SCALE;
                    let want = if sig { sigmoid_approx(v) } else { tansig_approx(v) };
                    assert!(same(&[got], &[want]), "case {case} lane {k} sigmoid {sig}");
                }
                assert!(same(&a[len - rest..], &x[len - rest..]), "case {case}: the rest is left alone");
            }
            // The recurrent layer, from a random state.
            let state0: [f32; GRU_NEURONS] = core::array::from_fn(|_| rng.unit() * 2.0);
            let (mut a, mut b) = (state0, state0);
            gru(&LAYER1, &mut a, &x);
            compute_gru_scalar(&LAYER1, &mut b, &x);
            assert!(same(&a, &b), "case {case}");
        }
        // A run of the whole network, where the state carries over.
        let (mut a, mut b) = ([0.0f32; 32], [0.0f32; 32]);
        for step in 0..2000 {
            let feat: [f32; 25] = core::array::from_fn(|_| rng.unit() * 6.0);
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
