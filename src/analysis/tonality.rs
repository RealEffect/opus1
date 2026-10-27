//! The analysis itself (libopus `src/analysis.c`): the state, one 20 ms
//! block (`tonality_analysis`), the reading of the results
//! (`tonality_get_info`) and the entry point (`run_analysis`).
//!
//! Each step keeps the order of the C statements, also where an early
//! return leaves a part of the state written: the next frames read that
//! state, so the order shows in the output. The comments "C 123" give the
//! line of analysis.c that a step ports.
//!
//! Rounding: `mac(acc, a, b)` stands where the libopus build fuses
//! `acc + a * b` into one rounding. Plain `*` and `+` stand where it rounds
//! each step, among them the two sums that clang turns into vector products
//! (C 864 and 871). `log`, `log10` and a few comparisons are in double
//! precision as in C.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::mlp::{self, LAYER0, LAYER1, LAYER2, MAX_NEURONS};
use super::tables::{ANALYSIS_WINDOW, DCT_TABLE, STD_FEATURE_BIAS, TBANDS};
use super::{AnalysisInfo, AnalysisPcm, LEAK_BANDS, PcmRef, SecondChannel, is_digital_silence};
use crate::kiss_fft::{KissCpx, KissFftState, opus_fft_impl};
use crate::in_place::{write_fields, write_zero};
use crate::mathops::mac;

/// libopus `NB_FRAMES`: frames of band energies kept for the stationarity
/// and the spectral variability.
pub const NB_FRAMES: usize = 8;
/// libopus `NB_TBANDS`: bands of the analysis.
pub const NB_TBANDS: usize = 18;
/// libopus `ANALYSIS_BUF_SIZE`: 30 ms at 24 kHz.
pub const ANALYSIS_BUF_SIZE: usize = 720;
/// libopus `ANALYSIS_COUNT_MAX`: the frame counter stops here.
pub const ANALYSIS_COUNT_MAX: i32 = 10000;
/// libopus `DETECT_SIZE`: results kept in the ring `info`.
pub const DETECT_SIZE: usize = 100;

/// libopus `TRANSITION_PENALTY`.
const TRANSITION_PENALTY: f32 = 10.0;
/// libopus `NB_TONAL_SKIP_BANDS`.
const NB_TONAL_SKIP_BANDS: usize = 9;
/// libopus `LEAKAGE_OFFSET`.
const LEAKAGE_OFFSET: f32 = 2.5;
/// libopus `LEAKAGE_SLOPE`.
const LEAKAGE_SLOPE: f32 = 2.0;
/// libopus `SCALE_ENER` of the float build: the square of 1/32768.
const SCALE_ENER: f32 = 1.0 / 32768.0 / 32768.0;
/// `.5f*1.442695f` of C 644 and 675, which clang folds into one constant.
/// 1.442695 is the base-2 logarithm of e with the digits of libopus.
#[allow(clippy::approx_constant)]
const HALF_LOG2_E: f32 = 0.5 * 1.442695;
/// `(float)(.5f/M_PI)` of C 581 and 585: computed in double, then rounded.
pub(super) const INV_TWO_PI: f32 = (0.5f64 / core::f64::consts::PI) as f32;
/// `pi4` of C 465: `(float)(M_PI*M_PI*M_PI*M_PI)`.
const PI4: f32 = (core::f64::consts::PI * core::f64::consts::PI * core::f64::consts::PI
    * core::f64::consts::PI) as f32;
/// `40.f*16.f*pi4` of C 601 and 603, evaluated from the left in `f32`.
pub(super) const TONALITY_SCALE: f32 = 40.0 * 16.0 * PI4;
/// The size of the FFT of the analysis.
pub(super) const N: usize = 480;
/// Half of it.
pub(super) const N2: usize = 240;
/// The most samples `downmix_and_resample` downmixes in one call: 20 ms at
/// 48 kHz.
const MAX_DOWNMIX: usize = 960;

/// `MIN16`/`MIN32` of celt/arch.h: `a < b ? a : b`.
#[inline(always)]
fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `MAX16`/`MAX32` of celt/arch.h: `a > b ? a : b`.
#[inline(always)]
fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// libopus `float2int` (celt/float_cast.h): round to the nearest integer,
/// a value halfway between two goes to the even one.
///
/// 64-bit ARM uses `vcvtns_s32_f32`, as libopus does, which also limits
/// the result to the `i32` range and gives 0 for a NaN. Elsewhere the
/// rounding is done in `f32` (as `lrintf` does in the default rounding
/// mode) and `as i32` limits the range. The analysis only rounds small
/// phase differences, far inside the range.
#[inline(always)]
fn float2int(x: f32) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of every 64-bit ARM processor.
        unsafe { core::arch::aarch64::vcvtns_s32_f32(x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // A float at or beyond 2^23 has no fraction bits. Below, adding
        // 2^23 leaves no fraction bits either, so the sum rounds to an
        // integer, to nearest and ties to even.
        const TWO_23: f32 = 8_388_608.0;
        // NaN must take this branch, so the comparison is negated.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let r = if !(x.abs() < TWO_23) {
            x
        } else if x >= 0.0 {
            (x + TWO_23) - TWO_23
        } else {
            (x - TWO_23) + TWO_23
        };
        r as i32
    }
}

/// The constants of libopus `fast_atan2f` (celt/mathops.h:55-59), with the
/// digits of libopus.
#[allow(clippy::excessive_precision)]
pub(super) const ATAN_A: f32 = 0.43157974;
#[allow(clippy::excessive_precision)]
pub(super) const ATAN_B: f32 = 0.67848403;
#[allow(clippy::excessive_precision)]
pub(super) const ATAN_C: f32 = 0.08595542;
/// `cE` of `fast_atan2f`: `(float)PI/2`, with PI 3.1415926535897931 of
/// celt/mathops.h.
pub(super) const ATAN_E: f32 = core::f32::consts::PI / 2.0;

/// libopus `fast_atan2f` (celt/mathops.h:60-77): an approximation of
/// `atan2(y, x)` within about 1e-4 radians. Returns 0 when `x` and `y` are
/// both below about 1e-9.
///
/// The three sums inside are fused multiply-adds, as in the libopus build.
/// `x*x + y*y` of the first test is not, because C computes the squares in
/// separate statements.
#[inline]
pub fn fast_atan2f(y: f32, x: f32) -> f32 {
    let x2 = x * x;
    let y2 = y * y;
    if x2 + y2 < 1e-18 {
        return 0.0;
    }
    let c1 = if y < 0.0 { -ATAN_E } else { ATAN_E };
    if x2 < y2 {
        let den = mac(y2, ATAN_B, x2) * mac(y2, ATAN_C, x2);
        // -x*y*(...)/den + c1: the negation is exact, so this is
        // c1 - x*y*(...)/den, as the libopus build computes it.
        c1 - (x * y) * mac(y2, ATAN_A, x2) / den
    } else {
        let den = mac(x2, ATAN_B, y2) * mac(x2, ATAN_C, y2);
        let xy = x * y;
        let c2 = if xy < 0.0 { -ATAN_E } else { ATAN_E };
        xy * mac(x2, ATAN_A, y2) / den + c1 - c2
    }
}

/// The state of the analysis (libopus `TonalityAnalysisState`,
/// src/analysis.h:47-81).
///
/// The fields have the C names in snake case and the C sizes. libopus's
/// `arch` and `application` are left out: this crate chooses its vector
/// code at compile time, and `analysis.c` never reads `application`.
///
/// The struct holds 25,968 bytes: 14 KB of state as in C, caches of values
/// computed from it, and scratch buffers that C keeps on the stack (the FFT
/// buffer, the per-bin tonality and the downmixed input), so that the
/// analysis needs little stack.
#[derive(Clone)]
pub struct TonalityAnalysisState {
    /// The input sampling rate: 16000, 24000 or 48000. It survives a reset.
    pub fs: i32,
    /// Phase of each FFT bin in the last block, in turns.
    pub angle: [f32; 240],
    /// Change of the phase from the block before.
    pub d_angle: [f32; 240],
    /// The fourth power of the remainder of the second phase difference.
    pub d2_angle: [f32; 240],
    /// The input at 24 kHz: 240 samples kept from the last block, then the
    /// new ones.
    pub inmem: [f32; ANALYSIS_BUF_SIZE],
    /// Usable samples in `inmem`.
    pub mem_fill: usize,
    pub prev_band_tonality: [f32; NB_TBANDS],
    pub prev_tonality: f32,
    pub prev_bandwidth: i32,
    /// C `E`: band energies of the last `NB_FRAMES` blocks.
    pub e: [[f32; NB_TBANDS]; NB_FRAMES],
    /// C `logE`: their natural logarithms.
    pub log_e: [[f32; NB_TBANDS]; NB_FRAMES],
    /// C `lowE` and `highE`: trackers of the lowest and highest log energy
    /// of each band.
    pub low_e: [f32; NB_TBANDS],
    pub high_e: [f32; NB_TBANDS],
    /// C `meanE`: a decaying maximum of each band energy, for the
    /// bandwidth decision. The last entry is the band above 12 kHz.
    pub mean_e: [f32; NB_TBANDS + 1],
    /// The cepstral values of the last four blocks, 8 per block.
    pub mem: [f32; 32],
    pub cmean: [f32; 8],
    pub std: [f32; 9],
    /// C `Etracker`.
    pub e_tracker: f32,
    /// C `lowECount`.
    pub low_e_count: f32,
    /// C `E_count`: the row of `e` and `log_e` that the next block writes.
    pub e_count: usize,
    /// Blocks analysed, up to `ANALYSIS_COUNT_MAX`.
    pub count: i32,
    /// Samples of the next call's buffer that the analysis has already
    /// seen (expert frame duration). See `run`.
    pub analysis_offset: i32,
    /// The slot of `info` that the next block writes.
    pub write_pos: usize,
    /// The slot of `info` of the frame that is coded next.
    pub read_pos: usize,
    /// The 2.5 ms step inside the slot `read_pos`.
    pub read_subframe: i32,
    /// The energy above 12 kHz of the samples in `inmem` (48 kHz input).
    pub hp_ener_accum: f32,
    /// The analysis has started (C `initialized`, an `int`).
    pub initialized: bool,
    /// The state of the recurrent layer of the network (24 values used).
    pub rnn_state: [f32; MAX_NEURONS],
    /// The state of the 2:1 decimator.
    pub downmix_state: [f32; 3],
    /// The results of the last `DETECT_SIZE` blocks.
    pub info: [AnalysisInfo; DETECT_SIZE],
    cache: Cache,
    scratch: Scratch,
}

/// Values that follow from `e` and `log_e` and that the analysis would
/// otherwise compute again in every block. Not part of the C state.
#[derive(Clone)]
struct Cache {
    /// The square root of each value of `e`, written when that value is
    /// written.
    e_sqrt: [[f32; NB_TBANDS]; NB_FRAMES],
    /// `dist[i][j]`: the squared distance between rows `i` and `j` of
    /// `log_e` of the spectral variability (C 761-769).
    ///
    /// Swapping the rows only negates each difference, so `dist[i][j]` and
    /// `dist[j][i]` are the same bits. A block that reaches the spectral
    /// variability has written one whole row of `log_e`, the row
    /// `e_count`. A block that stops earlier writes only part of that row,
    /// and does not advance `e_count`, so the next block that reaches the
    /// spectral variability rewrites the same row. So the distances of the
    /// pairs with that row are the only ones to compute.
    dist: [[f32; NB_FRAMES]; NB_FRAMES],
}

impl Cache {
    fn new() -> Self {
        Self { e_sqrt: [[0.0; NB_TBANDS]; NB_FRAMES], dist: [[0.0; NB_FRAMES]; NB_FRAMES] }
    }

    /// Writes the cleared cache of [`new`](Self::new) to `this` (see
    /// `crate::in_place`).
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    unsafe fn write_new(this: *mut Self) {
        // SAFETY: the caller guarantees that `this` is valid for writes.
        unsafe {
            write_fields!(this, {
                e_sqrt = |p| write_zero(p),
                dist = |p| write_zero(p),
            });
        }
    }
}

/// Buffers that hold data only during one block. Not part of the C state.
#[derive(Clone)]
struct Scratch {
    /// The FFT input and output (in place).
    fft: [KissCpx; N],
    bins: BinScratch,
    /// The downmixed input before resampling.
    downmix: [f32; MAX_DOWNMIX],
}

impl Scratch {
    /// Writes cleared buffers to `this` (see `crate::in_place`).
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    unsafe fn write_new(this: *mut Self) {
        // SAFETY: the caller guarantees that `this` is valid for writes.
        unsafe {
            write_fields!(this, {
                fft = |p| write_zero(p),
                bins = |p| BinScratch::write_new(p),
                downmix = |p| write_zero(p),
            });
        }
    }
}

/// The 480-point FFT of the 48 kHz CELT mode (C `celt_mode->mdct.kfft[0]`).
pub(super) fn fft_state() -> &'static KissFftState {
    crate::modes::default_mode().mdct.kfft(0).expect("the 48 kHz mode has a 480-point FFT")
}

impl TonalityAnalysisState {
    /// libopus `tonality_analysis_init`. `fs` is the input rate, which
    /// must be 16000, 24000 or 48000 when the analysis runs.
    pub fn new(fs: i32) -> Self {
        let mut st = core::mem::MaybeUninit::<Self>::uninit();
        // SAFETY: `write_new` initializes every field of `st`.
        unsafe {
            Self::write_new(st.as_mut_ptr(), fs);
            st.assume_init()
        }
    }

    /// Writes the state of [`new`](Self::new) to `this`, which may be
    /// uninitialized memory, without building it on the stack first (see
    /// `crate::in_place`). The state is 26 KB.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    pub(crate) unsafe fn write_new(this: *mut Self, fs: i32) {
        // SAFETY: the caller guarantees that `this` is valid for writes.
        // Each closure below writes the whole field it is given.
        unsafe {
            write_fields!(this, {
                fs: fs,
                angle = |p| write_zero(p),
                d_angle = |p| write_zero(p),
                d2_angle = |p| write_zero(p),
                inmem = |p| write_zero(p),
                mem_fill: 0,
                prev_band_tonality: [0.0; NB_TBANDS],
                prev_tonality: 0.0,
                prev_bandwidth: 0,
                e = |p| write_zero(p),
                log_e = |p| write_zero(p),
                low_e: [0.0; NB_TBANDS],
                high_e: [0.0; NB_TBANDS],
                mean_e: [0.0; NB_TBANDS + 1],
                mem: [0.0; 32],
                cmean: [0.0; 8],
                std: [0.0; 9],
                e_tracker: 0.0,
                low_e_count: 0.0,
                e_count: 0,
                count: 0,
                analysis_offset: 0,
                write_pos: 0,
                read_pos: 0,
                read_subframe: 0,
                hp_ener_accum: 0.0,
                initialized: false,
                rnn_state = |p| write_zero(p),
                downmix_state: [0.0; 3],
                info = |p: *mut [AnalysisInfo; DETECT_SIZE]| {
                    let first = p.cast::<AnalysisInfo>();
                    for i in 0..DETECT_SIZE {
                        first.add(i).write(AnalysisInfo::default());
                    }
                },
                cache = |p| Cache::write_new(p),
                scratch = |p| Scratch::write_new(p),
            });
        }
    }

    /// libopus `tonality_analysis_reset`: clears everything but `fs`, as
    /// after `new`. libopus calls it at a discontinuity of the input.
    pub fn reset(&mut self) {
        self.angle = [0.0; 240];
        self.d_angle = [0.0; 240];
        self.d2_angle = [0.0; 240];
        self.inmem = [0.0; ANALYSIS_BUF_SIZE];
        self.mem_fill = 0;
        self.prev_band_tonality = [0.0; NB_TBANDS];
        self.prev_tonality = 0.0;
        self.prev_bandwidth = 0;
        self.e = [[0.0; NB_TBANDS]; NB_FRAMES];
        self.log_e = [[0.0; NB_TBANDS]; NB_FRAMES];
        self.low_e = [0.0; NB_TBANDS];
        self.high_e = [0.0; NB_TBANDS];
        self.mean_e = [0.0; NB_TBANDS + 1];
        self.mem = [0.0; 32];
        self.cmean = [0.0; 8];
        self.std = [0.0; 9];
        self.e_tracker = 0.0;
        self.low_e_count = 0.0;
        self.e_count = 0;
        self.count = 0;
        self.analysis_offset = 0;
        self.write_pos = 0;
        self.read_pos = 0;
        self.read_subframe = 0;
        self.hp_ener_accum = 0.0;
        self.initialized = false;
        self.rnn_state = [0.0; MAX_NEURONS];
        self.downmix_state = [0.0; 3];
        self.info = [AnalysisInfo::default(); DETECT_SIZE];
        self.cache = Cache::new();
    }

    /// Whether the analysis ran since the last reset (C `initialized`).
    /// libopus resets the state when the analysis stops running, for
    /// example when the complexity drops below 7.
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// C `read_pos` and `read_subframe`. libopus saves them before
    /// `run_analysis` and restores them before it reads the results of
    /// each frame of a packet of several frames.
    pub fn read_cursor(&self) -> (usize, i32) {
        (self.read_pos, self.read_subframe)
    }

    /// Restores what `read_cursor` returned.
    pub fn set_read_cursor(&mut self, cursor: (usize, i32)) {
        self.read_pos = cursor.0;
        self.read_subframe = cursor.1;
    }

    /// libopus `run_analysis` (analysis.c:954-980): analyses the new
    /// samples of the caller's buffer, then returns the result for the next
    /// `frame_size` samples (`tonality_get_info`).
    ///
    /// The analysis works in blocks of 20 ms. It analyses
    /// `pcm.frame_size` samples (rounded down to an even number, at most
    /// 1.9 s), minus the `analysis_offset` samples that the last call had
    /// already analysed. It then keeps `pcm.frame_size - frame_size` as the
    /// offset of the next call. When the caller's buffer is longer than the
    /// coded frame (expert frame duration), the analysis therefore assumes
    /// that the next call's buffer starts with the samples after the coded
    /// frame, as libopus does.
    ///
    /// `lsb_depth` (8 to 24) is the precision of the input: it sets the
    /// noise floor of the bandwidth decision and the level below which a
    /// block counts as digital silence.
    pub fn run(&mut self, pcm: AnalysisPcm<'_>, frame_size: usize, lsb_depth: i32) -> AnalysisInfo {
        debug_assert!((8..=24).contains(&lsb_depth), "lsb_depth {lsb_depth}");
        let fs = self.fs;
        let block = fs / 50;
        let mut analysis_frame_size = pcm.frame_size as i32;
        analysis_frame_size -= analysis_frame_size & 1;
        // Avoid overflow/wrap-around of the analysis buffer (C 966).
        analysis_frame_size = analysis_frame_size.min((DETECT_SIZE as i32 - 5) * fs / 50);
        let mut pcm_len = analysis_frame_size - self.analysis_offset;
        let mut offset = self.analysis_offset;
        while pcm_len > 0 {
            self.tonality_analysis(&pcm, block.min(pcm_len), offset, lsb_depth);
            offset += block;
            pcm_len -= block;
        }
        // A coded frame longer than the caller's buffer is a caller error
        // that C does not check (the next call would read before the
        // buffer). Here the offset then stays at 0.
        self.analysis_offset = (analysis_frame_size - frame_size as i32).max(0);
        self.get_info(frame_size)
    }

    /// libopus `tonality_get_info` (analysis.c:232-408): the result for
    /// the next `len` samples, which advances the read position by `len`.
    ///
    /// It takes the slot of the frame, the largest tonality of up to 3
    /// slots ahead, the widest bandwidth of up to 6 neighbouring slots, and
    /// turns the music probabilities of the slots ahead into the switching
    /// thresholds `music_prob_min` and `music_prob_max`.
    pub fn get_info(&mut self, len: usize) -> AnalysisInfo {
        const D: i32 = DETECT_SIZE as i32;
        let write_pos = self.write_pos as i32;
        let len = len as i32;
        let mut pos = self.read_pos as i32;
        let mut curr_lookahead = write_pos - pos;
        if curr_lookahead < 0 {
            curr_lookahead += D;
        }

        self.read_subframe += len / (self.fs / 400);
        let mut read_pos = self.read_pos as i32;
        while self.read_subframe >= 8 {
            self.read_subframe -= 8;
            read_pos += 1;
        }
        if read_pos >= D {
            read_pos -= D;
        }
        self.read_pos = read_pos as usize;

        // On long frames, look at the second analysis window rather than
        // the first.
        if len > self.fs / 50 && pos != write_pos {
            pos += 1;
            if pos == D {
                pos = 0;
            }
        }
        if pos == write_pos {
            pos -= 1;
        }
        if pos < 0 {
            pos = D - 1;
        }
        let pos0 = pos;
        let mut out = self.info[pos as usize];
        if !out.valid {
            return out;
        }
        let mut tonality_max = out.tonality;
        let mut tonality_avg = out.tonality;
        let mut tonality_count = 1;
        // Look at the neighbouring frames and pick the largest bandwidth
        // found (to be safe).
        let mut bandwidth_span = 6;
        // If possible, look ahead for a tone to compensate for the delay in
        // the tone detector.
        for _ in 0..3 {
            pos += 1;
            if pos == D {
                pos = 0;
            }
            if pos == write_pos {
                break;
            }
            let slot = &self.info[pos as usize];
            tonality_max = max(tonality_max, slot.tonality);
            tonality_avg += slot.tonality;
            tonality_count += 1;
            out.bandwidth = out.bandwidth.max(slot.bandwidth);
            bandwidth_span -= 1;
        }
        pos = pos0;
        // Look back in time to see if any has a wider bandwidth than the
        // current frame.
        for _ in 0..bandwidth_span {
            pos -= 1;
            if pos < 0 {
                pos = D - 1;
            }
            if pos == write_pos {
                break;
            }
            out.bandwidth = out.bandwidth.max(self.info[pos as usize].bandwidth);
        }
        out.tonality = max(tonality_avg / tonality_count as f32, tonality_max - 0.2);

        let mut mpos = pos0;
        let mut vpos = pos0;
        // With enough look-ahead, compensate for the delay of about 5
        // frames of the music probability and of 1 frame of the activity
        // probability.
        if curr_lookahead > 15 {
            mpos += 5;
            if mpos >= D {
                mpos -= D;
            }
            vpos += 1;
            if vpos >= D {
                vpos -= D;
            }
        }

        // The thresholds of a switch between speech and music: see the long
        // comment at analysis.c:327-359.
        let mut prob_min = 1.0f32;
        let mut prob_max = 0.0f32;
        let vad_prob = self.info[vpos as usize].activity_probability;
        let mut prob_count = max(0.1, vad_prob);
        let mut prob_avg = max(0.1, vad_prob) * self.info[mpos as usize].music_prob;
        loop {
            mpos += 1;
            if mpos == D {
                mpos = 0;
            }
            if mpos == write_pos {
                break;
            }
            vpos += 1;
            if vpos == D {
                vpos = 0;
            }
            if vpos == write_pos {
                break;
            }
            let pos_vad = self.info[vpos as usize].activity_probability;
            prob_min =
                min(mac(prob_avg, -TRANSITION_PENALTY, vad_prob - pos_vad) / prob_count, prob_min);
            prob_max =
                max(mac(prob_avg, TRANSITION_PENALTY, vad_prob - pos_vad) / prob_count, prob_max);
            prob_count += max(0.1, pos_vad);
            prob_avg = mac(prob_avg, max(0.1, pos_vad), self.info[mpos as usize].music_prob);
        }
        out.music_prob = prob_avg / prob_count;
        prob_min = min(prob_avg / prob_count, prob_min);
        prob_max = max(prob_avg / prob_count, prob_max);
        prob_min = max(prob_min, 0.0);
        prob_max = min(prob_max, 1.0);

        // Without enough look-ahead, do the best to make a decent decision.
        if curr_lookahead < 10 {
            let mut pmin = prob_min;
            let mut pmax = prob_max;
            pos = pos0;
            // Look for the minimum and maximum in the past.
            for _ in 0..(self.count - 1).min(15) {
                pos -= 1;
                if pos < 0 {
                    pos = D - 1;
                }
                pmin = min(pmin, self.info[pos as usize].music_prob);
                pmax = max(pmax, self.info[pos as usize].music_prob);
            }
            // Bias against switching on active audio.
            pmin = max(0.0, mac(pmin, -0.1, vad_prob));
            pmax = min(1.0, mac(pmax, 0.1, vad_prob));
            let weight = mac(1.0, -0.1, curr_lookahead as f32);
            prob_min = mac(prob_min, weight, pmin - prob_min);
            prob_max = mac(prob_max, weight, pmax - prob_max);
        }
        out.music_prob_min = prob_min;
        out.music_prob_max = prob_max;
        out
    }

    /// libopus `tonality_analysis` (analysis.c:445-952) on `len` samples
    /// of the caller's buffer from sample `offset` on (at the input rate).
    /// It runs the analysis of a block when 30 ms at 24 kHz have come
    /// together.
    fn tonality_analysis(&mut self, pcm: &AnalysisPcm<'_>, len: i32, offset: i32, lsb_depth: i32) {
        let fs = self.fs;
        if !self.initialized {
            self.mem_fill = 240;
            self.initialized = true;
        }
        let count = self.count;
        let alpha = 1.0 / (10.min(1 + count)) as f32;
        let alpha_e = 1.0 / (25.min(1 + count)) as f32;
        // Noise floor related decay for bandwidth detection: -2.2 dB/second.
        let mut alpha_e2 = 1.0 / (100.min(1 + count)) as f32;
        if count <= 1 {
            alpha_e2 = 1.0;
        }

        // len and offset are now at 24 kHz.
        let (len, offset) = match fs {
            48000 => (len / 2, offset / 2),
            16000 => (3 * len / 2, 3 * offset / 2),
            _ => (len, offset),
        };

        let mem_fill = self.mem_fill as i32;
        let buf = ANALYSIS_BUF_SIZE as i32;
        self.hp_ener_accum += downmix_and_resample(
            pcm,
            &mut self.inmem[mem_fill as usize..],
            &mut self.downmix_state,
            &mut self.scratch.downmix,
            len.min(buf - mem_fill),
            offset,
            fs,
        );
        if mem_fill + len < buf {
            self.mem_fill = (mem_fill + len) as usize;
            // Not enough samples for a block yet.
            return;
        }
        let hp_ener = self.hp_ener_accum;
        let slot = self.write_pos;
        self.write_pos += 1;
        if self.write_pos >= DETECT_SIZE {
            self.write_pos -= DETECT_SIZE;
        }

        // C 529, is_digital_silence32, which is is_digital_silence in the
        // float build.
        let is_silence = is_digital_silence(&self.inmem, lsb_depth);

        let sc = &mut self.scratch;
        let fft = fft_state();
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        #[cfg(target_arch = "aarch64")]
        unsafe { super::neon::window_for_fft(&self.inmem, fft, &mut sc.fft) };
        // SAFETY: SSE2 is part of this target (see the `cfg`).
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        unsafe { super::x86::window_for_fft(&self.inmem, fft, &mut sc.fft) };
        #[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))))]
        window_for_fft_scalar(&self.inmem, fft, &mut sc.fft);
        self.inmem.copy_within(ANALYSIS_BUF_SIZE - 240.., 0);
        let remaining = len - (buf - mem_fill);
        self.hp_ener_accum = downmix_and_resample(
            pcm,
            &mut self.inmem[240..],
            &mut self.downmix_state,
            &mut sc.downmix,
            remaining,
            offset + buf - mem_fill,
            fs,
        );
        self.mem_fill = (240 + remaining) as usize;
        if is_silence {
            // On silence, copy the previous analysis (C 549-558).
            let prev_pos = (slot + DETECT_SIZE - 1) % DETECT_SIZE;
            self.info[slot] = self.info[prev_pos];
            return;
        }
        opus_fft_impl(fft, &mut sc.fft);
        // A NaN in the input makes the whole output NaN, so one value is
        // enough to check (C 562).
        if sc.fft[0].r.is_nan() {
            self.info[slot].valid = false;
            return;
        }
        let out = &sc.fft;
        // Phase-based tonality of each bin (C 570-608), and the energy of
        // each bin.
        let mut phase = Phase { angle: &mut self.angle, d_angle: &mut self.d_angle, d2_angle: &mut self.d2_angle };
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        #[cfg(target_arch = "aarch64")]
        let first_scalar_bin = unsafe { super::neon::tonality_bins(out, &mut phase, &mut sc.bins) };
        // SAFETY: SSE2 is part of this target (see the `cfg`).
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        let first_scalar_bin = unsafe { super::x86::tonality_bins(out, &mut phase, &mut sc.bins) };
        #[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))))]
        let first_scalar_bin = 1;
        tonality_bins_scalar(out, first_scalar_bin, &mut phase, &mut sc.bins);
        let BinScratch { tonality, tonality2, noisiness, energy: bin_energy } = &mut sc.bins;
        for i in 2..N2 - 1 {
            let tt = min(tonality2[i], max(tonality2[i - 1], tonality2[i + 1]));
            tonality[i] = 0.9 * max(tonality[i], tt - 0.1);
        }

        let mut frame_tonality = 0.0f32;
        let mut max_frame_tonality = 0.0f32;
        self.info[slot].activity = 0.0;
        let mut frame_noisiness = 0.0f32;
        let mut frame_stationarity = 0.0f32;
        if self.count == 0 {
            self.low_e = [1e10; NB_TBANDS];
            self.high_e = [-1e10; NB_TBANDS];
        }
        let mut relative_e = 0.0f32;
        let mut frame_loudness = 0.0f32;
        let mut band_log2 = [0.0f32; NB_TBANDS + 1];
        let mut log_e = [0.0f32; NB_TBANDS];
        let mut band_tonality = [0.0f32; NB_TBANDS];
        // The energy of the very first band is special because of DC.
        {
            let x1r = 2.0 * out[0].r;
            let x2r = 2.0 * out[0].i;
            let mut e = mac(x2r * x2r, x1r, x1r);
            for i in 1..4 {
                e += bin_energy[i];
            }
            e *= SCALE_ENER;
            band_log2[0] = HALF_LOG2_E * ((e + 1e-10) as f64).ln() as f32;
        }
        let mut slope = 0.0f32;
        let e_count = self.e_count;
        for b in 0..NB_TBANDS {
            let mut e = 0.0f32;
            let mut t_e = 0.0f32;
            let mut n_e = 0.0f32;
            for i in TBANDS[b]..TBANDS[b + 1] {
                let bin_e = SCALE_ENER * bin_energy[i];
                e += bin_e;
                t_e = mac(t_e, bin_e, max(0.0, tonality[i]));
                n_e = mac(n_e, bin_e + bin_e, 0.5 - noisiness[i]);
            }
            // Check for extreme band energies that could cause NaNs later.
            // (C: `!(E<1e9f) || celt_isnan(E)`.)
            if e >= 1e9 || e.is_nan() {
                self.info[slot].valid = false;
                return;
            }

            self.e[e_count][b] = e;
            self.cache.e_sqrt[e_count][b] = e.sqrt();
            frame_noisiness += n_e / (1e-15 + e);

            frame_loudness += (e + 1e-10).sqrt();
            let ln = ((e + 1e-10) as f64).ln() as f32;
            log_e[b] = ln;
            band_log2[b + 1] = HALF_LOG2_E * ln;
            self.log_e[e_count][b] = log_e[b];
            if self.count == 0 {
                self.low_e[b] = log_e[b];
                self.high_e[b] = log_e[b];
            }
            if self.high_e[b] as f64 > self.low_e[b] as f64 + 7.5 {
                if self.high_e[b] - log_e[b] > log_e[b] - self.low_e[b] {
                    self.high_e[b] -= 0.01;
                } else {
                    self.low_e[b] += 0.01;
                }
            }
            if log_e[b] > self.high_e[b] {
                self.high_e[b] = log_e[b];
                self.low_e[b] = max(self.high_e[b] - 15.0, self.low_e[b]);
            } else if log_e[b] < self.low_e[b] {
                self.low_e[b] = log_e[b];
                self.high_e[b] = min(self.low_e[b] + 15.0, self.high_e[b]);
            }
            relative_e += (log_e[b] - self.low_e[b]) / (1e-5 + (self.high_e[b] - self.low_e[b]));

            let mut l1 = 0.0f32;
            let mut l2 = 0.0f32;
            for i in 0..NB_FRAMES {
                l1 += self.cache.e_sqrt[i][b];
                l2 += self.e[i][b];
            }

            let mut stationarity =
                min(0.99, l1 / (1e-15 + (NB_FRAMES as f32 * l2) as f64).sqrt() as f32);
            stationarity *= stationarity;
            stationarity *= stationarity;
            frame_stationarity += stationarity;
            band_tonality[b] = max(t_e / (1e-15 + e), stationarity * self.prev_band_tonality[b]);
            frame_tonality += band_tonality[b];
            if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
                frame_tonality -= band_tonality[b + NB_TONAL_SKIP_BANDS - NB_TBANDS];
            }
            max_frame_tonality = max(
                max_frame_tonality,
                mac(1.0, 0.03, (b as i32 - NB_TBANDS as i32) as f32) * frame_tonality,
            );
            slope = mac(slope, band_tonality[b], (b as i32 - 8) as f32);
            self.prev_band_tonality[b] = band_tonality[b];
        }

        // Leakage between bands (C 726-754).
        let mut leakage_from = [0.0f32; NB_TBANDS + 1];
        let mut leakage_to = [0.0f32; NB_TBANDS + 1];
        leakage_from[0] = band_log2[0];
        leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET;
        for b in 1..NB_TBANDS + 1 {
            let leak_slope = LEAKAGE_SLOPE * (TBANDS[b] - TBANDS[b - 1]) as f32 / 4.0;
            leakage_from[b] = min(leakage_from[b - 1] + leak_slope, band_log2[b]);
            leakage_to[b] = max(leakage_to[b - 1] - leak_slope, band_log2[b] - LEAKAGE_OFFSET);
        }
        for b in (0..NB_TBANDS - 1).rev() {
            let leak_slope = LEAKAGE_SLOPE * (TBANDS[b + 1] - TBANDS[b]) as f32 / 4.0;
            leakage_from[b] = min(leakage_from[b + 1] + leak_slope, leakage_from[b]);
            leakage_to[b] = max(leakage_to[b + 1] - leak_slope, leakage_to[b]);
        }
        // C 742 then clears the bands from NB_TBANDS + 1 to LEAK_BANDS; there
        // are none.
        const _: () = assert!(NB_TBANDS + 1 == LEAK_BANDS);
        for b in 0..NB_TBANDS + 1 {
            // Two terms. The first, from leakage_to, is the boost a weak
            // band needs against the analysis leakage of louder neighbours.
            // The second, from leakage_from, applies to a loud band whose
            // quantization noise leaks into weaker neighbours.
            let boost = max(0.0, leakage_to[b] - band_log2[b])
                + max(0.0, band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET));
            // `as i32` saturates, like the conversion of C on 64-bit ARM.
            let q6 = (0.5 + (64.0 * boost) as f64).floor() as i32;
            self.info[slot].leak_boost[b] = q6.min(255) as u8;
        }

        // Spectral variability (C 756-775). Only the distances to the row
        // written in this block are new (see `Cache::dist`). The sums run
        // over the bands in order, as in C; the rows are independent sums.
        let row = &self.log_e[e_count];
        let mut new_dist = [0.0f32; NB_FRAMES];
        for k in 0..NB_TBANDS {
            for (d, other) in new_dist.iter_mut().zip(&self.log_e) {
                let tmp = row[k] - other[k];
                *d = mac(*d, tmp, tmp);
            }
        }
        for (j, &d) in new_dist.iter().enumerate() {
            self.cache.dist[e_count][j] = d;
            self.cache.dist[j][e_count] = d;
        }
        let mut spec_variability = 0.0f32;
        for i in 0..NB_FRAMES {
            let mut mindist = 1e15f32;
            for j in 0..NB_FRAMES {
                if j != i {
                    mindist = min(mindist, self.cache.dist[i][j]);
                }
            }
            spec_variability += mindist;
        }
        // C computes the root in double precision. The root of an `f32`
        // rounded to `f32` gives the same bits.
        spec_variability = (spec_variability / NB_FRAMES as f32 / NB_TBANDS as f32).sqrt();

        // Bandwidth (C 776-850).
        let mut bandwidth_mask = 0.0f32;
        let mut bandwidth = 0i32;
        let mut max_e = 0.0f32;
        let mut noise_floor = 5.7e-4f32 / (1i32 << 0.max(lsb_depth - 8)) as f32;
        noise_floor *= noise_floor;
        let mut below_max_pitch = 0.0f32;
        let mut above_max_pitch = 0.0f32;
        let mut is_masked = [false; NB_TBANDS + 1];
        for b in 0..NB_TBANDS {
            let band_start = TBANDS[b];
            let band_end = TBANDS[b + 1];
            let width = (band_end - band_start) as f32;
            let mut e = 0.0f32;
            for &bin_e in &bin_energy[band_start..band_end] {
                e += bin_e;
            }
            e *= SCALE_ENER;
            max_e = max(max_e, e);
            if band_start < 64 {
                below_max_pitch += e;
            } else {
                above_max_pitch += e;
            }
            self.mean_e[b] = max((1.0 - alpha_e2) * self.mean_e[b], e);
            let em = max(e, self.mean_e[b]);
            // The band is active only if it is
            // 1) less than 90 dB below the peak band (the most masking
            //    possible from the threshold of hearing and the
            //    loudness-dependent slope of the spreading function), and
            // 2) above the noise floor of the PCM quantization.
            // b+1, because the first CELT band is not part of the bands here.
            if e * 1e9 > max_e && (em > 3.0 * noise_floor * width || e > noise_floor * width) {
                bandwidth = b as i32 + 1;
            }
            // Whether the band is masked (see below).
            // C: `prev_bandwidth >= b+1`.
            let mask_ratio = if self.prev_bandwidth > b as i32 { 0.01 } else { 0.05 };
            is_masked[b] = e < mask_ratio * bandwidth_mask;
            // A follower with a slope of 13 dB per Bark for the spreading.
            bandwidth_mask = max(0.05 * bandwidth_mask, e);
        }
        // The last two bands have no spectrum, only the energy above 12 kHz.
        // The high-pass filter leaks some energy of low frequencies, so the
        // threshold is higher, without cutting off the band by accident.
        if fs == 48000 {
            let e = hp_ener * (1.0 / 3600.0);
            let noise_ratio = if self.prev_bandwidth == 20 { 10.0 } else { 30.0 };
            above_max_pitch += e;
            self.mean_e[NB_TBANDS] = max((1.0 - alpha_e2) * self.mean_e[NB_TBANDS], e);
            let em = max(e, self.mean_e[NB_TBANDS]);
            if em > 3.0 * noise_ratio * noise_floor * 160.0 || e > noise_ratio * noise_floor * 160.0 {
                bandwidth = 20;
            }
            let mask_ratio = if self.prev_bandwidth == 20 { 0.01 } else { 0.05 };
            is_masked[NB_TBANDS] = e < mask_ratio * bandwidth_mask;
        }
        self.info[slot].max_pitch_ratio =
            if above_max_pitch > below_max_pitch { below_max_pitch / above_max_pitch } else { 1.0 };
        // Aliasing of the resampling can put a little energy into the first
        // band that is cut. So a masked last band does not count.
        if bandwidth == 20 && is_masked[NB_TBANDS] {
            bandwidth -= 2;
        } else if bandwidth > 0 && bandwidth <= NB_TBANDS as i32 && is_masked[bandwidth as usize - 1] {
            bandwidth -= 1;
        }
        if self.count <= 2 {
            bandwidth = 20;
        }
        frame_loudness = 20.0 * (frame_loudness as f64).log10() as f32;
        self.e_tracker = max(self.e_tracker - 0.003, frame_loudness);
        self.low_e_count *= 1.0 - alpha_e;
        if frame_loudness < self.e_tracker - 30.0 {
            self.low_e_count += alpha_e;
        }

        // Cepstral values of the log energies and of the middle of the
        // trackers. The products are rounded: clang computes them as
        // vectors and adds them one by one (C 860-873).
        let mut bfcc = [0.0f32; 8];
        let mut mid_e = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = 0.0f32;
            for b in 0..16 {
                sum += DCT_TABLE[i * 16 + b] * log_e[b];
            }
            bfcc[i] = sum;
        }
        for i in 0..8 {
            let mut sum = 0.0f32;
            for b in 0..16 {
                sum += DCT_TABLE[i * 16 + b] * 0.5 * (self.high_e[b] + self.low_e[b]);
            }
            mid_e[i] = sum;
        }

        frame_stationarity /= NB_TBANDS as f32;
        relative_e /= NB_TBANDS as f32;
        if self.count < 10 {
            relative_e = 0.5;
        }
        frame_noisiness /= NB_TBANDS as f32;
        let activity = mac(frame_noisiness, 1.0 - frame_noisiness, relative_e);
        self.info[slot].activity = activity;
        frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
        frame_tonality = max(frame_tonality, self.prev_tonality * 0.8);
        self.prev_tonality = frame_tonality;

        slope /= (8 * 8) as f32;
        self.info[slot].tonality_slope = slope;

        self.e_count = (self.e_count + 1) % NB_FRAMES;
        self.count = (self.count + 1).min(ANALYSIS_COUNT_MAX);
        self.info[slot].tonality = frame_tonality;

        // The features of the network (C 895-929). The nesting of `mac`
        // copies how clang contracts each C expression: the leftmost
        // product is fused, the next one rounded, and each further term is
        // fused into the sum.
        let mem = &mut self.mem;
        let mut features = [0.0f32; 25];
        for i in 0..4 {
            features[i] = mac(
                mac(
                    mac(0.49195 * (mem[i] + mem[i + 16]), -0.12299, bfcc[i] + mem[i + 24]),
                    0.69693,
                    mem[i + 8],
                ),
                -1.4349,
                self.cmean[i],
            );
        }
        for i in 0..4 {
            self.cmean[i] = mac(alpha * bfcc[i], 1.0 - alpha, self.cmean[i]);
        }
        for i in 0..4 {
            features[4 + i] = mac(0.31623 * (mem[i] - mem[i + 16]), 0.63246, bfcc[i] - mem[i + 24]);
        }
        for i in 0..3 {
            features[8 + i] = mac(
                mac(-0.26726 * (mem[i] + mem[i + 16]), 0.53452, bfcc[i] + mem[i + 24]),
                -0.53452,
                mem[i + 8],
            );
        }
        if self.count > 5 {
            for i in 0..9 {
                self.std[i] = mac(alpha * features[i] * features[i], 1.0 - alpha, self.std[i]);
            }
        }
        for i in 0..4 {
            features[i] = bfcc[i] - mid_e[i];
        }
        for i in 0..8 {
            mem[i + 24] = mem[i + 16];
            mem[i + 16] = mem[i + 8];
            mem[i + 8] = mem[i];
            mem[i] = bfcc[i];
        }
        for i in 0..9 {
            features[11 + i] = self.std[i].sqrt() - STD_FEATURE_BIAS[i];
        }
        // This overwrites the deviation of feature 7 written just above, as
        // in C.
        features[18] = spec_variability - 0.78;
        features[20] = frame_tonality - 0.154723;
        features[21] = activity - 0.724643;
        features[22] = frame_stationarity - 0.743717;
        features[23] = slope + 0.069216;
        features[24] = self.low_e_count - 0.067930;

        let mut layer_out = [0.0f32; MAX_NEURONS];
        let mut frame_probs = [0.0f32; 2];
        mlp::compute_dense(&LAYER0, &mut layer_out, &features);
        mlp::compute_gru(&LAYER1, &mut self.rnn_state, &layer_out);
        mlp::compute_dense(&LAYER2, &mut frame_probs, &self.rnn_state);

        let info = &mut self.info[slot];
        // Probability of speech or music rather than noise.
        info.activity_probability = frame_probs[1];
        info.music_prob = frame_probs[0];
        info.bandwidth = bandwidth;
        self.prev_bandwidth = bandwidth;
        info.noisiness = frame_noisiness;
        info.valid = true;
    }
}

/// The block of `inmem` times the analysis window (C 535-542), scaled by
/// the `scale` of the FFT and stored in the bit-reversed order of the FFT:
/// the first loop of libopus `opus_fft_c` (celt/kiss_fft.c:615-635), so
/// that `opus_fft_impl` does the rest. The windowed sample is rounded before
/// the scaling, as in C, where the two products are in separate loops.
///
/// The two real blocks of the analysis share one complex FFT: the older
/// one is the real part, the newer one the imaginary part. 64-bit ARM uses
/// the NEON version in `neon.rs` and x86-64 the SSE version in `x86.rs`;
/// this one is their reference in the tests.
#[cfg_attr(
    any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2")),
    allow(dead_code)
)]
pub(super) fn window_for_fft_scalar(inmem: &[f32; ANALYSIS_BUF_SIZE], st: &KissFftState, fout: &mut [KissCpx; N]) {
    let scale = st.scale();
    let bitrev = &st.bitrev[..N];
    for i in 0..N2 {
        let w = ANALYSIS_WINDOW[i];
        fout[bitrev[i] as usize] = KissCpx::new(w * inmem[i] * scale, w * inmem[N2 + i] * scale);
        fout[bitrev[N - i - 1] as usize] =
            KissCpx::new(w * inmem[N - i - 1] * scale, w * inmem[N + N2 - i - 1] * scale);
    }
}

/// The phase of each FFT bin in the last block, its change, and the fourth
/// power of the remainder of its second difference: the fields `angle`,
/// `d_angle` and `d2_angle` of the state.
pub(super) struct Phase<'a> {
    pub angle: &'a mut [f32; N2],
    pub d_angle: &'a mut [f32; N2],
    pub d2_angle: &'a mut [f32; N2],
}

/// What the analysis of one block computes per FFT bin. Bin 0 is not used.
#[derive(Clone)]
pub(super) struct BinScratch {
    /// C `tonality`: tonality with a delay of 2 blocks.
    pub tonality: [f32; N2],
    /// C `tonality2`: tonality without the delay.
    pub tonality2: [f32; N2],
    /// C `noisiness`.
    pub noisiness: [f32; N2],
    /// C `binE` before the scaling by `SCALE_ENER`: the energy of the bin in
    /// the two blocks.
    pub energy: [f32; N2],
}

impl BinScratch {
    /// Cleared buffers, for the tests of the NEON and SSE versions.
    #[cfg(all(test, any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))))]
    pub(super) fn new() -> Self {
        Self { tonality: [0.0; N2], tonality2: [0.0; N2], noisiness: [0.0; N2], energy: [0.0; N2] }
    }

    /// Writes cleared buffers to `this` (see `crate::in_place`).
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    unsafe fn write_new(this: *mut Self) {
        // SAFETY: the caller guarantees that `this` is valid for writes.
        unsafe {
            write_fields!(this, {
                tonality = |p| write_zero(p),
                tonality2 = |p| write_zero(p),
                noisiness = |p| write_zero(p),
                energy = |p| write_zero(p),
            });
        }
    }
}

/// The phase-based tonality of FFT bins `first..240` (C 570-608) and
/// their energies (C 639-640, 653-654 and 793-794), one bin at a time.
/// The NEON version in `neon.rs` and the SSE version in `x86.rs` do the same
/// four bins at a time.
pub(super) fn tonality_bins_scalar(out: &[KissCpx; N], first: usize, phase: &mut Phase<'_>, r: &mut BinScratch) {
    for i in first..N2 {
        let x1r = out[i].r + out[N - i].r;
        let x1i = out[i].i - out[N - i].i;
        let x2r = out[i].i + out[N - i].i;
        let x2i = out[N - i].r - out[i].r;

        let angle = INV_TWO_PI * fast_atan2f(x1i, x1r);
        let d_angle = angle - phase.angle[i];
        let d2_angle = d_angle - phase.d_angle[i];

        let angle2 = INV_TWO_PI * fast_atan2f(x2i, x2r);
        let d_angle2 = angle2 - angle;
        let d2_angle2 = d_angle2 - d_angle;

        let mut mod1 = d2_angle - float2int(d2_angle) as f32;
        r.noisiness[i] = mod1.abs();
        mod1 *= mod1;
        mod1 *= mod1;

        let mut mod2 = d2_angle2 - float2int(d2_angle2) as f32;
        r.noisiness[i] += mod2.abs();
        mod2 *= mod2;
        mod2 *= mod2;

        // 2*mod2 is exact, so fused and unfused give the same bits.
        let avg_mod = 0.25 * (phase.d2_angle[i] + mod1 + 2.0 * mod2);
        // This introduces an extra delay of 2 frames in the detection.
        r.tonality[i] = 1.0 / mac(1.0, TONALITY_SCALE, avg_mod) - 0.015;
        // No delay on this detection, but it is less reliable.
        r.tonality2[i] = 1.0 / mac(1.0, TONALITY_SCALE, mod2) - 0.015;

        phase.angle[i] = angle2;
        phase.d_angle[i] = d_angle2;
        phase.d2_angle[i] = mod2;

        // The energy of the bin in the two real blocks packed into the
        // complex FFT. C adds four squares from the left. clang fuses the
        // first square into the sum with the second one, which it rounds,
        // and fuses each further square into the sum.
        let (a, b, c, d) = (out[i].r, out[N - i].r, out[i].i, out[N - i].i);
        r.energy[i] = mac(mac(mac(b * b, a, a), c, c), d, d);
    }
}

/// One output sample of libopus `silk_resampler_down2_hp` of the float
/// build (analysis.c:126-157), from the input samples `even` and `odd`.
/// Returns the output and adds the square of the high-band output to
/// `hp_ener`.
///
/// The three all-pass states are independent chains of a subtraction, a
/// product and a sum. They are computed first, so that a processor that
/// runs instructions in order can start the next sample's chains while the
/// sums of this sample's outputs wait. Each value is the same expression as
/// in C.
#[inline(always)]
fn down2_hp_step(s: &mut [f32; 3], hp_ener: &mut f32, even: f32, odd: f32) -> f32 {
    let [s0, s1, s2] = *s;
    // The all-pass section of the even input sample, and the two of the
    // odd input sample (the second one, of -odd, gives the high band).
    let x0 = 0.6074371 * (even - s0);
    let x1 = 0.15063 * (odd - s1);
    let x2 = 0.15063 * (-odd - s2);
    *s = [even + x0, odd + x1, -odd + x2];
    // The low band is the sum of the two sections, the high band their
    // difference, each added from the left as in C.
    let first = s0 + x0;
    let out32 = first + s1 + x1;
    let out32_hp = first + s2 + x2;
    *hp_ener = mac(*hp_ener, out32_hp, out32_hp);
    0.5 * out32
}

/// libopus `silk_resampler_down2_hp` of the float build (analysis.c:115-163):
/// a 2:1 decimator from two all-pass sections. Writes `input.len() / 2`
/// samples to `out` and returns the energy of the high half band.
///
/// The loop makes four outputs per pass. The sums of one output do not
/// depend on the all-pass states of the next, so the compiler can
/// interleave them, which helps processors that run instructions in order.
#[inline]
fn down2_hp(s: &mut [f32; 3], out: &mut [f32], input: &[f32]) -> f32 {
    let mut st = *s;
    let mut hp_ener = 0.0f32;
    let n = input.len() / 2;
    let (blocks, rest) = input[..2 * n].as_chunks::<8>();
    let (out4, out_rest) = out[..n].split_at_mut(4 * blocks.len());
    for (o, q) in out4.as_chunks_mut::<4>().0.iter_mut().zip(blocks) {
        o[0] = down2_hp_step(&mut st, &mut hp_ener, q[0], q[1]);
        o[1] = down2_hp_step(&mut st, &mut hp_ener, q[2], q[3]);
        o[2] = down2_hp_step(&mut st, &mut hp_ener, q[4], q[5]);
        o[3] = down2_hp_step(&mut st, &mut hp_ener, q[6], q[7]);
    }
    for (o, pair) in out_rest.iter_mut().zip(rest.as_chunks::<2>().0) {
        *o = down2_hp_step(&mut st, &mut hp_ener, pair[0], pair[1]);
    }
    *s = st;
    hp_ener
}

/// `down2_hp` of each sample of `input` repeated three times (the 16 kHz
/// path of `downmix_and_resample`), without a buffer for the repeated
/// samples: two samples `a`, `b` become `a a a b b b`, which give the three
/// outputs of the pairs `(a, a)`, `(a, b)` and `(b, b)`. An odd last sample
/// `c` gives one output of `(c, c)`, as C does: `3 * n / 2` outputs in
/// all.
#[inline]
fn down2_hp_repeat3(s: &mut [f32; 3], out: &mut [f32], input: &[f32]) {
    let mut st = *s;
    let mut hp_ener = 0.0f32;
    let (pairs, rest) = input.as_chunks::<2>();
    let (out3, out_rest) = out[..input.len() * 3 / 2].split_at_mut(pairs.len() * 3);
    for (o, &[a, b]) in out3.as_chunks_mut::<3>().0.iter_mut().zip(pairs) {
        o[0] = down2_hp_step(&mut st, &mut hp_ener, a, a);
        o[1] = down2_hp_step(&mut st, &mut hp_ener, a, b);
        o[2] = down2_hp_step(&mut st, &mut hp_ener, b, b);
    }
    if let (Some(&c), Some(o)) = (rest.first(), out_rest.first_mut()) {
        *o = down2_hp_step(&mut st, &mut hp_ener, c, c);
    }
    *s = st;
}

/// libopus `downmix_and_resample` (analysis.c:165-214): `subframe` samples
/// at 24 kHz from sample `offset` (also at 24 kHz) of the caller's buffer,
/// downmixed to one channel and resampled to 24 kHz into `y`. Returns the
/// energy above 12 kHz for 48 kHz input, scaled back to the -1 to 1 range,
/// and 0 otherwise.
fn downmix_and_resample(
    pcm: &AnalysisPcm<'_>,
    y: &mut [f32],
    s: &mut [f32; 3],
    tmp: &mut [f32; MAX_DOWNMIX],
    subframe: i32,
    offset: i32,
    fs: i32,
) -> f32 {
    if subframe == 0 {
        return 0.0;
    }
    let (subframe, offset) = match fs {
        48000 => (subframe * 2, offset * 2),
        16000 => (subframe * 2 / 3, offset * 2 / 3),
        _ => {
            debug_assert_eq!(fs, 24000, "the analysis supports 16, 24 and 48 kHz");
            (subframe, offset)
        }
    };
    let subframe = subframe as usize;
    let tmp = &mut tmp[..subframe];
    downmix(pcm, tmp, offset as usize);
    let halve = match pcm.c2 {
        SecondChannel::AllOthers => pcm.channels == 2,
        SecondChannel::Channel(_) => true,
        SecondChannel::None => false,
    };
    if halve {
        for t in tmp.iter_mut() {
            *t *= 0.5;
        }
    }
    let mut ret = 0.0f32;
    match fs {
        48000 => ret = down2_hp(s, y, tmp),
        16000 => {
            // Each sample three times, then 2:1: a poor resampler, but
            // enough for the analysis, which ignores the aliasing between 8
            // and 12 kHz. The energy is not used.
            down2_hp_repeat3(s, y, tmp);
        }
        _ => y[..subframe].copy_from_slice(tmp),
    }
    ret * (1.0 / 32768.0 / 32768.0)
}

/// libopus `downmix_float` and `downmix_int` (src/opus_encoder.c:747-802):
/// `y.len()` samples from sample `offset` on, channel `c1` plus the
/// channels that `c2` names.
fn downmix(pcm: &AnalysisPcm<'_>, y: &mut [f32], offset: usize) {
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    #[cfg(target_arch = "aarch64")]
    if unsafe { super::neon::downmix(pcm, y, offset) } {
        return;
    }
    // SAFETY: SSE2 is part of this target (see the `cfg`).
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if unsafe { super::x86::downmix(pcm, y, offset) } {
        return;
    }
    downmix_scalar(pcm, y, offset);
}

/// `downmix` for every layout, one sample at a time. It is also the
/// reference of the NEON and SSE versions in the tests.
// The limits are written as in C, followed by the NaN test.
#[allow(clippy::manual_clamp)]
pub(super) fn downmix_scalar(pcm: &AnalysisPcm<'_>, y: &mut [f32], offset: usize) {
    let ch = pcm.channels;
    let c1 = pcm.c1;
    match pcm.pcm {
        PcmRef::F32(x) => {
            let x = &x[offset * ch..];
            for (j, v) in y.iter_mut().enumerate() {
                *v = x[j * ch + c1] * 32768.0;
            }
            // The libopus build fuses the product into the sum. The product
            // is exact except where it overflows, and there the fused form
            // matters: an infinite first channel plus a huge second one.
            match pcm.c2 {
                SecondChannel::Channel(c2) => {
                    for (j, v) in y.iter_mut().enumerate() {
                        *v = mac(*v, x[j * ch + c2], 32768.0);
                    }
                }
                SecondChannel::AllOthers => {
                    for c in 1..ch {
                        for (j, v) in y.iter_mut().enumerate() {
                            *v = mac(*v, x[j * ch + c], 32768.0);
                        }
                    }
                }
                SecondChannel::None => {}
            }
            // Cap the signal to +6 dBFS to avoid problems in the analysis.
            for v in y.iter_mut() {
                if *v < -65536.0 {
                    *v = -65536.0;
                }
                if *v > 65536.0 {
                    *v = 65536.0;
                }
                if v.is_nan() {
                    *v = 0.0;
                }
            }
        }
        PcmRef::I16(x) => {
            let x = &x[offset * ch..];
            for (j, v) in y.iter_mut().enumerate() {
                *v = x[j * ch + c1] as f32;
            }
            match pcm.c2 {
                SecondChannel::Channel(c2) => {
                    for (j, v) in y.iter_mut().enumerate() {
                        *v += x[j * ch + c2] as f32;
                    }
                }
                SecondChannel::AllOthers => {
                    for c in 1..ch {
                        for (j, v) in y.iter_mut().enumerate() {
                            *v += x[j * ch + c] as f32;
                        }
                    }
                }
                SecondChannel::None => {}
            }
        }
        PcmRef::I32(x) => {
            // C `INT24TOSIG(a)`: `(float)(a)*(1.f/256.f)`.
            let sig = |a: i32| a as f32 * (1.0 / 256.0);
            let x = &x[offset * ch..];
            for (j, v) in y.iter_mut().enumerate() {
                *v = sig(x[j * ch + c1]);
            }
            match pcm.c2 {
                SecondChannel::Channel(c2) => {
                    for (j, v) in y.iter_mut().enumerate() {
                        *v += sig(x[j * ch + c2]);
                    }
                }
                SecondChannel::AllOthers => {
                    for c in 1..ch {
                        for (j, v) in y.iter_mut().enumerate() {
                            *v += sig(x[j * ch + c]);
                        }
                    }
                }
                SecondChannel::None => {}
            }
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn constants_have_the_bits_of_libopus() {
        // The static mode of libopus stores the scale as 0.0020833334f,
        // which is the `f32` nearest to 1/480.
        assert_eq!(fft_state().scale(), 1.0 / 480.0);
        assert_eq!(fft_state().nfft(), 480);
        assert_eq!(INV_TWO_PI, 0.159_154_94);
        assert_eq!(PI4, 97.409_09);
        assert_eq!(TONALITY_SCALE, 62_341.816);
        assert_eq!(HALF_LOG2_E, 0.721_347_5);
    }

    #[test]
    fn float2int_rounds_half_to_even() {
        assert_eq!(float2int(0.5), 0);
        assert_eq!(float2int(1.5), 2);
        assert_eq!(float2int(2.5), 2);
        assert_eq!(float2int(-0.5), 0);
        assert_eq!(float2int(-1.5), -2);
        assert_eq!(float2int(-2.49), -2);
        assert_eq!(float2int(1e20), i32::MAX);
        assert_eq!(float2int(-1e20), i32::MIN);
    }

    #[test]
    fn fast_atan2f_is_close_to_atan2() {
        let mut worst = 0.0f32;
        for i in 0..200 {
            for j in 0..200 {
                let y = (i as f32 - 100.0) * 0.37;
                let x = (j as f32 - 100.0) * 0.53;
                // On the negative real axis the approximation gives 0, not
                // pi: with y == 0, x*y is a zero and the two corrections
                // cancel. libopus has the same result there.
                if y == 0.0 {
                    continue;
                }
                let d = (fast_atan2f(y, x) - y.atan2(x)).abs();
                let d = d.min((d - 2.0 * core::f32::consts::PI).abs());
                worst = worst.max(d);
            }
        }
        assert!(worst < 2e-4, "worst {worst}");
        assert_eq!(fast_atan2f(0.0, 0.0), 0.0);
        assert_eq!(fast_atan2f(0.0, -1.0), 0.0);
    }

    /// `down2_hp` gives the bits of the plain loop of libopus, for every
    /// length.
    #[test]
    fn decimator_matches_the_loop_of_libopus() {
        let x: Vec<f32> = (0..300).map(|i| ((i as f32) * 0.29).sin() * 30000.0 - (i % 5) as f32).collect();
        for len in 0..60 {
            let mut s1 = [1.5f32, -2.0, 0.25];
            let mut s2 = s1;
            let mut out1 = vec![0.0f32; len / 2 + 1];
            let mut out2 = out1.clone();
            let mut hp1 = 0.0f32;
            for k in 0..len / 2 {
                out1[k] = down2_hp_step(&mut s1, &mut hp1, x[2 * k], x[2 * k + 1]);
            }
            let hp2 = down2_hp(&mut s2, &mut out2, &x[..len]);
            assert_eq!((hp1.to_bits(), s1.map(f32::to_bits)), (hp2.to_bits(), s2.map(f32::to_bits)), "len {len}");
            assert_eq!(out1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), out2.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
    }

    /// The 16 kHz path without a buffer gives the bits of `down2_hp` on the
    /// samples repeated three times, for even and odd lengths.
    #[test]
    fn repeat3_decimator_matches_the_repeated_input() {
        let x: Vec<f32> = (0..400).map(|i| ((i as f32) * 0.37).sin() * 20000.0 + (i % 7) as f32).collect();
        for n in 0..40 {
            let input = &x[n * 3..n * 3 + n];
            let repeated: Vec<f32> = input.iter().flat_map(|&v| [v, v, v]).collect();
            let mut s1 = [0.5f32, -3.0, 7.25];
            let mut s2 = s1;
            let mut out1 = vec![0.0f32; 3 * n / 2 + 1];
            let mut out2 = out1.clone();
            down2_hp(&mut s1, &mut out1, &repeated);
            down2_hp_repeat3(&mut s2, &mut out2, input);
            assert_eq!(s1.map(f32::to_bits), s2.map(f32::to_bits), "n {n}");
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&out1), bits(&out2), "n {n}");
        }
    }

    #[test]
    fn reset_gives_the_state_of_new() {
        let mut a = TonalityAnalysisState::new(48000);
        let x: Vec<f32> = (0..9600).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let pcm = AnalysisPcm {
            pcm: PcmRef::F32(&x),
            channels: 1,
            c1: 0,
            c2: SecondChannel::AllOthers,
            frame_size: 960,
        };
        for f in 0..10 {
            let p = AnalysisPcm { pcm: PcmRef::F32(&x[f * 960..]), ..pcm };
            a.run(p, 960, 24);
        }
        assert!(a.initialized());
        a.reset();
        let b = TonalityAnalysisState::new(48000);
        assert_eq!(a.angle, b.angle);
        assert_eq!(a.inmem, b.inmem);
        assert_eq!(a.info, b.info);
        assert_eq!(a.rnn_state, b.rnn_state);
        assert_eq!(a.count, b.count);
        assert!(!a.initialized());
    }
}
