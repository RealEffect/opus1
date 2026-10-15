use crate::silk::macros::*;
use core::mem::MaybeUninit;

pub const SILK_RESAMPLER_1_3_COEFS: [i16; 20] = [
    16102, -15162, -13, 0, 20, 26, 5, -31, -43, -4, 65, 90, 7, -157, -248, -44, 593, 1583, 2612,
    3271,
];
const RESAMPLER_DOWN_ORDER_FIR2: usize = 36;
const RESAMPLER_DOWN_BATCH_48K: usize = 480;

#[derive(Clone)]
pub struct SilkResamplerDown1_3 {
    pub ar2: [i32; 2],
    pub fir: [i32; RESAMPLER_DOWN_ORDER_FIR2],
}

impl Default for SilkResamplerDown1_3 {
    fn default() -> Self {
        Self {
            ar2: [0; 2],
            fir: [0; RESAMPLER_DOWN_ORDER_FIR2],
        }
    }
}

pub fn silk_resampler_down_1_3(
    state: &mut SilkResamplerDown1_3,
    output: &mut [i16],
    input: &[i16],
) {
    let total_in = input.len();
    const BUF_SIZE: usize = RESAMPLER_DOWN_BATCH_48K + RESAMPLER_DOWN_ORDER_FIR2;
    let a0 = SILK_RESAMPLER_1_3_COEFS[0];
    let a1 = SILK_RESAMPLER_1_3_COEFS[1];
    let fir_coefs = &SILK_RESAMPLER_1_3_COEFS[2..];

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < total_in {
        let n = (total_in - in_pos).min(RESAMPLER_DOWN_BATCH_48K);

        let mut buf = [0i32; BUF_SIZE];
        buf[..RESAMPLER_DOWN_ORDER_FIR2].copy_from_slice(&state.fir);

        let buf_out = &mut buf[RESAMPLER_DOWN_ORDER_FIR2..RESAMPLER_DOWN_ORDER_FIR2 + n];
        for k in 0..n {
            let out32 = state.ar2[0].wrapping_add((input[in_pos + k] as i32) << 8);
            buf_out[k] = out32;
            let out_q10 = out32 << 2;
            state.ar2[0] = state.ar2[1].wrapping_add(silk_smulwb(out_q10, a0 as i32));
            state.ar2[1] = silk_smulwb(out_q10, a1 as i32);
        }

        let n_out = n / 3;
        for j in 0..n_out {
            let bp = j * 3;
            let mut res_q6 = 0i32;

            let mut i = 0;
            while i < 18 {
                let coef0 = fir_coefs[i] as i32;
                let sym0 = buf[bp + i].wrapping_add(buf[bp + 35 - i]);
                res_q6 = res_q6.wrapping_add(silk_smulwb(sym0, coef0));

                if i + 1 < 18 {
                    let coef1 = fir_coefs[i + 1] as i32;
                    let sym1 = buf[bp + i + 1].wrapping_add(buf[bp + 35 - i - 1]);
                    res_q6 = res_q6.wrapping_add(silk_smulwb(sym1, coef1));
                }
                i += 2;
            }

            if out_pos < output.len() {
                output[out_pos] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
                out_pos += 1;
            }
        }

        state
            .fir
            .copy_from_slice(&buf[n..n + RESAMPLER_DOWN_ORDER_FIR2]);

        in_pos += n;
    }
}

#[derive(Clone, Default)]
pub struct SilkResamplerDown1_6 {
    pub stage1: SilkResamplerDown1_3,
    pub stage2: [i32; 2],
}

pub fn silk_resampler_down_1_6(
    state: &mut SilkResamplerDown1_6,
    output: &mut [i16],
    input: &[i16],
) {
    const BATCH_IN: usize = RESAMPLER_DOWN_BATCH_48K;
    const BATCH_MID: usize = BATCH_IN / 3;

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < input.len() {
        let n_in = (input.len() - in_pos).min(BATCH_IN);
        let n_mid = n_in / 3;
        let n_out = n_mid / 2;

        let mut mid = [0i16; BATCH_MID];
        silk_resampler_down_1_3(
            &mut state.stage1,
            &mut mid[..n_mid],
            &input[in_pos..in_pos + n_in],
        );
        silk_resampler_down2(
            &mut state.stage2,
            &mut output[out_pos..out_pos + n_out],
            &mid[..n_mid],
            n_mid as i32,
        );

        in_pos += n_in;
        out_pos += n_out;
    }
}

pub const SILK_RESAMPLER_DOWN2_0: i32 = 9872;
pub const SILK_RESAMPLER_DOWN2_1: i32 = 39809 - 65536;

pub const SILK_RESAMPLER_2_3_COEFS_LQ: [i16; 6] = [-2797, -6507, 4697, 10739, 1567, 8276];

const SILK_RESAMPLER_UP2_HQ_0: [i16; 3] = [1746, 14986, (39083 - 65536) as i16];
const SILK_RESAMPLER_UP2_HQ_1: [i16; 3] = [6854, 25769, (55542 - 65536) as i16];

const SILK_RESAMPLER_FRAC_FIR_12: [[i16; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

const RESAMPLER_MAX_BATCH_SIZE_MS: i32 = 10;
const RESAMPLER_ORDER_FIR_12: usize = 8;

/// The 8 taps of each of the 12 interpolation phases of the fractional
/// up-sampler, in the order of the input samples: the 4 taps of phase `t`
/// from `SILK_RESAMPLER_FRAC_FIR_12`, then the 4 taps of phase `11 - t` in
/// reverse.
///
/// Rows 12 to 15 are never used. They make the table 16 rows long, so that
/// a phase masked with 15 is a valid index without a bounds check.
const SILK_RESAMPLER_FRAC_FIR_12_TAPS: [[i16; RESAMPLER_ORDER_FIR_12]; 16] = {
    let mut taps = [[0i16; RESAMPLER_ORDER_FIR_12]; 16];
    let mut phase = 0;
    while phase < 12 {
        let mut j = 0;
        while j < 4 {
            taps[phase][j] = SILK_RESAMPLER_FRAC_FIR_12[phase][j];
            taps[phase][7 - j] = SILK_RESAMPLER_FRAC_FIR_12[11 - phase][j];
            j += 1;
        }
        phase += 1;
    }
    taps
};

/// Longest input batch of the fractional up-sampler, in samples.
const IIR_FIR_MAX_BATCH_IN: usize = 480;
/// Its work buffer: the FIR history and two samples per input sample.
const IIR_FIR_BUF_LEN: usize = 2 * IIR_FIR_MAX_BATCH_IN + RESAMPLER_ORDER_FIR_12;

const DELAY_MATRIX_DEC: [[i8; 6]; 3] =
    [[4, 0, 2, 0, 0, 0], [0, 9, 4, 7, 4, 4], [0, 3, 12, 7, 7, 7]];

/// `delay_matrix_enc` (resampler.c:52-59): input-delay compensation values
/// that equalize the total delay across encoder resampling modes.
const DELAY_MATRIX_ENC: [[i8; 3]; 6] = [
    // in\out   8   12   16
    /*  8 */ [6, 0, 3],
    /* 12 */ [0, 7, 3],
    /* 16 */ [0, 1, 10],
    /* 24 */ [0, 2, 6],
    /* 48 */ [18, 10, 12],
    /* 96 */ [0, 0, 44],
];

const RESAMPLER_DOWN_ORDER_FIR0: usize = 18;
const RESAMPLER_DOWN_ORDER_FIR1: usize = 24;

/// `silk_Resampler_3_4_COEFS`: AR2 coefficients followed by 3 phase-FIR
/// halves (resampler_rom.c).
const SILK_RESAMPLER_3_4_COEFS: [i16; 2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2] = [
    -20694, -13867, //
    -49, 64, 17, -157, 353, -496, 163, 11047, 22205, //
    -39, 6, 91, -170, 186, 23, -896, 6336, 19928, //
    -19, -36, 102, -89, -24, 328, -951, 2568, 15909,
];

/// `silk_Resampler_2_3_COEFS`.
const SILK_RESAMPLER_2_3_COEFS: [i16; 2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2] = [
    -14457, -14019, //
    64, 128, -122, 36, 310, -768, 584, 9267, 17733, //
    12, 128, 18, -142, 288, -117, -865, 4123, 14459,
];

/// `silk_Resampler_1_2_COEFS`.
const SILK_RESAMPLER_1_2_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR1 / 2] = [
    616, -14323, //
    -10, 39, 58, -46, -84, 120, 184, -315, -541, 1284, 5380, 9024,
];

/// `silk_Resampler_1_4_COEFS`.
const SILK_RESAMPLER_1_4_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
    22500, -15099, //
    3, -14, -20, -15, 2, 25, 37, 25, -16, -71, -107, -79, 50, 292, 623, 982, 1288, 1464,
];

/// `silk_Resampler_1_6_COEFS`.
const SILK_RESAMPLER_1_6_COEFS: [i16; 2 + RESAMPLER_DOWN_ORDER_FIR2 / 2] = [
    27540, -15257, //
    17, 12, 8, 1, -10, -22, -30, -32, -22, 3, 44, 100, 168, 243, 317, 381, 429, 455,
];

/// A slice of samples as a slice that may be written without being read.
///
/// The kernels below take their output as `&mut [MaybeUninit<T>]`. The same
/// kernel can then fill a caller's slice and a work buffer that has not
/// been initialized, and no reference to uninitialized samples is ever
/// handed out as `&mut [T]`.
///
/// # Safety
/// The caller must only store initialized values through the result. Safe
/// code could otherwise put an uninitialized value into the caller's
/// samples.
#[inline(always)]
unsafe fn as_uninit_mut<T: Copy>(samples: &mut [T]) -> &mut [MaybeUninit<T>] {
    // SAFETY: `MaybeUninit<T>` has the size and alignment of `T`, and every
    // `T` is a valid `MaybeUninit<T>`. The result borrows `samples`
    // mutably for its whole lifetime.
    unsafe { &mut *(samples as *mut [T] as *mut [MaybeUninit<T>]) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResamplerMode {
    Copy,
    Up2HQ,
    IirFir,
    DownFir,
}

#[derive(Clone)]
pub struct SilkResampler {
    s_iir: [i32; 6],

    s_fir: [i16; RESAMPLER_ORDER_FIR_12],

    /// `sFIR.i32` history of the private down-FIR decimator.
    s_down_fir: [i32; RESAMPLER_DOWN_ORDER_FIR2],

    delay_buf: [i16; 48],

    input_delay: i32,

    fs_in_khz: i32,

    fs_out_khz: i32,

    batch_size: i32,

    inv_ratio_q16: i32,

    mode: ResamplerMode,

    /// `FIR_Fracs`: interpolation phases of the down-FIR (0 = unused).
    fir_fracs: i32,

    /// FIR order of the down-FIR (0 = unused).
    down_order: usize,

    /// Down-FIR coefficient table (AR2 pair followed by the FIR halves).
    down_coefs: &'static [i16],
}

impl Default for SilkResampler {
    fn default() -> Self {
        Self {
            s_iir: [0; 6],
            s_fir: [0; RESAMPLER_ORDER_FIR_12],
            s_down_fir: [0; RESAMPLER_DOWN_ORDER_FIR2],
            delay_buf: [0; 48],
            input_delay: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            batch_size: 0,
            inv_ratio_q16: 0,
            mode: ResamplerMode::Copy,
            fir_fracs: 0,
            down_order: 0,
            down_coefs: &[],
        }
    }
}

fn rate_id(rate_hz: i32) -> usize {
    match rate_hz {
        8000 => 0,
        12000 => 1,
        16000 => 2,
        24000 => 3,
        48000 => 4,
        _ => 5,
    }
}

impl SilkResampler {
    pub fn is_initialized(&self) -> bool {
        self.fs_in_khz > 0
    }

    pub fn init(&mut self, fs_hz_in: i32, fs_hz_out: i32) -> i32 {
        *self = Self::default();

        let in_id = rate_id(fs_hz_in);
        let out_id = rate_id(fs_hz_out);

        if in_id > 2 || out_id > 5 || fs_hz_out <= 0 {
            // fs_hz_out <= 0 would divide by zero below (issue #27 deep scan).
            return -1;
        }

        self.input_delay = DELAY_MATRIX_DEC[in_id][out_id] as i32;
        self.init_common(fs_hz_in, fs_hz_out)
    }

    /// Encoder-direction init: `silk_resampler_init(..., forEnc = 1)`
    /// (resampler.c:77-152). Input 8/12/16/24/48 kHz, output 8/12/16 kHz,
    /// delays from `delay_matrix_enc`.
    pub fn init_for_enc(&mut self, fs_hz_in: i32, fs_hz_out: i32) -> i32 {
        *self = Self::default();

        let in_id = rate_id(fs_hz_in);
        let out_id = rate_id(fs_hz_out);

        if in_id > 4 || out_id > 2 || fs_hz_out <= 0 || fs_hz_in < 1000 {
            return -1;
        }

        self.input_delay = DELAY_MATRIX_ENC[in_id][out_id] as i32;
        self.init_common(fs_hz_in, fs_hz_out)
    }

    /// The part of `silk_resampler_init` that does not depend on the
    /// direction: pick the resampling method for the rate ratio and compute
    /// the inverse ratio in Q16.
    ///
    /// The method depends on the ratio only (resampler.c): 2x up-sampling,
    /// any other up-sampling (IIR + FIR), down-sampling (AR2 + FIR with one
    /// coefficient table per ratio), or a plain copy. The decoder also
    /// down-samples: an 8 or 12 kHz decoder receives 12 or 16 kHz SILK
    /// frames. The down-sampler used to be set up for the encoder only, and
    /// the decoder's down-sampling went through the up-sampling method.
    fn init_common(&mut self, fs_hz_in: i32, fs_hz_out: i32) -> i32 {
        self.fs_in_khz = fs_hz_in / 1000;
        self.fs_out_khz = fs_hz_out / 1000;
        self.batch_size = self.fs_in_khz * RESAMPLER_MAX_BATCH_SIZE_MS;

        let mut up2x = 0;
        if fs_hz_out > fs_hz_in {
            if fs_hz_out == fs_hz_in * 2 {
                self.mode = ResamplerMode::Up2HQ;
            } else {
                self.mode = ResamplerMode::IirFir;
                up2x = 1;
            }
        } else if fs_hz_out < fs_hz_in {
            let (fracs, order, coefs): (i32, usize, &'static [i16]) = if fs_hz_out * 4 == fs_hz_in * 3
            {
                (3, RESAMPLER_DOWN_ORDER_FIR0, &SILK_RESAMPLER_3_4_COEFS)
            } else if fs_hz_out * 3 == fs_hz_in * 2 {
                (2, RESAMPLER_DOWN_ORDER_FIR0, &SILK_RESAMPLER_2_3_COEFS)
            } else if fs_hz_out * 2 == fs_hz_in {
                (1, RESAMPLER_DOWN_ORDER_FIR1, &SILK_RESAMPLER_1_2_COEFS)
            } else if fs_hz_out * 3 == fs_hz_in {
                (1, RESAMPLER_DOWN_ORDER_FIR2, &SILK_RESAMPLER_1_3_COEFS)
            } else if fs_hz_out * 4 == fs_hz_in {
                (1, RESAMPLER_DOWN_ORDER_FIR2, &SILK_RESAMPLER_1_4_COEFS)
            } else if fs_hz_out * 6 == fs_hz_in {
                (1, RESAMPLER_DOWN_ORDER_FIR2, &SILK_RESAMPLER_1_6_COEFS)
            } else {
                return -1;
            };
            self.mode = ResamplerMode::DownFir;
            self.fir_fracs = fracs;
            self.down_order = order;
            self.down_coefs = coefs;
        } else {
            self.mode = ResamplerMode::Copy;
        }

        // Ratio of input to output samples in Q16, rounded up.
        self.inv_ratio_q16 = ((((fs_hz_in as i64) << (14 + up2x)) / fs_hz_out as i64) << 2) as i32;
        while silk_smulww(self.inv_ratio_q16, fs_hz_out) < (fs_hz_in << up2x) {
            self.inv_ratio_q16 += 1;
        }

        0
    }

    pub fn process(&mut self, out: &mut [i16], input: &[i16], in_len: i32) -> i32 {
        if in_len < self.fs_in_khz {
            return -1;
        }

        let n_samples = self.fs_in_khz - self.input_delay;

        self.delay_buf[self.input_delay as usize..self.fs_in_khz as usize]
            .copy_from_slice(&input[..n_samples as usize]);

        match self.mode {
            ResamplerMode::Copy => {
                // Clamp writes to the caller-provided output capacity. Well-formed
                // callers always pass an output of at least `in_len * fs_out/fs_in`
                // samples; the clamps only prevent a panic on mis-sized buffers
                // (issue #15 hardening).
                let head = (self.fs_in_khz as usize).min(out.len());
                out[..head].copy_from_slice(&self.delay_buf[..head]);
                let remaining = (in_len - self.fs_in_khz) as usize;
                let dst_start = self.fs_out_khz as usize;
                if dst_start <= out.len() {
                    let copy_len = remaining.min(out.len() - dst_start);
                    out[dst_start..dst_start + copy_len]
                        .copy_from_slice(&input[n_samples as usize..n_samples as usize + copy_len]);
                }
            }
            ResamplerMode::Up2HQ => {
                // 2x upsampling writes 2*n samples for n inputs; clamp `n` to what
                // the output slice can hold (issue #15 hardening).
                let first_len = (self.fs_in_khz as usize).min(out.len() / 2);
                silk_resampler_private_up2_hq(
                    &mut self.s_iir,
                    &mut out[..],
                    &self.delay_buf[..first_len],
                    first_len as i32,
                );
                let dst_start = self.fs_out_khz as usize;
                if out.len() > dst_start {
                    let out_rest = &mut out[dst_start..];
                    let rest_len =
                        ((in_len - self.fs_in_khz).max(0) as usize).min(out_rest.len() / 2);
                    silk_resampler_private_up2_hq(
                        &mut self.s_iir,
                        out_rest,
                        &input[n_samples as usize..n_samples as usize + rest_len],
                        rest_len as i32,
                    );
                }
            }
            ResamplerMode::IirFir => {
                // The first millisecond comes from the delay buffer, the
                // rest from the input.
                let rest_len = (in_len - self.fs_in_khz) as usize;
                iir_fir_resample(
                    &mut self.s_iir,
                    &mut self.s_fir,
                    self.batch_size as usize,
                    self.inv_ratio_q16,
                    out,
                    [
                        &self.delay_buf[..self.fs_in_khz as usize],
                        &input[n_samples as usize..n_samples as usize + rest_len],
                    ],
                );
            }
            ResamplerMode::DownFir => {
                let rest_len = (in_len - self.fs_in_khz) as usize;
                down_fir_resample(
                    &mut self.s_iir,
                    &mut self.s_down_fir,
                    self.batch_size as usize,
                    self.inv_ratio_q16,
                    self.down_order,
                    self.fir_fracs,
                    self.down_coefs,
                    out,
                    [
                        &self.delay_buf[..self.fs_in_khz as usize],
                        &input[n_samples as usize..n_samples as usize + rest_len],
                    ],
                );
            }
        }

        let delay = self.input_delay as usize;
        if delay > 0 {
            let src_start = (in_len as usize).saturating_sub(delay);
            self.delay_buf[..delay].copy_from_slice(&input[src_start..src_start + delay]);
        }

        0
    }
}

/// Longest input batch of the down-sampler, in samples (10 ms at 48 kHz).
const DOWN_FIR_MAX_BATCH_IN: usize = 480;
/// Its work buffer: the FIR history and one sample per input sample.
const DOWN_FIR_BUF_LEN: usize = DOWN_FIR_MAX_BATCH_IN + RESAMPLER_DOWN_ORDER_FIR2;

/// The taps of an interpolating down-FIR (ratios 3/4 and 2/3): 18 taps for
/// each fractional position.
struct DownFirTaps {
    /// The taps in the order of the input samples: the 9 coefficients of
    /// position `i` from the libopus table, then the 9 of position
    /// `fracs - 1 - i` in reverse.
    ///
    /// Rows from `fracs` on are never used. They make the table 4 rows
    /// long, so that a position masked with 3 is a valid index without a
    /// bounds check.
    taps: [[i16; RESAMPLER_DOWN_ORDER_FIR0]; 4],
    /// The same taps as operands of the vector kernel, shifted left by 15:
    /// vqdmulhq_s32(x, t << 15) is (x * t) >> 16. Each row holds taps 0 to
    /// 15 and then taps 14 to 17 with the first two set to zero. A fifth
    /// vector, loaded at sample 14, then covers the last two taps without
    /// reading past the 18 samples.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    wide: [[i32; 20]; 4],
}

const fn down_fir_taps(coefs: &[i16], fracs: usize) -> DownFirTaps {
    const HALF: usize = RESAMPLER_DOWN_ORDER_FIR0 / 2;
    let mut taps = [[0i16; RESAMPLER_DOWN_ORDER_FIR0]; 4];
    let mut wide = [[0i32; 20]; 4];
    let mut i = 0;
    while i < fracs {
        let mut k = 0;
        while k < HALF {
            taps[i][k] = coefs[2 + HALF * i + k];
            taps[i][RESAMPLER_DOWN_ORDER_FIR0 - 1 - k] = coefs[2 + HALF * (fracs - 1 - i) + k];
            k += 1;
        }
        let mut k = 0;
        while k < 16 {
            wide[i][k] = (taps[i][k] as i32) << 15;
            k += 1;
        }
        wide[i][18] = (taps[i][16] as i32) << 15;
        wide[i][19] = (taps[i][17] as i32) << 15;
        i += 1;
    }
    DownFirTaps { taps, wide }
}

static SILK_RESAMPLER_3_4_TAPS: DownFirTaps = down_fir_taps(&SILK_RESAMPLER_3_4_COEFS, 3);
static SILK_RESAMPLER_2_3_TAPS: DownFirTaps = down_fir_taps(&SILK_RESAMPLER_2_3_COEFS, 2);

/// Port of `silk_resampler_private_down_FIR` (resampler_private_down_FIR.c):
/// down-sampling. Each batch of the input runs through a second-order AR
/// filter into a work buffer, and the output samples are computed from that
/// buffer with a FIR filter.
#[allow(clippy::too_many_arguments)]
fn down_fir_resample(
    s_iir: &mut [i32; 6],
    s_fir: &mut [i32; RESAMPLER_DOWN_ORDER_FIR2],
    batch_size: usize,
    index_increment_q16: i32,
    order: usize,
    fracs: i32,
    coefs: &'static [i16],
    mut out: &mut [i16],
    blocks: [&[i16]; 2],
) {
    // libopus leaves this buffer uninitialized as well. Every batch writes
    // the part it reads.
    let mut buf_storage = [MaybeUninit::<i32>::uninit(); DOWN_FIR_BUF_LEN];

    for block in blocks {
        for input in block.chunks(batch_size.max(1)) {
            let n = input.len();
            let buf_len = order + n;
            assert!(order <= RESAMPLER_DOWN_ORDER_FIR2 && buf_len <= DOWN_FIR_BUF_LEN);

            let (history, filtered) = buf_storage[..buf_len].split_at_mut(order);
            history.write_copy_of_slice(&s_fir[..order]);
            ar2(&mut s_iir[..2], filtered, input, coefs[0] as i32, coefs[1] as i32);
            // SAFETY: all `buf_len` elements were written just above: the
            // first `order` by the copy of the FIR history, the other `n`
            // by the AR filter. `ar2` writes every element of its
            // output (it checks that there is one input sample for each).
            let buf: &[i32] = unsafe { buf_storage[..buf_len].assume_init_ref() };

            // One output sample per step of the index while it is below
            // the batch length, but no more than fit into `out`.
            let written = if index_increment_q16 > 0 {
                let steps = (n << 16).div_ceil(index_increment_q16 as usize);
                steps.min(out.len())
            } else {
                0
            };
            let (batch_out, rest) = core::mem::take(&mut out).split_at_mut(written);
            out = rest;
            match order {
                RESAMPLER_DOWN_ORDER_FIR0 => {
                    let taps = if fracs == 3 {
                        &SILK_RESAMPLER_3_4_TAPS
                    } else {
                        &SILK_RESAMPLER_2_3_TAPS
                    };
                    down_fir_interpolated(batch_out, buf, index_increment_q16, fracs, taps);
                }
                RESAMPLER_DOWN_ORDER_FIR1 => {
                    let half: &[i16; RESAMPLER_DOWN_ORDER_FIR1 / 2] =
                        coefs[2..].try_into().unwrap();
                    down_fir_symmetric(batch_out, buf, index_increment_q16, half);
                }
                _ => {
                    let half: &[i16; RESAMPLER_DOWN_ORDER_FIR2 / 2] =
                        coefs[2..].try_into().unwrap();
                    down_fir_symmetric(batch_out, buf, index_increment_q16, half);
                }
            }

            s_fir[..order].copy_from_slice(&buf[n..]);
        }
    }
}

/// Second-order AR filter of the down-samplers (libopus
/// `silk_resampler_private_AR2`, silk/resampler_private_AR2.c): one output
/// in Q8 per input sample. `s` holds the two states in its first two
/// elements.
///
/// The states are kept in local variables during the loop, so that they are
/// not stored and read back for every sample.
///
/// Every element of `out_q8` is written.
#[inline(always)]
fn ar2(
    s: &mut [i32],
    out_q8: &mut [MaybeUninit<i32>],
    input: &[i16],
    a0_q14: i32,
    a1_q14: i32,
) {
    assert!(out_q8.len() == input.len());
    let (mut s0, mut s1) = (s[0], s[1]);
    for (o, &x) in out_q8.iter_mut().zip(input) {
        let out32 = s0.wrapping_add((x as i32) << 8);
        o.write(out32);
        // The state update uses the output in Q10.
        let out_q10 = out32.wrapping_shl(2);
        s0 = s1.wrapping_add(silk_smulwb(out_q10, a0_q14));
        s1 = silk_smulwb(out_q10, a1_q14);
    }
    s[0] = s0;
    s[1] = s1;
}

/// FIR decimation with 18 taps that depend on the fractional part of the
/// position (ratios 3/4 and 2/3). One output sample per element of `out`:
/// sample `i` is computed at the position `i * index_increment_q16` in
/// `buf`.
#[inline(always)]
fn down_fir_interpolated(
    out: &mut [i16],
    buf: &[i32],
    index_increment_q16: i32,
    fracs: i32,
    taps: &DownFirTaps,
) {
    #[cfg(target_arch = "aarch64")]
    down_fir_interpolated_neon(out, buf, index_increment_q16, fracs, taps);
    #[cfg(not(target_arch = "aarch64"))]
    down_fir_interpolated_scalar(out, buf, 0, index_increment_q16, fracs, taps);
}

/// The row of the tap tables for the fractional part of `index_q16`.
#[inline(always)]
fn down_fir_phase(index_q16: i32, fracs: i32) -> usize {
    // The fractional position is below `fracs`, which is 2 or 3. The mask
    // only tells the compiler that it is inside the tables.
    (silk_smulwb(index_q16 & 0xFFFF, fracs) & 3) as usize
}

/// `down_fir_interpolated` for the samples from `index_q16` on.
#[inline(always)]
fn down_fir_interpolated_scalar(
    out: &mut [i16],
    buf: &[i32],
    mut index_q16: i32,
    index_increment_q16: i32,
    fracs: i32,
    taps: &DownFirTaps,
) {
    for o in out.iter_mut() {
        let pos = (index_q16 >> 16) as usize;
        let x: &[i32; RESAMPLER_DOWN_ORDER_FIR0] =
            buf[pos..pos + RESAMPLER_DOWN_ORDER_FIR0].try_into().unwrap();
        let t = &taps.taps[down_fir_phase(index_q16, fracs)];

        let mut res_q6 = 0i32;
        for k in 0..RESAMPLER_DOWN_ORDER_FIR0 {
            res_q6 = silk_smlawb(res_q6, x[k], t[k] as i32);
        }
        *o = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
        index_q16 = index_q16.wrapping_add(index_increment_q16);
    }
}

/// `down_fir_interpolated_scalar` with four output samples per step.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn down_fir_interpolated_neon(
    out: &mut [i16],
    buf: &[i32],
    index_increment_q16: i32,
    fracs: i32,
    taps: &DownFirTaps,
) {
    use core::arch::aarch64::*;

    let mut index_q16 = 0i32;
    let n4 = down_fir_vector_len(
        out.len(),
        buf.len(),
        RESAMPLER_DOWN_ORDER_FIR0,
        index_increment_q16,
    );

    // SAFETY: NEON is part of the aarch64 baseline. `down_fir_vector_len`
    // has checked that every one of the `n4` samples reads 18 samples of
    // `buf` that lie inside it; the loads below read `x[0..16]` and
    // `x[14..18]` of those. The rows of `taps.wide` have 20 elements. A
    // step at `i` has `i + 4 <= n4 <= out.len()` and writes `out[i..i + 4]`.
    unsafe {
        let products = |index_q16: i32| -> int32x4_t {
            let x = buf.as_ptr().add((index_q16 >> 16) as usize);
            let w = taps.wide[down_fir_phase(index_q16, fracs)].as_ptr();
            let p0 = vqdmulhq_s32(vld1q_s32(x), vld1q_s32(w));
            let p1 = vqdmulhq_s32(vld1q_s32(x.add(4)), vld1q_s32(w.add(4)));
            let p2 = vqdmulhq_s32(vld1q_s32(x.add(8)), vld1q_s32(w.add(8)));
            let p3 = vqdmulhq_s32(vld1q_s32(x.add(12)), vld1q_s32(w.add(12)));
            let p4 = vqdmulhq_s32(vld1q_s32(x.add(14)), vld1q_s32(w.add(16)));
            vaddq_s32(vaddq_s32(vaddq_s32(p0, p1), vaddq_s32(p2, p3)), p4)
        };
        let mut i = 0;
        while i < n4 {
            let p0 = products(index_q16);
            let p1 = products(index_q16.wrapping_add(index_increment_q16));
            let p2 = products(index_q16.wrapping_add(index_increment_q16.wrapping_mul(2)));
            let p3 = products(index_q16.wrapping_add(index_increment_q16.wrapping_mul(3)));
            down_fir_store4(out.as_mut_ptr().add(i), p0, p1, p2, p3);
            index_q16 = index_q16.wrapping_add(index_increment_q16.wrapping_mul(4));
            i += 4;
        }
    }
    down_fir_interpolated_scalar(&mut out[n4..], buf, index_q16, index_increment_q16, fracs, taps);
}

/// How many of `n_out` output samples a vector kernel may compute: a
/// multiple of four, all with their `window` input samples inside the
/// `buf_len` samples of the buffer. With valid arguments this is `n_out`
/// rounded down to a multiple of four; the scalar kernel computes the rest
/// (and panics on an argument that is not valid).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn down_fir_vector_len(
    n_out: usize,
    buf_len: usize,
    window: usize,
    index_increment_q16: i32,
) -> usize {
    let n4 = n_out & !3;
    if n4 == 0 || index_increment_q16 <= 0 {
        return 0;
    }
    // The position never decreases, so the last sample has the largest one.
    let last_pos = ((n4 as u64 - 1) * index_increment_q16 as u64) >> 16;
    if last_pos + window as u64 <= buf_len as u64 { n4 } else { 0 }
}

/// Store four output samples: the lanes of each `p` add up to one sample
/// in Q6, which is rounded to Q0 and saturated to 16 bits.
///
/// # Safety
/// `out` must be valid for writing four samples.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn down_fir_store4(
    out: *mut i16,
    p0: core::arch::aarch64::int32x4_t,
    p1: core::arch::aarch64::int32x4_t,
    p2: core::arch::aarch64::int32x4_t,
    p3: core::arch::aarch64::int32x4_t,
) {
    use core::arch::aarch64::*;
    // SAFETY: NEON is part of the aarch64 baseline; the caller guarantees
    // that `out` is valid for four samples.
    unsafe {
        let res_q6 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
        vst1_s16(out, vqmovn_s32(vrshrq_n_s32::<6>(res_q6)));
    }
}

/// FIR decimation with `2 * HALF` symmetric taps (ratios 1/2, 1/3, 1/4 and
/// 1/6): `coefs` holds one half, and each coefficient multiplies the sum of
/// the two samples at equal distance from the middle. One output sample per
/// element of `out`: sample `i` is computed at the position
/// `i * index_increment_q16` in `buf`.
#[inline(always)]
fn down_fir_symmetric<const HALF: usize>(
    out: &mut [i16],
    buf: &[i32],
    index_increment_q16: i32,
    coefs: &[i16; HALF],
) {
    #[cfg(target_arch = "aarch64")]
    down_fir_symmetric_neon(out, buf, index_increment_q16, coefs);
    #[cfg(not(target_arch = "aarch64"))]
    down_fir_symmetric_scalar(out, buf, 0, index_increment_q16, coefs);
}

/// `down_fir_symmetric` for the samples from `index_q16` on.
#[inline(always)]
fn down_fir_symmetric_scalar<const HALF: usize>(
    out: &mut [i16],
    buf: &[i32],
    mut index_q16: i32,
    index_increment_q16: i32,
    coefs: &[i16; HALF],
) {
    for o in out.iter_mut() {
        let pos = (index_q16 >> 16) as usize;
        let (first, second) = buf[pos..pos + 2 * HALF].split_at(HALF);
        let first: &[i32; HALF] = first.try_into().unwrap();
        let second: &[i32; HALF] = second.try_into().unwrap();

        let mut res_q6 = 0i32;
        for k in 0..HALF {
            let pair = first[k].wrapping_add(second[HALF - 1 - k]);
            res_q6 = silk_smlawb(res_q6, pair, coefs[k] as i32);
        }
        *o = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
        index_q16 = index_q16.wrapping_add(index_increment_q16);
    }
}

/// `down_fir_symmetric_scalar` with four output samples per step.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn down_fir_symmetric_neon<const HALF: usize>(
    out: &mut [i16],
    buf: &[i32],
    index_increment_q16: i32,
    coefs: &[i16; HALF],
) {
    use core::arch::aarch64::*;

    // Four pairs per vector. With 18 pairs the fifth vector starts at pair
    // 16; its last two coefficients are zero.
    const MAX_VECTORS: usize = 5;
    let vectors = HALF.div_ceil(4);
    assert!(vectors <= MAX_VECTORS && 4 * vectors <= 2 * HALF);
    // vqdmulhq_s32(x, c << 15) is (x * c) >> 16.
    let mut wide = [0i32; 4 * MAX_VECTORS];
    for (w, &c) in wide.iter_mut().zip(coefs.iter()) {
        *w = (c as i32) << 15;
    }

    let mut index_q16 = 0i32;
    let n4 = down_fir_vector_len(out.len(), buf.len(), 2 * HALF, index_increment_q16);

    // SAFETY: NEON is part of the aarch64 baseline. `down_fir_vector_len`
    // has checked that every one of the `n4` samples reads `2 * HALF`
    // samples of `buf` that lie inside it. Vector `v` reads
    // `x[4 * v..4 * v + 4]` and `x[2 * HALF - 4 - 4 * v..2 * HALF - 4 * v]`
    // of those; both are inside because `4 * vectors <= 2 * HALF` (checked
    // above). `wide` has `4 * MAX_VECTORS` elements. A step at `i` has
    // `i + 4 <= n4 <= out.len()` and writes `out[i..i + 4]`.
    unsafe {
        let products = |index_q16: i32| -> int32x4_t {
            let x = buf.as_ptr().add((index_q16 >> 16) as usize);
            let mut sum = vdupq_n_s32(0);
            for v in 0..vectors {
                let first = vld1q_s32(x.add(4 * v));
                // The mirrored samples, in reverse order.
                let second = vld1q_s32(x.add(2 * HALF - 4 - 4 * v));
                let second = vrev64q_s32(second);
                let second = vextq_s32::<2>(second, second);
                let pairs = vaddq_s32(first, second);
                sum = vaddq_s32(sum, vqdmulhq_s32(pairs, vld1q_s32(wide.as_ptr().add(4 * v))));
            }
            sum
        };
        let mut i = 0;
        while i < n4 {
            let p0 = products(index_q16);
            let p1 = products(index_q16.wrapping_add(index_increment_q16));
            let p2 = products(index_q16.wrapping_add(index_increment_q16.wrapping_mul(2)));
            let p3 = products(index_q16.wrapping_add(index_increment_q16.wrapping_mul(3)));
            down_fir_store4(out.as_mut_ptr().add(i), p0, p1, p2, p3);
            index_q16 = index_q16.wrapping_add(index_increment_q16.wrapping_mul(4));
            i += 4;
        }
    }
    down_fir_symmetric_scalar(&mut out[n4..], buf, index_q16, index_increment_q16, coefs);
}

/// Port of `silk_resampler_private_IIR_FIR` (resampler_private_IIR_FIR.c):
/// up-sampling by any ratio other than 2. Each batch of the input is
/// up-sampled by 2 into a work buffer, and the output samples are
/// interpolated from that buffer with an 8-tap FIR filter.
fn iir_fir_resample(
    s_iir: &mut [i32; 6],
    s_fir: &mut [i16; RESAMPLER_ORDER_FIR_12],
    batch_size: usize,
    index_increment_q16: i32,
    mut out: &mut [i16],
    blocks: [&[i16]; 2],
) {
    // libopus leaves this buffer uninitialized as well. Every batch writes
    // the part it reads.
    let mut buf_storage = [MaybeUninit::<i16>::uninit(); IIR_FIR_BUF_LEN];

    for block in blocks {
        for input in block.chunks(batch_size.max(1)) {
            let n = input.len();
            let buf_len = 2 * n + RESAMPLER_ORDER_FIR_12;
            assert!(buf_len <= IIR_FIR_BUF_LEN);

            let (history, upsampled) =
                buf_storage[..buf_len].split_at_mut(RESAMPLER_ORDER_FIR_12);
            history.write_copy_of_slice(s_fir);
            up2_hq(s_iir, upsampled, input);
            // SAFETY: all `buf_len` elements were written just above: the
            // first 8 by the copy of the FIR history, the other `2 * n` by
            // the 2x up-sampler. `up2_hq` writes every element of its
            // output (it checks that there are two for each input sample).
            let buf: &[i16] = unsafe { buf_storage[..buf_len].assume_init_ref() };

            let written = iir_fir_interpol(out, buf, (n as i32) << 17, index_increment_q16);
            out = &mut core::mem::take(&mut out)[written..];

            s_fir.copy_from_slice(&buf[2 * n..]);
        }
    }
}

/// The interpolation of `iir_fir_resample`: one output sample per step of
/// `index_increment_q16` while the index is below `max_index_q16`. The
/// integer part of the index is the position in `buf`, the fractional part
/// selects one of 12 sets of filter taps. Returns the number of samples
/// written; samples that do not fit into `out` are dropped.
#[inline(always)]
fn iir_fir_interpol(
    out: &mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    index_increment_q16: i32,
) -> usize {
    if max_index_q16 <= 0 || index_increment_q16 <= 0 {
        return 0;
    }
    let steps = (max_index_q16 as usize).div_ceil(index_increment_q16 as usize);
    let n = steps.min(out.len());

    #[cfg(target_arch = "aarch64")]
    iir_fir_interpol_neon(&mut out[..n], buf, 0, index_increment_q16);
    #[cfg(not(target_arch = "aarch64"))]
    iir_fir_interpol_scalar(&mut out[..n], buf, 0, index_increment_q16);
    n
}

/// The filter taps for the fractional part of `index_q16`.
#[inline(always)]
fn iir_fir_taps(index_q16: i32) -> &'static [i16; RESAMPLER_ORDER_FIR_12] {
    // The phase is below 12. The mask only tells the compiler that it is
    // inside the table.
    let phase = (silk_smulwb(index_q16 & 0xFFFF, 12) & 15) as usize;
    &SILK_RESAMPLER_FRAC_FIR_12_TAPS[phase]
}

/// One output sample per element of `out`, the first at `index_q16`.
#[inline(always)]
fn iir_fir_interpol_scalar(
    out: &mut [i16],
    buf: &[i16],
    mut index_q16: i32,
    index_increment_q16: i32,
) {
    for o in out.iter_mut() {
        let taps = iir_fir_taps(index_q16);
        let pos = (index_q16 >> 16) as usize;
        let x: &[i16; RESAMPLER_ORDER_FIR_12] =
            buf[pos..pos + RESAMPLER_ORDER_FIR_12].try_into().unwrap();

        let mut res_q15 = 0i32;
        for j in 0..RESAMPLER_ORDER_FIR_12 {
            res_q15 = silk_smlabb(res_q15, x[j] as i32, taps[j] as i32);
        }
        *o = silk_sat16(silk_rshift_round(res_q15, 15)) as i16;
        index_q16 = index_q16.wrapping_add(index_increment_q16);
    }
}

/// `iir_fir_interpol_scalar` with four output samples per step.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn iir_fir_interpol_neon(
    out: &mut [i16],
    buf: &[i16],
    mut index_q16: i32,
    index_increment_q16: i32,
) {
    use core::arch::aarch64::*;

    let n = out.len();
    let n4 = n & !3;
    if n4 > 0 {
        // The position in `buf` never decreases, so the last sample of the
        // vector part has the largest one.
        debug_assert!(index_q16 >= 0 && index_increment_q16 > 0);
        let last_index_q16 = index_q16 as i64 + (n4 as i64 - 1) * index_increment_q16 as i64;
        assert!(
            index_q16 >= 0
                && index_increment_q16 > 0
                && (last_index_q16 >> 16) as usize + RESAMPLER_ORDER_FIR_12 <= buf.len()
        );

        // SAFETY: NEON is part of the aarch64 baseline. Every load from
        // `buf` reads 8 samples at a position from 0 to that of the last
        // sample, which ends inside `buf` (checked above). The taps are
        // 8-element arrays. A step at `i` has `i + 4 <= n` and writes
        // `out[i..i + 4]`.
        unsafe {
            // The 8 products of one output sample, added in pairs.
            let products = |index_q16: i32| -> int32x4_t {
                let x = vld1q_s16(buf.as_ptr().add((index_q16 >> 16) as usize));
                let taps = vld1q_s16(iir_fir_taps(index_q16).as_ptr());
                vmlal_high_s16(vmull_s16(vget_low_s16(x), vget_low_s16(taps)), x, taps)
            };
            let mut i = 0;
            while i < n4 {
                let p0 = products(index_q16);
                let p1 = products(index_q16.wrapping_add(index_increment_q16));
                let p2 = products(index_q16.wrapping_add(index_increment_q16.wrapping_mul(2)));
                let p3 = products(index_q16.wrapping_add(index_increment_q16.wrapping_mul(3)));
                // One sum per output sample in each lane.
                let res_q15 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
                // Round to Q0 and saturate to 16 bits.
                vst1_s16(
                    out.as_mut_ptr().add(i),
                    vqmovn_s32(vrshrq_n_s32::<15>(res_q15)),
                );
                index_q16 = index_q16.wrapping_add(index_increment_q16.wrapping_mul(4));
                i += 4;
            }
        }
    }
    iir_fir_interpol_scalar(&mut out[n4..], buf, index_q16, index_increment_q16);
}

/// One first-order all-pass section of the 2x up-sampler: returns its
/// output and updates its state. `coef_q16` is positive and below 65536.
///
/// libopus stores the larger coefficients minus 65536, so that they fit in
/// 16 bits, and computes `y + ((y * c) >> 16)` (`silk_SMLAWB`). That is
/// `(y * (c + 65536)) >> 16` exactly, because `y * 65536` has no bits below
/// bit 16. One form for all sections keeps the path from one state to the
/// next equally short for all of them.
#[inline(always)]
fn up2_allpass(input: i32, state: &mut i32, coef_q16: i32) -> i32 {
    let y = input.wrapping_sub(*state);
    let x = ((y as i64 * coef_q16 as i64) >> 16) as i32;
    let output = state.wrapping_add(x);
    *state = input.wrapping_add(x);
    output
}

/// Coefficients of the two all-pass chains of the 2x up-sampler in Q16:
/// `UP2_HQ_COEFS[stage]` is the pair for the even and the odd output.
const UP2_HQ_COEFS: [[i32; 2]; 3] = [
    [
        SILK_RESAMPLER_UP2_HQ_0[0] as i32,
        SILK_RESAMPLER_UP2_HQ_1[0] as i32,
    ],
    [
        SILK_RESAMPLER_UP2_HQ_0[1] as i32,
        SILK_RESAMPLER_UP2_HQ_1[1] as i32,
    ],
    [
        SILK_RESAMPLER_UP2_HQ_0[2] as i32 + 65536,
        SILK_RESAMPLER_UP2_HQ_1[2] as i32 + 65536,
    ],
];

/// Port of `silk_resampler_private_up2_HQ`: up-sampling by 2. Each input
/// sample runs through two chains of three all-pass sections; one chain
/// gives the even output sample, the other the odd one. `s` holds the six
/// states.
pub fn silk_resampler_private_up2_hq(s: &mut [i32], out: &mut [i16], input: &[i16], len: i32) {
    let len = len as usize;
    if len == 0 {
        return;
    }
    let input = &input[..len];
    let state: &mut [i32; 6] = (&mut s[..6]).try_into().unwrap();
    // SAFETY: `up2_hq` only stores initialized samples.
    let out = unsafe { as_uninit_mut(&mut out[..2 * len]) };
    up2_hq(state, out, input);
}

/// The 2x up-sampler: two output samples per input sample. Every element
/// of `out` is written.
#[inline(always)]
fn up2_hq(s: &mut [i32; 6], out: &mut [MaybeUninit<i16>], input: &[i16]) {
    #[cfg(target_arch = "aarch64")]
    up2_hq_neon(s, out, input);
    #[cfg(not(target_arch = "aarch64"))]
    up2_hq_scalar(s, out, input);
}

/// The states are kept in local variables during the loop. Stored in the
/// slice they would be written to memory and read back for every sample,
/// with a bounds check on each access.
#[inline(always)]
fn up2_hq_scalar(s: &mut [i32; 6], out: &mut [MaybeUninit<i16>], input: &[i16]) {
    assert!(out.len() == 2 * input.len());
    let [mut s0, mut s1, mut s2, mut s3, mut s4, mut s5] = *s;

    for (o, &x) in out.as_chunks_mut::<2>().0.iter_mut().zip(input) {
        let in32 = (x as i32) << 10;

        let out32 = up2_allpass(in32, &mut s0, UP2_HQ_COEFS[0][0]);
        let out32 = up2_allpass(out32, &mut s1, UP2_HQ_COEFS[1][0]);
        let out32 = up2_allpass(out32, &mut s2, UP2_HQ_COEFS[2][0]);
        o[0].write(silk_sat16(silk_rshift_round(out32, 10)) as i16);

        let out32 = up2_allpass(in32, &mut s3, UP2_HQ_COEFS[0][1]);
        let out32 = up2_allpass(out32, &mut s4, UP2_HQ_COEFS[1][1]);
        let out32 = up2_allpass(out32, &mut s5, UP2_HQ_COEFS[2][1]);
        o[1].write(silk_sat16(silk_rshift_round(out32, 10)) as i16);
    }

    *s = [s0, s1, s2, s3, s4, s5];
}

/// The two chains side by side: lane 0 of each vector belongs to the even
/// output sample, lane 1 to the odd one. One multiplication serves both
/// chains.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn up2_hq_neon(s: &mut [i32; 6], out: &mut [MaybeUninit<i16>], input: &[i16]) {
    use core::arch::aarch64::*;

    assert!(out.len() == 2 * input.len());
    // Two input samples per step, so that a step stores four output samples.
    let (in_pairs, in_rest) = input.as_chunks::<2>();
    let (out_quads, out_rest) = out.as_chunks_mut::<4>();

    // SAFETY: NEON is part of the aarch64 baseline. The loads read 2
    // elements of 2-element arrays, the first store writes the 4 elements
    // of a 4-element array, and the last one the first 2 of the 6 states.
    unsafe {
        // vqdmulh_s32(y, c << 15) is (y * c) >> 16.
        let c0 = vshl_n_s32::<15>(vld1_s32(UP2_HQ_COEFS[0].as_ptr()));
        let c1 = vshl_n_s32::<15>(vld1_s32(UP2_HQ_COEFS[1].as_ptr()));
        let c2 = vshl_n_s32::<15>(vld1_s32(UP2_HQ_COEFS[2].as_ptr()));
        let mut st0 = vld1_s32([s[0], s[3]].as_ptr());
        let mut st1 = vld1_s32([s[1], s[4]].as_ptr());
        let mut st2 = vld1_s32([s[2], s[5]].as_ptr());

        // The three all-pass sections for one input sample.
        let mut step = |x: i16| -> int32x2_t {
            let in32 = vdup_n_s32((x as i32) << 10);

            let x0 = vqdmulh_s32(vsub_s32(in32, st0), c0);
            let out0 = vadd_s32(st0, x0);
            st0 = vadd_s32(in32, x0);

            let x1 = vqdmulh_s32(vsub_s32(out0, st1), c1);
            let out1 = vadd_s32(st1, x1);
            st1 = vadd_s32(out0, x1);

            let x2 = vqdmulh_s32(vsub_s32(out1, st2), c2);
            let out2 = vadd_s32(st2, x2);
            st2 = vadd_s32(out1, x2);
            out2
        };

        for (o, x) in out_quads.iter_mut().zip(in_pairs) {
            let first = step(x[0]);
            let second = step(x[1]);
            // Round to Q0 and saturate to 16 bits.
            let res = vqrshrn_n_s32::<10>(vcombine_s32(first, second));
            vst1_s16(o.as_mut_ptr() as *mut i16, res);
        }

        vst1_s32(s.as_mut_ptr(), vzip1_s32(st0, st1));
        s[2] = vget_lane_s32::<0>(st2);
        s[3] = vget_lane_s32::<1>(st0);
        s[4] = vget_lane_s32::<1>(st1);
        s[5] = vget_lane_s32::<1>(st2);
    }
    // The last sample of an odd number of input samples.
    up2_hq_scalar(s, out_rest, in_rest);
}

/// Port of `silk_resampler_down2`: down-sampling by 2. The even input
/// samples run through one first-order all-pass section and the odd ones
/// through another; an output sample is the sum of the two section outputs.
/// `s` holds the two states.
///
/// The states are kept in local variables during the loop, so that they are
/// not stored and read back, with a bounds check, for every sample.
pub fn silk_resampler_down2(s: &mut [i32], out: &mut [i16], input: &[i16], in_len: i32) {
    // libopus stores this coefficient minus 65536, so that it fits in 16
    // bits, and computes `y + ((y * c) >> 16)` (`silk_SMLAWB`). That is
    // `(y * (c + 65536)) >> 16` exactly, because `y * 65536` has no bits
    // below bit 16. This form is one multiplication and one shift, like the
    // other section.
    const DOWN2_1_Q16: i32 = SILK_RESAMPLER_DOWN2_1 + 65536;

    let len2 = (in_len >> 1) as usize;
    if len2 == 0 {
        return;
    }
    let input = &input[..2 * len2];
    let out = &mut out[..len2];
    let (mut s0, mut s1) = (s[0], s[1]);

    for (o, pair) in out.iter_mut().zip(input.as_chunks::<2>().0) {
        // All-pass section for the even input sample, in Q10.
        let in32 = (pair[0] as i32) << 10;
        let y = in32.wrapping_sub(s0);
        let x = ((y as i64 * DOWN2_1_Q16 as i64) >> 16) as i32;
        let mut out32 = s0.wrapping_add(x);
        s0 = in32.wrapping_add(x);

        // All-pass section for the odd input sample, added to the output
        // of the first section.
        let in32 = (pair[1] as i32) << 10;
        let y = in32.wrapping_sub(s1);
        let x = silk_smulwb(y, SILK_RESAMPLER_DOWN2_0);
        out32 = out32.wrapping_add(s1);
        out32 = out32.wrapping_add(x);
        s1 = in32.wrapping_add(x);

        *o = silk_sat16(silk_rshift_round(out32, 11)) as i16;
    }

    s[0] = s0;
    s[1] = s1;
}

/// Port of `silk_resampler_private_AR2`: the second-order AR filter in
/// front of the FIR filters of the down-samplers. One output in Q8 per input
/// sample; `s` holds the two states.
pub fn silk_resampler_private_ar2(
    s: &mut [i32],
    out_q8: &mut [i32],
    input: &[i16],
    a_q14: &[i16],
    len: i32,
) {
    let len = len as usize;
    if len == 0 {
        return;
    }
    // SAFETY: `ar2` only stores initialized samples.
    let out = unsafe { as_uninit_mut(&mut out_q8[..len]) };
    ar2(s, out, &input[..len], a_q14[0] as i32, a_q14[1] as i32);
}

const RESAMPLER_MAX_BATCH_SIZE_IN: i32 = 480;
const ORDER_FIR: usize = 4;

/// Port of `silk_resampler_down2_3` (silk/resampler_down2_3.c):
/// down-sampling by 2/3. A second-order AR filter into a work buffer in Q8,
/// then two output samples for every three samples of that buffer. `s` holds
/// the four samples of FIR history and the two AR states.
pub fn silk_resampler_down2_3(s: &mut [i32], out: &mut [i16], input: &[i16], in_len: i32) {
    const BUF_LEN: usize = RESAMPLER_MAX_BATCH_SIZE_IN as usize + ORDER_FIR;
    const C: [i32; 4] = [
        SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32,
        SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
        SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32,
        SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
    ];

    // libopus leaves this buffer uninitialized as well. Its first
    // `ORDER_FIR` elements are written right below and again at the end of
    // each batch; a batch writes the `n` elements behind them before it
    // reads anything.
    let mut buf = [MaybeUninit::<i32>::uninit(); BUF_LEN];
    let mut in_idx = 0;
    let mut out_idx = 0;
    let mut remaining_len = in_len;

    buf[..ORDER_FIR].write_copy_of_slice(&s[..ORDER_FIR]);

    while remaining_len > 0 {
        let n = remaining_len.min(RESAMPLER_MAX_BATCH_SIZE_IN) as usize;

        ar2(
            &mut s[ORDER_FIR..ORDER_FIR + 2],
            &mut buf[ORDER_FIR..ORDER_FIR + n],
            &input[in_idx..in_idx + n],
            SILK_RESAMPLER_2_3_COEFS_LQ[0] as i32,
            SILK_RESAMPLER_2_3_COEFS_LQ[1] as i32,
        );
        // SAFETY: the first `ORDER_FIR` elements were written before the
        // loop or at the end of the previous batch, and the next `n` by
        // `ar2` just above, which writes every element of its output.
        let filtered: &[i32] = unsafe { buf[..ORDER_FIR + n].assume_init_ref() };

        // Two output samples per three input samples; an incomplete group
        // at the end of the batch gives none.
        let groups = n / 3;
        let (pairs, _) = out[out_idx..out_idx + 2 * groups].as_chunks_mut::<2>();
        for (o, group) in pairs.iter_mut().zip(0..groups) {
            let x: &[i32; 5] = filtered[3 * group..3 * group + 5].try_into().unwrap();

            let mut res_q6 = silk_smulwb(x[0], C[0]);
            res_q6 = silk_smlawb(res_q6, x[1], C[1]);
            res_q6 = silk_smlawb(res_q6, x[2], C[3]);
            res_q6 = silk_smlawb(res_q6, x[3], C[2]);
            o[0] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;

            let mut res_q6 = silk_smulwb(x[1], C[2]);
            res_q6 = silk_smlawb(res_q6, x[2], C[3]);
            res_q6 = silk_smlawb(res_q6, x[3], C[1]);
            res_q6 = silk_smlawb(res_q6, x[4], C[0]);
            o[1] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
        }
        out_idx += 2 * groups;

        in_idx += n;
        remaining_len -= n as i32;

        // The last `ORDER_FIR` samples are the history of the next batch,
        // or of the next call.
        let history: [i32; ORDER_FIR] = filtered[n..].try_into().unwrap();
        if remaining_len > 0 {
            buf[..ORDER_FIR].write_copy_of_slice(&history);
        } else {
            s[..ORDER_FIR].copy_from_slice(&history);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;

    /// Small deterministic pseudo-random generator.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() as u64 % (hi as i64 - lo as i64 + 1) as u64) as i32
        }
        /// A value that is often at a limit of `i16`.
        fn i16_edgy(&mut self) -> i16 {
            match self.next() % 8 {
                0 => i16::MAX,
                1 => i16::MIN,
                2 => 0,
                _ => self.next() as i16,
            }
        }
        /// A value that is often at a limit of `i32`.
        fn i32_edgy(&mut self) -> i32 {
            match self.next() % 8 {
                0 => i32::MAX,
                1 => i32::MIN,
                2 => 0,
                _ => self.next() as i32,
            }
        }
        /// Speech-like samples for even trials, samples at the limits for
        /// odd ones.
        fn signal(&mut self, n: usize, trial: usize) -> Vec<i16> {
            (0..n)
                .map(|_| if trial % 2 == 0 { self.range(-12000, 12000) as i16 } else { self.i16_edgy() })
                .collect()
        }
        /// Filter states: of the size real signals produce for even trials,
        /// anything for odd ones.
        fn states<const N: usize>(&mut self, trial: usize) -> [i32; N] {
            core::array::from_fn(|_| {
                if trial % 2 == 0 { self.range(-(1 << 26), 1 << 26) } else { self.i32_edgy() }
            })
        }
    }

    /// The view of an output slice that the kernels write to.
    fn uninit<T: Copy>(out: &mut [T]) -> &mut [MaybeUninit<T>] {
        // SAFETY: the kernels only store initialized samples.
        unsafe { as_uninit_mut(out) }
    }

    /// Output lengths every kernel test runs: empty, one sample, every tail
    /// length of the 2- and 4-sample steps, and frame-sized ones.
    const LENGTHS: [usize; 20] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 31, 80, 81, 159, 160];

    /// The Q16 index increment of every rate pair that uses the fractional
    /// up-sampler or the down-sampler, and two arbitrary ones.
    fn increments(up: bool) -> Vec<i32> {
        let mut found = Vec::new();
        for fs_in in [8000, 12000, 16000, 24000, 48000] {
            for fs_out in [8000, 12000, 16000, 24000, 48000] {
                let mut rs = SilkResampler::default();
                let ok = if fs_in <= 16000 {
                    rs.init(fs_in, fs_out) == 0
                } else {
                    fs_out <= 16000 && rs.init_for_enc(fs_in, fs_out) == 0
                };
                let wanted = if up { ResamplerMode::IirFir } else { ResamplerMode::DownFir };
                if ok && rs.mode == wanted && !found.contains(&rs.inv_ratio_q16) {
                    found.push(rs.inv_ratio_q16);
                }
            }
        }
        assert!(found.len() >= 4);
        found.extend(if up { [1, 131071] } else { [65536, 400000] });
        found
    }

    // -- 2x up-sampler ---------------------------------------------------------

    /// `silk_resampler_private_up2_HQ` as libopus writes it, with the states
    /// in the slice.
    fn up2_hq_reference(s: &mut [i32; 6], out: &mut [i16], input: &[i16]) {
        for k in 0..input.len() {
            let in32 = (input[k] as i32) << 10;

            let y = in32.wrapping_sub(s[0]);
            let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_0[0] as i32);
            let out32_1 = s[0].wrapping_add(x);
            s[0] = in32.wrapping_add(x);
            let y = out32_1.wrapping_sub(s[1]);
            let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_0[1] as i32);
            let out32_2 = s[1].wrapping_add(x);
            s[1] = out32_1.wrapping_add(x);
            let y = out32_2.wrapping_sub(s[2]);
            let x = silk_smlawb(y, y, SILK_RESAMPLER_UP2_HQ_0[2] as i32);
            let out32_1 = s[2].wrapping_add(x);
            s[2] = out32_2.wrapping_add(x);
            out[2 * k] = silk_sat16(silk_rshift_round(out32_1, 10)) as i16;

            let y = in32.wrapping_sub(s[3]);
            let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_1[0] as i32);
            let out32_1 = s[3].wrapping_add(x);
            s[3] = in32.wrapping_add(x);
            let y = out32_1.wrapping_sub(s[4]);
            let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_1[1] as i32);
            let out32_2 = s[4].wrapping_add(x);
            s[4] = out32_1.wrapping_add(x);
            let y = out32_2.wrapping_sub(s[5]);
            let x = silk_smlawb(y, y, SILK_RESAMPLER_UP2_HQ_1[2] as i32);
            let out32_1 = s[5].wrapping_add(x);
            s[5] = out32_2.wrapping_add(x);
            out[2 * k + 1] = silk_sat16(silk_rshift_round(out32_1, 10)) as i16;
        }
    }

    #[test]
    fn up2_hq_kernels_match_the_reference() {
        let mut rng = Lcg(21);
        for &n in &LENGTHS {
            for trial in 0..16 {
                let input = rng.signal(n, trial);
                let state: [i32; 6] = rng.states(trial / 2);

                let mut expect_state = state;
                let mut expect = vec![0i16; 2 * n];
                up2_hq_reference(&mut expect_state, &mut expect, &input);

                let mut s = state;
                let mut out = vec![0x5555i16; 2 * n];
                up2_hq_scalar(&mut s, uninit(&mut out), &input);
                assert_eq!((&out, s), (&expect, expect_state), "scalar: n={n} trial={trial}");
                #[cfg(target_arch = "aarch64")]
                {
                    let mut s = state;
                    let mut out = vec![0x5555i16; 2 * n];
                    up2_hq_neon(&mut s, uninit(&mut out), &input);
                    assert_eq!((&out, s), (&expect, expect_state), "NEON: n={n} trial={trial}");
                }
                // The public function, with slices longer than it needs.
                let mut s = state;
                let mut out = vec![0x5555i16; 2 * n + 3];
                let mut longer = input.clone();
                longer.push(77);
                silk_resampler_private_up2_hq(&mut s, &mut out, &longer, n as i32);
                assert_eq!((&out[..2 * n], s), (&expect[..], expect_state), "public: n={n} trial={trial}");
                assert_eq!(out[2 * n..], [0x5555; 3], "public wrote past its output: n={n}");
            }
        }
    }

    // -- fractional up-sampler -------------------------------------------------

    /// The interpolation loop of `silk_resampler_private_IIR_FIR` with the
    /// libopus table of 4 taps per phase.
    fn iir_fir_interpol_reference(out: &mut [i16], buf: &[i16], inc: i32) {
        let mut index_q16 = 0i32;
        for o in out.iter_mut() {
            let t = silk_smulwb(index_q16 & 0xFFFF, 12) as usize;
            let b = &buf[(index_q16 >> 16) as usize..];
            let fir = &SILK_RESAMPLER_FRAC_FIR_12;
            let mut res = silk_smulbb(b[0] as i32, fir[t][0] as i32);
            res = silk_smlabb(res, b[1] as i32, fir[t][1] as i32);
            res = silk_smlabb(res, b[2] as i32, fir[t][2] as i32);
            res = silk_smlabb(res, b[3] as i32, fir[t][3] as i32);
            res = silk_smlabb(res, b[4] as i32, fir[11 - t][3] as i32);
            res = silk_smlabb(res, b[5] as i32, fir[11 - t][2] as i32);
            res = silk_smlabb(res, b[6] as i32, fir[11 - t][1] as i32);
            res = silk_smlabb(res, b[7] as i32, fir[11 - t][0] as i32);
            *o = silk_sat16(silk_rshift_round(res, 15)) as i16;
            index_q16 += inc;
        }
    }

    #[test]
    fn iir_fir_interpol_kernels_match_the_reference() {
        let mut rng = Lcg(22);
        for inc in increments(true) {
            for &n in &LENGTHS {
                for trial in 0..6 {
                    // Exactly as long as the last output sample needs.
                    let last_pos = if n == 0 { 0 } else { ((n as i64 - 1) * inc as i64 >> 16) as usize };
                    let buf = rng.signal(last_pos + RESAMPLER_ORDER_FIR_12, trial);
                    let mut expect = vec![0i16; n];
                    iir_fir_interpol_reference(&mut expect, &buf, inc);

                    let mut out = vec![0x5555i16; n];
                    iir_fir_interpol_scalar(&mut out, &buf, 0, inc);
                    assert_eq!(out, expect, "scalar: inc={inc} n={n} trial={trial}");
                    #[cfg(target_arch = "aarch64")]
                    {
                        let mut out = vec![0x5555i16; n];
                        iir_fir_interpol_neon(&mut out, &buf, 0, inc);
                        assert_eq!(out, expect, "NEON: inc={inc} n={n} trial={trial}");
                    }
                    // The dispatching function: it stops at `max_index_q16`
                    // or at the end of `out`, whichever comes first.
                    let max_index_q16 = (n as i64 * inc as i64) as i32;
                    let mut out = vec![0x5555i16; n + 2];
                    assert_eq!(iir_fir_interpol(&mut out, &buf, max_index_q16, inc), n);
                    assert_eq!(out[..n], expect[..], "dispatch: inc={inc} n={n} trial={trial}");
                    assert_eq!(out[n..], [0x5555; 2]);
                    let short = n / 2;
                    let mut out = vec![0x5555i16; short];
                    assert_eq!(iir_fir_interpol(&mut out, &buf, max_index_q16, inc), short);
                    assert_eq!(out[..], expect[..short], "short output: inc={inc} n={n}");
                }
            }
        }
    }

    /// The 8-tap table against the libopus table it is built from.
    #[test]
    fn frac_fir_tap_table_matches_the_libopus_table() {
        for phase in 0..12 {
            let taps = &SILK_RESAMPLER_FRAC_FIR_12_TAPS[phase];
            for j in 0..4 {
                assert_eq!(taps[j], SILK_RESAMPLER_FRAC_FIR_12[phase][j]);
                assert_eq!(taps[4 + j], SILK_RESAMPLER_FRAC_FIR_12[11 - phase][3 - j]);
            }
        }
    }

    // -- down-sampler ------------------------------------------------------------

    /// The loop of `silk_resampler_private_AR2` as libopus writes it.
    fn ar2_loop_reference(s: &mut [i32], out_q8: &mut [i32], input: &[i16], a0: i32, a1: i32) {
        for k in 0..input.len() {
            let out32 = s[0].wrapping_add((input[k] as i32) << 8);
            out_q8[k] = out32;
            let out_q10 = out32.wrapping_shl(2);
            s[0] = s[1].wrapping_add(silk_smulwb(out_q10, a0));
            s[1] = silk_smulwb(out_q10, a1);
        }
    }

    #[test]
    fn ar2_matches_the_reference() {
        let mut rng = Lcg(23);
        for coefs in [&SILK_RESAMPLER_3_4_COEFS[..], &SILK_RESAMPLER_1_2_COEFS, &SILK_RESAMPLER_1_6_COEFS] {
            for &n in &LENGTHS {
                for trial in 0..8 {
                    let input = rng.signal(n, trial);
                    let state: [i32; 6] = rng.states(trial / 2);
                    let (a0, a1) = (coefs[0] as i32, coefs[1] as i32);
                    let (mut s_ref, mut s_new) = (state, state);
                    let mut expect = vec![0i32; n];
                    ar2_loop_reference(&mut s_ref, &mut expect, &input, a0, a1);
                    let mut out = vec![0x5555_5555i32; n];
                    ar2(&mut s_new, uninit(&mut out), &input, a0, a1);
                    assert_eq!((&out, s_new), (&expect, s_ref), "n={n} trial={trial}");
                }
            }
        }
    }

    /// The three interpolation loops of `silk_resampler_private_down_FIR`,
    /// reading the libopus coefficient tables.
    fn down_fir_reference(out: &mut [i16], buf: &[i32], inc: i32, order: usize, fracs: i32, coefs: &[i16]) {
        let fir = &coefs[2..];
        let mut index_q16 = 0i32;
        for o in out.iter_mut() {
            let p = (index_q16 >> 16) as usize;
            let res_q6 = match order {
                RESAMPLER_DOWN_ORDER_FIR0 => {
                    let ind = silk_smulwb(index_q16 & 0xFFFF, fracs) as usize;
                    let mut r = silk_smulwb(buf[p], fir[9 * ind] as i32);
                    for k in 1..9 {
                        r = silk_smlawb(r, buf[p + k], fir[9 * ind + k] as i32);
                    }
                    let mirror = &fir[9 * (fracs as usize - 1 - ind)..];
                    for k in 0..9 {
                        r = silk_smlawb(r, buf[p + 17 - k], mirror[k] as i32);
                    }
                    r
                }
                _ => {
                    let mut r = 0i32;
                    for k in 0..order / 2 {
                        let pair = buf[p + k].wrapping_add(buf[p + order - 1 - k]);
                        r = silk_smlawb(r, pair, fir[k] as i32);
                    }
                    r
                }
            };
            *o = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
            index_q16 += inc;
        }
    }

    /// A work buffer for `n` output samples: exactly as long as the last
    /// one needs.
    fn down_fir_buf(rng: &mut Lcg, n: usize, inc: i32, order: usize, trial: usize) -> Vec<i32> {
        let last_pos = if n == 0 { 0 } else { ((n as i64 - 1) * inc as i64 >> 16) as usize };
        (0..last_pos + order)
            .map(|_| match trial % 3 {
                0 => rng.range(-(1 << 23), 1 << 23),
                1 => rng.range(-(1 << 30), 1 << 30),
                _ => rng.i32_edgy(),
            })
            .collect()
    }

    #[test]
    fn down_fir_interpolated_kernels_match_the_reference() {
        let mut rng = Lcg(24);
        for (fracs, coefs, taps) in [
            (3, &SILK_RESAMPLER_3_4_COEFS[..], &SILK_RESAMPLER_3_4_TAPS),
            (2, &SILK_RESAMPLER_2_3_COEFS[..], &SILK_RESAMPLER_2_3_TAPS),
        ] {
            for inc in increments(false) {
                for &n in &LENGTHS {
                    for trial in 0..6 {
                        let buf = down_fir_buf(&mut rng, n, inc, RESAMPLER_DOWN_ORDER_FIR0, trial);
                        let mut expect = vec![0i16; n];
                        down_fir_reference(&mut expect, &buf, inc, RESAMPLER_DOWN_ORDER_FIR0, fracs, coefs);

                        let mut out = vec![0x5555i16; n];
                        down_fir_interpolated_scalar(&mut out, &buf, 0, inc, fracs, taps);
                        assert_eq!(out, expect, "scalar: fracs={fracs} inc={inc} n={n} trial={trial}");
                        #[cfg(target_arch = "aarch64")]
                        {
                            let mut out = vec![0x5555i16; n];
                            down_fir_interpolated_neon(&mut out, &buf, inc, fracs, taps);
                            assert_eq!(out, expect, "NEON: fracs={fracs} inc={inc} n={n} trial={trial}");
                        }
                        let mut out = vec![0x5555i16; n];
                        down_fir_interpolated(&mut out, &buf, inc, fracs, taps);
                        assert_eq!(out, expect, "dispatch: fracs={fracs} inc={inc} n={n} trial={trial}");
                    }
                }
            }
        }
    }

    fn check_down_fir_symmetric<const HALF: usize>(rng: &mut Lcg, coefs: &[i16]) {
        let half: &[i16; HALF] = coefs[2..].try_into().unwrap();
        for inc in increments(false) {
            for &n in &LENGTHS {
                for trial in 0..6 {
                    let buf = down_fir_buf(rng, n, inc, 2 * HALF, trial);
                    let mut expect = vec![0i16; n];
                    down_fir_reference(&mut expect, &buf, inc, 2 * HALF, 1, coefs);

                    let mut out = vec![0x5555i16; n];
                    down_fir_symmetric_scalar(&mut out, &buf, 0, inc, half);
                    assert_eq!(out, expect, "scalar: taps={} inc={inc} n={n} trial={trial}", 2 * HALF);
                    #[cfg(target_arch = "aarch64")]
                    {
                        let mut out = vec![0x5555i16; n];
                        down_fir_symmetric_neon(&mut out, &buf, inc, half);
                        assert_eq!(out, expect, "NEON: taps={} inc={inc} n={n} trial={trial}", 2 * HALF);
                    }
                    let mut out = vec![0x5555i16; n];
                    down_fir_symmetric(&mut out, &buf, inc, half);
                    assert_eq!(out, expect, "dispatch: taps={} inc={inc} n={n} trial={trial}", 2 * HALF);
                }
            }
        }
    }

    #[test]
    fn down_fir_symmetric_kernels_match_the_reference() {
        let mut rng = Lcg(25);
        check_down_fir_symmetric::<12>(&mut rng, &SILK_RESAMPLER_1_2_COEFS);
        check_down_fir_symmetric::<18>(&mut rng, &SILK_RESAMPLER_1_3_COEFS);
        check_down_fir_symmetric::<18>(&mut rng, &SILK_RESAMPLER_1_4_COEFS);
        check_down_fir_symmetric::<18>(&mut rng, &SILK_RESAMPLER_1_6_COEFS);
    }

    /// The vector kernels compute their read positions without an index
    /// check per sample. A buffer that is one sample too short for the last
    /// output must therefore make them panic (in the check before the loop,
    /// or in the scalar kernel they hand over to), not read past the buffer.
    #[test]
    fn interpolation_kernels_panic_on_a_buffer_that_is_too_short() {
        fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
            std::panic::catch_unwind(f).is_err()
        }
        let half_1_2: &[i16; 12] = SILK_RESAMPLER_1_2_COEFS[2..].try_into().unwrap();
        let half_1_3: &[i16; 18] = SILK_RESAMPLER_1_3_COEFS[2..].try_into().unwrap();
        for &n in &[1usize, 3, 4, 5, 8, 9, 12, 80, 160] {
            for missing in [1usize, 2, 19] {
                for inc in increments(false) {
                    // The buffer the last of `n` samples needs, minus some.
                    let short = |window: usize| -> Vec<i32> {
                        let last_pos = ((n as i64 - 1) * inc as i64 >> 16) as usize;
                        vec![1i32; (last_pos + window).saturating_sub(missing)]
                    };
                    let what = format!("n={n} missing={missing} inc={inc}");

                    let buf = short(RESAMPLER_DOWN_ORDER_FIR0);
                    assert!(
                        panics(|| {
                            let mut out = vec![0i16; n];
                            down_fir_interpolated(&mut out, &buf, inc, 3, &SILK_RESAMPLER_3_4_TAPS);
                        }),
                        "interpolated: {what}"
                    );
                    let buf = short(24);
                    assert!(
                        panics(|| {
                            let mut out = vec![0i16; n];
                            down_fir_symmetric(&mut out, &buf, inc, half_1_2);
                        }),
                        "symmetric 24: {what}"
                    );
                    let buf = short(36);
                    assert!(
                        panics(|| {
                            let mut out = vec![0i16; n];
                            down_fir_symmetric(&mut out, &buf, inc, half_1_3);
                        }),
                        "symmetric 36: {what}"
                    );
                }
                for inc in increments(true) {
                    let last_pos = ((n as i64 - 1) * inc as i64 >> 16) as usize;
                    let len = (last_pos + RESAMPLER_ORDER_FIR_12).saturating_sub(missing);
                    let buf = vec![1i16; len];
                    assert!(
                        panics(|| {
                            let mut out = vec![0i16; n];
                            #[cfg(target_arch = "aarch64")]
                            iir_fir_interpol_neon(&mut out, &buf, 0, inc);
                            #[cfg(not(target_arch = "aarch64"))]
                            iir_fir_interpol_scalar(&mut out, &buf, 0, inc);
                        }),
                        "up-sampler: n={n} missing={missing} inc={inc}"
                    );
                }
            }
        }
        // An index increment that is zero or negative never reaches the
        // vector loop: the position would not grow.
        for inc in [0, -1, -65536, i32::MIN] {
            let buf = vec![1i32; 64];
            let mut out = vec![0x5555i16; 8];
            let result = std::panic::catch_unwind(move || {
                down_fir_symmetric(&mut out, &buf, inc, half_1_2);
                out
            });
            // Increment 0 reads position 0 for every sample; a negative one
            // leaves the buffer at the second sample and panics.
            assert_eq!(result.is_ok(), inc == 0, "inc={inc}");
        }
    }

    /// The tap tables of the interpolating down-FIR against the libopus
    /// coefficient tables they are built from.
    #[test]
    fn down_fir_tap_tables_match_the_libopus_tables() {
        for (fracs, coefs, table) in [
            (3usize, &SILK_RESAMPLER_3_4_COEFS[..], &SILK_RESAMPLER_3_4_TAPS),
            (2, &SILK_RESAMPLER_2_3_COEFS[..], &SILK_RESAMPLER_2_3_TAPS),
        ] {
            for i in 0..fracs {
                for k in 0..9 {
                    assert_eq!(table.taps[i][k], coefs[2 + 9 * i + k]);
                    assert_eq!(table.taps[i][17 - k], coefs[2 + 9 * (fracs - 1 - i) + k]);
                }
                for k in 0..16 {
                    assert_eq!(table.wide[i][k], (table.taps[i][k] as i32) << 15);
                }
                assert_eq!(table.wide[i][16..18], [0, 0]);
                assert_eq!(table.wide[i][18], (table.taps[i][16] as i32) << 15);
                assert_eq!(table.wide[i][19], (table.taps[i][17] as i32) << 15);
            }
        }
    }

    // -- the down-samplers of the pitch analysis ---------------------------------

    /// `silk_resampler_down2` as libopus writes it.
    fn down2_reference(s: &mut [i32; 2], out: &mut [i16], input: &[i16]) {
        for k in 0..input.len() / 2 {
            let in32 = (input[2 * k] as i32) << 10;
            let y = in32.wrapping_sub(s[0]);
            let x = silk_smlawb(y, y, SILK_RESAMPLER_DOWN2_1);
            let mut out32 = s[0].wrapping_add(x);
            s[0] = in32.wrapping_add(x);

            let in32 = (input[2 * k + 1] as i32) << 10;
            let y = in32.wrapping_sub(s[1]);
            let x = silk_smulwb(y, SILK_RESAMPLER_DOWN2_0);
            out32 = out32.wrapping_add(s[1]);
            out32 = out32.wrapping_add(x);
            s[1] = in32.wrapping_add(x);

            out[k] = silk_sat16(silk_rshift_round(out32, 11)) as i16;
        }
    }

    #[test]
    fn down2_matches_the_reference() {
        let mut rng = Lcg(26);
        for &n in &LENGTHS {
            for trial in 0..16 {
                let input = rng.signal(n, trial);
                let state: [i32; 2] = rng.states(trial / 2);
                let (mut s_ref, mut s_new) = (state, state);
                let mut expect = vec![0i16; n / 2];
                down2_reference(&mut s_ref, &mut expect, &input);
                let mut out = vec![0x5555i16; n / 2 + 1];
                silk_resampler_down2(&mut s_new, &mut out, &input, n as i32);
                assert_eq!((&out[..n / 2], s_new), (&expect[..], s_ref), "n={n} trial={trial}");
                assert_eq!(out[n / 2], 0x5555);
            }
        }
    }

    /// `silk_resampler_private_AR2` and `silk_resampler_down2_3` as libopus
    /// writes them, with the states in the slice and a cleared buffer.
    fn ar2_reference(s: &mut [i32], out_q8: &mut [i32], input: &[i16], a_q14: &[i16]) {
        ar2_loop_reference(s, out_q8, input, a_q14[0] as i32, a_q14[1] as i32);
    }

    fn down2_3_reference(s: &mut [i32; 6], out: &mut Vec<i16>, input: &[i16]) {
        let c = &SILK_RESAMPLER_2_3_COEFS_LQ;
        let mut buf = [0i32; RESAMPLER_MAX_BATCH_SIZE_IN as usize + ORDER_FIR];
        buf[..ORDER_FIR].copy_from_slice(&s[..ORDER_FIR]);
        let mut in_idx = 0;
        let mut remaining = input.len();
        while remaining > 0 {
            let n = remaining.min(RESAMPLER_MAX_BATCH_SIZE_IN as usize);
            ar2_reference(&mut s[ORDER_FIR..], &mut buf[ORDER_FIR..ORDER_FIR + n], &input[in_idx..in_idx + n], c);
            let mut p = 0;
            let mut counter = n as i32;
            while counter > 2 {
                let mut r = silk_smulwb(buf[p], c[2] as i32);
                r = silk_smlawb(r, buf[p + 1], c[3] as i32);
                r = silk_smlawb(r, buf[p + 2], c[5] as i32);
                r = silk_smlawb(r, buf[p + 3], c[4] as i32);
                out.push(silk_sat16(silk_rshift_round(r, 6)) as i16);
                let mut r = silk_smulwb(buf[p + 1], c[4] as i32);
                r = silk_smlawb(r, buf[p + 2], c[5] as i32);
                r = silk_smlawb(r, buf[p + 3], c[3] as i32);
                r = silk_smlawb(r, buf[p + 4], c[2] as i32);
                out.push(silk_sat16(silk_rshift_round(r, 6)) as i16);
                p += 3;
                counter -= 3;
            }
            in_idx += n;
            remaining -= n;
            if remaining > 0 {
                buf.copy_within(n..n + ORDER_FIR, 0);
            } else {
                s[..ORDER_FIR].copy_from_slice(&buf[n..n + ORDER_FIR]);
            }
        }
    }

    #[test]
    fn down2_3_and_ar2_match_the_reference() {
        let mut rng = Lcg(27);
        // Up to more than one batch of 480 samples.
        for n in LENGTHS.into_iter().chain([240, 479, 480, 481, 483, 960, 1000]) {
            for trial in 0..8 {
                let input = rng.signal(n, trial);
                let state: [i32; 6] = rng.states(trial / 2);

                let mut s_ref = state;
                let mut expect = Vec::new();
                down2_3_reference(&mut s_ref, &mut expect, &input);
                let mut s_new = state;
                let mut out = vec![0x5555i16; expect.len() + 1];
                silk_resampler_down2_3(&mut s_new, &mut out, &input, n as i32);
                assert_eq!((&out[..expect.len()], s_new), (&expect[..], s_ref), "n={n} trial={trial}");
                assert_eq!(out[expect.len()], 0x5555);

                let (mut s_ref, mut s_new) = ([state[0], state[1]], [state[0], state[1]]);
                let mut expect = vec![0i32; n];
                ar2_reference(&mut s_ref, &mut expect, &input, &SILK_RESAMPLER_2_3_COEFS_LQ);
                let mut out_q8 = vec![0x5555_5555i32; n];
                silk_resampler_private_ar2(&mut s_new, &mut out_q8, &input, &SILK_RESAMPLER_2_3_COEFS_LQ, n as i32);
                assert_eq!((&out_q8, s_new), (&expect, s_ref), "AR2: n={n} trial={trial}");
            }
        }
    }
}
