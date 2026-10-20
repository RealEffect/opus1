//! The input features of LACE, NoLACE and BBWENet, and the two 10 ms
//! cross-fades (libopus `dnn/osce_features.c`).
//!
//! LACE and NoLACE read 93 values per 5 ms subframe, computed from what the
//! SILK decoder decoded (`osce_calculate_features`):
//!
//! | Values | What |
//! |---|---|
//! | 0-63 | the log spectrum of the LPC synthesis filter, 64 bands |
//! | 64-81 | the cepstrum of the decoded signal, 18 bands |
//! | 82-86 | the normalized correlation at the pitch lag - 2 to + 2 |
//! | 87-91 | the 5 LTP coefficients |
//! | 92 | the log gain |
//!
//! BBWENet reads 114 values per 10 ms (`osce_bwe_calculate_features`): 32
//! log-magnitude bands and the cosine and sine of the phase advance of the
//! first 41 frequency bins.
//!
//! The roundings follow the disassembly of the reference build (libopus
//! 1.6.1 on 64-bit ARM, `-O2 -fno-vectorize -fno-slp-vectorize`): every
//! accumulation, including those of the filter banks and the correlations,
//! is a chain of fused multiply-adds, and the `double` steps of the C code
//! are done in `f64`.

// The tables keep the digits of the C source.
#![allow(clippy::excessive_precision)]

use super::OsceFrame;
use super::math::{fma, ln, sqrt32, sqrt64};
use crate::dnn::freq::{FREQ_SIZE, NB_BANDS, WINDOW_SIZE, dct, forward_transform};
use crate::kiss_fft::KissCpx;

/// `OSCE_FEATURES_MAX_HISTORY`: past samples kept for the cepstrum and the
/// correlation.
pub const OSCE_FEATURES_MAX_HISTORY: usize = 350;
/// `OSCE_FEATURE_DIM`: features per subframe.
pub const OSCE_FEATURE_DIM: usize = 93;
/// `OSCE_NO_PITCH_VALUE`: the pitch lag given to the models in subframes
/// that are not voiced.
pub const OSCE_NO_PITCH_VALUE: i32 = 7;
/// `OSCE_CLEAN_SPEC_NUM_BANDS`.
const OSCE_CLEAN_SPEC_NUM_BANDS: usize = 64;
/// `OSCE_NOISY_SPEC_NUM_BANDS`.
const OSCE_NOISY_SPEC_NUM_BANDS: usize = 18;
/// `OSCE_CLEAN_SPEC_START`.
const OSCE_CLEAN_SPEC_START: usize = 0;
/// `OSCE_NOISY_CEPSTRUM_START`.
const OSCE_NOISY_CEPSTRUM_START: usize = 64;
/// `OSCE_ACORR_START`.
const OSCE_ACORR_START: usize = 82;
/// `OSCE_LTP_START`.
const OSCE_LTP_START: usize = 87;
/// `OSCE_LTP_LENGTH`, equal to SILK's `LTP_ORDER`.
const OSCE_LTP_LENGTH: usize = 5;
/// `OSCE_LOG_GAIN_START`.
const OSCE_LOG_GAIN_START: usize = 92;
/// `OSCE_SPEC_WINDOW_SIZE`.
const OSCE_SPEC_WINDOW_SIZE: usize = 320;
/// `OSCE_SPEC_NUM_FREQS`.
const OSCE_SPEC_NUM_FREQS: usize = 161;
/// SILK's `TYPE_VOICED`.
const TYPE_VOICED: i32 = 2;

/// `OSCE_BWE_MAX_INSTAFREQ_BIN`: the last bin of the phase features.
pub const OSCE_BWE_MAX_INSTAFREQ_BIN: usize = 40;
/// `OSCE_BWE_HALF_WINDOW_SIZE`: samples per 10 ms frame at 16 kHz.
pub const OSCE_BWE_HALF_WINDOW_SIZE: usize = 160;
/// `OSCE_BWE_NUM_BANDS`.
pub const OSCE_BWE_NUM_BANDS: usize = 32;
/// `OSCE_BWE_FEATURE_DIM`: features per 10 ms frame.
pub const OSCE_BWE_FEATURE_DIM: usize = 114;

// The asserts of libopus on these sizes.
const _: () = assert!(OSCE_SPEC_WINDOW_SIZE == WINDOW_SIZE && OSCE_SPEC_NUM_FREQS == FREQ_SIZE);
const _: () = assert!(OSCE_NOISY_SPEC_NUM_BANDS == NB_BANDS);

/// The feature state of one SILK channel (libopus `OSCEFeatureState`).
#[derive(Clone)]
pub struct OsceFeatureState {
    /// The smoothed bit count: `0.9·smooth + 0.1·bits` per frame.
    pub numbits_smooth: f32,
    /// The pitch hangover counter. libopus 1.6.1 disables the hangover
    /// (`OSCE_HANGOVER_BUGFIX` is not defined), so it stays 0.
    pub pitch_hangover_count: i32,
    /// The pitch lag of the last voiced subframe.
    pub last_lag: i32,
    /// The signal type of the last subframe.
    pub last_type: i32,
    /// The last 350 samples, divided by 32768.
    pub signal_history: [f32; OSCE_FEATURES_MAX_HISTORY],
    /// The bypass counter of `osce_enhance_frame`: 2 after a reset (the
    /// next frame comes out unenhanced), 1 (the next frame is cross-faded
    /// from unenhanced to enhanced), then 0.
    pub reset: i32,
}

impl Default for OsceFeatureState {
    /// The cleared state (`OPUS_CLEAR`).
    fn default() -> Self {
        OsceFeatureState {
            numbits_smooth: 0.0,
            pitch_hangover_count: 0,
            last_lag: 0,
            last_type: 0,
            signal_history: [0.0; OSCE_FEATURES_MAX_HISTORY],
            reset: 0,
        }
    }
}

/// The feature state of the bandwidth extension of one SILK channel
/// (libopus `OSCEBWEFeatureState`).
#[derive(Clone)]
pub struct OsceBweFeatureState {
    /// The last 160 samples, divided by 32768.
    pub signal_history: [f32; OSCE_BWE_HALF_WINDOW_SIZE],
    /// The first 41 bins of the previous spectrum, real and imaginary
    /// parts interleaved.
    pub last_spec: [f32; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
}

impl Default for OsceBweFeatureState {
    /// The cleared state, as `silk_init_decoder` leaves it (not the state
    /// of `osce_bwe_reset`, which sets the real parts of `last_spec` to
    /// 1e-9).
    fn default() -> Self {
        OsceBweFeatureState {
            signal_history: [0.0; OSCE_BWE_HALF_WINDOW_SIZE],
            last_spec: [0.0; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
        }
    }
}

/// `center_bins_clean`: the centre bins of the 64 bands of the LPC spectrum.
const CENTER_BINS_CLEAN: [usize; 64] = [
    0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38,
    40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 73, 75, 78,
    80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115, 118,
    120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 160,
];
/// `center_bins_noisy`: the centre bins of the 18 bands of the cepstrum.
const CENTER_BINS_NOISY: [usize; 18] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160,
];
/// `center_bins_bwe`: the centre bins of the 32 bands of the BWE features.
const CENTER_BINS_BWE: [usize; 32] = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
    80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 160,
];
/// `band_weights_clean`.
const BAND_WEIGHTS_CLEAN: [f32; 64] = [
    0.666666666667, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.400000000000, 0.400000000000, 0.400000000000, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.400000000000, 0.400000000000, 0.400000000000, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.250000000000, 0.333333333333,
];
/// `band_weights_noisy`.
const BAND_WEIGHTS_NOISY: [f32; 18] = [
    0.400000000000, 0.250000000000, 0.250000000000, 0.250000000000,
    0.250000000000, 0.250000000000, 0.250000000000, 0.250000000000,
    0.166666666667, 0.125000000000, 0.125000000000, 0.125000000000,
    0.083333333333, 0.062500000000, 0.062500000000, 0.050000000000,
    0.041666666667, 0.080000000000,
];
/// `band_weights_bwe`: written with `double` literals in C, so each value is
/// rounded to `double` first, then to `float`.
const BAND_WEIGHTS_BWE: [f32; 32] = [
    0.333333333_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.200000000_f64 as f32,
    0.200000000_f64 as f32, 0.200000000_f64 as f32, 0.133333333_f64 as f32, 0.181818182_f64 as f32,
];
/// `osce_window`: a sine window of 320 samples, for the spectra and the
/// cross-fades.
pub const OSCE_WINDOW: [f32; 320] = [
    0.004908718808, 0.014725683311, 0.024541228523, 0.034354408400, 0.044164277127,
    0.053969889210, 0.063770299562, 0.073564563600, 0.083351737332, 0.093130877450,
    0.102901041421, 0.112661287575, 0.122410675199, 0.132148264628, 0.141873117332,
    0.151584296010, 0.161280864678, 0.170961888760, 0.180626435180, 0.190273572448,
    0.199902370753, 0.209511902052, 0.219101240157, 0.228669460829, 0.238215641862,
    0.247738863176, 0.257238206902, 0.266712757475, 0.276161601717, 0.285583828929,
    0.294978530977, 0.304344802381, 0.313681740399, 0.322988445118, 0.332264019538,
    0.341507569661, 0.350718204573, 0.359895036535, 0.369037181064, 0.378143757022,
    0.387213886697, 0.396246695891, 0.405241314005, 0.414196874117, 0.423112513073,
    0.431987371563, 0.440820594212, 0.449611329655, 0.458358730621, 0.467061954019,
    0.475720161014, 0.484332517110, 0.492898192230, 0.501416360796, 0.509886201809,
    0.518306898929, 0.526677640552, 0.534997619887, 0.543266035038, 0.551482089078,
    0.559644990127, 0.567753951426, 0.575808191418, 0.583806933818, 0.591749407690,
    0.599634847523, 0.607462493302, 0.615231590581, 0.622941390558, 0.630591150148,
    0.638180132051, 0.645707604824, 0.653172842954, 0.660575126926, 0.667913743292,
    0.675187984742, 0.682397150168, 0.689540544737, 0.696617479953, 0.703627273726,
    0.710569250438, 0.717442741007, 0.724247082951, 0.730981620454, 0.737645704427,
    0.744238692572, 0.750759949443, 0.757208846506, 0.763584762206, 0.769887082016,
    0.776115198508, 0.782268511401, 0.788346427627, 0.794348361383, 0.800273734191,
    0.806121974951, 0.811892519997, 0.817584813152, 0.823198305781, 0.828732456844,
    0.834186732948, 0.839560608398, 0.844853565250, 0.850065093356, 0.855194690420,
    0.860241862039, 0.865206121757, 0.870086991109, 0.874883999665, 0.879596685080,
    0.884224593137, 0.888767277786, 0.893224301196, 0.897595233788, 0.901879654283,
    0.906077149740, 0.910187315596, 0.914209755704, 0.918144082372, 0.921989916403,
    0.925746887127, 0.929414632439, 0.932992798835, 0.936481041442, 0.939879024058,
    0.943186419177, 0.946402908026, 0.949528180593, 0.952561935658, 0.955503880820,
    0.958353732530, 0.961111216112, 0.963776065795, 0.966348024735, 0.968826845041,
    0.971212287799, 0.973504123096, 0.975702130039, 0.977806096779, 0.979815820533,
    0.981731107599, 0.983551773378, 0.985277642389, 0.986908548290, 0.988444333892,
    0.989884851171, 0.991229961288, 0.992479534599, 0.993633450666, 0.994691598273,
    0.995653875433, 0.996520189401, 0.997290456679, 0.997964603026, 0.998542563469,
    0.999024282300, 0.999409713092, 0.999698818696, 0.999891571247, 0.999987952167,
    0.999987952167, 0.999891571247, 0.999698818696, 0.999409713092, 0.999024282300,
    0.998542563469, 0.997964603026, 0.997290456679, 0.996520189401, 0.995653875433,
    0.994691598273, 0.993633450666, 0.992479534599, 0.991229961288, 0.989884851171,
    0.988444333892, 0.986908548290, 0.985277642389, 0.983551773378, 0.981731107599,
    0.979815820533, 0.977806096779, 0.975702130039, 0.973504123096, 0.971212287799,
    0.968826845041, 0.966348024735, 0.963776065795, 0.961111216112, 0.958353732530,
    0.955503880820, 0.952561935658, 0.949528180593, 0.946402908026, 0.943186419177,
    0.939879024058, 0.936481041442, 0.932992798835, 0.929414632439, 0.925746887127,
    0.921989916403, 0.918144082372, 0.914209755704, 0.910187315596, 0.906077149740,
    0.901879654283, 0.897595233788, 0.893224301196, 0.888767277786, 0.884224593137,
    0.879596685080, 0.874883999665, 0.870086991109, 0.865206121757, 0.860241862039,
    0.855194690420, 0.850065093356, 0.844853565250, 0.839560608398, 0.834186732948,
    0.828732456844, 0.823198305781, 0.817584813152, 0.811892519997, 0.806121974951,
    0.800273734191, 0.794348361383, 0.788346427627, 0.782268511401, 0.776115198508,
    0.769887082016, 0.763584762206, 0.757208846506, 0.750759949443, 0.744238692572,
    0.737645704427, 0.730981620454, 0.724247082951, 0.717442741007, 0.710569250438,
    0.703627273726, 0.696617479953, 0.689540544737, 0.682397150168, 0.675187984742,
    0.667913743292, 0.660575126926, 0.653172842954, 0.645707604824, 0.638180132051,
    0.630591150148, 0.622941390558, 0.615231590581, 0.607462493302, 0.599634847523,
    0.591749407690, 0.583806933818, 0.575808191418, 0.567753951426, 0.559644990127,
    0.551482089078, 0.543266035038, 0.534997619887, 0.526677640552, 0.518306898929,
    0.509886201809, 0.501416360796, 0.492898192230, 0.484332517110, 0.475720161014,
    0.467061954019, 0.458358730621, 0.449611329655, 0.440820594212, 0.431987371563,
    0.423112513073, 0.414196874117, 0.405241314005, 0.396246695891, 0.387213886697,
    0.378143757022, 0.369037181064, 0.359895036535, 0.350718204573, 0.341507569661,
    0.332264019538, 0.322988445118, 0.313681740399, 0.304344802381, 0.294978530977,
    0.285583828929, 0.276161601717, 0.266712757475, 0.257238206902, 0.247738863176,
    0.238215641862, 0.228669460829, 0.219101240157, 0.209511902052, 0.199902370753,
    0.190273572448, 0.180626435180, 0.170961888760, 0.161280864678, 0.151584296010,
    0.141873117332, 0.132148264628, 0.122410675199, 0.112661287575, 0.102901041421,
    0.093130877450, 0.083351737332, 0.073564563600, 0.063770299562, 0.053969889210,
    0.044164277127, 0.034354408400, 0.024541228523, 0.014725683311, 0.004908718808,
];

/// libopus `apply_filterbank` (`dnn/osce_features.c:183-209`): triangular
/// bands between the centre bins, each band scaled by its weight. Each
/// output is a chain of fused multiply-adds:
/// `out[b] = fma(w[b]·frac, x, out[b])`,
/// `out[b+1] = fma(w[b+1]·(1 - frac), x, out[b+1])`.
fn apply_filterbank(x_out: &mut [f32], x_in: &[f32], center_bins: &[usize], band_weights: &[f32]) {
    let num_bands = center_bins.len();
    x_out[0] = 0.0;
    for b in 0..num_bands - 1 {
        x_out[b + 1] = 0.0;
        let (lo, hi) = (center_bins[b], center_bins[b + 1]);
        let width = (hi - lo) as f32;
        for (i, &x) in x_in.iter().enumerate().take(hi).skip(lo) {
            let frac = (hi - i) as f32 / width;
            x_out[b] = fma(band_weights[b] * frac, x, x_out[b]);
            x_out[b + 1] = fma(band_weights[b + 1] * (1.0 - frac), x, x_out[b + 1]);
        }
    }
    let last = num_bands - 1;
    x_out[last] = fma(band_weights[last], x_in[center_bins[last]], x_out[last]);
}

/// libopus `mag_spec_320_onesided` (`dnn/osce_features.c:212-226`): the 161
/// magnitudes of the 320-point spectrum, `320·sqrt(re² + im²)` with the
/// square root and the product in double precision.
fn mag_spec_320_onesided(out: &mut [f32; OSCE_SPEC_NUM_FREQS], input: &[f32; OSCE_SPEC_WINDOW_SIZE]) {
    let mut spec = [KissCpx::default(); FREQ_SIZE];
    forward_transform(&mut spec, input);
    for (o, c) in out.iter_mut().zip(spec.iter()) {
        *o = (OSCE_SPEC_WINDOW_SIZE as f64 * sqrt64(fma(c.r, c.r, c.i * c.i) as f64)) as f32;
    }
}

/// libopus `calculate_log_spectrum_from_lpc` (`dnn/osce_features.c:229-257`):
/// the log of the inverse magnitude of the LPC filter `1 - Σ a_k z^-k`, in
/// the 64 bands of `CENTER_BINS_CLEAN`, times 0.3.
fn calculate_log_spectrum_from_lpc(spec: &mut [f32], a_q12: &[i16], lpc_order: usize) {
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    buffer[0] = 1.0;
    for (b, &a) in buffer[1..=lpc_order].iter_mut().zip(a_q12) {
        *b = -(a as f32) / 4096.0;
    }
    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);
    for m in mag.iter_mut() {
        *m = 1.0 / (*m + 1e-9);
    }
    apply_filterbank(spec, &mag, &CENTER_BINS_CLEAN, &BAND_WEIGHTS_CLEAN);
    for s in spec[..OSCE_CLEAN_SPEC_NUM_BANDS].iter_mut() {
        // `0.3f * log(s + 1e-9f)`: the sum in float, the rest in double.
        *s = (0.3f32 as f64 * ln((*s + 1e-9) as f64)) as f32;
    }
}

/// libopus `calculate_cepstrum` (`dnn/osce_features.c:259-290`): windowed
/// spectrum of 320 samples, 18 bands, log, then the DCT of `freq.c`.
fn calculate_cepstrum(cepstrum: &mut [f32], signal: &[f32]) {
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    for ((b, &w), &x) in buffer.iter_mut().zip(OSCE_WINDOW.iter()).zip(signal) {
        *b = w * x;
    }
    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);
    let mut spec = [0.0f32; NB_BANDS];
    apply_filterbank(&mut spec, &mag, &CENTER_BINS_NOISY, &BAND_WEIGHTS_NOISY);
    for s in spec.iter_mut() {
        *s = ln((*s + 1e-9) as f64) as f32;
    }
    let mut out = [0.0f32; NB_BANDS];
    dct(&mut out, &spec);
    cepstrum[..NB_BANDS].copy_from_slice(&out);
}

/// libopus `calculate_acorr` (`dnn/osce_features.c:292-312`): the
/// normalized correlation of the 80 samples at `start` with the samples
/// `lag - 2` to `lag + 2` earlier, `xy / sqrt(xx·yy + 1e-9)`.
fn calculate_acorr(acorr: &mut [f32], signal: &[f32], start: usize, lag: usize) {
    for (k, a) in acorr[..5].iter_mut().enumerate() {
        let off = start + k - 2 - lag;
        let (mut xx, mut xy, mut yy) = (0.0f32, 0.0f32, 0.0f32);
        for n in 0..80 {
            let x = signal[start + n];
            let y = signal[off + n];
            xx = fma(x, x, xx);
            yy = fma(y, y, yy);
            xy = fma(x, y, xy);
        }
        *a = (xy as f64 / sqrt64(fma(xx, yy, 1e-9) as f64)) as f32;
    }
}

/// libopus `pitch_postprocessing` (`dnn/osce_features.c:314-366`) as 1.6.1
/// compiles it: the hangover is disabled, so a voiced subframe gives its
/// lag and any other subframe `OSCE_NO_PITCH_VALUE`.
fn pitch_postprocessing(st: &mut OsceFeatureState, lag: i32, signal_type: i32) -> i32 {
    let new_lag = if signal_type != TYPE_VOICED {
        st.pitch_hangover_count = 0;
        OSCE_NO_PITCH_VALUE
    } else {
        st.last_lag = lag;
        st.pitch_hangover_count = 0;
        lag
    };
    st.last_type = signal_type;
    new_lag
}

/// libopus `osce_calculate_features` (`dnn/osce_features.c:368-458`) for one
/// 20 ms frame at 16 kHz (`frame.nb_subfr == 4`, `xq` of 320 samples).
///
/// Writes 4 × 93 features, the bit count and the smoothed bit count, and
/// the pitch lag of each subframe, and updates the state. The spectrum and
/// the cepstrum are computed for subframes 0 and 2 and copied to 1 and 3.
pub fn osce_calculate_features(
    st: &mut OsceFeatureState,
    frame: &OsceFrame<'_>,
    xq: &[i16],
    features: &mut [f32; 4 * OSCE_FEATURE_DIM],
    numbits: &mut [f32; 2],
    periods: &mut [i32; 4],
) {
    const NUM_SUBFRAMES: usize = 4;
    const NUM_SAMPLES: usize = NUM_SUBFRAMES * 80;
    let xq = &xq[..NUM_SAMPLES];
    let lpc_order = frame.lpc_order as usize;
    assert!(lpc_order <= 16, "LPC order above 16");

    st.numbits_smooth = fma(0.9, st.numbits_smooth, 0.1 * frame.num_bits as f32);
    numbits[0] = frame.num_bits as f32;
    numbits[1] = st.numbits_smooth;

    let mut buffer = [0.0f32; OSCE_FEATURES_MAX_HISTORY + NUM_SAMPLES];
    buffer[..OSCE_FEATURES_MAX_HISTORY].copy_from_slice(&st.signal_history);
    for (b, &x) in buffer[OSCE_FEATURES_MAX_HISTORY..].iter_mut().zip(xq) {
        *b = x as f32 / 32768.0;
    }

    for k in 0..NUM_SUBFRAMES {
        let start = OSCE_FEATURES_MAX_HISTORY + k * 80;
        let (done, rest) = features.split_at_mut(k * OSCE_FEATURE_DIM);
        let pf = &mut rest[..OSCE_FEATURE_DIM];

        // The clean spectrum from the LPC coefficients, and the noisy
        // cepstrum from the signal, every other subframe.
        if k % 2 == 0 {
            calculate_log_spectrum_from_lpc(
                &mut pf[OSCE_CLEAN_SPEC_START..],
                &frame.pred_coef_q12[k >> 1][..lpc_order],
                lpc_order,
            );
            calculate_cepstrum(&mut pf[OSCE_NOISY_CEPSTRUM_START..], &buffer[start - 160..start + 160]);
        } else {
            let prev = &done[(k - 1) * OSCE_FEATURE_DIM..];
            pf[OSCE_CLEAN_SPEC_START..OSCE_CLEAN_SPEC_START + OSCE_CLEAN_SPEC_NUM_BANDS]
                .copy_from_slice(&prev[OSCE_CLEAN_SPEC_START..OSCE_CLEAN_SPEC_START + OSCE_CLEAN_SPEC_NUM_BANDS]);
            pf[OSCE_NOISY_CEPSTRUM_START..OSCE_NOISY_CEPSTRUM_START + NB_BANDS]
                .copy_from_slice(&prev[OSCE_NOISY_CEPSTRUM_START..OSCE_NOISY_CEPSTRUM_START + NB_BANDS]);
        }

        periods[k] = pitch_postprocessing(st, frame.pitch_l[k], frame.signal_type);

        // The correlation around the pitch lag. A voiced lag of the SILK
        // decoder lies in 32..=288 at 16 kHz.
        let lag = periods[k] as usize;
        assert!(lag + 2 <= start, "pitch lag too long for the feature history");
        calculate_acorr(&mut pf[OSCE_ACORR_START..], &buffer, start, lag);

        for i in 0..OSCE_LTP_LENGTH {
            pf[OSCE_LTP_START + i] = frame.ltp_coef_q14[k * OSCE_LTP_LENGTH + i] as f32 / 16384.0;
        }

        pf[OSCE_LOG_GAIN_START] = ln((frame.gains_q16[k] as f32 / 65536.0 + 1e-9) as f64) as f32;
    }

    st.signal_history.copy_from_slice(&buffer[NUM_SAMPLES..]);
}

/// libopus `osce_bwe_calculate_features` (`dnn/osce_features.c:462-542`):
/// 114 features for each 10 ms of `xq` (160 or 320 samples at 16 kHz).
///
/// For each 10 ms: the windowed spectrum of the last 320 samples; 32
/// log-magnitude bands (`log(band + 1e-9)` in double precision); and for
/// the bins 0 to 40 the cosine and the sine of the phase advance against
/// the previous spectrum, `re(X·conj(X'))/|X·conj(X')|` and
/// `im(X·conj(X'))/|X·conj(X')|`.
pub fn osce_bwe_calculate_features(st: &mut OsceBweFeatureState, features: &mut [f32], xq: &[i16]) {
    let num_samples = xq.len();
    assert!(num_samples.is_multiple_of(OSCE_BWE_HALF_WINDOW_SIZE));
    let num_frames = num_samples / OSCE_BWE_HALF_WINDOW_SIZE;
    const NSPEC: usize = 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2;

    for frame in 0..num_frames {
        let feat = &mut features[frame * OSCE_BWE_FEATURE_DIM..(frame + 1) * OSCE_BWE_FEATURE_DIM];
        feat.fill(0.0);
        let (lmspec, instafreq) = feat.split_at_mut(OSCE_BWE_NUM_BANDS);
        let x = &xq[frame * OSCE_BWE_HALF_WINDOW_SIZE..(frame + 1) * OSCE_BWE_HALF_WINDOW_SIZE];

        let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
        buffer[..OSCE_BWE_HALF_WINDOW_SIZE].copy_from_slice(&st.signal_history);
        for (b, &v) in buffer[OSCE_BWE_HALF_WINDOW_SIZE..].iter_mut().zip(x) {
            *b = v as f32 / 32768.0;
        }
        st.signal_history.copy_from_slice(&buffer[OSCE_BWE_HALF_WINDOW_SIZE..]);
        for (b, &w) in buffer.iter_mut().zip(OSCE_WINDOW.iter()) {
            *b *= w;
        }

        let mut fft = [KissCpx::default(); FREQ_SIZE];
        forward_transform(&mut fft, &buffer);

        let mut spec = [0.0f32; NSPEC];
        for k in 0..=OSCE_BWE_MAX_INSTAFREQ_BIN {
            // `320 * re + 1e-9`: the product in float, the sum in double.
            spec[2 * k] = ((OSCE_SPEC_WINDOW_SIZE as f32 * fft[k].r) as f64 + 1e-9) as f32;
            spec[2 * k + 1] = OSCE_SPEC_WINDOW_SIZE as f32 * fft[k].i;
            let (re1, im1) = (spec[2 * k], spec[2 * k + 1]);
            let (re2, im2) = (st.last_spec[2 * k], st.last_spec[2 * k + 1]);
            let aux_r = fma(re1, re2, im1 * im2);
            let aux_i = fma(im1, re2, -(re1 * im2));
            // `(float)sqrt(double)` of a float is the float square root.
            let aux_abs = sqrt32(fma(aux_r, aux_r, aux_i * aux_i));
            instafreq[k] = (aux_r as f64 / (aux_abs as f64 + 1e-9)) as f32;
            instafreq[k + OSCE_BWE_MAX_INSTAFREQ_BIN + 1] = (aux_i as f64 / (aux_abs as f64 + 1e-9)) as f32;
        }

        let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
        for (m, c) in mag.iter_mut().zip(fft.iter()) {
            *m = (OSCE_SPEC_WINDOW_SIZE as f64 * sqrt64(fma(c.r, c.r, c.i * c.i) as f64)) as f32;
        }
        apply_filterbank(lmspec, &mag, &CENTER_BINS_BWE, &BAND_WEIGHTS_BWE);
        for l in lmspec.iter_mut() {
            // `log(l + 1e-9)`: the sum in double, unlike the other features.
            *l = ln(*l as f64 + 1e-9) as f32;
        }

        st.last_spec = spec;
    }
}

/// libopus `osce_cross_fade_10ms` (`dnn/osce_features.c:545-554`): the first
/// 160 samples go from `x_in` to `x_enhanced` along the rising half of
/// `OSCE_WINDOW`: `e = fma(w, e, (1 - w)·x)`.
pub fn osce_cross_fade_10ms(x_enhanced: &mut [f32], x_in: &[f32]) {
    for i in 0..160 {
        let w = OSCE_WINDOW[i];
        x_enhanced[i] = fma(w, x_enhanced[i], (1.0 - w) * x_in[i]);
    }
}

/// libopus `osce_bwe_cross_fade_10ms` (`dnn/osce_features.c:557-572`): the
/// first 480 samples (10 ms at 48 kHz) go from `x_fadeout` to `x_fadein`,
/// in place on `x_fadein`. The weight steps a third of the way to the next
/// window value for each of the 3 output samples of an input sample;
/// each sample is `(int)(fma(w, in, (1 - w)·out) + 0.5)` with the addition
/// in double precision and truncation toward zero.
pub fn osce_bwe_cross_fade_10ms(x_fadein: &mut [i16], x_fadeout: &[i16]) {
    let x_fadein = &mut x_fadein[..480];
    let x_fadeout = &x_fadeout[..480];
    let f = 1.0f32 / 3.0;
    for i in 0..160 {
        let diff = if i == 159 { 0.0 } else { OSCE_WINDOW[i + 1] - OSCE_WINDOW[i] };
        let mut w = OSCE_WINDOW[i];
        for j in 0..3 {
            let n = 3 * i + j;
            let v = fma(w, x_fadein[n] as f32, (1.0 - w) * x_fadeout[n] as f32);
            x_fadein[n] = ((v as f64 + 0.5) as i32) as i16;
            if j < 2 {
                w = fma(diff, f, w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_the_sine_window() {
        for (i, &w) in OSCE_WINDOW.iter().enumerate() {
            let want = (core::f64::consts::PI * (i as f64 + 0.5) / 320.0).sin();
            assert!((w as f64 - want).abs() < 1e-7, "{i}");
        }
    }

    /// `(int)(x + 0.5)` truncates toward zero: a cross-fade of a signal with
    /// itself keeps samples at or above zero and moves negative samples up
    /// by one.
    #[test]
    fn the_bwe_cross_fade_of_a_signal_with_itself_rounds_as_libopus() {
        let a: [i16; 480] = core::array::from_fn(|i| ((i as i32 * 7919) % 65536 - 32768) as i16);
        let mut b = a;
        osce_bwe_cross_fade_10ms(&mut b, &a);
        for (&x, &y) in a.iter().zip(b.iter()) {
            assert_eq!(y, if x >= 0 { x } else { x + 1 });
        }
    }

    #[test]
    fn the_cross_fade_starts_at_the_input_and_ends_at_the_enhanced_signal() {
        let x = [1.0f32; 320];
        let mut e = [0.0f32; 320];
        osce_cross_fade_10ms(&mut e, &x);
        assert!(e[0] > 0.99 && e[159] < 0.01);
        assert!(e[160..].iter().all(|&v| v == 0.0));
    }
}
