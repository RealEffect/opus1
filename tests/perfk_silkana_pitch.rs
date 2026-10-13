//! `silk_pitch_analysis_core` computes its divisions in groups and keeps the
//! values of the last search stage per subframe. These tests run the
//! function on generated signals and compare a checksum of all outputs with
//! the checksum of the version before that change.

use opus_rs::silk::pitch_analysis::silk_pitch_analysis_core;

/// Samples of history in front of the subframes, in milliseconds.
const HISTORY_MS: usize = 20;
/// Length of one subframe in milliseconds.
const SUBFRAME_MS: usize = 5;

/// xorshift64 generator with a fixed seed, so the test data never changes.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `low..=high`.
    fn range(&mut self, low: i32, high: i32) -> i32 {
        low + (self.next() % (high as i64 - low as i64 + 1) as u64) as i32
    }
}

/// A train of short pulses plus noise. The distance of the pulses starts at
/// `period` samples and changes by `drift_q8 / 256` samples per pulse.
fn pulse_train(
    rng: &mut XorShift,
    len: usize,
    period: usize,
    drift_q8: i32,
    amplitude: i32,
    noise: i32,
) -> Vec<i16> {
    let mut signal = vec![0i32; len];
    let mut position = rng.range(0, period as i32 - 1) as usize;
    let mut period_q8 = (period as i32) << 8;
    while position < len {
        for (offset, gain_q2) in [4, -2, 1].into_iter().enumerate() {
            if position + offset < len {
                signal[position + offset] += amplitude * gain_q2 / 4;
            }
        }
        period_q8 += drift_q8;
        position += (period_q8 >> 8).max(2) as usize;
    }
    signal
        .into_iter()
        .map(|sample| (sample + rng.range(-noise, noise)).clamp(-32768, 32767) as i16)
        .collect()
}

/// FNV-1a hash over 32-bit values.
struct Checksum(u64);

impl Checksum {
    fn add(&mut self, value: i32) {
        for byte in value.to_le_bytes() {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}

/// Run the pitch analysis on many signals for one sampling rate, frame
/// length and complexity. Returns the checksum of all outputs and the number
/// of signals that were found voiced.
pub fn run(fs_khz: usize, nb_subfr: usize, complexity: usize) -> (u64, usize) {
    let mut rng =
        XorShift(0x9E37_79B9_7F4A_7C15 ^ ((fs_khz * 64 + nb_subfr * 8 + complexity) as u64));
    let len = (HISTORY_MS + nb_subfr * SUBFRAME_MS) * fs_khz;
    let mut checksum = Checksum(0xCBF2_9CE4_8422_2325);
    let mut voiced = 0;

    let mut signals: Vec<(Vec<i16>, i32)> = Vec::new();
    // Pulse trains over the whole range of pitch lags (2 ms to 18 ms), at
    // several levels, with and without a change of the pitch.
    for round in 0..240 {
        let period = rng.range(2 * fs_khz as i32, 18 * fs_khz as i32) as usize;
        let drift_q8 = [0, 0, 40, -40, 200, -200][round % 6];
        let amplitude = [60, 800, 6000, 20000, 32000][round % 5];
        let noise = [0, amplitude / 50, amplitude / 8, amplitude / 2][round % 4];
        let signal = pulse_train(&mut rng, len, period, drift_q8, amplitude, noise);
        // The lag of the previous frame: none, the true one, or another one.
        let prev_lag = match round % 3 {
            0 => 0,
            1 => period as i32,
            _ => rng.range(2 * fs_khz as i32, 18 * fs_khz as i32),
        };
        signals.push((signal, prev_lag));
    }
    // Signals without a pitch: noise at three levels, silence, a constant.
    for level in [3, 500, 32767] {
        let noise: Vec<i16> = (0..len).map(|_| rng.range(-level, level) as i16).collect();
        signals.push((noise, 0));
    }
    signals.push((vec![0i16; len], 0));
    signals.push((vec![12345i16; len], 100));
    // Full-scale square waves.
    for period in [3 * fs_khz, 7 * fs_khz, 17 * fs_khz] {
        let square: Vec<i16> = (0..len)
            .map(|n| {
                if (n / (period / 2)) % 2 == 0 {
                    32767
                } else {
                    -32768
                }
            })
            .collect();
        signals.push((square, period as i32));
    }

    for (index, (signal, prev_lag)) in signals.iter().enumerate() {
        // Thresholds as the encoder sets them, and lower ones.
        let search_thres1_q16 = [52429, 45875, 30000][index % 3];
        let search_thres2_q13 = [4915, 3000, 1500, 6500][index % 4];
        let mut pitch_out = [0i32; 4];
        let mut lag_index = 0i16;
        let mut contour_index = 0i8;
        let mut ltp_corr_q15 = 0i32;
        let unvoiced = silk_pitch_analysis_core(
            signal,
            &mut pitch_out,
            &mut lag_index,
            &mut contour_index,
            &mut ltp_corr_q15,
            *prev_lag,
            search_thres1_q16,
            search_thres2_q13,
            fs_khz as i32,
            complexity,
            nb_subfr,
        );
        if unvoiced == 0 {
            voiced += 1;
        }
        checksum.add(unvoiced);
        for lag in pitch_out {
            checksum.add(lag);
        }
        checksum.add(lag_index as i32);
        checksum.add(contour_index as i32);
        checksum.add(ltp_corr_q15);
    }
    (checksum.0, voiced)
}

/// Sampling rate in kHz, number of subframes, complexity of the pitch
/// analysis, checksum of the outputs before the change.
pub const EXPECTED: [(usize, usize, usize, u64); 18] = [
    (8, 4, 0, 0xFCF6_3CBE_0AF2_5DE8),
    (8, 4, 1, 0xF54F_3E57_5F47_88F8),
    (8, 4, 2, 0x7726_F695_D21E_9722),
    (8, 2, 0, 0xEDB4_0E9E_C17A_E98F),
    (8, 2, 1, 0xE69D_AC9F_8D5A_8942),
    (8, 2, 2, 0xAFDD_4ABA_7F00_EFF0),
    (12, 4, 0, 0x9B15_C771_32E9_A7A1),
    (12, 4, 1, 0x6F5D_B641_4C49_8E91),
    (12, 4, 2, 0x08F7_F5EE_D8F8_03DE),
    (12, 2, 0, 0xF8DE_2EFC_B496_A3CE),
    (12, 2, 1, 0x0A30_7C56_D102_FD74),
    (12, 2, 2, 0xBA71_E57E_4548_779F),
    (16, 4, 0, 0x5E28_41B9_1E09_288A),
    (16, 4, 1, 0xB3A7_617F_4B2A_F776),
    (16, 4, 2, 0x27D9_8350_3C89_D7C6),
    (16, 2, 0, 0x027C_B905_8456_4E34),
    (16, 2, 1, 0xE14D_43D4_0904_EC45),
    (16, 2, 2, 0xB5F6_9169_B000_C096),
];

#[test]
fn outputs_match_the_version_before_the_change() {
    for (fs_khz, nb_subfr, complexity, expected) in EXPECTED {
        let (checksum, voiced) = run(fs_khz, nb_subfr, complexity);
        assert_eq!(
            checksum, expected,
            "{fs_khz} kHz, {nb_subfr} subframes, complexity {complexity}: {checksum:#018x}"
        );
        // The last search stage only runs for voiced signals at 12 and
        // 16 kHz. A good part of the 248 signals must reach it.
        assert!(
            voiced >= 60,
            "{fs_khz} kHz, {nb_subfr} subframes, complexity {complexity}: {voiced} voiced"
        );
    }
}
