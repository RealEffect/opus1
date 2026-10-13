//! `silk_ana_filt_bank_1` computes its two all-pass filters in another order
//! than libopus writes them, so that each filter state waits for fewer
//! operations. These tests compare it with the loop in the order of libopus.

use opus_rs::silk::ana_filt_bank_1::silk_ana_filt_bank_1;
use opus_rs::silk::macros::{silk_sat16, silk_smlawb, silk_smulwb};

const A_FB1_20: i32 = 5394 << 1;
const A_FB1_21: i32 = -24290;

/// The function before the change. The additions wrap explicitly (in a
/// release build they wrap as well).
fn reference(input: &[i16], s: &mut [i32], out_l: &mut [i16], out_h: &mut [i16], n: usize) {
    if !(2..=4096).contains(&n) || n > input.len() {
        return;
    }
    let n2 = n >> 1;
    if n2 > out_l.len() || n2 > out_h.len() {
        return;
    }
    for k in 0..n2 {
        let in32_even = (input[2 * k] as i32) << 10;
        let y_even = in32_even.wrapping_sub(s[0]);
        let x_even = silk_smlawb(y_even, y_even, A_FB1_21);
        let out_1 = s[0].wrapping_add(x_even);
        s[0] = in32_even.wrapping_add(x_even);

        let in32_odd = (input[2 * k + 1] as i32) << 10;
        let y_odd = in32_odd.wrapping_sub(s[1]);
        let x_odd = silk_smulwb(y_odd, A_FB1_20);
        let out_2 = s[1].wrapping_add(x_odd);
        s[1] = in32_odd.wrapping_add(x_odd);

        out_l[k] = silk_sat16(out_2.wrapping_add(out_1).wrapping_add(1024) >> 11) as i16;
        out_h[k] = silk_sat16(out_2.wrapping_sub(out_1).wrapping_add(1024) >> 11) as i16;
    }
}

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

/// Run both functions from the same state on consecutive blocks and require
/// the same outputs and the same state after every block.
fn check(blocks: &[Vec<i16>], n_of: impl Fn(usize) -> usize, start: [i32; 2], what: &str) {
    let mut expected_state = start;
    let mut state = start;
    for (index, input) in blocks.iter().enumerate() {
        let n = n_of(input.len());
        // A marker value shows what was written.
        let mut expected_l = vec![0x5a5ai16; input.len() / 2 + 2];
        let mut expected_h = expected_l.clone();
        let mut out_l = expected_l.clone();
        let mut out_h = expected_l.clone();
        reference(
            input,
            &mut expected_state,
            &mut expected_l,
            &mut expected_h,
            n,
        );
        silk_ana_filt_bank_1(input, &mut state, &mut out_l, &mut out_h, n);
        assert_eq!(out_l, expected_l, "{what}: low band, block {index}, n {n}");
        assert_eq!(out_h, expected_h, "{what}: high band, block {index}, n {n}");
        assert_eq!(state, expected_state, "{what}: state, block {index}, n {n}");
    }
}

#[test]
fn speech_like_blocks_match_reference() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for len in [64usize, 80, 120, 160, 240, 320] {
        // A slowly varying signal plus noise, 25 frames in a row.
        let mut level = 0i32;
        let blocks: Vec<Vec<i16>> = (0..25)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        level = (level * 15 / 16 + rng.range(-2000, 2000)).clamp(-30000, 30000);
                        (level + rng.range(-500, 500)).clamp(-32768, 32767) as i16
                    })
                    .collect()
            })
            .collect();
        check(&blocks, |n| n, [0, 0], "speech-like");
    }
}

#[test]
fn every_length_matches_reference() {
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    // Lengths from 0 upwards, even and odd, also with `n` below the length
    // of the input.
    for len in 0..=70usize {
        let blocks: Vec<Vec<i16>> = (0..6)
            .map(|_| (0..len).map(|_| rng.range(-32768, 32767) as i16).collect())
            .collect();
        let start = [
            rng.range(-(1 << 26), 1 << 26),
            rng.range(-(1 << 26), 1 << 26),
        ];
        check(&blocks, |n| n, start, "full length");
        check(&blocks, |n| n.saturating_sub(1), start, "one sample less");
        check(&blocks, |n| n / 2, start, "half");
        check(&blocks, |n| n + 1, start, "longer than the input");
    }
}

#[test]
fn extreme_values_match_reference() {
    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    let extremes = [i16::MIN, i16::MAX, -1, 0, 1];
    for round in 0..2_000 {
        let len = 2 * rng.range(1, 40) as usize + (round % 2);
        let blocks: Vec<Vec<i16>> = (0..3)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        if round % 3 == 0 {
                            extremes[rng.range(0, 4) as usize]
                        } else {
                            rng.range(-32768, 32767) as i16
                        }
                    })
                    .collect()
            })
            .collect();
        // Any 32-bit state: the sums wrap around.
        let start = [rng.next() as i32, rng.next() as i32];
        check(&blocks, |n| n, start, "extreme values");
    }
    for value in extremes {
        for state in [i32::MIN, i32::MAX, 0, -1] {
            let blocks = vec![vec![value; 320]; 4];
            check(&blocks, |n| n, [state, state], "constant input");
        }
    }
}

#[test]
fn short_output_buffers_leave_everything_untouched() {
    let input = [1000i16; 16];
    let mut state = [5, 7];
    let mut out_l = [0x5a5ai16; 7];
    let mut out_h = [0x5a5ai16; 8];
    silk_ana_filt_bank_1(&input, &mut state, &mut out_l, &mut out_h, 16);
    assert_eq!((state, out_l, out_h), ([5, 7], [0x5a5a; 7], [0x5a5a; 8]));
    let mut out_l = [0x5a5ai16; 8];
    let mut out_h = [0x5a5ai16; 7];
    silk_ana_filt_bank_1(&input, &mut state, &mut out_l, &mut out_h, 16);
    assert_eq!((state, out_l, out_h), ([5, 7], [0x5a5a; 8], [0x5a5a; 7]));
    // `n` of 0 and 1, and above 4096, do nothing either.
    let long = vec![1000i16; 5000];
    let mut long_l = vec![0x5a5ai16; 2500];
    let mut long_h = long_l.clone();
    for n in [0usize, 1, 4097, 5000] {
        silk_ana_filt_bank_1(&long, &mut state, &mut long_l, &mut long_h, n);
        assert_eq!(state, [5, 7]);
        assert!(long_l.iter().chain(&long_h).all(|&v| v == 0x5a5a));
    }
}
