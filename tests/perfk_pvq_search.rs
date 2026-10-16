//! Properties of `pvq_search` that hold on every target and for every one
//! of its internal paths (2 or 4 coefficients, up to 16, 17 to 31, 32 and
//! more; with and without the projection step).
//!
//! The search writes `k` pulses into `y[..n]`. Every entry of `y[..n]` must
//! be written, whatever the buffer held before; nothing after `n` may change;
//! the magnitudes add up to `k`; and a pulse has the sign of its coefficient.

use opus1::pvq::pvq_search;

/// Small deterministic generator, so that the test needs no dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

const SENTINEL: i32 = 0x5a5a_5a5a;

fn check(x: &[f32], k: i32, what: &str) {
    let n = x.len();
    let mut y = vec![SENTINEL; n + 2];
    pvq_search(x, &mut y, k, n);
    assert_eq!(&y[n..], &[SENTINEL, SENTINEL], "{what}: n {n} k {k}: wrote past n");
    let pulses: i64 = y[..n].iter().map(|v| v.unsigned_abs() as i64).sum();
    assert_eq!(pulses, k.max(0) as i64, "{what}: n {n} k {k}: y {:?}", &y[..n]);
    // With two coefficients and a sum of magnitudes below 1e-15 the search
    // puts all pulses on the first one with a positive sign. Everywhere else
    // a pulse has the sign of its coefficient.
    let sum: f32 = x.iter().map(|v| v.abs()).sum();
    if n != 2 || sum >= 1.0e-15 {
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if xi < 0.0 {
                assert!(yi <= 0, "{what}: n {n} k {k}: x[{i}] {xi} y {yi}");
            } else {
                assert!(yi >= 0, "{what}: n {n} k {k}: x[{i}] {xi} y {yi}");
            }
        }
    }
    // The result does not depend on what the buffer held before.
    let mut again = vec![-7i32; n + 2];
    pvq_search(x, &mut again, k, n);
    assert_eq!(&again[..n], &y[..n], "{what}: n {n} k {k}: depends on old contents");
}

#[test]
fn search_writes_k_pulses_with_the_signs_of_the_input() {
    let mut rng = Lcg(21);
    let sizes: Vec<usize> = (1..=64).chain([72, 88, 96, 144, 176]).collect();
    for &n in &sizes {
        for k in [0, 1, 2, 3, 4, 5, 6, 8, 11, 16, 24, 40, 96, 128] {
            for _ in 0..4 {
                let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
                check(&x, k, "random");
            }
            // A few large coefficients among small ones.
            let mut sparse: Vec<f32> = (0..n).map(|_| rng.unit() * 1.0e-3).collect();
            for _ in 0..1 + n / 8 {
                sparse[rng.next_u32() as usize % n] = rng.unit();
            }
            check(&sparse, k, "sparse");
            // Sums below the threshold of the projection step.
            check(&vec![0.0; n], k, "zero");
            let tiny: Vec<f32> = (0..n).map(|_| rng.unit() * 1.0e-20).collect();
            check(&tiny, k, "tiny");
            // One coefficient only, and all negative.
            let mut one = vec![0.0f32; n];
            one[rng.next_u32() as usize % n] = -0.75;
            check(&one, k, "one");
            let negative: Vec<f32> = (0..n).map(|_| -rng.unit().abs() - 0.01).collect();
            check(&negative, k, "negative");
            // Equal magnitudes: every position is as good as the others.
            let flat: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
            check(&flat, k, "flat");
        }
    }
}

#[test]
fn search_with_no_dimension_writes_nothing() {
    let mut y = [SENTINEL; 2];
    for k in [0, 1, 2, 9] {
        pvq_search(&[], &mut y, k, 0);
        assert_eq!(y, [SENTINEL; 2]);
    }
}
