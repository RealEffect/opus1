//! The pulse search against the search of libopus itself (`op_pvq_search_c`,
//! celt/vq.c), called directly in the C library the tests link.
//!
//! The two are compared where opus-rs is meant to run the libopus algorithm:
//! bands of 2 values, and bands of up to 31 values with 2 to 4 pulses. For
//! more pulses in a band of up to 16 values aarch64 has its own scoring.
//!
//! The pulse vectors must be the same. One difference is allowed where the
//! library was not built by clang for Apple silicon: libopus is compiled C,
//! and its compiler may round a product and a sum once (a fused
//! multiply-add) or separately. That can change the last bit of a sum and so
//! the winner of an exact tie. Such a case is accepted when the two vectors
//! match the input equally well, and it must be rare.
//!
//! On Apple silicon the library has one fused multiply-add in this function,
//! `xy = xy + X[j] * y[j]` in the projection step, and `pvq_search_libopus`
//! (src/pvq.rs) has the same one on aarch64. There no difference is allowed.

use audiopus_sys as _;
use opus_rs::pvq::pvq_search;
use std::os::raw::c_int;

unsafe extern "C" {
    /// `opus_val16 op_pvq_search_c(celt_norm *X, int *iy, int K, int N, int arch)`
    /// of the float build. It replaces `X` by its magnitudes.
    fn op_pvq_search_c(x: *mut f32, iy: *mut c_int, k: c_int, n: c_int, arch: c_int) -> f32;
}

/// Small deterministic generator, so that the tests need no dependency.
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

fn libopus_search(x: &[f32], k: i32) -> Vec<i32> {
    let mut x = x.to_vec();
    let mut iy = vec![0 as c_int; x.len()];
    // SAFETY: both buffers hold `x.len()` values, which is what the function
    // reads and writes.
    unsafe { op_pvq_search_c(x.as_mut_ptr(), iy.as_mut_ptr(), k, x.len() as c_int, 0) };
    iy
}

/// How well the pulse vector `y` matches `x`: the cosine of the angle
/// between them.
fn match_quality(x: &[f32], y: &[i32]) -> f64 {
    let (mut xy, mut yy) = (0f64, 0f64);
    for (&a, &b) in x.iter().zip(y) {
        xy += f64::from(a) * f64::from(b);
        yy += f64::from(b) * f64::from(b);
    }
    xy / yy.sqrt().max(1e-30)
}

/// A band of `n` values with unit energy. Every second one has a strong
/// value, like a band with a tone in it; every eighth has exact ties.
fn band(rng: &mut Lcg, n: usize, trial: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    if trial.is_multiple_of(2) {
        x[trial / 2 % n] *= 6.0;
    }
    if trial % 8 == 7 {
        for v in x.iter_mut() {
            *v = if *v < 0.0 { -0.5 } else { 0.5 };
        }
    }
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    x.iter_mut().for_each(|v| *v /= norm);
    x
}

/// How many of `trials` vectors may differ from libopus by a tie: none where
/// the library and opus-rs round the same sums once (see the top of this
/// file), one in 200 elsewhere.
fn allowed_ties(trials: usize) -> usize {
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        0
    } else {
        trials / 200
    }
}

/// Compares `pvq_search` with libopus for the given sizes and pulse counts.
/// Returns the number of trials and the number of different vectors.
fn compare(sizes: &[usize], pulses: &[i32], seed: u64) -> (usize, usize) {
    let mut rng = Lcg(seed);
    let (mut trials, mut different) = (0, 0);
    for &n in sizes {
        for &k in pulses {
            for trial in 0..1500 {
                let x = band(&mut rng, n, trial);
                let want = libopus_search(&x, k);
                let mut got = vec![0i32; n];
                pvq_search(&x, &mut got, k, n);
                trials += 1;
                if got != want {
                    different += 1;
                    let (q_got, q_want) = (match_quality(&x, &got), match_quality(&x, &want));
                    assert!(
                        (q_got - q_want).abs() < 1e-6,
                        "n {n} k {k} trial {trial}: {got:?} (match {q_got:.6}) is not the vector \
                         of libopus {want:?} (match {q_want:.6}) for {x:?}"
                    );
                }
            }
        }
    }
    (trials, different)
}

/// With 2 to 4 pulses in a band of up to 31 values (not 2 or 4 values, which
/// have their own code) the search is the one of libopus.
#[test]
fn few_pulses_give_the_vector_of_libopus() {
    let sizes = [3usize, 5, 6, 7, 8, 9, 12, 15, 16, 17, 22, 24, 31];
    let (trials, different) = compare(&sizes, &[2, 3, 4], 0x51);
    println!("{different} of {trials} vectors differ from libopus, all of them ties");
    assert!(different <= allowed_ties(trials), "{different} of {trials} vectors differ");
}

/// A band of two values runs the search of libopus for every number of
/// pulses the codec can ask for.
#[test]
fn two_values_give_the_vector_of_libopus() {
    let pulses: Vec<i32> = (1..=40).chain([64, 100, 128]).collect();
    let (trials, different) = compare(&[2], &pulses, 0x52);
    println!("{different} of {trials} vectors differ from libopus, all of them ties");
    assert!(different <= allowed_ties(trials), "{different} of {trials} vectors differ");
}


/// Two inputs where the last bit of the correlation sum decides between two
/// pulse vectors that match equally well. With product and sum rounded
/// separately in the projection step, opus-rs gave [-7, 7] and [-12, 10]
/// here; the libopus build for Apple silicon gives [-8, 6] and [-13, 9]
/// (6 such cases in 8.4 million random bands of two values).
#[test]
fn ties_between_two_values_fall_as_in_libopus() {
    let cases: [([f32; 2], i32); 2] = [([-0.7554538, 0.655202], 14), ([-0.7959981, 0.60529906], 22)];
    let mut different = 0;
    for (x, k) in cases {
        let want = libopus_search(&x, k);
        let mut got = vec![0i32; 2];
        pvq_search(&x, &mut got, k, 2);
        println!("{x:?} with {k} pulses: {got:?}, libopus {want:?}");
        if got != want {
            different += 1;
            let (q_got, q_want) = (match_quality(&x, &got), match_quality(&x, &want));
            assert!((q_got - q_want).abs() < 1e-6, "{got:?} is not a tie with {want:?}");
        }
    }
    assert!(different <= allowed_ties(cases.len()), "{different} of {} ties fall differently", cases.len());
}
