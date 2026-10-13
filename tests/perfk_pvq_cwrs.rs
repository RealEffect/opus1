//! `icwrs` and `cwrsi` (pulse vector to index and back) against a reference
//! written from the definition of the index.
//!
//! The reference uses the exact number of pulse vectors, computed with the
//! recurrence in 128-bit integers. It shares no code and no table with
//! `src/pvq.rs`. The pairs `(n, k)` cover both branches of `cwrsi` (more
//! pulses than dimensions and the opposite), the ends of every row of the
//! lookup table, and pairs outside the table.

use opus_rs::pvq::{MAX_PVQ_K, celt_pvq_v, cwrsi, decode_pulses, encode_pulses, icwrs};
use opus_rs::range_coder::RangeCoder;

/// `u[n][k]` is U(n, k) of libopus `celt/cwrs.c`, from its recurrence.
/// V(n, k) = U(n, k) + U(n, k + 1) is the number of pulse vectors with `n`
/// dimensions and `k` pulses.
struct Exact {
    u: Vec<Vec<u128>>,
}

impl Exact {
    fn new(n_max: usize, k_max: usize) -> Self {
        let mut u = vec![vec![0u128; k_max + 2]; n_max + 1];
        u[0][0] = 1;
        for n in 1..=n_max {
            for k in 1..=k_max + 1 {
                u[n][k] = if n == 1 {
                    1
                } else {
                    u[n - 1][k]
                        .saturating_add(u[n][k - 1])
                        .saturating_add(u[n - 1][k - 1])
                };
            }
        }
        Exact { u }
    }

    fn v(&self, n: usize, k: usize) -> u128 {
        self.u[n][k].saturating_add(self.u[n][k + 1])
    }

    /// The pulse vector with index `i`, from the definition.
    fn decode(&self, n: usize, k: usize, mut i: u128) -> Vec<i32> {
        assert!(i < self.v(n, k));
        let mut y = Vec::with_capacity(n);
        let mut k = k;
        for dims in (2..=n).rev() {
            // Indices from U(dims, k + 1) on belong to a negative entry.
            let q = self.u[dims][k + 1];
            let negative = i >= q;
            if negative {
                i -= q;
            }
            // `rest` pulses remain for the dimensions after this one.
            let mut rest = k;
            while self.u[dims][rest] > i {
                rest -= 1;
            }
            i -= self.u[dims][rest];
            let m = (k - rest) as i32;
            y.push(if negative { -m } else { m });
            k = rest;
        }
        // One dimension left: index 0 is +k, index 1 is -k.
        assert!(i <= 1);
        y.push(if i == 0 { k as i32 } else { -(k as i32) });
        y
    }

    /// The index of the pulse vector `y`, from the definition.
    fn encode(&self, y: &[i32]) -> u128 {
        let n = y.len();
        let mut i = (y[n - 1] < 0) as u128;
        let mut k = y[n - 1].unsigned_abs() as usize;
        for j in (0..n - 1).rev() {
            i += self.u[n - j][k];
            k += y[j].unsigned_abs() as usize;
            if y[j] < 0 {
                i += self.u[n - j][k + 1];
            }
        }
        i
    }
}

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

    fn below(&mut self, bound: u128) -> u128 {
        let r = ((self.next_u32() as u128) << 32) | self.next_u32() as u128;
        r % bound
    }
}

/// The pairs to test: at most `MAX_PVQ_K` pulses and fewer than 2^32 pulse
/// vectors. That is the domain the codec uses.
fn pairs(exact: &Exact) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    // Every small pair: both branches of `cwrsi` and the change between them.
    for n in 2..=26 {
        for k in 1..=26 {
            out.push((n, k));
        }
    }
    // The last column of each table row, and the pair just beyond it.
    let cmax = [
        176usize, 176, 176, 176, 176, 176, 96, 54, 37, 28, 24, 19, 18, 16, 14,
    ];
    for (r, &c) in cmax.iter().enumerate().skip(1) {
        for (n, k) in [
            (c, r - 1),
            (c, r),
            (c - 1, r),
            (r, c - 1),
            (r, c),
            (r + 1, c - 1),
        ] {
            if n >= 2 && k >= 1 {
                out.push((n, k));
            }
        }
    }
    // Sizes the codec uses for wide bands with few pulses.
    for (n, k) in [(32, 5), (36, 4), (44, 6), (48, 5), (64, 4), (72, 3), (88, 4)] {
        out.push((n, k));
    }
    for (n, k) in [(96, 4), (144, 3), (176, 2), (176, 4), (176, 5), (100, 5)] {
        out.push((n, k));
    }
    // The largest pulse counts.
    for n in 2..=5 {
        for k in [64, 100, 127, 128] {
            out.push((n, k));
        }
    }
    out.retain(|&(n, k)| n <= 176 && k <= MAX_PVQ_K && exact.v(n, k) < (1u128 << 32));
    out.sort();
    out.dedup();
    out
}

/// Indices to try: all of them when there are few, else both ends, every
/// boundary between the values of the first entry, and random ones.
fn indices(exact: &Exact, rng: &mut Lcg, n: usize, k: usize) -> Vec<u128> {
    let v = exact.v(n, k);
    if v <= 3000 {
        return (0..v).collect();
    }
    let mut out = vec![0, 1, 2, v / 2, v - 2, v - 1];
    let q = exact.u[n][k + 1];
    for rest in 0..=k {
        let b = exact.u[n][rest];
        for cand in [b.wrapping_sub(1), b, b + 1, q + b - 1, q + b, q + b + 1] {
            if cand < v {
                out.push(cand);
            }
        }
    }
    for _ in 0..200 {
        out.push(rng.below(v));
    }
    out
}

#[test]
fn cwrsi_and_icwrs_match_the_definition() {
    let exact = Exact::new(176, 176);
    let mut rng = Lcg(11);
    let pairs = pairs(&exact);
    assert!(pairs.len() > 400, "only {} pairs", pairs.len());
    let mut vectors = 0u64;
    for &(n, k) in &pairs {
        let v = exact.v(n, k);
        assert_eq!(celt_pvq_v(n as u32, k as u32) as u128, v, "V({n},{k})");
        for i in indices(&exact, &mut rng, n, k) {
            let want = exact.decode(n, k, i);
            // Two extra entries: they must stay untouched.
            let mut got = vec![-99i32; n + 2];
            cwrsi(n as u32, k as u32, i as u32, &mut got);
            assert_eq!(&got[..n], &want[..], "cwrsi n {n} k {k} i {i}");
            assert_eq!(&got[n..], &[-99, -99], "cwrsi wrote past n: n {n} k {k}");
            assert_eq!(
                icwrs(n as u32, k as u32, &want) as u128,
                i,
                "icwrs n {n} k {k} y {want:?}"
            );
            vectors += 1;
        }
    }
    println!("{} pairs, {vectors} vectors", pairs.len());
    assert!(vectors > 100_000, "only {vectors} vectors");
}

/// A random pulse vector with exactly `k` pulses.
fn random_vector(rng: &mut Lcg, n: usize, k: usize) -> Vec<i32> {
    let mut y = vec![0i32; n];
    // Put the pulses in a few positions, so that large entries occur.
    let spots = 1 + rng.next_u32() as usize % n.min(6);
    for _ in 0..k {
        let pos = (rng.next_u32() as usize % spots) * (n / spots);
        y[pos] += 1;
    }
    for v in y.iter_mut() {
        if rng.next_u32() & 1 == 1 {
            *v = -*v;
        }
    }
    y
}

#[test]
fn icwrs_matches_the_definition_on_random_vectors() {
    let exact = Exact::new(176, 176);
    let mut rng = Lcg(12);
    for &(n, k) in &pairs(&exact) {
        for _ in 0..40 {
            let y = random_vector(&mut rng, n, k);
            let want = exact.encode(&y);
            assert!(want < exact.v(n, k));
            assert_eq!(icwrs(n as u32, k as u32, &y) as u128, want, "n {n} k {k} y {y:?}");
            let mut back = vec![0i32; n];
            cwrsi(n as u32, k as u32, want as u32, &mut back);
            assert_eq!(back, y, "n {n} k {k}");
        }
    }
}

/// `icwrs` does not use its `k` argument for the result: a vector with fewer
/// or more pulses than `k` gets the index its own pulse count defines.
#[test]
fn icwrs_ignores_a_wrong_pulse_count() {
    let exact = Exact::new(176, 176);
    let mut rng = Lcg(13);
    for (n, k) in [(8usize, 6usize), (12, 9), (16, 4), (24, 3), (4, 20), (3, 60)] {
        for _ in 0..50 {
            let y = random_vector(&mut rng, n, k);
            let want = exact.encode(&y) as u32;
            for claimed in [0u32, 1, k as u32 - 1, k as u32, k as u32 + 1, 500, u32::MAX] {
                assert_eq!(icwrs(n as u32, claimed, &y), want, "n {n} k {k} claimed {claimed}");
            }
        }
    }
}

#[test]
fn one_dimension() {
    for k in [1u32, 2, 7, 128] {
        let mut y = [0i32; 1];
        cwrsi(1, k, 0, &mut y);
        assert_eq!(y, [k as i32]);
        assert_eq!(icwrs(1, k, &y), 0);
        cwrsi(1, k, 1, &mut y);
        assert_eq!(y, [-(k as i32)]);
        assert_eq!(icwrs(1, k, &y), 1);
    }
}

#[test]
fn pulses_survive_the_range_coder() {
    let exact = Exact::new(176, 176);
    let mut rng = Lcg(14);
    let pairs = pairs(&exact);
    // Groups of 24 vectors per packet: at most 24 * 32 bits.
    for group in pairs.chunks(24) {
        let vectors: Vec<Vec<i32>> = group
            .iter()
            .map(|&(n, k)| random_vector(&mut rng, n, k))
            .collect();
        let mut enc = RangeCoder::new_encoder(256);
        for (&(n, k), y) in group.iter().zip(&vectors) {
            encode_pulses(y, n as u32, k as u32, &mut enc);
        }
        enc.done();
        assert_eq!(enc.error, 0);
        let packet: Vec<u8> = enc.buf[..256].to_vec();
        let mut dec = RangeCoder::new_decoder(&packet);
        for (&(n, k), y) in group.iter().zip(&vectors) {
            let mut got = vec![0i32; n];
            decode_pulses(&mut got, n as u32, k as u32, &mut dec);
            assert_eq!(&got, y, "n {n} k {k}");
        }
    }
    // No pulses: nothing is read and the vector is cleared.
    let mut dec = RangeCoder::new_decoder(&[0u8; 8]);
    let before = dec.tell();
    let mut y = [5i32; 9];
    decode_pulses(&mut y, 9, 0, &mut dec);
    assert_eq!(y, [0; 9]);
    assert_eq!(dec.tell(), before);
}
