//! Encoder behaviour that differed from libopus and that the LFE tests ran
//! into. Every test here states what libopus does. All three failed, and
//! were ignored, until the pulse search in src/pvq.rs became a port of
//! `op_pvq_search_c` for 2 to 4 pulses and for bands of two values. Two
//! pass and stay as guards; `lfe_tone_packets_are_the_ones_libopus_writes`
//! is ignored again, for a difference of the pitch pre-filter (see there).
//! Run them with
//! `cargo test --release --test celtenc_known_gaps -- --nocapture`
//! to see the measured numbers.

mod celtenc_common;
use celtenc_common::*;

/// The pulse search of libopus (`op_pvq_search_c`, celt/vq.c, float build).
///
/// It places `k` pulses one by one. Each pulse goes where it leaves the
/// largest `Rxy * Rxy / Ryy`: `Rxy` is the correlation of the pulse vector
/// with the input, `Ryy` is the energy of the pulse vector.
fn libopus_pulse_search(x_in: &[f32], k: i32) -> Vec<i32> {
    let n = x_in.len();
    let mut x: Vec<f32> = x_in.iter().map(|v| v.abs()).collect();
    let mut iy = vec![0i32; n];
    let mut y = vec![0f32; n];
    let (mut xy, mut yy) = (0f32, 0f32);
    let mut pulses_left = k;
    // First a projection onto the pyramid, when there are many pulses.
    if k > (n as i32 >> 1) {
        let mut sum: f32 = x.iter().sum();
        if !(sum > 1e-15 && sum < 64.0) {
            x.fill(0.0);
            x[0] = 1.0;
            sum = 1.0;
        }
        let rcp = (k as f32 + 0.8) * (1.0 / sum);
        for j in 0..n {
            iy[j] = (rcp * x[j]).floor() as i32;
            y[j] = iy[j] as f32;
            yy += y[j] * y[j];
            xy += x[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp + tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }
    for _ in 0..pulses_left {
        yy += 1.0;
        let mut best_id = 0;
        let mut best_num = (xy + x[0]) * (xy + x[0]);
        let mut best_den = yy + y[0];
        for j in 1..n {
            let rxy = (xy + x[j]) * (xy + x[j]);
            let ryy = yy + y[j];
            if best_den * rxy > ryy * best_num {
                best_den = ryy;
                best_num = rxy;
                best_id = j;
            }
        }
        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }
    for j in 0..n {
        if x_in[j] < 0.0 {
            iy[j] = -iy[j];
        }
    }
    iy
}

/// How well the pulse vector `y` matches `x`: 1 is a perfect match.
fn match_quality(x: &[f32], y: &[i32]) -> f64 {
    let (mut xy, mut yy) = (0f64, 0f64);
    for (&a, &b) in x.iter().zip(y) {
        xy += f64::from(a) * f64::from(b);
        yy += f64::from(b) * f64::from(b);
    }
    xy / yy.sqrt().max(1e-30)
}

/// With 2, 3 or 4 pulses in a band of up to 16 coefficients, `pvq_search`
/// finds pulse vectors that match the input as well as libopus's.
///
/// Measured now: worse than libopus in 0 of 2000 vectors for every band
/// size and pulse count below.
///
/// What this guards against: on aarch64 `pvq_search_neon` in src/pvq.rs,
/// branch `k <= 4`, used to pick the position with the largest `Rxy / Ryy`.
/// libopus picks the largest `Rxy * Rxy / Ryy`. For 8 coefficients, where
/// every second vector has one strong coefficient, the result was worse in
/// 32 to 42 % of the vectors, and the match with the input dropped by 0.03
/// to 0.045 on average (1 is a perfect match).
///
/// Bands with 2 to 4 pulses are the common case at low and medium
/// bitrates, so this cost every CELT stream quality, not only LFE.
#[test]
fn pulse_search_with_few_pulses_matches_libopus() {
    use opus1::internals::pvq::pvq_search;
    let mut noise = Noise(5);
    let mut worst = 0f64;
    for n in [3usize, 6, 8, 12, 16] {
        for k in 2..=4 {
            let (mut worse, mut loss) = (0, 0f64);
            let trials = 2000;
            for trial in 0..trials {
                let mut x: Vec<f32> = (0..n).map(|_| noise.next()).collect();
                // Every second vector has one strong coefficient, like a
                // band with a tone in it.
                if trial % 2 == 0 {
                    x[trial / 2 % n] *= 6.0;
                }
                let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                x.iter_mut().for_each(|v| *v /= norm);
                let mut y = vec![0i32; n];
                pvq_search(&x, &mut y, k, n);
                let reference = libopus_pulse_search(&x, k);
                let d = match_quality(&x, &reference) - match_quality(&x, &y);
                if d > 1e-6 {
                    worse += 1;
                    loss += d;
                }
            }
            let mean_loss = loss / trials as f64;
            println!(
                "{n} coefficients, {k} pulses: worse than libopus in {worse} of {trials}, \
                 mean loss {mean_loss:.4}"
            );
            worst = worst.max(mean_loss);
        }
    }
    assert!(
        worst < 1e-3,
        "the pulse search loses up to {worst:.4} against libopus"
    );
}

/// LFE packets of 11 to 13 bytes decode within 1 dB of libopus's.
///
/// At these sizes band 0 gets 2 to 4 pulses, which is where the pulse
/// search above was off: the packets decoded 1.5 to 2.8 dB below
/// libopus's. Measured now: 6.44, 8.18 and 10.58 dB against 6.36, 8.32 and
/// 10.58 dB for libopus. At the other sizes from 8 to 41 bytes the SNR was
/// already within 0.4 dB of libopus (tests/celtenc_lfe.rs).
#[test]
fn lfe_quality_matches_libopus_in_packets_of_11_to_13_bytes() {
    let pcm = lfe_signal(100);
    let mut largest_gap = 0f64;
    for rate in [4400, 4800, 5200] {
        let mut cfg = Cfg::lfe(rate);
        cfg.vbr = false;
        let rs_snr = snr_db(&pcm, &dec_c(&enc_rs(&cfg, &pcm), 1), 1, 5);
        let c_snr = snr_db(&pcm, &dec_c(&enc_c(&cfg, &pcm), 1), 1, 5);
        println!(
            "LFE CBR {rate} b/s ({} bytes): SNR opus1 {rs_snr:.2} dB, libopus {c_snr:.2} dB",
            cfg.cbr_packet_bytes()
        );
        largest_gap = largest_gap.max(c_snr - rs_snr);
    }
    assert!(
        largest_gap <= 1.0,
        "LFE SNR is up to {largest_gap:.2} dB below libopus"
    );
}

/// On a steady tone the LFE packets of opus1 are the ones libopus writes.
///
/// Measured on a 50 Hz tone in 40-byte packets, when this test was written:
/// the frame headers (energy, post-filter, allocation) are the same in 99
/// of 100 frames, and 95 of 100 packets are identical.
///
/// Later, frames 20 and 23 got the pitch pre-filter period 960 where
/// libopus has 959, and 25 of 100 packets were identical. The period comes
/// from the frequency `tone_detect` finds, which sits near the rounding
/// boundary of the period here, so it depends on the rounding of every step
/// before it. The disassembly of libopus 1.6.1 on 64-bit ARM shows fused
/// multiply-adds in `dc_reject` (the encoder's input high-pass) and in
/// `tone_lpc`, and a double-precision check and `acos` in `tone_detect`.
/// With src/celt.rs and src/hp_cutoff.rs rounding the same way, all 100
/// packets are identical.
///
/// What this guards against: `pvq_search` used `pvq_search_n2` for a band
/// or half band of 2 coefficients. It rounds the projection of the input
/// onto the pyramid. libopus (`op_pvq_search_c`) rounds that projection
/// down and places the remaining pulses one by one. The vectors of
/// `pvq_search_n2` match the input as well or better, so this was not a
/// loss of quality on average, but the stream was not libopus's: only 5 of
/// 100 packets were identical.
#[test]
fn lfe_tone_packets_are_the_ones_libopus_writes() {
    let pcm: Vec<f32> = (0..100 * FRAME)
        .map(|i| (0.4 * (std::f64::consts::TAU * 50.0 * i as f64 / FS as f64).sin()) as f32)
        .collect();
    let mut cfg = Cfg::lfe(16_000);
    cfg.vbr = false;
    let rs = enc_rs(&cfg, &pcm);
    let c = enc_c(&cfg, &pcm);
    let same_header = headers(&rs, 1, 13, 3)
        .iter()
        .zip(&headers(&c, 1, 13, 3))
        .filter(|(a, b)| a == b)
        .count();
    let identical = rs.iter().zip(&c).filter(|(a, b)| a == b).count();
    println!(
        "{} packets: {same_header} with the frame header of libopus, {identical} identical",
        rs.len()
    );
    assert!(
        same_header * 10 >= rs.len() * 9,
        "frame headers: {same_header}"
    );
    assert!(
        identical * 10 >= rs.len() * 9,
        "only {identical} of {} packets are the ones libopus writes",
        rs.len()
    );
}

