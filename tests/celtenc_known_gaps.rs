//! Encoder behaviour that still differs from libopus and that the LFE tests
//! ran into. Every test here states what libopus does, fails on the current
//! code, and is ignored with the reason. The cause is outside
//! `CeltEncoder`. Run them with
//! `cargo test --release --test celtenc_known_gaps -- --ignored --nocapture`
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
/// finds worse pulse vectors than libopus on aarch64.
///
/// Measured for 8 coefficients, where every second vector has one strong
/// coefficient: the result is worse in 32 to 42 % of the vectors. Averaged
/// over all vectors the match with the input drops by 0.03 to 0.045 (1 is
/// a perfect match). With 5 and more pulses the two searches agree except
/// for near ties.
///
/// Cause: `pvq_search_neon` in src/pvq.rs, branch `k <= 4`. It picks the
/// position with the largest `Rxy / Ryy`. libopus picks the largest
/// `Rxy * Rxy / Ryy`. Squaring `best_num` and `rxy` in that loop makes the
/// two searches agree; `pvq_search_small_k` in the same file already does.
///
/// Bands with 2 to 4 pulses are the common case at low and medium
/// bitrates, so this costs every CELT stream quality, not only LFE.
#[test]
#[ignore = "src/pvq.rs pvq_search_neon, k <= 4: compares Rxy/Ryy where libopus compares Rxy*Rxy/Ryy"]
fn pulse_search_with_few_pulses_matches_libopus() {
    use opus_rs::pvq::pvq_search;
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

/// LFE packets of 11 to 13 bytes decode 1.5 to 2.8 dB below libopus's.
///
/// At these sizes band 0 gets 2 to 4 pulses, which is where the pulse
/// search above is off. At the other sizes from 8 to 41 bytes the SNR is
/// within 0.4 dB of libopus (tests/celtenc_lfe.rs). With the two squares
/// added in `pvq_search_neon` the gap at 11 to 13 bytes closes to 0.2 dB.
#[test]
#[ignore = "consequence of the pulse search with 2 to 4 pulses (src/pvq.rs pvq_search_neon)"]
fn lfe_quality_matches_libopus_in_packets_of_11_to_13_bytes() {
    let pcm = lfe_signal(100);
    let mut largest_gap = 0f64;
    for rate in [4400, 4800, 5200] {
        let mut cfg = Cfg::lfe(rate);
        cfg.vbr = false;
        let rs_snr = snr_db(&pcm, &dec_c(&enc_rs(&cfg, &pcm), 1), 1, 5);
        let c_snr = snr_db(&pcm, &dec_c(&enc_c(&cfg, &pcm), 1), 1, 5);
        println!(
            "LFE CBR {rate} b/s ({} bytes): SNR opus-rs {rs_snr:.2} dB, libopus {c_snr:.2} dB",
            cfg.cbr_packet_bytes()
        );
        largest_gap = largest_gap.max(c_snr - rs_snr);
    }
    assert!(
        largest_gap <= 1.0,
        "LFE SNR is up to {largest_gap:.2} dB below libopus"
    );
}

/// On a steady tone the LFE packets of opus-rs are not the ones libopus
/// writes, although both encoders make the same decisions for the frame.
///
/// Measured on a 50 Hz tone in 40-byte packets: the frame headers (energy,
/// post-filter, allocation) are the same in 99 of 100 frames, but only 5
/// of 100 packets are identical. The pulse vectors differ.
///
/// Cause: `pvq_search_n2` in src/pvq.rs, which `pvq_search` uses for a band
/// or half band of 2 coefficients. It rounds the projection of the input
/// onto the pyramid. libopus (`op_pvq_search_c`) rounds that projection
/// down and places the remaining pulses one by one. The vectors of
/// `pvq_search_n2` match the input as well or better, so this is not a loss
/// of quality on average, but the stream is not libopus's. With
/// `pvq_search_scalar` in its place, tried in a scratch copy, 95 of 100
/// packets are identical.
#[test]
#[ignore = "src/pvq.rs pvq_search_n2: rounds a projection where libopus searches pulse by pulse"]
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
