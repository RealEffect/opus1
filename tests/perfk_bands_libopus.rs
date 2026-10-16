//! Stereo band coding against libopus, at the level of whole CELT streams:
//! both encoders code the same input, the libopus decoder decodes both
//! streams, and the two results are compared with the input.

mod celtenc_common;
use celtenc_common::*;

/// A stereo signal whose right channel is the left one with the opposite
/// sign, plus a little independent noise: tones between 300 Hz and 12 kHz
/// that drift slowly, so that the frames differ.
fn out_of_phase_signal(frames: usize) -> Vec<f32> {
    let mut noise = Noise(77);
    let tones = [310.0f64, 870.0, 1730.0, 2950.0, 4100.0, 5600.0, 7300.0, 9100.0, 11800.0];
    let mut pcm = Vec::with_capacity(frames * FRAME * 2);
    for i in 0..frames * FRAME {
        let t = i as f64 / FS as f64;
        let mut s = 0.0f64;
        for (k, f) in tones.iter().enumerate() {
            let drift = 1.0 + 0.01 * (0.7 * t + k as f64).sin();
            s += (std::f64::consts::TAU * f * drift * t).sin() / (1.0 + k as f64 * 0.3);
        }
        let s = 0.12 * s as f32;
        pcm.push(s + 0.002 * noise.next());
        pcm.push(-s + 0.002 * noise.next());
    }
    pcm
}

/// Bands that are coded as one channel plus a phase flag (intensity stereo)
/// keep their content when the two channels have opposite signs.
///
/// libopus negates the right channel of such a band before it mixes the two
/// (celt/bands.c, compute_theta: `if (inv) Y[j] = -Y[j]`), and sends the
/// flag so that the decoder negates it again. Without the negation the mix
/// of two opposite channels is close to zero, and the band is coded from
/// rounding noise: its level is right, its content is not.
///
/// Measured per bitrate: the signal-to-noise ratio of the opus1 stream and
/// of the libopus stream, both decoded by libopus, from 3 to 12 kHz, where
/// the encoder uses intensity stereo at these bitrates.
///
/// With the negation: -0.2, 1.5 and 8.8 dB at 24, 32 and 48 kb/s, against
/// 0.4, 1.3 and 11.2 dB for libopus. Without it: -3.3, -3.3 and -3.0 dB,
/// which is worse than coding nothing. The 2.3 dB that are left at 48 kb/s
/// have another cause, which this test does not locate.
#[test]
fn out_of_phase_stereo_keeps_its_high_bands() {
    let pcm = out_of_phase_signal(120);
    let edges = [3000.0f64, 12000.0];
    let mut largest_gap = f64::MIN;
    for bitrate in [24_000, 32_000, 48_000] {
        let cfg = Cfg::new(2, bitrate);
        let rs = dec_c(&enc_rs(&cfg, &pcm), 2);
        let c = dec_c(&enc_c(&cfg, &pcm), 2);
        let mut gap = f64::MIN;
        for channel in 0..2 {
            let rs_snr = range_snr_db(&pcm, &rs, 2, channel, 10, &edges)[0];
            let c_snr = range_snr_db(&pcm, &c, 2, channel, 10, &edges)[0];
            println!(
                "{bitrate} b/s, channel {channel}, 3 to 12 kHz: SNR opus1 {rs_snr:.2} dB, \
                 libopus {c_snr:.2} dB"
            );
            gap = gap.max(c_snr - rs_snr);
        }
        largest_gap = largest_gap.max(gap);
    }
    assert!(
        largest_gap <= 3.0,
        "the high bands of an out-of-phase stereo signal are up to {largest_gap:.2} dB below libopus"
    );
}
