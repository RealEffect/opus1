//! SILK-only packets from libopus must decode bit-exactly at 24 kHz.
//!
//! SILK is fixed-point, so opus-rs and libopus must produce the same 16-bit
//! samples from the same packets. The streams here cover narrowband,
//! mediumband and wideband SILK at bitrates from 6 to 64 kb/s, encoder
//! complexity 0 and 10, and four kinds of signal. The decoder runs at
//! 24 kHz (the SILK output is resampled from 8 / 12 / 16 kHz).
//!
//! The chirp rows are the regression for the stability limit `A_LIMIT`
//! (see `lpc_inverse_pred_gain_matches_libopus` in k24_silk_internals.rs):
//! a pure tone gives the sharply resonant filters that reach it.

mod k24_common;
use k24_common::*;

#[test]
fn silk_only_streams_decode_bit_exactly_at_24k() {
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("chirp", chirp(16000, 32000)),
        ("multitone", multitone(16000, 32000)),
        ("fullscale", full_scale(16000, 32000)),
        ("speech", speech(16000, 2.0)),
    ];
    let mut failures = Vec::new();
    let mut rows = 0;
    for (signal, pcm) in &signals {
        for (bw, bw_name) in [(BW_NB, "NB"), (BW_MB, "MB"), (BW_WB, "WB")] {
            for bitrate in [6000, 10000, 16000, 24000, 32000, 40000, 64000] {
                for complexity in [0, 10] {
                    let mut enc = CEnc::voip(16000, 1);
                    enc.force_mode(MODE_SILK)
                        .bandwidth(bw)
                        .bitrate(bitrate)
                        .complexity(complexity);
                    let packets = enc.encode_all(pcm, 320);
                    let reference = c_decode_all(&packets, 24000, 1);
                    let ours = rs_decode_all(&packets, 24000, 1).expect("opus-rs decode");
                    let cmp = compare(&reference, &ours, 1, 480, 24000);
                    rows += 1;
                    if cmp.max_abs_diff != 0.0 {
                        failures.push(format!(
                            "{signal} SILK-{bw_name} {bitrate} b/s complexity {complexity}: {cmp}"
                        ));
                    }
                }
            }
        }
    }
    println!("{} of {rows} streams bit-exact", rows - failures.len());
    assert!(failures.is_empty(), "not bit-exact:\n{}", failures.join("\n"));
}
