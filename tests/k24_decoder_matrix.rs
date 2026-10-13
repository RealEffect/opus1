//! Decoder matrix: every coding mode and bandwidth libopus can produce,
//! decoded by opus-rs and by libopus at every output rate, compared sample
//! by sample.
//!
//! Streams: SILK-only narrowband / mediumband / wideband, Hybrid
//! super-wideband / fullband, CELT-only narrowband / wideband /
//! super-wideband / fullband; encoder input at 8, 12, 16, 24 and 48 kHz;
//! mono and stereo; 20 ms frames of speech with energy up to Nyquist.
//! Decoder output rates: 8, 12, 16, 24 and 48 kHz.
//!
//! Required agreement with libopus on the same packets:
//! - SILK-only: bit-exact (SILK is fixed-point).
//! - Hybrid at 8 / 12 / 16 kHz: bit-exact (the CELT layer lies above the
//!   output band and contributes nothing).
//! - Hybrid at 24 / 48 kHz and CELT-only: every frame better than 80 dB
//!   (CELT is floating-point).
//!
//! The 8 and 12 kHz columns are the regression for the decoder-direction
//! down-sampler (see `resampler_matches_libopus_for_every_rate_pair`):
//! wideband SILK decoded at 8 kHz scored 0 dB against libopus.

mod k24_common;
use k24_common::*;

const MODES: [(i32, i32, &str, i32); 9] = [
    (MODE_SILK, BW_NB, "SILK-NB", 16000),
    (MODE_SILK, BW_MB, "SILK-MB", 20000),
    (MODE_SILK, BW_WB, "SILK-WB", 24000),
    (MODE_HYBRID, BW_SWB, "Hybrid-SWB", 32000),
    (MODE_HYBRID, BW_FB, "Hybrid-FB", 40000),
    (MODE_CELT, BW_NB, "CELT-NB", 32000),
    (MODE_CELT, BW_WB, "CELT-WB", 40000),
    (MODE_CELT, BW_SWB, "CELT-SWB", 48000),
    (MODE_CELT, BW_FB, "CELT-FB", 64000),
];
const RATES: [usize; 5] = [8000, 12000, 16000, 24000, 48000];

#[test]
fn every_mode_and_bandwidth_at_every_decoder_rate() {
    // Lowest worst-frame SNR seen per (mode, decoder rate).
    let mut table = vec![[f64::INFINITY; 5]; MODES.len()];
    let mut failures = Vec::new();
    let mut streams = 0;
    for &enc_rate in &RATES {
        let mono = speech_fullband(enc_rate, 1.5);
        for channels in [1usize, 2] {
            let pcm = if channels == 2 { to_stereo(&mono, enc_rate) } else { mono.clone() };
            for (mi, &(mode, bw, name, bitrate)) in MODES.iter().enumerate() {
                let mut enc = CEnc::audio(enc_rate, channels);
                enc.force_mode(mode).bandwidth(bw).bitrate(bitrate * channels as i32);
                let packets = enc.encode_all(&pcm, enc_rate / 50);
                let t = toc(packets[packets.len() / 2][0]);
                if format!("{}-{}", t.mode, t.bandwidth) != name {
                    continue; // this encoder rate cannot carry the bandwidth
                }
                streams += 1;
                for (ri, &dec_rate) in RATES.iter().enumerate() {
                    let reference = c_decode_all(&packets, dec_rate, channels);
                    let label = format!("enc {enc_rate} ch{channels} {name} -> dec {dec_rate}");
                    let ours = match rs_decode_all(&packets, dec_rate, channels) {
                        Ok(v) => v,
                        Err((i, e)) => {
                            failures.push(format!("{label}: packet {i}: {e}"));
                            continue;
                        }
                    };
                    let cmp = compare(&reference, &ours, channels, dec_rate / 50, dec_rate);
                    table[mi][ri] = table[mi][ri].min(cmp.worst_frame_snr_db);
                    let exact_required = t.mode == "SILK" || (t.mode == "Hybrid" && dec_rate <= 16000);
                    let ok = if exact_required {
                        cmp.max_abs_diff == 0.0
                    } else {
                        cmp.worst_frame_snr_db > 80.0 && cmp.lag == 0
                    };
                    if !ok || ours.len() != reference.len() {
                        failures.push(format!("{label}: {cmp}"));
                    }
                }
            }
        }
    }
    println!("{streams} streams; lowest worst-frame SNR against libopus in dB (200 = bit-exact):");
    println!("{:<12} {:>7} {:>7} {:>7} {:>7} {:>7}", "stream", "8k", "12k", "16k", "24k", "48k");
    for (mi, row) in table.iter().enumerate() {
        println!(
            "{:<12} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            MODES[mi].2, row[0], row[1], row[2], row[3], row[4]
        );
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
