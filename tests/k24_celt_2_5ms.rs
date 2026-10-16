//! 2.5 ms CELT frames must decode like libopus.
//!
//! In a 2.5 ms frame the lowest eight bands are one MDCT bin wide. Such a
//! band codes only a sign (libopus `quant_band_n1`), and its value is also
//! stored as source material for filling higher bands that received no
//! bits ("folding"). libopus stores it as `SHR32(X[0], 4)`, and in the
//! floating-point build `SHR32` does nothing: the stored value is +-1.
//!
//! opus1 stored +-1/16, the fixed-point reading of that line. Bands
//! folded from a mix of one-bin bands and wider bands then got the wrong
//! shape. Only 2.5 ms frames have one-bin bands, so 5, 10 and 20 ms frames
//! were unaffected. Against libopus's decode of the same packets, mono
//! 2.5 ms streams between 9 and 48 kb/s scored 24 to 65 dB over the whole
//! stream and 9 to 32 dB in their worst 2.5 ms frame.

mod k24_common;
use k24_common::*;

#[test]
fn celt_2_5ms_frames_decode_like_libopus() {
    let pcm = speech_fullband(48000, 1.5);
    let mut lowest = f64::INFINITY;
    // Mono only: stereo short frames are covered, together with a second
    // fault they had, in k24_celt_stereo_short_frames.rs.
    {
        let channels = 1usize;
        for (bw, name) in [(BW_NB, "NB"), (BW_WB, "WB"), (BW_SWB, "SWB"), (BW_FB, "FB")] {
            for bitrate in [9000, 12000, 16000, 24000, 32000, 48000, 96000] {
                for cbr in [false, true] {
                    let mut enc = CEnc::audio(48000, channels);
                    enc.force_mode(MODE_CELT).bandwidth(bw).bitrate(bitrate).vbr(!cbr);
                    let packets = enc.encode_all(&pcm, 120);
                    assert_eq!(toc(packets[10][0]).frame_tenth_ms, 25);
                    for rate in [24000usize, 48000] {
                        let reference = c_decode_all(&packets, rate, channels);
                        let ours = rs_decode_all(&packets, rate, channels).expect("opus1 decode");
                        // Compare in 20 ms windows: a 2.5 ms frame is too
                        // short for a meaningful per-frame ratio in quiet
                        // passages.
                        let cmp = compare(&reference, &ours, channels, rate / 50, rate);
                        lowest = lowest.min(cmp.worst_frame_snr_db);
                        assert!(
                            cmp.worst_frame_snr_db > 80.0,
                            "CELT-{name} 2.5 ms, {bitrate} b/s per channel, {} ch{channels}, at {rate} Hz: {cmp}",
                            if cbr { "CBR" } else { "VBR" }
                        );
                    }
                }
            }
        }
    }
    println!("lowest worst-window SNR over all 2.5 ms streams: {lowest:.1} dB");
}
