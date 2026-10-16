//! 10 ms SILK frames (SILK-only and Hybrid) must decode like libopus.
//!
//! A 10 ms SILK frame has two subframes and codes the pitch contour with
//! its own probability table (`silk_pitch_contour_10_ms_iCDF`, or the
//! narrowband variant). libopus `silk_decoder_set_fs` picks the table from
//! `nb_subfr`, which `silk_Decode` sets from the packet duration first.
//!
//! opus1's `silk_decoder_set_fs` forced `nb_subfr` to 4 before choosing,
//! so a 10 ms frame read its pitch contour with the 20 ms table. The range
//! decoder lost synchronisation at the first voiced frame and the rest of
//! the stream was noise: every 10 ms SILK or Hybrid stream scored about
//! -2 dB against libopus's decode of the same packets.

mod k24_common;
use k24_common::*;

/// Encode 2 s of speech in 10 ms frames with libopus and compare the two
/// decoders at `dec_rate`.
fn compare_10ms(mode: i32, bw: i32, bitrate: i32, channels: usize, dec_rate: usize) -> (String, Cmp) {
    let mono = speech_fullband(48000, 2.0);
    let pcm = if channels == 2 { to_stereo(&mono, 48000) } else { mono };
    let mut enc = CEnc::voip(48000, channels);
    enc.force_mode(mode).bandwidth(bw).bitrate(bitrate * channels as i32);
    let packets = enc.encode_all(&pcm, 480);
    let reference = c_decode_all(&packets, dec_rate, channels);
    let ours = rs_decode_all(&packets, dec_rate, channels).expect("opus1 decode");
    assert_eq!(ours.len(), reference.len());
    // Skip the first 80 ms: Hybrid streams carry a separate small start-up
    // difference in the CELT layer (see k24_celt_reset_state.rs).
    let skip = dec_rate * 80 / 1000 * channels;
    (
        toc_summary(&packets),
        compare(&reference[skip..], &ours[skip..], channels, dec_rate / 100, dec_rate),
    )
}

#[test]
fn silk_only_10ms_frames_decode_bit_exactly() {
    for channels in [1, 2] {
        for (bw, name, bitrate) in [(BW_NB, "NB", 16000), (BW_MB, "MB", 20000), (BW_WB, "WB", 24000)] {
            for dec_rate in [16000, 24000, 48000] {
                let (tocs, cmp) = compare_10ms(MODE_SILK, bw, bitrate, channels, dec_rate);
                println!("SILK-{name} ch{channels} at {dec_rate}: {tocs}: {cmp}");
                assert!(tocs.contains("10ms"), "expected 10 ms frames, got {tocs}");
                assert_eq!(
                    cmp.max_abs_diff, 0.0,
                    "SILK-{name} 10 ms, {channels} ch, decoded at {dec_rate} Hz: {cmp}"
                );
            }
        }
    }
}

#[test]
fn hybrid_10ms_frames_decode_like_libopus() {
    for channels in [1, 2] {
        for (bw, name, bitrate) in [(BW_SWB, "SWB", 32000), (BW_FB, "FB", 40000)] {
            for dec_rate in [24000, 48000] {
                let (tocs, cmp) = compare_10ms(MODE_HYBRID, bw, bitrate, channels, dec_rate);
                println!("Hybrid-{name} ch{channels} at {dec_rate}: {tocs}: {cmp}");
                assert!(tocs.contains("Hybrid") && tocs.contains("10ms"), "got {tocs}");
                assert!(
                    cmp.worst_frame_snr_db > 80.0,
                    "Hybrid-{name} 10 ms, {channels} ch, decoded at {dec_rate} Hz: {cmp}"
                );
            }
        }
    }
}
