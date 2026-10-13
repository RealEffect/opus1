//! Stereo CELT frames of every duration must decode like libopus.
//!
//! In 2.5 ms and 5 ms frames some bands are two MDCT bins wide. libopus
//! `quant_band_stereo` codes such a band with a special case (the side
//! channel is the mid rotated by 90 degrees and costs one sign bit). The
//! phase-inversion flag still applies to it: the `if (inv)` that negates
//! the right channel sits after the special case, outside the `N != 2`
//! test.
//!
//! opus-rs returned from the special case before that step, so a two-bin
//! band with the phase-inversion flag set came out with the right channel
//! at the wrong sign. The flag is only coded in a narrow range of bit
//! budgets (about 20 bytes per frame), which is why only 2.5 ms at
//! 64 kb/s and 5 ms at 32 kb/s showed it: 16 to 29 dB in the worst 20 ms
//! against libopus's decode of the same packets.

mod k24_common;
use k24_common::*;

#[test]
fn stereo_celt_frames_of_every_duration_decode_like_libopus() {
    let pcm = to_stereo(&speech_fullband(48000, 1.5), 48000);
    let mut lowest = f64::INFINITY;
    for frame_tenth_ms in [25usize, 50, 100, 200] {
        for bitrate in [16000, 24000, 32000, 48000, 64000, 96000, 128000, 256000] {
            for (bw, name) in [(BW_NB, "NB"), (BW_WB, "WB"), (BW_FB, "FB")] {
                // Keep the stream stereo: at low bitrates libopus would
                // otherwise code stereo input as mono.
                let mut enc = CEnc::audio(48000, 2);
                enc.force_mode(MODE_CELT)
                    .bandwidth(bw)
                    .bitrate(bitrate)
                    .force_channels(2);
                let packets = enc.encode_all(&pcm, 48000 * frame_tenth_ms / 10000);
                let reference = c_decode_all(&packets, 24000, 2);
                let ours = rs_decode_all(&packets, 24000, 2).expect("opus-rs decode");
                // 20 ms comparison windows whatever the frame duration.
                let cmp = compare(&reference, &ours, 2, 480, 24000);
                lowest = lowest.min(cmp.worst_frame_snr_db);
                assert!(
                    cmp.worst_frame_snr_db > 80.0,
                    "CELT-{name} stereo, {} ms, {bitrate} b/s, at 24000 Hz: {cmp}",
                    frame_tenth_ms as f32 / 10.0
                );
            }
        }
    }
    println!("lowest worst-window SNR over all stereo streams: {lowest:.1} dB");
}
