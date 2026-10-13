//! A stream that switches between SILK-only, Hybrid and CELT-only must
//! decode like libopus through every switch.
//!
//! libopus `opus_decode_frame` treats the three kinds of switch differently:
//!
//! - Between SILK-only and Hybrid nothing special happens: the SILK layer
//!   continues and the CELT layer starts from a reset state (or, when it
//!   stops, fades out through 2.5 ms of CELT silence).
//! - Next to a switch to or from CELT-only the encoder appends a 5 ms
//!   full-band CELT frame (the "redundancy" frame) to the SILK-only or
//!   Hybrid frame. The decoder cross-fades it with the main frame.
//! - A switch to or from CELT-only without that frame is bridged with 5 ms
//!   of concealment in the previous mode.
//!
//! opus-rs treated every change of mode as the third kind, and skipped the
//! CELT layer of a frame that carried a redundancy frame instead of
//! decoding the redundancy frame. The concealment call also advanced the
//! SILK state as if a packet had been lost. Result: 0.5 to 12 dB against
//! libopus in the frame of each switch, and wrong output for several frames
//! after it, including the plain SILK-only to Hybrid switch that an encoder
//! in automatic mode makes whenever its bitrate allows more bandwidth.

mod k24_common;
use k24_common::*;

/// Encode with libopus, changing the forced mode and bandwidth every
/// 400 ms, and return the packets.
fn switching_stream(enc_rate: usize, channels: usize, plan: &[(i32, i32)], frame_ms: usize) -> Vec<Vec<u8>> {
    let mono = speech_fullband(enc_rate, 0.4 * plan.len() as f64);
    let pcm = if channels == 2 { to_stereo(&mono, enc_rate) } else { mono };
    let mut enc = CEnc::audio(enc_rate, channels);
    enc.bitrate(40000 * channels as i32).force_channels(channels as i32);
    let frame = enc_rate / 1000 * frame_ms;
    let per_step = 400 / frame_ms;
    pcm.chunks_exact(frame * channels)
        .enumerate()
        .map(|(i, f)| {
            let (mode, bw) = plan[(i / per_step) % plan.len()];
            enc.force_mode(mode).bandwidth(bw);
            enc.encode(f)
        })
        .collect()
}

/// Number of places where consecutive packets differ in mode or bandwidth.
fn count_switches(packets: &[Vec<u8>]) -> usize {
    packets
        .windows(2)
        .filter(|w| {
            let (a, b) = (toc(w[0][0]), toc(w[1][0]));
            a.mode != b.mode || a.bandwidth != b.bandwidth
        })
        .count()
}

fn check(packets: &[Vec<u8>], channels: usize, dec_rate: usize, frame_ms: usize, label: &str) {
    let reference = c_decode_all(packets, dec_rate, channels);
    let ours = rs_decode_all(packets, dec_rate, channels).expect("opus-rs decode");
    assert_eq!(ours.len(), reference.len(), "{label}");
    let cmp = compare(&reference, &ours, channels, dec_rate / 1000 * frame_ms, dec_rate);
    println!("{label}: {cmp}");
    assert!(cmp.worst_frame_snr_db > 80.0, "{label}: {cmp}");
}

/// The modes a 24 kHz encoder can produce: SILK up to wideband, Hybrid and
/// CELT up to super-wideband.
#[test]
fn switches_in_a_24k_stream_decode_like_libopus() {
    let plan = [
        (MODE_SILK, BW_WB),
        (MODE_HYBRID, BW_SWB),
        (MODE_CELT, BW_SWB),
        (MODE_HYBRID, BW_SWB),
        (MODE_SILK, BW_WB),
        (MODE_CELT, BW_SWB),
        (MODE_SILK, BW_NB),
        (MODE_CELT, BW_WB),
        (MODE_SILK, BW_MB),
        (MODE_HYBRID, BW_SWB),
    ];
    for channels in [1usize, 2] {
        for frame_ms in [10usize, 20] {
            let packets = switching_stream(24000, channels, &plan, frame_ms);
            assert!(count_switches(&packets) >= 9, "stream must switch: {}", toc_summary(&packets));
            for dec_rate in [16000usize, 24000, 48000] {
                check(
                    &packets,
                    channels,
                    dec_rate,
                    frame_ms,
                    &format!("24 kHz stream, {channels} ch, {frame_ms} ms, decoded at {dec_rate}"),
                );
            }
        }
    }
}

/// The same with a 48 kHz encoder, which adds the fullband modes.
#[test]
fn switches_in_a_48k_stream_decode_like_libopus() {
    let plan = [
        (MODE_SILK, BW_WB),
        (MODE_HYBRID, BW_FB),
        (MODE_CELT, BW_FB),
        (MODE_HYBRID, BW_SWB),
        (MODE_SILK, BW_MB),
        (MODE_CELT, BW_NB),
        (MODE_HYBRID, BW_FB),
        (MODE_SILK, BW_NB),
        (MODE_CELT, BW_SWB),
        (MODE_SILK, BW_WB),
    ];
    for channels in [1usize, 2] {
        let packets = switching_stream(48000, channels, &plan, 20);
        assert!(count_switches(&packets) >= 9, "stream must switch: {}", toc_summary(&packets));
        for dec_rate in [24000usize, 48000] {
            check(
                &packets,
                channels,
                dec_rate,
                20,
                &format!("48 kHz stream, {channels} ch, decoded at {dec_rate}"),
            );
        }
    }
}
