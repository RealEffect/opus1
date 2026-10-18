//! The CELT decoder's band-energy history must start, and be cleared, the
//! way libopus does it.
//!
//! libopus keeps the log-energy of each band for the two previous frames
//! (`oldLogE`, `oldLogE2`). Two rules were missing in opus1:
//!
//! 1. After creation or a reset both histories are -28 (silence), not 0.
//!    The anti-collapse step fills a band that lost its energy in a
//!    transient frame with noise, scaled by how much louder the band is
//!    than its history. With a history of 0 the first transient frame got
//!    audible noise (about 40 dB below the signal) that libopus does not
//!    produce, and the post-filter kept it ringing for about six frames.
//!    Speech always starts with a transient frame.
//! 2. After every frame the bands outside the coded range are cleared
//!    (energy 0, history -28). Without this, a stream that widens its
//!    bandwidth predicts the newly coded bands from stale energies.

mod k24_common;
use k24_common::*;

/// Per-frame SNR of opus1 against libopus for one packet stream.
fn snrs(packets: &[Vec<u8>], rate: usize, channels: usize, frame: usize) -> Vec<f64> {
    let reference = c_decode_all(packets, rate, channels);
    let ours = rs_decode_all(packets, rate, channels).expect("opus1 decode");
    assert_eq!(ours.len(), reference.len());
    frame_snrs(&reference, &ours, channels, frame)
        .iter()
        .map(|(snr, power)| if *power > 1e-9 { *snr } else { 200.0 })
        .collect()
}

/// CELT-only and Hybrid streams from their very first frame.
#[test]
fn first_frames_match_libopus() {
    let cases = [
        (MODE_CELT, BW_FB, 64000, "CELT-FB"),
        (MODE_CELT, BW_SWB, 48000, "CELT-SWB"),
        (MODE_CELT, BW_WB, 40000, "CELT-WB"),
        (MODE_CELT, BW_NB, 32000, "CELT-NB"),
        (MODE_HYBRID, BW_SWB, 32000, "Hybrid-SWB"),
        (MODE_HYBRID, BW_FB, 40000, "Hybrid-FB"),
    ];
    let mono = speech_fullband(48000, 1.0);
    for channels in [1usize, 2] {
        let pcm = if channels == 2 { to_stereo(&mono, 48000) } else { mono.clone() };
        for (mode, bw, bitrate, name) in cases {
            for frame_ms in [10usize, 20] {
                let mut enc = CEnc::audio(48000, channels);
                enc.force_mode(mode).bandwidth(bw).bitrate(bitrate * channels as i32);
                let packets = enc.encode_all(&pcm, 48 * frame_ms);
                for rate in [24000usize, 48000] {
                    let s = snrs(&packets, rate, channels, rate / 1000 * frame_ms);
                    let (worst_at, worst) = s
                        .iter()
                        .enumerate()
                        .fold((0, f64::INFINITY), |m, (i, v)| if *v < m.1 { (i, *v) } else { m });
                    println!(
                        "{name} {frame_ms} ms ch{channels} at {rate}: first frames {:?}, worst {worst:.1} dB (frame {worst_at})",
                        s.iter().take(4).map(|v| v.round()).collect::<Vec<_>>()
                    );
                    assert!(
                        worst > 80.0,
                        "{name} {frame_ms} ms, {channels} ch, at {rate} Hz: frame {worst_at} is {worst:.1} dB against libopus"
                    );
                }
            }
        }
    }
}

/// A CELT-only stream whose bandwidth changes every 200 ms.
#[test]
fn bandwidth_changes_match_libopus() {
    let pcm = speech_fullband(48000, 2.0);
    let plan = [BW_FB, BW_NB, BW_FB, BW_WB, BW_SWB, BW_NB, BW_SWB, BW_FB, BW_WB, BW_FB];
    let mut enc = CEnc::audio(48000, 1);
    enc.force_mode(MODE_CELT).bitrate(64000);
    let packets: Vec<Vec<u8>> = pcm
        .as_chunks::<960>().0.iter()
        .enumerate()
        .map(|(i, f)| {
            enc.bandwidth(plan[(i / 10) % plan.len()]);
            enc.encode(f)
        })
        .collect();
    println!("{}", toc_summary(&packets));
    for rate in [24000usize, 48000] {
        let s = snrs(&packets, rate, 1, rate / 50);
        let (worst_at, worst) = s
            .iter()
            .enumerate()
            .fold((0, f64::INFINITY), |m, (i, v)| if *v < m.1 { (i, *v) } else { m });
        println!("bandwidth changes at {rate}: worst {worst:.1} dB (frame {worst_at})");
        assert!(worst > 80.0, "at {rate} Hz: frame {worst_at} is {worst:.1} dB against libopus");
    }
}
