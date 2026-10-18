//! Hybrid super-wideband packets must decode like libopus.
//!
//! Super-wideband is the widest bandwidth a 24 kHz stream can carry, and
//! Hybrid super-wideband is what both libopus and opus1's own encoder
//! produce for 24 kHz speech at voice bitrates. Such a packet codes CELT
//! bands 17 and 18 only (libopus `opus_decode_frame`: end band 19 for
//! super-wideband).
//!
//! The decoder took the end band from the ToC byte only for CELT-only
//! packets and assumed 21 bands for every Hybrid packet. Reading 21 bands
//! from a 19-band packet desynchronises the range decoder, so everything
//! above 8 kHz came out as noise. Measured against libopus's decode of the
//! same packets, worst 20 ms frame: a libopus-encoded chirp -4.6 dB, and
//! opus1's own 24 kHz VoIP encoding of full-band speech 13.4 dB. Both
//! are above 120 dB with the end band taken from the ToC.
//!
//! The fault hides on band-limited input: 16 kHz speech resampled to
//! 24 kHz has nothing above 8 kHz, and scored 96 dB even with the fault.

mod k24_common;
use k24_common::*;
use opus1::{Application, Bitrate, OpusEncoder};

/// Lowest per-frame SNR (opus1 against libopus) from frame `skip` on.
fn worst_frame_snr(packets: &[Vec<u8>], rate: usize, skip: usize) -> f64 {
    let reference = c_decode_all(packets, rate, 1);
    let ours = rs_decode_all(packets, rate, 1).expect("opus1 decode");
    assert_eq!(ours.len(), reference.len());
    frame_snrs(&reference, &ours, 1, rate / 50)
        .iter()
        .skip(skip)
        .filter(|(_, power)| *power > 1e-7)
        .map(|(snr, _)| *snr)
        .fold(f64::INFINITY, f64::min)
}

/// libopus-encoded Hybrid super-wideband, signals with energy above 8 kHz.
#[test]
fn libopus_hybrid_swb_packets_decode_like_libopus() {
    let signals: [(&str, Vec<f32>); 3] = [
        ("chirp", chirp(24000, 48000)),
        ("multitone", multitone(24000, 48000)),
        ("speech", speech_fullband(24000, 2.0)),
    ];
    for (name, pcm) in &signals {
        let mut enc = CEnc::voip(24000, 1);
        enc.force_mode(MODE_HYBRID).bandwidth(BW_SWB).bitrate(32000);
        let packets = enc.encode_all(pcm, 480);
        let t = toc(packets[10][0]);
        assert_eq!((t.mode, t.bandwidth), ("Hybrid", "SWB"), "{name}");
        for rate in [24000, 48000] {
            // The first frames are skipped: they carry a separate, small
            // start-up difference (see k24_celt_reset_state.rs).
            let worst = worst_frame_snr(&packets, rate, 4);
            println!("{name} at {rate} Hz: worst frame {worst:.1} dB");
            assert!(
                worst > 80.0,
                "{name} decoded at {rate} Hz: worst frame {worst:.1} dB against libopus"
            );
        }
    }
}

/// The owner's main path: opus1's own encoder, 24 kHz speech, VoIP,
/// 24 kb/s, 20 ms frames. It produces Hybrid super-wideband packets, and
/// opus1's decoder must read them the way libopus does.
#[test]
fn own_encoder_24k_voip_packets_decode_like_libopus() {
    // Speech with energy in the 8-12 kHz band: the plain fixture is 16 kHz
    // speech and would leave the CELT layer of the Hybrid packets empty.
    let pcm = speech_fullband(24000, 3.0);
    let mut enc = OpusEncoder::new(24000, 1, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(24000)).unwrap();
    let mut buf = [0u8; 1500];
    let packets: Vec<Vec<u8>> = pcm
        .as_chunks::<480>().0.iter()
        .map(|f| {
            let n = enc.encode(f, 480, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .collect();
    println!("own encoder: {}", toc_summary(&packets));
    assert!(
        packets.iter().any(|p| toc(p[0]).mode == "Hybrid"),
        "expected Hybrid packets from the 24 kHz VoIP encoder"
    );
    let worst = worst_frame_snr(&packets, 24000, 4);
    println!("own encoder at 24000 Hz: worst frame {worst:.1} dB");
    assert!(worst > 80.0, "worst frame {worst:.1} dB against libopus");
}
