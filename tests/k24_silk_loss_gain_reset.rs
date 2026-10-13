//! After a lost SILK packet the next good packet must decode like libopus
//! when the signal level fell during the loss.
//!
//! SILK codes the first gain of a packet as an absolute index, and the
//! decoder does not let it be more than 16 steps (about 22 dB) below the
//! last gain index it decoded (`silk_gains_dequant`). The encoder never
//! lowers the gain faster than 4 steps per 5 ms, so this limit does nothing
//! while every packet arrives.
//!
//! After a loss the decoder's last index is old. libopus therefore puts it
//! back to its start value at the end of `silk_Decode` (silk/dec_API.c:
//! "On packet loss, remove the gain clamping to prevent having the energy
//! bounce back if we lose packets when the energy is going down").
//!
//! opus-rs kept the old index. When a talker stopped during a lost packet,
//! the first good packet after it was decoded with a gain only 16 steps
//! below the last speech gain instead of its own, far lower, gain. The
//! pause after the speech came back too loud for one to three packets.
//!
//! Measured on the unfixed code with the two signals below (0.72 s of
//! speech, then digital silence or faint noise), decoder at 24 kHz, one
//! packet lost, every position tried: 17 of 690 positions differed from
//! libopus's decode, all of them with the packet at the end of the speech
//! lost. The packet after the loss was between 27.0 dB (20 ms packets) and
//! -8.7 dB (60 ms packets) against libopus. Narrowband, mediumband and
//! wideband were all affected. Every position is bit-exact now.

mod k24_common;
use k24_common::*;

/// Decode `packets` with packet `lost` missing, with both decoders. Returns
/// the first packet whose output differs from libopus and the SNR of that
/// packet against libopus, or `None` if every sample is equal.
fn first_difference(packets: &[Vec<u8>], lost: usize, frame: usize) -> Option<(usize, f64)> {
    let mut c_dec = CDec::new(24000, 1);
    let mut rs_dec = opus_rs::OpusDecoder::new(24000, 1).unwrap();
    let mut buf = vec![0f32; frame];
    for (i, p) in packets.iter().enumerate() {
        let reference = if i == lost {
            c_dec.decode(None, frame, false).unwrap()
        } else {
            c_dec.decode(Some(p), frame, false).unwrap()
        };
        let n = if i == lost {
            rs_dec.decode(&[], frame, &mut buf).unwrap()
        } else {
            rs_dec.decode(p, frame, &mut buf).unwrap()
        };
        assert_eq!(n, frame);
        if reference != buf {
            let (mut sig, mut err) = (0f64, 0f64);
            for (a, b) in reference.iter().zip(&buf) {
                sig += (*a as f64).powi(2);
                err += ((*a - *b) as f64).powi(2);
            }
            return Some((i, snr_db(sig, err)));
        }
    }
    None
}

/// Every position of a single lost packet, in streams where the level
/// falls: speech that stops and is followed by digital silence or by faint
/// noise.
#[test]
fn packet_after_a_loss_is_bit_exact_when_the_level_fell() {
    let mut then_silence = speech(16000, 0.72);
    then_silence.extend(vec![0f32; 9600]);
    let mut then_noise: Vec<f32> = speech(16000, 0.72).iter().map(|v| (v * 2.0).clamp(-1.0, 1.0)).collect();
    let mut seed = 5u32;
    then_noise.extend((0..9600).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.0002
    }));

    let mut failures = Vec::new();
    let mut checked = 0;
    for (signal, pcm) in [("speech then silence", &then_silence), ("speech then faint noise", &then_noise)] {
        for (bw, name, bitrate) in [(BW_NB, "NB", 12000), (BW_MB, "MB", 16000), (BW_WB, "WB", 24000)] {
            for ms in [20usize, 40, 60] {
                let mut enc = CEnc::voip(16000, 1);
                enc.force_mode(MODE_SILK).bandwidth(bw).bitrate(bitrate);
                let packets = enc.encode_all(pcm, 16 * ms);
                // Lose each packet in turn (never the first or the last).
                for lost in 1..packets.len() - 1 {
                    checked += 1;
                    if let Some((at, snr)) = first_difference(&packets, lost, 24 * ms) {
                        failures.push(format!(
                            "{signal}, SILK-{name} {ms} ms packets, packet {lost} lost: packet {at} is {snr:.1} dB against libopus"
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 500, "only {checked} loss positions checked");
    assert!(
        failures.is_empty(),
        "{} of {checked} loss positions do not decode like libopus:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
