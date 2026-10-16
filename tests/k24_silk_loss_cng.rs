//! Lost SILK frames and discontinuous transmission must decode like
//! libopus over whole streams, at a 24 kHz output rate.
//!
//! Two things that only show after a stream has had a pause:
//!
//! 1. Comfort noise. libopus `silk_CNG` learns the background noise from
//!    frames without voice activity and adds noise of that spectrum and
//!    level to every concealed frame. With discontinuous transmission
//!    (DTX) the encoder stops sending during silence and the decoder
//!    conceals for seconds, so this noise is all the listener hears.
//!    opus1 had no comfort noise: concealed frames after the first pause
//!    scored about 4 dB against libopus, and the frames after a DTX gap
//!    15 dB.
//! 2. Voiced concealment followed by an unvoiced frame. libopus decodes the
//!    first two subframes of that frame as voiced, with one prediction tap
//!    of 0.25 at the concealment's pitch lag, to avoid an abrupt change.
//!    opus1 switched the frame type but left the taps at zero (14 dB in
//!    the affected frame).

mod k24_common;
use k24_common::*;

/// Decode with both decoders, `lost[i]` marking lost packets, and return
/// the number of differing samples and the index of the first frame that
/// differs.
fn compare_with_losses(packets: &[Vec<u8>], lost: &[bool], rate: usize, channels: usize) -> (usize, Option<usize>) {
    let frame = rate / 50;
    let mut c_dec = CDec::new(rate, channels);
    let mut rs_dec = opus1::OpusDecoder::new(rate as i32, channels).unwrap();
    let mut buf = vec![0f32; frame * channels];
    let (mut differing, mut first) = (0, None);
    for (i, p) in packets.iter().enumerate() {
        let reference = if lost[i] {
            c_dec.decode(None, frame, false).unwrap()
        } else {
            c_dec.decode(Some(p), frame, false).unwrap()
        };
        if lost[i] {
            rs_dec.decode(&[], frame, &mut buf).unwrap();
        } else {
            rs_dec.decode(p, frame, &mut buf).unwrap();
        }
        let d = reference.iter().zip(&buf).filter(|(a, b)| a != b).count();
        if d > 0 && first.is_none() {
            first = Some(i);
        }
        differing += d;
    }
    (differing, first)
}

/// 3 s of speech (with pauses) and two loss patterns.
#[test]
fn losses_in_long_silk_streams_are_bit_exact() {
    let pcm = speech(16000, 3.0);
    for (bw, name, bitrate) in [(BW_NB, "NB", 16000), (BW_MB, "MB", 20000), (BW_WB, "WB", 24000)] {
        let mut enc = CEnc::voip(16000, 1);
        enc.force_mode(MODE_SILK).bandwidth(bw).bitrate(bitrate);
        let packets = enc.encode_all(&pcm, 320);
        let patterns: [(&str, Vec<bool>); 2] = [
            ("1 lost in 10", (0..packets.len()).map(|i| i % 10 == 7).collect()),
            ("3 lost in a row every 20", (0..packets.len()).map(|i| (10..13).contains(&(i % 20))).collect()),
        ];
        for (pattern, lost) in &patterns {
            let (differing, first) = compare_with_losses(&packets, lost, 24000, 1);
            assert_eq!(
                differing, 0,
                "SILK-{name}, {pattern}: {differing} samples differ from libopus, first in frame {first:?}"
            );
        }
    }
}

/// Speech, two seconds of low-level noise, speech again, encoded by libopus
/// with DTX: during the noise it sends almost nothing.
#[test]
fn dtx_stream_is_bit_exact() {
    let mut pcm = speech(16000, 1.0);
    let mut seed = 99u32;
    pcm.extend((0..32000).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.002
    }));
    pcm.extend(speech(16000, 1.0));
    for (bw, name, bitrate) in [(BW_NB, "NB", 16000), (BW_WB, "WB", 24000)] {
        let mut enc = CEnc::voip(16000, 1);
        enc.force_mode(MODE_SILK).bandwidth(bw).bitrate(bitrate).dtx(true);
        let packets = enc.encode_all(&pcm, 320);
        let dtx_packets = packets.iter().filter(|p| p.len() <= 2).count();
        assert!(dtx_packets > 10, "expected DTX packets, got {dtx_packets}");
        let reference = c_decode_all(&packets, 24000, 1);
        let ours = rs_decode_all(&packets, 24000, 1).expect("opus1 decode");
        let cmp = compare(&reference, &ours, 1, 480, 24000);
        println!("SILK-{name} DTX, {dtx_packets} of {} packets empty: {cmp}", packets.len());
        assert_eq!(cmp.max_abs_diff, 0.0, "SILK-{name} DTX: {cmp}");
        // The comfort noise must be there: the DTX stretch is not silent.
        let gap = &reference[24000 * 3 / 2..24000 * 5 / 2];
        let power: f64 = gap.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / gap.len() as f64;
        assert!(power > 1e-9, "libopus DTX stretch is silent; the test would prove nothing");
    }
}
