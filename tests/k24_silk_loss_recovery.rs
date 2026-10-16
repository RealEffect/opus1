//! The first good SILK frame after a lost one must decode like libopus.
//!
//! In that frame libopus widens the bandwidth of the prediction filter by
//! the factor `BWE_AFTER_LOSS_Q16` = 63570 (0.97 in Q16, silk/define.h),
//! which damps the filter while the decoder state recovers.
//!
//! opus1 had 64738 (0.988). The filter coefficients of that frame were
//! therefore different, and so was every sample of it and, through the
//! decoder state, the frames after it: 9 to 16 dB against libopus in the
//! frame after each loss, although the concealed frame itself was
//! bit-exact.

mod k24_common;
use k24_common::*;

#[test]
fn frame_after_a_single_loss_is_bit_exact() {
    // 0.6 s of speech: three single losses, all before the stream's first
    // pause (so comfort noise plays no part yet).
    let pcm = speech(16000, 0.6);
    for (bw, name, bitrate) in [(BW_NB, "NB", 16000), (BW_MB, "MB", 20000), (BW_WB, "WB", 24000)] {
        let mut enc = CEnc::voip(16000, 1);
        enc.force_mode(MODE_SILK).bandwidth(bw).bitrate(bitrate);
        let packets = enc.encode_all(&pcm, 320);
        let lost: Vec<bool> = (0..packets.len()).map(|i| i % 10 == 7).collect();
        for rate in [16000usize, 24000] {
            let frame = rate / 50;
            let mut c_dec = CDec::new(rate, 1);
            let mut rs_dec = opus1::OpusDecoder::new(rate as i32, 1).unwrap();
            let mut buf = vec![0f32; frame];
            for (i, p) in packets.iter().enumerate() {
                let reference = if lost[i] {
                    c_dec.decode(None, frame, false).unwrap()
                } else {
                    c_dec.decode(Some(p), frame, false).unwrap()
                };
                let n = if lost[i] {
                    rs_dec.decode(&[], frame, &mut buf).unwrap()
                } else {
                    rs_dec.decode(p, frame, &mut buf).unwrap()
                };
                assert_eq!(n, frame);
                let differing = reference.iter().zip(&buf).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differing,
                    0,
                    "SILK-{name} at {rate} Hz: frame {i} ({}) differs from libopus in {differing} samples",
                    if lost[i] {
                        "lost"
                    } else if i > 0 && lost[i - 1] {
                        "first good frame after a loss"
                    } else {
                        "good"
                    }
                );
            }
        }
    }
}
