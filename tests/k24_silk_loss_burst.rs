//! Several SILK frames lost in a row must be concealed like libopus.
//!
//! libopus `silk_PLC_conceal` repeats the last pitch period with a
//! long-term prediction gain that it lowers after every subframe. It lowers
//! the coefficients in the concealment state itself (`B_Q14` points to
//! `psPLC->LTPCoef_Q14`), so the next lost frame continues from the lowered
//! values and the concealed speech keeps fading.
//!
//! opus1 worked on a copy and discarded it, so every lost frame started
//! again from the full gain. The first lost frame was bit-exact; the second
//! in a row scored 25 dB against libopus and the third 8 dB.

mod k24_common;
use k24_common::*;
use opus1::{Channels, SampleRate};

#[test]
fn lost_frames_in_a_row_are_bit_exact() {
    // 0.4 s of speech with frames 10, 11 and 12 lost, before the stream's
    // first pause (so comfort noise plays no part yet).
    let pcm = speech(16000, 0.4);
    for (bw, name, bitrate) in [(BW_NB, "NB", 16000), (BW_WB, "WB", 24000)] {
        let mut enc = CEnc::voip(16000, 1);
        enc.force_mode(MODE_SILK).bandwidth(bw).bitrate(bitrate);
        let packets = enc.encode_all(&pcm, 320);
        let rate = 24000usize;
        let frame = rate / 50;
        let mut c_dec = CDec::new(rate, 1);
        let mut rs_dec = opus1::OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::Mono,
        )
        .unwrap();
        let mut buf = vec![0f32; frame];
        for (i, p) in packets.iter().enumerate() {
            let lost = (10..13).contains(&i);
            let reference = if lost {
                c_dec.decode(None, frame, false).unwrap()
            } else {
                c_dec.decode(Some(p), frame, false).unwrap()
            };
            if lost {
                rs_dec.decode(&[], frame, &mut buf).unwrap();
            } else {
                rs_dec.decode(p, frame, &mut buf).unwrap();
            }
            let differing = reference.iter().zip(&buf).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing, 0,
                "SILK-{name}: frame {i}{} differs from libopus in {differing} samples",
                if lost { " (lost)" } else { "" }
            );
        }
    }
}
