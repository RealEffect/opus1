//! The CELT part of Hybrid frames (start band 17) against libopus.
//!
//! Both encoders code the CELT part on its own here, without the SILK
//! part that a Hybrid packet has in front of it. The C encoder gets
//! `CELT_SET_START_BAND(17)`. Such a frame cannot be decoded alone, so the
//! tests compare the symbols both encoders wrote.

mod celtenc_common;

use celtenc_common::c_celt::Stream;
use celtenc_common::*;

fn headers(st: &Stream, payloads: &[Vec<u8>]) -> Vec<Header> {
    let lm = st.format().lm();
    payloads
        .iter()
        .map(|p| parse_header_from(p, st.stream_channels, st.start_band, st.end_band, lm))
        .collect()
}

/// libopus codes the allocation trim of a Hybrid frame as 5, without
/// `alloc_trim_analysis` (celt_encoder.c 2414). opus1 ran the analysis
/// on the two to four bands of the frame.
#[test]
fn hybrid_frames_have_the_neutral_trim() {
    let pcm = music(48_000, 3, 0);
    // Stereo needs the intensity band limited to start..end as well
    // (celt_encoder.c 2408); that line is corrected on its own.
    for (channels, bitrate) in [(1, 16_000), (1, 32_000), (1, 64_000)] {
        let pcm = if channels == 2 {
            stereo_of(&pcm, &music(48_000, 3, 1), 0.3)
        } else {
            pcm.clone()
        };
        let mut st = Stream::new(48_000, channels, 40, bitrate);
        st.start_band = 17;
        st.complexity = 9;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let (hc, hr) = (headers(&st, &c), headers(&st, &rs));
        let trims = |h: &[Header]| h.iter().filter(|x| x.trim == Some(5)).count();
        let same = hc.iter().zip(&hr).filter(|(a, b)| a == b).count();
        let same_bytes = c.iter().zip(&rs).filter(|(a, b)| a == b).count();
        println!(
            "{channels} channels, {bitrate} b/s: trim 5 in {} frames (libopus {}) of {}; \
             same header in {same}, same bytes in {same_bytes}",
            trims(&hr),
            trims(&hc),
            hc.len()
        );
        assert_eq!(trims(&hc), hc.len());
        assert_eq!(trims(&hr), hr.len());
        assert!(same * 100 >= hc.len() * 95, "same header in {same} frames");
    }
}
