//! Stereo decisions of the CELT encoder against libopus
//! (celt_encoder.c, `celt_encode_with_ec`).

mod celtenc_common;

use celtenc_common::c_celt::Stream;
use celtenc_common::*;

/// The allocation of every frame of a stream.
fn allocations(st: &Stream, payloads: &[Vec<u8>]) -> Vec<Allocation> {
    let lm = st.format().lm();
    payloads
        .iter()
        .map(|p| {
            parse_frame(p, st.stream_channels, st.end_band, lm)
                .1
                .unwrap_or_default()
        })
        .collect()
}

/// Two different instruments, one per channel, at 48 kHz.
fn two_sources(seconds: usize) -> Vec<f32> {
    stereo_of(&music(48_000, seconds, 0), &music(48_000, seconds, 1), 0.0)
}

/// The bit allocator of libopus lowers the encoder's intensity band to the
/// number of coded bands, or sets it to 0 when the packet has no room to
/// signal it (rate.c, interp_bits2pulses; celt_encoder.c passes
/// `&st->intensity`). The hysteresis of the next frame starts from that
/// value. opus1 gave the allocator a copy, so its next frame started from
/// the unlowered band.
#[test]
fn the_allocator_lowers_the_intensity_band_for_the_next_frame() {
    let pcm = stereo_of(&music(48_000, 6, 0), &music(48_000, 6, 1), 0.3);
    let mut st = Stream::new(48_000, 2, 5, 64_000);
    st.complexity = 4;
    st.prediction = 1;
    let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
    let a = st.agreement(&c, &rs);
    let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
    println!(
        "intensity band and dual stereo the same in {} of {} frames; \
         opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
        a.stereo, a.frames
    );
    assert!(a.at_least(a.stereo, 80), "{a:?}");
}

/// libopus decides between left/right and mid/side coding with
/// `stereo_analysis`, but only for frames longer than 2.5 ms: "Always use MS
/// for 2.5 ms frames until we can do a better analysis" (celt_encoder.c
/// 2401). opus1 ran the analysis for 2.5 ms frames too, and coded some
/// frames of two unrelated channels as left and right.
#[test]
fn frames_of_2_5_ms_are_coded_as_mid_and_side() {
    let pcm = two_sources(3);
    for per_channel in [64_000, 96_000] {
        let st = Stream::new(48_000, 2, 5, 2 * per_channel);
        let (c_bytes, rs_bytes) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let c = allocations(&st, &c_bytes);
        let rs = allocations(&st, &rs_bytes);
        let dual = |a: &[Allocation]| a.iter().filter(|x| x.dual_stereo).count();
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs_bytes), st.snr_db(&pcm, &c_bytes));
        println!(
            "{per_channel} b/s per channel: left/right in {} frames of {}, libopus {}; \
             opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            dual(&rs),
            rs.len(),
            dual(&c)
        );
        assert_eq!(dual(&c), 0);
        assert_eq!(dual(&rs), 0);
        assert!(snr_rs > snr_c - 0.5, "{snr_rs:.2} dB against {snr_c:.2} dB");
    }

    // Longer frames do use left/right coding for this signal, in both
    // encoders.
    let st = Stream::new(48_000, 2, 40, 2 * 64_000);
    let c = allocations(&st, &st.enc_c(&pcm));
    let rs = allocations(&st, &st.enc_rs(&pcm));
    let dual = |a: &[Allocation]| a.iter().filter(|x| x.dual_stereo).count();
    println!("20 ms: left/right in {} frames of {}, libopus {}", dual(&rs), rs.len(), dual(&c));
    assert!(dual(&c) * 2 > c.len());
    assert!(dual(&rs) * 2 > rs.len());
}
