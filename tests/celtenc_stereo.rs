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

/// Stereo input whose right channel is the left one times a constant (one
/// source panned to one side). Every band then has the angle 0, and libopus
/// codes it from the mix of `intensity_stereo` (celt/bands.c, lines
/// 379-403). Tones from -134 to -20 dBFS, 128 to 750 kb/s per channel,
/// frames of 2.5 to 20 ms, constant and variable bitrate: every packet is
/// libopus's.
///
/// Before `intensity_stereo` added libopus's `EPSILON` terms and fused its
/// products as clang does, 30 of these 103,500 packets differed, in 4
/// streams (at -110 dBFS and at -40 dBFS).
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn a_source_panned_to_one_side_is_coded_as_libopus_codes_it() {
    let fs = 48_000usize;
    // A debug build (the `no_std` check runs one) codes the levels and
    // settings where packets differed before, and fewer of the others.
    let debug = cfg!(debug_assertions);
    let levels: &[f64] = if debug { &[-110.0, -40.0] } else { &[-134.0, -110.0, -90.0, -70.0, -40.0, -20.0] };
    let tones: &[(f64, f32)] = if debug { &[(440.0, 0.5), (1000.0, 0.25)] } else { &[(440.0, 0.5), (1000.0, 0.25), (3000.0, 0.7)] };
    let frames: &[usize] = if debug { &[10, 40] } else { &[5, 10, 20, 40] };
    let rates: &[i32] = if debug { &[128, 510] } else { &[128, 256, 510, 750] };
    let mut packets = 0;
    for &db in levels {
        for &(freq, gain_right) in tones {
            let amp = 10f64.powf(db / 20.0);
            let pcm: Vec<f32> = (0..fs)
                .flat_map(|i| {
                    let left = (amp * (std::f64::consts::TAU * freq * i as f64 / fs as f64).sin()) as f32;
                    [left, gain_right * left]
                })
                .collect();
            for &frame_ms_x2 in frames {
                for &kbps in rates {
                    for vbr in [false, true] {
                        let mut st = Stream::new(fs, 2, frame_ms_x2, kbps * 1000 * 2);
                        st.vbr = vbr;
                        if !vbr && st.cbr_bytes() > 1275 {
                            continue;
                        }
                        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
                        for (k, (a, b)) in c.iter().zip(&rs).enumerate() {
                            assert!(
                                a == b,
                                "{db} dBFS, {freq} Hz, right = {gain_right} x left, frame {frame_ms_x2}/2 ms, \
                                 {kbps} kb/s per channel, vbr {vbr}: frame {k} differs"
                            );
                        }
                        packets += c.len();
                    }
                }
            }
        }
    }
    println!("{packets} packets, all equal to libopus's");
}
