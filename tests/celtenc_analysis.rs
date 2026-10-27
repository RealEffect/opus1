//! The analyses of the CELT encoder against libopus (celt_encoder.c,
//! `celt_encode_with_ec` and the functions it calls).
//!
//! Each test reads the symbols both encoders wrote for the same input and
//! counts in how many frames a field has the same value. The pitch
//! pre-filter is off (`CELT_SET_PREDICTION` 1): its period comes from the
//! pitch search in src/pitch.rs, and a different period changes the
//! signal every later analysis sees.

mod celtenc_common;

use celtenc_common::c_celt::{Agreement, Stream};
use celtenc_common::*;

/// The streams most tests look at: mono and stereo, speech and music,
/// 20 ms and 5 ms frames, constant and variable bitrate.
fn streams() -> Vec<(String, Stream, Vec<f32>)> {
    let speech = speech(48_000, 4);
    let m0 = music(48_000, 4, 0);
    let m1 = music(48_000, 4, 1);
    let late: Vec<f32> = (0..speech.len())
        .map(|i| if i >= 12 { 0.6 * speech[i - 12] } else { 0.0 })
        .collect();
    let stereo_speech = stereo_of(&speech, &late, 0.0);
    let stereo_music = stereo_of(&m0, &m1, 0.3);
    let mut out = Vec::new();
    for (name, channels, pcm) in [
        ("mono speech", 1, speech),
        ("mono music", 1, m0),
        ("stereo speech", 2, stereo_speech),
        ("stereo music", 2, stereo_music),
    ] {
        for (frame_ms_x2, per_channel, vbr, complexity) in [
            (40, 24_000, false, 9),
            (40, 48_000, true, 5),
            (10, 64_000, false, 9),
            (40, 32_000, false, 4),
        ] {
            let mut st = Stream::new(48_000, channels, frame_ms_x2, per_channel * channels as i32);
            st.vbr = vbr;
            st.complexity = complexity;
            st.prediction = 1;
            let label = format!(
                "{name}, {} ms, {} kb/s per channel, {}, complexity {complexity}",
                frame_ms_x2 as f32 / 2.0,
                per_channel / 1000,
                if vbr { "VBR" } else { "CBR" }
            );
            out.push((label, st, pcm.clone()));
        }
    }
    out
}

/// Encode every stream of `streams` with both encoders and compare.
fn compare_all() -> Vec<(String, Agreement)> {
    streams()
        .into_iter()
        .map(|(label, st, pcm)| {
            let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
            let a = st.agreement(&c, &rs);
            println!("{label}: {a:?}");
            (label, a)
        })
        .collect()
}

/// libopus `dynalloc_analysis`, which opus1 had in a simplified form:
/// - it runs before the coarse energy is quantized, so the 2.5 ms rule
///   compares with the energies of the frame before, not of this one;
/// - `last` carries over from the first channel to the second;
/// - the boosts of a CBR frame stop at 2/3 of its bytes as in libopus,
///   and the band that reaches the limit is boosted as far as the packet
///   allows;
/// - a band's boost count is not cut to its cap; the coding loop stops
///   at the cap;
/// - the VBR target gets the bits the analysis expected, not the bits that
///   were coded;
/// - the weights of the spreading decision (`spread_weight`): 32 for
///   every band before.
#[test]
fn dynamic_allocation_and_spreading_follow_libopus() {
    for (label, a) in compare_all() {
        assert!(a.at_least(a.spread, 97), "{label}: spread {} of {}", a.spread, a.frames);
        // The boosts come from the band energies, which differ slightly
        // where the MDCT of opus1 rounds differently (src/mdct.rs,
        // src/kiss_fft.rs). With the rounding of libopus there, at least
        // 95 % of the frames are the same.
        assert!(a.at_least(a.boost, 80), "{label}: boost {} of {}", a.boost, a.frames);
    }
}

/// Bursts of noise with a soft attack, mono at 48 kHz: every 250 ms the
/// level goes from -60 dB to -10 dB over `attack_ms`.
fn soft_bursts(seconds: usize, attack_ms: f64) -> Vec<f32> {
    let mut noise = Noise(9);
    (0..48_000 * seconds)
        .map(|i| {
            let t = (i % 12_000) as f64 / 48.0; // ms into the period
            let level = if t < 150.0 {
                0.001
            } else {
                let x = ((t - 150.0) / attack_ms).min(1.0);
                0.001 + 0.3 * 0.5 * (1.0 - (std::f64::consts::PI * x).cos())
            };
            (level * f64::from(noise.next())) as f32
        })
        .collect()
}

/// libopus looks once more for a transient after the band energies are
/// known: a frame whose energies are on average more than 6 dB above
/// those of the frame before is coded with short blocks
/// (`patch_transient_decision`, celt_encoder.c 2215-2232). opus1 had only
/// the time-domain analysis, which misses an attack spread over several
/// milliseconds.
#[test]
fn energy_jumps_make_a_transient() {
    for attack_ms in [30.0, 60.0] {
        let pcm = soft_bursts(4, attack_ms);
        for complexity in [5, 9] {
            let mut st = Stream::new(48_000, 1, 40, 64_000);
            st.complexity = complexity;
            st.prediction = 1;
            let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
            let count = |p: &[Vec<u8>]| {
                p.iter()
                    .filter(|x| parse_header(x, 1, 21, 3).transient)
                    .count()
            };
            let a = st.agreement(&c, &rs);
            let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
            println!(
                "attack {attack_ms} ms, complexity {complexity}: transient in {} frames \
                 (libopus {}), same flag in {} of {}; opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
                count(&rs),
                count(&c),
                a.transient,
                a.frames
            );
            assert!(count(&c) >= 12, "the signal has no transients for libopus");
            assert!(a.at_least(a.transient, 98), "{a:?}");
        }
    }
}

/// libopus tells the allocator how many bands the frames before coded
/// (`lastCodedBands`): bands below it are kept with a lower threshold, so
/// that they do not drop in and out from frame to frame. That number moves
/// by at most one band per frame (celt_encoder.c 2620-2633). The allocator
/// may keep bands up to `end - 1` (`signalBandwidth`). opus1 gave it 0
/// and 20 instead. On noise the bands at the top then dropped out in some
/// frames, and the noise bursts below came out 1.4 dB worse than from
/// libopus.
#[test]
fn coded_bands_follow_libopus() {
    for (label, a) in compare_all() {
        assert!(
            a.at_least(a.coded_bands, 97),
            "{label}: coded bands {} of {}",
            a.coded_bands,
            a.frames
        );
    }
    let pcm = soft_bursts(4, 30.0);
    let mut st = Stream::new(48_000, 1, 40, 64_000);
    st.complexity = 9;
    st.prediction = 1;
    let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
    let a = st.agreement(&c, &rs);
    let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
    println!("noise bursts: {a:?}, opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB");
    assert!(a.at_least(a.coded_bands, 97), "{a:?}");
    assert!(snr_rs > snr_c - 0.2, "{snr_rs:.2} dB against {snr_c:.2} dB");
}

/// libopus takes the logarithm of a band energy as it is (`amp2Log2`): an
/// empty band, whose energy is the square root of 1e-27, gets about -45.
/// opus1 raised the energy to 1e-10 first (-33). At 16 kHz the bands above
/// 8 kHz are empty, so opus1 spent bits on coding a level that libopus
/// does not code, and lost up to 3 dB with 2.5 ms frames.
#[test]
fn empty_bands_have_the_energy_of_libopus() {
    let pcm = music(16_000, 4, 0);
    for (frame_ms_x2, bitrate) in [(5, 48_000), (5, 64_000), (40, 32_000)] {
        let mut st = Stream::new(16_000, 1, frame_ms_x2, bitrate);
        st.complexity = 4;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let a = st.agreement(&c, &rs);
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        println!(
            "{} ms, {bitrate} b/s: coarse energy the same in {} of {} frames; \
             opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            frame_ms_x2 as f32 / 2.0,
            a.coarse,
            a.frames
        );
        assert!(a.at_least(a.coarse, 80), "{a:?}");
        assert!(snr_rs > snr_c - 0.3, "{snr_rs:.2} dB against {snr_c:.2} dB");
    }
}

/// Mono VBR streams, with the size of each packet.
fn vbr_streams(channels: usize) -> Vec<(String, Stream, Vec<f32>)> {
    streams()
        .into_iter()
        .filter(|(label, st, _)| label.contains("VBR") && st.channels == channels)
        .collect()
}

/// libopus raises the VBR target of a frame whose spectrum is above a slow
/// average of the spectra before it, and lowers it below (temporal VBR,
/// celt_encoder.c 2189-2206 and `compute_vbr`). opus1 passed 0, so its
/// packets had the size of libopus in about half or less of the frames.
#[test]
fn vbr_follows_the_spectral_level() {
    for (label, st, pcm) in vbr_streams(1) {
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let a = st.agreement(&c, &rs);
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        let bytes = |p: &[Vec<u8>]| p.iter().map(|x| x.len()).sum::<usize>() as f64 / p.len() as f64;
        println!(
            "{label}: same size in {} of {} frames, {:.1} bytes (libopus {:.1}); \
             opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            a.size,
            a.frames,
            bytes(&rs),
            bytes(&c)
        );
        assert!(a.at_least(a.size, 70), "{label}: {a:?}");
    }
}

/// The rest of libopus `compute_vbr` and of the VBR update after it
/// (celt_encoder.c 1606-1700 and 2427-2534):
/// - the target goes by the rate the allocation goes by (`equiv_rate`),
///   not by the bitrate setting;
/// - the noise-floor limit (`maxDepth`) applies to every frame; opus1
///   skipped it when the loudest band was less than 1 above its floor;
/// - a stereo saving below 0.1 raises the target;
/// - the drift of the target is updated with the float arithmetic of
///   libopus, not with a Q15 approximation of it;
/// - a frame takes at most 1275 bytes per 20 ms.
#[test]
fn vbr_target_follows_libopus() {
    for channels in [1, 2] {
        for (label, st, pcm) in vbr_streams(channels) {
            let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
            let a = st.agreement(&c, &rs);
            println!("{label}: same size in {} of {} frames", a.size, a.frames);
            let percent = if channels == 1 { 95 } else { 80 };
            assert!(a.at_least(a.size, percent), "{label}: {a:?}");
        }
    }
}

/// libopus estimates from frame to frame how many bits mid/side coding
/// saves (`st->stereo_saving`): the estimate may rise by 0.25 per frame and
/// the VBR target goes down by up to 1.0 of it (alloc_trim_analysis,
/// compute_vbr). opus1 started from 0 in every frame, so the estimate
/// never got above 0.25, and a stereo signal with two similar channels got
/// more bits than from libopus.
#[test]
fn stereo_saving_builds_up_over_frames() {
    let m = music(48_000, 4, 0);
    let late: Vec<f32> = (0..m.len())
        .map(|i| if i >= 3 { 0.9 * m[i - 3] } else { 0.0 })
        .collect();
    for pcm in [stereo_of(&m, &m, 0.0), stereo_of(&m, &late, 0.0)] {
        let mut st = Stream::new(48_000, 2, 40, 96_000);
        st.vbr = true;
        st.complexity = 9;
        st.prediction = 1;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let bytes = |p: &[Vec<u8>]| p.iter().map(|x| x.len()).sum::<usize>() as f64 / p.len() as f64;
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        let a = st.agreement(&c, &rs);
        println!(
            "{:.1} bytes per frame, libopus {:.1}; same size in {} of {}; \
             opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            bytes(&rs),
            bytes(&c),
            a.size,
            a.frames
        );
        assert!((bytes(&rs) / bytes(&c) - 1.0).abs() < 0.01);
        assert!(a.at_least(a.size, 90), "{a:?}");
    }
}

/// libopus leans the coarse energy of a band towards what its quantizer
/// left over in the frame before, when the energy of the band changed by
/// less than 2 (12 dB): "a constant offset is better than fluctuations"
/// (celt_encoder.c 2283-2297, `energyError`). opus1 did not, and wrote
/// other coarse energy steps than libopus in about half of the frames.
#[test]
fn coarse_energy_leans_towards_the_last_error() {
    for (label, a) in compare_all() {
        // A VBR frame of another size has another budget for the energy.
        // Frames of 5 ms differ in more places (time-frequency analysis).
        if label.contains("VBR") || label.contains(" 5 ms") {
            continue;
        }
        assert!(a.at_least(a.coarse, 80), "{label}: coarse {} of {}", a.coarse, a.frames);
    }
}

/// libopus `transient_analysis` sums a table of about 6*64/x over the
/// energy envelope of each channel (`inv_table`, celt_encoder.c 288-297).
/// opus1 had five entries of that table one too low: 71, 72, 83, 84 and
/// 109. The sum decides the transient flag, `tf_estimate`, and which
/// channel of a stereo frame the time-frequency analysis looks at
/// (`tf_chan`). On this stereo speech, the wrong table made opus1 look
/// at the other channel in 49 of 300 frames, and the time-frequency
/// resolution differed from libopus in 37.
///
/// Stereo speech, 48 kHz, 20 ms, 64 kb/s per channel, CBR, complexity 4
/// (no pitch pre-filter on either side).
#[test]
fn the_transient_table_is_the_one_of_libopus() {
    let speech = speech(48_000, 6);
    let late: Vec<f32> = (0..speech.len())
        .map(|i| if i >= 12 { 0.6 * speech[i - 12] } else { 0.0 })
        .collect();
    let pcm = stereo_of(&speech, &late, 0.0);
    let mut st = Stream::new(48_000, 2, 40, 128_000);
    st.complexity = 4;
    let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
    let a = st.agreement(&c, &rs);
    println!("{a:?}");
    assert_eq!(a.transient, a.frames, "transient flags");
    assert!(a.at_least(a.tf, 97), "tf {} of {}", a.tf, a.frames);
}

/// libopus moves `lastCodedBands` by at most one band per frame
/// (celt_encoder.c 2633-2636). The allocator keeps the bands below it
/// with a lower threshold, and the VBR target counts its bins. Taking the
/// coded bands of the last frame as they are lets the count jump.
///
/// Music, with broadband noise added every other 100 ms, mono, 48 kHz,
/// 20 ms, 24 kb/s, constrained VBR, complexity 9. Without the smoothing
/// the number of coded bands differed from libopus in 10 of 300 frames.
#[test]
fn the_coded_bands_move_one_band_per_frame() {
    let m = music(48_000, 6, 0);
    let mut rng = Noise(3);
    let pcm: Vec<f32> = (0..m.len())
        .map(|i| {
            if (i / 4800) % 2 == 0 {
                m[i]
            } else {
                0.2 * rng.next() + 0.3 * m[i]
            }
        })
        .collect();
    let mut st = Stream::new(48_000, 1, 40, 24_000);
    st.complexity = 9;
    st.vbr = true;
    st.prediction = 1;
    let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
    let a = st.agreement(&c, &rs);
    println!("{a:?}");
    assert!(a.at_least(a.coded_bands, 99), "coded bands {} of {}", a.coded_bands, a.frames);
}

/// Under VBR, libopus lets a frame take at most 1275 bytes per 20 ms:
/// 159 bytes for 2.5 ms, 318 for 5 ms (celt_encoder.c 2448, "Don't attempt
/// to use more than 510 kb/s, even for frames smaller than 20 ms").
#[test]
fn short_vbr_frames_stay_below_510_kbps() {
    let base = music(48_000, 2, 0);
    let pcm = stereo_of(&base, &speech(48_000, 2), 0.3);
    for (frame_ms_x2, cap) in [(5usize, 159usize), (10, 318)] {
        let mut st = Stream::new(48_000, 2, frame_ms_x2, 510_000);
        st.complexity = 9;
        st.vbr = true;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let largest = |p: &[Vec<u8>]| p.iter().map(|x| x.len()).max().unwrap_or(0);
        let same = c.iter().zip(&rs).filter(|(a, b)| a.len() == b.len()).count();
        println!(
            "{} ms: largest packet libopus {}, opus1 {}; same size in {same} of {}",
            frame_ms_x2 as f32 / 2.0,
            largest(&c),
            largest(&rs),
            c.len()
        );
        assert_eq!(largest(&c), cap);
        assert!(largest(&rs) <= cap);
        assert!(same * 100 >= c.len() * 99);
    }
}

/// The decaying output of the Opus layer's DC filter, after a signal stops.
///
/// The filter (libopus `dc_reject`, 3 Hz) keeps an estimate of the DC offset
/// of each channel. When the input becomes digital silence, the filter's
/// output is minus that estimate, which decays by 0.24 % per sample at
/// 8 kHz. It takes about half a second to fall below the silence threshold
/// of 2^-24. In that time the CELT encoder codes a signal with a peak level
/// of about 1e-3 down to 1e-7, whose two channels are scaled copies of each
/// other. At such levels the `EPSILON` terms of libopus matter:
///
/// - in the transient analysis, which adds `EPSILON` (1e-15) to the mean
///   energy of the high-passed signal and to each value of its envelope;
/// - in `intensity_stereo` (see celtenc_bands_libopus.rs).
///
/// Every packet must be libopus's: mono and stereo input, one or two coded
/// channels, 8, 16 and 48 kHz, frames of 2.5 to 20 ms, constant and
/// variable bitrate.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn the_decaying_output_of_the_dc_filter_is_coded_as_libopus_codes_it() {
    use opus1::internals::hp_cutoff::dc_reject_float;
    // A debug build (the `no_std` check runs one) codes a part of the matrix.
    let debug = cfg!(debug_assertions);
    let rates: &[usize] = if debug { &[48_000, 8_000] } else { &[48_000, 16_000, 8_000] };
    let durations: &[usize] = if debug { &[10, 40] } else { &[5, 10, 20, 40] };
    let settings: &[(i32, bool)] =
        if debug { &[(64_000, false), (48_000, true)] } else { &[(24_000, false), (64_000, false), (160_000, false), (48_000, true)] };
    let mut frames = 0;
    for &fs in rates {
        // 0.3 s of music, then 1.7 s of digital silence, twice.
        let mut left = music(fs, 4, 0);
        let mut right = music(fs, 4, 1);
        for x in [&mut left, &mut right] {
            for (i, v) in x.iter_mut().enumerate() {
                if i % (2 * fs) >= 3 * fs / 10 {
                    *v = 0.0;
                }
            }
        }
        for channels in [1usize, 2] {
            let raw: Vec<f32> = if channels == 1 {
                left.clone()
            } else {
                left.iter().zip(&right).flat_map(|(&l, &r)| [l, r]).collect()
            };
            for &frame_ms_x2 in durations {
                // The input as the Opus layer gives it to CELT.
                let frame = fs * frame_ms_x2 / 2000;
                let mut pcm = vec![0f32; raw.len()];
                let mut mem = [0f32; 4];
                for (i, o) in raw.chunks(frame * channels).zip(pcm.chunks_mut(frame * channels)) {
                    dc_reject_float(i, 3, o, &mut mem, frame, channels, fs as i32);
                }
                for stream_channels in 1..=channels {
                    for &(bitrate, vbr) in settings {
                        let mut st = Stream::new(fs, channels, frame_ms_x2, bitrate * channels as i32);
                        st.stream_channels = stream_channels;
                        st.vbr = vbr;
                        st.complexity = 5;
                        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
                        for (k, (a, b)) in c.iter().zip(&rs).enumerate() {
                            assert!(
                                a == b,
                                "{fs} Hz, {channels} ch coded as {stream_channels}, frame {frame_ms_x2}/2 ms, \
                                 {bitrate} b/s per channel, vbr {vbr}: frame {k} differs"
                            );
                        }
                        frames += c.len();
                    }
                }
            }
        }
    }
    println!("{frames} frames, all equal to libopus's");
}
