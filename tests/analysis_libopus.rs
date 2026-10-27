//! The speech and music analysis against `run_analysis` of libopus, called
//! directly.
//!
//! Every test runs the libopus analysis (the Homebrew build of libopus
//! 1.6.1 that the test suite links) and the opus1 analysis on the same
//! input with the same arguments. After every call it checks that the
//! returned `AnalysisInfo` and the whole state (every field of
//! `TonalityAnalysisState`, the ring of 100 results included) are the same
//! bits. See `analysis_common` for how the C functions are reached.
//!
//! The arithmetic of the analysis differs between builds of libopus. opus1
//! follows the Homebrew build on 64-bit ARM, so the tests run only there.
//! On other targets opus1 rounds products separately, as libopus builds
//! without fused multiply-add do, and no such build is linked here.
#![cfg(all(feature = "std", feature = "analysis", target_arch = "aarch64"))]
// With `test-libopus-dnn` the tests link a libopus built with its neural
// network code and without vector loops. Its analysis rounds two sums of
// the cepstrum differently from the Homebrew build that opus1 follows
// (docs/guide/10-differences-from-libopus.md), so these tests run only without
// that feature.
#![cfg(not(feature = "test-libopus-dnn"))]

mod analysis_common;

use analysis_common::*;

const RATES: [i32; 3] = [16000, 24000, 48000];

/// The frame durations of libopus, in steps of 2.5 ms: 2.5, 5, 10, 20, 40,
/// 60, 80, 100 and 120 ms.
const DURATIONS: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];

/// Samples of `steps` times 2.5 ms.
fn frame(fs: i32, steps: usize) -> usize {
    fs as usize / 400 * steps
}

/// The mixed test signal, about 7 s: speech, music, digital silence, tones,
/// noise, speech again, clicks, a sweep and very quiet noise.
fn mixed(fs: i32) -> Vec<f32> {
    concat(&[
        speech(fs, 2.0),
        music(fs, 1.5, 7),
        vec![0.0; fs as usize / 4],
        tones(fs, 0.5, &[440.0, 1250.0, 3150.0], 0.5),
        noise(fs, 0.5, 0.05, 3),
        speech(fs, 0.7),
        clicks(fs, 0.4, fs as usize / 25),
        chirp(fs, 0.6, 0.4),
        noise(fs, 0.3, 1e-4, 11),
    ])
}

/// The C structs have the sizes of libopus, and `new` gives the state of
/// `tonality_analysis_init`, also for rates where the analysis does not
/// run.
#[test]
fn state_layout_and_init_match_libopus() {
    assert_eq!(std::mem::size_of::<CInfo>(), 64);
    assert_eq!(std::mem::size_of::<CState>(), 14004);
    for fs in [8000, 12000, 16000, 24000, 48000] {
        let p = Pair::new(fs, format!("init {fs}"));
        assert!(!p.rs.initialized());
    }
}

/// `is_digital_silence` against the libopus function, also with NaN, which
/// the ternaries of libopus handle in their own way.
#[test]
fn digital_silence_matches_libopus() {
    use opus1::internals::analysis::is_digital_silence;
    let mut rng = Lcg(77);
    let specials = [0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e-30, -1e-30];
    for trial in 0..20000 {
        let n = 1 + rng.below(40) as usize;
        let lsb = 8 + rng.below(17) as i32;
        let thr = 1.0 / (1i32 << lsb) as f32;
        let x: Vec<f32> = (0..n)
            .map(|_| match rng.below(6) {
                0 => specials[rng.below(7) as usize],
                1 => thr,
                2 => -thr,
                3 => thr * (1.0 + 1e-7),
                4 => thr * rng.uniform(),
                _ => rng.uniform(),
            })
            .collect();
        let c = unsafe { c_is_digital_silence(x.as_ptr(), n as i32, 1, lsb) } != 0;
        assert_eq!(is_digital_silence(&x, lsb), c, "trial {trial}: {x:?}, lsb {lsb}");
    }
}

#[test]
fn speech_at_48_khz_mono_20_ms() {
    let fs = 48000;
    let mut p = Pair::new(fs, "48 kHz mono speech");
    p.feed(&Pcm::F32(speech(fs, 6.0)), Layout::encoder(1), 960, 24);
    assert!(p.valid > 250, "valid results: {}", p.valid);
}

/// Every input rate and frame duration, mono and stereo, float and 16-bit
/// input.
#[test]
fn every_rate_duration_and_input_type() {
    for fs in RATES {
        let sig = mixed(fs);
        let stereo = interleave(&sig, &[0.8, -0.5]);
        for d in DURATIONS {
            let n = frame(fs, d);
            for (channels, x) in [(1usize, &sig), (2, &stereo)] {
                let mut p = Pair::new(fs, format!("{fs} Hz, {channels} ch, {n} samples, float"));
                p.feed(&Pcm::F32(x.clone()), Layout::encoder(channels), n, 24);
                assert!(p.valid > 0);

                let mut p = Pair::new(fs, format!("{fs} Hz, {channels} ch, {n} samples, 16-bit"));
                p.feed(&Pcm::I16(to_i16(x)), Layout::encoder(channels), n, 16);
                assert!(p.valid > 0);
            }
        }
    }
}

/// The channel choices of the multistream encoder: a coupled stream adds
/// two channels and halves the sum, a mono stream reads one channel. Also
/// `c2 = -2` with more than two channels, which sums them all without
/// halving.
#[test]
fn multistream_channel_layouts() {
    let layouts = [
        Layout { channels: 3, c1: 1, c2: 2 },
        Layout { channels: 3, c1: 2, c2: -1 },
        Layout { channels: 2, c1: 1, c2: -1 },
        Layout { channels: 6, c1: 0, c2: -2 },
        Layout { channels: 8, c1: 6, c2: 7 },
        Layout { channels: 8, c1: 3, c2: -1 },
    ];
    for fs in RATES {
        let a = concat(&[speech(fs, 1.5), music(fs, 1.0, 5)]);
        let b = concat(&[music(fs, 1.0, 9), noise(fs, 0.5, 0.2, 4), speech(fs, 1.0)]);
        for layout in layouts {
            // Channel c is a different mix of the two sources.
            let ch = layout.channels;
            let mut x = vec![0.0f32; a.len() * ch];
            for i in 0..a.len() {
                for c in 0..ch {
                    let g = (c as f32 + 1.0) / ch as f32;
                    x[i * ch + c] = 0.6 * (g * a[i] + (1.0 - g) * b[i]);
                }
            }
            for d in [4, 8] {
                let n = frame(fs, d);
                let mut p = Pair::new(fs, format!("{fs} Hz, {layout:?}, {n} samples, float"));
                p.feed(&Pcm::F32(x.clone()), layout, n, 24);
                assert!(p.valid > 0);
                let mut p = Pair::new(fs, format!("{fs} Hz, {layout:?}, {n} samples, 16-bit"));
                p.feed(&Pcm::I16(to_i16(&x)), layout, n, 16);
                assert!(p.valid > 0);
            }
        }
    }
}

/// `lsb_depth` sets the noise floor of the bandwidth decision and the
/// level of digital silence. Quiet noise and tones from -10 to -150 dBFS,
/// signals just below and above the silence level, and silence after a
/// signal.
#[test]
fn bit_depths_and_quiet_signals() {
    for fs in RATES {
        let mut parts = Vec::new();
        for db in [-10.0f32, -40.0, -70.0, -90.0, -110.0, -130.0, -150.0] {
            let amp = 10f32.powf(db / 20.0);
            parts.push(noise(fs, 0.3, amp, db.to_bits() as u64));
            parts.push(tones(fs, 0.2, &[300.0, 5000.0, 11000.0], amp));
        }
        // The silence level for lsb_depth 24 is 2^-24 at the scale of
        // 16-bit samples, so 2^-39 in -1 to 1. Just below, then just above.
        let level = (2.0f32).powi(-39);
        parts.push(vec![level; fs as usize / 5]);
        parts.push(vec![-level * 1.5; fs as usize / 5]);
        parts.push(speech(fs, 0.5));
        parts.push(vec![0.0; fs as usize / 2]);
        parts.push(noise(fs, 0.2, 1e-7, 2));
        let x = concat(&parts);
        for lsb in [8, 12, 16, 20, 24] {
            for d in [2, 8, 24] {
                let n = frame(fs, d);
                let mut p = Pair::new(fs, format!("{fs} Hz, lsb {lsb}, {n} samples"));
                p.feed(&Pcm::F32(x.clone()), Layout::encoder(1), n, lsb);
                assert!(p.valid > 0);
            }
            let mut p = Pair::new(fs, format!("{fs} Hz, lsb {lsb}, 16-bit"));
            p.feed(&Pcm::I16(to_i16(&x)), Layout::encoder(1), frame(fs, 8), lsb);
        }
    }
}

/// With an expert frame duration, the caller's buffer is longer than the
/// coded frame. The analysis reads the whole buffer, keeps how far it got
/// (`analysis_offset`), and assumes that the next buffer starts with the
/// samples after the coded frame. Here the caller does what libopus
/// assumes: each buffer starts at the first sample not yet coded.
#[test]
fn expert_frame_duration_lookahead() {
    for fs in RATES {
        let x = mixed(fs);
        let stereo = interleave(&x, &[0.7, 0.4]);
        // (buffer, coded frame) in steps of 2.5 ms.
        let cases = [(8, 1), (8, 2), (8, 4), (24, 8), (16, 8), (48, 8), (48, 24), (40, 16)];
        for (channels, sig) in [(1usize, &x), (2, &stereo)] {
            let pcm = Pcm::F32(sig.clone());
            let layout = Layout::encoder(channels);
            for (buf, coded) in cases {
                let (buf, coded) = (frame(fs, buf), frame(fs, coded));
                let mut p = Pair::new(fs, format!("{fs} Hz, {channels} ch, buffer {buf}, frame {coded}"));
                let total = sig.len() / channels;
                let mut pos = 0;
                while pos + buf <= total {
                    let b = pcm.slice(pos, pos + buf, channels);
                    p.run(&b, layout, buf, coded, 24);
                    pos += coded;
                }
                assert!(p.valid > 0);
            }
        }
        // The buffer length changes from call to call, odd lengths (the
        // analysis drops the last sample), and buffers longer than the
        // 1.9 s the analysis takes at most.
        let pcm = Pcm::F32(x.clone());
        let mut p = Pair::new(fs, format!("{fs} Hz, changing buffers"));
        let mut rng = Lcg(fs as u64);
        let total = x.len();
        let mut pos = 0;
        let unit = frame(fs, 1);
        loop {
            let coded = unit * DURATIONS[rng.below(9) as usize];
            let extra = match rng.below(6) {
                0 => 0,
                1 => 1,
                2 => unit * rng.below(8) as usize,
                3 => unit * rng.below(48) as usize + 1,
                4 => fs as usize * 2,
                _ => unit * 8,
            };
            let buf = coded + extra;
            if pos + buf > total {
                break;
            }
            let b = pcm.slice(pos, pos + buf, 1);
            p.run(&b, Layout::encoder(1), buf, coded, 24);
            pos += coded;
        }
        assert!(p.valid > 0);
    }
}

/// Inputs at and beyond the limits: the float downmix limits the sum to
/// ±65536 (+6 dBFS) and replaces a NaN by 0. Infinities, NaN, huge values,
/// subnormal values, full-scale square waves, and a sum of two channels
/// whose product with 32768 overflows (where the fused multiply-add of the
/// libopus build gives -inf, and rounded products would give NaN).
#[test]
fn extreme_inputs() {
    for fs in RATES {
        let n = fs as usize;
        let sq = |amp: f32, period: usize| -> Vec<f32> {
            (0..n / 2).map(|i| if (i / period).is_multiple_of(2) { amp } else { -amp }).collect()
        };
        let mut rng = Lcg(5);
        let bursts: Vec<f32> = (0..n)
            .map(|i| match i % 997 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 1e30,
                4 => -3e38,
                _ => 0.3 * rng.uniform(),
            })
            .collect();
        let mono = concat(&[
            sq(2.0, 37),
            sq(1.0, 5),
            sq(1e30, 100),
            vec![f32::NAN; n / 10],
            speech(fs, 0.5),
            vec![f32::INFINITY; n / 10],
            vec![1e-40; n / 4],
            vec![0.99; n / 4],
            bursts,
            speech(fs, 0.5),
        ]);
        for d in [1, 8, 24] {
            let len = frame(fs, d);
            let mut p = Pair::new(fs, format!("{fs} Hz, extreme mono, {len} samples"));
            p.feed(&Pcm::F32(mono.clone()), Layout::encoder(1), len, 24);
        }
        // Two channels whose scaled sum overflows.
        let mut stereo = interleave(&mono, &[1.0, -1.0]);
        for i in (0..stereo.len() / 2).step_by(13) {
            stereo[2 * i] = -3e38;
            stereo[2 * i + 1] = 3e38;
        }
        for layout in [Layout::encoder(2), Layout { channels: 2, c1: 0, c2: 1 }] {
            let mut p = Pair::new(fs, format!("{fs} Hz, extreme stereo, {layout:?}"));
            p.feed(&Pcm::F32(stereo.clone()), layout, frame(fs, 8), 24);
        }
        // 16-bit full scale, also summed over 6 channels without a limit.
        let full: Vec<i16> = (0..n * 6)
            .map(|i| if (i / 6 / 3) % 2 == 0 { i16::MIN } else { i16::MAX })
            .collect();
        for layout in [Layout::encoder(6), Layout { channels: 6, c1: 2, c2: 5 }] {
            let mut p = Pair::new(fs, format!("{fs} Hz, 16-bit full scale, {layout:?}"));
            p.feed(&Pcm::I16(full.clone()), layout, frame(fs, 8), 16);
        }
    }
}

/// The paths that the downmix makes unreachable from `run`: a NaN after
/// the FFT (C 562) and a band energy of 1e9 or more (C 662). Both set
/// `valid = 0` and return early, after part of the state was written; the
/// following frames then read stale slots of the ring. The test writes
/// such values into the first 240 samples of `inmem` of both states, which
/// the next call keeps (it writes its samples after them).
#[test]
fn invalid_blocks_from_the_buffer() {
    for fs in RATES {
        let x = mixed(fs);
        let n20 = frame(fs, 8);
        let pcm = Pcm::F32(x.clone());
        let mut p = Pair::new(fs, format!("{fs} Hz, injected buffers"));
        let mut pos = 0;
        let mut k = 0;
        let (mut band_check, mut nan_check) = (0, 0);
        while pos + n20 <= x.len() {
            match k % 23 {
                5 => {
                    // Band energies far above 1e9.
                    for (i, v) in p.rs.inmem.iter_mut().enumerate() {
                        *v = if i % 2 == 0 { 3e9 } else { -3e9 };
                    }
                    p.c.st.inmem = p.rs.inmem;
                }
                9 => {
                    // A NaN among the 240 samples that the next call keeps,
                    // not at the end, so that the silence test is false
                    // whatever the order of its maximum.
                    p.rs.inmem[100] = f32::NAN;
                    p.c.st.inmem[100] = f32::NAN;
                }
                14 => {
                    p.rs.inmem[10] = f32::INFINITY;
                    p.c.st.inmem[10] = f32::INFINITY;
                }
                17 => {
                    // Energy just below and around the limit in one band.
                    let a = 2.0e6 + 1e5 * (k % 7) as f32;
                    for (i, v) in p.rs.inmem.iter_mut().enumerate() {
                        *v = a * (i as f32 * 0.3).sin();
                    }
                    p.c.st.inmem = p.rs.inmem;
                }
                _ => {}
            }
            p.check_state("after the injection");
            let b = pcm.slice(pos, pos + n20, 1);
            let info = p.run(&b, Layout::encoder(1), n20, n20, 24);
            // The result of a 20 ms call is the block it completed. It is
            // not checked when the rest of the block is digital silence,
            // which skips both checks.
            match k % 23 {
                5 if !info.valid => band_check += 1,
                9 if !info.valid => nan_check += 1,
                _ => {}
            }
            pos += n20;
            k += 1;
        }
        assert!(p.valid > 0);
        assert!(band_check >= 10 && nan_check >= 10, "{band_check} {nan_check}");
    }
}

/// `tonality_analysis_reset` in the middle of a stream (libopus resets at
/// `OPUS_RESET_STATE` and when the complexity drops below 7), a reset
/// followed by digital silence (which copies a cleared slot), and long
/// silence, which keeps copying the slot before and wraps the ring.
#[test]
fn reset_silence_and_restart() {
    for fs in RATES {
        let n = frame(fs, 8);
        let a = Pcm::F32(concat(&[speech(fs, 1.0), vec![0.0; fs as usize * 3], music(fs, 1.0, 1)]));
        let silence = Pcm::F32(vec![0.0; fs as usize]);
        let b = Pcm::F32(speech(fs, 1.0));
        let mut p = Pair::new(fs, format!("{fs} Hz, resets"));
        p.feed(&a, Layout::encoder(1), n, 24);
        p.reset();
        p.feed(&silence, Layout::encoder(1), n, 24);
        p.feed(&b, Layout::encoder(1), n, 24);
        p.reset();
        p.feed(&b, Layout::encoder(1), frame(fs, 2), 24);
        p.reset();
        p.feed(&a, Layout::encoder(1), frame(fs, 24), 24);
        assert!(p.valid > 0);
    }
}

/// A packet of several frames, as `opus_encode_native` codes it: one
/// `run_analysis` for the whole packet, then the read position goes back
/// to where it was, and `tonality_get_info` is called for each frame. Also
/// single SILK frames of 40 and 60 ms, which read the second slot.
#[test]
fn packets_of_several_frames_rewind_the_read_position() {
    for fs in RATES {
        let x = Pcm::F32(mixed(fs));
        let total = x.len();
        // (packet, frame) in steps of 2.5 ms.
        for (packet, part) in [(16, 8), (24, 8), (48, 8), (32, 8), (40, 8), (16, 16), (24, 24), (48, 16), (8, 4)] {
            let (packet, part) = (frame(fs, packet), frame(fs, part));
            let mut p = Pair::new(fs, format!("{fs} Hz, packet {packet} in frames of {part}"));
            let mut pos = 0;
            while pos + packet <= total {
                let b = x.slice(pos, pos + packet, 1);
                let cursor = (p.rs.read_pos, p.rs.read_subframe);
                let c_cursor = (p.c.st.read_pos, p.c.st.read_subframe);
                p.run(&b, Layout::encoder(1), packet, packet, 24);
                p.rs.set_read_cursor((cursor.0, cursor.1));
                p.c.st.read_pos = c_cursor.0;
                p.c.st.read_subframe = c_cursor.1;
                p.check_state("after the rewind");
                for _ in 0..packet / part {
                    p.get_info(part);
                }
                pos += packet;
            }
            assert!(p.valid > 0);
        }
    }
}

/// More than `ANALYSIS_COUNT_MAX` (10,000) blocks, so that the counter
/// stops, and a long run of the recurrent network: 205 s at 24 kHz, stereo.
#[test]
fn long_run_past_the_frame_counter() {
    let fs = 24000;
    let one = concat(&[speech(fs, 6.0), music(fs, 3.0, 2), noise(fs, 1.0, 0.01, 8)]);
    let mut p = Pair::new(fs, "205 s at 24 kHz");
    let n = frame(fs, 8);
    let mut seconds = 0.0;
    let mut k = 0u64;
    while seconds < 205.0 {
        let sig = interleave(&one, &[0.9, 0.3 + 0.05 * (k % 5) as f32]);
        p.feed(&Pcm::F32(sig), Layout::encoder(2), n, 24);
        seconds += one.len() as f32 / fs as f32;
        k += 1;
    }
    assert_eq!(p.rs.count, 10000);
    assert!(p.valid > 9000);
}

/// Random calls: frame durations, buffer lengths, layouts, input types,
/// bit depths, resets and signal pieces chosen by a seeded generator.
#[test]
fn random_call_patterns() {
    let layouts = [
        Layout::encoder(1),
        Layout::encoder(2),
        Layout { channels: 3, c1: 0, c2: 2 },
        Layout { channels: 3, c1: 1, c2: -1 },
        Layout::encoder(4),
    ];
    for seed in 0..12u64 {
        let mut rng = Lcg(seed * 7919 + 1);
        let fs = RATES[rng.below(3) as usize];
        let layout = layouts[rng.below(5) as usize];
        let float = rng.below(2) == 0;
        let lsb = if float { [16, 20, 24][rng.below(3) as usize] } else { 16 };
        let mut p = Pair::new(fs, format!("seed {seed}, {fs} Hz, {layout:?}, float {float}"));
        let unit = frame(fs, 1);
        for _ in 0..300 {
            let coded = unit * DURATIONS[rng.below(9) as usize];
            let buf = if rng.below(4) == 0 { coded + unit * rng.below(16) as usize } else { coded };
            let kind = rng.below(8);
            let src = match kind {
                0 => speech(fs, buf as f32 / fs as f32 + 0.01),
                1 => music(fs, buf as f32 / fs as f32 + 0.01, rng.next_u32() as u64),
                2 => vec![0.0; buf],
                3 => noise(fs, buf as f32 / fs as f32 + 0.01, 0.5, rng.next_u32() as u64),
                4 => tones(fs, buf as f32 / fs as f32 + 0.01, &[100.0 + rng.below(8000) as f64], 0.8),
                5 => clicks(fs, buf as f32 / fs as f32 + 0.01, 1 + rng.below(500) as usize),
                6 => noise(fs, buf as f32 / fs as f32 + 0.01, 1e-6, rng.next_u32() as u64),
                _ => chirp(fs, buf as f32 / fs as f32 + 0.01, 0.3),
            };
            let gains: Vec<f32> = (0..layout.channels).map(|c| 1.0 - 0.3 * c as f32).collect();
            let x = interleave(&src[..buf], &gains);
            let pcm = if float { Pcm::F32(x) } else { Pcm::I16(to_i16(&x)) };
            p.run(&pcm, layout, buf, coded, lsb);
            if rng.below(60) == 0 {
                p.reset();
            }
        }
    }
}
