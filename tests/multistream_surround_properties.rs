//! Properties of the surround masking analysis that need no reference.
//!
//! The other files compare the mask with a reference in `f64`, with
//! `surround_analysis` of libopus and with the packets of the C surround
//! encoder. The tests here state what follows from the algorithm itself and
//! require it exactly, bit for bit:
//!
//! - after a reset, and after a frame that libopus drops on every channel,
//!   the analysis is in the state of a new one;
//! - the left and the right side of the mix are treated alike;
//! - the low-frequency-effects channel has no influence on the other
//!   channels;
//! - a centre channel counts alike on both sides;
//! - any `f32` bit pattern in the input gives a mask within the bounds the
//!   algorithm sets.
//!
//! The last test prints the cost of the analysis for several layouts, rates
//! and frame sizes.
// The multistream API needs the `std` feature.
#![cfg(feature = "std")]

mod ms_support;

use ms_support::{noise, silence};
use opus_rs::multistream::surround::SurroundProbe;

const NB_BANDS: usize = 21;
const RATES: [i32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];

/// The masks of all frames of `pcm`, one after the other.
fn masks(probe: &mut SurroundProbe, pcm: &[f32], channels: usize, frame_size: usize) -> Vec<f32> {
    let mut all = Vec::new();
    for frame in pcm.chunks_exact(frame_size * channels) {
        let mut mask = vec![f32::NAN; NB_BANDS * channels];
        assert!(probe.run(frame, frame_size, &mut mask), "no mask");
        all.extend_from_slice(&mask);
    }
    all
}

/// The bits of every value, so that a comparison tells -0.0 from 0.0 and
/// does not stumble over a value that is not a number.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The 21 values of channel `c` in one mask.
fn of_channel(mask: &[f32], c: usize) -> &[f32] {
    &mask[NB_BANDS * c..NB_BANDS * (c + 1)]
}

/// Noise with a constant offset, with another level and another offset on
/// every channel.
///
/// The offset makes the pre-emphasis memory count. With the memory, the
/// filter turns the offset into a small constant. Without it, the first
/// sample of a frame is a step of the full offset.
fn noise_with_offset(channels: usize, samples: usize, seed: u32) -> Vec<f32> {
    let mut pcm = noise(channels, samples, 0.05, seed);
    for (i, v) in pcm.iter_mut().enumerate() {
        let c = i % channels;
        *v = *v / (1 + c) as f32 + 0.3 - 0.07 * c as f32;
    }
    pcm
}

#[test]
fn reset_clears_the_pre_emphasis_memory_too() {
    // The analysis keeps two memories per channel: the last 120 filtered
    // samples and the memory of the pre-emphasis filter. Below 48 kHz the
    // filter memory is 0 at the end of every frame, because the last sample
    // at 48 kHz is an inserted zero. Only at 48 kHz can a test see whether
    // a reset clears it. The other rates are run as well.
    for rate in RATES {
        for channels in 3..=8usize {
            for frame_400 in [1usize, 8, 16] {
                let frame_size = rate as usize / 400 * frame_400;
                let what = format!("{channels} channels, {rate} Hz, frame {frame_size}");
                let pcm = noise_with_offset(channels, 3 * frame_size, 211);
                let (start, last) = pcm.split_at(2 * frame_size * channels);

                let fresh = masks(
                    &mut SurroundProbe::new(channels, rate),
                    last,
                    channels,
                    frame_size,
                );

                let mut used = SurroundProbe::new(channels, rate);
                masks(&mut used, start, channels, frame_size);
                used.reset();
                let after_reset = masks(&mut used, last, channels, frame_size);
                assert_eq!(bits(&after_reset), bits(&fresh), "{what}: after a reset");

                // The memories matter: without the reset the result differs.
                let mut carried = SurroundProbe::new(channels, rate);
                masks(&mut carried, start, channels, frame_size);
                let without_reset = masks(&mut carried, last, channels, frame_size);
                assert_ne!(bits(&without_reset), bits(&fresh), "{what}: no reset");
            }
        }
    }
}

#[test]
fn the_pre_emphasis_memory_alone_changes_the_next_frame() {
    // A frame that leaves the 120 kept samples at exactly 0 and the filter
    // memory at a value that is not 0. Then a frame of silence shows the
    // filter memory alone: with it the first sample is a small step, and
    // after a reset the frame is the silence of a new analysis.
    let (channels, rate, frame_size) = (6usize, 48_000, 960usize);
    let preemph = opus_rs::modes::default_mode().preemph[0];
    let mut first = silence(channels, frame_size);
    // From sample 838 on, every sample is what the filter predicts from the
    // sample before it. The filter output is then exactly 0 from sample 839
    // on, which covers the 120 kept samples. The scaling by 32768 is exact,
    // so the test can compute the same product as the filter.
    let mut x = 4096.0f32;
    for i in 838..frame_size {
        first[i * channels] = x;
        x = (preemph * (x * 32768.0)) / 32768.0;
    }
    // The memory after the frame, in the 16-bit range CELT works in.
    let memory = preemph * (first[(frame_size - 1) * channels] * 32768.0);
    assert!((0.1..1.0).contains(&memory), "{memory}");
    let quiet = silence(channels, frame_size);

    let fresh = masks(
        &mut SurroundProbe::new(channels, rate),
        &quiet,
        channels,
        frame_size,
    );

    let mut probe = SurroundProbe::new(channels, rate);
    masks(&mut probe, &first, channels, frame_size);
    let carried = masks(&mut probe, &quiet, channels, frame_size);
    // The front left channel now holds one step, which is far above the
    // level of silence in every band.
    for b in 0..NB_BANDS {
        assert!(
            carried[b] > fresh[b] + 10.0,
            "band {b}: {} with the memory, {} for silence",
            carried[b],
            fresh[b]
        );
    }
    // The right side of the mix never held signal: the 120 kept samples of
    // every channel were 0, and the right channels are still at silence.
    assert_eq!(
        bits(of_channel(&carried, 2)),
        bits(of_channel(&fresh, 2)),
        "front right"
    );

    let mut probe = SurroundProbe::new(channels, rate);
    masks(&mut probe, &first, channels, frame_size);
    probe.reset();
    let after_reset = masks(&mut probe, &quiet, channels, frame_size);
    assert_eq!(bits(&after_reset), bits(&fresh));
}

#[test]
fn a_frame_dropped_on_every_channel_leaves_the_state_of_a_new_analysis() {
    // libopus clears a channel whose frame holds a sample that is not a
    // number or an energy of 1e18 or more. It clears the 120 kept samples
    // and the filter memory with it. When that happens on every channel,
    // the next frame is analysed as by a new analysis.
    for (channels, rate, frame_size) in [
        (6usize, 48_000, 960usize),
        (8, 48_000, 120),
        (3, 48_000, 2880),
        (7, 16_000, 320),
        (4, 8_000, 20),
        (5, 24_000, 240),
    ] {
        for (name, bad) in [
            ("not a number", f32::NAN),
            ("infinite", f32::INFINITY),
            ("-infinite", f32::NEG_INFINITY),
            ("the largest f32", f32::MAX),
            ("1e9", 1e9),
        ] {
            let what = format!("{channels} channels, {rate} Hz, frame {frame_size}, {name}");
            let pcm = noise_with_offset(channels, 4 * frame_size, 223);
            let frames: Vec<&[f32]> = pcm.chunks_exact(frame_size * channels).collect();
            let fresh = masks(
                &mut SurroundProbe::new(channels, rate),
                frames[3],
                channels,
                frame_size,
            );

            let mut probe = SurroundProbe::new(channels, rate);
            masks(&mut probe, frames[0], channels, frame_size);
            masks(&mut probe, frames[1], channels, frame_size);
            // One bad sample per channel, each at another place.
            let mut dropped = frames[2].to_vec();
            for c in 0..channels {
                dropped[(c * 7 % frame_size) * channels + c] = bad;
            }
            let of_dropped = masks(&mut probe, &dropped, channels, frame_size);
            // A dropped frame is analysed as silence.
            let of_silence = masks(
                &mut SurroundProbe::new(channels, rate),
                &silence(channels, frame_size),
                channels,
                frame_size,
            );
            assert_eq!(bits(&of_dropped), bits(&of_silence), "{what}: dropped");
            let after = masks(&mut probe, frames[3], channels, frame_size);
            assert_eq!(bits(&after), bits(&fresh), "{what}: the frame after");
        }
    }
}

/// The pairs of channels that mirror each other in the surround layout of
/// `channels` channels (Vorbis order), and the channels in the centre.
fn mirror_pairs(channels: usize) -> (Vec<(usize, usize)>, Vec<usize>) {
    match channels {
        // Left, centre, right.
        3 => (vec![(0, 2)], vec![1]),
        // Front left, front right, rear left, rear right.
        4 => (vec![(0, 1), (2, 3)], vec![]),
        // Front left, centre, front right, rear left, rear right.
        5 | 6 => (vec![(0, 2), (3, 4)], vec![1]),
        // The same, then rear centre.
        7 => (vec![(0, 2), (3, 4)], vec![1, 5]),
        // Front left, centre, front right, side left, side right, rear
        // left, rear right.
        8 => (vec![(0, 2), (3, 4), (5, 6)], vec![1]),
        _ => panic!("no surround layout for {channels} channels"),
    }
}

/// The masks of `pcm` and of `pcm` with every left channel swapped with its
/// right partner, frame by frame.
fn masks_and_swapped_masks(
    channels: usize,
    rate: i32,
    frame_size: usize,
    pcm: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (pairs, _) = mirror_pairs(channels);
    let mut swapped = pcm.to_vec();
    for x in swapped.chunks_exact_mut(channels) {
        for &(l, r) in &pairs {
            x.swap(l, r);
        }
    }
    let mut a = SurroundProbe::new(channels, rate);
    let mut b = SurroundProbe::new(channels, rate);
    (
        masks(&mut a, pcm, channels, frame_size),
        masks(&mut b, &swapped, channels, frame_size),
    )
}

const SWAP_CASES: [(i32, usize); 4] = [(48_000, 960), (48_000, 240), (12_000, 240), (8_000, 320)];

#[test]
fn swapping_left_and_right_swaps_the_masks() {
    // With a silent centre, the left side of the mix holds the left
    // channels and the right side the right channels, added in the same
    // order. Swapping every left channel with its right partner then swaps
    // their masks and leaves the mask of the centre as it is, bit for bit.
    for channels in 3..=8usize {
        let (pairs, centres) = mirror_pairs(channels);
        for (rate, frame_size) in SWAP_CASES {
            let what = format!("{channels} channels, {rate} Hz, frame {frame_size}");
            let mut pcm = noise_with_offset(channels, 3 * frame_size, 227);
            for x in pcm.chunks_exact_mut(channels) {
                for &c in &centres {
                    x[c] = 0.0;
                }
            }
            let (a, b) = masks_and_swapped_masks(channels, rate, frame_size, &pcm);
            let frames = a.chunks_exact(NB_BANDS * channels);
            for (f, (a, b)) in frames.zip(b.chunks_exact(NB_BANDS * channels)).enumerate() {
                for &(l, r) in &pairs {
                    assert_eq!(
                        bits(of_channel(a, l)),
                        bits(of_channel(b, r)),
                        "{what}, frame {f}: channel {l} against the swapped {r}"
                    );
                    assert_eq!(
                        bits(of_channel(a, r)),
                        bits(of_channel(b, l)),
                        "{what}, frame {f}: channel {r} against the swapped {l}"
                    );
                }
                for &c in &centres {
                    assert_eq!(
                        bits(of_channel(a, c)),
                        bits(of_channel(b, c)),
                        "{what}, frame {f}: centre channel {c}"
                    );
                }
                // With two or more channels on a side, the masks of the
                // two sides differ, so a mix-up of the sides would show.
                if pairs.len() >= 2 {
                    let (l, r) = pairs[0];
                    assert_ne!(bits(of_channel(a, l)), bits(of_channel(a, r)), "{what}");
                }
            }
        }
    }
}

#[test]
fn a_centre_channel_counts_alike_on_both_sides() {
    // The same with signal in the centre. A centre channel is added to both
    // sides of the mix, 3 dB down, so the masks still mirror each other.
    //
    // At a usual level they do so bit for bit: every side starts at -28,
    // which is too far below the signal to count, and `logSum` gives the
    // same result for its two arguments in either order. A signal 200 dB
    // down lies close to -28. The sides then add their channels to it in
    // another order, and `logSum` is only an approximation, so the masks
    // mirror each other closely but not bit for bit.
    for (gain, tolerance) in [(1.0f32, 0.0f32), (1e-10, 0.03)] {
        let mut worst = 0f32;
        for channels in [3usize, 5, 6, 7, 8] {
            let (pairs, centres) = mirror_pairs(channels);
            for (rate, frame_size) in SWAP_CASES {
                let what =
                    format!("gain {gain:e}, {channels} channels, {rate} Hz, frame {frame_size}");
                let mut pcm = noise_with_offset(channels, 3 * frame_size, 227);
                for x in pcm.chunks_exact_mut(channels) {
                    for v in x.iter_mut() {
                        *v *= gain;
                    }
                    // The centre channels are the loudest, so they count
                    // on both sides.
                    for &c in &centres {
                        x[c] *= 3.0;
                    }
                }
                let (a, b) = masks_and_swapped_masks(channels, rate, frame_size, &pcm);
                let mut sides_differ = 0f32;
                let frames = a.chunks_exact(NB_BANDS * channels);
                for (a, b) in frames.zip(b.chunks_exact(NB_BANDS * channels)) {
                    let mut compare = |x: &[f32], y: &[f32]| {
                        for (&x, &y) in x.iter().zip(y) {
                            worst = worst.max((x - y).abs());
                        }
                    };
                    for &(l, r) in &pairs {
                        compare(of_channel(a, l), of_channel(b, r));
                        compare(of_channel(a, r), of_channel(b, l));
                        for (&x, &y) in of_channel(a, l).iter().zip(of_channel(a, r)) {
                            sides_differ = sides_differ.max((x - y).abs());
                        }
                    }
                    for &c in &centres {
                        compare(of_channel(a, c), of_channel(b, c));
                    }
                }
                assert!(
                    worst <= tolerance,
                    "{what}: a mask and its mirror image differ by {worst}"
                );
                // The left and the right channels have other levels, so
                // their masks are not the same.
                assert!(
                    sides_differ > 0.2,
                    "{what}: the sides differ by {sides_differ}"
                );
            }
        }
        println!("gain {gain:e}: a mask and its mirror image differ by at most {worst}");
    }
}

#[test]
fn the_low_frequency_channel_changes_no_other_mask() {
    // The low-frequency-effects channel is the last one of the layouts with
    // 6, 7 and 8 channels. It is not part of the mix: whatever it holds,
    // the masks of the other channels are the same bits, and its own mask
    // is 0.
    for channels in [6usize, 7, 8] {
        for (rate, frame_size) in [(48_000, 960usize), (48_000, 120), (16_000, 960)] {
            let what = format!("{channels} channels, {rate} Hz, frame {frame_size}");
            let quiet = {
                let mut pcm = noise_with_offset(channels, 3 * frame_size, 229);
                for x in pcm.chunks_exact_mut(channels) {
                    x[channels - 1] = 0.0;
                }
                pcm
            };
            let mut loud = quiet.clone();
            let rumble = noise(1, 3 * frame_size, 0.9, 233);
            for (x, &v) in loud.chunks_exact_mut(channels).zip(&rumble) {
                x[channels - 1] = v;
            }
            let a = masks(
                &mut SurroundProbe::new(channels, rate),
                &quiet,
                channels,
                frame_size,
            );
            let b = masks(
                &mut SurroundProbe::new(channels, rate),
                &loud,
                channels,
                frame_size,
            );
            assert_eq!(bits(&a), bits(&b), "{what}");
            for mask in a.chunks_exact(NB_BANDS * channels) {
                assert!(
                    of_channel(mask, channels - 1).iter().all(|&v| v == 0.0),
                    "{what}: the low-frequency-effects mask is not 0"
                );
                // Every other channel is part of the mix and has a mask.
                for c in 0..channels - 1 {
                    assert!(
                        of_channel(mask, c).iter().any(|&v| v != 0.0),
                        "{what}: channel {c} has no mask"
                    );
                }
            }
        }
    }
    // The layouts with 3, 4 and 5 channels have no such channel: taking the
    // signal off any channel changes the mask of another one.
    for channels in [3usize, 4, 5] {
        let (rate, frame_size) = (48_000, 960usize);
        let pcm = noise_with_offset(channels, 2 * frame_size, 239);
        let a = masks(
            &mut SurroundProbe::new(channels, rate),
            &pcm,
            channels,
            frame_size,
        );
        for c in 0..channels {
            let mut changed = pcm.clone();
            for x in changed.chunks_exact_mut(channels) {
                x[c] = 0.0;
            }
            let b = masks(
                &mut SurroundProbe::new(channels, rate),
                &changed,
                channels,
                frame_size,
            );
            let others_changed = (0..channels).filter(|&o| o != c).any(|o| {
                let pairs = a.chunks_exact(NB_BANDS * channels);
                pairs
                    .zip(b.chunks_exact(NB_BANDS * channels))
                    .any(|(a, b)| bits(of_channel(a, o)) != bits(of_channel(b, o)))
            });
            assert!(
                others_changed,
                "{channels} channels: channel {c} masks nothing"
            );
        }
    }
}

/// A pseudo-random generator (xorshift32).
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

#[test]
fn any_bit_pattern_in_the_input_gives_a_mask_within_its_bounds() {
    // Noise in which one sample of 64 is a random `f32` bit pattern: huge
    // values, tiny values, infinities and values that are not numbers. The
    // analysis must give a mask for every frame, and every value must be a
    // finite number within the bounds the algorithm sets.
    //
    // The upper bound: `logSum` never returns less than the larger of its
    // arguments, so a side of the mix is never below a channel on it. A
    // left or right channel is therefore at most at its mask, and a centre
    // channel, which is added 0.5 down, at most 0.5 above. The mask is
    // then shifted by the offset for the number of channels.
    //
    // The lower bound: an empty band is at -44.8 minus the band mean, which
    // is 6.44 at most. libopus drops a channel whose energy is 1e18 or
    // more, so a band amplitude stays below 1e9. That is a level of 30,
    // and four channels on one side add at most 1.5.
    let mut rng = Rng(0x5eed_0241);
    let (mut frames_run, mut frames_with_patterns) = (0usize, 0usize);
    let mut highest = f32::MIN;
    let mut lowest = f32::MAX;
    for channels in 3..=8usize {
        let (_, centres) = mirror_pairs(channels);
        let lfe = (channels >= 6).then_some(channels - 1);
        let channel_offset = 0.5 * (2.0 / (channels - 1) as f32).log2();
        for rate in RATES {
            for frame_400 in [1usize, 2, 4, 8, 24, 48] {
                let frame_size = rate as usize / 400 * frame_400;
                let frames = (48 / frame_400).clamp(2, 12);
                let what = format!("{channels} channels, {rate} Hz, frame {frame_size}");
                let mut probe = SurroundProbe::new(channels, rate);
                let mut mask = vec![0f32; NB_BANDS * channels];
                let mut pcm = vec![0f32; frame_size * channels];
                for f in 0..frames {
                    let mut patterns = false;
                    for v in pcm.iter_mut() {
                        let r = rng.next();
                        // Every third frame is plain noise.
                        *v = if r.is_multiple_of(64) && f % 3 != 2 {
                            patterns = true;
                            f32::from_bits(rng.next())
                        } else {
                            (r >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                        };
                    }
                    assert!(probe.run(&pcm, frame_size, &mut mask), "{what}");
                    frames_run += 1;
                    frames_with_patterns += patterns as usize;
                    for c in 0..channels {
                        let upper = if lfe == Some(c) {
                            0.0
                        } else if centres.contains(&c) {
                            0.5 - channel_offset + 1e-4
                        } else {
                            -channel_offset + 1e-4
                        };
                        for (b, &v) in of_channel(&mask, c).iter().enumerate() {
                            assert!(
                                v > -90.0 && v <= upper,
                                "{what}, frame {f}, channel {c}, band {b}: {v}, upper bound {upper}"
                            );
                            if lfe == Some(c) {
                                assert_eq!(v, 0.0, "{what}, frame {f}, band {b}");
                            } else {
                                highest = highest.max(v - upper);
                                lowest = lowest.min(v);
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "{frames_run} frames, {frames_with_patterns} with bit patterns: lowest value {lowest}, closest to the upper bound {highest}"
    );
    // Both kinds of frames were run, and the upper bound is reached.
    assert!(frames_with_patterns * 4 > frames_run);
    assert!(frames_with_patterns * 4 < frames_run * 3);
    assert!(highest > -1e-3, "{highest}");
}

#[test]
fn the_analysis_can_move_between_threads() {
    // The multistream encoder owns the analysis, and an encoder is handed
    // from thread to thread.
    fn is_send_and_sync<T: Send + Sync>() {}
    is_send_and_sync::<SurroundProbe>();

    let (channels, rate, frame_size) = (6usize, 48_000, 960usize);
    let pcm = noise_with_offset(channels, 2 * frame_size, 241);
    let want = masks(
        &mut SurroundProbe::new(channels, rate),
        &pcm,
        channels,
        frame_size,
    );
    let (first, second) = pcm.split_at(frame_size * channels);
    let mut probe = SurroundProbe::new(channels, rate);
    let mut got = masks(&mut probe, first, channels, frame_size);
    let second = second.to_vec();
    let rest = std::thread::spawn(move || masks(&mut probe, &second, channels, frame_size))
        .join()
        .unwrap();
    got.extend_from_slice(&rest);
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn cost_by_layout_rate_and_frame_size() {
    // The time of the fastest of 20 batches, so that other work on the
    // machine does not count. Run with `--nocapture` to see the numbers.
    for (channels, rate, frame_400) in [
        (6usize, 48_000, 8usize),
        (8, 48_000, 8),
        (3, 48_000, 8),
        (6, 48_000, 1),
        (6, 48_000, 4),
        (6, 48_000, 24),
        (6, 48_000, 48),
        (8, 48_000, 48),
        (6, 24_000, 8),
        (6, 16_000, 8),
        (6, 8_000, 8),
        (8, 8_000, 1),
    ] {
        let frame_size = rate as usize / 400 * frame_400;
        let frames = (400 / frame_400).max(10);
        let pcm = noise_with_offset(channels, frames * frame_size, 251);
        let mut probe = SurroundProbe::new(channels, rate);
        let mut mask = vec![0f32; NB_BANDS * channels];
        let mut best = f64::INFINITY;
        let mut checksum = 0f32;
        for _ in 0..20 {
            let start = std::time::Instant::now();
            for frame in pcm.chunks_exact(frame_size * channels) {
                assert!(probe.run(std::hint::black_box(frame), frame_size, &mut mask));
                checksum += mask[7];
            }
            best = best.min(start.elapsed().as_secs_f64() / frames as f64);
        }
        assert!(checksum.is_finite());
        let duration = frame_400 as f64 * 0.0025;
        println!(
            "surround analysis, {channels} channels, {rate} Hz, {} ms: {:.2} microseconds per frame ({:.3} % of real time)",
            duration * 1e3,
            best * 1e6,
            best / duration * 100.0
        );
        // A loose limit for optimized builds: 2 % of real time.
        if !cfg!(debug_assertions) {
            assert!(
                best < 0.02 * duration,
                "{channels} channels, {rate} Hz, {} ms: {:.2} microseconds per frame",
                duration * 1e3,
                best * 1e6
            );
        }
    }
}
