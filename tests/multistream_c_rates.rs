//! `opus1::multistream::rate_alloc` against C libopus.
//!
//! The C multistream encoder gives every stream encoder a bitrate before it
//! encodes a frame. That bitrate can be read back from the stream encoder
//! (`OPUS_MULTISTREAM_GET_ENCODER_STATE`, then `OPUS_GET_BITRATE`). These tests
//! require `rate_allocation` to give exactly the same numbers.
//!
//! The byte budget functions are compared with what the C encoder does with
//! an output buffer of a given size: the error for a buffer that is too
//! small, the exact packet size with constant bitrate, the size limit of
//! every stream, and the bitrate it sets on the last stream.
#![cfg(feature = "std")]

mod ms_support;

use ms_support::*;
use opus1::multistream::layout::{ChannelLayout, MappingType};
use opus1::multistream::rate_alloc::{
    MS_FRAME_TMP, cbr_last_stream_bitrate, cbr_max_data_bytes, clamp_user_bitrate, rate_allocation,
    smallest_packet, stream_max_bytes,
};

/// What a C stream encoder reports for `OPUS_GET_BITRATE` after it was set
/// to `rate` and coded a frame of `coded_frame` samples.
///
/// `OPUS_SET_BITRATE` raises the value to 500 and limits it to 750 000 per
/// channel (opus_encoder.c). `OPUS_GET_BITRATE` reports at most what fits
/// into 1276 bytes per frame (`user_bitrate_to_bitrate`).
fn reported(rate: i32, stream_channels: usize, fs: i32, coded_frame: usize) -> i32 {
    let set = rate.clamp(500, 750_000 * stream_channels as i32);
    let cap = (1276i64 * 8 * (6 * fs as i64 / coded_frame as i64) / 6) as i32;
    set.min(cap)
}

/// One case: the C encoder `enc` with `layout`, set to `bitrate`, encodes one
/// frame. Returns a description of the difference between the per-stream
/// bitrates of C and those of `rate_allocation`, or `None` when they agree.
fn compare_rates(
    enc: &mut CMsEncoder,
    layout: &ChannelLayout,
    mapping_type: MappingType,
    lfe_stream: Option<usize>,
    bitrate: i32,
    frame_size: usize,
) -> Option<String> {
    let pcm = vec![0.01f32; frame_size * enc.channels];
    compare_rates_for_signal(
        enc,
        layout,
        mapping_type,
        lfe_stream,
        bitrate,
        frame_size,
        &pcm,
    )
}

/// `compare_rates` with the input signal given by the caller: one frame,
/// interleaved. The allocation must not depend on the signal.
fn compare_rates_for_signal(
    enc: &mut CMsEncoder,
    layout: &ChannelLayout,
    mapping_type: MappingType,
    lfe_stream: Option<usize>,
    bitrate: i32,
    frame_size: usize,
    pcm: &[f32],
) -> Option<String> {
    let fs = enc.fs;
    let (channels, streams, coupled) = (enc.channels, enc.streams, enc.coupled);
    assert_eq!(
        enc.try_ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, bitrate),
        0
    );
    // Room for the largest packet. In a smaller buffer the first streams can
    // leave the last ones too few bytes to code anything, and a stream
    // encoder that codes nothing reports its bitrate differently.
    let packet = enc
        .encode_float(pcm, frame_size, max_packet_bytes(streams))
        .expect("C encode");

    // The multistream encoder clamps an explicit bitrate before it uses it.
    let user = clamp_user_bitrate(bitrate, channels).unwrap();
    let mut rate = [0i32; 255];
    let sum = rate_allocation(
        layout,
        mapping_type,
        lfe_stream,
        user,
        frame_size,
        fs,
        &mut rate,
    )
    .unwrap();
    assert_eq!(
        sum as i64,
        rate[..streams].iter().map(|&r| r as i64).sum::<i64>()
    );
    assert!(rate[..streams].iter().all(|&r| r >= 500));

    // A stream encoder reports its bitrate against the frame it really
    // coded. A frame above 20 ms can be coded as several shorter frames.
    let parts = split_streams(&packet, streams).unwrap();
    let want: Vec<i32> = (0..streams)
        .map(|s| {
            let coded_frame = toc(parts[s].toc).frame_48k * fs as usize / 48_000;
            reported(rate[s], if s < coupled { 2 } else { 1 }, fs, coded_frame)
        })
        .collect();
    let got = enc.stream_bitrates();
    let total = enc.bitrate();
    if got != want {
        return Some(format!(
            "C streams {got:?}, opus1 {want:?} (before limits {:?})",
            &rate[..streams]
        ));
    }
    if total as i64 != want.iter().map(|&r| r as i64).sum::<i64>() {
        return Some(format!("C total {total}, opus1 streams {want:?}"));
    }
    None
}

/// The 14 bitrates of the design grid: automatic, maximum, and explicit
/// values from far below to far above what the layouts need.
const BITRATES: [i32; 14] = [
    OPUS_AUTO,
    OPUS_BITRATE_MAX,
    6_000,
    16_000,
    24_000,
    48_000,
    64_000,
    96_000,
    128_000,
    192_000,
    256_000,
    320_000,
    510_000,
    1_000_000,
];

/// The 18 layouts of the design grid, as (mapping family, channels).
const LAYOUTS: [(u8, usize); 18] = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (255, 1),
    (255, 3),
    (255, 5),
    (2, 1),
    (2, 4),
    (2, 6),
    (2, 9),
    (2, 11),
    (0, 1),
    (0, 2),
];

/// Runs `compare_rates` for surround layouts over a grid and returns the
/// number of cases. Panics with the first differences if there are any.
fn run_grid(
    sample_rates: &[i32],
    layouts: &[(u8, usize)],
    frame_divs: &[(i32, i32)],
    bitrates: &[i32],
) -> usize {
    let mut cases = 0;
    let mut mismatches = Vec::new();
    for &fs in sample_rates {
        for &(family, channels) in layouts {
            let (layout, mapping_type, lfe_stream) =
                ChannelLayout::surround(channels, family).unwrap();
            // frame_size = fs * num / den
            for &(num, den) in frame_divs {
                let frame_size = (fs * num / den) as usize;
                for &bitrate in bitrates {
                    // A new encoder for every case, as an application would have.
                    let mut enc =
                        CMsEncoder::surround(fs, channels, family as i32, APP_AUDIO).unwrap();
                    cases += 1;
                    if let Some(diff) = compare_rates(
                        &mut enc,
                        &layout,
                        mapping_type,
                        lfe_stream,
                        bitrate,
                        frame_size,
                    ) {
                        mismatches.push(format!(
                            "fs {fs}, family {family}, {channels} channels, frame {frame_size}, bitrate {bitrate}: {diff}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {cases} cases differ from libopus:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    cases
}

#[test]
fn rate_allocation_matches_libopus_on_the_design_grid() {
    // 3 sample rates x 18 layouts x 4 frame sizes (20, 10, 5, 2.5 ms) x 14 bitrates.
    let cases = run_grid(
        &[48_000, 24_000, 16_000],
        &LAYOUTS,
        &[(1, 50), (1, 100), (1, 200), (1, 400)],
        &BITRATES,
    );
    assert_eq!(cases, 3024);
}

#[test]
fn rate_allocation_matches_libopus_at_8_and_12_khz() {
    // The two sampling rates the design grid leaves out.
    let cases = run_grid(&[8_000, 12_000], &LAYOUTS, &[(1, 50), (1, 400)], &BITRATES);
    assert_eq!(cases, 2 * 18 * 2 * 14);
}

#[test]
fn rate_allocation_matches_libopus_for_long_frames() {
    // 40, 60, 80, 100 and 120 ms. The allocation uses the whole frame size.
    let layouts = [
        (1u8, 1usize),
        (1, 2),
        (1, 6),
        (1, 8),
        (255, 3),
        (2, 6),
        (0, 2),
    ];
    let bitrates = [
        OPUS_AUTO,
        OPUS_BITRATE_MAX,
        16_000,
        96_000,
        320_000,
        1_000_000,
    ];
    let frames = [(1, 25), (3, 50), (2, 25), (1, 10), (3, 25)];
    let cases = run_grid(&[48_000, 16_000], &layouts, &frames, &bitrates);
    assert_eq!(cases, 2 * 7 * 5 * 6);
}

#[test]
fn rate_allocation_follows_a_bitrate_change() {
    // One encoder, many frames: the allocation depends only on the current
    // setting and frame size, not on what came before.
    for (family, channels) in [(1u8, 6usize), (1, 8), (2, 6), (255, 3)] {
        let fs = 48_000;
        let (layout, mapping_type, lfe_stream) = ChannelLayout::surround(channels, family).unwrap();
        let mut enc = CMsEncoder::surround(fs, channels, family as i32, APP_AUDIO).unwrap();
        for (i, &bitrate) in BITRATES.iter().chain(BITRATES.iter().rev()).enumerate() {
            let frame_size = [960, 480, 240, 120, 1920, 2880][i % 6];
            let diff = compare_rates(
                &mut enc,
                &layout,
                mapping_type,
                lfe_stream,
                bitrate,
                frame_size,
            );
            assert_eq!(
                diff, None,
                "family {family}, {channels} channels, step {i}, bitrate {bitrate}"
            );
        }
    }
}

#[test]
fn rate_allocation_matches_libopus_for_generic_layouts() {
    // The generic constructor has no mapping type and no LFE stream.
    // (channels, streams, coupled, mapping)
    let layouts: [(usize, usize, usize, &[u8]); 8] = [
        (1, 1, 0, &[0]),
        (2, 1, 1, &[0, 1]),
        (2, 2, 0, &[0, 1]),
        (3, 2, 1, &[0, 1, 2]),
        (4, 2, 2, &[0, 1, 2, 3]),
        (6, 4, 2, &[0, 4, 1, 2, 3, 5]),
        // More input channels than coded channels: the bitrate limit of the
        // encoder counts input channels.
        (6, 2, 1, &[255, 2, 0, 0, 1, 2]),
        (8, 1, 0, &[0, 255, 255, 255, 255, 255, 255, 255]),
    ];
    let mut cases = 0;
    for fs in [48_000, 16_000] {
        for (channels, streams, coupled, mapping) in layouts {
            let layout = ChannelLayout::new(channels, streams, coupled, mapping).unwrap();
            layout.validate_for_encoder().unwrap();
            for frame_div in [50, 100, 400] {
                let frame_size = (fs / frame_div) as usize;
                for bitrate in BITRATES {
                    let mut enc =
                        CMsEncoder::generic(fs, channels, streams, coupled, mapping, APP_AUDIO)
                            .unwrap();
                    let diff = compare_rates(
                        &mut enc,
                        &layout,
                        MappingType::None,
                        None,
                        bitrate,
                        frame_size,
                    );
                    assert_eq!(
                        diff, None,
                        "fs {fs}, mapping {mapping:?}, frame {frame_size}, bitrate {bitrate}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 2 * 8 * 3 * 14);
}

#[test]
fn rate_allocation_has_an_lfe_stream_only_in_surround_mode() {
    // libopus keeps an LFE stream only in a surround encoder. The generic
    // constructor with the 5.1 layout treats the last stream as an ordinary
    // mono stream, and an ambisonics encoder has no LFE stream at all.
    // `rate_allocation` must do the same when the mapping type is not
    // `Surround`, even if the caller names an LFE stream.
    let (layout, mapping_type, lfe_stream) = ChannelLayout::surround(6, 1).unwrap();
    assert_eq!((mapping_type, lfe_stream), (MappingType::Surround, Some(3)));
    let (ambisonics, ambisonics_type, _) = ChannelLayout::surround(6, 2).unwrap();
    assert_eq!(ambisonics_type, MappingType::Ambisonics);
    let mut cases = 0;
    for fs in [48_000, 16_000] {
        for frame_div in [50, 400] {
            let frame_size = (fs / frame_div) as usize;
            for bitrate in BITRATES {
                let what = format!("fs {fs}, frame {frame_size}, bitrate {bitrate}");
                let mut enc = CMsEncoder::generic(
                    fs,
                    6,
                    layout.streams(),
                    layout.coupled_streams(),
                    layout.mapping(),
                    APP_AUDIO,
                )
                .unwrap();
                let diff = compare_rates(
                    &mut enc,
                    &layout,
                    MappingType::None,
                    lfe_stream,
                    bitrate,
                    frame_size,
                );
                assert_eq!(diff, None, "generic 5.1, {what}");

                let mut enc = CMsEncoder::surround(fs, 6, 2, APP_AUDIO).unwrap();
                let diff = compare_rates(
                    &mut enc,
                    &ambisonics,
                    MappingType::Ambisonics,
                    Some(ambisonics.streams() - 1),
                    bitrate,
                    frame_size,
                );
                assert_eq!(diff, None, "ambisonics, {what}");
                cases += 2;
            }
        }
    }
    assert_eq!(cases, 2 * 2 * 14 * 2);

    // With and without the LFE stream the values differ, so the comparison
    // above can tell the two apart.
    let mut surround = [0i32; 4];
    let mut generic = [0i32; 4];
    rate_allocation(
        &layout,
        MappingType::Surround,
        lfe_stream,
        256_000,
        960,
        48_000,
        &mut surround,
    )
    .unwrap();
    rate_allocation(
        &layout,
        MappingType::None,
        lfe_stream,
        256_000,
        960,
        48_000,
        &mut generic,
    )
    .unwrap();
    assert_eq!(surround, [95_120, 95_120, 57_560, 8_195]);
    assert_eq!(generic, [78_666, 78_666, 49_333, 49_333]);
}

/// Deterministic pseudo-random numbers for the tests below.
struct Rng(u32);

impl Rng {
    /// A number from 0 to `n - 1`.
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as usize % n
    }
}

/// A random layout that an encoder accepts, with at most `max_channels`
/// input channels, as (channels, streams, coupled streams, mapping). Every
/// decoded channel has an input channel. The other input channels are not
/// coded or repeat a decoded channel. The order is random.
fn random_encoder_layout(rng: &mut Rng, max_channels: usize) -> (usize, usize, usize, Vec<u8>) {
    loop {
        let channels = 1 + rng.below(max_channels);
        let streams = 1 + rng.below(channels);
        let coupled = rng.below(streams + 1);
        let coded_channels = streams + coupled;
        if coded_channels > channels {
            continue;
        }
        let mut mapping: Vec<u8> = (0..coded_channels as u8).collect();
        while mapping.len() < channels {
            mapping.push(if rng.below(2) == 0 {
                255
            } else {
                rng.below(coded_channels) as u8
            });
        }
        for i in (1..mapping.len()).rev() {
            mapping.swap(i, rng.below(i + 1));
        }
        return (channels, streams, coupled, mapping);
    }
}

/// Every frame size of libopus as a fraction of a second: 2.5, 5, 10, 20, 40,
/// 60, 80, 100 and 120 ms.
const FRAME_FRACTIONS: [(i32, i32); 9] = [
    (1, 400),
    (1, 200),
    (1, 100),
    (1, 50),
    (1, 25),
    (3, 50),
    (2, 25),
    (1, 10),
    (3, 25),
];

#[test]
fn rate_allocation_matches_libopus_for_random_generic_layouts() {
    // Layouts with up to 40 input channels, with input channels that are not
    // coded or that repeat a decoded channel. Every sampling rate, frame
    // size and application, and three kinds of input signal: the allocation
    // depends on none of the last two.
    let mut rng = Rng(0x00C0_FFEE);
    let cases = 600;
    // Stream bitrates that `reported` passed on unchanged. Only they compare
    // the exact value of the allocation with libopus.
    let mut exact = 0;
    let mut streams_total = 0;
    for case in 0..cases {
        let (channels, streams, coupled, mapping) = random_encoder_layout(&mut rng, 40);
        let layout = ChannelLayout::new(channels, streams, coupled, &mapping).unwrap();
        layout.validate_for_encoder().unwrap();
        let fs = [8_000, 12_000, 16_000, 24_000, 48_000][rng.below(5)];
        let (num, den) = FRAME_FRACTIONS[rng.below(9)];
        let frame_size = (fs * num / den) as usize;
        let application = [APP_VOIP, APP_AUDIO, APP_LOWDELAY][rng.below(3)];
        let ch = channels as i32;
        let coded_channels = (streams + coupled) as i32;
        // libopus multiplies in 32 bits, which holds up to about 8.4 Mb/s for
        // one stream. The test after this one covers the values above.
        let bitrate = match rng.below(8) {
            0 => OPUS_AUTO,
            1 => OPUS_BITRATE_MAX,
            2 => 1 + rng.below(3_000) as i32,
            3 => 500 * ch - 2 + rng.below(5) as i32,
            4 => (750_000 * ch - 2 + rng.below(5) as i32).min(8_000_000),
            5 => 1 + rng.below(8_000_000) as i32,
            _ => 1 + rng.below(100_000 * coded_channels as usize) as i32,
        };
        let pcm = match rng.below(3) {
            0 => voiced(fs, channels, frame_size),
            1 => noise(channels, frame_size, 0.3, case as u32),
            _ => silence(channels, frame_size),
        };
        let mut enc =
            CMsEncoder::generic(fs, channels, streams, coupled, &mapping, application).unwrap();
        let diff = compare_rates_for_signal(
            &mut enc,
            &layout,
            MappingType::None,
            None,
            bitrate,
            frame_size,
            &pcm,
        );
        assert_eq!(
            diff, None,
            "case {case}: fs {fs}, {channels} channels, {streams} streams, {coupled} coupled, mapping {mapping:?}, frame {frame_size}, application {application}, bitrate {bitrate}"
        );

        let mut rate = [0i32; 255];
        let user = clamp_user_bitrate(bitrate, channels).unwrap();
        rate_allocation(
            &layout,
            MappingType::None,
            None,
            user,
            frame_size,
            fs,
            &mut rate,
        )
        .unwrap();
        // The report limit is lowest when the whole frame is coded as one.
        exact += (0..streams)
            .filter(|&s| {
                let stream_channels = if s < coupled { 2 } else { 1 };
                reported(rate[s], stream_channels, fs, frame_size) == rate[s]
            })
            .count();
        streams_total += streams;
    }
    println!(
        "random generic layouts: {cases} cases, {streams_total} streams, {exact} compared exactly"
    );
    assert!(
        exact * 2 > streams_total,
        "only {exact} of {streams_total} stream bitrates were below the limits of the stream encoder"
    );
}

#[test]
fn rate_allocation_above_the_32_bit_range_of_libopus() {
    // One coupled stream, and many input channels that are not coded. The
    // bitrate limit of the multistream encoder counts input channels, so the
    // stream can be given far more than it can use. For the rate of a coupled
    // stream libopus multiplies `channel_rate` by 512 in 32 bits. Above about
    // 8.4 Mb/s for one stream the product does not fit, and the C standard
    // leaves the result open.
    //
    // `rate_allocation` computes in 64 bits and gives the stream the whole
    // bitrate. The libopus 1.6.1 build these tests link behaves the same: its
    // stream encoder ends at its own limit of 750 000 b/s per channel. A
    // product that wrapped around in 32 bits would give a few kb/s instead.
    // If this test fails with another build of libopus while the checks on
    // `rate` hold, that build wraps, and opus1 is still right.
    let mut above = 0;
    for channels in [12usize, 16, 64, 255] {
        let mut mapping = vec![255u8; channels];
        mapping[0] = 0;
        mapping[1] = 1;
        let layout = ChannelLayout::new(channels, 1, 1, &mapping).unwrap();
        layout.validate_for_encoder().unwrap();
        for bitrate in [
            8_380_000,
            8_450_000,
            9_000_000,
            750_000 * channels as i32,
            i32::MAX,
        ] {
            for frame_size in [120usize, 960] {
                let fs = 48_000;
                let what = format!("{channels} channels, bitrate {bitrate}, frame {frame_size}");
                let user = clamp_user_bitrate(bitrate, channels).unwrap();
                let mut rate = [0i32; 1];
                let sum = rate_allocation(
                    &layout,
                    MappingType::None,
                    None,
                    user,
                    frame_size,
                    fs,
                    &mut rate,
                )
                .unwrap();
                // The one stream gets everything. The division by two of the
                // allocation can lose 1 b/s.
                assert!(
                    rate[0] == user || rate[0] == user - 1,
                    "{what}: {}",
                    rate[0]
                );
                assert_eq!(sum, rate[0], "{what}");
                // `channel_rate` is half of what is left after the offsets.
                let channel_offset = 40 * (fs / frame_size as i32).max(50) as i64;
                let channel_rate = (user as i64 - 20_000 - 2 * channel_offset) / 2;
                above += (channel_rate * 512 > i32::MAX as i64) as usize;

                let mut enc = CMsEncoder::generic(fs, channels, 1, 1, &mapping, APP_AUDIO).unwrap();
                let diff = compare_rates(
                    &mut enc,
                    &layout,
                    MappingType::None,
                    None,
                    bitrate,
                    frame_size,
                );
                assert_eq!(diff, None, "{what}");
                // At 2.5 ms the limit of the stream encoder is what libopus
                // reports. At 20 ms a packet cannot hold that much.
                let want = if frame_size == 120 {
                    1_500_000
                } else {
                    510_400
                };
                assert_eq!(enc.stream_bitrate(0), want, "{what}");
            }
        }
    }
    // All cases but the first bitrate are above the 32-bit range.
    assert_eq!(above, 4 * 4 * 2);
}

#[test]
fn rate_allocation_matches_libopus_at_the_bitrate_limits() {
    // The multistream encoder keeps an explicit bitrate between 500 and
    // 750 000 bits per second per channel. Values at and beyond both limits
    // must give the stream bitrates of libopus.
    let mut cases = 0;
    for (family, channels) in [
        (1u8, 1usize),
        (1, 2),
        (1, 6),
        (1, 8),
        (255, 3),
        (2, 6),
        (2, 11),
    ] {
        let ch = channels as i32;
        let bitrates = [
            1,
            499,
            500 * ch - 1,
            500 * ch,
            500 * ch + 1,
            750_000 * ch - 1,
            750_000 * ch,
            750_000 * ch + 1,
            i32::MAX,
        ];
        cases += run_grid(
            &[48_000, 16_000],
            &[(family, channels)],
            &[(1, 50), (1, 400), (3, 50)],
            &bitrates,
        );
    }
    assert_eq!(cases, 7 * 2 * 3 * 9);
}

#[test]
fn user_bitrate_clamp_matches_libopus() {
    // The multistream encoder accepts or rejects a bitrate like
    // `clamp_user_bitrate`. The clamped value shows in the stream bitrates,
    // which the tests above compare; here only accept or reject is compared.
    for channels in [1usize, 2, 6, 8] {
        let mut enc = CMsEncoder::surround(48_000, channels, 1, APP_AUDIO).unwrap();
        for value in [
            OPUS_AUTO,
            OPUS_BITRATE_MAX,
            -1001,
            -999,
            -2,
            0,
            1,
            499,
            500,
            64_000,
            6_000_001,
            i32::MAX,
        ] {
            let c_ok = enc.try_ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, value) == 0;
            let rs = clamp_user_bitrate(value, channels);
            assert_eq!(rs.is_ok(), c_ok, "{channels} channels, bitrate {value}");
        }
    }
}

/// Counters of `check_budget`, to show that the cases reach every rule.
#[derive(Default, Debug)]
struct BudgetStats {
    cases: usize,
    too_small: usize,
    cbr_packets: usize,
    /// Constant bitrate packets smaller than the output buffer.
    cbr_below_buffer: usize,
    /// Streams that used every byte they were allowed.
    streams_at_limit: usize,
    /// Streams whose limit left 2 bytes for the length field.
    two_byte_length: usize,
    /// Packets of the smallest possible size.
    smallest_packets: usize,
    last_stream_rates: usize,
}

/// One frame through the C encoder with an output buffer of `buffer` bytes,
/// against the byte budget functions.
#[allow(clippy::too_many_arguments)]
fn check_budget(
    family: u8,
    channels: usize,
    fs: i32,
    frame_size: usize,
    bitrate: i32,
    vbr: bool,
    buffer: usize,
    stats: &mut BudgetStats,
) {
    let what = format!(
        "family {family}, {channels} channels, fs {fs}, frame {frame_size}, bitrate {bitrate}, vbr {vbr}, buffer {buffer}"
    );
    let (layout, mapping_type, lfe_stream) = ChannelLayout::surround(channels, family).unwrap();
    let streams = layout.streams();
    let mut enc = CMsEncoder::surround(fs, channels, family as i32, APP_AUDIO).unwrap();
    enc.set_bitrate(bitrate).set_vbr(vbr);
    let pcm = noise(channels, frame_size, 0.1, 5);
    let result = enc.encode_float(&pcm, frame_size, buffer);
    stats.cases += 1;

    // A buffer below the smallest packet is an error on both sides.
    let smallest = smallest_packet(streams, frame_size, fs).unwrap();
    if buffer < smallest {
        assert_eq!(result, Err(OPUS_BUFFER_TOO_SMALL), "{what}");
        // Except at 100 ms, where every stream needs 2 bytes, the budget of
        // the first stream alone already shows that the buffer is too small.
        if fs as usize / frame_size != 10 {
            assert!(
                stream_max_bytes(buffer, 0, streams, 0, frame_size, fs).is_err(),
                "{what}"
            );
        }
        stats.too_small += 1;
        return;
    }
    let packet = result.unwrap_or_else(|e| panic!("{what}: C encode failed: {e}"));

    // The size limit of the whole packet.
    let user = clamp_user_bitrate(bitrate, channels).unwrap();
    let mut rate = [0i32; 255];
    let rate_sum = rate_allocation(
        &layout,
        mapping_type,
        lfe_stream,
        user,
        frame_size,
        fs,
        &mut rate,
    )
    .unwrap();
    let max_data_bytes = if vbr {
        buffer
    } else {
        cbr_max_data_bytes(user, rate_sum, streams, frame_size, fs, buffer).unwrap()
    };
    assert!(
        max_data_bytes <= buffer && max_data_bytes >= smallest,
        "{what}: limit {max_data_bytes}"
    );
    if vbr {
        assert!(
            packet.len() <= max_data_bytes,
            "{what}: packet of {} bytes",
            packet.len()
        );
    } else {
        // With constant bitrate the last stream is padded up to the limit.
        assert_eq!(
            packet.len(),
            max_data_bytes,
            "{what}: constant bitrate packet size"
        );
        stats.cbr_packets += 1;
        stats.cbr_below_buffer += (max_data_bytes < buffer) as usize;
    }

    // The size limit of every stream. `tot_size` is what the streams before
    // it really took in the C packet.
    let parts = split_streams(&packet, streams).unwrap_or_else(|e| panic!("{what}: {e}"));
    let mut tot_size = 0;
    for (s, part) in parts.iter().enumerate() {
        let curr_max = stream_max_bytes(max_data_bytes, tot_size, streams, s, frame_size, fs)
            .unwrap_or_else(|e| panic!("{what}: stream {s} after {tot_size} bytes: {e}"));
        assert!(curr_max <= MS_FRAME_TMP, "{what}");
        if max_data_bytes == smallest {
            // In the smallest packet every stream has exactly its limit: 1
            // byte, or 2 bytes for a 100 ms frame.
            assert_eq!(
                part.packet.len(),
                curr_max,
                "{what}: stream {s} of the smallest packet"
            );
            stats.smallest_packets += (s == 0) as usize;
        }
        if s + 1 < streams {
            // The stream and its length field fit in front of the minimum
            // that the later streams need.
            let length_field = part.used - part.packet.len();
            assert!(
                part.packet.len() <= curr_max,
                "{what}: stream {s} has {} bytes, the limit is {curr_max}",
                part.packet.len()
            );
            assert!(
                tot_size + part.used + 2 * (streams - s - 1) - 1 <= max_data_bytes,
                "{what}: stream {s}"
            );
            stats.streams_at_limit += (part.packet.len() == curr_max) as usize;
            stats.two_byte_length += (part.packet.len() == curr_max && length_field == 2) as usize;
        } else if !vbr {
            // libopus sets the last stream to the bitrate that fills its
            // bytes, then pads the packet to the full size.
            let want = cbr_last_stream_bitrate(curr_max, frame_size, fs).unwrap();
            let coded_frame = toc(part.toc).frame_48k * fs as usize / 48_000;
            let want = reported(
                want,
                if s < layout.coupled_streams() { 2 } else { 1 },
                fs,
                coded_frame,
            );
            assert_eq!(
                enc.stream_bitrate(s),
                want,
                "{what}: bitrate of the last stream ({curr_max} bytes)"
            );
            stats.last_stream_rates += 1;
        }
        tot_size += part.used;
    }
    assert_eq!(tot_size, packet.len(), "{what}");
}

#[test]
fn byte_budget_matches_libopus() {
    let mut stats = BudgetStats::default();
    for (family, channels) in [(1u8, 6usize), (1, 8), (255, 3), (2, 6), (1, 2), (1, 1)] {
        let streams = ChannelLayout::surround(channels, family)
            .unwrap()
            .0
            .streams();
        for fs in [48_000, 16_000] {
            // 2.5, 10, 20, 40, 60 and 100 ms. 100 ms has its own rule.
            for (num, den) in [(1, 400), (1, 100), (1, 50), (1, 25), (3, 50), (1, 10)] {
                let frame_size = (fs * num / den) as usize;
                let smallest = smallest_packet(streams, frame_size, fs).unwrap();
                // Around the smallest packet, around the point where a length
                // field grows to 2 bytes, and far above every packet.
                let first = 2 * streams + 250;
                let buffers = [
                    smallest.saturating_sub(1),
                    smallest,
                    smallest + 1,
                    smallest + 9,
                    100,
                    first,
                    first + 1,
                    first + 2,
                    max_packet_bytes(streams),
                ];
                for bitrate in [OPUS_AUTO, OPUS_BITRATE_MAX, 16_000, 96_000, 510_000] {
                    for vbr in [false, true] {
                        for buffer in buffers {
                            check_budget(
                                family, channels, fs, frame_size, bitrate, vbr, buffer, &mut stats,
                            );
                        }
                    }
                }
            }
        }
    }
    println!("{stats:?}");
    assert_eq!(stats.cases, 6 * 2 * 6 * 9 * 5 * 2);
    // Every rule was reached many times.
    assert!(stats.too_small >= 100, "{stats:?}");
    assert!(stats.cbr_packets >= 1000, "{stats:?}");
    assert!(stats.cbr_below_buffer >= 100, "{stats:?}");
    assert!(stats.streams_at_limit >= 100, "{stats:?}");
    assert!(stats.two_byte_length >= 10, "{stats:?}");
    assert!(stats.smallest_packets >= 100, "{stats:?}");
    assert!(stats.last_stream_rates >= 1000, "{stats:?}");
}

#[test]
fn byte_budget_matches_libopus_for_many_streams() {
    // The bytes kept back for later streams grow with the number of streams.
    // With 40, 120 and 255 mono streams they are most of a small packet.
    let mut stats = BudgetStats::default();
    // (channels, sampling rate, frame size as a fraction of a second)
    for (channels, fs, num, den) in [
        (40usize, 48_000, 1, 50),
        (40, 48_000, 1, 10),
        (120, 16_000, 1, 50),
        (255, 48_000, 1, 400),
    ] {
        let frame_size = (fs * num / den) as usize;
        // Mapping family 255: one mono stream per channel.
        let streams = channels;
        let smallest = smallest_packet(streams, frame_size, fs).unwrap();
        let buffers = [
            smallest - 1,
            smallest,
            smallest + 1,
            smallest + streams,
            3 * streams + 300,
            max_packet_bytes(streams),
        ];
        let ch = channels as i32;
        for bitrate in [OPUS_AUTO, 500 * ch, 6_000 * ch, 64_000 * ch] {
            for vbr in [false, true] {
                for buffer in buffers {
                    check_budget(
                        255, channels, fs, frame_size, bitrate, vbr, buffer, &mut stats,
                    );
                }
            }
        }
    }
    println!("{stats:?}");
    assert_eq!(stats.cases, 4 * 6 * 4 * 2);
    assert_eq!(stats.too_small, 4 * 4 * 2, "{stats:?}");
    assert!(stats.cbr_packets >= 60, "{stats:?}");
    assert!(stats.cbr_below_buffer >= 20, "{stats:?}");
    assert!(stats.streams_at_limit >= 1000, "{stats:?}");
    assert!(stats.smallest_packets >= 20, "{stats:?}");
    assert!(stats.last_stream_rates >= 60, "{stats:?}");
}

#[test]
fn smallest_packet_matches_libopus() {
    // libopus refuses an output buffer below the smallest packet and accepts
    // one of exactly that size, for every frame size it supports.
    for (family, channels) in [(1u8, 1usize), (1, 2), (1, 6), (1, 8), (255, 5), (2, 11)] {
        let streams = ChannelLayout::surround(channels, family)
            .unwrap()
            .0
            .streams();
        for fs in [48_000, 8_000] {
            // 2.5, 5, 10, 20, 40, 60, 80, 100, 120 ms.
            for (num, den) in [
                (1, 400),
                (1, 200),
                (1, 100),
                (1, 50),
                (1, 25),
                (3, 50),
                (2, 25),
                (1, 10),
                (3, 25),
            ] {
                let frame_size = (fs * num / den) as usize;
                let smallest = smallest_packet(streams, frame_size, fs).unwrap();
                let want = if num == 1 && den == 10 {
                    3 * streams - 1
                } else {
                    2 * streams - 1
                };
                assert_eq!(smallest, want);
                let what =
                    format!("family {family}, {channels} channels, fs {fs}, frame {frame_size}");
                let pcm = noise(channels, frame_size, 0.1, 9);
                for vbr in [true, false] {
                    let mut enc =
                        CMsEncoder::surround(fs, channels, family as i32, APP_AUDIO).unwrap();
                    enc.set_vbr(vbr);
                    if smallest > 1 {
                        assert_eq!(
                            enc.encode_float(&pcm, frame_size, smallest - 1),
                            Err(OPUS_BUFFER_TOO_SMALL),
                            "{what}"
                        );
                    }
                    let packet = enc
                        .encode_float(&pcm, frame_size, smallest)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(packet.len() <= smallest, "{what}");
                    assert!(split_streams(&packet, streams).is_ok(), "{what}");
                }
            }
        }
    }
}
