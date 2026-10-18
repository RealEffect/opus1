//! The rate rules of libopus `celt_encode_with_ec` (celt/celt_encoder.c),
//! checked against the CELT encoder of libopus itself:
//!
//! - the bits per frame under VBR, `bitrate_to_bits`, at bitrates that do
//!   not divide into whole bits per frame;
//! - the limit of the bitrate to 750 kb/s per channel (`OPUS_SET_BITRATE`);
//! - the byte count of a silent frame under VBR (`effectiveBytes`).
//!
//! Both encoders get the same samples and settings. At complexity 0 they
//! write the same packets in these settings, so every packet must be the
//! same here.

mod celtenc_common;

use celtenc_common::c_celt::Stream;
use celtenc_common::*;

/// Encodes `pcm` with both encoders and asserts that every packet is the
/// same. Returns the sizes of the packets of libopus.
fn assert_same_packets(st: &Stream, pcm: &[f32], what: &str) -> Vec<usize> {
    let c = st.enc_c(pcm);
    let rs = st.enc_rs(pcm);
    assert_eq!(c.len(), rs.len(), "{what}");
    for (k, (pc, pr)) in c.iter().zip(&rs).enumerate() {
        assert!(
            pc == pr,
            "{what}: frame {k} differs: libopus {} bytes, opus1 {} bytes",
            pc.len(),
            pr.len()
        );
    }
    c.iter().map(|p| p.len()).collect()
}

/// `seconds` of the music signal for `channels` channels, interleaved.
fn music_for(channels: usize, seconds: usize) -> Vec<f32> {
    let left = music(FS, seconds, 0);
    if channels == 1 {
        return left;
    }
    let right = music(FS, seconds, 1);
    left.iter().zip(&right).flat_map(|(&l, &r)| [l, r]).collect()
}

/// A stream at 48 kHz with variable bitrate and complexity 0.
fn vbr_stream(channels: usize, frame_ms_x2: usize, bitrate_per_channel: i32) -> Stream {
    let mut st = Stream::new(FS, channels, frame_ms_x2, bitrate_per_channel * channels as i32);
    st.vbr = true;
    st.complexity = 0;
    st
}

/// libopus computes the bits of a frame as `bitrate * 6 / (6 * Fs / N)`,
/// rounded down, and the target in 1/8 bit from that. These bitrates leave
/// a remainder at every frame size.
#[test]
fn vbr_at_bitrates_that_are_not_whole_bits_per_frame() {
    for channels in [1usize, 2] {
        let pcm = music_for(channels, 1);
        for frame_ms_x2 in [5usize, 10, 20, 40] {
            for bitrate in [6_001, 24_007, 50_001, 96_333] {
                let st = vbr_stream(channels, frame_ms_x2, bitrate);
                let what = format!("{channels} ch, frame {frame_ms_x2}/2 ms, {bitrate} b/s per channel");
                assert_same_packets(&st, &pcm, &what);
            }
        }
    }
}

/// libopus limits the bitrate of its CELT encoder to 750 kb/s per channel.
/// A frame is never larger than 510 kb/s allows. The bitrate also sets the
/// byte count the analyses go by (`effectiveBytes`) and the state of the
/// VBR control, but no packet of this test depends on the limit itself:
/// without it opus1 writes the same packets. The unit test
/// `bitrate_is_limited_to_750_kbps_per_channel` in src/celt.rs checks the
/// limit. This test still catches rules that go by the byte count, such as
/// a boost of band 0 above 320 bytes per frame, which libopus 1.6.1 does
/// not have.
///
/// Above the limit each encoder must write the packets it writes at the
/// limit, and the packets of the two encoders must be the same.
#[test]
fn bitrate_is_limited_to_750_kbps_per_channel() {
    for channels in [1usize, 2] {
        let pcm = music_for(channels, 1);
        for frame_ms_x2 in [5usize, 10] {
            let at_limit = vbr_stream(channels, frame_ms_x2, 750_000);
            let (c_limit, rs_limit) = (at_limit.enc_c(&pcm), at_limit.enc_rs(&pcm));
            for bitrate in [760_000, 1_000_000, 1_500_000] {
                let st = vbr_stream(channels, frame_ms_x2, bitrate);
                let what = format!("{channels} ch, frame {frame_ms_x2}/2 ms, {bitrate} b/s per channel");
                assert!(st.enc_c(&pcm) == c_limit, "{what}: libopus does not limit the bitrate");
                assert!(st.enc_rs(&pcm) == rs_limit, "{what}: opus1 does not limit the bitrate");
                assert_same_packets(&st, &pcm, &what);
            }
        }
    }
}

/// Under VBR libopus codes a silent frame in two bytes and takes those two
/// bytes as the frame's effective byte count for the analyses that still
/// run. The signal has silent stretches between the music, each long
/// enough for several frames. (No setting here showed a packet that
/// depends on that byte count; the test keeps silent frames under VBR
/// equal to libopus's.)
#[test]
fn silent_frames_under_vbr() {
    for channels in [1usize, 2] {
        let mut pcm = music_for(channels, 2);
        let stretch = FS / 5 * channels;
        for (k, chunk) in pcm.chunks_mut(stretch).enumerate() {
            if k % 3 == 1 {
                chunk.fill(0.0);
            }
        }
        for frame_ms_x2 in [5usize, 20, 40] {
            for complexity in [0, 4, 10] {
                let mut st = vbr_stream(channels, frame_ms_x2, 48_000);
                st.complexity = complexity;
                let what = format!("{channels} ch, frame {frame_ms_x2}/2 ms, complexity {complexity}");
                let sizes = assert_same_packets(&st, &pcm, &what);
                assert!(sizes.contains(&2), "{what}: no silent frame");
            }
        }
    }
}

/// The silence flag of a stereo input coded as mono.
///
/// libopus measures the peak level of a frame over `C*(N-overlap)` and
/// `C*overlap` samples of the input (celt_encoder.c, lines 1969-1971).
/// `C` is the number of coded channels, but the input holds both input
/// channels, interleaved. So with one coded channel libopus measures only
/// the first half of the frame. A frame whose first half is digital silence
/// is coded as a silent frame even when its second half has a signal.
///
/// The signal has stretches of digital silence of 13 to 96 ms, so their
/// ends fall at every place in a frame. The test counts the frames where
/// measuring the whole frame would give another flag, and requires some.
#[test]
fn silence_flag_of_stereo_input_coded_as_mono() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut pick = move |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 16) as usize % n
    };
    // A debug build (the `no_std` check runs one) codes a part of the matrix.
    let debug = cfg!(debug_assertions);
    let rates: &[usize] = if debug { &[48_000, 8_000] } else { &[48_000, 16_000, 8_000] };
    let durations: &[usize] = if debug { &[5, 20] } else { &[5, 10, 20, 40] };
    let settings: &[(bool, i32)] = if debug { &[(true, 0), (false, 0)] } else { &[(true, 0), (true, 5), (false, 0), (false, 10)] };
    for &fs in rates {
        let left = music(fs, 2, 0);
        let right = music(fs, 2, 1);
        let mut pcm: Vec<f32> = left.iter().zip(&right).flat_map(|(&l, &r)| [l, r]).collect();
        let mut pos = 0;
        let mut silent = false;
        while pos < left.len() {
            let len = (fs * (13 + pick(84)) / 1000).min(left.len() - pos);
            if silent {
                pcm[2 * pos..2 * (pos + len)].fill(0.0);
            }
            silent = !silent;
            pos += len;
        }
        let threshold = 1.0 / (1u32 << 24) as f32;
        for &frame_ms_x2 in durations {
            let mut st = Stream::new(fs, 2, frame_ms_x2, 32_000);
            st.stream_channels = 1;
            // Frames where libopus's measure and a measure of the whole
            // frame disagree.
            let api_overlap = 120 * fs / 48_000;
            let (mut first_half_max, mut whole_max) = (0f32, 0f32);
            let mut disagree = 0;
            for frame in pcm.chunks_exact(st.frame * 2) {
                let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
                let body = st.frame - api_overlap;
                let first_half = peak(&frame[..body]).max(first_half_max);
                first_half_max = peak(&frame[body..st.frame]);
                let whole = peak(&frame[..2 * body]).max(whole_max);
                whole_max = peak(&frame[2 * body..]);
                let by_libopus = first_half.max(first_half_max) <= threshold;
                let by_whole_frame = whole.max(whole_max) <= threshold;
                disagree += (by_libopus != by_whole_frame) as usize;
            }
            assert!(disagree > 0, "{fs} Hz, frame {frame_ms_x2}/2 ms: no frame shows the rule");
            for &(vbr, complexity) in settings {
                st.vbr = vbr;
                st.complexity = complexity;
                let what = format!(
                    "{fs} Hz, frame {frame_ms_x2}/2 ms, vbr {vbr}, complexity {complexity}, {disagree} frames"
                );
                assert_same_packets(&st, &pcm, &what);
            }
        }
    }
}
