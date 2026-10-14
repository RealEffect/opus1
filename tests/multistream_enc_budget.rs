//! The byte budget of the opus-rs multistream encoder, stream by stream,
//! against the C libopus encoder.
//!
//! tests/multistream_enc_structure.rs compares the size of a whole
//! constant-bitrate packet and the bitrates of the streams under variable
//! bitrate. The tests here look inside the constant-bitrate packet:
//!
//! - With the application `RestrictedLowDelay` every stream is CELT, and a
//!   CELT stream at constant bitrate has exactly the bytes of its bitrate.
//!   The two encoders must then agree on every stream: its size, its
//!   padding and the bitrate its encoder was set to. For the last stream
//!   that bitrate is what the other streams left.
//! - Constant bitrate with the maximum bitrate: the packet fills the output
//!   buffer, whatever its size.
//! - `bitrate()` after a packet that has streams without audio.
//!
//! One fault of the single-stream encoder (src/lib.rs) shows here. An
//! ignored test holds the full rule.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use audiopus_sys as sys;
use ms_support::*;
use multistream_enc_common::*;
use opus_rs::{Application, OpusEncoder};

/// What a C stream encoder reports for `OPUS_GET_BITRATE` when it was set to
/// `rate` and its last coded frame has `coded_frame` samples: at most what
/// 1276 bytes per frame carry (opus_encoder.c, `user_bitrate_to_bitrate`).
fn reported(rate: i32, fs: i32, coded_frame: usize) -> i32 {
    let cap = (1276i64 * 8 * (6 * fs as i64 / coded_frame as i64) / 6) as i32;
    rate.min(cap)
}

/// Whether a constant-bitrate stream at `rate` bits per second has 20 to 23
/// bits per frame.
///
/// Both encoders write a packet without audio when the bitrate is below 3
/// bytes per frame. libopus first rounds the bitrate to the size of the
/// constant-bitrate packet (opus_encoder.c:1331). A bitrate of 20 to 23
/// bits per frame rounds to 3 bytes, so the frame is coded.
/// `a_cbr_stream_of_three_bytes_is_coded_like_libopus` shows it on a single
/// stream. opus-rs once tested the bitrate before rounding it and did not
/// code such a frame.
fn rounds_up_to_three_bytes(rate: i32, frame_size: usize, fs: i32) -> bool {
    let bits = rate as i64 * frame_size as i64 / fs as i64;
    (20..24).contains(&bits)
}

#[test]
fn low_delay_cbr_streams_have_the_sizes_and_bitrates_of_libopus() {
    let application = Application::RestrictedLowDelay;
    let mut differences = Vec::new();
    // Frames with a stream of 20 to 23 bits, which libopus still codes.
    let (mut frames, mut three_bytes) = (0, 0);
    for fs in [48_000, 24_000, 16_000] {
        for case in CASES {
            let channels = case.channels as i32;
            for per_channel in [4_000, 9_000, 24_000, 70_000, 400_000] {
                for frame_size in frame_sizes(fs) {
                    let (mut rust, mut c) = encoders(case, fs, application);
                    configure(&mut rust, &mut c, per_channel * channels, false);
                    let streams = c.streams;
                    let pcm = noise(case.channels, 4 * frame_size, 0.3, 11);
                    let mut out = vec![0u8; max_packet_bytes(streams)];
                    for (f, frame) in pcm.chunks_exact(frame_size * case.channels).enumerate() {
                        let label = format!(
                            "{} {fs} Hz {per_channel} b/s per channel {} frame {f}",
                            case.name,
                            ms(frame_size, fs)
                        );
                        let n = rust
                            .encode(frame, frame_size, &mut out)
                            .unwrap_or_else(|e| panic!("{label}: {e}"));
                        let c_packet = c.encode_float(frame, frame_size, out.len()).unwrap();
                        // The streams before the last keep the bitrate of
                        // the allocation.
                        three_bytes += usize::from((0..streams - 1).any(|s| {
                            let rate = rust.stream_encoder(s).unwrap().bitrate_bps;
                            rounds_up_to_three_bytes(rate, frame_size, fs)
                        }));
                        frames += 1;
                        assert_eq!(n, c_packet.len(), "{label}: packet size");
                        let parts = split_streams(&out[..n], streams).unwrap();
                        let c_parts = split_streams(&c_packet, streams).unwrap();
                        for s in 0..streams {
                            let (a, b) = (&parts[s], &c_parts[s]);
                            if (a.used, a.padding, a.frames(), a.payload_len)
                                != (b.used, b.padding, b.frames(), b.payload_len)
                            {
                                differences.push(format!(
                                    "{label}: stream {s}: {a} with {} bytes of padding, C {b} with {}",
                                    a.padding, b.padding
                                ));
                            }
                            let set = rust.stream_encoder(s).unwrap().bitrate_bps;
                            let c_frame = b.samples(fs) / b.frames();
                            if reported(set, fs, c_frame) != c.stream_bitrate(s) {
                                differences.push(format!(
                                    "{label}: stream {s}: bitrate {set}, C {}",
                                    c.stream_bitrate(s)
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "{frames} frames compared, {three_bytes} of them with a stream of 20 to 23 bits, {} \
         differences",
        differences.len()
    );
    for d in differences.iter().take(30) {
        println!("{d}");
    }
    assert!(differences.is_empty(), "{} differences", differences.len());
    // Every frame is compared, and the settings reach the streams of 20
    // to 23 bits. Measured: 3960 and 24.
    assert!(
        frames == 3960 && three_bytes >= 20,
        "{frames} compared, {three_bytes} with a stream of 20 to 23 bits"
    );
}

/// C single-stream encoder, freed on drop.
struct CEncoder(*mut sys::OpusEncoder);

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.0) }
    }
}

/// The rule behind `rounds_up_to_three_bytes`, on a plain `OpusEncoder`
/// without any multistream code: under constant bitrate libopus rounds the
/// bitrate to the packet size before it decides that a frame has too few
/// bits (opus_encoder.c:1331). The single-stream encoder does the same now;
/// this test was ignored while it used the bitrate that was set.
#[test]
fn a_cbr_stream_of_three_bytes_is_coded_like_libopus() {
    let fs = 48_000;
    let mut differ = Vec::new();
    // 20 to 23 bits per frame: 4000 b/s at 5 ms, 9000 b/s at 2.5 ms,
    // 1000 b/s at 20 ms, 2000 b/s at 10 ms.
    for (frame_size, bitrate) in [(240usize, 4_000), (120, 9_000), (960, 1_000), (480, 2_000)] {
        assert!(rounds_up_to_three_bytes(bitrate, frame_size, fs));
        let mut err = 0;
        let c = CEncoder(unsafe { sys::opus_encoder_create(fs, 1, APP_LOWDELAY, &mut err) });
        assert!(!c.0.is_null() && err == 0);
        unsafe {
            assert_eq!(
                sys::opus_encoder_ctl(c.0, sys::OPUS_SET_BITRATE_REQUEST, bitrate),
                0
            );
            assert_eq!(sys::opus_encoder_ctl(c.0, sys::OPUS_SET_VBR_REQUEST, 0), 0);
        }
        let mut rust = OpusEncoder::new(fs, 1, Application::RestrictedLowDelay).unwrap();
        rust.bitrate_bps = bitrate;
        rust.use_cbr = true;
        let pcm = noise(1, frame_size, 0.3, 3);
        let (mut a, mut b) = (vec![0u8; 1500], vec![0u8; 1500]);
        let n = rust.encode(&pcm, frame_size, &mut a).unwrap();
        let m = unsafe {
            sys::opus_encode_float(c.0, pcm.as_ptr(), frame_size as i32, b.as_mut_ptr(), 1500)
        };
        assert!(m > 0);
        let (r, cp) = (
            parse_stream(&a[..n]).unwrap(),
            parse_stream(&b[..m as usize]).unwrap(),
        );
        // libopus codes 2 bytes after the TOC byte.
        assert_eq!((cp.used, cp.payload_len), (3, 2), "{bitrate} b/s: C {cp}");
        if (r.used, r.payload_len) != (cp.used, cp.payload_len) {
            differ.push(format!("{bitrate} b/s, {frame_size} samples: {r}, C {cp}"));
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The buffers of the maximum-bitrate tests, in bytes: near the smallest
/// packet, ordinary sizes, and more than the streams can fill.
const BUFFERS: [usize; 7] = [9, 40, 300, 1_500, 4_000, 9_000, 20_000];

/// Encodes three frames at constant bitrate with the maximum bitrate into
/// buffers of `BUFFERS`. Returns the number of packets both encoders wrote,
/// and a description of every frame opus-rs refused.
fn maximum_bitrate_cbr(frame_ms: &[usize], buffers: &[usize]) -> (usize, Vec<String>) {
    let mut packets = 0;
    let mut refused = Vec::new();
    for fs in [48_000, 16_000] {
        for case in CASES {
            for &ms_ in frame_ms {
                let frame_size = fs as usize * ms_ / 1000;
                for &len in buffers {
                    for application in [Application::Audio, Application::RestrictedLowDelay] {
                        let (mut rust, mut c) = encoders(case, fs, application);
                        let streams = c.streams;
                        if len < 2 * streams - 1 {
                            continue;
                        }
                        configure(&mut rust, &mut c, OPUS_BITRATE_MAX, false);
                        let mut dec = c_decoder(&rust);
                        let pcm = voiced(fs, case.channels, 3 * frame_size);
                        for (f, frame) in pcm.chunks_exact(frame_size * case.channels).enumerate() {
                            let label = format!(
                                "{} {fs} Hz {application:?} {ms_} ms, buffer {len}, frame {f}",
                                case.name
                            );
                            let mut out = vec![0u8; len];
                            let c_packet = c
                                .encode_float(frame, frame_size, len)
                                .unwrap_or_else(|e| panic!("{label}: C error {e}"));
                            // libopus pads the packet to the whole buffer.
                            assert_eq!(c_packet.len(), len, "{label}: C packet size");
                            let n = match rust.encode(frame, frame_size, &mut out) {
                                Ok(n) => n,
                                Err(e) => {
                                    assert_eq!(e, "Output buffer too small", "{label}");
                                    refused.push(label);
                                    rust.reset_state();
                                    dec = c_decoder(&rust);
                                    continue;
                                }
                            };
                            assert_eq!(n, len, "{label}: packet size");
                            let decoded = dec
                                .decode_float(Some(&out[..n]), frame_size, false)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "{label}: C decode error {e}: {}",
                                        describe_packet(&out[..n], streams)
                                    )
                                });
                            assert_eq!(decoded.len(), frame_size * case.channels, "{label}");
                            packets += 1;
                        }
                    }
                }
            }
        }
    }
    (packets, refused)
}

/// Frames up to 20 ms: the packet has the size of the buffer, as in
/// libopus, for every buffer.
#[test]
fn constant_bitrate_with_the_maximum_bitrate_fills_the_buffer() {
    let (packets, refused) = maximum_bitrate_cbr(&[5, 10, 20], &BUFFERS);
    println!("{packets} packets");
    assert!(refused.is_empty(), "opus-rs refused: {refused:#?}");
    assert!(packets > 2500, "only {packets} packets");
}

/// 40 and 60 ms frames with a buffer the streams cannot fill: the same.
#[test]
fn constant_bitrate_with_the_maximum_bitrate_and_long_frames_in_a_large_buffer() {
    let (packets, refused) = maximum_bitrate_cbr(&[40, 60], &[9_000, 20_000]);
    println!("{packets} packets");
    assert!(refused.is_empty(), "opus-rs refused: {refused:#?}");
    assert!(packets > 500, "only {packets} packets");
}

/// 40 and 60 ms frames with a buffer the first streams fill. A later stream
/// is then left one or two bytes. libopus writes a packet without audio for
/// it; opus-rs refuses the frame, for 5.1 with buffers of up to 4000 bytes.
/// The cause is in the single-stream encoder: see
/// `tight_output_buffers_with_40_and_60_ms_frames_like_libopus` in
/// tests/multistream_enc_controls.rs.
#[test]
fn constant_bitrate_with_the_maximum_bitrate_and_long_frames_in_any_buffer() {
    let (packets, refused) = maximum_bitrate_cbr(&[40, 60], &BUFFERS);
    assert!(
        refused.is_empty(),
        "opus-rs refused {} frames and coded {packets}, for example {:#?}",
        refused.len(),
        &refused[..refused.len().min(5)]
    );
}

/// `bitrate()` counts a stream with at most what 1276 bytes per frame
/// carry, for the frame duration of the last frame the stream coded. A
/// packet without audio does not change that duration in libopus.
///
/// With the maximum bitrate a stream is set to 750 000 bits per second per
/// channel, more than a 20 ms frame carries (510 400), so the duration
/// shows in `bitrate()`. A frame coded into a large buffer is followed by
/// frames of other durations coded into the smallest buffer, where no
/// stream has room for audio.
#[test]
fn bitrate_after_a_packet_without_audio_is_that_of_libopus() {
    let fs = 48_000;
    // Every stream is CELT, so both encoders split a long frame alike.
    let application = Application::RestrictedLowDelay;
    for case in CASES {
        let (mut rust, mut c) = encoders(case, fs, application);
        configure(&mut rust, &mut c, OPUS_BITRATE_MAX, true);
        let streams = c.streams;
        let smallest = 2 * streams - 1;
        let large = max_packet_bytes(streams);
        let pcm = noise(case.channels, 2880, 0.2, 21);
        // (frame size, buffer): coded frames set the duration, the frames
        // into the smallest buffer must leave it alone.
        let steps = [
            (960usize, large),
            (120, smallest),
            (240, smallest),
            (480, smallest),
            (240, large),
            (960, smallest),
            (120, smallest),
            (2880, large),
            (120, smallest),
            (480, large),
            (960, smallest),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (k, (frame_size, len)) in steps.into_iter().enumerate() {
            let label = format!(
                "{}: step {k}, {frame_size} samples into {len} bytes",
                case.name
            );
            let mut out = vec![0u8; len];
            let n = rust
                .encode(&pcm, frame_size, &mut out)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let c_packet = c.encode_float(&pcm, frame_size, len).unwrap();
            let parts = split_streams(&out[..n], streams).unwrap();
            let c_parts = split_streams(&c_packet, streams).unwrap();
            for s in 0..streams {
                // Both encoders agree on which streams carry audio.
                assert_eq!(
                    parts[s].payload_len == 0,
                    len == smallest,
                    "{label}: stream {s}: {}",
                    parts[s]
                );
                assert_eq!(
                    c_parts[s].payload_len == 0,
                    len == smallest,
                    "{label}: C stream {s}: {}",
                    c_parts[s]
                );
            }
            assert_eq!(rust.bitrate(), c.bitrate(), "{label}: bitrate()");
            seen.insert(rust.bitrate());
        }
        // The steps must make a difference, or the test says nothing: the
        // limit of a 20 ms frame, of a 60 ms packet (three 20 ms frames)
        // and of shorter frames.
        assert!(
            seen.len() >= 2,
            "{}: bitrate() is always {seen:?}",
            case.name
        );
    }
}

/// libopus checks the buffer against the smallest packet before it does
/// anything else. A call that fails there leaves the encoder as it was: the
/// streams have not been given a bitrate, and the packets that follow are
/// those of an encoder that never saw the call.
#[test]
fn a_buffer_below_the_smallest_packet_leaves_the_encoder_untouched() {
    let fs = 48_000;
    let frame_size = 960;
    for case in CASES {
        let (mut rust, mut c) = encoders(case, fs, Application::Audio);
        let (mut fresh, _) = encoders(case, fs, Application::Audio);
        let bitrate = 40_000 * case.channels as i32;
        configure(&mut rust, &mut c, bitrate, true);
        fresh.set_bitrate(rust_bitrate(bitrate)).unwrap();
        let streams = c.streams;
        let pcm = voiced(fs, case.channels, 6 * frame_size);
        let before = rust.bitrate();
        assert_eq!(before, c.bitrate(), "{}: new encoder", case.name);
        for len in 0..2 * streams - 1 {
            let mut out = vec![0u8; len];
            assert_eq!(
                rust.encode(&pcm, frame_size, &mut out),
                Err("Output buffer too small"),
                "{}: buffer of {len} bytes",
                case.name
            );
            assert_eq!(
                c.encode_float(&pcm, frame_size, len),
                Err(OPUS_BUFFER_TOO_SMALL),
                "{}: buffer of {len} bytes",
                case.name
            );
            // No stream encoder was given a bitrate yet, in either encoder.
            assert_eq!(
                rust.bitrate(),
                before,
                "{}: buffer of {len} bytes",
                case.name
            );
            assert_eq!(c.bitrate(), before, "{}: buffer of {len} bytes", case.name);
            assert_eq!(rust.final_range(), 0);
        }
        assert_eq!(
            rust_encode_all(&mut rust, &pcm, frame_size),
            rust_encode_all(&mut fresh, &pcm, frame_size),
            "{}: packets after the refused calls",
            case.name
        );
    }
}
