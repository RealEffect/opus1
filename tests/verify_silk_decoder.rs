//! The SILK decoder and the SILK resampler against libopus.
//!
//! The SILK packets come from the libopus encoder that the tests link, and
//! the decoded samples must be those of `opus_decode` of the same libopus:
//! SILK decodes with integers only, so the samples are equal. The resampler
//! must give the samples of libopus's `silk_resampler` (silk/resampler.c)
//! for the same input.

use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::dec_api::SilkDecoder;
use opus1::internals::silk::decode_frame::FLAG_DECODE_NORMAL;
use opus1::internals::silk::resampler::SilkResampler;
use opus1::packet::Framing;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_resampler_init(state: *mut u8, fs_in: i32, fs_out: i32, for_enc: c_int) -> c_int;
    fn silk_resampler(state: *mut u8, out: *mut i16, input: *const i16, in_len: i32) -> c_int;
}

/// A sine of `freq` Hz at half of full scale, `frames` frames of 20 ms.
fn sine(rate: i32, freq: f32, frames: usize) -> Vec<f32> {
    (0..rate as usize / 50 * frames)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.5)
        .collect()
}

/// Encodes `pcm` with libopus (VoIP, 20 ms frames, constant bitrate,
/// complexity 0) and decodes each packet with libopus and with opus1's
/// SILK decoder. libopus's `silk_Decode` passes the samples through its
/// resampler even when the output rate is the SILK rate, which delays
/// them; opus1's samples go through its resampler too. Returns the number
/// of samples compared; panics at the first difference.
fn decode_libopus_packets(rate: i32, bitrate: i32, pcm: &[f32], toc_config: u8) -> usize {
    let frame = rate as usize / 50;
    let mut enc = opus::Encoder::new(rate as u32, opus::Channels::Mono, opus::Application::Voip).unwrap();
    enc.set_bitrate(opus::Bitrate::Bits(bitrate)).unwrap();
    enc.set_vbr(false).unwrap();
    enc.set_complexity(0).unwrap();
    let mut c_dec = opus::Decoder::new(rate as u32, opus::Channels::Mono).unwrap();
    let mut dec = SilkDecoder::new();
    assert_eq!(dec.init(rate, 1), 0, "init at {rate} Hz");
    assert_eq!(dec.frame_length() as usize, frame);
    let mut resampler = SilkResampler::default();
    assert_eq!(resampler.init(rate, rate), 0);

    let mut compared = 0;
    for (i, input) in pcm.chunks_exact(frame).enumerate() {
        let mut packet = vec![0u8; 1275];
        let n = enc.encode_float(input, &mut packet).unwrap();
        packet.truncate(n);
        assert_eq!(packet[0] >> 3, toc_config, "packet {i}: not one SILK frame of 20 ms");
        let mut c_out = vec![0i16; frame];
        assert_eq!(c_dec.decode(&packet, &mut c_out, false).unwrap(), frame);

        // The constant bitrate pads the packet: take its one frame.
        let parsed = opus1::packet::parse(&packet, Framing::Normal).unwrap();
        let frames: Vec<&[u8]> = parsed.frames(&packet).collect();
        assert_eq!(frames.len(), 1);
        let mut rc = RangeCoder::new_decoder(frames[0]);
        let mut silk = vec![0i16; frame];
        let got = dec.decode(&mut rc, &mut silk, FLAG_DECODE_NORMAL, true, 20, rate);
        assert_eq!(got as usize, frame, "packet {i}");
        let mut out = vec![0i16; frame];
        assert_eq!(resampler.process(&mut out, &silk, frame as i32), 0);
        assert_eq!(out, c_out, "packet {i}: samples");
        compared += frame;
    }
    compared
}

/// A 440 Hz sine at 8 kHz, 10 kb/s: narrowband SILK.
#[test]
fn test_decode_c_encoded_silk_8khz() {
    let pcm = sine(8000, 440.0, 25);
    assert_eq!(decode_libopus_packets(8000, 10000, &pcm, 1), pcm.len());
}

/// A 440 Hz sine at 16 kHz, 20 kb/s: wideband SILK.
#[test]
fn test_decode_synth_16khz() {
    let pcm = sine(16000, 440.0, 25);
    assert_eq!(decode_libopus_packets(16000, 20000, &pcm, 9), pcm.len());
}

/// Test decoder reset functionality
#[test]
fn test_decoder_reset() {
    let mut decoder = SilkDecoder::new();

    // Initialize at 16kHz
    let ret = decoder.init(16000, 1);
    assert_eq!(ret, 0, "Init 16kHz failed");
    assert_eq!(decoder.sample_rate(), 16000);

    // Reset and reinit at 8kHz
    decoder.reset();
    let ret = decoder.init(8000, 1);
    assert_eq!(ret, 0, "Init 8kHz failed");
    assert_eq!(decoder.sample_rate(), 8000);

    // Verify state is reset
    assert_eq!(decoder.channel_state[0].first_frame_after_reset, 1);
}

/// Test that decoder handles invalid sample rates correctly
#[test]
fn test_decoder_invalid_sample_rate() {
    let mut decoder = SilkDecoder::new();

    // SILK only supports 8, 12, 16 kHz
    let ret = decoder.init(48000, 1);
    assert!(ret < 0, "Should reject 48kHz");

    let ret = decoder.init(44100, 1);
    assert!(ret < 0, "Should reject 44.1kHz");

    let ret = decoder.init(24000, 1);
    assert!(ret < 0, "Should reject 24kHz");
}

/// Resamples `frames` calls of 10 ms of a sine (`freq` Hz) from `fs_in` to
/// `fs_out` with opus1 and with libopus, one stream, and compares the
/// output of every call. Returns opus1's output.
fn compare_resampler(fs_in: i32, fs_out: i32, freq: f32, frames: usize) -> Vec<i16> {
    // Makes sure the test binary links the static library that
    // `audiopus_sys` brings along.
    assert!(!unsafe { audiopus_sys::opus_get_version_string() }.is_null());
    let (len_in, len_out) = (fs_in as usize / 100, fs_out as usize / 100);
    let mut resampler = SilkResampler::default();
    assert_eq!(resampler.init(fs_in, fs_out), 0, "{fs_in} to {fs_out} Hz");
    // libopus's state is opaque and far smaller than this buffer.
    let mut c_state = vec![0u8; 4096];
    // SAFETY: the buffer is larger than `silk_resampler_state_struct`.
    assert_eq!(unsafe { silk_resampler_init(c_state.as_mut_ptr(), fs_in, fs_out, 0) }, 0);

    let mut all = Vec::new();
    for frame in 0..frames {
        let input: Vec<i16> = (0..len_in)
            .map(|i| {
                let t = (frame * len_in + i) as f32 / fs_in as f32;
                ((2.0 * std::f32::consts::PI * freq * t).sin() * 16384.0) as i16
            })
            .collect();
        let mut output = vec![0i16; len_out];
        assert_eq!(resampler.process(&mut output, &input, len_in as i32), 0);
        let mut c_output = vec![0i16; len_out];
        // SAFETY: `input` holds `len_in` samples and `c_output` room for
        // the `len_out` samples they give.
        let ret =
            unsafe { silk_resampler(c_state.as_mut_ptr(), c_output.as_mut_ptr(), input.as_ptr(), len_in as i32) };
        assert_eq!(ret, 0);
        assert_eq!(output, c_output, "{fs_in} to {fs_out} Hz, call {frame}");
        all.extend(output);
    }
    all
}

/// 8 kHz to 48 kHz (IIR and FIR interpolation).
#[test]
fn test_resampler_8khz_to_48khz() {
    let out = compare_resampler(8000, 48000, 440.0, 1);
    assert!(out.iter().any(|&x| x != 0));
}

/// 12 kHz to 24 kHz (Up2HQ mode, direct 2x upsample).
#[test]
fn test_resampler_12khz_to_24khz() {
    let out = compare_resampler(12000, 24000, 400.0, 1);
    assert!(out.iter().any(|&x| x != 0));
}

/// 16 kHz to 48 kHz (IIR and FIR interpolation).
#[test]
fn test_resampler_16khz_to_48khz() {
    let out = compare_resampler(16000, 48000, 440.0, 1);
    assert!(out.iter().any(|&x| x != 0));
}

/// 8 kHz to 8 kHz: a copy with the delay of libopus.
#[test]
fn test_resampler_8khz_copy_mode() {
    compare_resampler(8000, 8000, 440.0, 2);
}

/// The filter state carries over from call to call: five calls of 10 ms
/// in one stream.
#[test]
fn test_resampler_delay_buffer_continuity() {
    compare_resampler(8000, 48000, 440.0, 5);
}

/// 8 kHz to 16 kHz.
#[test]
fn test_resampler_8khz_to_16khz() {
    let out = compare_resampler(8000, 16000, 440.0, 1);
    assert!(out.iter().any(|&x| x != 0));
}
