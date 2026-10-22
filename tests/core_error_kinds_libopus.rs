//! The kind of the errors of opus1's encoder and decoder, against the error
//! code that libopus 1.6.1 returns for the same call.
//!
//! `Error::to_raw` must give the libopus code, so that code written for
//! libopus can handle the errors of opus1 the same way.

mod core_common;

use audiopus_sys as sys;
use core_common::*;
use opus1::{Application, Bitrate, Channels, Error, ErrorKind, OpusDecoder, OpusEncoder, SampleRate};

/// opus1's result as a libopus return code: 0 for success, else the error
/// code.
fn code<T>(r: Result<T, Error>) -> i32 {
    match r {
        Ok(_) => 0,
        Err(e) => e.to_raw(),
    }
}

#[test]
fn every_kind_has_its_libopus_code() {
    let kinds = [
        (ErrorKind::BadArg, sys::OPUS_BAD_ARG),
        (ErrorKind::BufferTooSmall, sys::OPUS_BUFFER_TOO_SMALL),
        (ErrorKind::InternalError, sys::OPUS_INTERNAL_ERROR),
        (ErrorKind::InvalidPacket, sys::OPUS_INVALID_PACKET),
        (ErrorKind::Unimplemented, sys::OPUS_UNIMPLEMENTED),
        (ErrorKind::InvalidState, sys::OPUS_INVALID_STATE),
        (ErrorKind::AllocFail, sys::OPUS_ALLOC_FAIL),
    ];
    for (kind, raw) in kinds {
        assert_eq!(kind.to_raw(), raw, "{kind:?}");
    }
}

#[test]
fn encoder_errors_have_the_codes_of_libopus() {
    let mut rust =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
    let mut c = CEncoder::new(48_000, 1, APP_AUDIO);
    let pcm = vec![0.0f32; 5760];
    let mut out = vec![0u8; 1500];
    // 441 samples is not a frame duration of Opus.
    let want = c.try_encode(&pcm, 441, 1500).unwrap_err();
    assert_eq!(code(rust.encode(&pcm[..441], 441, &mut out)), want, "frame size");
    assert_eq!(want, sys::OPUS_BAD_ARG);
    // An empty output buffer.
    let want = c.try_encode(&pcm, 960, 0).unwrap_err();
    assert_eq!(code(rust.encode(&pcm[..960], 960, &mut [])), want, "empty output");
    assert_eq!(want, sys::OPUS_BAD_ARG);
    // 100 ms cannot be coded in 1 byte.
    let want = c.try_encode(&pcm, 4800, 1).unwrap_err();
    assert_eq!(code(rust.encode(&pcm[..4800], 4800, &mut out[..1])), want, "100 ms in 1 byte");
    assert_eq!(want, sys::OPUS_BUFFER_TOO_SMALL);
    // Settings that libopus refuses.
    assert_eq!(code(rust.set_complexity(11)), c.try_set(sys::OPUS_SET_COMPLEXITY_REQUEST, 11));
    assert_eq!(
        code(rust.set_bitrate(Bitrate::BitsPerSecond(0))),
        c.try_set(sys::OPUS_SET_BITRATE_REQUEST, 0)
    );
    assert_eq!(
        code(rust.set_packet_loss_perc(101)),
        c.try_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 101)
    );
    assert_eq!(code(rust.set_lsb_depth(7)), c.try_set(sys::OPUS_SET_LSB_DEPTH_REQUEST, 7));
}

#[test]
fn decoder_errors_have_the_codes_of_libopus() {
    let mut enc =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
    let mut buf = vec![0u8; 1500];
    let len = enc.encode(&[0.0; 960], 960, &mut buf).unwrap();
    let packet = &buf[..len];
    let mut rust = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    let mut c = CDecoder::new(48_000, 1);
    let mut out = vec![0.0f32; 5760];
    // Frame count code 3 without the byte that holds the count.
    let want = c.decode(Some(&[0x03]), 960, false).unwrap_err();
    assert_eq!(code(rust.decode(&[0x03], 960, &mut out)), want, "invalid packet");
    assert_eq!(want, sys::OPUS_INVALID_PACKET);
    // A 20 ms packet into room for 10 ms.
    let want = c.decode(Some(packet), 480, false).unwrap_err();
    assert_eq!(code(rust.decode(packet, 480, &mut out)), want, "frame_size below the packet");
    assert_eq!(want, sys::OPUS_BUFFER_TOO_SMALL);
    // Lost and recovered frames must last a whole number of 2.5 ms.
    let want = c.decode(None, 100, false).unwrap_err();
    assert_eq!(code(rust.decode(&[], 100, &mut out)), want, "loss of 100 samples");
    assert_eq!(want, sys::OPUS_BAD_ARG);
    let want = c.decode(Some(packet), 100, true).unwrap_err();
    assert_eq!(code(rust.decode_fec(packet, 100, &mut out)), want, "FEC of 100 samples");
    // A `frame_size` of 0 is checked before anything else, also before the
    // packet is read: BadArg for a valid packet, an invalid one, a lost one
    // and FEC, with every output type.
    let mut out16 = vec![0i16; 5760];
    let mut out24 = vec![0i32; 5760];
    for (data, what) in [(Some(packet), "a valid packet"), (Some(&[0x03][..]), "an invalid packet"), (None, "a loss")] {
        let bytes = data.unwrap_or(&[]);
        for fec in [false, true] {
            let what = format!("frame_size 0 with {what}, FEC {fec}");
            let want = c.decode(data, 0, fec).unwrap_err();
            assert_eq!(want, sys::OPUS_BAD_ARG, "{what}");
            assert_eq!(c.decode_i16(data, 0, fec).unwrap_err(), want, "{what}: opus_decode");
            assert_eq!(c.decode24(data, 0, fec).unwrap_err(), want, "{what}: opus_decode24");
            let (f32_result, i16_result, i24_result) = if fec {
                (
                    code(rust.decode_fec(bytes, 0, &mut out)),
                    code(rust.decode_fec_i16(bytes, 0, &mut out16)),
                    code(rust.decode_fec24(bytes, 0, &mut out24)),
                )
            } else {
                (
                    code(rust.decode(bytes, 0, &mut out)),
                    code(rust.decode_i16(bytes, 0, &mut out16)),
                    code(rust.decode24(bytes, 0, &mut out24)),
                )
            };
            assert_eq!(f32_result, want, "{what}: decode");
            assert_eq!(i16_result, want, "{what}: decode_i16");
            assert_eq!(i24_result, want, "{what}: decode24");
        }
    }
    // Settings that libopus refuses.
    assert_eq!(code(rust.set_gain(40_000)), c.try_set(sys::OPUS_SET_GAIN_REQUEST, 40_000));
    assert_eq!(code(rust.set_complexity(-1)), c.try_set(sys::OPUS_SET_COMPLEXITY_REQUEST, -1));
}
