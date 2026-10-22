#![cfg(feature = "test-libopus-dnn")]
//! The decoder controls of the neural network features against libopus
//! 1.6.1 built with them (`scripts/android/build-libopus.sh --dnn`): deep
//! PLC, OSCE and the bandwidth extension, with built-in weights.
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,osce-bwe --test dnn_decoder_controls_libopus
//! ```
//!
//! `core_decoder_dnn_controls.rs` compares the same requests with the
//! Homebrew libopus, which has none of the features.

use audiopus_sys as sys;
use opus1::{Channels, OpusDecoder, SampleRate};
use std::os::raw::c_int;

const OPUS_GET_COMPLEXITY_REQUEST: c_int = 4011;
const OPUS_SET_DNN_BLOB_REQUEST: c_int = 4052;
const OPUS_SET_OSCE_BWE_REQUEST: c_int = 4054;
#[cfg(feature = "osce-bwe")]
const OPUS_GET_OSCE_BWE_REQUEST: c_int = 4055;

struct CDec(*mut sys::OpusDecoder);

impl CDec {
    fn new(rate: i32, channels: i32) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate, channels, &mut err) };
        assert_eq!(err, sys::OPUS_OK);
        CDec(st)
    }

    fn set(&mut self, request: c_int, value: i32) -> c_int {
        unsafe { sys::opus_decoder_ctl(self.0, request, value) }
    }

    fn get(&mut self, request: c_int) -> (c_int, i32) {
        let mut value = -12345;
        let ret = unsafe { sys::opus_decoder_ctl(self.0, request, &mut value as *mut i32) };
        (ret, value)
    }

    fn reset(&mut self) {
        assert_eq!(unsafe { sys::opus_decoder_ctl(self.0, sys::OPUS_RESET_STATE) }, sys::OPUS_OK);
    }
}

impl Drop for CDec {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.0) }
    }
}

#[test]
fn complexity_matches_a_libopus_with_the_features() {
    let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    let mut c = CDec::new(48000, 1);
    assert_eq!((d.complexity(), c.get(OPUS_GET_COMPLEXITY_REQUEST)), (0, (sys::OPUS_OK, 0)));
    for value in [-1, 0, 4, 5, 6, 7, 10, 11, i32::MIN, i32::MAX] {
        let want = c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, value);
        let got = match d.set_complexity(value) {
            Ok(()) => sys::OPUS_OK,
            Err(e) => e.to_raw(),
        };
        assert_eq!(got, want, "complexity {value}");
        assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, d.complexity()));
    }
    d.reset_state();
    c.reset();
    assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, d.complexity()));
}

/// libopus takes 0 and 1 and refuses other values with `OPUS_BAD_ARG`;
/// opus1 takes a `bool`, so it has no such values. Both start off and keep
/// the switch over a reset.
#[cfg(feature = "osce-bwe")]
#[test]
fn the_bwe_switch_matches_a_libopus_with_the_extension() {
    let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    let mut c = CDec::new(48000, 1);
    assert_eq!(c.get(OPUS_GET_OSCE_BWE_REQUEST), (sys::OPUS_OK, 0));
    assert_eq!(d.osce_bwe(), Ok(false));
    for on in [true, false, true] {
        assert_eq!(c.set(OPUS_SET_OSCE_BWE_REQUEST, on as i32), sys::OPUS_OK);
        d.set_osce_bwe(on).unwrap();
        assert_eq!(c.get(OPUS_GET_OSCE_BWE_REQUEST), (sys::OPUS_OK, on as i32));
        assert_eq!(d.osce_bwe(), Ok(on));
    }
    for bad in [-1, 2, 100] {
        assert_eq!(c.set(OPUS_SET_OSCE_BWE_REQUEST, bad), sys::OPUS_BAD_ARG);
        assert_eq!(c.get(OPUS_GET_OSCE_BWE_REQUEST), (sys::OPUS_OK, 1));
    }
    d.reset_state();
    c.reset();
    assert_eq!(c.get(OPUS_GET_OSCE_BWE_REQUEST), (sys::OPUS_OK, 1));
    assert_eq!(d.osce_bwe(), Ok(true));
}

/// A libopus with built-in weights has no `OPUS_SET_DNN_BLOB` (it exists
/// only with `USE_WEIGHTS_FILE`). opus1 has `set_dnn_blob` in every build
/// with a neural network feature.
#[test]
fn libopus_with_builtin_weights_has_no_blob_request() {
    let mut c = CDec::new(48000, 1);
    let blob = [0u8; 64];
    let ret = unsafe { sys::opus_decoder_ctl(c.0, OPUS_SET_DNN_BLOB_REQUEST, blob.as_ptr(), blob.len() as i32) };
    assert_eq!(ret, sys::OPUS_UNIMPLEMENTED);
    let _ = &mut c;
    #[cfg(feature = "deep-plc")]
    {
        let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
        assert!(d.set_dnn_blob(&opus1::DnnBlob::builtin()).is_ok());
    }
}

/// libopus's multistream decoder forwards `OPUS_SET_COMPLEXITY` but not
/// `OPUS_SET_OSCE_BWE`; opus1 forwards both.
#[test]
fn the_multistream_decoder_of_libopus_does_not_forward_the_bwe_switch() {
    let mapping = [0u8, 1];
    let mut err = 0;
    let st = unsafe { sys::opus_multistream_decoder_create(48000, 2, 2, 0, mapping.as_ptr(), &mut err) };
    assert_eq!(err, sys::OPUS_OK);
    assert_eq!(unsafe { sys::opus_multistream_decoder_ctl(st, sys::OPUS_SET_COMPLEXITY_REQUEST, 7) }, sys::OPUS_OK);
    assert_eq!(unsafe { sys::opus_multistream_decoder_ctl(st, OPUS_SET_OSCE_BWE_REQUEST, 1) }, sys::OPUS_UNIMPLEMENTED);
    unsafe { sys::opus_multistream_decoder_destroy(st) };

    let table = opus1::multistream::ChannelMappingTable {
        channels: 2,
        stream_count: 2,
        coupled_count: 0,
        mapping: mapping.to_vec(),
    };
    let mut d = opus1::multistream::MultistreamDecoder::new(SampleRate::Hz48000, table).unwrap();
    d.set_complexity(7).unwrap();
    assert_eq!(d.stream_decoder(1).unwrap().complexity(), 7);
    #[cfg(feature = "osce-bwe")]
    {
        d.set_osce_bwe(true).unwrap();
        assert_eq!(d.stream_decoder(1).unwrap().osce_bwe(), Ok(true));
    }
}
