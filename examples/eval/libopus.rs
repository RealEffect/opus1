//! libopus through `audiopus_sys`: the encoder of the paired comparison,
//! and the decoder behind the `DemoDecoder` trait of
//! `opus1::internals::eval::demo`, so that the `opus_demo -d` loop drives
//! it.
//!
//! `opus_decode24` and `opus_packet_has_lbrr` are public functions of
//! libopus 1.6.1 (`include/opus.h` 541 and 778) that `audiopus_sys` does
//! not declare, so `opus_decode24` is declared here.

use audiopus_sys as ffi;
use opus1::internals::eval::demo::DemoDecoder;
use std::ffi::CStr;
use std::os::raw::c_int;

unsafe extern "C" {
    fn opus_decode24(
        st: *mut ffi::OpusDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut i32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
}

const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
pub const MODE_SILK_ONLY: c_int = 1000;
pub const MODE_HYBRID: c_int = 1001;
pub const MODE_CELT_ONLY: c_int = 1002;
const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
const OPUS_GET_LAST_PACKET_DURATION_REQUEST: c_int = 4039;
const OPUS_SET_IGNORE_EXTENSIONS_REQUEST: c_int = 4058;

/// `opus_get_version_string()`, such as "libopus 1.6.1".
pub fn version() -> String {
    // SAFETY: returns a pointer to a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::opus_get_version_string()) }.to_string_lossy().into_owned()
}

/// A libopus encoder.
pub struct CEncoder {
    state: *mut ffi::OpusEncoder,
    channels: usize,
}

// SAFETY: the encoder state is used by one thread at a time.
unsafe impl Send for CEncoder {}

impl CEncoder {
    /// `application` is one of the `OPUS_APPLICATION_*` values.
    pub fn new(rate: u32, channels: usize, application: c_int) -> Result<Self, String> {
        let mut error: c_int = 0;
        // SAFETY: plain C constructor; `error` is a valid out pointer.
        let state = unsafe { ffi::opus_encoder_create(rate as i32, channels as c_int, application, &mut error) };
        if state.is_null() || error != ffi::OPUS_OK {
            return Err(format!("opus_encoder_create failed with {error}"));
        }
        Ok(Self { state, channels })
    }

    pub fn ctl(&mut self, request: c_int, value: c_int) -> Result<(), String> {
        // SAFETY: `state` is a live encoder; every request used here takes
        // one `opus_int32` argument.
        let ret = unsafe { ffi::opus_encoder_ctl(self.state, request, value) };
        if ret == ffi::OPUS_OK { Ok(()) } else { Err(format!("opus_encoder_ctl({request}, {value}) failed with {ret}")) }
    }

    pub fn get(&self, request: c_int) -> Result<i32, String> {
        let mut value: i32 = 0;
        // SAFETY: `state` is a live encoder; the request writes one
        // `opus_int32` through the pointer.
        let ret = unsafe { ffi::opus_encoder_ctl(self.state, request, &mut value as *mut i32) };
        if ret == ffi::OPUS_OK { Ok(value) } else { Err(format!("opus_encoder_ctl({request}) failed with {ret}")) }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn set_force_mode(&mut self, mode: c_int) -> Result<(), String> {
        self.ctl(OPUS_SET_FORCE_MODE_REQUEST, mode)
    }

    pub fn final_range(&self) -> u32 {
        self.get(OPUS_GET_FINAL_RANGE_REQUEST).unwrap_or(0) as u32
    }

    /// `opus_encode_float`. Returns the packet length.
    pub fn encode(&mut self, pcm: &[f32], frame: usize, out: &mut [u8]) -> Result<usize, String> {
        assert!(pcm.len() >= frame * self.channels);
        // SAFETY: `pcm` holds `frame * channels` samples (checked above) and
        // `out` is writable for `out.len()` bytes.
        let ret = unsafe {
            ffi::opus_encode_float(self.state, pcm.as_ptr(), frame as c_int, out.as_mut_ptr(), out.len() as i32)
        };
        if ret < 0 { Err(format!("opus_encode_float failed with {ret}")) } else { Ok(ret as usize) }
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        // SAFETY: `state` came from `opus_encoder_create` and is freed once.
        unsafe { ffi::opus_encoder_destroy(self.state) }
    }
}

/// A libopus decoder, as `opus_demo -d <rate> <channels> -ignore_extensions`
/// creates it.
pub struct CDecoder {
    state: *mut ffi::OpusDecoder,
    channels: usize,
}

// SAFETY: the decoder state is used by one thread at a time.
unsafe impl Send for CDecoder {}

impl CDecoder {
    pub fn new(rate: u32, channels: usize) -> Result<Self, String> {
        let mut error: c_int = 0;
        // SAFETY: plain C constructor; `error` is a valid out pointer.
        let state = unsafe { ffi::opus_decoder_create(rate as i32, channels as c_int, &mut error) };
        if state.is_null() || error != ffi::OPUS_OK {
            return Err(format!("opus_decoder_create failed with {error}"));
        }
        // SAFETY: live decoder; the request takes one `opus_int32`.
        let ret = unsafe { ffi::opus_decoder_ctl(state, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, 1 as c_int) };
        if ret != ffi::OPUS_OK {
            // SAFETY: created above, freed once.
            unsafe { ffi::opus_decoder_destroy(state) };
            return Err(format!("OPUS_SET_IGNORE_EXTENSIONS failed with {ret}"));
        }
        Ok(Self { state, channels })
    }

    fn get(&self, request: c_int) -> i32 {
        let mut value: i32 = 0;
        // SAFETY: live decoder; the request writes one `opus_int32`.
        unsafe { ffi::opus_decoder_ctl(self.state, request, &mut value as *mut i32) };
        value
    }
}

impl DemoDecoder for CDecoder {
    fn channels(&self) -> usize {
        self.channels
    }

    fn decode24(&mut self, data: &[u8], frame_size: usize, decode_fec: bool, out: &mut [i32]) -> Result<usize, &'static str> {
        if out.len() < frame_size * self.channels {
            return Err("output buffer too small");
        }
        let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr() };
        // SAFETY: `out` has room for `frame_size * channels` samples
        // (checked above); `data` is readable for its length.
        let ret = unsafe {
            opus_decode24(self.state, ptr, data.len() as i32, out.as_mut_ptr(), frame_size as c_int, decode_fec as c_int)
        };
        if ret < 0 { Err("libopus decode error") } else { Ok(ret as usize) }
    }

    fn last_packet_duration(&self) -> usize {
        self.get(OPUS_GET_LAST_PACKET_DURATION_REQUEST).max(0) as usize
    }

    fn final_range(&self) -> u32 {
        self.get(OPUS_GET_FINAL_RANGE_REQUEST) as u32
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        // SAFETY: `state` came from `opus_decoder_create` and is freed once.
        unsafe { ffi::opus_decoder_destroy(self.state) }
    }
}
