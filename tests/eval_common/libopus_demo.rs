//! libopus's decoder behind the `DemoDecoder` trait of
//! `opus1::internals::eval::demo`, so that the `opus_demo -d` loop of the
//! port drives libopus exactly as it drives opus1.
//!
//! It calls `opus_decode24`, as `opus_demo` does, and
//! `opus_packet_has_lbrr`. Both are public functions of libopus 1.6.1
//! (`include/opus.h` 541 and 778) that `audiopus_sys` does not declare, so
//! they are declared here. The Homebrew library the tests link exports
//! them.

use audiopus_sys as sys;
use opus1::internals::eval::demo::DemoDecoder;
use std::os::raw::c_int;

unsafe extern "C" {
    fn opus_decode24(
        st: *mut sys::OpusDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut i32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    fn opus_packet_has_lbrr(packet: *const u8, len: i32) -> c_int;
}

pub const OPUS_GET_FINAL_RANGE_REQUEST: c_int = 4031;
pub const OPUS_GET_LAST_PACKET_DURATION_REQUEST: c_int = 4039;
pub const OPUS_SET_IGNORE_EXTENSIONS_REQUEST: c_int = 4058;

/// libopus `opus_packet_has_lbrr`: 1, 0, or a negative error code.
pub fn libopus_has_lbrr(packet: &[u8]) -> i32 {
    unsafe { opus_packet_has_lbrr(packet.as_ptr(), packet.len() as i32) }
}

/// A libopus decoder, as `opus_demo -d <rate> <channels> -ignore_extensions`
/// creates it.
pub struct LibopusDecoder {
    st: *mut sys::OpusDecoder,
    channels: usize,
}

impl LibopusDecoder {
    pub fn new(rate: u32, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as c_int, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        let ret = unsafe { sys::opus_decoder_ctl(st, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, 1 as c_int) };
        assert_eq!(ret, 0, "OPUS_SET_IGNORE_EXTENSIONS");
        Self { st, channels }
    }

    fn get(&self, request: c_int) -> i32 {
        let mut value: i32 = 0;
        let ret = unsafe { sys::opus_decoder_ctl(self.st, request, &mut value as *mut i32) };
        assert_eq!(ret, 0, "opus_decoder_ctl({request})");
        value
    }

    /// `opus_decode_float`. An empty `data` is a lost packet.
    pub fn decode_float(&mut self, data: &[u8], frame_size: usize, fec: bool) -> Result<Vec<f32>, i32> {
        let mut pcm = vec![0f32; frame_size * self.channels];
        let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr() };
        let n = unsafe {
            sys::opus_decode_float(self.st, ptr, data.len() as i32, pcm.as_mut_ptr(), frame_size as c_int, fec as c_int)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    /// `opus_decode`: 16-bit output. An empty `data` is a lost packet.
    pub fn decode16(&mut self, data: &[u8], frame_size: usize, fec: bool) -> Result<Vec<i16>, i32> {
        let mut pcm = vec![0i16; frame_size * self.channels];
        let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr() };
        let n = unsafe {
            sys::opus_decode(self.st, ptr, data.len() as i32, pcm.as_mut_ptr(), frame_size as c_int, fec as c_int)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    /// `opus_decode24`.
    pub fn decode24_vec(&mut self, data: &[u8], frame_size: usize, fec: bool) -> Result<Vec<i32>, i32> {
        let mut pcm = vec![0i32; frame_size * self.channels];
        let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr() };
        let n = unsafe {
            opus_decode24(self.st, ptr, data.len() as i32, pcm.as_mut_ptr(), frame_size as c_int, fec as c_int)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    pub fn reset(&mut self) {
        let ret = unsafe { sys::opus_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
    }
}

impl Drop for LibopusDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// The error messages of the libopus codes (`opus_strerror`).
fn error_message(code: c_int) -> &'static str {
    match code {
        -1 => "invalid argument",
        -2 => "buffer too small",
        -3 => "internal error",
        -4 => "corrupted stream",
        -5 => "request not implemented",
        -6 => "invalid state",
        -7 => "memory allocation failed",
        _ => "unknown error",
    }
}

impl DemoDecoder for LibopusDecoder {
    fn channels(&self) -> usize {
        self.channels
    }

    fn decode24(&mut self, data: &[u8], frame_size: usize, decode_fec: bool, out: &mut [i32]) -> Result<usize, &'static str> {
        assert!(out.len() >= frame_size * self.channels, "output buffer too small");
        let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr() };
        let n = unsafe {
            opus_decode24(self.st, ptr, data.len() as i32, out.as_mut_ptr(), frame_size as c_int, decode_fec as c_int)
        };
        if n < 0 { Err(error_message(n)) } else { Ok(n as usize) }
    }

    fn last_packet_duration(&self) -> usize {
        self.get(OPUS_GET_LAST_PACKET_DURATION_REQUEST) as usize
    }

    fn final_range(&self) -> u32 {
        self.get(OPUS_GET_FINAL_RANGE_REQUEST) as u32
    }
}
