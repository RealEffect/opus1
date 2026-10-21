//! Test support for the multistream tests. Use it with `mod ms_support;`.
//!
//! What is here:
//!
//! 1. Wrappers around the C libopus multistream encoder and decoder
//!    (`audiopus_sys`). They free the C state when dropped.
//! 2. The control request numbers that `audiopus_sys` does not name.
//! 3. Test signals with a different content on every channel.
//! 4. Measures: the power of a tone, and the signal-to-noise ratio (SNR) of
//!    one signal against another.
//! 5. A reader that splits a multistream packet into its streams.
//!
//! Nothing here uses opus1. Every helper is checked against C libopus in
//! `tests/multistream_c_selftest.rs`.
//!
//! All signals are interleaved `f32`: sample `i` of channel `c` is at index
//! `i * channels + c`. A frame size is always in samples per channel.

#![allow(dead_code)]

use audiopus_sys as sys;
use std::f64::consts::PI;
use std::os::raw::c_int;

// The 24-bit multistream calls of libopus 1.6, which `audiopus_sys` does
// not declare.
unsafe extern "C" {
    fn opus_multistream_encode24(
        st: *mut sys::OpusMSEncoder,
        pcm: *const i32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> c_int;
    fn opus_multistream_decode24(
        st: *mut sys::OpusMSDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut i32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Bitrate value: libopus picks the bitrate itself.
pub const OPUS_AUTO: i32 = sys::OPUS_AUTO;
/// Bitrate value: use as many bits as the packet can hold.
pub const OPUS_BITRATE_MAX: i32 = sys::OPUS_BITRATE_MAX;

/// libopus error code: an argument is invalid.
pub const OPUS_BAD_ARG: i32 = sys::OPUS_BAD_ARG;
/// libopus error code: the output buffer is too small.
pub const OPUS_BUFFER_TOO_SMALL: i32 = sys::OPUS_BUFFER_TOO_SMALL;
/// libopus error code: the packet is corrupted.
pub const OPUS_INVALID_PACKET: i32 = sys::OPUS_INVALID_PACKET;
/// libopus error code: the request is not implemented.
pub const OPUS_UNIMPLEMENTED: i32 = sys::OPUS_UNIMPLEMENTED;

/// Application value for speech.
pub const APP_VOIP: i32 = sys::OPUS_APPLICATION_VOIP;
/// Application value for music and mixed content.
pub const APP_AUDIO: i32 = sys::OPUS_APPLICATION_AUDIO;
/// Application value for the lowest delay (CELT only).
pub const APP_LOWDELAY: i32 = sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY;

/// Private request `OPUS_SET_FORCE_MODE` (opus_private.h). Its value is one of
/// the three `MODE_*` constants below, or `OPUS_AUTO`.
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
/// Value for `OPUS_SET_FORCE_MODE`: SILK only.
pub const MODE_SILK_ONLY: i32 = 1000;
/// Value for `OPUS_SET_FORCE_MODE`: Hybrid (SILK and CELT).
pub const MODE_HYBRID: i32 = 1001;
/// Value for `OPUS_SET_FORCE_MODE`: CELT only.
pub const MODE_CELT_ONLY: i32 = 1002;
/// Private request `OPUS_SET_LFE` (celt.h): 1 marks a stream as the
/// low-frequency-effects (LFE) stream.
pub const OPUS_SET_LFE_REQUEST: i32 = 10024;
/// Private request `OPUS_SET_ENERGY_MASK` (celt.h). Its argument is a pointer
/// to 21 `f32` values per channel of the stream.
pub const OPUS_SET_ENERGY_MASK_REQUEST: i32 = 10026;
/// Request `OPUS_GET_FINAL_RANGE`. Its argument is a pointer to `u32`.
pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;
/// `OPUS_GET_VOICE_RATIO` (opus_private.h).
pub const OPUS_GET_VOICE_RATIO_REQUEST: i32 = 11019;
/// Request `OPUS_MULTISTREAM_GET_ENCODER_STATE`. Its arguments are the stream
/// index and a pointer to `*mut OpusEncoder`.
pub const OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST: i32 = 5120;
/// Request `OPUS_MULTISTREAM_GET_DECODER_STATE`. Its arguments are the stream
/// index and a pointer to `*mut OpusDecoder`.
pub const OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST: i32 = 5122;

/// Largest frame of one channel in samples: 120 ms at 48 kHz.
pub const MAX_FRAME: usize = 5760;

// ---------------------------------------------------------------------------
// C multistream encoder
// ---------------------------------------------------------------------------

/// C libopus multistream encoder. The fields describe its layout.
pub struct CMsEncoder {
    st: *mut sys::OpusMSEncoder,
    /// Sampling rate in Hz.
    pub fs: i32,
    /// Number of input channels.
    pub channels: usize,
    /// Number of streams in one packet.
    pub streams: usize,
    /// Number of stereo streams. They come first in a packet.
    pub coupled: usize,
    /// Mapping table, one entry per input channel.
    pub mapping: Vec<u8>,
}

impl CMsEncoder {
    /// `opus_multistream_surround_encoder_create`: libopus picks the layout for a mapping family. `Err` holds the libopus error code.
    pub fn surround(fs: i32, channels: usize, family: i32, application: i32) -> Result<Self, i32> {
        let (mut streams, mut coupled, mut err): (c_int, c_int, c_int) = (0, 0, 0);
        // libopus writes one entry per channel and accepts at most 255.
        let mut mapping = [0u8; 256];
        let st = unsafe {
            sys::opus_multistream_surround_encoder_create(
                fs,
                channels as c_int,
                family,
                &mut streams,
                &mut coupled,
                mapping.as_mut_ptr(),
                application,
                &mut err,
            )
        };
        if st.is_null() {
            return Err(if err != 0 { err } else { sys::OPUS_ALLOC_FAIL });
        }
        Ok(Self {
            st,
            fs,
            channels,
            streams: streams as usize,
            coupled: coupled as usize,
            mapping: mapping[..channels].to_vec(),
        })
    }

    /// `opus_multistream_encoder_create`: the caller gives the layout. `Err` holds the libopus error code.
    pub fn generic(
        fs: i32,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &[u8],
        application: i32,
    ) -> Result<Self, i32> {
        assert!(
            mapping.len() >= channels || channels > 255,
            "the mapping needs one entry per channel"
        );
        // libopus reads one entry per channel after it has checked that
        // there are at most 255 channels.
        let mut padded = [0u8; 256];
        let n = mapping.len().min(256);
        padded[..n].copy_from_slice(&mapping[..n]);
        let mut err: c_int = 0;
        let st = unsafe {
            sys::opus_multistream_encoder_create(
                fs,
                channels as c_int,
                streams as c_int,
                coupled as c_int,
                padded.as_ptr(),
                application,
                &mut err,
            )
        };
        if st.is_null() {
            return Err(if err != 0 { err } else { sys::OPUS_ALLOC_FAIL });
        }
        Ok(Self {
            st,
            fs,
            channels,
            streams,
            coupled,
            mapping: mapping[..channels].to_vec(),
        })
    }

    /// Any control of the multistream encoder that sets one `i32`. Returns the libopus error code (0 is success).
    pub fn try_ctl(&mut self, request: i32, value: i32) -> i32 {
        unsafe { sys::opus_multistream_encoder_ctl(self.st, request, value) }
    }

    /// Like `try_ctl`, but panics when libopus reports an error.
    pub fn ctl(&mut self, request: i32, value: i32) -> &mut Self {
        let ret = self.try_ctl(request, value);
        assert_eq!(
            ret, 0,
            "opus_multistream_encoder_ctl({request}, {value}) = {ret}"
        );
        self
    }

    /// Any control of the multistream encoder that reads one `i32`.
    pub fn get(&mut self, request: i32) -> i32 {
        let mut v: i32 = 0;
        let ret =
            unsafe { sys::opus_multistream_encoder_ctl(self.st, request, &mut v as *mut i32) };
        assert_eq!(ret, 0, "opus_multistream_encoder_ctl({request}) = {ret}");
        v
    }

    /// Sets the total bitrate in bits per second, or `OPUS_AUTO` or `OPUS_BITRATE_MAX`.
    pub fn set_bitrate(&mut self, bps: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_BITRATE_REQUEST, bps)
    }

    /// Selects variable bitrate (`true`, the default) or constant bitrate (`false`) on every stream.
    pub fn set_vbr(&mut self, vbr: bool) -> &mut Self {
        self.ctl(sys::OPUS_SET_VBR_REQUEST, vbr as i32)
    }

    /// Sets the complexity, 0 to 10, on every stream.
    pub fn set_complexity(&mut self, complexity: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_COMPLEXITY_REQUEST, complexity)
    }

    /// Turns in-band forward error correction on or off on every stream.
    pub fn set_inband_fec(&mut self, on: bool) -> &mut Self {
        self.ctl(sys::OPUS_SET_INBAND_FEC_REQUEST, on as i32)
    }

    /// Sets the expected packet loss in percent on every stream.
    pub fn set_packet_loss_perc(&mut self, percent: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, percent)
    }

    /// `OPUS_GET_BITRATE`: the sum of the bitrates the stream encoders report.
    pub fn bitrate(&mut self) -> i32 {
        self.get(sys::OPUS_GET_BITRATE_REQUEST)
    }

    /// `OPUS_GET_LOOKAHEAD`: the delay of the encoder in samples.
    pub fn lookahead(&mut self) -> i32 {
        self.get(sys::OPUS_GET_LOOKAHEAD_REQUEST)
    }

    /// `OPUS_GET_FINAL_RANGE` after an encode: the final range values of all streams combined with XOR.
    pub fn final_range(&mut self) -> u32 {
        let mut v: u32 = 0;
        let ret = unsafe {
            sys::opus_multistream_encoder_ctl(
                self.st,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut v as *mut u32,
            )
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE = {ret}");
        v
    }

    /// The encoder of stream `s`, through `OPUS_MULTISTREAM_GET_ENCODER_STATE`. It belongs to this multistream encoder.
    fn stream_state(&mut self, s: usize) -> *mut sys::OpusEncoder {
        let mut sub: *mut sys::OpusEncoder = std::ptr::null_mut();
        let ret = unsafe {
            sys::opus_multistream_encoder_ctl(
                self.st,
                OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
                s as c_int,
                &mut sub as *mut *mut sys::OpusEncoder,
            )
        };
        assert_eq!(ret, 0, "OPUS_MULTISTREAM_GET_ENCODER_STATE({s}) = {ret}");
        assert!(!sub.is_null());
        sub
    }

    /// Any control of the encoder of stream `s` that sets one `i32`. Returns the libopus error code.
    pub fn stream_ctl(&mut self, s: usize, request: i32, value: i32) -> i32 {
        let sub = self.stream_state(s);
        unsafe { sys::opus_encoder_ctl(sub, request, value) }
    }

    /// Any control of the encoder of stream `s` that reads one `i32`.
    pub fn stream_get(&mut self, s: usize, request: i32) -> i32 {
        let sub = self.stream_state(s);
        let mut v: i32 = 0;
        let ret = unsafe { sys::opus_encoder_ctl(sub, request, &mut v as *mut i32) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}) on stream {s} = {ret}");
        v
    }

    /// `OPUS_GET_BITRATE` of the encoder of stream `s`. After an encode this is the bitrate the rate allocation gave the stream, limited to what one packet can hold.
    pub fn stream_bitrate(&mut self, s: usize) -> i32 {
        self.stream_get(s, sys::OPUS_GET_BITRATE_REQUEST)
    }

    /// `stream_bitrate` of every stream, in stream order.
    pub fn stream_bitrates(&mut self) -> Vec<i32> {
        (0..self.streams).map(|s| self.stream_bitrate(s)).collect()
    }

    /// `OPUS_GET_FINAL_RANGE` of the encoder of stream `s`.
    pub fn stream_final_range(&mut self, s: usize) -> u32 {
        let sub = self.stream_state(s);
        let mut v: u32 = 0;
        let ret =
            unsafe { sys::opus_encoder_ctl(sub, OPUS_GET_FINAL_RANGE_REQUEST, &mut v as *mut u32) };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE on stream {s} = {ret}");
        v
    }

    /// `opus_multistream_encode_float`: encodes one frame into a packet of at most `max_bytes` bytes. `Err` holds the libopus error code.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        max_bytes: usize,
    ) -> Result<Vec<u8>, i32> {
        assert!(
            pcm.len() >= frame_size * self.channels,
            "pcm is shorter than one frame"
        );
        let mut out = vec![0u8; max_bytes];
        let n = unsafe {
            sys::opus_multistream_encode_float(
                self.st,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                max_bytes as i32,
            )
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// `opus_multistream_encode24`: like `encode_float`, for 24-bit input
    /// in 32-bit integers.
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, max_bytes: usize) -> Result<Vec<u8>, i32> {
        assert!(pcm.len() >= frame_size * self.channels, "pcm is shorter than one frame");
        let mut out = vec![0u8; max_bytes];
        let n = unsafe {
            opus_multistream_encode24(self.st, pcm.as_ptr(), frame_size as c_int, out.as_mut_ptr(), max_bytes as i32)
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// `opus_multistream_encode`: like `encode_float`, for 16-bit input.
    pub fn encode_i16(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        max_bytes: usize,
    ) -> Result<Vec<u8>, i32> {
        assert!(
            pcm.len() >= frame_size * self.channels,
            "pcm is shorter than one frame"
        );
        let mut out = vec![0u8; max_bytes];
        let n = unsafe {
            sys::opus_multistream_encode(
                self.st,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                max_bytes as i32,
            )
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Encodes a whole signal frame by frame, with room for the largest packet. A trailing partial frame is dropped.
    pub fn encode_all(&mut self, pcm: &[f32], frame_size: usize) -> Vec<Vec<u8>> {
        let max_bytes = max_packet_bytes(self.streams);
        pcm.chunks_exact(frame_size * self.channels)
            .map(|frame| {
                self.encode_float(frame, frame_size, max_bytes)
                    .unwrap_or_else(|e| panic!("opus_multistream_encode_float failed: {e}"))
            })
            .collect()
    }
}

impl Drop for CMsEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_encoder_destroy(self.st) }
    }
}

/// Output buffer size that holds any packet of `streams` streams: 1275 bytes per frame and up to 6 frames per stream, plus headers.
pub fn max_packet_bytes(streams: usize) -> usize {
    streams * (6 * 1275 + 12)
}

// ---------------------------------------------------------------------------
// C multistream decoder
// ---------------------------------------------------------------------------

/// C libopus multistream decoder.
pub struct CMsDecoder {
    st: *mut sys::OpusMSDecoder,
    /// Sampling rate in Hz.
    pub fs: i32,
    /// Number of output channels.
    pub channels: usize,
    /// Number of streams in one packet.
    pub streams: usize,
    /// Number of stereo streams.
    pub coupled: usize,
}

impl CMsDecoder {
    /// `opus_multistream_decoder_create`. `Err` holds the libopus error code.
    pub fn new(
        fs: i32,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &[u8],
    ) -> Result<Self, i32> {
        assert!(
            mapping.len() >= channels || channels > 255,
            "the mapping needs one entry per channel"
        );
        let mut padded = [0u8; 256];
        let n = mapping.len().min(256);
        padded[..n].copy_from_slice(&mapping[..n]);
        let mut err: c_int = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(
                fs,
                channels as c_int,
                streams as c_int,
                coupled as c_int,
                padded.as_ptr(),
                &mut err,
            )
        };
        if st.is_null() {
            return Err(if err != 0 { err } else { sys::OPUS_ALLOC_FAIL });
        }
        Ok(Self {
            st,
            fs,
            channels,
            streams,
            coupled,
        })
    }

    /// A decoder with the layout and sampling rate of `enc`.
    pub fn for_encoder(enc: &CMsEncoder) -> Self {
        Self::new(enc.fs, enc.channels, enc.streams, enc.coupled, &enc.mapping)
            .unwrap_or_else(|e| panic!("opus_multistream_decoder_create failed: {e}"))
    }

    /// `opus_multistream_decode_float`. `packet` is `None` for a lost packet. `frame_size` is the room per channel, or the lost duration. With `fec`, the forward error correction data of `packet` rebuilds the frame before it. Returns the decoded samples, interleaved.
    pub fn decode_float(
        &mut self,
        packet: Option<&[u8]>,
        frame_size: usize,
        fec: bool,
    ) -> Result<Vec<f32>, i32> {
        let mut out = vec![0f32; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_multistream_decode_float(
                self.st,
                ptr,
                len,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize * self.channels);
        Ok(out)
    }

    /// `opus_multistream_decode24`: like `decode_float`, with 24-bit
    /// output in 32-bit integers.
    pub fn decode24(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<i32>, i32> {
        let mut out = vec![0i32; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            opus_multistream_decode24(self.st, ptr, len, out.as_mut_ptr(), frame_size as c_int, fec as c_int)
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize * self.channels);
        Ok(out)
    }

    /// A control of the multistream decoder that takes one `i32`, with
    /// libopus's return code.
    pub fn try_ctl(&mut self, request: i32, value: i32) -> i32 {
        unsafe { sys::opus_multistream_decoder_ctl(self.st, request, value) }
    }

    /// A control of the multistream decoder that reads one `i32`.
    pub fn get(&mut self, request: i32) -> i32 {
        let mut v: i32 = 0;
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.st, request, &mut v as *mut i32) };
        assert_eq!(ret, 0, "opus_multistream_decoder_ctl({request}) = {ret}");
        v
    }

    /// `opus_multistream_decode`: like `decode_float`, with 16-bit output.
    pub fn decode_i16(
        &mut self,
        packet: Option<&[u8]>,
        frame_size: usize,
        fec: bool,
    ) -> Result<Vec<i16>, i32> {
        let mut out = vec![0i16; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_multistream_decode(
                self.st,
                ptr,
                len,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize * self.channels);
        Ok(out)
    }

    /// Decodes a sequence of packets and joins the output. Panics when libopus rejects a packet.
    pub fn decode_all(&mut self, packets: &[Vec<u8>]) -> Vec<f32> {
        let mut out = Vec::new();
        for (i, p) in packets.iter().enumerate() {
            let pcm = self
                .decode_float(Some(p), MAX_FRAME, false)
                .unwrap_or_else(|e| {
                    panic!("opus_multistream_decode_float failed on packet {i}: {e}")
                });
            out.extend_from_slice(&pcm);
        }
        out
    }

    /// `OPUS_GET_FINAL_RANGE` after a decode: the final range values of all streams combined with XOR.
    pub fn final_range(&mut self) -> u32 {
        let mut v: u32 = 0;
        let ret = unsafe {
            sys::opus_multistream_decoder_ctl(
                self.st,
                OPUS_GET_FINAL_RANGE_REQUEST,
                &mut v as *mut u32,
            )
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE = {ret}");
        v
    }

    /// Any control of the decoder of stream `s` that reads one `i32`, through `OPUS_MULTISTREAM_GET_DECODER_STATE`.
    pub fn stream_get(&mut self, s: usize, request: i32) -> i32 {
        let mut sub: *mut sys::OpusDecoder = std::ptr::null_mut();
        let ret = unsafe {
            sys::opus_multistream_decoder_ctl(
                self.st,
                OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
                s as c_int,
                &mut sub as *mut *mut sys::OpusDecoder,
            )
        };
        assert_eq!(ret, 0, "OPUS_MULTISTREAM_GET_DECODER_STATE({s}) = {ret}");
        let mut v: i32 = 0;
        let ret = unsafe { sys::opus_decoder_ctl(sub, request, &mut v as *mut i32) };
        assert_eq!(ret, 0, "opus_decoder_ctl({request}) on stream {s} = {ret}");
        v
    }
}

impl Drop for CMsDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_decoder_destroy(self.st) }
    }
}

// ---------------------------------------------------------------------------
// C single-stream decoder
// ---------------------------------------------------------------------------

/// C libopus decoder for one ordinary stream, for example one stream that `split_streams` took out of a multistream packet.
pub struct CDecoder {
    st: *mut sys::OpusDecoder,
    /// Number of output channels, 1 or 2.
    pub channels: usize,
}

impl CDecoder {
    /// `opus_decoder_create`. Panics when libopus rejects the arguments.
    pub fn new(fs: i32, channels: usize) -> Self {
        let mut err: c_int = 0;
        let st = unsafe { sys::opus_decoder_create(fs, channels as c_int, &mut err) };
        assert!(
            !st.is_null() && err == 0,
            "opus_decoder_create({fs}, {channels}) = {err}"
        );
        Self { st, channels }
    }

    /// `opus_decode_float`. `packet` is `None` for a lost packet. Returns the decoded samples, interleaved.
    pub fn decode_float(
        &mut self,
        packet: Option<&[u8]>,
        frame_size: usize,
    ) -> Result<Vec<f32>, i32> {
        let mut out = vec![0f32; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_decode_float(self.st, ptr, len, out.as_mut_ptr(), frame_size as c_int, 0)
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize * self.channels);
        Ok(out)
    }

    /// `OPUS_GET_FINAL_RANGE` after a decode.
    pub fn final_range(&mut self) -> u32 {
        let mut v: u32 = 0;
        let ret = unsafe {
            sys::opus_decoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut v as *mut u32)
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE = {ret}");
        v
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// Amplitude of every tone of `tones`.
pub const TONE_AMPLITUDE: f32 = 0.25;

/// The LFE channel of a layout: the last channel of mapping family 1 with 6, 7 or 8 channels. `None` for every other layout.
pub fn lfe_channel(channels: usize, family: i32) -> Option<usize> {
    (family == 1 && (6..=8).contains(&channels)).then(|| channels - 1)
}

/// Frequency in Hz of the test tone on channel `c`: `400 + 150 * c`, and 80 on the LFE channel, which only keeps low frequencies.
pub fn tone_hz(c: usize, channels: usize, family: i32) -> f64 {
    if lfe_channel(channels, family) == Some(c) {
        80.0
    } else {
        400.0 + 150.0 * c as f64
    }
}

/// `samples` samples per channel with one sine per channel, at `tone_hz` and with amplitude `TONE_AMPLITUDE`. It shows which input channel reached which output channel.
pub fn tones(fs: i32, channels: usize, family: i32, samples: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let w = 2.0 * PI * tone_hz(c, channels, family) / fs as f64;
        for i in 0..samples {
            pcm[i * channels + c] = (TONE_AMPLITUDE as f64 * (w * i as f64).sin()) as f32;
        }
    }
    pcm
}

/// Pitch in Hz of channel `c` of `voiced` at the time `t` in seconds: `110 + 23 * c`, drifting by up to 15 Hz either way.
pub fn voiced_pitch_hz(c: usize, t: f64) -> f64 {
    110.0 + 23.0 * c as f64 + 15.0 * (2.0 * PI * 0.7 * t + c as f64).sin()
}

/// `samples` samples per channel of a speech-like signal: 8 harmonics of a slowly drifting pitch under a syllable-rate envelope, as in `tests/issue_42_stereo_channels.rs`. Every channel has its own pitch (`voiced_pitch_hz`) and its own envelope timing.
pub fn voiced(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let mut phase = 0.0f64;
        for i in 0..samples {
            let t = i as f64 / fs as f64;
            phase += 2.0 * PI * voiced_pitch_hz(c, t) / fs as f64;
            let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t + 0.9 * c as f64).sin();
            let mut v = 0.0;
            for h in 1..=8 {
                v += 0.25 / h as f64 * (h as f64 * phase).sin();
            }
            pcm[i * channels + c] = (v * env * 0.5) as f32;
        }
    }
    pcm
}

/// `samples` samples per channel of silence.
pub fn silence(channels: usize, samples: usize) -> Vec<f32> {
    vec![0f32; samples * channels]
}

/// `samples` samples per channel of white noise between `-amplitude` and `amplitude`. The same `seed` gives the same signal; every channel is different.
pub fn noise(channels: usize, samples: usize, amplitude: f32, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345) | 1;
    (0..samples * channels)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * amplitude
        })
        .collect()
}

/// Channel `c` of an interleaved signal.
pub fn channel(pcm: &[f32], channels: usize, c: usize) -> Vec<f32> {
    pcm.iter().skip(c).step_by(channels).copied().collect()
}

/// Converts to 16-bit samples like libopus `FLOAT2INT16`: scale by 32768, clip, round to the nearest integer (halves to the even one).
pub fn to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter()
        .map(|&s| (s * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16)
        .collect()
}

// ---------------------------------------------------------------------------
// Measures
// ---------------------------------------------------------------------------

/// Power of the sine at `hz` in the single-channel signal `x` (Goertzel recurrence under a Hann window). A sine of amplitude `a` gives `a * a / 2`.
pub fn goertzel_power(x: &[f32], hz: f64, fs: i32) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let coeff = 2.0 * (2.0 * PI * hz / fs as f64).cos();
    let (mut q1, mut q2, mut window_sum) = (0f64, 0f64, 0f64);
    for (i, &s) in x.iter().enumerate() {
        let window = 0.5 - 0.5 * (2.0 * PI * (i as f64 + 0.5) / n as f64).cos();
        let q0 = coeff * q1 - q2 + window * s as f64;
        q2 = q1;
        q1 = q0;
        window_sum += window;
    }
    let magnitude_sq = (q1 * q1 + q2 * q2 - q1 * q2 * coeff).max(0.0);
    2.0 * magnitude_sq / (window_sum * window_sum)
}

/// For every channel of a decoded `tones` signal: the channel whose test tone is strongest there, and the power of that tone divided by the power of the next strongest test tone. Routing is right when entry `c` is `(c, large ratio)`. The first `skip` samples per channel are left out.
pub fn dominant_tones(
    decoded: &[f32],
    fs: i32,
    channels: usize,
    family: i32,
    skip: usize,
) -> Vec<(usize, f64)> {
    (0..channels)
        .map(|c| {
            let x = channel(decoded, channels, c);
            let x = &x[skip.min(x.len())..];
            let mut powers: Vec<(usize, f64)> = (0..channels)
                .map(|k| (k, goertzel_power(x, tone_hz(k, channels, family), fs)))
                .collect();
            powers.sort_by(|a, b| b.1.total_cmp(&a.1));
            let second = powers.get(1).map_or(0.0, |p| p.1);
            (powers[0].0, powers[0].1 / second.max(1e-30))
        })
        .collect()
}

/// `10 * log10(signal / noise)`, limited to -200 to 200 dB so that silence and exact matches stay finite.
pub fn snr_db(signal: f64, noise: f64) -> f64 {
    if noise <= 0.0 {
        200.0
    } else if signal <= 0.0 {
        -200.0
    } else {
        (10.0 * (signal / noise).log10()).clamp(-200.0, 200.0)
    }
}

/// Sum of squares of `reference[from..to]`, and of its difference to `test` read `lag` samples later. Samples that `test` does not have are skipped.
fn error_sums(reference: &[f32], test: &[f32], from: usize, to: usize, lag: isize) -> (f64, f64) {
    let (mut signal, mut noise) = (0f64, 0f64);
    for (i, &r) in reference.iter().enumerate().take(to).skip(from) {
        let j = i as isize + lag;
        if j < 0 || j as usize >= test.len() {
            continue;
        }
        let (r, t) = (r as f64, test[j as usize] as f64);
        signal += r * r;
        noise += (r - t) * (r - t);
    }
    (signal, noise)
}

/// SNR in dB of the single-channel signal `test` against `reference`, when `test` is `lag` samples late. The first `skip` samples of `reference` are left out.
pub fn snr_at_lag(reference: &[f32], test: &[f32], lag: isize, skip: usize) -> f64 {
    let (signal, noise) = error_sums(reference, test, skip, reference.len(), lag);
    snr_db(signal, noise)
}

/// Best-lag SNR: the SNR in dB of `test` against `reference` at the delay where the two match best, and that delay in samples (positive when `test` is late). Delays from `-max_lag` to `max_lag` are tried on a stretch of at most 9600 samples after `skip`; the SNR is then measured over everything after `skip`.
pub fn best_lag_snr(reference: &[f32], test: &[f32], max_lag: usize, skip: usize) -> (f64, isize) {
    let from = skip.max(max_lag).min(reference.len());
    let to = (from + 9600)
        .min(reference.len().saturating_sub(max_lag))
        .max(from);
    let max_lag = max_lag as isize;
    let mut best = (f64::NEG_INFINITY, 0isize);
    for lag in -max_lag..=max_lag {
        let (signal, noise) = error_sums(reference, test, from, to, lag);
        let snr = snr_db(signal, noise);
        // On a tie the smaller delay wins, so equal signals report lag 0.
        if snr > best.0 || (snr == best.0 && lag.abs() < best.1.abs()) {
            best = (snr, lag);
        }
    }
    (snr_at_lag(reference, test, best.1, skip), best.1)
}

// ---------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------

/// Coding mode named by a table-of-contents (TOC) byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Silk,
    Hybrid,
    Celt,
}

/// Audio bandwidth named by a TOC byte: narrowband (4 kHz), mediumband (6 kHz), wideband (8 kHz), super-wideband (12 kHz), fullband (20 kHz).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bandwidth {
    Nb,
    Mb,
    Wb,
    Swb,
    Fb,
}

/// What the TOC byte of an Opus packet says (RFC 6716 section 3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Toc {
    pub mode: Mode,
    pub bandwidth: Bandwidth,
    /// Duration of one frame in samples at 48 kHz: 120 is 2.5 ms, 960 is 20 ms, 2880 is 60 ms.
    pub frame_48k: usize,
    /// The packet is coded as stereo.
    pub stereo: bool,
    /// Frame count code, 0 to 3: one frame, two equal frames, two different frames, any number of frames.
    pub code: u8,
}

/// Decodes a TOC byte.
pub fn toc(byte: u8) -> Toc {
    let (mode, bandwidth, frame_48k) = if byte & 0x80 != 0 {
        let bw = [Bandwidth::Nb, Bandwidth::Wb, Bandwidth::Swb, Bandwidth::Fb]
            [((byte >> 5) & 3) as usize];
        (Mode::Celt, bw, 120usize << ((byte >> 3) & 3))
    } else if byte & 0x60 == 0x60 {
        let bw = if byte & 0x10 != 0 {
            Bandwidth::Fb
        } else {
            Bandwidth::Swb
        };
        (Mode::Hybrid, bw, if byte & 0x08 != 0 { 960 } else { 480 })
    } else {
        let bw = [Bandwidth::Nb, Bandwidth::Mb, Bandwidth::Wb][((byte >> 5) & 3) as usize];
        (
            Mode::Silk,
            bw,
            [480, 960, 1920, 2880][((byte >> 3) & 3) as usize],
        )
    };
    Toc {
        mode,
        bandwidth,
        frame_48k,
        stereo: byte & 0x04 != 0,
        code: byte & 0x03,
    }
}

impl std::fmt::Display for Toc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = match self.mode {
            Mode::Silk => "SILK",
            Mode::Hybrid => "HYBR",
            Mode::Celt => "CELT",
        };
        let bw = match self.bandwidth {
            Bandwidth::Nb => "NB",
            Bandwidth::Mb => "MB",
            Bandwidth::Wb => "WB",
            Bandwidth::Swb => "SWB",
            Bandwidth::Fb => "FB",
        };
        write!(
            f,
            "{mode}/{bw}/{}ms/{}/c{}",
            self.frame_48k as f64 / 48.0,
            if self.stereo { "stereo" } else { "mono" },
            self.code
        )
    }
}

/// One stream of a multistream packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamPacket {
    /// TOC byte of the stream. Decode it with `toc`.
    pub toc: u8,
    /// Size of every frame in bytes.
    pub frame_sizes: Vec<usize>,
    /// Payload length: the sum of the frame sizes, without the TOC byte, length fields and padding.
    pub payload_len: usize,
    /// Number of padding bytes.
    pub padding: usize,
    /// Number of bytes the stream takes in the multistream packet, with its header and padding.
    pub used: usize,
    /// The stream as an ordinary Opus packet that a single-stream decoder accepts.
    pub packet: Vec<u8>,
}

impl StreamPacket {
    /// Number of frames in the stream.
    pub fn frames(&self) -> usize {
        self.frame_sizes.len()
    }

    /// Duration of the stream in samples per channel at the sampling rate `fs`.
    pub fn samples(&self, fs: i32) -> usize {
        self.frames() * toc(self.toc).frame_48k * fs as usize / 48_000
    }
}

impl std::fmt::Display for StreamPacket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{} frames={} payload={}B used={}B]",
            toc(self.toc),
            self.frames(),
            self.payload_len,
            self.used
        )
    }
}

/// libopus `parse_size`: a frame length of one or two bytes at `data[pos]`, with `len` bytes left. Returns the length and the number of bytes it took.
fn parse_size(data: &[u8], pos: usize, len: i64) -> Option<(i64, i64)> {
    if len < 1 {
        None
    } else if data[pos] < 252 {
        Some((data[pos] as i64, 1))
    } else if len < 2 {
        None
    } else {
        Some((4 * data[pos + 1] as i64 + data[pos] as i64, 2))
    }
}

/// Port of libopus `opus_packet_parse_impl` (opus.c): parses the packet at the start of `data`. With `self_delimited`, the packet carries the length of its last frame and may be followed by more data.
fn parse_packet(data: &[u8], self_delimited: bool) -> Result<StreamPacket, &'static str> {
    const INVALID: &str = "invalid packet";
    if data.is_empty() {
        return Err("empty packet");
    }
    let toc_byte = data[0];
    let framesize = toc(toc_byte).frame_48k as i64;
    let mut pos = 1usize;
    let mut len = data.len() as i64 - 1;
    let mut last_size = len;
    let mut cbr = false;
    let mut pad = 0i64;
    let count: usize;
    let mut size = [0i64; 48];
    match toc_byte & 0x3 {
        // One frame.
        0 => count = 1,
        // Two frames of the same size.
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if len & 0x1 != 0 {
                    return Err(INVALID);
                }
                last_size = len / 2;
                size[0] = last_size;
            }
        }
        // Two frames of different sizes.
        2 => {
            count = 2;
            let (s, bytes) = parse_size(data, pos, len).ok_or(INVALID)?;
            len -= bytes;
            if s > len {
                return Err(INVALID);
            }
            pos += bytes as usize;
            size[0] = s;
            last_size = len - s;
        }
        // Any number of frames.
        _ => {
            if len < 1 {
                return Err(INVALID);
            }
            let ch = data[pos];
            pos += 1;
            count = (ch & 0x3F) as usize;
            if count == 0 || framesize * count as i64 > 5760 {
                return Err(INVALID);
            }
            len -= 1;
            // Bit 6 says that padding follows.
            if ch & 0x40 != 0 {
                loop {
                    if len <= 0 {
                        return Err(INVALID);
                    }
                    let p = data[pos] as i64;
                    pos += 1;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p };
                    len -= tmp;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(INVALID);
            }
            // Bit 7 says that the frames have different sizes.
            cbr = ch & 0x80 == 0;
            if !cbr {
                last_size = len;
                for s in size.iter_mut().take(count - 1) {
                    let (v, bytes) = parse_size(data, pos, len).ok_or(INVALID)?;
                    len -= bytes;
                    if v > len {
                        return Err(INVALID);
                    }
                    pos += bytes as usize;
                    *s = v;
                    last_size -= bytes + v;
                }
                if last_size < 0 {
                    return Err(INVALID);
                }
            } else if !self_delimited {
                last_size = len / count as i64;
                if last_size * count as i64 != len {
                    return Err(INVALID);
                }
                for s in size.iter_mut().take(count - 1) {
                    *s = last_size;
                }
            }
        }
    }
    // Self-delimited framing has one more length, for the last frame.
    let length_field = pos;
    let mut length_field_bytes = 0usize;
    if self_delimited {
        let (s, bytes) = parse_size(data, pos, len).ok_or(INVALID)?;
        len -= bytes;
        if s > len {
            return Err(INVALID);
        }
        pos += bytes as usize;
        length_field_bytes = bytes as usize;
        if cbr {
            if s * count as i64 > len {
                return Err(INVALID);
            }
            for v in size.iter_mut().take(count) {
                *v = s;
            }
        } else {
            if bytes + s > last_size {
                return Err(INVALID);
            }
            size[count - 1] = s;
        }
    } else {
        if last_size > 1275 {
            return Err(INVALID);
        }
        size[count - 1] = last_size;
    }
    let frame_sizes: Vec<usize> = size[..count].iter().map(|&s| s as usize).collect();
    let payload_len: usize = frame_sizes.iter().sum();
    let used = pos + payload_len + pad as usize;
    if used > data.len() {
        return Err(INVALID);
    }
    // Without the extra length field the bytes are an ordinary packet.
    let mut packet = data[..length_field].to_vec();
    packet.extend_from_slice(&data[length_field + length_field_bytes..used]);
    Ok(StreamPacket {
        toc: toc_byte,
        frame_sizes,
        payload_len,
        padding: pad as usize,
        used,
        packet,
    })
}

/// Parses one ordinary Opus packet: its TOC byte, frame sizes and padding.
pub fn parse_stream(packet: &[u8]) -> Result<StreamPacket, String> {
    parse_packet(packet, false).map_err(|e| e.to_string())
}

/// Walks the `streams` streams of a multistream packet and returns each one's TOC byte, payload length and bytes. Every stream but the last is self-delimited (RFC 6716 appendix B). Fails when a stream is malformed or when the streams do not all have the same duration, like libopus `opus_multistream_packet_validate`.
pub fn split_streams(packet: &[u8], streams: usize) -> Result<Vec<StreamPacket>, String> {
    let mut out = Vec::with_capacity(streams);
    let mut rest = packet;
    for s in 0..streams {
        if rest.is_empty() {
            return Err(format!("stream {s}: the packet ends before this stream"));
        }
        let stream =
            parse_packet(rest, s + 1 != streams).map_err(|e| format!("stream {s}: {e}"))?;
        rest = &rest[stream.used..];
        out.push(stream);
    }
    let duration = |p: &StreamPacket| p.frames() * toc(p.toc).frame_48k;
    if out.iter().any(|p| duration(p) != duration(&out[0])) {
        return Err("the streams do not all have the same duration".to_string());
    }
    Ok(out)
}

/// Text for a multistream packet, one bracket per stream, for failure messages.
pub fn describe_packet(packet: &[u8], streams: usize) -> String {
    match split_streams(packet, streams) {
        Ok(parts) => parts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" "),
        Err(e) => format!("(unreadable: {e})"),
    }
}
