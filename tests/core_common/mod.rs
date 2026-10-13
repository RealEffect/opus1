//! Shared helpers for the `core_*` tests: thin wrappers over the raw C
//! libopus API (`audiopus_sys`) that expose what those tests compare
//! against. The test signals and the comparison functions come from
//! `k24_common`.

#![allow(dead_code)]

use audiopus_sys as sys;

/// Private libopus requests that `audiopus_sys` does not name
/// (opus_private.h, celt.h).
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
pub const OPUS_SET_LFE_REQUEST: i32 = 10024;
pub const OPUS_SET_ENERGY_MASK_REQUEST: i32 = 10026;
pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;

pub const MODE_SILK: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT: i32 = 1002;

pub const BW_NB: i32 = sys::OPUS_BANDWIDTH_NARROWBAND;
pub const BW_MB: i32 = sys::OPUS_BANDWIDTH_MEDIUMBAND;
pub const BW_WB: i32 = sys::OPUS_BANDWIDTH_WIDEBAND;
pub const BW_SWB: i32 = sys::OPUS_BANDWIDTH_SUPERWIDEBAND;
pub const BW_FB: i32 = sys::OPUS_BANDWIDTH_FULLBAND;

pub const APP_VOIP: i32 = sys::OPUS_APPLICATION_VOIP;
pub const APP_AUDIO: i32 = sys::OPUS_APPLICATION_AUDIO;
pub const APP_LOWDELAY: i32 = sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY;

/// C libopus encoder.
pub struct CEncoder {
    st: *mut sys::OpusEncoder,
    pub channels: usize,
    /// The energy mask libopus has a pointer to. The buffer is allocated
    /// once and never moves; a new mask overwrites its values.
    mask: Box<[f32; 42]>,
}

impl CEncoder {
    pub fn new(rate: usize, channels: usize, application: i32) -> Self {
        let mut err = 0;
        let st = unsafe {
            sys::opus_encoder_create(rate as i32, channels as i32, application, &mut err)
        };
        assert_eq!(err, 0, "opus_encoder_create({rate}, {channels})");
        Self { st, channels, mask: Box::new([0.0; 42]) }
    }

    /// `OPUS_SET_ENERGY_MASK`: 21 values per channel, or `None` for no
    /// mask. libopus keeps a pointer to the values and reads them in every
    /// later `encode`.
    pub fn energy_mask(&mut self, mask: Option<&[f32]>) -> &mut Self {
        let ptr = match mask {
            Some(mask) => {
                let n = 21 * self.channels;
                self.mask[..n].copy_from_slice(&mask[..n]);
                self.mask.as_ptr()
            }
            None => std::ptr::null(),
        };
        let ret = unsafe { sys::opus_encoder_ctl(self.st, OPUS_SET_ENERGY_MASK_REQUEST, ptr) };
        assert_eq!(ret, 0, "OPUS_SET_ENERGY_MASK = {ret}");
        self
    }

    /// A "set" control that takes one `opus_int32`.
    pub fn set(&mut self, request: i32, value: i32) -> &mut Self {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value}) = {ret}");
        self
    }

    /// A "get" control that returns one `opus_int32`.
    pub fn get(&mut self, request: i32) -> i32 {
        let mut value = 0i32;
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, &mut value as *mut i32) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}) = {ret}");
        value
    }

    pub fn bitrate(&mut self, bps: i32) -> &mut Self {
        self.set(sys::OPUS_SET_BITRATE_REQUEST, bps)
    }
    pub fn force_mode(&mut self, mode: i32) -> &mut Self {
        self.set(OPUS_SET_FORCE_MODE_REQUEST, mode)
    }
    pub fn bandwidth(&mut self, bw: i32) -> &mut Self {
        self.set(sys::OPUS_SET_BANDWIDTH_REQUEST, bw)
    }
    pub fn max_bandwidth(&mut self, bw: i32) -> &mut Self {
        self.set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw)
    }
    pub fn complexity(&mut self, c: i32) -> &mut Self {
        self.set(sys::OPUS_SET_COMPLEXITY_REQUEST, c)
    }
    pub fn vbr(&mut self, on: bool) -> &mut Self {
        self.set(sys::OPUS_SET_VBR_REQUEST, on as i32)
    }
    pub fn fec(&mut self, on: bool, loss_perc: i32) -> &mut Self {
        self.set(sys::OPUS_SET_INBAND_FEC_REQUEST, on as i32);
        self.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, loss_perc)
    }
    pub fn force_channels(&mut self, ch: i32) -> &mut Self {
        self.set(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, ch)
    }
    pub fn lfe(&mut self, on: bool) -> &mut Self {
        self.set(OPUS_SET_LFE_REQUEST, on as i32)
    }

    /// `OPUS_GET_FINAL_RANGE` after the last `encode`.
    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_encoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32)
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE");
        value
    }

    pub fn lookahead(&mut self) -> i32 {
        self.get(sys::OPUS_GET_LOOKAHEAD_REQUEST)
    }

    pub fn reset(&mut self) {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
    }

    /// Encode one frame of interleaved `f32` PCM into at most `max_bytes`.
    pub fn encode_max(&mut self, pcm: &[f32], max_bytes: usize) -> Vec<u8> {
        let frame = pcm.len() / self.channels;
        let mut out = vec![0u8; max_bytes];
        let n = unsafe {
            sys::opus_encode_float(
                self.st,
                pcm.as_ptr(),
                frame as i32,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        assert!(n > 0, "opus_encode_float failed: {n}");
        out.truncate(n as usize);
        out
    }

    pub fn encode(&mut self, pcm: &[f32]) -> Vec<u8> {
        self.encode_max(pcm, 4000)
    }

    /// Encode a whole signal in frames of `frame` samples per channel. A
    /// trailing partial frame is dropped.
    pub fn encode_all(&mut self, pcm: &[f32], frame: usize) -> Vec<Vec<u8>> {
        pcm.chunks_exact(frame * self.channels).map(|f| self.encode(f)).collect()
    }

    /// Like `encode_all`, and also the final range after every packet.
    pub fn encode_all_with_ranges(&mut self, pcm: &[f32], frame: usize) -> (Vec<Vec<u8>>, Vec<u32>) {
        let mut packets = Vec::new();
        let mut ranges = Vec::new();
        for f in pcm.chunks_exact(frame * self.channels) {
            packets.push(self.encode(f));
            ranges.push(self.final_range());
        }
        (packets, ranges)
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// C libopus decoder.
pub struct CDecoder {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
}

impl CDecoder {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        Self { st, channels }
    }

    /// `opus_decode_float`. `None` is a lost packet. Returns the interleaved
    /// samples, or the libopus error code.
    pub fn decode(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<f32>, i32> {
        let mut pcm = vec![0f32; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_decode_float(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as i32, fec as i32)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    /// `opus_decode`: 16-bit output.
    pub fn decode_i16(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<i16>, i32> {
        let mut pcm = vec![0i16; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_decode(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as i32, fec as i32)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    /// `OPUS_GET_FINAL_RANGE` after the last decode call.
    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_decoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32)
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE");
        value
    }

    pub fn reset(&mut self) {
        let ret = unsafe { sys::opus_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// Signal-to-noise ratio of `test` against `reference` in dB, over the
/// samples both have. 200 means identical.
pub fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (a, b) in reference.iter().zip(test) {
        sig += (*a as f64).powi(2);
        err += ((*a - *b) as f64).powi(2);
    }
    if err <= 0.0 {
        200.0
    } else if sig <= 0.0 {
        -200.0
    } else {
        (10.0 * (sig / err).log10()).min(200.0)
    }
}
