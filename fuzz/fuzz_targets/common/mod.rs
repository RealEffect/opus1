//! Code shared by the fuzz targets that compare opus1 with libopus.
//!
//! libopus is the C reference implementation of Opus. It is linked through
//! `audiopus_sys`, the raw C interface. These wrappers create and free the C
//! objects and return the C result codes as they are, so that a target can
//! compare them with the `Result` of opus1.
//!
//! Two functions of libopus are not part of its public header but are
//! exported by the static library: `opus_packet_parse_impl` (it can parse a
//! self-delimited packet) and `opus_repacketizer_out_range_impl` (it can
//! write a self-delimited or padded packet). The tests of the main crate
//! call internal libopus functions in the same way.

#![allow(dead_code)]

/// Encoded packets for the targets with the neural network features.
#[cfg(feature = "dred")]
pub mod pool;

use audiopus_sys as sys;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use opus1::{Bitrate, FrameDuration};
use std::os::raw::{c_int, c_uchar};

/// The sampling rates Opus supports.
pub const RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Requests that `audiopus_sys` does not name (opus_private.h).
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
pub const OPUS_SET_LFE_REQUEST: i32 = 10024;
pub const OPUS_SET_ENERGY_MASK_REQUEST: i32 = 10026;
pub const MODE_SILK_ONLY: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT_ONLY: i32 = 1002;

unsafe extern "C" {
    /// libopus opus.c. `frames`, `size`, `payload_offset`, `packet_offset`,
    /// `padding` and `padding_len` may be null, except `size`.
    pub fn opus_packet_parse_impl(
        data: *const c_uchar,
        len: i32,
        self_delimited: c_int,
        out_toc: *mut c_uchar,
        frames: *mut *const c_uchar,
        size: *mut i16,
        payload_offset: *mut c_int,
        packet_offset: *mut i32,
        padding: *mut *const c_uchar,
        padding_len: *mut i32,
    ) -> c_int;

    /// libopus repacketizer.c. `extensions` may be null when
    /// `nb_extensions` is 0.
    pub fn opus_repacketizer_out_range_impl(
        rp: *mut sys::OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: i32,
        self_delimited: c_int,
        pad: c_int,
        extensions: *const u8,
        nb_extensions: c_int,
    ) -> i32;
}

/// A pointer for C to read `data` from. An empty slice gives a null
/// pointer, which libopus takes as a lost packet.
pub fn data_ptr(data: &[u8]) -> *const c_uchar {
    if data.is_empty() { std::ptr::null() } else { data.as_ptr() }
}

/// The libopus decoder (`opus_decoder_create`).
pub struct CDecoder {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
}

impl CDecoder {
    pub fn new(fs: i32, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(fs, channels as c_int, &mut err) };
        assert!(err == 0 && !st.is_null(), "opus_decoder_create({fs}, {channels}) = {err}");
        Self { st, channels }
    }

    /// `opus_decode_float`, with `out` as the output buffer. Returns the C
    /// result: samples per channel, or an error code.
    pub fn decode_float(&mut self, data: &[u8], out: &mut [f32], frame_size: usize, fec: bool) -> i32 {
        assert!(out.len() >= frame_size * self.channels);
        unsafe {
            sys::opus_decode_float(
                self.st,
                data_ptr(data),
                data.len() as i32,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        }
    }

    /// `opus_decode` (16-bit output).
    pub fn decode_i16(&mut self, data: &[u8], out: &mut [i16], frame_size: usize, fec: bool) -> i32 {
        assert!(out.len() >= frame_size * self.channels);
        unsafe {
            sys::opus_decode(
                self.st,
                data_ptr(data),
                data.len() as i32,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        }
    }

    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_decoder_ctl(self.st, sys::OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32)
        };
        assert_eq!(ret, 0);
        value
    }

    pub fn reset(&mut self) {
        let ret = unsafe { sys::opus_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0);
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// The libopus encoder (`opus_encoder_create`), driven by raw controls.
pub struct CEncoder {
    st: *mut sys::OpusEncoder,
    pub fs: i32,
    pub channels: usize,
}

impl CEncoder {
    pub fn new(fs: i32, channels: usize, application: i32) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_encoder_create(fs, channels as c_int, application, &mut err) };
        assert!(err == 0 && !st.is_null(), "opus_encoder_create({fs}, {channels}) = {err}");
        Self { st, fs, channels }
    }

    /// A control that takes one `opus_int32`. Returns the C result: a value
    /// out of range is an error, which the caller may ignore.
    pub fn set(&mut self, request: i32, value: i32) -> i32 {
        unsafe { sys::opus_encoder_ctl(self.st, request, value) }
    }

    /// `opus_encode_float` into `out`. Returns the packet length or an
    /// error code.
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> i32 {
        assert!(pcm.len() >= frame_size * self.channels);
        unsafe {
            sys::opus_encode_float(
                self.st,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        }
    }

    pub fn reset(&mut self) {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0);
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// The libopus multistream decoder (`opus_multistream_decoder_create`).
pub struct CMsDecoder {
    st: *mut sys::OpusMSDecoder,
    pub channels: usize,
}

impl CMsDecoder {
    /// Returns the C error code when libopus rejects the layout.
    pub fn new(fs: i32, channels: usize, streams: usize, coupled: usize, mapping: &[u8]) -> std::result::Result<Self, i32> {
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(
                fs,
                channels as c_int,
                streams as c_int,
                coupled as c_int,
                mapping.as_ptr(),
                &mut err,
            )
        };
        if err != 0 || st.is_null() {
            return Err(err);
        }
        Ok(Self { st, channels })
    }

    pub fn decode_float(&mut self, data: &[u8], out: &mut [f32], frame_size: usize, fec: bool) -> i32 {
        assert!(out.len() >= frame_size * self.channels);
        unsafe {
            sys::opus_multistream_decode_float(
                self.st,
                data_ptr(data),
                data.len() as i32,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        }
    }

    pub fn decode_i16(&mut self, data: &[u8], out: &mut [i16], frame_size: usize, fec: bool) -> i32 {
        assert!(out.len() >= frame_size * self.channels);
        unsafe {
            sys::opus_multistream_decode(
                self.st,
                data_ptr(data),
                data.len() as i32,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        }
    }

    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_multistream_decoder_ctl(
                self.st,
                sys::OPUS_GET_FINAL_RANGE_REQUEST,
                &mut value as *mut u32,
            )
        };
        assert_eq!(ret, 0);
        value
    }

    pub fn reset(&mut self) {
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0);
    }
}

impl Drop for CMsDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_decoder_destroy(self.st) }
    }
}

/// The libopus multistream encoder (`opus_multistream_encoder_create` and
/// `opus_multistream_surround_encoder_create`), driven by raw controls.
pub struct CMsEncoder {
    st: *mut sys::OpusMSEncoder,
    pub channels: usize,
}

impl CMsEncoder {
    /// An encoder for a layout of `streams` streams, the first `coupled` of
    /// them stereo, and a mapping. Returns the C error code when libopus
    /// refuses the layout.
    pub fn new(
        fs: i32,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &[u8],
        application: i32,
    ) -> std::result::Result<Self, i32> {
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_encoder_create(
                fs,
                channels as c_int,
                streams as c_int,
                coupled as c_int,
                mapping.as_ptr(),
                application,
                &mut err,
            )
        };
        if err != 0 || st.is_null() {
            return Err(err);
        }
        Ok(Self { st, channels })
    }

    /// An encoder with the layout of a mapping family. Returns the encoder
    /// and its streams, coupled streams and mapping, or the C error code.
    pub fn surround(
        fs: i32,
        channels: usize,
        family: u8,
        application: i32,
    ) -> std::result::Result<(Self, usize, usize, Vec<u8>), i32> {
        let (mut streams, mut coupled) = (0, 0);
        let mut mapping = [0u8; 256];
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_surround_encoder_create(
                fs,
                channels as c_int,
                family as c_int,
                &mut streams,
                &mut coupled,
                mapping.as_mut_ptr(),
                application,
                &mut err,
            )
        };
        if err != 0 || st.is_null() {
            return Err(err);
        }
        Ok((Self { st, channels }, streams as usize, coupled as usize, mapping[..channels].to_vec()))
    }

    /// A control that takes one `opus_int32`. Returns the C result.
    pub fn set(&mut self, request: i32, value: i32) -> i32 {
        unsafe { sys::opus_multistream_encoder_ctl(self.st, request, value) }
    }

    /// `opus_multistream_encode_float` into `out`. Returns the packet
    /// length or an error code.
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> i32 {
        assert!(pcm.len() >= frame_size * self.channels);
        unsafe {
            sys::opus_multistream_encode_float(
                self.st,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        }
    }
}

impl Drop for CMsEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_encoder_destroy(self.st) }
    }
}

/// The libopus repacketizer (`opus_repacketizer_create`).
pub struct CRepacketizer(pub *mut sys::OpusRepacketizer);

impl CRepacketizer {
    pub fn new() -> Self {
        let rp = unsafe { sys::opus_repacketizer_create() };
        assert!(!rp.is_null());
        Self(rp)
    }

    pub fn init(&mut self) {
        unsafe { sys::opus_repacketizer_init(self.0) };
    }

    /// `opus_repacketizer_cat`. libopus keeps pointers into `data`, so the
    /// caller keeps it alive until the packet is written.
    pub fn cat(&mut self, data: &[u8]) -> i32 {
        unsafe { sys::opus_repacketizer_cat(self.0, data_ptr(data), data.len() as i32) }
    }

    pub fn nb_frames(&mut self) -> i32 {
        unsafe { sys::opus_repacketizer_get_nb_frames(self.0) }
    }

    /// `opus_repacketizer_out_range_impl` over all frames, without
    /// extensions.
    pub fn out(&mut self, out: &mut [u8], self_delimited: bool, pad: bool) -> i32 {
        let n = self.nb_frames();
        unsafe {
            opus_repacketizer_out_range_impl(
                self.0,
                0,
                n,
                out.as_mut_ptr(),
                out.len() as i32,
                self_delimited as c_int,
                pad as c_int,
                std::ptr::null(),
                0,
            )
        }
    }
}

impl Drop for CRepacketizer {
    fn drop(&mut self) {
        unsafe { sys::opus_repacketizer_destroy(self.0) }
    }
}

/// `opus_packet_get_nb_samples`: the duration of a packet, or an error code.
pub fn c_nb_samples(data: &[u8], fs: i32) -> i32 {
    unsafe { sys::opus_packet_get_nb_samples(data_ptr(data), data.len() as i32, fs) }
}

/// What libopus `opus_packet_parse_impl` returns for a packet that parses.
#[derive(Debug, PartialEq, Eq)]
pub struct CParsed {
    pub toc: u8,
    pub sizes: Vec<u16>,
    pub payload_offset: usize,
    pub packet_offset: usize,
    pub padding_len: usize,
}

/// `opus_packet_parse_impl`: the parsed packet, or the C error code.
pub fn c_parse(data: &[u8], self_delimited: bool) -> std::result::Result<CParsed, i32> {
    let mut toc = 0u8;
    let mut frames = [std::ptr::null::<u8>(); 48];
    let mut size = [0i16; 48];
    let mut payload_offset = 0;
    let mut packet_offset = 0;
    let mut padding = std::ptr::null::<u8>();
    let mut padding_len = 0;
    let count = unsafe {
        opus_packet_parse_impl(
            data_ptr(data),
            data.len() as i32,
            self_delimited as c_int,
            &mut toc,
            frames.as_mut_ptr(),
            size.as_mut_ptr(),
            &mut payload_offset,
            &mut packet_offset,
            &mut padding,
            &mut padding_len,
        )
    };
    if count < 0 {
        return Err(count);
    }
    let count = count as usize;
    // The frame pointers must follow the sizes, starting at the payload.
    let mut at = payload_offset as usize;
    for i in 0..count {
        assert_eq!(frames[i] as usize - data.as_ptr() as usize, at, "C frame pointer {i}");
        at += size[i] as usize;
    }
    Ok(CParsed {
        toc,
        sizes: size[..count].iter().map(|&s| s as u16).collect(),
        payload_offset: payload_offset as usize,
        packet_offset: packet_offset as usize,
        padding_len: padding_len as usize,
    })
}

/// A small deterministic generator: the 64-bit linear congruential
/// generator of Knuth's MMIX.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// A value in [-1, 1).
    pub fn signed(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 30) as f32 - 1.0
    }
}

/// Test audio for an encoder. Each frame continues the one before, so the
/// encoder sees a signal and not unrelated blocks.
pub struct Signal {
    kind: u8,
    level: f32,
    freq: f32,
    rng: Lcg,
    /// Samples per channel produced so far.
    t: u64,
}

impl Signal {
    pub fn new(u: &mut Unstructured) -> Result<Self> {
        let mut s = Self { kind: 0, level: 0.5, freq: 440.0, rng: Lcg(0), t: 0 };
        s.change(u)?;
        Ok(s)
    }

    /// Picks another kind of signal, level and frequency.
    pub fn change(&mut self, u: &mut Unstructured) -> Result<()> {
        self.kind = u.int_in_range(0..=8)?;
        // The largest levels make frames that the encoder must treat as
        // insane input (a sum of squares of 1e9 or more): it codes silence.
        self.level = *u.choose(&[0.0, 1e-4, 0.01, 0.1, 0.5, 0.99, 1.0, 2.0, 30.0, 1e3, 1e6])?;
        self.freq = u.int_in_range(20..=23_000)? as f32;
        self.rng = Lcg(u.arbitrary::<u64>()?);
        Ok(())
    }

    /// One frame of `frame_size` samples per channel, interleaved.
    /// `extra` gives the bytes the "fuzzer data" kind plays.
    pub fn frame(&mut self, fs: i32, channels: usize, frame_size: usize, extra: &[u8], out: &mut Vec<f32>) {
        out.clear();
        let w = 2.0 * std::f32::consts::PI * self.freq / fs as f32;
        for i in 0..frame_size {
            let n = self.t + i as u64;
            for c in 0..channels {
                let phase = (n % (1 << 20)) as f32 * w;
                let v = match self.kind {
                    0 => 0.0,
                    1 => phase.sin(),
                    2 => self.rng.signed(),
                    3 => {
                        if n % 480 == 0 { 1.0 } else { 0.0 }
                    }
                    4 => {
                        // Speech-like: noise bursts with a pitch.
                        let env = ((n / 1600) % 2) as f32;
                        env * (phase.sin() + 0.3 * self.rng.signed())
                    }
                    5 => {
                        // The two channels differ: tone left, noise right.
                        if c == 0 { phase.sin() } else { self.rng.signed() }
                    }
                    6 => {
                        if extra.is_empty() {
                            0.0
                        } else {
                            let b = extra[(i * channels + c) % extra.len()];
                            (b as i8) as f32 / 128.0
                        }
                    }
                    7 => {
                        // Full-scale square wave.
                        if (phase / std::f32::consts::PI) as i64 % 2 == 0 { 1.0 } else { -1.0 }
                    }
                    _ => {
                        // Noise with samples that are not numbers or are
                        // infinite now and then.
                        match self.rng.next() % 64 {
                            0 => f32::NAN,
                            1 => f32::INFINITY,
                            2 => f32::NEG_INFINITY,
                            _ => self.rng.signed(),
                        }
                    }
                };
                out.push(v * self.level);
            }
        }
        self.t += frame_size as u64;
    }
}

/// libopus `bitrate_to_bits` (opus_private.h): bits per second to bits
/// per frame of `frame_size` samples at `fs`.
pub fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: usize) -> i32 {
    (bitrate as i64 * 6 / (6 * fs as i64 / frame_size as i64)) as i32
}

/// libopus `bits_to_bitrate` (opus_private.h): bits per frame to bits per
/// second.
pub fn bits_to_bitrate(bits: i32, fs: i32, frame_size: usize) -> i32 {
    (bits as i64 * (6 * fs as i64 / frame_size as i64) / 6) as i32
}

/// The length libopus gives a constant-bitrate packet of one stream, as the
/// lowest and the highest length allowed (opus_encoder.c,
/// `opus_encode_native`):
///
/// - The bitrate is that of the setting (`user_bitrate_to_bitrate`: the
///   automatic one is 60 * fs / frame_size + fs * channels, the maximum
///   1.5 Mb/s), at most what the buffer holds, and the buffer counts with
///   at most 6 * 1276 bytes.
/// - The packet has the whole bytes of that bitrate, rounded to the nearest
///   byte and at most the buffer (`cbr_bytes`), and at least 1 byte.
/// - A packet that is too small to code audio is padded to that length too,
///   but its frame count code can need 2 bytes (C 1340-1406).
/// - At the maximum bitrate a packet made of several frames is padded to
///   the whole buffer (C 1749): CELT or Hybrid of 40 ms or more, SILK of
///   80 ms or more.
///
/// `toc` is the first byte of the packet; it tells a SILK-only packet.
pub fn cbr_packet_len(
    setting: Bitrate,
    fs: i32,
    channels: usize,
    frame_size: usize,
    out_len: usize,
    toc: u8,
) -> (usize, usize) {
    let max_data_bytes = out_len.min(6 * 1276) as i32;
    let max_bitrate = bits_to_bitrate(max_data_bytes * 8, fs, frame_size);
    let user = match setting {
        Bitrate::Auto => 60 * fs / frame_size as i32 + fs * channels as i32,
        Bitrate::Max => 1_500_000,
        Bitrate::BitsPerSecond(bps) => bps,
    };
    let bitrate = user.min(max_bitrate);
    let cbr_bytes = ((bitrate_to_bits(bitrate, fs, frame_size) + 4) / 8).min(max_data_bytes);
    let bitrate = bits_to_bitrate(cbr_bytes * 8, fs, frame_size);
    let max_data_bytes = cbr_bytes.max(1);
    let frame_rate = fs / frame_size as i32;
    let no_audio = max_data_bytes < 3
        || bitrate < 3 * frame_rate * 8
        || (frame_rate < 50 && (max_data_bytes * frame_rate < 300 || bitrate < 2400));
    if no_audio {
        return (max_data_bytes as usize, max_data_bytes.max(2) as usize);
    }
    let silk_only = toc >> 3 < 12;
    let several_frames = (frame_size * 50 > fs as usize && !silk_only) || frame_size * 50 > 3 * fs as usize;
    if setting == Bitrate::Max && several_frames {
        return (out_len, out_len);
    }
    (max_data_bytes as usize, max_data_bytes as usize)
}

/// The frame sizes an encoder takes at `fs`: 2.5, 5, 10, 20, 40, 60, 80,
/// 100 and 120 ms.
pub fn encoder_frame_sizes(fs: i32) -> [usize; 9] {
    let f = fs as usize / 400;
    [f, 2 * f, 4 * f, 8 * f, 16 * f, 24 * f, 32 * f, 40 * f, 48 * f]
}

/// The frame an encode call of `frame_size` samples codes with the expert
/// frame duration `duration`, or `None` when the call is refused: libopus
/// `frame_size_select`, written out here from its rules.
pub fn selected_frame_size(fs: i32, frame_size: usize, duration: FrameDuration) -> Option<usize> {
    let fs = fs as usize;
    if frame_size < fs / 400 {
        return None;
    }
    let ms_times_10 = match duration {
        FrameDuration::Argument => None,
        FrameDuration::Ms2_5 => Some(25),
        FrameDuration::Ms5 => Some(50),
        FrameDuration::Ms10 => Some(100),
        FrameDuration::Ms20 => Some(200),
        FrameDuration::Ms40 => Some(400),
        FrameDuration::Ms60 => Some(600),
        FrameDuration::Ms80 => Some(800),
        FrameDuration::Ms100 => Some(1000),
        FrameDuration::Ms120 => Some(1200),
    };
    let size = ms_times_10.map_or(frame_size, |t| fs * t / 10_000);
    (size <= frame_size && encoder_frame_sizes(fs as i32).contains(&size)).then_some(size)
}

/// Changes some bytes of a packet: flips bits, cuts it short, or appends
/// bytes. The first byte (the table-of-contents byte) is kept unless
/// `keep_toc` is false.
pub fn mutate(u: &mut Unstructured, packet: &mut Vec<u8>, keep_toc: bool) -> Result<()> {
    let n = u.int_in_range(0..=3)?;
    for _ in 0..n {
        match u.int_in_range(0..=3)? {
            0 | 1 => {
                if packet.len() > keep_toc as usize {
                    let lo = keep_toc as usize;
                    let i = u.int_in_range(lo..=packet.len() - 1)?;
                    packet[i] ^= u.arbitrary::<u8>()? | 1;
                }
            }
            2 => {
                let keep = u.int_in_range(0..=packet.len())?;
                packet.truncate(keep.max(keep_toc as usize));
            }
            _ => {
                let extra = u.int_in_range(1..=8)?;
                for _ in 0..extra {
                    packet.push(u.arbitrary()?);
                }
            }
        }
    }
    Ok(())
}

/// The output of opus1 is within 50 dB of libopus's, or differs by less
/// than `floor` per sample on average (in squared units). This holds while
/// both decoders have decoded the same packets in full: opus1's decoder
/// differs from libopus's only in rounding (docs/10).
pub fn assert_close<T: Copy + Into<f64>>(what: &str, rust: &[T], c: &[T], floor: f64) {
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (&r, &c) in rust.iter().zip(c) {
        let (r, c): (f64, f64) = (r.into(), c.into());
        sig += c * c;
        err += (r - c) * (r - c);
    }
    let n = rust.len().max(1) as f64;
    if err > sig * 1e-5 + floor * n {
        let snr = 10.0 * (sig / err.max(1e-30)).log10();
        panic!("{what}: output differs from libopus, SNR {snr:.1} dB over {} samples", rust.len());
    }
}

/// The multistream packet of `streams` streams parses, and no stream has a
/// frame of 0 or 1 bytes, which a decoder conceals. With one stream it is
/// an ordinary packet.
pub fn whole_frames(packet: &[u8], streams: usize) -> bool {
    let mut off = 0;
    for s in 0..streams {
        match c_parse(&packet[off.min(packet.len())..], s + 1 != streams) {
            Ok(p) if p.sizes.iter().all(|&n| n > 1) => off += p.packet_offset,
            _ => return false,
        }
    }
    true
}

/// Whether each stream of a multistream packet of `streams` streams is
/// CELT-only (bit 7 of its first byte), or `None` when it does not parse.
fn celt_only_streams(packet: &[u8], streams: usize) -> Option<Vec<bool>> {
    let mut off = 0;
    let mut modes = Vec::with_capacity(streams);
    for s in 0..streams {
        let p = c_parse(&packet[off.min(packet.len())..], s + 1 != streams).ok()?;
        modes.push(p.toc & 0x80 != 0);
        off += p.packet_offset;
    }
    Some(modes)
}

/// The coding mode of the last packet each stream decoded.
///
/// When a stream goes from CELT-only to SILK-only or Hybrid, the decoder
/// conceals 5 ms in CELT and fades them into the new frame, unless the
/// packet carries a redundant CELT frame. That concealment searches the
/// decoded signal for its pitch, and a rounding difference can change what
/// it finds. So from such a packet on, the output of the two decoders is
/// not compared.
#[derive(Default)]
pub struct Modes(Option<Vec<bool>>);

impl Modes {
    /// Whether decoding `packet` takes a stream from CELT-only to another
    /// mode.
    pub fn leaves_celt(&self, packet: &[u8], streams: usize) -> bool {
        match (&self.0, celt_only_streams(packet, streams)) {
            (Some(prev), Some(now)) => prev.iter().zip(&now).any(|(&p, &n)| p && !n),
            _ => false,
        }
    }

    /// Both decoders decoded `packet`.
    pub fn decoded(&mut self, packet: &[u8], streams: usize) {
        if let Some(now) = celt_only_streams(packet, streams) {
            self.0 = Some(now);
        }
    }

    /// The decoders were reset.
    pub fn reset(&mut self) {
        self.0 = None;
    }
}

// ---------------------------------------------------------------------------
// Packet extensions
// ---------------------------------------------------------------------------
//
// libopus's extension code (src/extensions.c) and `opus_packet_pad_impl`
// are internal, but the static library keeps their names, as for
// `opus_packet_parse_impl` above. `tests/core_packet_extensions_libopus.rs`
// calls them in the same way.

/// libopus `opus_extension_data` (src/opus_private.h).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CExt {
    pub id: c_int,
    pub frame: c_int,
    pub data: *const c_uchar,
    pub len: i32,
}

pub const C_EXT_EMPTY: CExt = CExt { id: 0, frame: 0, data: std::ptr::null(), len: 0 };

/// libopus `OpusExtensionIterator` (src/opus_private.h).
#[repr(C)]
pub struct CIter {
    pub data: *const c_uchar,
    pub curr_data: *const c_uchar,
    pub repeat_data: *const c_uchar,
    pub last_long: *const c_uchar,
    pub src_data: *const c_uchar,
    pub len: i32,
    pub curr_len: i32,
    pub repeat_len: i32,
    pub src_len: i32,
    pub trailing_short_len: i32,
    pub nb_frames: c_int,
    pub frame_max: c_int,
    pub curr_frame: c_int,
    pub repeat_frame: c_int,
    pub repeat_l: c_uchar,
}

unsafe extern "C" {
    pub fn opus_extension_iterator_init(iter: *mut CIter, data: *const c_uchar, len: i32, nb_frames: i32);
    pub fn opus_extension_iterator_reset(iter: *mut CIter);
    pub fn opus_extension_iterator_set_frame_max(iter: *mut CIter, frame_max: c_int);
    pub fn opus_extension_iterator_next(iter: *mut CIter, ext: *mut CExt) -> c_int;
    pub fn opus_extension_iterator_find(iter: *mut CIter, ext: *mut CExt, id: c_int) -> c_int;
    pub fn opus_packet_extensions_count(data: *const c_uchar, len: i32, nb_frames: c_int) -> i32;
    pub fn opus_packet_extensions_count_ext(data: *const c_uchar, len: i32, nb_frame_exts: *mut i32, nb_frames: c_int) -> i32;
    pub fn opus_packet_extensions_parse(data: *const c_uchar, len: i32, exts: *mut CExt, nb: *mut i32, nb_frames: c_int) -> i32;
    pub fn opus_packet_extensions_parse_ext(
        data: *const c_uchar,
        len: i32,
        exts: *mut CExt,
        nb: *mut i32,
        nb_frame_exts: *const i32,
        nb_frames: c_int,
    ) -> i32;
    pub fn opus_packet_extensions_generate(
        data: *mut c_uchar,
        len: i32,
        exts: *const CExt,
        nb: i32,
        nb_frames: c_int,
        pad: c_int,
    ) -> i32;
    pub fn opus_packet_pad_impl(data: *mut c_uchar, len: i32, new_len: i32, pad: c_int, exts: *const CExt, nb: c_int) -> i32;
}

/// An extension as plain values: ID, frame, payload.
pub type Plain = (u8, usize, Vec<u8>);

pub fn plain(e: &opus1::extensions::Extension) -> Plain {
    (e.id, e.frame, e.data.to_vec())
}

pub fn c_plain(e: &CExt) -> Plain {
    let data = if e.len > 0 { unsafe { std::slice::from_raw_parts(e.data, e.len as usize) }.to_vec() } else { Vec::new() };
    (e.id as u8, e.frame as usize, data)
}

pub fn to_c(exts: &[opus1::extensions::Extension]) -> Vec<CExt> {
    exts.iter()
        .map(|e| CExt { id: e.id as c_int, frame: e.frame as c_int, data: e.data.as_ptr(), len: e.data.len() as i32 })
        .collect()
}

/// A list of up to `max` extensions for `nb_frames` frames from the input,
/// payloads cut from `pool`. Most are valid; some have an ID, a frame or a
/// short payload that `generate` refuses. Some lists repeat one pattern in
/// every frame from some frame on, so that `generate` writes repeats.
pub fn arbitrary_extensions<'a>(
    u: &mut Unstructured,
    pool: &'a [u8],
    nb_frames: usize,
    max: usize,
) -> Result<Vec<opus1::extensions::Extension<'a>>> {
    use opus1::extensions::Extension;
    let payload = |u: &mut Unstructured, id: u8| -> Result<&'a [u8]> {
        let len = if id < 32 {
            if u.ratio(1, 30)? { 2 } else { u.int_in_range(0..=1)? }
        } else {
            match u.int_in_range(0..=9)? {
                0 => u.int_in_range(0..=700)?,
                _ => u.int_in_range(0..=8)?,
            }
        };
        let len = len.min(pool.len());
        let start = u.int_in_range(0..=pool.len() - len)?;
        Ok(&pool[start..start + len])
    };
    let id = |u: &mut Unstructured| -> Result<u8> {
        Ok(if u.ratio(1, 40)? { u.arbitrary()? } else { u.int_in_range(3..=127)? })
    };
    let frame = |u: &mut Unstructured| -> Result<usize> {
        Ok(if u.ratio(1, 40)? { u.int_in_range(0..=60)? } else { u.int_in_range(0..=nb_frames.max(1) - 1)? })
    };
    let mut exts = Vec::new();
    if nb_frames > 0 && u.ratio(1, 3)? {
        // One pattern in every frame from `first` on.
        let first = u.int_in_range(0..=nb_frames - 1)?;
        let mut pattern = Vec::new();
        for _ in 0..u.int_in_range(1..=4)? {
            let i = id(u)?;
            pattern.push((i, payload(u, i)?));
        }
        for f in first..nb_frames {
            for &(id, data) in &pattern {
                if exts.len() < max {
                    exts.push(Extension { id, frame: f, data });
                }
            }
        }
    }
    let n = u.int_in_range(0..=max.saturating_sub(exts.len()).min(24))?;
    for _ in 0..n {
        let i = id(u)?;
        let data = payload(u, i)?;
        let at = u.int_in_range(0..=exts.len())?;
        exts.insert(at, Extension { id: i, frame: frame(u)?, data });
    }
    Ok(exts)
}
