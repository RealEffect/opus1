//! What the tests of the neural network features inside the decoder
//! share (`dnn_plc_decoder_libopus.rs`, `dnn_osce_decoder_libopus.rs`):
//! libopus decoders with the controls of these features, the speech
//! signal, the coding of test streams by libopus and by opus1, and the
//! comparison of outputs.
//!
//! A test file that uses this module also declares `core_common`,
//! `dnn_plc_common` and `k24_common`.

// Each test file uses a part of this module.
#![allow(dead_code)]

use crate::core_common::*;
use crate::{dnn_plc_common, k24_common};
use audiopus_sys as sys;
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusEncoder};
use std::os::raw::c_int;
use std::path::Path;

pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
/// `OPUS_SET_OSCE_BWE_REQUEST`; a libopus without the bandwidth extension
/// answers `OPUS_UNIMPLEMENTED`.
pub const OPUS_SET_OSCE_BWE_REQUEST: c_int = 4054;
/// `OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST`.
pub const OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST: c_int = 5122;

// ---------------------------------------------------------------------------
// libopus decoders
// ---------------------------------------------------------------------------

/// A libopus decoder (`opus_decoder_create`).
pub struct CDec {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
}

impl CDec {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        CDec { st, channels }
    }

    pub fn set_complexity(&mut self, complexity: i32) {
        let ret = unsafe { sys::opus_decoder_ctl(self.st, OPUS_SET_COMPLEXITY_REQUEST, complexity) };
        assert_eq!(ret, 0, "OPUS_SET_COMPLEXITY({complexity})");
    }

    /// `OPUS_SET_OSCE_BWE`. Panics when the linked libopus has no
    /// bandwidth extension.
    pub fn set_osce_bwe(&mut self, on: bool) {
        let ret = unsafe { sys::opus_decoder_ctl(self.st, OPUS_SET_OSCE_BWE_REQUEST, on as i32) };
        assert_eq!(ret, 0, "OPUS_SET_OSCE_BWE: the linked libopus has no bandwidth extension");
    }

    pub fn reset(&mut self) {
        assert_eq!(unsafe { sys::opus_decoder_ctl(self.st, sys::OPUS_RESET_STATE) }, 0);
    }

    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe { sys::opus_decoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32) };
        assert_eq!(ret, 0);
        value
    }

    /// `opus_decode_float`; `None` is a lost packet.
    pub fn decode(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<f32>, i32> {
        let mut pcm = vec![0f32; frame_size * self.channels];
        let (ptr, len) = packet.map_or((std::ptr::null(), 0), |p| (p.as_ptr(), p.len() as i32));
        let n = unsafe { sys::opus_decode_float(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as i32, fec as i32) };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    /// `opus_decode`; `None` is a lost packet.
    pub fn decode_i16(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<i16>, i32> {
        let mut pcm = vec![0i16; frame_size * self.channels];
        let (ptr, len) = packet.map_or((std::ptr::null(), 0), |p| (p.as_ptr(), p.len() as i32));
        let n = unsafe { sys::opus_decode(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as i32, fec as i32) };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }
}

impl Drop for CDec {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// A libopus multistream decoder (`opus_multistream_decoder_create`).
pub struct CMsDec {
    st: *mut sys::OpusMSDecoder,
    pub channels: usize,
    streams: usize,
}

impl CMsDec {
    pub fn new(rate: usize, channels: usize, streams: usize, coupled: usize, mapping: &[u8]) -> Self {
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(
                rate as i32,
                channels as c_int,
                streams as c_int,
                coupled as c_int,
                mapping.as_ptr(),
                &mut err,
            )
        };
        assert_eq!(err, 0, "opus_multistream_decoder_create");
        CMsDec { st, channels, streams }
    }

    pub fn set_complexity(&mut self, complexity: i32) {
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.st, OPUS_SET_COMPLEXITY_REQUEST, complexity) };
        assert_eq!(ret, 0);
    }

    /// `OPUS_SET_OSCE_BWE` on each stream decoder, which libopus reaches
    /// through `OPUS_MULTISTREAM_GET_DECODER_STATE`: its multistream
    /// decoder does not forward the request.
    pub fn set_osce_bwe(&mut self, on: bool) {
        for s in 0..self.streams {
            let mut dec: *mut sys::OpusDecoder = std::ptr::null_mut();
            let ret = unsafe {
                sys::opus_multistream_decoder_ctl(
                    self.st,
                    OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
                    s as c_int,
                    &mut dec as *mut *mut sys::OpusDecoder,
                )
            };
            assert_eq!(ret, 0);
            let ret = unsafe { sys::opus_decoder_ctl(dec, OPUS_SET_OSCE_BWE_REQUEST, on as i32) };
            assert_eq!(ret, 0, "OPUS_SET_OSCE_BWE: the linked libopus has no bandwidth extension");
        }
    }

    pub fn reset(&mut self) {
        assert_eq!(unsafe { sys::opus_multistream_decoder_ctl(self.st, sys::OPUS_RESET_STATE) }, 0);
    }

    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_multistream_decoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32)
        };
        assert_eq!(ret, 0);
        value
    }

    pub fn decode(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<f32>, i32> {
        let mut pcm = vec![0f32; frame_size * self.channels];
        let (ptr, len) = packet.map_or((std::ptr::null(), 0), |p| (p.as_ptr(), p.len() as i32));
        let n = unsafe {
            sys::opus_multistream_decode_float(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as c_int, fec as c_int)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    pub fn decode_i16(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Result<Vec<i16>, i32> {
        let mut pcm = vec![0i16; frame_size * self.channels];
        let (ptr, len) = packet.map_or((std::ptr::null(), 0), |p| (p.as_ptr(), p.len() as i32));
        let n = unsafe {
            sys::opus_multistream_decode(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as c_int, fec as c_int)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }
}

impl Drop for CMsDec {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_decoder_destroy(self.st) }
    }
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// Speech at 48 kHz, half of full scale: the four excerpts of
/// `fixtures/tsp/` (10 s), and with `corpus` also files of the TSP
/// corpus, `seconds` long, starting at file `first`.
pub fn speech48(seconds: f64, first: usize, corpus: bool) -> Vec<f32> {
    let mut files: Vec<std::path::PathBuf> = ["FA01_05", "MD19_04", "FJ55_02", "MH43_04"]
        .iter()
        .map(|f| Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/tsp/{f}.wav")))
        .collect();
    if corpus {
        files.extend(dnn_plc_common::tsp_files().into_iter().step_by(37));
    }
    let need = (seconds * 48000.0) as usize;
    let mut out = Vec::with_capacity(need);
    let mut k = first;
    while out.len() < need {
        let (s, rate) = dnn_plc_common::read_wav(&files[k % files.len()]);
        assert_eq!(rate, 48000);
        out.extend(s.iter().map(|&v| 0.5 * v as f32 / 32768.0));
        k += 1;
    }
    out.truncate(need);
    out
}

/// `seconds` of input for an encoder at `rate` with `channels`: speech,
/// with a short pause of low noise every 2 s. The second channel is the
/// speech of another talker at a lower level.
pub fn input(rate: usize, channels: usize, seconds: f64, seed: usize, corpus: bool) -> Vec<f32> {
    let a = k24_common::resample(&speech48(seconds, seed, corpus), 48000, rate);
    let b = k24_common::resample(&speech48(seconds, seed + 1, corpus), 48000, rate);
    let mut state = 0x2545_f491u32.wrapping_add(seed as u32);
    let mut out = Vec::with_capacity(a.len() * channels);
    for i in 0..a.len() {
        let pause = (i * 4 / rate) % 8 == 7;
        for c in 0..channels {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((state >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.001;
            let s = if c == 0 { a[i] } else { 0.6 * b[i] + 0.3 * a[i] };
            out.push(if pause { noise } else { s });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// How a stream is coded.
#[derive(Clone, Debug)]
pub struct Coding {
    pub label: String,
    pub rate: usize,
    pub channels: usize,
    pub application: i32,
    /// `OPUS_SET_FORCE_MODE`, if any.
    pub mode: Option<i32>,
    /// `OPUS_SET_BANDWIDTH`, if any.
    pub bandwidth: Option<i32>,
    /// Frame durations in units of 2.5 ms, one after the other.
    pub durations: Vec<usize>,
    /// Bitrates in b/s; the bitrate changes every 10 packets.
    pub bitrates: Vec<i32>,
    pub fec: bool,
    /// Discontinuous transmission (libopus's encoder only): frames of
    /// silence are sent as 1-byte packets, which the decoder conceals.
    pub dtx: bool,
}

impl Coding {
    pub fn new(label: &str, rate: usize, channels: usize, mode: Option<i32>, bandwidth: Option<i32>) -> Self {
        Coding {
            label: label.to_string(),
            rate,
            channels,
            application: APP_VOIP,
            mode,
            bandwidth,
            durations: vec![8],
            bitrates: vec![24000 * channels as i32],
            fec: false,
            dtx: false,
        }
    }
    pub fn durations(mut self, d: &[usize]) -> Self {
        self.durations = d.to_vec();
        self
    }
    pub fn bitrates(mut self, b: &[i32]) -> Self {
        self.bitrates = b.to_vec();
        self
    }
    pub fn fec(mut self) -> Self {
        self.fec = true;
        self
    }
    pub fn application(mut self, app: i32) -> Self {
        self.application = app;
        self
    }
    pub fn dtx(mut self) -> Self {
        self.dtx = true;
        self
    }
}

/// Cuts `pcm` into the frames of `coding` and codes each with `encode`.
pub fn encode_frames(coding: &Coding, pcm: &[f32], mut encode: impl FnMut(&[f32], i32) -> Vec<u8>) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut pos = 0;
    for k in 0.. {
        let n = coding.durations[k % coding.durations.len()] * coding.rate / 400 * coding.channels;
        if pos + n > pcm.len() {
            break;
        }
        packets.push(encode(&pcm[pos..pos + n], coding.bitrates[k / 10 % coding.bitrates.len()]));
        pos += n;
    }
    packets
}

/// Packets of libopus's encoder.
pub fn c_encode(coding: &Coding, pcm: &[f32]) -> Vec<Vec<u8>> {
    let mut enc = CEncoder::new(coding.rate, coding.channels, coding.application);
    if let Some(mode) = coding.mode {
        enc.force_mode(mode);
    }
    if let Some(bw) = coding.bandwidth {
        enc.bandwidth(bw);
    }
    if coding.fec {
        enc.fec(true, 20);
    }
    if coding.dtx {
        enc.set(sys::OPUS_SET_DTX_REQUEST, 1);
    }
    encode_frames(coding, pcm, |frame, bitrate| {
        enc.bitrate(bitrate);
        enc.encode(frame)
    })
}

/// Packets of opus1's encoder.
pub fn rs_encode(coding: &Coding, pcm: &[f32]) -> Vec<Vec<u8>> {
    let application = match coding.application {
        APP_VOIP => Application::Voip,
        APP_AUDIO => Application::Audio,
        _ => Application::RestrictedLowDelay,
    };
    let mut enc = OpusEncoder::new(coding.rate as i32, coding.channels, application).unwrap();
    enc.set_force_mode(coding.mode.map(|m| match m {
        MODE_SILK => ForcedMode::SilkOnly,
        MODE_HYBRID => ForcedMode::Hybrid,
        _ => ForcedMode::CeltOnly,
    }));
    if let Some(bw) = coding.bandwidth {
        enc.set_bandwidth(match bw {
            BW_NB => Bandwidth::Narrowband,
            BW_MB => Bandwidth::Mediumband,
            BW_WB => Bandwidth::Wideband,
            BW_SWB => Bandwidth::Superwideband,
            _ => Bandwidth::Fullband,
        });
    }
    if coding.fec {
        enc.set_inband_fec(true);
        enc.set_packet_loss_perc(20).unwrap();
    }
    let mut buf = vec![0u8; 4000];
    let channels = coding.channels;
    encode_frames(coding, pcm, |frame, bitrate| {
        enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
        let len = enc.encode(frame, frame.len() / channels, &mut buf).unwrap();
        buf[..len].to_vec()
    })
}

/// The two streams' packets in turns of `run` packets: a switch between
/// them has no redundancy frame.
pub fn splice(a: &[Vec<u8>], b: &[Vec<u8>], run: usize) -> Vec<Vec<u8>> {
    let n = a.len().min(b.len());
    (0..n).map(|k| if (k / run).is_multiple_of(2) { a[k].clone() } else { b[k].clone() }).collect()
}

/// The first sample where two outputs differ, as text.
pub fn first_difference<T: PartialEq + std::fmt::Debug + Copy>(
    want: &[T],
    got: &[T],
    same: impl Fn(T, T) -> bool,
) -> Option<String> {
    if want.len() != got.len() {
        return Some(format!("{} samples, opus1 {}", want.len(), got.len()));
    }
    let i = want.iter().zip(got).position(|(&a, &b)| !same(a, b))?;
    Some(format!("sample {i}: libopus {:?}, opus1 {:?}", want[i], got[i]))
}

/// The coding mode of a packet's first frame, from its first byte.
pub fn toc_mode(packet: &[u8]) -> i32 {
    let config = packet[0] >> 3;
    if config >= 16 {
        MODE_CELT
    } else if config >= 12 {
        MODE_HYBRID
    } else {
        MODE_SILK
    }
}

/// Whether a packet is SILK-only wideband, the SILK packets the bandwidth
/// extension takes.
pub fn is_silk_wb(packet: &[u8]) -> bool {
    (8..12).contains(&(packet[0] >> 3))
}
