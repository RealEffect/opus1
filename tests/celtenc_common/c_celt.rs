//! The CELT encoder inside C libopus, called directly.
//!
//! The tests link libopus as a static library. Its CELT encoder functions
//! are not part of the public API, but a static library keeps their names,
//! so they can be declared and called here. This is the encoder that
//! `opus_encode` calls for a CELT-only or Hybrid frame, without the Opus
//! layer in front of it: no DC filter, no mode decision, no analysis.
//!
//! `CeltEncoder` of opus1 is the port of exactly this code. Both get the
//! same samples and the same settings, so their packets can be compared
//! byte by byte.
#![allow(dead_code)]

use super::{Cfg, Format, celt_toc_lm, parse_frame};
use opus1::internals::analysis::{AnalysisInfo, LEAK_BANDS};
use opus1::internals::celt::CeltEncoder;
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;
use std::os::raw::{c_int, c_uchar, c_void};

unsafe extern "C" {
    fn celt_encoder_get_size(channels: c_int) -> c_int;
    fn celt_encoder_init(
        st: *mut c_void,
        sampling_rate: i32,
        channels: c_int,
        arch: c_int,
    ) -> c_int;
    fn opus_custom_encoder_ctl(st: *mut c_void, request: c_int, ...) -> c_int;
    fn celt_encode_with_ec(
        st: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        compressed: *mut c_uchar,
        nb_compressed_bytes: c_int,
        enc: *mut c_void,
    ) -> c_int;
}

// Requests of `opus_custom_encoder_ctl` (opus_defines.h, celt.h).
pub const OPUS_SET_BITRATE: c_int = 4002;
pub const OPUS_SET_VBR: c_int = 4006;
pub const OPUS_SET_COMPLEXITY: c_int = 4010;
pub const OPUS_SET_PACKET_LOSS_PERC: c_int = 4014;
pub const OPUS_SET_VBR_CONSTRAINT: c_int = 4020;
pub const OPUS_RESET_STATE: c_int = 4028;
pub const OPUS_SET_LSB_DEPTH: c_int = 4036;
pub const CELT_SET_PREDICTION: c_int = 10002;
pub const CELT_SET_CHANNELS: c_int = 10008;
pub const CELT_SET_START_BAND: c_int = 10010;
pub const CELT_SET_END_BAND: c_int = 10012;
pub const CELT_SET_SIGNALLING: c_int = 10016;
pub const CELT_SET_ANALYSIS: c_int = 10022;
pub const OPUS_SET_LFE: c_int = 10024;
/// libopus `OPUS_BITRATE_MAX`.
pub const BITRATE_MAX: i32 = -1;

/// libopus `AnalysisInfo` (celt/celt.h:65-79), 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CAnalysisInfo {
    pub valid: c_int,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: c_int,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    pub leak_boost: [u8; LEAK_BANDS],
}

impl From<&AnalysisInfo> for CAnalysisInfo {
    fn from(a: &AnalysisInfo) -> Self {
        CAnalysisInfo {
            valid: a.valid as c_int,
            tonality: a.tonality,
            tonality_slope: a.tonality_slope,
            noisiness: a.noisiness,
            activity: a.activity,
            music_prob: a.music_prob,
            music_prob_min: a.music_prob_min,
            music_prob_max: a.music_prob_max,
            bandwidth: a.bandwidth,
            activity_probability: a.activity_probability,
            max_pitch_ratio: a.max_pitch_ratio,
            leak_boost: a.leak_boost,
        }
    }
}

/// The CELT encoder of C libopus (`CELTEncoder`).
pub struct CCelt {
    // 8-byte units keep the state aligned for its pointers and floats.
    state: Vec<u64>,
    pub channels: usize,
}

impl CCelt {
    /// A new encoder, set up the way `opus_encoder_init` sets up its CELT
    /// encoder: no frame-size byte in front of the payload.
    pub fn new(fs: usize, channels: usize) -> Self {
        let bytes = unsafe { celt_encoder_get_size(channels as c_int) } as usize;
        assert!(bytes > 0, "celt_encoder_get_size");
        let mut enc = CCelt {
            state: vec![0u64; bytes.div_ceil(8)],
            channels,
        };
        let ret = unsafe { celt_encoder_init(enc.ptr(), fs as i32, channels as c_int, 0) };
        assert_eq!(ret, 0, "celt_encoder_init");
        enc.ctl(CELT_SET_SIGNALLING, 0);
        enc
    }

    fn ptr(&mut self) -> *mut c_void {
        self.state.as_mut_ptr().cast()
    }

    /// A request with one integer argument.
    pub fn ctl(&mut self, request: c_int, value: c_int) {
        let ret = unsafe { opus_custom_encoder_ctl(self.ptr(), request, value) };
        assert_eq!(ret, 0, "opus_custom_encoder_ctl({request}, {value})");
    }

    /// `CELT_SET_ANALYSIS`: the analysis the next frames are coded with.
    pub fn set_analysis(&mut self, info: &AnalysisInfo) {
        let c_info = CAnalysisInfo::from(info);
        let ret = unsafe {
            opus_custom_encoder_ctl(self.ptr(), CELT_SET_ANALYSIS, &c_info as *const CAnalysisInfo)
        };
        assert_eq!(ret, 0, "CELT_SET_ANALYSIS");
    }

    /// `OPUS_RESET_STATE`.
    pub fn reset(&mut self) {
        let ret = unsafe { opus_custom_encoder_ctl(self.ptr(), OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
    }

    /// Encode one frame of interleaved samples into at most `max_bytes`
    /// bytes. Returns the CELT payload.
    pub fn encode(&mut self, pcm: &[f32], frame: usize, max_bytes: usize) -> Vec<u8> {
        assert_eq!(pcm.len(), frame * self.channels);
        let mut out = vec![0u8; max_bytes];
        let n = unsafe {
            celt_encode_with_ec(
                self.ptr(),
                pcm.as_ptr(),
                frame as c_int,
                out.as_mut_ptr(),
                max_bytes as c_int,
                std::ptr::null_mut(),
            )
        };
        assert!(n > 0, "celt_encode_with_ec failed: {n}");
        out.truncate(n as usize);
        out
    }
}

/// In how many frames of two streams each field has the same value
/// (`Stream::agreement`).
#[derive(Clone, Debug, Default)]
pub struct Agreement {
    pub frames: usize,
    /// Packet size in bytes.
    pub size: usize,
    /// Post-filter on or off, with period, gain and tap set.
    pub postfilter: usize,
    pub transient: usize,
    pub intra: usize,
    /// Coarse energy steps of all bands.
    pub coarse: usize,
    pub tf: usize,
    pub spread: usize,
    /// Dynamic allocation steps of all bands.
    pub boost: usize,
    pub trim: usize,
    /// Intensity band and dual stereo flag.
    pub stereo: usize,
    pub coded_bands: usize,
    /// Bits for the shape of each band.
    pub pulses: usize,
    /// Fine energy bits and values.
    pub fine: usize,
}

impl Agreement {
    /// `count` is at least `percent` % of the frames.
    pub fn at_least(&self, count: usize, percent: usize) -> bool {
        count * 100 >= self.frames * percent
    }
}

/// Settings of one CELT stream, for both encoders.
#[derive(Clone, Debug)]
pub struct Stream {
    /// Sampling rate of the input, in Hz.
    pub fs: usize,
    /// Channels of the input.
    pub channels: usize,
    /// Channels coded in the packets (`CELT_SET_CHANNELS`).
    pub stream_channels: usize,
    /// Samples per channel in one frame, at `fs`.
    pub frame: usize,
    pub bitrate: i32,
    pub vbr: bool,
    pub constrained_vbr: bool,
    pub complexity: i32,
    /// First coded band: 0, or 17 for the CELT part of a Hybrid frame.
    pub start_band: usize,
    /// Number of the band after the last coded one.
    pub end_band: usize,
    /// `CELT_SET_PREDICTION`: 0, 1 or 2. 2 is the default of libopus.
    pub prediction: i32,
    /// `OPUS_SET_PACKET_LOSS_PERC`.
    pub loss: i32,
}

impl Stream {
    /// `frame_ms_x2` is the frame duration in half milliseconds: 5 for
    /// 2.5 ms, 10, 20 or 40 for 20 ms.
    pub fn new(fs: usize, channels: usize, frame_ms_x2: usize, bitrate: i32) -> Self {
        Stream {
            fs,
            channels,
            stream_channels: channels,
            frame: fs * frame_ms_x2 / 2000,
            bitrate,
            vbr: false,
            constrained_vbr: true,
            complexity: 5,
            start_band: 0,
            end_band: 21,
            prediction: 2,
            loss: 0,
        }
    }

    pub fn format(&self) -> Format {
        Format {
            fs: self.fs,
            frame: self.frame,
        }
    }

    /// Bytes of one CELT payload at a constant bitrate. This is the packet
    /// size of `opus_encode` less the TOC byte.
    pub fn cbr_bytes(&self) -> usize {
        self.format().cbr_packet_bytes(self.bitrate) - 1
    }

    /// Room for one CELT payload: the constant size, or the largest payload
    /// when the bitrate is variable.
    pub fn max_bytes(&self) -> usize {
        if self.vbr { 1275 } else { self.cbr_bytes() }
    }

    /// The C encoder with these settings, as `opus_encode` leaves its CELT
    /// encoder before a frame (opus_encoder.c: with a constant bitrate CELT
    /// stays at `OPUS_BITRATE_MAX` and fills the bytes it is given).
    pub fn c_encoder(&self) -> CCelt {
        let mut enc = CCelt::new(self.fs, self.channels);
        self.c_settings(&mut enc);
        enc
    }

    /// Apply the settings to a C encoder. `OPUS_RESET_STATE` keeps them, so
    /// this is needed once.
    pub fn c_settings(&self, enc: &mut CCelt) {
        enc.ctl(OPUS_SET_COMPLEXITY, self.complexity);
        enc.ctl(CELT_SET_START_BAND, self.start_band as c_int);
        enc.ctl(CELT_SET_END_BAND, self.end_band as c_int);
        enc.ctl(CELT_SET_CHANNELS, self.stream_channels as c_int);
        enc.ctl(CELT_SET_PREDICTION, self.prediction);
        enc.ctl(OPUS_SET_PACKET_LOSS_PERC, self.loss);
        enc.ctl(OPUS_SET_VBR, self.vbr as c_int);
        enc.ctl(OPUS_SET_VBR_CONSTRAINT, self.constrained_vbr as c_int);
        enc.ctl(
            OPUS_SET_BITRATE,
            if self.vbr { self.bitrate } else { BITRATE_MAX },
        );
    }

    /// The opus1 encoder with the same settings.
    pub fn rs_encoder(&self) -> CeltEncoder {
        let mut enc = CeltEncoder::with_sampling_rate(default_mode(), self.channels, self.fs as i32);
        self.rs_settings(&mut enc);
        enc
    }

    /// Apply the settings to an opus1 encoder.
    pub fn rs_settings(&self, enc: &mut CeltEncoder) {
        enc.complexity = self.complexity;
        enc.set_end_band(self.end_band);
        enc.set_bitrate(if self.vbr { self.bitrate } else { BITRATE_MAX })
            .expect("bitrate");
        enc.set_vbr(self.vbr);
        enc.set_constrained_vbr(self.constrained_vbr);
        enc.set_prediction(self.prediction).expect("prediction");
        enc.set_packet_loss_perc(self.loss).expect("packet loss");
        enc.set_stream_channels(self.stream_channels)
            .expect("stream channels");
        enc.set_start_band(self.start_band).expect("start band");
    }

    /// One frame of interleaved samples through the C encoder.
    pub fn c_frame(&self, enc: &mut CCelt, frame: &[f32]) -> Vec<u8> {
        enc.encode(frame, self.frame, self.max_bytes())
    }

    /// One frame of interleaved samples through the opus1 encoder.
    pub fn rs_frame(&self, enc: &mut CeltEncoder, frame: &[f32]) -> Vec<u8> {
        let bytes = self.max_bytes();
        let mut rc = RangeCoder::new_encoder(bytes as u32);
        enc.encode(frame, self.frame, &mut rc);
        rc.done();
        assert_eq!(rc.error, 0, "range coder error");
        rc.buf[..rc.storage as usize].to_vec()
    }

    /// Encode interleaved `pcm` with the C encoder. Returns the CELT
    /// payloads.
    pub fn enc_c(&self, pcm: &[f32]) -> Vec<Vec<u8>> {
        let mut enc = self.c_encoder();
        pcm.chunks_exact(self.frame * self.channels)
            .map(|frame| self.c_frame(&mut enc, frame))
            .collect()
    }

    /// Encode interleaved `pcm` with opus1. Returns the CELT payloads.
    pub fn enc_rs(&self, pcm: &[f32]) -> Vec<Vec<u8>> {
        let mut enc = self.rs_encoder();
        pcm.chunks_exact(self.frame * self.channels)
            .map(|frame| self.rs_frame(&mut enc, frame))
            .collect()
    }

    /// Encode interleaved `pcm` with both encoders. `before` is called
    /// ahead of every frame with the frame number and the two encoders, so
    /// it can change their settings in the middle of the stream. Returns
    /// the payloads of libopus and of opus1.
    pub fn enc_both_with(
        &self,
        pcm: &[f32],
        mut before: impl FnMut(usize, &mut CCelt, &mut CeltEncoder),
    ) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut c_enc = self.c_encoder();
        let mut rs_enc = self.rs_encoder();
        pcm.chunks_exact(self.frame * self.channels)
            .enumerate()
            .map(|(k, frame)| {
                before(k, &mut c_enc, &mut rs_enc);
                (
                    self.c_frame(&mut c_enc, frame),
                    self.rs_frame(&mut rs_enc, frame),
                )
            })
            .unzip()
    }

    /// The payloads as CELT-only Opus packets, which `dec_c_fmt` decodes.
    /// Only for streams that start at band 0.
    pub fn packets(&self, payloads: &[Vec<u8>]) -> Vec<Vec<u8>> {
        assert_eq!(self.start_band, 0);
        let toc = celt_toc_lm(self.end_band, self.stream_channels, self.format().lm());
        payloads
            .iter()
            .map(|p| {
                let mut pkt = Vec::with_capacity(p.len() + 1);
                pkt.push(toc);
                pkt.extend_from_slice(p);
                pkt
            })
            .collect()
    }

    /// Signal-to-noise ratio in dB of the stream `payloads`, decoded by C
    /// libopus, against the input `pcm`. The first 100 ms are left out.
    pub fn snr_db(&self, pcm: &[f32], payloads: &[Vec<u8>]) -> f64 {
        let fmt = self.format();
        let decoded = super::dec_c_fmt(&self.packets(payloads), self.channels, fmt);
        let reference: Vec<f32> = if self.stream_channels < self.channels {
            // The stream carries the mean of the two channels, and the
            // decoder plays it on both.
            pcm.as_chunks::<2>().0.iter()
                .flat_map(|s| {
                    let m = 0.5 * (s[0] + s[1]);
                    [m, m]
                })
                .collect()
        } else {
            pcm.to_vec()
        };
        super::snr_db_delay(&reference, &decoded, self.channels, self.fs / 10, fmt.delay())
    }

    /// Number of frames at the start of the two streams that are the same,
    /// byte for byte.
    pub fn same_prefix(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// Number of frames that are the same in both streams.
    pub fn same_frames(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x == y).count()
    }

    /// For each field of the header and of the allocation, in how many
    /// frames the two streams have the same value.
    pub fn agreement(&self, c: &[Vec<u8>], rs: &[Vec<u8>]) -> Agreement {
        let lm = self.format().lm();
        let mut a = Agreement {
            frames: c.len(),
            ..Agreement::default()
        };
        for (pc, pr) in c.iter().zip(rs) {
            let (hc, ac) = parse_frame(pc, self.stream_channels, self.end_band, lm);
            let (hr, ar) = parse_frame(pr, self.stream_channels, self.end_band, lm);
            let (ac, ar) = (ac.unwrap_or_default(), ar.unwrap_or_default());
            a.size += (pc.len() == pr.len()) as usize;
            a.postfilter += (hc.postfilter == hr.postfilter) as usize;
            a.transient += (hc.transient == hr.transient) as usize;
            a.intra += (hc.intra == hr.intra) as usize;
            a.coarse += (hc.coarse == hr.coarse) as usize;
            a.tf += (hc.tf == hr.tf) as usize;
            a.spread += (hc.spread == hr.spread) as usize;
            a.boost += (hc.boost == hr.boost) as usize;
            a.trim += (hc.trim == hr.trim) as usize;
            a.stereo += (ac.intensity == ar.intensity && ac.dual_stereo == ar.dual_stereo) as usize;
            a.coded_bands += (ac.coded_bands == ar.coded_bands) as usize;
            a.pulses += (ac.pulses == ar.pulses) as usize;
            a.fine += (ac.fine_bits == ar.fine_bits && ac.fine == ar.fine) as usize;
        }
        a
    }

    /// `Cfg` with the channel count and band range of this stream, for
    /// `parse_header`.
    pub fn cfg(&self) -> Cfg {
        let mut cfg = Cfg::new(self.stream_channels, self.bitrate);
        cfg.end_band = self.end_band;
        cfg.vbr = self.vbr;
        cfg.constrained_vbr = self.constrained_vbr;
        cfg
    }
}
