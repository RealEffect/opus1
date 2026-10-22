//! The surround mask against C libopus.
//!
//! libopus computes the mask inside its multistream encoder and does not
//! give it out. What can be seen from outside is what the stream encoders
//! do with it. So the tests here rebuild the C surround encoder from its
//! parts:
//!
//! - the C surround encoder codes a signal and the packet is split into its
//!   streams;
//! - C single-stream encoders, set up exactly like the streams of the
//!   surround encoder, code the same channels. They get the mask that
//!   opus1 computed, or no mask.
//!
//! When the opus1 mask is the mask libopus computed, the single-stream
//! encoders write the same bytes as the surround encoder. Without a mask
//! they write other packets.
//!
//! The two masks are computed in `f32` by different code, so they differ by
//! rounding. In a few of 10000 packets that is enough to change a decision
//! of the encoder. It was seen at 12, 16 and 24 kHz on a signal with bands
//! that hold no signal: such a band holds the rounding noise of the MDCT,
//! which is largest when zeros are inserted between the samples. In three
//! of four such cases that were looked at, a mask shifted by 1e-4 or 1e-3
//! gives the bytes of libopus back.
//!
//! One test is a control: it takes the mask from `surround_analysis` of
//! libopus, called directly, instead of opus1. Every packet is then the
//! same. So the single-stream encoders are set up like the streams of the
//! surround encoder, and a packet that differs with the opus1 mask
//! differs because of the mask.
//!
//! The last test hands the mask to opus1 stream encoders and compares
//! their packet sizes with libopus.
//!
//! The encoder without a mask is never the C multistream encoder made with
//! `opus_multistream_encoder_create` for the same layout. That one runs no
//! surround analysis, but it also has no low-frequency-effects stream. It
//! therefore gives every stream another bitrate (5.1 at 256 kb/s: 78.7,
//! 78.7, 49.3 and 49.3 kb/s instead of 95.1, 95.1, 57.6 and 8.2), and it
//! does not force CELT on the stereo streams. For these reasons alone its
//! streams are about 20 % smaller than those of the surround encoder.
// The multistream API needs the `std` feature.
#![cfg(feature = "std")]

mod ms_support;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use ms_support::{
    APP_AUDIO, CMsEncoder, MODE_CELT_ONLY, Mode, OPUS_SET_ENERGY_MASK_REQUEST,
    OPUS_SET_FORCE_MODE_REQUEST, OPUS_SET_LFE_REQUEST, max_packet_bytes, noise, split_streams, toc,
    tones, voiced,
};
use opus1::internals::multistream::surround::SurroundProbe;
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};
use std::os::raw::{c_int, c_void};

const NB_BANDS: usize = 21;
/// `celt_mode->overlap` of the one CELT mode libopus has.
const OVERLAP: usize = 120;
/// Private request `CELT_GET_MODE` (celt.h). `opus_encoder_ctl` hands it to
/// the CELT encoder. Its argument is a pointer to `*const CELTMode`.
const CELT_GET_MODE_REQUEST: c_int = 10015;

/// libopus `opus_copy_channel_in_func` (opus_private.h), float build.
type CopyChannelIn = unsafe extern "C" fn(
    dst: *mut f32,
    dst_stride: c_int,
    src: *const c_void,
    src_stride: c_int,
    src_channel: c_int,
    frame_size: c_int,
    user_data: *mut c_void,
);

unsafe extern "C" {
    /// opus_multistream_encoder.c 230. It is not a `static` function, and
    /// libopus is linked as a static library, so the test can call it.
    /// `tests/multistream_surround_c_direct.rs` compares its result with
    /// the opus1 mask value by value.
    fn surround_analysis(
        celt_mode: *const c_void,
        pcm: *const c_void,
        band_log_e: *mut f32,
        mem: *mut f32,
        preemph_mem: *mut f32,
        len: c_int,
        overlap: c_int,
        channels: c_int,
        rate: c_int,
        copy_channel_in: CopyChannelIn,
        arch: c_int,
    );
}

/// `opus_copy_channel_in_float` of opus_multistream_encoder.c.
unsafe extern "C" fn copy_channel_in_float(
    dst: *mut f32,
    dst_stride: c_int,
    src: *const c_void,
    src_stride: c_int,
    src_channel: c_int,
    frame_size: c_int,
    _user_data: *mut c_void,
) {
    let src = src as *const f32;
    for i in 0..frame_size as usize {
        unsafe {
            *dst.add(i * dst_stride as usize) =
                *src.add(i * src_stride as usize + src_channel as usize);
        }
    }
}

/// C `surround_analysis` with the two memories the C multistream encoder
/// keeps for it from frame to frame.
struct CAnalysis {
    mode: *const c_void,
    channels: usize,
    rate: i32,
    mem: Vec<f32>,
    preemph_mem: Vec<f32>,
}

impl CAnalysis {
    fn new(channels: usize, rate: i32) -> Self {
        // The CELT mode is a constant table inside libopus. It is read
        // from an encoder as the C multistream encoder reads it.
        let mut err = 0;
        let mut mode: *const c_void = std::ptr::null();
        unsafe {
            let enc = sys::opus_encoder_create(48_000, 1, APP_AUDIO, &mut err);
            assert!(!enc.is_null() && err == 0, "opus_encoder_create = {err}");
            let ret =
                sys::opus_encoder_ctl(enc, CELT_GET_MODE_REQUEST, &mut mode as *mut *const c_void);
            sys::opus_encoder_destroy(enc);
            assert_eq!(ret, 0, "CELT_GET_MODE = {ret}");
        }
        assert!(!mode.is_null());
        Self {
            mode,
            channels,
            rate,
            mem: vec![0.0; channels * OVERLAP],
            preemph_mem: vec![0.0; channels],
        }
    }

    /// The mask of one frame of interleaved samples: 21 values per channel.
    fn run(&mut self, pcm: &[f32], frame_size: usize) -> Vec<f32> {
        assert_eq!(pcm.len(), frame_size * self.channels);
        let mut band_smr = vec![f32::NAN; NB_BANDS * self.channels];
        // `arch` 0 selects the plain C code in a build of libopus that
        // picks vector code at run time, and is ignored in other builds.
        unsafe {
            surround_analysis(
                self.mode,
                pcm.as_ptr() as *const c_void,
                band_smr.as_mut_ptr(),
                self.mem.as_mut_ptr(),
                self.preemph_mem.as_mut_ptr(),
                frame_size as c_int,
                OVERLAP as c_int,
                self.channels as c_int,
                self.rate,
                copy_channel_in_float,
                0,
            );
        }
        band_smr
    }
}

/// Where the mask of the single-stream encoders comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskFrom {
    /// The analysis of opus1, which is what the tests are about.
    OpusRs,
    /// `surround_analysis` of libopus, as a control.
    LibOpus,
}

/// A C single-stream encoder with an energy mask.
struct CStream {
    st: *mut sys::OpusEncoder,
    channels: usize,
    /// libopus keeps a pointer to the mask and reads it in every encode
    /// call, so the values stay here at a fixed address.
    mask: Box<[f32; 2 * NB_BANDS]>,
}

impl CStream {
    fn new(fs: i32, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_encoder_create(fs, channels as i32, APP_AUDIO, &mut err) };
        assert!(!st.is_null() && err == 0, "opus_encoder_create = {err}");
        Self {
            st,
            channels,
            mask: Box::new([0.0; 2 * NB_BANDS]),
        }
    }

    fn ctl(&mut self, request: i32, value: i32) {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value}) = {ret}");
    }

    /// `OPUS_SET_ENERGY_MASK` with 21 values per channel.
    fn set_mask(&mut self, mask: &[f32]) {
        let n = NB_BANDS * self.channels;
        self.mask[..n].copy_from_slice(&mask[..n]);
        let ret = unsafe {
            sys::opus_encoder_ctl(self.st, OPUS_SET_ENERGY_MASK_REQUEST, self.mask.as_ptr())
        };
        assert_eq!(ret, 0, "OPUS_SET_ENERGY_MASK = {ret}");
    }

    fn encode(&mut self, pcm: &[f32], frame_size: usize, max_bytes: usize) -> Vec<u8> {
        assert_eq!(pcm.len(), frame_size * self.channels);
        let mut out = vec![0u8; max_bytes];
        let n = unsafe {
            sys::opus_encode_float(
                self.st,
                pcm.as_ptr(),
                frame_size as i32,
                out.as_mut_ptr(),
                max_bytes as i32,
            )
        };
        assert!(n > 0, "opus_encode_float = {n}");
        out.truncate(n as usize);
        out
    }
}

impl Drop for CStream {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// The input channels of every stream of `enc`: left and right for a
/// stereo stream, one channel for a mono stream.
fn stream_channels(enc: &CMsEncoder) -> Vec<(usize, Option<usize>)> {
    let find = |v: usize| enc.mapping.iter().position(|&m| m as usize == v).unwrap();
    (0..enc.streams)
        .map(|s| {
            if s < enc.coupled {
                (find(2 * s), Some(find(2 * s + 1)))
            } else {
                (find(s + enc.coupled), None)
            }
        })
        .collect()
}

/// The samples of one stream, interleaved, from a frame of all channels.
fn stream_pcm(frame: &[f32], channels: usize, of: (usize, Option<usize>)) -> Vec<f32> {
    frame
        .chunks_exact(channels)
        .flat_map(|x| std::iter::once(x[of.0]).chain(of.1.map(|r| x[r])))
        .collect()
}

/// The mask of one stream: the 21 values of its first channel, then those
/// of its second channel.
fn stream_mask(band_smr: &[f32], of: (usize, Option<usize>)) -> Vec<f32> {
    let one = |c: usize| &band_smr[NB_BANDS * c..NB_BANDS * (c + 1)];
    let mut mask = one(of.0).to_vec();
    if let Some(right) = of.1 {
        mask.extend_from_slice(one(right));
    }
    mask
}

/// The bandwidth the C surround encoder sets on every stream
/// (opus_multistream_encoder.c 941-952).
fn surround_bandwidth(bitrate: i32, fs: i32, frame_size: usize, channels: usize) -> i32 {
    let mut equiv_rate = bitrate;
    if frame_size as i32 * 50 < fs {
        equiv_rate -= 60 * (fs / frame_size as i32 - 50) * channels as i32;
    }
    if equiv_rate > 10_000 * channels as i32 {
        sys::OPUS_BANDWIDTH_FULLBAND
    } else if equiv_rate > 7_000 * channels as i32 {
        sys::OPUS_BANDWIDTH_SUPERWIDEBAND
    } else if equiv_rate > 5_000 * channels as i32 {
        sys::OPUS_BANDWIDTH_WIDEBAND
    } else {
        sys::OPUS_BANDWIDTH_NARROWBAND
    }
}

/// How the C surround encoder is run.
#[derive(Debug, Clone, Copy)]
struct Setup {
    fs: i32,
    channels: usize,
    /// Samples per channel in one frame.
    frame_size: usize,
    /// Total bitrate in bits per second.
    bitrate: i32,
    /// `OPUS_SET_VBR_CONSTRAINT`: libopus sets it by default. The packet
    /// sizes then stay close to the bitrate. Without it they follow the
    /// signal, and with it the mask, more freely.
    constrained_vbr: bool,
}

impl std::fmt::Display for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} channels, {} Hz, {} ms, {} b/s, {} VBR",
            self.channels,
            self.fs,
            self.frame_size as f64 * 1000.0 / self.fs as f64,
            self.bitrate,
            if self.constrained_vbr {
                "constrained"
            } else {
                "unconstrained"
            }
        )
    }
}

/// What one stream wrote over a whole signal.
#[derive(Default, Debug, Clone)]
struct StreamTotals {
    /// Number of frames coded.
    frames: usize,
    /// Bytes of the stream in the packets of the C surround encoder.
    surround: usize,
    /// Bytes of a C single-stream encoder that got the opus1 mask.
    with_mask: usize,
    /// Bytes of a C single-stream encoder without a mask.
    without_mask: usize,
    /// Frames in which the encoder with the opus1 mask wrote the bytes of
    /// the surround encoder.
    same_with_mask: usize,
    /// The same count for the encoder without a mask.
    same_without_mask: usize,
    /// The same count for the encoder whose mask had `change` added.
    same_with_changed_mask: usize,
    /// Frames the surround encoder coded with CELT alone.
    celt_frames: usize,
}

/// The result of `rebuild_c_surround`.
struct Rebuilt {
    /// One entry per stream.
    streams: Vec<StreamTotals>,
    /// The bitrate the surround encoder gave every stream.
    rates: Vec<i32>,
    /// The stream of the low-frequency-effects channel. It ignores the mask.
    lfe_stream: Option<usize>,
}

impl Rebuilt {
    /// The streams that use the mask.
    fn masked_streams(&self) -> impl Iterator<Item = (usize, &StreamTotals)> {
        self.streams
            .iter()
            .enumerate()
            .filter(|(s, _)| self.lfe_stream != Some(*s))
    }
}

/// Codes `pcm` with the C surround encoder and, stream by stream, with C
/// single-stream encoders set up like its streams: one set gets the mask of
/// `mask_from`, one gets no mask, and one gets that mask plus `change` in
/// every band.
fn rebuild_c_surround(setup: Setup, pcm: &[f32], change: f32, mask_from: MaskFrom) -> Rebuilt {
    let Setup {
        fs,
        channels,
        frame_size,
        bitrate,
        constrained_vbr,
    } = setup;
    let mut ms = CMsEncoder::surround(fs, channels, 1, APP_AUDIO).unwrap();
    ms.set_bitrate(bitrate);
    ms.ctl(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, constrained_vbr as i32);
    let layout = stream_channels(&ms);
    // opus_multistream_encoder.c 498-507: the last stream of a layout with
    // 6 or more channels codes the low-frequency-effects channel.
    let lfe_stream = (channels >= 6).then(|| ms.streams - 1);
    let new_streams = || -> Vec<CStream> {
        layout
            .iter()
            .enumerate()
            .map(|(s, of)| {
                let mut enc = CStream::new(fs, if of.1.is_some() { 2 } else { 1 });
                if lfe_stream == Some(s) {
                    enc.ctl(OPUS_SET_LFE_REQUEST, 1);
                }
                enc.ctl(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, constrained_vbr as i32);
                enc
            })
            .collect()
    };
    let mut with_mask = new_streams();
    let mut without_mask = new_streams();
    let mut with_changed_mask = new_streams();
    let mut probe = SurroundProbe::new(channels, fs);
    let mut c_analysis = CAnalysis::new(channels, fs);
    let mut band_smr = vec![0f32; NB_BANDS * channels];
    let mut streams = vec![StreamTotals::default(); ms.streams];
    let mut rates = Vec::new();
    let bandwidth = surround_bandwidth(bitrate, fs, frame_size, channels);

    for frame in pcm.chunks_exact(frame_size * channels) {
        let packet = ms
            .encode_float(frame, frame_size, max_packet_bytes(ms.streams))
            .unwrap();
        let parts = split_streams(&packet, ms.streams).unwrap();
        // The bitrate the surround encoder gave each stream for this frame.
        rates = ms.stream_bitrates();
        assert!(
            probe.run(frame, frame_size, &mut band_smr),
            "{setup}: no mask"
        );
        if mask_from == MaskFrom::LibOpus {
            band_smr = c_analysis.run(frame, frame_size);
        }

        for (s, &of) in layout.iter().enumerate() {
            let x = stream_pcm(frame, channels, of);
            let mask = stream_mask(&band_smr, of);
            let changed_mask: Vec<f32> = mask.iter().map(|v| v + change).collect();
            // The room the surround encoder leaves a stream when the packet
            // buffer is large (opus_multistream_encoder.c 1016-1024).
            let max_bytes = if s + 1 == ms.streams { 7662 } else { 7660 };
            let code = |enc: &mut CStream, mask: Option<&[f32]>| -> Vec<u8> {
                // What the surround encoder sets on a stream before every
                // frame (opus_multistream_encoder.c 938-959, 1014).
                enc.ctl(sys::OPUS_SET_BITRATE_REQUEST, rates[s]);
                enc.ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth);
                if of.1.is_some() {
                    enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
                    enc.ctl(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, 2);
                }
                if let Some(mask) = mask {
                    enc.set_mask(mask);
                }
                enc.encode(&x, frame_size, max_bytes)
            };
            let masked = code(&mut with_mask[s], Some(&mask));
            let plain = code(&mut without_mask[s], None);
            let changed = code(&mut with_changed_mask[s], Some(&changed_mask));
            let t = &mut streams[s];
            t.frames += 1;
            t.surround += parts[s].packet.len();
            t.with_mask += masked.len();
            t.without_mask += plain.len();
            t.same_with_mask += (masked == parts[s].packet) as usize;
            t.same_without_mask += (plain == parts[s].packet) as usize;
            t.same_with_changed_mask += (changed == parts[s].packet) as usize;
            t.celt_frames += (toc(parts[s].toc).mode == Mode::Celt) as usize;
        }
    }
    Rebuilt {
        streams,
        rates,
        lfe_stream,
    }
}

/// Tones on a noise floor, with a different level on every channel.
fn mixed_signal(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = tones(fs, channels, 1, samples);
    let floor = noise(channels, samples, 0.05, 71);
    for (i, (x, &n)) in pcm.iter_mut().zip(&floor).enumerate() {
        let c = i % channels;
        *x = *x * (1.0 / (1 + c) as f32) + n * (1.0 / (1u32 << c) as f32);
    }
    pcm
}

/// Noise on every channel, with the front left channel 30 dB above the rest.
fn loud_front_left(channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = noise(channels, samples, 0.01, 73);
    for v in pcm.iter_mut().step_by(channels) {
        *v *= 30.0;
    }
    pcm
}

/// Tones on noise on the first and the last but one channel, and silence on
/// every other channel. In 5.1 these are front left and a rear channel.
fn two_channels_only(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = mixed_signal(fs, channels, samples);
    for (i, v) in pcm.iter_mut().enumerate() {
        let c = i % channels;
        if c != 0 && c != channels - 2 {
            *v = 0.0;
        }
    }
    pcm
}

/// One of five signals: tones on noise, a loud channel next to quiet ones,
/// noise of one level, a speech-like signal, and two channels of signal
/// among silent ones.
fn signal(which: usize, fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    match which % 5 {
        0 => mixed_signal(fs, channels, samples),
        1 => loud_front_left(channels, samples),
        2 => noise(channels, samples, 0.2, 5),
        3 => voiced(fs, channels, samples),
        _ => two_channels_only(fs, channels, samples),
    }
}

/// Rebuilds the C surround encoder for 3 to 8 channels, 8 to 48 kHz and
/// frames of 2.5 to 60 ms, and counts the stream packets that the C
/// single-stream encoders with the mask of `mask_from` write byte for byte.
fn check_every_layout_rate_and_frame_size(constrained_vbr: bool, mask_from: MaskFrom) {
    // Packets of the streams that use the mask: all of them, those that are
    // the same with the opus1 mask, and those that are the same without
    // a mask.
    let (mut total, mut same, mut same_without) = (0usize, 0usize, 0usize);
    let mut case = 0;
    for channels in 3..=8usize {
        for fs in [8_000, 12_000, 16_000, 24_000, 48_000] {
            for frame_400 in [1usize, 2, 4, 8, 16, 24] {
                let frame_size = fs as usize / 400 * frame_400;
                let frames = (60 / frame_400).max(6);
                for per_channel in [16_000, 40_000] {
                    let bitrate = per_channel * channels as i32;
                    let setup = Setup {
                        fs,
                        channels,
                        frame_size,
                        bitrate,
                        constrained_vbr,
                    };
                    let pcm = signal(case, fs, channels, frames * frame_size);
                    case += 1;
                    let rebuilt = rebuild_c_surround(setup, &pcm, 0.0, mask_from);
                    for (s, t) in rebuilt.masked_streams() {
                        total += t.frames;
                        same += t.same_with_mask;
                        same_without += t.same_without_mask;
                        if t.same_with_mask != t.frames {
                            println!(
                                "{setup}, stream {s}: {} of {} packets are the same ({} and {} bytes)",
                                t.same_with_mask, t.frames, t.with_mask, t.surround
                            );
                        }
                    }
                    // The low-frequency-effects stream ignores the mask.
                    if let Some(s) = rebuilt.lfe_stream {
                        let t = &rebuilt.streams[s];
                        assert_eq!(t.same_with_mask, t.frames, "{setup}: LFE stream");
                        assert_eq!(t.same_without_mask, t.frames, "{setup}: LFE stream");
                    }
                }
            }
        }
    }
    println!(
        "{case} cases, {total} stream packets: {same} the same with the mask of {mask_from:?}, {same_without} without a mask"
    );
    if mask_from == MaskFrom::LibOpus {
        // The control: with the mask of libopus itself every packet is the
        // same. Measured on aarch64 with libopus 1.6.1: 21080 of 21080.
        // The limit leaves room for a build of libopus
        // whose encoder computes its own mask with vector code that rounds
        // differently from the plain C code called here.
        assert!(
            same * 1000 >= total * 999,
            "only {same} of {total} stream packets are the same with the mask of libopus"
        );
        return;
    }
    // Measured on aarch64 with libopus 1.6.1: all 21080 packets are the
    // same in constrained VBR, and 21076 in unconstrained VBR. The other 4
    // are in 3 of the 360 cases, all on the speech-like signal at 12 or
    // 24 kHz, and their streams differ by at most one byte in total. One
    // packet that differs changes the state of its encoder, and with it
    // the later packets of its case, so the limit of 99.5 % leaves room
    // for a few more cases on another machine.
    // A mask that is off by 0.01 leaves about half of the packets the same
    // in unconstrained VBR.
    assert!(
        same * 1000 >= total * 995,
        "only {same} of {total} stream packets are the same"
    );
    // The mask matters: without it most packets differ.
    assert!(
        same_without * 4 < total,
        "{same_without} of {total} packets are the same without a mask"
    );
}

#[test]
fn c_encoders_with_the_opus1_mask_write_the_streams_of_the_c_surround_encoder() {
    check_every_layout_rate_and_frame_size(true, MaskFrom::OpusRs);
}

#[test]
fn c_encoders_with_the_opus1_mask_write_the_c_surround_streams_in_unconstrained_vbr() {
    check_every_layout_rate_and_frame_size(false, MaskFrom::OpusRs);
}

#[test]
fn c_encoders_with_the_libopus_mask_write_the_c_surround_streams() {
    // A control for the two tests above. The mask now comes from
    // `surround_analysis` of libopus, called with the memories the C
    // surround encoder keeps. Every stream packet is then the same. This
    // shows that the single-stream encoders are set up like the streams of
    // the surround encoder. The few packets that differ with the opus1
    // mask therefore differ because of the mask, not because of the setup.
    //
    // Unconstrained VBR is the case in which 4 packets differ with the
    // opus1 mask. In constrained VBR the control also gives 21080 of
    // 21080; it is left out to keep the test short.
    check_every_layout_rate_and_frame_size(false, MaskFrom::LibOpus);
}

#[test]
fn c_encoders_with_the_opus1_mask_write_the_c_surround_streams_for_long_frames() {
    // 80, 100 and 120 ms: the analysis takes the largest band levels of
    // four, five and six MDCTs.
    let (mut total, mut same, mut same_without) = (0usize, 0usize, 0usize);
    let mut case = 0;
    for (channels, fs) in [
        (6usize, 48_000),
        (8, 48_000),
        (6, 16_000),
        (5, 24_000),
        (4, 8_000),
    ] {
        for frame_400 in [32usize, 40, 48] {
            let frame_size = fs as usize / 400 * frame_400;
            let bitrate = 32_000 * channels as i32;
            let setup = Setup {
                fs,
                channels,
                frame_size,
                bitrate,
                constrained_vbr: case % 2 == 0,
            };
            // The speech-like signal is left out: see the top of this file.
            let pcm = signal(case % 3, fs, channels, 4 * frame_size);
            case += 1;
            let rebuilt = rebuild_c_surround(setup, &pcm, 0.0, MaskFrom::OpusRs);
            for (s, t) in rebuilt.masked_streams() {
                total += t.frames;
                same += t.same_with_mask;
                same_without += t.same_without_mask;
                assert_eq!(t.same_with_mask, t.frames, "{setup}, stream {s}");
            }
        }
    }
    println!(
        "{case} cases, {total} stream packets: {same} the same with the opus1 mask, {same_without} without a mask"
    );
    // Measured: 180 of 180, and 42 without a mask.
    assert!(
        same_without * 2 < total,
        "{same_without} of {total} packets are the same without a mask"
    );
}

#[test]
fn a_mask_that_is_off_by_a_little_does_not_reproduce_the_c_streams() {
    // How sharp the comparison above is. Unconstrained VBR, where the packet
    // size follows the mask most directly.
    let (mut total, mut same, mut same_changed) = (0usize, 0usize, 0usize);
    for (channels, fs, frame_size, bitrate) in [
        (6usize, 48_000, 960usize, 256_000),
        (6, 48_000, 960, 96_000),
        (8, 48_000, 480, 384_000),
        (6, 16_000, 320, 160_000),
        (4, 24_000, 240, 128_000),
    ] {
        let setup = Setup {
            fs,
            channels,
            frame_size,
            bitrate,
            constrained_vbr: false,
        };
        let pcm = mixed_signal(fs, channels, 40 * frame_size);
        // 0.05 in log2 units is 0.3 dB.
        let rebuilt = rebuild_c_surround(setup, &pcm, 0.05, MaskFrom::OpusRs);
        for (_, t) in rebuilt.masked_streams() {
            total += t.frames;
            same += t.same_with_mask;
            same_changed += t.same_with_changed_mask;
        }
    }
    println!(
        "{total} stream packets: {same} the same with the opus1 mask, {same_changed} with the mask plus 0.05"
    );
    // Measured: 600 of 600, and 111 of 600 with the changed mask. A band
    // where the mask is outside -2 to 0.25 is limited by the encoder, so a
    // small change there has no effect.
    assert!(same * 100 >= total * 99, "{same} of {total}");
    assert!(same_changed * 3 < total, "{same_changed} of {total}");
}

/// Bytes that opus1 stream encoders write for `pcm`, stream by stream,
/// when they are set up like the streams of the C surround encoder with the
/// stream bitrates `rates`. With `use_mask`, every encoder gets the opus1
/// mask of its channels before each frame.
///
/// Every stream is told to use CELT. libopus tells the stereo streams, and
/// chooses CELT itself for a mono stream at the bitrates used here.
fn rust_stream_bytes(setup: Setup, pcm: &[f32], rates: &[i32], use_mask: bool) -> Vec<usize> {
    let Setup {
        fs,
        channels,
        frame_size,
        bitrate,
        ..
    } = setup;
    let ms = CMsEncoder::surround(fs, channels, 1, APP_AUDIO).unwrap();
    let layout = stream_channels(&ms);
    let lfe_stream = (channels >= 6).then(|| ms.streams - 1);
    let bandwidth = match surround_bandwidth(bitrate, fs, frame_size, channels) {
        sys::OPUS_BANDWIDTH_FULLBAND => Bandwidth::Fullband,
        sys::OPUS_BANDWIDTH_SUPERWIDEBAND => Bandwidth::Superwideband,
        sys::OPUS_BANDWIDTH_WIDEBAND => Bandwidth::Wideband,
        _ => Bandwidth::Narrowband,
    };
    let mut encoders: Vec<OpusEncoder> = layout
        .iter()
        .enumerate()
        .map(|(s, of)| {
            let stream_channels = if of.1.is_some() { 2 } else { 1 };
            let mut enc = OpusEncoder::new(
                SampleRate::try_from(fs).unwrap(),
                Channels::try_from(stream_channels).unwrap(),
                Application::Audio,
            )
            .unwrap();
            enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(rates[s] as u32).unwrap()));
            enc.set_vbr(true);
            enc.set_bandwidth(Some(bandwidth));
            enc.set_force_mode(Some(ForcedMode::CeltOnly));
            enc.set_lfe(lfe_stream == Some(s));
            enc
        })
        .collect();
    let mut probe = SurroundProbe::new(channels, fs);
    let mut band_smr = vec![0f32; NB_BANDS * channels];
    let mut bytes = vec![0usize; layout.len()];
    let mut out = vec![0u8; 7660];
    for frame in pcm.chunks_exact(frame_size * channels) {
        assert!(probe.run(frame, frame_size, &mut band_smr));
        for (s, &of) in layout.iter().enumerate() {
            let x = stream_pcm(frame, channels, of);
            if use_mask {
                encoders[s].set_energy_mask(Some(&stream_mask(&band_smr, of))).unwrap();
            }
            bytes[s] += encoders[s].encode(&x, frame_size, &mut out).unwrap();
        }
    }
    bytes
}

/// `a` against `b` in percent: +1.0 when `a` is 1 % larger.
fn percent(a: usize, b: usize) -> f64 {
    (a as f64 / b as f64 - 1.0) * 100.0
}

#[test]
fn opus1_stream_sizes_follow_the_mask_as_in_libopus() {
    // The mask of this package goes to opus1 stream encoders through
    // `OpusEncoder::set_energy_mask`, and the stream sizes are compared with
    // libopus. For libopus, "with the mask" is the C surround encoder, and
    // "without" is C single-stream encoders set up like its streams.
    //
    // Under the default constrained VBR a CELT stream writes between its
    // bitrate and about 2.4 % more. The mask decides where in that range a
    // stream ends up: a channel that stands out goes to the top, a masked
    // one stays at the bitrate.
    let frames = 50;
    let mut failures = Vec::new();
    // Streams compared; streams whose size the mask changed by more than
    // 1 % in libopus; and of those, the streams that both encoders moved
    // from the same size by the same amount.
    let (mut compared, mut moved, mut moved_alike) = (0, 0, 0);
    for (channels, bitrate) in [(6usize, 256_000), (6, 384_000), (8, 384_000), (8, 512_000)] {
        let setup = Setup {
            fs: 48_000,
            channels,
            frame_size: 960,
            bitrate,
            constrained_vbr: true,
        };
        for (which, name) in ["tones on noise", "loud front left", "noise", "speech-like"]
            .into_iter()
            .enumerate()
        {
            let pcm = signal(which, setup.fs, channels, frames * setup.frame_size);
            let c = rebuild_c_surround(setup, &pcm, 0.0, MaskFrom::OpusRs);
            let with = rust_stream_bytes(setup, &pcm, &c.rates, true);
            let without = rust_stream_bytes(setup, &pcm, &c.rates, false);
            println!("{setup}, {name}:");
            for (s, t) in c.masked_streams() {
                let c_change = percent(t.surround, t.without_mask);
                let rs_change = percent(with[s], without[s]);
                println!(
                    "  stream {s}: libopus {} -> {} bytes ({c_change:+.2} %), opus1 {} -> {} bytes ({rs_change:+.2} %)",
                    t.without_mask, t.surround, without[s], with[s]
                );
                assert_eq!(
                    t.celt_frames, t.frames,
                    "{setup}, {name}, stream {s}: libopus did not use CELT"
                );
                compared += 1;
                let mut fail =
                    |what: String| failures.push(format!("{setup}, {name}, stream {s}: {what}"));
                // With the mask, an opus1 stream has the size of the
                // libopus stream. Measured: at most 0.8 % apart.
                let off = percent(with[s], t.surround);
                if off.abs() >= 1.0 {
                    fail(format!("opus1 is {off:+.2} % off libopus with the mask"));
                }
                if c_change.abs() > 1.0 {
                    moved += 1;
                    // Where the mask moves the libopus stream, it does not
                    // move the opus1 stream the other way. The two can
                    // start from different sizes: without a mask opus1
                    // does not follow libopus as closely as with one.
                    if rs_change * c_change.signum() < -0.5 {
                        fail(format!(
                            "the mask changes libopus by {c_change:+.2} % and opus1 by {rs_change:+.2} %"
                        ));
                    }
                    // Same size without a mask, same change with it.
                    if percent(without[s], t.without_mask).abs() < 0.25
                        && (rs_change - c_change).abs() < 0.25
                    {
                        moved_alike += 1;
                    }
                }
            }
        }
    }
    println!(
        "{compared} streams compared; the mask moved {moved} of them by more than 1 % in libopus, and {moved_alike} of these by the same amount from the same size in opus1"
    );
    assert!(
        failures.is_empty(),
        "{} of {compared} streams:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The mask must have moved something. Measured: 37 of 56 streams.
    assert!(
        moved * 2 > compared,
        "the mask moved only {moved} of {compared} streams"
    );
    // Measured: 20 streams.
    assert!(moved_alike >= 12, "only {moved_alike} streams moved alike");
}
