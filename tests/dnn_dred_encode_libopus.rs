#![cfg(all(feature = "test-libopus-dnn", feature = "dred"))]
//! The DRED encoder against libopus 1.6.1 built with DRED, without loop
//! vectorization (`libopus-host-1.6.1-dred-novec`, a float build).
//!
//! - CELT-only streams at 48 kHz, mono and stereo, 2.5 to 60 ms, VBR and
//!   CBR, with and without FEC, at several bitrates, losses and DRED
//!   durations: every packet is libopus's, byte for byte, DRED data and
//!   padding included.
//! - Hybrid streams at 48 kHz: the DRED data equals libopus's where its
//!   inputs are libopus's (the `Audio` application with VBR). The SILK
//!   layer is opus1's fixed-point one, so the rest of the packet differs.
//!   With `Voip` the input filter follows SILK's pitch analysis, and with
//!   CBR the room left for DRED follows the size of the main frame, so
//!   there the DRED data may differ; the tests print how often.
//! - The multistream encoder with DRED set on its stream encoders.
//! - libopus's DRED decoder reads opus1's DRED data as opus1's does, and
//!   the audio both recover from it over bursts of lost packets is the
//!   same and closer to the input than plain packet loss concealment.
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dred-novec LIBOPUS_NO_PKG=1 \
//!   LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dred-ref cargo test --release \
//!   --features test-libopus-dnn,dred,osce-bwe --test dnn_dred_encode_libopus
//! ```

#[path = "dnn_plc_common/mod.rs"]
mod common;

use audiopus_sys as sys;
use opus1::internals::dnn::dred::DRED_EXTENSION_ID;
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusDecoder, OpusDred, OpusDredDecoder, OpusEncoder, packet};
use std::ffi::CStr;
use std::os::raw::{c_int, c_uchar, c_void};

/// libopus `OpusDRED` (dnn/dred_decoder.h).
#[repr(C)]
struct CDred {
    fec_features: [f32; 2 * 52 * 20],
    state: [f32; 50],
    latents: [f32; 26 * 26],
    nb_latents: c_int,
    process_stage: c_int,
    dred_offset: c_int,
}

unsafe extern "C" {
    fn opus_dred_decoder_create(error: *mut c_int) -> *mut c_void;
    fn opus_dred_decoder_destroy(dec: *mut c_void);
    fn opus_dred_alloc(error: *mut c_int) -> *mut CDred;
    fn opus_dred_free(dred: *mut CDred);
    fn opus_dred_parse(
        dred_dec: *mut c_void,
        dred: *mut CDred,
        data: *const c_uchar,
        len: i32,
        max_dred_samples: i32,
        sampling_rate: i32,
        dred_end: *mut c_int,
        defer_processing: c_int,
    ) -> c_int;
    fn opus_decoder_dred_decode_float(
        st: *mut sys::OpusDecoder,
        dred: *const CDred,
        dred_offset: i32,
        pcm: *mut f32,
        frame_size: i32,
    ) -> c_int;
}

const OPUS_SET_BITRATE: c_int = 4002;
const OPUS_SET_MAX_BANDWIDTH: c_int = 4004;
const OPUS_SET_VBR: c_int = 4006;
const OPUS_SET_COMPLEXITY: c_int = 4010;
const OPUS_SET_INBAND_FEC: c_int = 4012;
const OPUS_SET_PACKET_LOSS_PERC: c_int = 4014;
const OPUS_RESET_STATE: c_int = 4028;
const OPUS_SET_DRED_DURATION: c_int = 4050;
const OPUS_GET_DRED_DURATION: c_int = 4051;
const OPUS_SET_FORCE_MODE: c_int = 11002;
const OPUS_MULTISTREAM_GET_ENCODER_STATE: c_int = 5120;
const OPUS_BANDWIDTH_WIDEBAND: c_int = 1103;
const MODE_SILK_ONLY: c_int = 1000;
const MODE_HYBRID: c_int = 1001;
const MODE_CELT_ONLY: c_int = 1002;

fn linked() {
    let version = unsafe { CStr::from_ptr(sys::opus_get_version_string()) };
    assert!(version.to_string_lossy().contains("1.6.1"), "{version:?}");
}

#[derive(Clone, Copy, Debug)]
struct Cfg {
    fs: i32,
    channels: usize,
    application: Application,
    bitrate: i32,
    vbr: bool,
    complexity: i32,
    loss: i32,
    fec: bool,
    dred: i32,
    /// Samples per channel of each packet.
    frame: usize,
    mode: ForcedMode,
}

const FS: i32 = 48000;

/// Input at `fs`: the speech fixture (16 kHz) brought to `fs` by linear
/// interpolation, with 1.2 s of digital silence and 1.2 s of quiet noise
/// inside, so that DRED's handling of inactive frames runs. The second
/// channel is the first one delayed by 37 samples, at 0.7 times the level,
/// with some noise.
fn input(fs: i32, channels: usize, seconds: usize) -> Vec<i16> {
    let speech = common::fixture();
    let fs = fs as usize;
    let n = seconds * fs;
    let mut mono = Vec::with_capacity(n);
    let mut seed = 7u32;
    for i in 0..n {
        let t = i % (17 * fs);
        let v = if (3 * fs..(42 * fs) / 10).contains(&t) {
            0.0
        } else if ((9 * fs)..(102 * fs) / 10).contains(&t) {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) % 61) as f64 - 30.0
        } else {
            let pos = t as f64 * 16000.0 / fs as f64;
            let k = pos as usize % (speech.len() - 1);
            let f = pos.fract();
            speech[k] as f64 * (1.0 - f) + speech[k + 1] as f64 * f
        };
        mono.push(v);
    }
    let mut out = Vec::with_capacity(n * channels);
    for i in 0..n {
        out.push(mono[i] as i16);
        if channels == 2 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let d = if i >= 37 { mono[i - 37] } else { 0.0 };
            out.push((0.7 * d + ((seed >> 16) % 201) as f64 - 100.0) as i16);
        }
    }
    out
}

/// libopus's encoder with the settings of `cfg`.
struct CEnc(*mut sys::OpusEncoder);

impl CEnc {
    fn new(cfg: &Cfg) -> Self {
        linked();
        let mut err = 0;
        let app = cfg.application as c_int;
        let enc = unsafe { sys::opus_encoder_create(cfg.fs, cfg.channels as c_int, app, &mut err) };
        assert_eq!(err, 0);
        let mode = match cfg.mode {
            ForcedMode::SilkOnly => MODE_SILK_ONLY,
            ForcedMode::Hybrid => MODE_HYBRID,
            ForcedMode::CeltOnly => MODE_CELT_ONLY,
        };
        unsafe {
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_BITRATE, cfg.bitrate), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_VBR, cfg.vbr as c_int), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY, cfg.complexity), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC, cfg.fec as c_int), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC, cfg.loss), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION, cfg.dred), 0, "a libopus without DRED");
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE, mode), 0);
            if cfg.mode == ForcedMode::SilkOnly {
                assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_MAX_BANDWIDTH, OPUS_BANDWIDTH_WIDEBAND), 0);
            }
        }
        CEnc(enc)
    }

    fn encode(&mut self, pcm: &[i16], frame: usize) -> Vec<u8> {
        let mut buf = vec![0u8; 1500];
        let n = unsafe { sys::opus_encode(self.0, pcm.as_ptr(), frame as c_int, buf.as_mut_ptr(), buf.len() as i32) };
        assert!(n > 0, "libopus: {n}");
        buf.truncate(n as usize);
        buf
    }
}

impl Drop for CEnc {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.0) };
    }
}

fn opus1_encoder(cfg: &Cfg) -> OpusEncoder {
    let mut enc = OpusEncoder::new(cfg.fs, cfg.channels, cfg.application).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(cfg.bitrate)).unwrap();
    enc.set_vbr(cfg.vbr);
    enc.set_complexity(cfg.complexity).unwrap();
    enc.set_inband_fec(cfg.fec);
    enc.set_packet_loss_perc(cfg.loss).unwrap();
    enc.set_dred_duration(cfg.dred).unwrap();
    enc.set_force_mode(Some(cfg.mode));
    if cfg.mode == ForcedMode::SilkOnly {
        // Above wideband libopus codes Hybrid packets even when SILK-only
        // is forced.
        enc.set_max_bandwidth(Bandwidth::Wideband).unwrap();
    }
    enc
}

fn encode_opus1(enc: &mut OpusEncoder, pcm: &[i16], frame: usize) -> Vec<u8> {
    let mut buf = vec![0u8; 1500];
    let n = enc.encode_i16(pcm, frame, &mut buf).unwrap();
    buf.truncate(n);
    buf
}

/// The DRED extension of a packet: its frame and its data (with the two
/// experimental bytes).
fn dred_of(data: &[u8]) -> Option<(usize, Vec<u8>)> {
    let parsed = packet::parse(data, false).unwrap();
    let mut it = opus1::extensions::ExtensionIterator::new(parsed.padding(data), parsed.count);
    let ext = it.find(DRED_EXTENSION_ID).unwrap()?;
    Some((ext.frame, ext.data.to_vec()))
}

/// What a stream gave: packets, packets equal to libopus's, packets with
/// DRED data in each encoder, and packets whose DRED data is equal.
#[derive(Default, Debug)]
struct Counts {
    packets: usize,
    same_packets: usize,
    dred_c: usize,
    dred_r: usize,
    same_dred: usize,
    dred_bytes: usize,
}

/// Encodes `seconds` of input with both encoders and counts.
fn compare(cfg: &Cfg, seconds: usize) -> Counts {
    let pcm = input(cfg.fs, cfg.channels, seconds);
    let mut c = CEnc::new(cfg);
    let mut r = opus1_encoder(cfg);
    let mut counts = Counts::default();
    for (i, frame) in pcm.chunks_exact(cfg.frame * cfg.channels).enumerate() {
        let a = c.encode(frame, cfg.frame);
        let b = encode_opus1(&mut r, frame, cfg.frame);
        let (da, db) = (dred_of(&a), dred_of(&b));
        counts.packets += 1;
        counts.same_packets += (a == b) as usize;
        counts.dred_c += da.is_some() as usize;
        counts.dred_r += db.is_some() as usize;
        counts.same_dred += (da == db) as usize;
        counts.dred_bytes += db.as_ref().map_or(0, |d| d.1.len());
        if i == 0 {
            assert_eq!(r.dred_duration(), cfg.dred);
        }
    }
    counts
}

/// Every packet equals libopus's, and DRED data was written.
fn assert_same_packets(cfg: &Cfg, seconds: usize) -> Counts {
    let n = compare(cfg, seconds);
    assert_eq!(n.same_packets, n.packets, "{cfg:?}: {n:?}");
    n
}

#[test]
fn celt_only_packets_with_dred_equal_libopus() {
    let mut total = Counts::default();
    let mut k = 0usize;
    for &channels in &[1usize, 2] {
        for &frame in &[120usize, 240, 480, 960] {
            for &vbr in &[true, false] {
                for &(bitrate, loss, fec, dred) in &[
                    (24000, 20, false, 104),
                    (32000, 10, true, 30),
                    (64000, 30, false, 60),
                    (128000, 50, true, 104),
                    (48000, 5, false, 10),
                ] {
                    k += 1;
                    let complexity = [0, 5, 6, 3][k % 4];
                    let cfg = Cfg {
                        fs: FS,
                        channels,
                        application: if k.is_multiple_of(3) { Application::Voip } else { Application::Audio },
                        bitrate: bitrate * channels as i32 / 2 + bitrate / 2,
                        vbr,
                        complexity,
                        loss,
                        fec,
                        dred,
                        frame,
                        mode: ForcedMode::CeltOnly,
                    };
                    let n = assert_same_packets(&cfg, 4);
                    total.packets += n.packets;
                    total.same_packets += n.same_packets;
                    total.dred_c += n.dred_c;
                    total.dred_r += n.dred_r;
                    total.same_dred += n.same_dred;
                    total.dred_bytes += n.dred_bytes;
                }
            }
        }
    }
    println!("{total:?}");
    // Packets of 2.5 and 5 ms at low bitrates have no room for DRED.
    assert!(total.dred_r > total.packets / 4, "{total:?}");
}

/// With the speech and music analysis (complexity 7 to 10), CELT-only
/// packets and their DRED data. The analysis of opus1 follows the Homebrew
/// build of libopus, whose roundings differ from the build without vector
/// loops in the last bits of some probabilities
/// (docs/10-differences-from-libopus.md), so a decision can differ now and
/// then; the test allows 1 % of the packets to differ.
#[test]
fn celt_only_packets_with_dred_and_the_analysis_equal_libopus() {
    let mut total = Counts::default();
    for (k, &(channels, frame, vbr, bitrate, loss, dred)) in [
        (1usize, 960usize, true, 32000, 20, 104),
        (2, 960, true, 96000, 30, 50),
        (1, 480, false, 48000, 10, 30),
        (2, 480, true, 64000, 50, 104),
        (1, 240, true, 40000, 15, 20),
    ]
    .iter()
    .enumerate()
    {
        let cfg = Cfg {
            fs: FS,
            channels,
            application: Application::Audio,
            bitrate,
            vbr,
            complexity: 7 + (k as i32 % 4),
            loss,
            fec: false,
            dred,
            frame,
            mode: ForcedMode::CeltOnly,
        };
        let n = compare(&cfg, 6);
        println!("{cfg:?}: {n:?}");
        total.packets += n.packets;
        total.same_packets += n.same_packets;
        total.same_dred += n.same_dred;
        total.dred_c += n.dred_c;
        total.dred_r += n.dred_r;
    }
    println!("{total:?}");
    assert!(total.same_packets * 100 >= total.packets * 99, "{total:?}");
    assert!(total.dred_r > total.packets / 2, "{total:?}");
}

/// CELT-only packets of 40 and 60 ms (two and three frames, joined by the
/// repacketizer, the DRED data with the first) with CBR. With VBR opus1
/// leaves out libopus's limit of each frame to its share of the bitrate on
/// purpose, so those packets differ (`encode_multiframe`).
#[test]
fn celt_only_packets_of_several_frames_with_dred_equal_libopus() {
    for (k, &(channels, frame, bitrate, loss, dred)) in [
        (1usize, 1920usize, 32000, 20, 104),
        (2, 2880, 64000, 30, 60),
        (1, 2880, 24000, 50, 30),
        (2, 1920, 96000, 10, 104),
    ]
    .iter()
    .enumerate()
    {
        let cfg = Cfg {
            fs: FS,
            channels,
            application: Application::Audio,
            bitrate,
            vbr: false,
            complexity: [5, 0, 3, 6][k],
            loss,
            fec: k % 2 == 1,
            dred,
            frame,
            mode: ForcedMode::CeltOnly,
        };
        let n = assert_same_packets(&cfg, 6);
        println!("{cfg:?}: {n:?}");
        assert!(n.dred_r >= 10, "{cfg:?}: {n:?}");
    }
}

/// CELT-only packets at 8, 12, 16 and 24 kHz, where DRED brings the input
/// to 16 kHz with each of its other filters.
#[test]
fn celt_only_packets_with_dred_at_other_rates_equal_libopus() {
    for (k, &(fs, channels, vbr, bitrate)) in [
        (8000, 1usize, true, 24000),
        (12000, 2, false, 32000),
        (16000, 1, false, 40000),
        (16000, 2, true, 48000),
        (24000, 1, true, 32000),
        (24000, 2, false, 64000),
        (12000, 1, true, 20000),
        (8000, 2, false, 40000),
    ]
    .iter()
    .enumerate()
    {
        let cfg = Cfg {
            fs,
            channels,
            application: if k % 2 == 0 { Application::Audio } else { Application::Voip },
            bitrate,
            vbr,
            complexity: [5, 0, 6, 3][k % 4],
            loss: [20, 30, 50, 10][k % 4] + 20 * (k / 4) as i32,
            fec: k % 3 == 1,
            dred: [104, 60, 30, 104][k % 4],
            frame: fs as usize / 50,
            mode: ForcedMode::CeltOnly,
        };
        let n = assert_same_packets(&cfg, 6);
        println!("{cfg:?}: {n:?}");
        assert!(n.dred_r >= 50, "{cfg:?}: {n:?}");
    }
}

/// SILK-only packets of 10 to 60 ms (wideband at most: libopus codes
/// Hybrid packets above). The DRED data equals libopus's with `Audio` and
/// VBR, as in Hybrid packets. In frames of 40 and 60 ms DRED
/// analyzes 20 ms at a time, and libopus moves on by the number of samples
/// of one channel; a stereo input is read from the wrong place for the
/// second part, in opus1 as in libopus (`DredEnc::compute_latents`).
#[test]
fn silk_only_dred_data_equals_libopus_with_audio_and_vbr() {
    for (k, &(fs, channels, frame_ms, bitrate, fec)) in [
        (16000, 1usize, 20usize, 20000, false),
        (16000, 2, 40, 32000, true),
        (48000, 2, 60, 24000, false),
        (24000, 1, 60, 16000, false),
        (12000, 2, 10, 32000, false),
        (8000, 1, 40, 16000, false),
    ]
    .iter()
    .enumerate()
    {
        let cfg = Cfg {
            fs,
            channels,
            application: Application::Audio,
            bitrate,
            vbr: true,
            complexity: [5, 0, 3, 6][k % 4],
            loss: [30, 50, 20, 40][k % 4],
            fec,
            dred: 104,
            frame: fs as usize * frame_ms / 1000,
            mode: ForcedMode::SilkOnly,
        };
        let n = compare(&cfg, 6);
        println!("{cfg:?}: {n:?}");
        assert_eq!(n.same_dred, n.packets, "{cfg:?}: {n:?}");
        assert!(n.dred_r > n.packets / 2, "{cfg:?}: {n:?}");
    }
}

/// Hybrid packets at 48 kHz. With `Audio` and VBR the DRED data is
/// libopus's: DRED reads the frame after the DC filter, which does not
/// depend on SILK, and the activity of SILK's voice activity detector,
/// whose input and arithmetic are those of the float build. The SILK layer
/// itself is opus1's fixed-point one, so the main frames differ.
#[test]
fn hybrid_dred_data_equals_libopus_with_audio_and_vbr() {
    for (k, &(channels, frame, bitrate, loss, fec, dred)) in [
        (1usize, 960usize, 24000, 20, false, 104),
        (2, 960, 32000, 30, true, 50),
        (1, 480, 28000, 30, false, 30),
        (1, 960, 32000, 50, true, 104),
        (2, 480, 40000, 15, false, 70),
    ]
    .iter()
    .enumerate()
    {
        let cfg = Cfg {
            fs: FS,
            channels,
            application: Application::Audio,
            bitrate,
            vbr: true,
            complexity: [5, 0, 3, 6, 1][k],
            loss,
            fec,
            dred,
            frame,
            mode: ForcedMode::Hybrid,
        };
        let n = compare(&cfg, 6);
        println!("{cfg:?}: {n:?}");
        assert_eq!(n.same_dred, n.packets, "{cfg:?}: {n:?}");
        assert!(n.dred_r > n.packets / 2, "{cfg:?}: {n:?}");
    }
}

/// Packets of 40 and 60 ms made of 20 ms frames (Hybrid and CELT-only)
/// with VBR. libopus limits each frame to its share of the bitrate, plus
/// the DRED bytes for the first frame; opus1 leaves out the share on
/// purpose (`encode_multiframe`). So the first frame has more room for
/// DRED in opus1, and libopus can cut its DRED data shorter. Prints how
/// often the DRED data is equal, and checks that opus1 writes DRED data
/// at least as often as libopus.
#[test]
fn multiframe_vbr_dred_data_report() {
    for &(mode, frame) in &[
        (ForcedMode::Hybrid, 1920usize),
        (ForcedMode::Hybrid, 2880),
        (ForcedMode::CeltOnly, 1920),
        (ForcedMode::CeltOnly, 2880),
    ] {
        let cfg = Cfg {
            fs: FS,
            channels: 1,
            application: Application::Audio,
            bitrate: 24000,
            vbr: true,
            complexity: 5,
            loss: 20,
            fec: false,
            dred: 104,
            frame,
            mode,
        };
        let n = compare(&cfg, 10);
        println!("{mode:?} {frame}: {n:?}");
        assert!(n.dred_r >= n.dred_c && n.dred_r > n.packets / 2, "{cfg:?}: {n:?}");
    }
}

/// Hybrid packets where the DRED data cannot always be libopus's: with
/// `Voip` the high-pass filter that DRED reads after follows the pitch
/// analysis of SILK; with CBR the room left for DRED is what the main
/// frame leaves; with the analysis (complexity 7 and above) see
/// `celt_only_packets_with_dred_and_the_analysis_equal_libopus`. Prints
/// how much is equal, and checks that both write DRED data about as often.
#[test]
fn hybrid_dred_data_report() {
    for &(application, vbr, complexity) in &[
        (Application::Voip, true, 5),
        (Application::Audio, false, 5),
        (Application::Voip, false, 9),
        (Application::Audio, true, 9),
    ] {
        let cfg = Cfg {
            fs: FS,
            channels: 1,
            application,
            bitrate: 32000,
            vbr,
            complexity,
            loss: 25,
            fec: false,
            dred: 104,
            frame: 960,
            mode: ForcedMode::Hybrid,
        };
        let n = compare(&cfg, 10);
        println!("{application:?} vbr {vbr} complexity {complexity}: {n:?}");
        assert!(n.dred_r.abs_diff(n.dred_c) * 20 <= n.packets, "{cfg:?}: {n:?}");
    }
}

/// `OPUS_SET_DRED_DURATION` and `OPUS_GET_DRED_DURATION` take the same
/// values as libopus's, and `OPUS_RESET_STATE` turns DRED off in both.
#[test]
fn dred_duration_control_and_reset_as_libopus() {
    let cfg = Cfg {
        fs: FS,
        channels: 1,
        application: Application::Audio,
        bitrate: 32000,
        vbr: true,
        complexity: 5,
        loss: 20,
        fec: false,
        dred: 104,
        frame: 960,
        mode: ForcedMode::CeltOnly,
    };
    let c = CEnc::new(&cfg);
    let mut r = opus1_encoder(&cfg);
    for v in [-1, 0, 1, 50, 104, 105, 1000] {
        let cr = unsafe { sys::opus_encoder_ctl(c.0, OPUS_SET_DRED_DURATION, v) };
        assert_eq!(cr == 0, r.set_dred_duration(v).is_ok(), "{v}");
        let mut got = -1;
        unsafe { sys::opus_encoder_ctl(c.0, OPUS_GET_DRED_DURATION, &mut got as *mut c_int) };
        assert_eq!(got, r.dred_duration(), "{v}");
    }
    unsafe { sys::opus_encoder_ctl(c.0, OPUS_RESET_STATE) };
    r.reset_state();
    let mut got = -1;
    unsafe { sys::opus_encoder_ctl(c.0, OPUS_GET_DRED_DURATION, &mut got as *mut c_int) };
    assert_eq!((got, r.dred_duration()), (0, 0));

    // After a reset with DRED on again, the packets are libopus's.
    let mut c = CEnc::new(&cfg);
    let mut r = opus1_encoder(&cfg);
    let pcm = input(FS, 1, 4);
    for (i, frame) in pcm.as_chunks::<960>().0.iter().enumerate() {
        if i == 100 {
            unsafe {
                sys::opus_encoder_ctl(c.0, OPUS_RESET_STATE);
                sys::opus_encoder_ctl(c.0, OPUS_SET_DRED_DURATION, 60);
            }
            r.reset_state();
            r.set_dred_duration(60).unwrap();
        }
        if i == 150 {
            unsafe { sys::opus_encoder_ctl(c.0, OPUS_SET_DRED_DURATION, 0) };
            r.set_dred_duration(0).unwrap();
        }
        if i == 160 {
            unsafe { sys::opus_encoder_ctl(c.0, OPUS_SET_DRED_DURATION, 20) };
            r.set_dred_duration(20).unwrap();
        }
        assert_eq!(c.encode(frame, 960), encode_opus1(&mut r, frame, 960), "packet {i}");
    }
}

/// The multistream encoder of libopus does not pass on
/// `OPUS_SET_DRED_DURATION`; it is set on each stream encoder
/// (`OPUS_MULTISTREAM_GET_ENCODER_STATE`, `stream_encoder_mut`). Each
/// stream then carries its DRED data, and the packets are libopus's.
#[test]
fn multistream_packets_with_dred_on_the_streams_equal_libopus() {
    linked();
    let (channels, streams, coupled) = (3usize, 2usize, 1usize);
    let mapping = [0u8, 1, 2];
    let mut err = 0;
    let c = unsafe {
        sys::opus_multistream_encoder_create(
            FS,
            channels as c_int,
            streams as c_int,
            coupled as c_int,
            mapping.as_ptr(),
            Application::Audio as c_int,
            &mut err,
        )
    };
    assert_eq!(err, 0);
    let mut r = MultistreamEncoder::new(FS, channels, streams, coupled, &mapping, Application::Audio).unwrap();
    unsafe {
        assert_eq!(sys::opus_multistream_encoder_ctl(c, OPUS_SET_BITRATE, 96000), 0);
        assert_eq!(sys::opus_multistream_encoder_ctl(c, OPUS_SET_COMPLEXITY, 5), 0);
        assert_eq!(sys::opus_multistream_encoder_ctl(c, OPUS_SET_PACKET_LOSS_PERC, 20), 0);
        // Not passed on to the streams, as in opus1, which has no such
        // setting on the multistream encoder.
        assert_ne!(sys::opus_multistream_encoder_ctl(c, OPUS_SET_DRED_DURATION, 50), 0);
    }
    r.set_bitrate(Bitrate::BitsPerSecond(96000)).unwrap();
    r.set_complexity(5).unwrap();
    r.set_packet_loss_perc(20).unwrap();
    for s in 0..streams {
        let mut enc: *mut sys::OpusEncoder = std::ptr::null_mut();
        unsafe {
            let state = &mut enc as *mut *mut sys::OpusEncoder;
            assert_eq!(sys::opus_multistream_encoder_ctl(c, OPUS_MULTISTREAM_GET_ENCODER_STATE, s as c_int, state), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION, 50), 0);
            assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_FORCE_MODE, MODE_CELT_ONLY), 0);
        }
        let e = r.stream_encoder_mut(s).unwrap();
        e.set_dred_duration(50).unwrap();
        e.set_force_mode(Some(ForcedMode::CeltOnly));
    }
    let mono = input(FS, 1, 4);
    let pcm: Vec<i16> = mono.iter().flat_map(|&v| [v, v / 2, v / 3]).collect();
    let mut with_dred = [0usize; 2];
    for (i, frame) in pcm.chunks_exact(960 * channels).enumerate() {
        let mut a = vec![0u8; 4000];
        let mut b = vec![0u8; 4000];
        let na = unsafe { sys::opus_multistream_encode(c, frame.as_ptr(), 960, a.as_mut_ptr(), a.len() as i32) };
        let nb = r.encode_i16(frame, 960, &mut b).unwrap();
        assert!(na > 0);
        assert_eq!(&a[..na as usize], &b[..nb], "packet {i}");
        // Every stream but the last is self-delimited.
        let mut rest = &b[..nb];
        for (s, count) in with_dred.iter_mut().enumerate() {
            let p = packet::parse(rest, s + 1 < streams).unwrap();
            let mut it = opus1::extensions::ExtensionIterator::new(p.padding(rest), p.count);
            *count += it.find(DRED_EXTENSION_ID).unwrap().is_some() as usize;
            rest = &rest[p.packet_offset..];
        }
    }
    unsafe { sys::opus_multistream_encoder_destroy(c) };
    println!("packets with DRED data per stream: {with_dred:?} of 200");
    assert!(with_dred.iter().all(|&n| n > 100), "{with_dred:?}");
}

/// libopus's DRED decoder and DRED decoding, freed on drop.
struct CDredDec {
    dec: *mut c_void,
    dred: *mut CDred,
    out: *mut sys::OpusDecoder,
}

impl CDredDec {
    fn new(channels: usize) -> Self {
        let mut err = 0;
        let dec = unsafe { opus_dred_decoder_create(&mut err) };
        assert!(err == 0 && !dec.is_null());
        let dred = unsafe { opus_dred_alloc(&mut err) };
        assert!(err == 0 && !dred.is_null());
        let out = unsafe { sys::opus_decoder_create(FS, channels as c_int, &mut err) };
        assert_eq!(err, 0);
        CDredDec { dec, dred, out }
    }
}

impl Drop for CDredDec {
    fn drop(&mut self) {
        unsafe {
            opus_dred_free(self.dred);
            opus_dred_decoder_destroy(self.dec);
            sys::opus_decoder_destroy(self.out);
        }
    }
}

/// The bursts of lost packets, each followed by `GAP` received packets.
const BURSTS: [usize; 8] = [1, 3, 6, 10, 15, 25, 35, 49];
const GAP: usize = 6;
const LEAD: usize = 20;

/// The energies in dB of the 10 ms blocks of the first channel.
fn block_db(pcm: &[f32], channels: usize) -> Vec<f64> {
    pcm.chunks(480 * channels)
        .map(|b| {
            let e: f64 = b.iter().step_by(channels).map(|&x| (x as f64) * (x as f64)).sum::<f64>() / 480.0;
            10.0 * (e + 1e-9).log10()
        })
        .collect()
}

/// Decodes the packets of `encoder` (libopus's or opus1's) with bursts of
/// losses, concealing each lost packet with the DRED data of the packet
/// after the burst, as `opus_demo` does, with libopus's decoder and with
/// opus1's, which must give the same audio. Also decodes with plain packet
/// loss concealment. Returns the mean distance in dB of the 10 ms block
/// energies from those of the lossless decode, over the lost packets,
/// with DRED and with plain concealment.
fn recover(cfg: &Cfg, opus1_packets: bool) -> (f64, f64) {
    let total = LEAD + BURSTS.iter().map(|b| b + GAP).sum::<usize>();
    let ch = cfg.channels;
    let pcm = input(FS, ch, (total * cfg.frame).div_ceil(FS as usize) + 1);
    let packets: Vec<Vec<u8>> = if opus1_packets {
        let mut r = opus1_encoder(cfg);
        pcm.chunks_exact(cfg.frame * ch).take(total).map(|f| encode_opus1(&mut r, f, cfg.frame)).collect()
    } else {
        let mut c = CEnc::new(cfg);
        pcm.chunks_exact(cfg.frame * ch).take(total).map(|f| c.encode(f, cfg.frame)).collect()
    };
    let frame = cfg.frame;
    let mut lost = vec![false; total];
    let mut at = LEAD;
    for &b in &BURSTS {
        lost[at..at + b].iter_mut().for_each(|l| *l = true);
        at += b + GAP;
    }
    let c = CDredDec::new(ch);
    let mut r = OpusDecoder::new(FS, ch).unwrap();
    let rd = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut plc = OpusDecoder::new(FS, ch).unwrap();
    let mut clean = OpusDecoder::new(FS, ch).unwrap();
    let (mut out_dred, mut out_plc, mut out_clean) = (Vec::new(), Vec::new(), Vec::new());
    let mut a = vec![0f32; frame * ch];
    let mut b = vec![0f32; frame * ch];
    let mut lost_count = 0;
    for (i, p) in packets.iter().enumerate() {
        clean.decode(p, frame, &mut b).unwrap();
        out_clean.extend_from_slice(&b);
        if lost[i] {
            plc.decode(&[], frame, &mut b).unwrap();
            out_plc.extend_from_slice(&b);
            lost_count += 1;
            continue;
        }
        plc.decode(p, frame, &mut b).unwrap();
        out_plc.extend_from_slice(&b);
        if lost_count > 0 {
            let max = FS.min((lost_count * frame) as i32);
            let mut end = 0;
            let ret = unsafe { opus_dred_parse(c.dec, c.dred, p.as_ptr(), p.len() as i32, max, FS, &mut end, 0) };
            let got = rd.parse(&mut dred, p, max, FS, false).unwrap();
            assert_eq!((got.samples, got.end), (ret, end), "packet {i}");
            for fr in 0..lost_count {
                if ret > 0 {
                    let offset = ((lost_count - fr) * frame) as i32;
                    let n = unsafe {
                        opus_decoder_dred_decode_float(c.out, c.dred, offset, a.as_mut_ptr(), frame as i32)
                    };
                    let m = r.dred_decode_float(&dred, offset, frame, &mut b).unwrap();
                    assert_eq!(n as usize, m);
                } else {
                    let null = std::ptr::null();
                    let n = unsafe { sys::opus_decode_float(c.out, null, 0, a.as_mut_ptr(), frame as c_int, 0) };
                    let m = r.decode(&[], frame, &mut b).unwrap();
                    assert_eq!(n as usize, m);
                }
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "packet {i}, loss {fr}");
                out_dred.extend_from_slice(&b);
            }
            lost_count = 0;
        }
        let n = unsafe { sys::opus_decode_float(c.out, p.as_ptr(), p.len() as i32, a.as_mut_ptr(), frame as c_int, 0) };
        let m = r.decode(p, frame, &mut b).unwrap();
        assert_eq!(n as usize, m);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "packet {i}");
        out_dred.extend_from_slice(&b);
    }
    let (ed, ep, ec) = (block_db(&out_dred, ch), block_db(&out_plc, ch), block_db(&out_clean, ch));
    let blocks_per_packet = frame / 480;
    let (mut sd, mut sp, mut n) = (0.0, 0.0, 0);
    for (i, &l) in lost.iter().enumerate() {
        if l {
            for k in i * blocks_per_packet..(i + 1) * blocks_per_packet {
                sd += (ed[k] - ec[k]).abs();
                sp += (ep[k] - ec[k]).abs();
                n += 1;
            }
        }
    }
    (sd / n as f64, sp / n as f64)
}

/// opus1's DRED data recovers lost packets with libopus's decoder, and the
/// audio is opus1's decoder's, bit for bit.
#[test]
fn libopus_recovers_losses_from_the_dred_data_of_opus1() {
    for cfg in [
        Cfg {
            fs: FS,
            channels: 1,
            application: Application::Voip,
            bitrate: 32000,
            vbr: true,
            complexity: 9,
            loss: 30,
            fec: false,
            dred: 104,
            frame: 960,
            mode: ForcedMode::Hybrid,
        },
        Cfg {
            fs: FS,
            channels: 2,
            application: Application::Audio,
            bitrate: 64000,
            vbr: false,
            complexity: 5,
            loss: 40,
            fec: true,
            dred: 104,
            frame: 480,
            mode: ForcedMode::CeltOnly,
        },
    ] {
        let (d, p) = recover(&cfg, true);
        println!("opus1's packets, {cfg:?}: DRED {d:.2} dB, PLC {p:.2} dB");
        assert!(d < p, "{cfg:?}: DRED {d} dB, PLC {p} dB");
    }
}

/// The other way around: libopus's DRED data with opus1's decoder.
#[test]
fn opus1_recovers_losses_from_the_dred_data_of_libopus() {
    let cfg = Cfg {
        fs: FS,
        channels: 1,
        application: Application::Voip,
        bitrate: 24000,
        vbr: true,
        complexity: 9,
        loss: 50,
        fec: false,
        dred: 104,
        frame: 960,
        mode: ForcedMode::Hybrid,
    };
    let (d, p) = recover(&cfg, false);
    println!("libopus's packets: DRED {d:.2} dB, PLC {p:.2} dB");
    assert!(d < p, "DRED {d} dB, PLC {p} dB");
}


/// The time per 20 ms frame of the encoder with DRED off and on, in
/// opus1 and in the linked libopus: 48 and 16 kHz mono, `Voip`, 32 kb/s,
/// 20 % expected loss, complexity 9, DRED of 1 s. Run alone:
///
/// ```text
/// ... cargo test --release --features test-libopus-dnn,dred,osce-bwe \
///   --test dnn_dred_encode_libopus -- --ignored --nocapture --test-threads 1 encoder_cost
/// ```
#[test]
#[ignore = "a measurement; run alone"]
fn encoder_cost_of_dred() {
    use std::time::Instant;
    for fs in [48000, 16000] {
        let frame = fs as usize / 50;
        let mut cfg = Cfg {
            fs,
            channels: 1,
            application: Application::Voip,
            bitrate: 32000,
            vbr: true,
            complexity: 9,
            loss: 20,
            fec: false,
            dred: 0,
            frame,
            mode: ForcedMode::Hybrid,
        };
        let pcm = input(fs, 1, 20);
        let frames = pcm.len() / frame;
        for dred in [0, 100] {
            cfg.dred = dred;
            let (mut best_c, mut best_r) = (f64::MAX, f64::MAX);
            for _ in 0..5 {
                let mut c = CEnc::new(&cfg);
                // Let the encoders choose the mode, as an application would.
                unsafe { sys::opus_encoder_ctl(c.0, OPUS_SET_FORCE_MODE, -1000) };
                let t = Instant::now();
                for f in pcm.chunks_exact(frame) {
                    std::hint::black_box(c.encode(f, frame));
                }
                best_c = best_c.min(t.elapsed().as_secs_f64());
                let mut r = opus1_encoder(&cfg);
                r.set_force_mode(None);
                let t = Instant::now();
                for f in pcm.chunks_exact(frame) {
                    std::hint::black_box(encode_opus1(&mut r, f, frame));
                }
                best_r = best_r.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{fs} Hz, DRED {dred}: libopus {:.1} us, opus1 {:.1} us per 20 ms frame",
                best_c * 1e6 / frames as f64,
                best_r * 1e6 / frames as f64
            );
        }
    }
}
