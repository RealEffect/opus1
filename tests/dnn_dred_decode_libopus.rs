// opus1 computes the 64-bit ARM bits of the neural network code on every
// target, and these tests compare with the bits of the linked libopus, so
// they are compiled for 64-bit ARM only (docs/10, Platforms).
#![cfg(all(feature = "test-libopus-dnn", feature = "dred", target_arch = "aarch64"))]
//! The DRED decoder against libopus 1.6.1 built with DRED, without loop
//! vectorization (`libopus-host-1.6.1-dred-novec`). Every comparison is
//! exact.
//!
//! - `opus_dred_parse` and `opus_dred_process` on the packets of libopus's
//!   encoder with DRED, at several rates, channel counts, frame sizes,
//!   bitrates, complexities, expected losses and DRED durations: the
//!   result, `dred_end`, the number of latent vectors, the offset, the
//!   initial state, the latent vectors and the decoded features.
//! - The same on DRED payloads that are random, cut short or damaged.
//! - `opus_decoder_dred_decode`, `_float` and `24` for bursts of 1 to 50
//!   lost packets, rebuilt from the packet after the burst as `opus_demo`
//!   does, at 16 and 48 kHz, mono and stereo.
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dred-novec LIBOPUS_NO_PKG=1 \
//!   LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dred-ref cargo test --release \
//!   --features test-libopus-dnn,dred --test dnn_dred_decode_libopus
//! ```

use audiopus_sys as sys;
use opus1::extensions::Extension;
use opus1::internals::dnn::dred::{DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID};
use opus1::{ErrorKind, OpusDecoder, OpusDred, OpusDredDecoder, packet};
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
    fn opus_dred_get_size() -> c_int;
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
    fn opus_dred_process(dred_dec: *mut c_void, src: *const CDred, dst: *mut CDred) -> c_int;
    fn opus_decoder_dred_decode(st: *mut sys::OpusDecoder, dred: *const CDred, dred_offset: i32, pcm: *mut i16, frame_size: i32) -> c_int;
    fn opus_decoder_dred_decode24(st: *mut sys::OpusDecoder, dred: *const CDred, dred_offset: i32, pcm: *mut i32, frame_size: i32) -> c_int;
    fn opus_decoder_dred_decode_float(st: *mut sys::OpusDecoder, dred: *const CDred, dred_offset: i32, pcm: *mut f32, frame_size: i32) -> c_int;
}

const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
const OPUS_SET_DRED_DURATION_REQUEST: c_int = 4050;

fn linked() {
    let version = unsafe { CStr::from_ptr(sys::opus_get_version_string()) };
    assert!(version.to_string_lossy().contains("1.6.1"), "{version:?}");
}

/// A libopus DRED decoder and DRED state, freed on drop.
struct CDredDec {
    dec: *mut c_void,
    dred: *mut CDred,
    dred2: *mut CDred,
}

impl CDredDec {
    fn new() -> Self {
        linked();
        assert_eq!(unsafe { opus_dred_get_size() } as usize, size_of::<CDred>(), "OpusDRED layout");
        let mut err = 0;
        let dec = unsafe { opus_dred_decoder_create(&mut err) };
        assert!(err == 0 && !dec.is_null());
        let dred = unsafe { opus_dred_alloc(&mut err) };
        assert!(err == 0 && !dred.is_null());
        let dred2 = unsafe { opus_dred_alloc(&mut err) };
        assert!(err == 0 && !dred2.is_null());
        CDredDec { dec, dred, dred2 }
    }

    /// `opus_dred_parse`: the result and `dred_end`.
    fn parse(&mut self, data: &[u8], max: i32, rate: i32, defer: bool) -> (i32, i32) {
        let mut end = -1;
        let ret = unsafe {
            opus_dred_parse(self.dec, self.dred, data.as_ptr(), data.len() as i32, max, rate, &mut end, defer as c_int)
        };
        (ret, end)
    }

    fn dred(&self) -> &CDred {
        unsafe { &*self.dred }
    }
}

impl Drop for CDredDec {
    fn drop(&mut self) {
        unsafe {
            opus_dred_free(self.dred);
            opus_dred_free(self.dred2);
            opus_dred_decoder_destroy(self.dec);
        }
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Exact equality, with the first difference in the message.
fn same<T: PartialEq + std::fmt::Debug>(a: &[T], b: &[T], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
        panic!("{what}: {n} of {} differ, first at {i}: {:?} != {:?}", a.len(), a[i], b[i]);
    }
}

/// The parsed fields of both states agree.
fn same_parse(c: &CDred, r: &OpusDred, what: &str) {
    let stage = if r.is_processed() {
        2
    } else if r.is_parsed() {
        1
    } else {
        -1
    };
    assert_eq!(c.process_stage, stage, "{what}: process_stage");
    if stage < 0 {
        return;
    }
    let n = c.nb_latents as usize;
    assert_eq!(r.nb_latents(), n, "{what}: nb_latents");
    assert_eq!(r.dred_offset(), c.dred_offset, "{what}: dred_offset");
    same(&bits(r.state()), &bits(&c.state), &format!("{what}: state"));
    same(&bits(r.latents()), &bits(&c.latents[..n * 26]), &format!("{what}: latents"));
    if stage == 2 {
        same(r.features(), &c.fec_features[..n * 80], &format!("{what}: features"));
    }
}

/// A libopus encoder with DRED.
#[derive(Clone, Copy, Debug)]
struct EncCfg {
    fs: i32,
    channels: i32,
    application: c_int,
    bitrate: i32,
    complexity: i32,
    loss: i32,
    dred: i32,
    /// Samples per channel of each packet, at `fs`.
    frame: usize,
}

const VOIP: c_int = sys::OPUS_APPLICATION_VOIP as c_int;
const AUDIO: c_int = sys::OPUS_APPLICATION_AUDIO as c_int;

const CONFIGS: [EncCfg; 6] = [
    EncCfg { fs: 16000, channels: 1, application: VOIP, bitrate: 64000, complexity: 10, loss: 40, dred: 100, frame: 320 },
    EncCfg { fs: 48000, channels: 1, application: VOIP, bitrate: 32000, complexity: 5, loss: 30, dred: 50, frame: 960 },
    EncCfg { fs: 48000, channels: 2, application: AUDIO, bitrate: 96000, complexity: 9, loss: 35, dred: 104, frame: 480 },
    EncCfg { fs: 16000, channels: 1, application: VOIP, bitrate: 20000, complexity: 3, loss: 25, dred: 20, frame: 640 },
    EncCfg { fs: 48000, channels: 1, application: VOIP, bitrate: 40000, complexity: 7, loss: 50, dred: 104, frame: 2880 },
    EncCfg { fs: 24000, channels: 2, application: VOIP, bitrate: 48000, complexity: 10, loss: 15, dred: 70, frame: 480 },
];

/// Speech-like test signal: a gliding tone with a varying level, pauses
/// and some noise; the second channel is shifted in time.
fn signal(fs: i32, channels: usize, n: usize) -> Vec<i16> {
    let mut seed = 1u32;
    let mut out = Vec::with_capacity(n * channels);
    for i in 0..n {
        for c in 0..channels {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let t = (i + 37 * c) as f64 / fs as f64;
            let f = 150.0 + 50.0 * (t * 3.0).sin();
            let level = (0.6 + 0.4 * (t * 7.0).sin()) * if (t * 0.9).fract() < 0.8 { 1.0 } else { 0.05 };
            let v = 6000.0 * (2.0 * std::f64::consts::PI * f * t).sin() * level;
            out.push((v + ((seed >> 16) % 1000) as f64 - 500.0) as i16);
        }
    }
    out
}

fn encode(cfg: &EncCfg, packets: usize) -> Vec<Vec<u8>> {
    let mut err = 0;
    let enc = unsafe { sys::opus_encoder_create(cfg.fs, cfg.channels, cfg.application, &mut err) };
    assert_eq!(err, 0);
    unsafe {
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, cfg.bitrate), 0);
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, cfg.complexity), 0);
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, cfg.loss), 0);
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION_REQUEST, cfg.dred), 0, "a libopus without DRED");
    }
    let ch = cfg.channels as usize;
    let pcm = signal(cfg.fs, ch, packets * cfg.frame);
    let out = pcm
        .chunks_exact(cfg.frame * ch)
        .map(|frame| {
            let mut buf = vec![0u8; 1500];
            let n = unsafe { sys::opus_encode(enc, frame.as_ptr(), cfg.frame as c_int, buf.as_mut_ptr(), buf.len() as i32) };
            assert!(n > 0);
            buf.truncate(n as usize);
            buf
        })
        .collect();
    unsafe { sys::opus_encoder_destroy(enc) };
    out
}

/// The (max_dred_samples, sampling_rate) pairs of the parse tests.
const PARSE_ARGS: [(i32, i32); 8] = [
    (0, 16000),
    (480, 48000),
    (16000, 16000),
    (48000, 48000),
    (2000, 8000),
    (100000, 48000),
    (-5000, 16000),
    (7, 44100),
];

/// Parses one packet with both libopus and opus1, with and without
/// deferred processing, and compares everything.
fn compare_parse(c: &mut CDredDec, r: &OpusDredDecoder, dred: &mut OpusDred, data: &[u8], max: i32, rate: i32, what: &str) -> i32 {
    for defer in [false, true] {
        let what = format!("{what} max {max} rate {rate} defer {defer}");
        let (ret, end) = c.parse(data, max, rate, defer);
        let got = r.parse(dred, data, max, rate, defer);
        match got {
            Ok(p) => {
                assert_eq!((p.samples, p.end), (ret, end), "{what}");
            }
            Err(e) => {
                assert!(ret < 0, "{what}: opus1 {e}, libopus {ret}");
            }
        }
        same_parse(c.dred(), dred, &what);
        if defer {
            // Processing into another state, as libopus allows.
            let cret = unsafe { opus_dred_process(c.dec, c.dred, c.dred2) };
            let mut dst = OpusDred::new();
            let rret = r.process(dred, &mut dst);
            assert_eq!(rret.is_ok(), cret == 0, "{what}: process {rret:?} {cret}");
            if cret == 0 {
                same_parse(unsafe { &*c.dred2 }, &dst, &format!("{what} process"));
                // Processing in place, twice: the second call does nothing.
                let cret = unsafe { opus_dred_process(c.dec, c.dred, c.dred) };
                assert_eq!(cret, 0);
                r.process_in_place(dred).unwrap();
                r.process_in_place(dred).unwrap();
                same_parse(c.dred(), dred, &format!("{what} in place"));
            }
        }
        if defer {
            return ret;
        }
    }
    unreachable!()
}

#[test]
fn parse_and_process_equal_libopus() {
    let mut c = CDredDec::new();
    let r = OpusDredDecoder::new();
    assert!(r.is_loaded());
    let mut dred = OpusDred::new();
    for (k, cfg) in CONFIGS.iter().enumerate() {
        let packets = encode(cfg, (48000 / cfg.frame).max(40));
        let mut with_dred = 0;
        let mut latents = 0;
        for (i, p) in packets.iter().enumerate() {
            let (max, rate) = PARSE_ARGS[(i + k) % PARSE_ARGS.len()];
            let ret = compare_parse(&mut c, &r, &mut dred, p, max, rate, &format!("config {k} packet {i}"));
            assert!(ret >= 0);
            // The whole DRED of the packet.
            compare_parse(&mut c, &r, &mut dred, p, 48000, 48000, &format!("config {k} packet {i} all"));
            if c.dred().process_stage > 0 {
                with_dred += 1;
                latents = latents.max(c.dred().nb_latents);
            }
        }
        assert!(with_dred > packets.len() / 2, "config {k}: DRED in {with_dred} of {} packets", packets.len());
        eprintln!("config {k}: DRED in {with_dred} of {} packets, up to {latents} latent vectors", packets.len());
    }
}

/// Splits a packet with DRED into its plain form and its DRED payload
/// (without the 2-byte header) and the frame that carried it.
fn split_dred(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>, usize)> {
    let parsed = packet::parse(data, false).ok()?;
    let mut it = opus1::extensions::ExtensionIterator::new(parsed.padding(data), parsed.count);
    let ext = it.find(DRED_EXTENSION_ID).ok()??;
    let payload = ext.data[2..].to_vec();
    let mut plain = data.to_vec();
    let len = packet::unpad(&mut plain, data.len()).unwrap();
    plain.truncate(len);
    Some((plain, payload, ext.frame))
}

/// A packet `plain` with `payload` as DRED data of frame `frame`, with the
/// DRED header.
fn with_dred(plain: &[u8], payload: &[u8], frame: usize) -> Vec<u8> {
    let mut data = vec![b'D', DRED_EXPERIMENTAL_VERSION];
    data.extend_from_slice(payload);
    let mut buf = plain.to_vec();
    buf.resize(plain.len() + data.len() + 64, 0);
    let ext = [Extension { id: DRED_EXTENSION_ID, frame, data: &data }];
    let new_len = buf.len();
    let n = packet::pad_with_extensions(&mut buf, plain.len(), new_len, false, &ext).unwrap();
    buf.truncate(n);
    buf
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

#[test]
fn damaged_and_random_payloads_parse_as_in_libopus() {
    let mut c = CDredDec::new();
    let r = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut rng = Rng(0x5EED);
    let mut cases = 0;
    for (k, cfg) in CONFIGS.iter().enumerate() {
        let packets = encode(cfg, 30);
        for (i, p) in packets.iter().enumerate() {
            let Some((plain, payload, frame)) = split_dred(p) else { continue };
            for trial in 0..6 {
                let mut bad = payload.clone();
                match trial {
                    // Cut short.
                    0 => bad.truncate(rng.next() as usize % (payload.len() + 1)),
                    // Bits flipped.
                    1 | 2 => {
                        for _ in 0..1 + rng.next() % 8 {
                            let at = rng.next() as usize % bad.len();
                            bad[at] ^= 1 << (rng.next() % 8);
                        }
                    }
                    // Random bytes of the same length.
                    3 => bad.iter_mut().for_each(|b| *b = rng.next() as u8),
                    // Random bytes of a random length.
                    4 => bad = (0..rng.next() % 400).map(|_| rng.next() as u8).collect(),
                    // All ones, then all zeros.
                    _ => bad.iter_mut().for_each(|b| *b = if i % 2 == 0 { 0xFF } else { 0 }),
                }
                let data = with_dred(&plain, &bad, frame);
                let (max, rate) = PARSE_ARGS[(i + trial) % PARSE_ARGS.len()];
                compare_parse(&mut c, &r, &mut dred, &data, max, rate, &format!("config {k} packet {i} trial {trial}"));
                compare_parse(&mut c, &r, &mut dred, &data, 48000, 48000, &format!("config {k} packet {i} trial {trial} all"));
                cases += 1;
            }
        }
    }
    assert!(cases > 300, "{cases}");
    // Packets that are not valid, or whose padding is not valid extension
    // data, give the error of libopus.
    for n in 0..2000 {
        let len = rng.next() as usize % 64;
        let mut data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        if n % 2 == 0 && len > 2 {
            // Code 3 with padding, so that the extension parser runs.
            data[0] = (data[0] & 0xFC) | 3;
            data[1] |= 0x40;
        }
        compare_parse(&mut c, &r, &mut dred, &data, 48000, 48000, &format!("random packet {n}"));
    }
}

/// The output format of a DRED decode.
#[derive(Clone, Copy, Debug)]
enum Out {
    I16,
    F32,
    I24,
}

/// Two decoders, libopus and opus1, decoding the same packets.
struct Pair {
    c: *mut sys::OpusDecoder,
    r: OpusDecoder,
    channels: usize,
}

impl Pair {
    fn new(fs: i32, channels: usize, complexity: i32) -> Self {
        let mut err = 0;
        let c = unsafe { sys::opus_decoder_create(fs, channels as c_int, &mut err) };
        assert_eq!(err, 0);
        assert_eq!(unsafe { sys::opus_decoder_ctl(c, OPUS_SET_COMPLEXITY_REQUEST, complexity) }, 0);
        let mut r = OpusDecoder::new(fs, channels).unwrap();
        r.set_complexity(complexity).unwrap();
        Pair { c, r, channels }
    }

    /// Decodes a packet (or a loss, with `data` empty) in both.
    fn decode(&mut self, data: &[u8], frame: usize, out: Out, what: &str) {
        let n = frame * self.channels;
        let ptr = if data.is_empty() { std::ptr::null() } else { data.as_ptr() };
        match out {
            Out::I16 => {
                let mut a = vec![0i16; n];
                let mut b = vec![0i16; n];
                let ca = unsafe { sys::opus_decode(self.c, ptr, data.len() as i32, a.as_mut_ptr(), frame as c_int, 0) };
                let rb = self.r.decode_i16(data, frame, &mut b).unwrap();
                assert_eq!(rb as i32, ca, "{what}");
                same(&a, &b, what);
            }
            Out::F32 | Out::I24 => {
                let mut a = vec![0f32; n];
                let mut b = vec![0f32; n];
                let ca = unsafe { sys::opus_decode_float(self.c, ptr, data.len() as i32, a.as_mut_ptr(), frame as c_int, 0) };
                let rb = self.r.decode(data, frame, &mut b).unwrap();
                assert_eq!(rb as i32, ca, "{what}");
                same(&bits(&a), &bits(&b), what);
            }
        }
    }

    /// `opus_decoder_dred_decode*` in both.
    fn dred_decode(&mut self, cd: &CDred, rd: &OpusDred, offset: i32, frame: usize, out: Out, what: &str) {
        let n = frame * self.channels;
        match out {
            Out::I16 => {
                let mut a = vec![0i16; n];
                let mut b = vec![0i16; n];
                let ca = unsafe { opus_decoder_dred_decode(self.c, cd, offset, a.as_mut_ptr(), frame as i32) };
                let rb = self.r.dred_decode(rd, offset, frame, &mut b).unwrap();
                assert_eq!(rb as i32, ca, "{what}");
                same(&a, &b, what);
            }
            Out::F32 => {
                let mut a = vec![0f32; n];
                let mut b = vec![0f32; n];
                let ca = unsafe { opus_decoder_dred_decode_float(self.c, cd, offset, a.as_mut_ptr(), frame as i32) };
                let rb = self.r.dred_decode_float(rd, offset, frame, &mut b).unwrap();
                assert_eq!(rb as i32, ca, "{what}");
                same(&bits(&a), &bits(&b), what);
            }
            Out::I24 => {
                let mut a = vec![0i32; n];
                let mut b = vec![0i32; n];
                let ca = unsafe { opus_decoder_dred_decode24(self.c, cd, offset, a.as_mut_ptr(), frame as i32) };
                let rb = self.r.dred_decode24(rd, offset, frame, &mut b).unwrap();
                assert_eq!(rb as i32, ca, "{what}");
                same(&a, &b, what);
            }
        }
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.c) };
    }
}

/// The bursts of lost packets, each followed by `GAP` received packets.
const BURSTS: [usize; 11] = [1, 2, 3, 4, 6, 9, 13, 20, 30, 41, 50];
const GAP: usize = 4;
const LEAD: usize = 12;

/// Decodes the packets of `cfg` with bursts of losses, concealing each
/// lost packet with the DRED data of the packet after the burst, as
/// `opus_demo` does, in libopus and in opus1.
fn dred_losses(cfg: &EncCfg, fs: i32, channels: usize, complexity: i32, out: Out) -> usize {
    let total = LEAD + BURSTS.iter().map(|b| b + GAP).sum::<usize>();
    let packets = encode(cfg, total);
    let frame = cfg.frame * fs as usize / cfg.fs as usize;
    let mut pair = Pair::new(fs, channels, complexity);
    let mut c = CDredDec::new();
    let r = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut lost = vec![false; total];
    let mut at = LEAD;
    for &b in &BURSTS {
        lost[at..at + b].iter_mut().for_each(|l| *l = true);
        at += b + GAP;
    }
    let tag = format!("{cfg:?} at {fs} Hz, {channels} ch, complexity {complexity}, {out:?}");
    let mut lost_count = 0;
    let mut dred_frames = 0;
    for (i, p) in packets.iter().enumerate() {
        if lost[i] {
            lost_count += 1;
            continue;
        }
        if lost_count > 0 {
            let dred_input = (lost_count * frame) as i32;
            let max = fs.min(dred_input.max(0));
            let (ret, end) = c.parse(p, max, fs, false);
            let got = r.parse(&mut dred, p, max, fs, false).unwrap();
            assert_eq!((got.samples, got.end), (ret, end), "{tag} packet {i}: parse");
            same_parse(c.dred(), &dred, &format!("{tag} packet {i}"));
            for fr in 0..lost_count {
                let what = format!("{tag} packet {i}, loss {fr} of {lost_count}");
                if ret > 0 {
                    let offset = ((lost_count - fr) * frame) as i32;
                    pair.dred_decode(c.dred(), &dred, offset, frame, out, &what);
                    dred_frames += 1;
                } else {
                    pair.decode(&[], frame, out, &what);
                }
            }
            lost_count = 0;
        }
        pair.decode(p, frame, out, &format!("{tag} packet {i}"));
        assert_eq!(pair.r.last_packet_duration(), frame);
    }
    dred_frames
}

#[test]
fn dred_decode_mono_16k_equals_libopus() {
    let n = dred_losses(&CONFIGS[0], 16000, 1, 0, Out::I16);
    assert!(n > 100, "{n}");
    dred_losses(&CONFIGS[3], 16000, 1, 10, Out::F32);
    dred_losses(&CONFIGS[1], 16000, 1, 5, Out::I24);
}

#[test]
fn dred_decode_stereo_48k_equals_libopus() {
    let n = dred_losses(&CONFIGS[2], 48000, 2, 10, Out::I16);
    assert!(n > 100, "{n}");
    dred_losses(&CONFIGS[5], 48000, 2, 0, Out::F32);
    dred_losses(&CONFIGS[0], 48000, 2, 7, Out::I24);
}

#[test]
fn dred_decode_mono_48k_and_stereo_16k_equal_libopus() {
    dred_losses(&CONFIGS[1], 48000, 1, 10, Out::F32);
    dred_losses(&CONFIGS[4], 48000, 1, 0, Out::I16);
    dred_losses(&CONFIGS[2], 16000, 2, 5, Out::I16);
    dred_losses(&CONFIGS[5], 16000, 2, 10, Out::I24);
}

#[test]
fn dred_decode_at_other_positions_equals_libopus() {
    // Offsets that do not fall on a frame of the packet, longer and shorter
    // calls, an unprocessed state and a state without DRED.
    let cfg = &CONFIGS[0];
    let packets = encode(cfg, 80);
    for (fs, channels) in [(16000, 1usize), (48000, 2)] {
        let f10 = fs as usize / 100;
        let mut pair = Pair::new(fs, channels, 10);
        let mut c = CDredDec::new();
        let r = OpusDredDecoder::new();
        let mut dred = OpusDred::new();
        let mut rng = Rng(fs as u64);
        for (i, p) in packets.iter().enumerate() {
            let frame = 2 * f10;
            if i < 10 || i % 5 != 0 {
                pair.decode(p, frame, Out::F32, &format!("packet {i}"));
                continue;
            }
            let defer = i % 3 == 0;
            let (ret, _) = c.parse(p, fs, fs, defer);
            r.parse(&mut dred, p, fs, fs, defer).unwrap();
            assert!(ret > 0);
            for call in 0..4 {
                // The last call is longer than 120 ms, which the integer
                // variants of opus1 conceal in pieces.
                let len = if call == 3 {
                    f10 / 4 * (49 + rng.next() as usize % 400)
                } else {
                    f10 / 4 * (1 + rng.next() as usize % 24)
                };
                let offset = (rng.next() % (2 * fs as u32)) as i32 - fs / 4;
                let what = format!("{fs} Hz packet {i} call {call}: {len} at {offset}");
                let out = match call {
                    0 | 2 => Out::F32,
                    1 => Out::I16,
                    _ if i % 2 == 0 => Out::I24,
                    _ => Out::I16,
                };
                pair.dred_decode(c.dred(), &dred, offset, len, out, &what);
            }
            pair.decode(p, frame, Out::F32, &format!("packet {i}"));
        }
    }
}

#[test]
fn dred_decode_checks_its_arguments_as_libopus() {
    let packets = encode(&CONFIGS[0], 20);
    let mut c = CDredDec::new();
    let r = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let p = &packets[15];
    assert!(c.parse(p, 16000, 16000, false).0 > 0);
    r.parse(&mut dred, p, 16000, 16000, false).unwrap();
    let mut pair = Pair::new(16000, 1, 10);
    for q in &packets[..15] {
        pair.decode(q, 320, Out::I16, "lead");
    }
    // frame_size 0 and frame sizes that are not a multiple of 2.5 ms.
    for frame in [0usize, 1, 39, 41, 330] {
        let mut a = vec![0i16; 400];
        let mut b = vec![0i16; 400];
        let ca = unsafe { opus_decoder_dred_decode(pair.c, c.dred(), 320, a.as_mut_ptr(), frame as i32) };
        assert!(ca < 0, "{frame}");
        assert!(pair.r.dred_decode(&dred, 320, frame, &mut b).is_err(), "{frame}");
    }
    // A buffer that is too small is an error in opus1 only.
    let mut small = vec![0i16; 100];
    assert!(pair.r.dred_decode(&dred, 320, 320, &mut small).is_err());
    // The state is unchanged: the next decode equals libopus's.
    pair.dred_decode(c.dred(), &dred, 320, 320, Out::I16, "after the errors");
    // process without a parsed state.
    let empty = OpusDred::new();
    let mut dst = OpusDred::new();
    assert_eq!(r.process(&empty, &mut dst).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
    // A packet without DRED: zeros, and the state is not usable.
    let plain = split_dred(p).unwrap().0;
    let (ret, end) = c.parse(&plain, 16000, 16000, false);
    let got = r.parse(&mut dred, &plain, 16000, 16000, false).unwrap();
    assert_eq!((got.samples, got.end), (ret, end));
    assert_eq!((ret, end), (0, 0));
    assert!(!dred.is_parsed());
    // Not a positive sampling rate: an error in opus1 (libopus divides by
    // it).
    assert_eq!(r.parse(&mut dred, p, 16000, 0, false).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
    // A decode with the unusable state conceals as without DRED.
    pair.dred_decode(c.dred(), &dred, 320, 320, Out::F32, "without DRED");
}
