// opus1 computes the 64-bit ARM bits of the neural network code on every
// target, and these tests compare with the bits of the linked libopus, so
// they are compiled for 64-bit ARM only (docs/10, Platforms).
#![cfg(all(feature = "test-libopus-dnn", target_arch = "aarch64"))]
#![cfg(feature = "osce")]
//! The OSCE port (`src/dnn/osce/`: LACE, NoLACE, the bandwidth extension,
//! their features and the filters of `nndsp.c`) against libopus 1.6.1's
//! own functions: `osce_enhance_frame`, `osce_reset`,
//! `osce_calculate_features`, `osce_bwe`, `osce_bwe_reset`,
//! `osce_bwe_calculate_features`, the two cross-fades,
//! `compute_overlap_window` and the three filters of `nndsp.c`.
//!
//! Those functions take libopus's whole SILK decoder state. A small C shim,
//! `scripts/dnn/osce_test_shim.c`, keeps one, fills the fields OSCE reads,
//! and calls them. `scripts/dnn/build-osce-shim.sh` compiles it with the
//! defines and flags of a reference build and links it with that build's
//! `libopus.a` into a shared library, which this test loads at run time
//! (`dlopen`). So the test links against any libopus, and one test binary
//! can compare with several reference builds.
//!
//! The inputs are real speech: files of the TSP corpus
//! (`$OPUS1_CORPUS/tsp48k/48k`, default `corpus/` in this repository or
//! one of its parent directories; `fixtures/` when the corpus is absent),
//! joined into long streams, encoded by opus1's encoder as SILK-only mono
//! packets of 10, 20, 40 and 60 ms at 6 to 32 kb/s, mostly wideband with
//! stretches of narrowband and mediumband, and decoded frame by frame with
//! opus1's SILK decoder functions, which give the parameters that OSCE
//! reads. Each decoded frame is checked against opus1's `OpusDecoder`.
//!
//! The streams switch the method (none, LACE, NoLACE) as a change of the
//! decoder complexity does, reset the channel as a lost frame and a decoder
//! reset do, unload the models for a while, and run the bandwidth
//! extension in 10 and 20 ms calls with resets. Every output sample must be
//! equal, for every kernel variant (`DnnArch`) this processor runs.
//!
//! Bit-exact reference (`plan.md` section 1): libopus built at `-O2` without
//! loop vectorization.
//!
//! ```text
//! scripts/dnn/build-osce-shim.sh target/android/libopus-host-1.6.1-dnn-novec
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,osce-bwe --test dnn_osce_libopus
//! ```
//!
//! The shim is found at `OPUS1_OSCE_SHIM` (a path), else at
//! `target/osce-shim/<name of LIBOPUS_LIB_DIR>/libosce_shim.dylib` (`.so`
//! on Linux) in this repository, where the script writes it. The test fails
//! with these instructions when it is missing.
//!
//! Other settings:
//!
//! - `OPUS1_OSCE_FILES`: how many corpus files each stream uses (default
//!   48; `all` for the whole corpus).
//! - `OPUS1_DNN_REF_O2=1` with `OPUS1_OSCE_SHIM` pointing at a shim built
//!   from the plain `-O2` build (`libopus-host-1.6.1-dnn`): measure instead
//!   of asserting equality. The test prints, per model, the frames that
//!   differ, the largest difference and the signal-to-noise ratio of the
//!   difference. Run it with `-- --nocapture --test-threads=1`.
//! - `OPUS1_OSCE_TIME=1`: also print the time per 20 ms frame of opus1 and
//!   of the shim's libopus for LACE, NoLACE and the bandwidth extension.

use opus1::internals::dnn::arch::{self, DnnArch};
use opus1::internals::dnn::builtin;
#[cfg(feature = "osce-bwe")]
use opus1::internals::dnn::models::bbwenet::{self as bw, BbwenetLayers};
use opus1::internals::dnn::models::lace::{self as la, LaceLayers};
use opus1::internals::dnn::models::nolace::{self as nl, NolaceLayers};
#[cfg(feature = "osce-bwe")]
use opus1::internals::dnn::osce::OsceBweChannel;
use opus1::internals::dnn::osce::features::{OSCE_FEATURE_DIM, OsceFeatureState, osce_calculate_features};
#[cfg(feature = "osce-bwe")]
use opus1::internals::dnn::osce::features::{OSCE_BWE_FEATURE_DIM, OsceBweFeatureState, osce_bwe_calculate_features};
use opus1::internals::dnn::osce::nndsp::{
    self, AdaCombParams, AdaCombState, AdaConvParams, AdaConvState, AdaShapeParams, AdaShapeState,
};
use opus1::internals::dnn::osce::{self, OSCE_DEFAULT_METHOD, OsceChannel, OsceFrame, OsceMethod, OsceModels};
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::decode_core::silk_decode_core;
use opus1::internals::silk::decode_frame::FLAG_DECODE_NORMAL;
use opus1::internals::silk::decode_indices::silk_decode_indices;
use opus1::internals::silk::decode_parameters::silk_decode_parameters;
use opus1::internals::silk::decode_pulses::silk_decode_pulses;
use opus1::internals::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use opus1::internals::silk::define::{CODE_CONDITIONALLY, CODE_INDEPENDENTLY};
use opus1::internals::silk::init_decoder::{silk_create_decoder, silk_decoder_set_fs};
use opus1::{
    Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusDecoder, OpusEncoder, SampleRate,
};
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// The shim
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
#[cfg(target_os = "macos")]
const RTLD_LOCAL: c_int = 4;
#[cfg(not(target_os = "macos"))]
const RTLD_LOCAL: c_int = 0;

#[cfg(target_os = "macos")]
const SHIM_FILE: &str = "libosce_shim.dylib";
#[cfg(not(target_os = "macos"))]
const SHIM_FILE: &str = "libosce_shim.so";

/// The interface version of `osce_test_shim.c` this test expects.
const SHIM_VERSION: c_int = 1;

type H = *mut c_void;

/// The functions of the shim (see `scripts/dnn/osce_test_shim.c`). The
/// bandwidth extension functions are used only with `osce-bwe`.
#[allow(clippy::type_complexity)]
#[cfg_attr(not(feature = "osce-bwe"), allow(dead_code))]
struct Api {
    path: PathBuf,
    version: unsafe extern "C" fn() -> c_int,
    create: unsafe extern "C" fn(c_int) -> H,
    destroy: unsafe extern "C" fn(H),
    arch: unsafe extern "C" fn(H) -> c_int,
    set_loaded: unsafe extern "C" fn(H, c_int),
    reset: unsafe extern "C" fn(H, c_int),
    method: unsafe extern "C" fn(H) -> c_int,
    bypass_count: unsafe extern "C" fn(H) -> c_int,
    enhance: unsafe extern "C" fn(H, c_int, c_int, c_int, c_int, *const i16, *const c_int, *const i16, *const i32, *mut i16, c_int),
    features: unsafe extern "C" fn(H, c_int, c_int, *const i16, *const c_int, *const i16, *const i32, *const i16, c_int, *mut f32, *mut f32, *mut c_int),
    features_reset: unsafe extern "C" fn(H),
    bwe_reset: unsafe extern "C" fn(H),
    bwe_clear: unsafe extern "C" fn(H),
    bwe: unsafe extern "C" fn(H, *mut i16, *const i16, c_int),
    bwe_features: unsafe extern "C" fn(H, *const i16, c_int, *mut f32),
    bwe_features_reset: unsafe extern "C" fn(H),
    cross_fade_10ms: unsafe extern "C" fn(*mut f32, *mut f32),
    bwe_cross_fade_10ms: unsafe extern "C" fn(*mut i16, *mut i16),
    overlap_window: unsafe extern "C" fn(*mut f32, c_int),
    blocks_reset: unsafe extern "C" fn(H),
    adacomb: unsafe extern "C" fn(H, c_int, *mut f32, *const f32, *const f32, c_int),
    adaconv: unsafe extern "C" fn(H, c_int, *mut f32, *const f32, *const f32),
    adashape: unsafe extern "C" fn(H, c_int, *mut f32, *const f32),
    sizeof: unsafe extern "C" fn(c_int) -> c_long,
}

// SAFETY: the shim's functions only touch the instance passed to them.
unsafe impl Sync for Api {}
unsafe impl Send for Api {}

/// Where the shim library is: `OPUS1_OSCE_SHIM`, else the default output of
/// `build-osce-shim.sh` for the libopus this test was built against.
fn shim_path() -> PathBuf {
    if let Some(p) = std::env::var_os("OPUS1_OSCE_SHIM").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let lib_dir = option_env!("LIBOPUS_LIB_DIR").unwrap_or("libopus");
    let name = Path::new(lib_dir).file_name().map(|n| n.to_owned()).unwrap_or_default();
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target/osce-shim").join(name).join(SHIM_FILE)
}

/// A function pointer of type `F` from a symbol address.
///
/// # Safety
///
/// `p` must be the address of a function with the signature `F`.
unsafe fn fn_ptr<F: Copy>(p: *mut c_void) -> F {
    assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: the caller guarantees the signature; the sizes are equal.
    unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) }
}

fn api() -> &'static Api {
    static API: OnceLock<Api> = OnceLock::new();
    API.get_or_init(|| {
        let path = shim_path();
        assert!(
            path.exists(),
            "the OSCE test shim {} is missing: build it with \
             `scripts/dnn/build-osce-shim.sh <reference build>` or set OPUS1_OSCE_SHIM",
            path.display()
        );
        let cpath = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        // SAFETY: dlopen of a library this repository builds; the symbols
        // are looked up by name and have the C signatures of the shim.
        unsafe {
            let lib = dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_LOCAL);
            if lib.is_null() {
                panic!("dlopen {}: {}", path.display(), CStr::from_ptr(dlerror()).to_string_lossy());
            }
            macro_rules! sym {
                ($name:literal) => {{
                    let c = CString::new($name).unwrap();
                    let p = dlsym(lib, c.as_ptr());
                    assert!(!p.is_null(), "the shim has no {}", $name);
                    fn_ptr(p)
                }};
            }
            let api = Api {
                path: path.clone(),
                version: sym!("osce_shim_version"),
                create: sym!("osce_shim_create"),
                destroy: sym!("osce_shim_destroy"),
                arch: sym!("osce_shim_arch"),
                set_loaded: sym!("osce_shim_set_loaded"),
                reset: sym!("osce_shim_reset"),
                method: sym!("osce_shim_method"),
                bypass_count: sym!("osce_shim_bypass_count"),
                enhance: sym!("osce_shim_enhance"),
                features: sym!("osce_shim_features"),
                features_reset: sym!("osce_shim_features_reset"),
                bwe_reset: sym!("osce_shim_bwe_reset"),
                bwe_clear: sym!("osce_shim_bwe_clear"),
                bwe: sym!("osce_shim_bwe"),
                bwe_features: sym!("osce_shim_bwe_features"),
                bwe_features_reset: sym!("osce_shim_bwe_features_reset"),
                cross_fade_10ms: sym!("osce_shim_cross_fade_10ms"),
                bwe_cross_fade_10ms: sym!("osce_shim_bwe_cross_fade_10ms"),
                overlap_window: sym!("osce_shim_overlap_window"),
                blocks_reset: sym!("osce_shim_blocks_reset"),
                adacomb: sym!("osce_shim_adacomb"),
                adaconv: sym!("osce_shim_adaconv"),
                adashape: sym!("osce_shim_adashape"),
                sizeof: sym!("osce_shim_sizeof"),
            };
            assert_eq!((api.version)(), SHIM_VERSION, "{} is another version of the shim", path.display());
            api
        }
    })
}

/// One shim instance: libopus's OSCE model and one SILK channel state.
struct Shim {
    api: &'static Api,
    h: H,
}

impl Shim {
    fn new() -> Shim {
        let api = api();
        // SAFETY: see `Api`.
        let h = unsafe { (api.create)(-1) };
        assert!(!h.is_null(), "osce_shim_create failed");
        Shim { api, h }
    }
    fn arch(&self) -> i32 {
        unsafe { (self.api.arch)(self.h) }
    }
    fn set_loaded(&mut self, loaded: bool) {
        unsafe { (self.api.set_loaded)(self.h, loaded as c_int) }
    }
    fn reset(&mut self, method: OsceMethod) {
        unsafe { (self.api.reset)(self.h, method as c_int) }
    }
    fn method(&self) -> i32 {
        unsafe { (self.api.method)(self.h) }
    }
    fn bypass_count(&self) -> i32 {
        unsafe { (self.api.bypass_count)(self.h) }
    }
    fn enhance(&mut self, f: &Frame, xq: &mut [i16]) {
        assert!(xq.len() >= f.nb_subfr as usize * 80 * f.fs_khz as usize / 16);
        unsafe {
            (self.api.enhance)(
                self.h,
                f.fs_khz,
                f.nb_subfr,
                f.lpc_order,
                f.signal_type,
                f.pred.as_ptr().cast(),
                f.pitch.as_ptr(),
                f.ltp.as_ptr(),
                f.gains.as_ptr(),
                xq.as_mut_ptr(),
                f.num_bits,
            )
        }
    }
    fn features(&mut self, f: &Frame, out: &mut [f32; 4 * OSCE_FEATURE_DIM], numbits: &mut [f32; 2], periods: &mut [i32; 4]) {
        assert!(f.xq.len() >= 320);
        unsafe {
            (self.api.features)(
                self.h,
                f.lpc_order,
                f.signal_type,
                f.pred.as_ptr().cast(),
                f.pitch.as_ptr(),
                f.ltp.as_ptr(),
                f.gains.as_ptr(),
                f.xq.as_ptr(),
                f.num_bits,
                out.as_mut_ptr(),
                numbits.as_mut_ptr(),
                periods.as_mut_ptr(),
            )
        }
    }
    fn features_reset(&mut self) {
        unsafe { (self.api.features_reset)(self.h) }
    }
    #[cfg(feature = "osce-bwe")]
    fn bwe_reset(&mut self) {
        unsafe { (self.api.bwe_reset)(self.h) }
    }
    #[cfg(feature = "osce-bwe")]
    fn bwe_clear(&mut self) {
        unsafe { (self.api.bwe_clear)(self.h) }
    }
    #[cfg(feature = "osce-bwe")]
    fn bwe(&mut self, xq48: &mut [i16], xq16: &[i16]) {
        assert!((xq16.len() == 160 || xq16.len() == 320) && xq48.len() >= 3 * xq16.len());
        unsafe { (self.api.bwe)(self.h, xq48.as_mut_ptr(), xq16.as_ptr(), xq16.len() as c_int) }
    }
    #[cfg(feature = "osce-bwe")]
    fn bwe_features(&mut self, xq16: &[i16], out: &mut [f32]) {
        assert!(out.len() >= xq16.len() / 160 * OSCE_BWE_FEATURE_DIM);
        unsafe { (self.api.bwe_features)(self.h, xq16.as_ptr(), xq16.len() as c_int, out.as_mut_ptr()) }
    }
    #[cfg(feature = "osce-bwe")]
    fn bwe_features_reset(&mut self) {
        unsafe { (self.api.bwe_features_reset)(self.h) }
    }
    fn blocks_reset(&mut self) {
        unsafe { (self.api.blocks_reset)(self.h) }
    }
    fn adacomb(&mut self, which: i32, x_out: &mut [f32], x_in: &[f32], features: &[f32], lag: usize) {
        assert!(x_out.len() >= 80 && x_in.len() >= 80);
        unsafe { (self.api.adacomb)(self.h, which, x_out.as_mut_ptr(), x_in.as_ptr(), features.as_ptr(), lag as c_int) }
    }
    fn adaconv(&mut self, which: i32, x_out: &mut [f32], x_in: &[f32], features: &[f32]) {
        unsafe { (self.api.adaconv)(self.h, which, x_out.as_mut_ptr(), x_in.as_ptr(), features.as_ptr()) }
    }
    fn adashape(&mut self, which: i32, x: &mut [f32], features: &[f32]) {
        unsafe { (self.api.adashape)(self.h, which, x.as_mut_ptr(), features.as_ptr()) }
    }
}

impl Drop for Shim {
    fn drop(&mut self) {
        unsafe { (self.api.destroy)(self.h) }
    }
}

/// `OPUS1_DNN_REF_O2=1`: measure the differences instead of asserting
/// equality.
fn measuring() -> bool {
    std::env::var("OPUS1_DNN_REF_O2").is_ok_and(|v| v == "1")
}

fn timing() -> bool {
    std::env::var("OPUS1_OSCE_TIME").is_ok_and(|v| v == "1")
}

// ---------------------------------------------------------------------------
// Speech, packets and decoded SILK frames
// ---------------------------------------------------------------------------

/// A small deterministic random number generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

/// The samples of a 16-bit PCM mono WAV file and its rate.
fn read_wav(path: &Path) -> (Vec<i16>, u32) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(&b[..4] == b"RIFF" && &b[8..12] == b"WAVE", "{} is not a WAV file", path.display());
    let mut pos = 12;
    let (mut rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let len = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &b[pos + 8..(pos + 8 + len).min(b.len())];
        if id == b"fmt " {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            assert!(channels == 1 && bits == 16, "{}: not 16-bit mono", path.display());
            return (body.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect(), rate);
        }
        pos += 8 + len + (len & 1);
    }
    panic!("{}: no data chunk", path.display());
}

/// The TSP corpus directory, if present.
fn corpus_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("OPUS1_CORPUS").filter(|d| !d.is_empty()) {
        let d = PathBuf::from(dir).join("tsp48k/48k");
        return d.is_dir().then_some(d);
    }
    // The repository root, or a parent of it (worktrees live in
    // `target/wt/` of the main checkout).
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().map(|a| a.join("corpus/tsp48k/48k")).find(|d| d.is_dir())
}

fn wav_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            wav_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "wav") {
            out.push(p);
        }
    }
}

/// The speech of one stream: `count` files of the corpus, starting at
/// `offset` and spread over it, at 48 kHz; or the fixtures.
fn speech(offset: usize) -> (Vec<i16>, u32) {
    let count = match std::env::var("OPUS1_OSCE_FILES") {
        Ok(v) if v == "all" => usize::MAX,
        Ok(v) => v.parse().expect("OPUS1_OSCE_FILES: a number or `all`"),
        Err(_) => 48,
    };
    let Some(dir) = corpus_dir() else {
        println!("the TSP corpus is absent: using fixtures/");
        let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let mut all = Vec::new();
        for name in ["tsp/FA01_05.wav", "tsp/FJ55_02.wav", "tsp/MD19_04.wav", "tsp/MH43_04.wav"] {
            let (s, r) = read_wav(&fx.join(name));
            assert_eq!(r, 48000);
            all.extend(s);
        }
        return (all, 48000);
    };
    let mut files = Vec::new();
    wav_files(&dir, &mut files);
    let n = count.min(files.len());
    let step = files.len() / n;
    let mut all = Vec::new();
    for k in 0..n {
        let (s, r) = read_wav(&files[(offset + k * step) % files.len()]);
        assert_eq!(r, 48000);
        all.extend(s);
    }
    (all, 48000)
}

/// SILK-only mono packets of `pcm`, in segments of changing frame
/// duration, bitrate and bandwidth.
fn encode(pcm: &[i16], rate: u32, rng: &mut Rng, with_narrowband: bool) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::Mono,
        Application::Voip,
    )
    .unwrap();
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    let samples_per_ms = rate as usize / 1000;
    let mut packets = Vec::new();
    let mut pos = 0;
    let mut out = [0u8; 1500];
    while pos < pcm.len() {
        // A segment of about 2 seconds with one setting.
        let ms = [20, 20, 20, 20, 10, 40, 60][rng.below(7) as usize];
        let bitrate = [6000, 9000, 12000, 16000, 24000, 32000][rng.below(6) as usize];
        let bw = if with_narrowband && rng.below(8) == 0 {
            [Bandwidth::Narrowband, Bandwidth::Mediumband][rng.below(2) as usize]
        } else {
            Bandwidth::Wideband
        };
        enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
        enc.set_bandwidth(bw);
        let frame = ms * samples_per_ms;
        for _ in 0..2000 / ms {
            if pos + frame > pcm.len() {
                return packets;
            }
            let n = enc.encode_i16(&pcm[pos..pos + frame], frame, &mut out).unwrap();
            packets.push(out[..n].to_vec());
            pos += frame;
        }
    }
    packets
}

/// One SILK frame as the SILK decoder gives it to OSCE.
#[derive(Clone)]
struct Frame {
    fs_khz: i32,
    nb_subfr: i32,
    lpc_order: i32,
    signal_type: i32,
    pred: [[i16; 16]; 2],
    pitch: [i32; 4],
    ltp: [i16; 20],
    gains: [i32; 4],
    num_bits: i32,
    /// The decoded samples (before enhancement).
    xq: Vec<i16>,
}

impl Frame {
    fn osce(&self) -> OsceFrame<'_> {
        OsceFrame {
            fs_khz: self.fs_khz,
            nb_subfr: self.nb_subfr,
            lpc_order: self.lpc_order,
            signal_type: self.signal_type,
            pred_coef_q12: &self.pred,
            pitch_l: &self.pitch,
            ltp_coef_q14: &self.ltp,
            gains_q16: &self.gains,
            num_bits: self.num_bits,
        }
    }
    fn eligible(&self) -> bool {
        self.fs_khz == 16 && self.nb_subfr == 4
    }
}

/// Decodes the frames of SILK-only mono packets with opus1's SILK
/// functions, as `silk_Decode` and `silk_decode_frame` do for packets
/// without LBRR data, keeping what OSCE reads.
fn decode_frames(packets: &[Vec<u8>]) -> Vec<Frame> {
    let mut st: SilkDecoderState = silk_create_decoder();
    let mut frames = Vec::new();
    for p in packets {
        let toc = p[0];
        let config = (toc >> 3) as usize;
        assert!(config < 12 && toc & 0x07 == 0, "not a mono SILK-only packet with one frame");
        let fs_khz = [8, 12, 16][config / 4];
        let (n_frames, nb_subfr) = [(1, 2), (1, 4), (2, 4), (3, 4)][config % 4];
        st.n_frames_decoded = 0;
        st.n_frames_per_packet = n_frames;
        st.nb_subfr = nb_subfr;
        assert!(silk_decoder_set_fs(&mut st, fs_khz, 16000) >= 0);
        let mut rc = RangeCoder::new_decoder(&p[1..]);
        for i in 0..n_frames as usize {
            st.vad_flags[i] = rc.decode_bit_logp(1) as i32;
        }
        st.lbrr_flag = rc.decode_bit_logp(1) as i32;
        assert_eq!(st.lbrr_flag, 0, "LBRR data is not handled here");
        st.lbrr_flags.fill(0);
        for f in 0..n_frames {
            let cond = if f == 0 { CODE_INDEPENDENTLY } else { CODE_CONDITIONALLY };
            let l = st.frame_length as usize;
            let ec_start = rc.tell();
            let index = st.n_frames_decoded;
            silk_decode_indices(&mut st, &mut rc, index, FLAG_DECODE_NORMAL, cond);
            let mut pulses = [0i16; 320];
            silk_decode_pulses(
                &mut rc,
                &mut pulses,
                st.indices.signal_type as i32,
                st.indices.quant_offset_type as i32,
                st.frame_length,
            );
            let mut ctrl = SilkDecoderControl::new();
            silk_decode_parameters(&mut st, &mut ctrl, cond);
            let mut xq = vec![0i16; l];
            silk_decode_core(&mut st, &mut ctrl, &mut xq, &pulses);
            let num_bits = rc.tell() - ec_start;
            let mv_len = (st.ltp_mem_length - st.frame_length) as usize;
            st.out_buf.rotate_left(l);
            st.out_buf[mv_len..mv_len + l].copy_from_slice(&xq);
            st.loss_cnt = 0;
            st.prev_signal_type = st.indices.signal_type as i32;
            st.first_frame_after_reset = 0;
            st.lag_prev = ctrl.pitch_l[st.nb_subfr as usize - 1];
            st.n_frames_decoded += 1;
            frames.push(Frame {
                fs_khz: st.fs_khz,
                nb_subfr: st.nb_subfr,
                lpc_order: st.lpc_order,
                signal_type: st.indices.signal_type as i32,
                pred: ctrl.pred_coef_q12,
                pitch: ctrl.pitch_l,
                ltp: ctrl.ltp_coef_q14,
                gains: ctrl.gains_q16,
                num_bits,
                xq,
            });
        }
    }
    frames
}

/// Checks the frames against opus1's `OpusDecoder` at 16 kHz. Its output
/// is the decoded SILK signal delayed by 13 samples: one sample of stereo
/// buffering and the 12 samples of delay of the SILK resampler from 16 to
/// 16 kHz.
///
/// The check stops before the first frame that is not at 16 kHz. Before a
/// SILK bandwidth switch the encoder adds a redundant CELT frame to the last
/// packet (`opusCanSwitch` in `src/opus_encoder.c`), which the Opus decoder
/// cross-fades into the last 2.5 ms of that packet's output; and the
/// resampler starts afresh at the new rate, so the last 13 samples do not
/// come out.
fn check_against_decoder(packets: &[Vec<u8>], frames: &[Frame]) {
    const DELAY: usize = 13;
    const REDUNDANCY: usize = 40;
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut out = vec![0i16; 960];
    let mut decoded = Vec::new();
    for p in packets {
        let n = dec.decode_i16(p, 960, &mut out).unwrap();
        decoded.extend_from_slice(&out[..n]);
    }
    let wideband = frames.iter().take_while(|f| f.fs_khz == 16).count();
    let silk: Vec<i16> = frames[..wideband].iter().flat_map(|f| f.xq.iter().copied()).collect();
    let end = if wideband < frames.len() { silk.len().saturating_sub(DELAY + REDUNDANCY) } else { silk.len() };
    let n = end.min(decoded.len() - DELAY);
    for i in 0..n {
        assert_eq!(decoded[i + DELAY], silk[i], "sample {i} of the SILK signal differs from OpusDecoder");
    }
}

/// The speech of stream `offset`, encoded and decoded.
fn frames_of_stream(offset: usize, seed: u64, with_narrowband: bool) -> Vec<Frame> {
    let (pcm, rate) = speech(offset);
    let mut rng = Rng(seed);
    let packets = encode(&pcm, rate, &mut rng, with_narrowband);
    let frames = decode_frames(&packets);
    check_against_decoder(&packets, &frames);
    frames
}

// ---------------------------------------------------------------------------
// Comparison and measurement
// ---------------------------------------------------------------------------

/// The differences between opus1 and the reference over a run.
#[derive(Default)]
struct Diff {
    frames: usize,
    frames_differ: usize,
    samples: usize,
    samples_differ: usize,
    max_abs: i32,
    signal: f64,
    noise: f64,
}

impl Diff {
    fn add(&mut self, want: &[i16], got: &[i16]) -> bool {
        self.frames += 1;
        let mut differ = false;
        for (&w, &g) in want.iter().zip(got) {
            let d = (w as i32 - g as i32).abs();
            self.samples += 1;
            self.signal += (w as f64) * (w as f64);
            self.noise += (d as f64) * (d as f64);
            if d != 0 {
                differ = true;
                self.samples_differ += 1;
                self.max_abs = self.max_abs.max(d);
            }
        }
        if differ {
            self.frames_differ += 1;
        }
        differ
    }
    fn report(&self, what: &str) {
        let snr = if self.noise == 0.0 { f64::INFINITY } else { 10.0 * (self.signal / self.noise).log10() };
        println!(
            "{what}: {} of {} frames differ, {} of {} samples, largest difference {} LSB, SNR of the difference {:.1} dB",
            self.frames_differ, self.frames, self.samples_differ, self.samples, self.max_abs, snr
        );
    }
}

fn compare(diff: &mut Diff, what: &str, index: usize, want: &[i16], got: &[i16]) {
    if diff.add(want, got) && !measuring() {
        let i = want.iter().zip(got).position(|(a, b)| a != b).unwrap();
        panic!("{what}: frame {index} differs first at sample {i}: libopus {}, opus1 {}", want[i], got[i]);
    }
}

fn assert_bits(what: &str, want: &[f32], got: &[f32]) {
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(w.to_bits(), g.to_bits(), "{what}: value {i}: libopus {w:e}, opus1 {g:e}");
    }
}

/// The kernel variants to test: every one this processor runs.
fn archs() -> Vec<DnnArch> {
    arch::available().to_vec()
}

fn models() -> &'static opus1::internals::dnn::models::ModelSet<'static> {
    builtin::models()
}

fn lace_layers() -> &'static LaceLayers<'static> {
    models().lace.as_ref().unwrap()
}
fn nolace_layers() -> &'static NolaceLayers<'static> {
    models().nolace.as_ref().unwrap()
}
#[cfg(feature = "osce-bwe")]
fn bbwenet_layers() -> &'static BbwenetLayers<'static> {
    models().bbwenet.as_ref().unwrap()
}

// ---------------------------------------------------------------------------
// Small pieces
// ---------------------------------------------------------------------------

#[test]
fn the_shim_is_the_expected_build() {
    let api = api();
    let shim = Shim::new();
    println!("shim: {}", api.path.display());
    println!("libopus arch: {}", shim.arch());
    // SAFETY: see `Api`.
    unsafe {
        println!(
            "sizeof silk_OSCE_struct {}, silk_OSCE_BWE_struct {}, OSCEModel {}, silk_decoder_state {}",
            (api.sizeof)(0),
            (api.sizeof)(1),
            (api.sizeof)(2),
            (api.sizeof)(3)
        );
    }
    assert_eq!(shim.method(), OSCE_DEFAULT_METHOD as i32);
    assert_eq!(shim.bypass_count(), 2);
    let ch = OsceChannel::new();
    assert_eq!(ch.method(), OSCE_DEFAULT_METHOD);
    assert_eq!(ch.bypass_count(), 2);
}

#[test]
fn overlap_windows_and_cross_fades() {
    let api = api();
    for (n, table) in [(40, &nndsp::OVERLAP_WINDOW_40[..]), (80, &nndsp::OVERLAP_WINDOW_80[..]), (120, &nndsp::OVERLAP_WINDOW_120[..])] {
        let mut want = vec![0.0f32; n];
        unsafe { (api.overlap_window)(want.as_mut_ptr(), n as c_int) };
        assert_bits(&format!("overlap window {n}"), &want, table);
        let mut got = vec![0.0f32; n];
        nndsp::compute_overlap_window(&mut got);
        if cfg!(target_os = "macos") {
            assert_bits(&format!("compute_overlap_window {n}"), &want, &got);
        }
    }
    let mut rng = Rng(5);
    for _ in 0..2000 {
        let mut x_in: [f32; 320] = std::array::from_fn(|_| rng.uniform());
        let x_enh: [f32; 320] = std::array::from_fn(|_| rng.uniform());
        let mut want = x_enh;
        unsafe { (api.cross_fade_10ms)(want.as_mut_ptr(), x_in.as_mut_ptr()) };
        let mut got = x_enh;
        osce::osce_cross_fade_10ms(&mut got, &x_in);
        assert_bits("osce_cross_fade_10ms", &want, &got);

        let scale = [1.0f32, 100.0, 32767.0][rng.below(3) as usize];
        let mut fade_out: [i16; 480] = std::array::from_fn(|_| (rng.uniform() * scale) as i16);
        let fade_in: [i16; 480] = std::array::from_fn(|_| (rng.uniform() * scale) as i16);
        let mut want = fade_in;
        unsafe { (api.bwe_cross_fade_10ms)(want.as_mut_ptr(), fade_out.as_mut_ptr()) };
        let mut got = fade_in;
        osce::osce_bwe_cross_fade_10ms(&mut got, &fade_out);
        assert_eq!(want, got, "osce_bwe_cross_fade_10ms");
    }
}

/// The three filters of `nndsp.c` with the layers of the models, on
/// speech and with random conditioning vectors in [-1, 1], 2000 calls each.
#[test]
fn nndsp_filters_match() {
    let frames = frames_of_stream(3, 11, false);
    let signal: Vec<f32> = frames.iter().filter(|f| f.fs_khz == 16).flat_map(|f| f.xq.iter().map(|&v| v as f32 / 32768.0)).collect();
    for &arch in &archs() {
        let mut shim = Shim::new();
        shim.blocks_reset();
        let mut rng = Rng(77);
        let lace = lace_layers();
        let nolace = nolace_layers();

        // AdaComb: LACE cf1 and NoLACE cf1.
        let mut comb = [AdaCombState::default(), AdaCombState::default()];
        let cf_lace = AdaCombParams {
            frame_size: la::LACE_FRAME_SIZE,
            overlap_size: la::LACE_OVERLAP_SIZE,
            kernel_size: la::LACE_CF1_KERNEL_SIZE,
            left_padding: la::LACE_CF1_LEFT_PADDING,
            filter_gain_a: la::LACE_CF1_FILTER_GAIN_A,
            filter_gain_b: la::LACE_CF1_FILTER_GAIN_B,
            log_gain_limit: la::LACE_CF1_LOG_GAIN_LIMIT,
        };
        let cf_nolace = AdaCombParams {
            frame_size: nl::NOLACE_FRAME_SIZE,
            overlap_size: nl::NOLACE_OVERLAP_SIZE,
            kernel_size: nl::NOLACE_CF1_KERNEL_SIZE,
            left_padding: nl::NOLACE_CF1_LEFT_PADDING,
            filter_gain_a: nl::NOLACE_CF1_FILTER_GAIN_A,
            filter_gain_b: nl::NOLACE_CF1_FILTER_GAIN_B,
            log_gain_limit: nl::NOLACE_CF1_LOG_GAIN_LIMIT,
        };
        // AdaConv: LACE af1, NoLACE af2.
        let mut conv_lace: AdaConvState<16, 16> = AdaConvState::default();
        let mut conv_nolace: AdaConvState<32, 64> = AdaConvState::default();
        let af_lace = AdaConvParams {
            frame_size: la::LACE_AF1_FRAME_SIZE,
            overlap_size: la::LACE_AF1_OVERLAP_SIZE,
            in_channels: la::LACE_AF1_IN_CHANNELS,
            out_channels: la::LACE_AF1_OUT_CHANNELS,
            kernel_size: la::LACE_AF1_KERNEL_SIZE,
            left_padding: la::LACE_AF1_LEFT_PADDING,
            filter_gain_a: la::LACE_AF1_FILTER_GAIN_A,
            filter_gain_b: la::LACE_AF1_FILTER_GAIN_B,
            shape_gain: la::LACE_AF1_SHAPE_GAIN,
        };
        let af_nolace = AdaConvParams {
            frame_size: nl::NOLACE_AF2_FRAME_SIZE,
            overlap_size: nl::NOLACE_AF2_OVERLAP_SIZE,
            in_channels: nl::NOLACE_AF2_IN_CHANNELS,
            out_channels: nl::NOLACE_AF2_OUT_CHANNELS,
            kernel_size: nl::NOLACE_AF2_KERNEL_SIZE,
            left_padding: nl::NOLACE_AF2_LEFT_PADDING,
            filter_gain_a: nl::NOLACE_AF2_FILTER_GAIN_A,
            filter_gain_b: nl::NOLACE_AF2_FILTER_GAIN_B,
            shape_gain: nl::NOLACE_AF2_SHAPE_GAIN,
        };
        // AdaShape: NoLACE tdshape1.
        let mut shape = AdaShapeState::default();
        let sh = AdaShapeParams {
            feature_dim: nl::NOLACE_TDSHAPE1_FEATURE_DIM,
            frame_size: nl::NOLACE_TDSHAPE1_FRAME_SIZE,
            avg_pool_k: nl::NOLACE_TDSHAPE1_AVG_POOL_K,
            interpolate_k: 1,
        };

        let calls = 2000.min(signal.len() / 160 - 1);
        for c in 0..calls {
            let x = &signal[c * 80..c * 80 + 160];
            let feat: [f32; 160] = std::array::from_fn(|_| rng.uniform());
            let lag = [7usize, 32, 100, 288][rng.below(4) as usize];

            let mut want = [0.0f32; 160];
            shim.adacomb(0, &mut want, x, &feat, lag);
            let mut got = [0.0f32; 80];
            got.copy_from_slice(&x[..80]);
            nndsp::adacomb_process_frame(&mut comb[0], &mut got, &feat, &lace.lace_cf1_kernel, &lace.lace_cf1_gain, &lace.lace_cf1_global_gain, lag, &cf_lace, &nndsp::OVERLAP_WINDOW_40, arch);
            assert_bits(&format!("LACE cf1, call {c}"), &want[..80], &got);

            shim.adacomb(1, &mut want, x, &feat, lag);
            got.copy_from_slice(&x[..80]);
            nndsp::adacomb_process_frame(&mut comb[1], &mut got, &feat, &nolace.nolace_cf1_kernel, &nolace.nolace_cf1_gain, &nolace.nolace_cf1_global_gain, lag, &cf_nolace, &nndsp::OVERLAP_WINDOW_40, arch);
            assert_bits(&format!("NoLACE cf1, call {c}"), &want[..80], &got);

            shim.adaconv(0, &mut want, x, &feat);
            nndsp::adaconv_process_frame(&mut conv_lace, &mut got, &x[..80], &feat, &lace.lace_af1_kernel, &lace.lace_af1_gain, &af_lace, &nndsp::OVERLAP_WINDOW_40, arch);
            assert_bits(&format!("LACE af1, call {c}"), &want[..80], &got);

            let mut got2 = [0.0f32; 160];
            shim.adaconv(1, &mut want, x, &feat);
            nndsp::adaconv_process_frame(&mut conv_nolace, &mut got2, x, &feat, &nolace.nolace_af2_kernel, &nolace.nolace_af2_gain, &af_nolace, &nndsp::OVERLAP_WINDOW_40, arch);
            assert_bits(&format!("NoLACE af2, call {c}"), &want, &got2);

            let mut want = [0.0f32; 80];
            want.copy_from_slice(&x[..80]);
            shim.adashape(0, &mut want, &feat);
            got.copy_from_slice(&x[..80]);
            nndsp::adashape_process_frame(&mut shape, &mut got, &feat, &nolace.nolace_tdshape1_alpha1_f, &nolace.nolace_tdshape1_alpha1_t, &nolace.nolace_tdshape1_alpha2, &sh, arch);
            assert_bits(&format!("NoLACE tdshape1, call {c}"), &want, &got);
        }

        #[cfg(feature = "osce-bwe")]
        bwe_filters(&mut shim, &signal, arch);
    }
}

/// The BBWENet filters: af1 (1 to 3 channels), af2 (3 to 3, 32 taps), af3
/// (3 to 1), tdshape1 and tdshape2 (interpolation 2).
#[cfg(feature = "osce-bwe")]
fn bwe_filters(shim: &mut Shim, signal: &[f32], arch: DnnArch) {
    let l = bbwenet_layers();
    let mut rng = Rng(99);
    let af1 = AdaConvParams {
        frame_size: bw::BBWENET_AF1_FRAME_SIZE,
        overlap_size: bw::BBWENET_AF1_OVERLAP_SIZE,
        in_channels: bw::BBWENET_AF1_IN_CHANNELS,
        out_channels: bw::BBWENET_AF1_OUT_CHANNELS,
        kernel_size: bw::BBWENET_AF1_KERNEL_SIZE,
        left_padding: bw::BBWENET_AF1_LEFT_PADDING,
        filter_gain_a: bw::BBWENET_AF1_FILTER_GAIN_A,
        filter_gain_b: bw::BBWENET_AF1_FILTER_GAIN_B,
        shape_gain: bw::BBWENET_AF1_SHAPE_GAIN,
    };
    let af2 = AdaConvParams {
        frame_size: bw::BBWENET_AF2_FRAME_SIZE,
        overlap_size: bw::BBWENET_AF2_OVERLAP_SIZE,
        in_channels: bw::BBWENET_AF2_IN_CHANNELS,
        out_channels: bw::BBWENET_AF2_OUT_CHANNELS,
        kernel_size: bw::BBWENET_AF2_KERNEL_SIZE,
        left_padding: bw::BBWENET_AF2_LEFT_PADDING,
        filter_gain_a: bw::BBWENET_AF2_FILTER_GAIN_A,
        filter_gain_b: bw::BBWENET_AF2_FILTER_GAIN_B,
        shape_gain: bw::BBWENET_AF2_SHAPE_GAIN,
    };
    let af3 = AdaConvParams {
        frame_size: bw::BBWENET_AF3_FRAME_SIZE,
        overlap_size: bw::BBWENET_AF3_OVERLAP_SIZE,
        in_channels: bw::BBWENET_AF3_IN_CHANNELS,
        out_channels: bw::BBWENET_AF3_OUT_CHANNELS,
        kernel_size: bw::BBWENET_AF3_KERNEL_SIZE,
        left_padding: bw::BBWENET_AF3_LEFT_PADDING,
        filter_gain_a: bw::BBWENET_AF3_FILTER_GAIN_A,
        filter_gain_b: bw::BBWENET_AF3_FILTER_GAIN_B,
        shape_gain: bw::BBWENET_AF3_SHAPE_GAIN,
    };
    let sh1 = AdaShapeParams {
        feature_dim: bw::BBWENET_TDSHAPE1_FEATURE_DIM,
        frame_size: bw::BBWENET_TDSHAPE1_FRAME_SIZE,
        avg_pool_k: bw::BBWENET_TDSHAPE1_AVG_POOL_K,
        interpolate_k: bw::BBWENET_TDSHAPE1_INTERPOLATE_K,
    };
    let sh2 = AdaShapeParams {
        feature_dim: bw::BBWENET_TDSHAPE2_FEATURE_DIM,
        frame_size: bw::BBWENET_TDSHAPE2_FRAME_SIZE,
        avg_pool_k: bw::BBWENET_TDSHAPE2_AVG_POOL_K,
        interpolate_k: bw::BBWENET_TDSHAPE2_INTERPOLATE_K,
    };
    let mut s1: AdaConvState<16, 48> = AdaConvState::default();
    let mut s2: AdaConvState<96, 288> = AdaConvState::default();
    let mut s3: AdaConvState<48, 48> = AdaConvState::default();
    let mut t1 = AdaShapeState::default();
    let mut t2 = AdaShapeState::default();
    let calls = 2000.min(signal.len() / 720 - 1);
    for c in 0..calls {
        let x = &signal[c * 240..c * 240 + 720];
        let feat: [f32; 128] = std::array::from_fn(|_| rng.uniform());
        let mut want = [0.0f32; 720];
        let mut got = [0.0f32; 720];

        shim.adaconv(2, &mut want, x, &feat);
        nndsp::adaconv_process_frame(&mut s1, &mut got, &x[..80], &feat, &l.bbwenet_af1_kernel, &l.bbwenet_af1_gain, &af1, &nndsp::OVERLAP_WINDOW_40, arch);
        assert_bits(&format!("BBWENet af1, call {c}"), &want[..240], &got[..240]);

        shim.adaconv(3, &mut want, x, &feat);
        nndsp::adaconv_process_frame(&mut s2, &mut got, &x[..480], &feat, &l.bbwenet_af2_kernel, &l.bbwenet_af2_gain, &af2, &nndsp::OVERLAP_WINDOW_80, arch);
        assert_bits(&format!("BBWENet af2, call {c}"), &want[..480], &got[..480]);

        shim.adaconv(4, &mut want, x, &feat);
        nndsp::adaconv_process_frame(&mut s3, &mut got, x, &feat, &l.bbwenet_af3_kernel, &l.bbwenet_af3_gain, &af3, &nndsp::OVERLAP_WINDOW_120, arch);
        assert_bits(&format!("BBWENet af3, call {c}"), &want[..240], &got[..240]);

        want[..160].copy_from_slice(&x[..160]);
        got[..160].copy_from_slice(&x[..160]);
        shim.adashape(1, &mut want, &feat);
        nndsp::adashape_process_frame(&mut t1, &mut got[..160], &feat, &l.bbwenet_tdshape1_alpha1_f, &l.bbwenet_tdshape1_alpha1_t, &l.bbwenet_tdshape1_alpha2, &sh1, arch);
        assert_bits(&format!("BBWENet tdshape1, call {c}"), &want[..160], &got[..160]);

        want[..240].copy_from_slice(&x[..240]);
        got[..240].copy_from_slice(&x[..240]);
        shim.adashape(2, &mut want, &feat);
        nndsp::adashape_process_frame(&mut t2, &mut got[..240], &feat, &l.bbwenet_tdshape2_alpha1_f, &l.bbwenet_tdshape2_alpha1_t, &l.bbwenet_tdshape2_alpha2, &sh2, arch);
        assert_bits(&format!("BBWENet tdshape2, call {c}"), &want[..240], &got[..240]);
    }
}

/// The 93 features per subframe, the bit counts and the pitch lags of
/// decoded speech, frame by frame.
#[test]
fn features_match() {
    let frames = frames_of_stream(5, 21, false);
    let mut shim = Shim::new();
    shim.features_reset();
    let mut st = OsceFeatureState::default();
    let mut n = 0;
    for (i, f) in frames.iter().enumerate() {
        if !f.eligible() {
            // libopus resets (clears) the features here.
            shim.features_reset();
            st = OsceFeatureState::default();
            continue;
        }
        let (mut want, mut got) = ([0.0f32; 4 * OSCE_FEATURE_DIM], [0.0f32; 4 * OSCE_FEATURE_DIM]);
        let (mut nb_want, mut nb_got) = ([0.0f32; 2], [0.0f32; 2]);
        let (mut p_want, mut p_got) = ([0i32; 4], [0i32; 4]);
        shim.features(f, &mut want, &mut nb_want, &mut p_want);
        osce_calculate_features(&mut st, &f.osce(), &f.xq, &mut got, &mut nb_got, &mut p_got);
        assert_eq!(p_want, p_got, "frame {i}: periods");
        assert_bits(&format!("frame {i}: numbits"), &nb_want, &nb_got);
        assert_bits(&format!("frame {i}: features"), &want, &got);
        n += 1;
    }
    println!("{n} frames of features compared");
    assert!(n > 1000);
}

/// The BWE features of decoded speech, in 10 and 20 ms calls.
#[cfg(feature = "osce-bwe")]
#[test]
fn bwe_features_match() {
    let frames = frames_of_stream(7, 31, false);
    let mut shim = Shim::new();
    shim.bwe_features_reset();
    let mut st = OsceBweFeatureState::default();
    let mut rng = Rng(3);
    for (i, f) in frames.iter().filter(|f| f.fs_khz == 16).enumerate() {
        let chunks: Vec<&[i16]> = if f.xq.len() == 320 && rng.below(3) == 0 { f.xq.chunks(160).collect() } else { vec![&f.xq[..]] };
        for x in chunks {
            let mut want = [0.0f32; 2 * OSCE_BWE_FEATURE_DIM];
            let mut got = [0.0f32; 2 * OSCE_BWE_FEATURE_DIM];
            shim.bwe_features(x, &mut want);
            osce_bwe_calculate_features(&mut st, &mut got, x);
            assert_bits(&format!("frame {i}: BWE features"), &want, &got);
        }
    }
}

// ---------------------------------------------------------------------------
// Whole models
// ---------------------------------------------------------------------------

/// What happens before a frame in the scenario of `run_enhancement`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Event {
    /// The decoder complexity changes the method (`dec_API.c:350-354`).
    Method(OsceMethod),
    /// The previous frame was lost: `osce_reset(method)` after the
    /// concealment, and this frame is not enhanced.
    Lost,
    /// `silk_reset_decoder`: `osce_reset(OSCE_DEFAULT_METHOD)`.
    DecoderReset,
    /// The models are unloaded or loaded again (libopus `model->loaded`).
    Loaded(bool),
}

/// Runs `osce_enhance_frame` of opus1 and of libopus over the frames, with
/// the events `scenario` gives before each frame, and compares every
/// output sample. Also runs `osce_enhance_frame_with_features` (features
/// with method NONE too) and checks that it gives the same output.
fn run_enhancement(what: &str, frames: &[Frame], arch: DnnArch, mut scenario: impl FnMut(usize) -> Event) -> Diff {
    let mut shim = Shim::new();
    let mut ch = OsceChannel::new();
    let mut ch_all = OsceChannel::new();
    let loaded = OsceModels::builtin(arch);
    let unloaded = OsceModels { lace: None, nolace: None, ..loaded };
    let mut is_loaded = true;
    let mut diff = Diff::default();
    let mut methods = [0usize; 3];
    for (i, f) in frames.iter().enumerate() {
        match scenario(i) {
            Event::Method(m) => {
                if ch.method() != m {
                    osce::osce_reset(&mut ch, m);
                    osce::osce_reset(&mut ch_all, m);
                }
                if shim.method() != m as i32 {
                    shim.reset(m);
                }
            }
            Event::Lost => {
                let m = ch.method();
                osce::osce_reset(&mut ch, m);
                osce::osce_reset(&mut ch_all, m);
                shim.reset(m);
                continue;
            }
            Event::DecoderReset => {
                osce::osce_reset(&mut ch, OSCE_DEFAULT_METHOD);
                osce::osce_reset(&mut ch_all, OSCE_DEFAULT_METHOD);
                shim.reset(OSCE_DEFAULT_METHOD);
            }
            Event::Loaded(l) => {
                is_loaded = l;
                shim.set_loaded(l);
            }
        }
        let models = if is_loaded { &loaded } else { &unloaded };
        let mut want = f.xq.clone();
        let mut got = f.xq.clone();
        let mut got_all = f.xq.clone();
        shim.enhance(f, &mut want);
        osce::osce_enhance_frame(&mut ch, models, &f.osce(), &mut got);
        osce::osce_enhance_frame_with_features(&mut ch_all, models, &f.osce(), &mut got_all);
        compare(&mut diff, what, i, &want, &got);
        assert_eq!(got, got_all, "{what}: frame {i}: skipping the features of method NONE changed the output");
        assert_eq!(shim.method(), ch.method() as i32, "{what}: frame {i}: method");
        assert_eq!(shim.bypass_count(), ch.bypass_count(), "{what}: frame {i}: bypass counter");
        if f.eligible() {
            methods[ch.method() as usize] += 1;
        }
    }
    println!(
        "{what} ({arch:?}): {} frames, eligible frames with method NONE {}, LACE {}, NoLACE {}",
        frames.len(),
        methods[0],
        methods[1],
        methods[2]
    );
    diff
}

/// LACE alone over a long stream, with the resets of lost frames and of
/// ineligible (10 ms, narrowband, mediumband) frames.
#[test]
fn lace_matches_libopus() {
    let frames = frames_of_stream(0, 1, true);
    for &arch in &archs() {
        let mut rng = Rng(100);
        let diff = run_enhancement("LACE", &frames, arch, |i| {
            if i == 0 {
                Event::Method(OsceMethod::Lace)
            } else if rng.below(200) == 0 {
                Event::Lost
            } else {
                Event::Method(OsceMethod::Lace)
            }
        });
        diff.report(&format!("LACE ({arch:?})"));
        if measuring() {
            break;
        }
    }
}

/// NoLACE alone over a long stream, as `lace_matches_libopus`.
#[test]
fn nolace_matches_libopus() {
    let frames = frames_of_stream(1, 2, true);
    for &arch in &archs() {
        let mut rng = Rng(200);
        let diff = run_enhancement("NoLACE", &frames, arch, |_| {
            if rng.below(200) == 0 { Event::Lost } else { Event::Method(OsceMethod::Nolace) }
        });
        diff.report(&format!("NoLACE ({arch:?})"));
        if measuring() {
            break;
        }
    }
}

/// Method switches (NONE, LACE, NoLACE), lost frames, decoder resets and
/// unloaded models in one stream.
#[test]
fn method_switches_and_resets_match_libopus() {
    let frames = frames_of_stream(2, 3, true);
    for &arch in &archs() {
        let mut rng = Rng(300);
        let mut method = OsceMethod::Nolace;
        let diff = run_enhancement("switches", &frames, arch, |_| match rng.below(400) {
            0..=5 => {
                method = [OsceMethod::None, OsceMethod::Lace, OsceMethod::Nolace][rng.below(3) as usize];
                Event::Method(method)
            }
            6 | 7 => Event::Lost,
            8 => {
                method = OSCE_DEFAULT_METHOD;
                Event::DecoderReset
            }
            9 => Event::Loaded(false),
            10..=13 => Event::Loaded(true),
            _ => Event::Method(method),
        });
        diff.report(&format!("switches ({arch:?})"));
        if measuring() {
            break;
        }
    }
}

/// The bandwidth extension over a long stream of decoded 16 kHz speech, in
/// 10 and 20 ms calls, with `osce_bwe_reset` (the start of a BBWE stretch)
/// and the cleared state of a new channel.
#[cfg(feature = "osce-bwe")]
#[test]
fn bwe_matches_libopus() {
    let frames = frames_of_stream(4, 4, false);
    let input: Vec<&Frame> = frames.iter().filter(|f| f.fs_khz == 16).collect();
    for &arch in &archs() {
        let mut shim = Shim::new();
        shim.bwe_clear();
        let mut ch = OsceBweChannel::new();
        let layers = bbwenet_layers();
        let mut rng = Rng(400);
        let mut diff = Diff::default();
        for (i, f) in input.iter().enumerate() {
            match rng.below(300) {
                0 => {
                    shim.bwe_reset();
                    osce::osce_bwe_reset(&mut ch);
                }
                1 => {
                    shim.bwe_clear();
                    ch = OsceBweChannel::new();
                }
                _ => {}
            }
            let chunks: Vec<&[i16]> = if f.xq.len() == 320 && rng.below(4) == 0 { f.xq.chunks(160).collect() } else { vec![&f.xq[..]] };
            for x in chunks {
                let mut want = vec![0i16; 3 * x.len()];
                let mut got = vec![0i16; 3 * x.len()];
                shim.bwe(&mut want, x);
                osce::osce_bwe(&mut ch, layers, arch, &mut got, x);
                compare(&mut diff, "BWE", i, &want, &got);
            }
        }
        diff.report(&format!("BWE ({arch:?})"));
        if measuring() {
            break;
        }
    }
}

/// NoLACE and then the bandwidth extension of its output, as a decoder at
/// complexity 7 with BWE runs them.
#[cfg(feature = "osce-bwe")]
#[test]
fn nolace_then_bwe_matches_libopus() {
    let frames = frames_of_stream(6, 6, false);
    let arch = arch::detect();
    let mut shim = Shim::new();
    let mut shim_bwe = Shim::new();
    shim.reset(OsceMethod::Nolace);
    shim_bwe.bwe_reset();
    let mut ch = OsceChannel::new();
    let mut bwe = OsceBweChannel::new();
    osce::osce_bwe_reset(&mut bwe);
    let models = OsceModels::builtin(arch);
    let mut diff = Diff::default();
    for (i, f) in frames.iter().filter(|f| f.fs_khz == 16).enumerate() {
        let mut want = f.xq.clone();
        let mut got = f.xq.clone();
        shim.enhance(f, &mut want);
        osce::osce_enhance_frame(&mut ch, &models, &f.osce(), &mut got);
        let mut want48 = vec![0i16; 3 * want.len()];
        let mut got48 = vec![0i16; 3 * got.len()];
        // Each side extends its own enhanced output.
        shim_bwe.bwe(&mut want48, &want);
        osce::osce_bwe(&mut bwe, models.bbwenet.unwrap(), arch, &mut got48, &got);
        compare(&mut diff, "NoLACE + BWE", i, &want48, &got48);
    }
    diff.report("NoLACE + BWE (48 kHz)");
}

// ---------------------------------------------------------------------------
// Time per frame
// ---------------------------------------------------------------------------

/// The time per 20 ms frame of opus1 and of the shim's libopus, with
/// `OPUS1_OSCE_TIME=1` (best of 5 runs over the same frames).
#[test]
fn time_per_frame() {
    if !timing() {
        println!("set OPUS1_OSCE_TIME=1 to time the models");
        return;
    }
    let frames: Vec<Frame> = frames_of_stream(8, 8, false).into_iter().filter(|f| f.eligible()).collect();
    let arch = arch::detect();
    let models = OsceModels::builtin(arch);
    let per_frame = |d: Duration| d.as_secs_f64() * 1e6 / frames.len() as f64;
    for method in [OsceMethod::None, OsceMethod::Lace, OsceMethod::Nolace] {
        let (mut best_rust, mut best_c) = (Duration::MAX, Duration::MAX);
        for _ in 0..5 {
            let mut ch = OsceChannel::new();
            osce::osce_reset(&mut ch, method);
            let t = Instant::now();
            for f in &frames {
                let mut x = f.xq.clone();
                osce::osce_enhance_frame(&mut ch, &models, &f.osce(), &mut x);
                std::hint::black_box(&x);
            }
            best_rust = best_rust.min(t.elapsed());
            let mut shim = Shim::new();
            shim.reset(method);
            let t = Instant::now();
            for f in &frames {
                let mut x = f.xq.clone();
                shim.enhance(f, &mut x);
                std::hint::black_box(&x);
            }
            best_c = best_c.min(t.elapsed());
        }
        println!(
            "{method:?}: opus1 {:.1} us, libopus {:.1} us per 20 ms frame ({} frames, {arch:?})",
            per_frame(best_rust),
            per_frame(best_c),
            frames.len()
        );
    }
    #[cfg(feature = "osce-bwe")]
    {
        let layers = bbwenet_layers();
        let (mut best_rust, mut best_c) = (Duration::MAX, Duration::MAX);
        let mut out = vec![0i16; 960];
        for _ in 0..5 {
            let mut ch = OsceBweChannel::new();
            let t = Instant::now();
            for f in &frames {
                osce::osce_bwe(&mut ch, layers, arch, &mut out, &f.xq);
                std::hint::black_box(&out);
            }
            best_rust = best_rust.min(t.elapsed());
            let mut shim = Shim::new();
            let t = Instant::now();
            for f in &frames {
                shim.bwe(&mut out, &f.xq);
                std::hint::black_box(&out);
            }
            best_c = best_c.min(t.elapsed());
        }
        println!(
            "BWE: opus1 {:.1} us, libopus {:.1} us per 20 ms frame",
            per_frame(best_rust),
            per_frame(best_c)
        );
    }
}
