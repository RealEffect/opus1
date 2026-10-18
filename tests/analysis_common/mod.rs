//! Shared code of the tests that compare the speech and music analysis with
//! libopus.
//!
//! libopus is linked as a static library (the Homebrew build of libopus
//! 1.6.1, see `.cargo/config.toml`). Its analysis functions are not
//! `static`, so the tests declare and call them directly:
//! `tonality_analysis_init`, `tonality_analysis_reset`, `run_analysis`,
//! `tonality_get_info`, `analysis_compute_dense`, `analysis_compute_gru`,
//! the downmix functions and the weight tables `layer0` to `layer2`.
//!
//! `CState` is a `#[repr(C)]` copy of `TonalityAnalysisState`
//! (src/analysis.h:47-81) and `CInfo` one of `AnalysisInfo`
//! (celt/celt.h:65-79). The tests compare the whole state of the two
//! implementations after every call, bit for bit.
#![allow(dead_code)]

// The tests call functions of libopus that no Rust code of the `opus` or
// `audiopus_sys` crates names. This makes sure the library is linked.
extern crate audiopus_sys;

use opus1::internals::analysis::{
    AnalysisInfo, AnalysisPcm, LEAK_BANDS, PcmRef, SecondChannel, TonalityAnalysisState,
};
use std::os::raw::{c_int, c_void};

// ---------------------------------------------------------------------------
// The C side
// ---------------------------------------------------------------------------

/// libopus `AnalysisInfo`, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CInfo {
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

/// libopus `TonalityAnalysisState` of the float build, 14,004 bytes.
#[repr(C)]
pub struct CState {
    pub arch: c_int,
    pub application: c_int,
    pub fs: i32,
    pub angle: [f32; 240],
    pub d_angle: [f32; 240],
    pub d2_angle: [f32; 240],
    pub inmem: [f32; 720],
    pub mem_fill: c_int,
    pub prev_band_tonality: [f32; 18],
    pub prev_tonality: f32,
    pub prev_bandwidth: c_int,
    pub e: [[f32; 18]; 8],
    pub log_e: [[f32; 18]; 8],
    pub low_e: [f32; 18],
    pub high_e: [f32; 18],
    pub mean_e: [f32; 19],
    pub mem: [f32; 32],
    pub cmean: [f32; 8],
    pub std: [f32; 9],
    pub e_tracker: f32,
    pub low_e_count: f32,
    pub e_count: c_int,
    pub count: c_int,
    pub analysis_offset: c_int,
    pub write_pos: c_int,
    pub read_pos: c_int,
    pub read_subframe: c_int,
    pub hp_ener_accum: f32,
    pub initialized: c_int,
    pub rnn_state: [f32; 32],
    pub downmix_state: [f32; 3],
    pub info: [CInfo; 100],
}

/// libopus `AnalysisDenseLayer` (src/mlp.h).
#[repr(C)]
pub struct CDenseLayer {
    pub bias: *const i8,
    pub input_weights: *const i8,
    pub nb_inputs: c_int,
    pub nb_neurons: c_int,
    pub sigmoid: c_int,
}

/// libopus `AnalysisGRULayer` (src/mlp.h).
#[repr(C)]
pub struct CGruLayer {
    pub bias: *const i8,
    pub input_weights: *const i8,
    pub recurrent_weights: *const i8,
    pub nb_inputs: c_int,
    pub nb_neurons: c_int,
}

/// libopus `downmix_func` (src/opus_private.h).
pub type DownmixFunc = unsafe extern "C" fn(
    x: *const c_void,
    y: *mut f32,
    subframe: c_int,
    offset: c_int,
    c1: c_int,
    c2: c_int,
    channels: c_int,
);

unsafe extern "C" {
    pub fn tonality_analysis_init(st: *mut CState, fs: i32);
    pub fn tonality_analysis_reset(st: *mut CState);
    pub fn tonality_get_info(st: *mut CState, info: *mut CInfo, len: c_int);
    pub fn run_analysis(
        st: *mut CState,
        celt_mode: *const c_void,
        analysis_pcm: *const c_void,
        analysis_frame_size: c_int,
        frame_size: c_int,
        c1: c_int,
        c2: c_int,
        channels: c_int,
        fs: i32,
        lsb_depth: c_int,
        downmix: DownmixFunc,
        info: *mut CInfo,
    );
    pub fn downmix_float(
        x: *const c_void,
        y: *mut f32,
        subframe: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        channels: c_int,
    );
    pub fn downmix_int(
        x: *const c_void,
        y: *mut f32,
        subframe: c_int,
        offset: c_int,
        c1: c_int,
        c2: c_int,
        channels: c_int,
    );
    #[link_name = "is_digital_silence"]
    pub fn c_is_digital_silence(pcm: *const f32, frame_size: c_int, channels: c_int, lsb_depth: c_int) -> c_int;
    pub fn opus_custom_mode_create(fs: i32, frame_size: c_int, error: *mut c_int) -> *const c_void;
    pub fn analysis_compute_dense(layer: *const CDenseLayer, output: *mut f32, input: *const f32);
    pub fn analysis_compute_gru(layer: *const CGruLayer, state: *mut f32, input: *const f32);
    pub static layer0: CDenseLayer;
    pub static layer1: CGruLayer;
    pub static layer2: CDenseLayer;
}

/// The one CELT mode of libopus (48 kHz, 960 samples), whose 480-point FFT
/// the analysis uses.
pub fn celt_mode() -> *const c_void {
    let mut err = 0;
    // SAFETY: plain call; the static mode is returned, nothing is allocated.
    let mode = unsafe { opus_custom_mode_create(48000, 960, &mut err) };
    assert!(!mode.is_null() && err == 0, "opus_custom_mode_create failed: {err}");
    mode
}

/// The samples of one call, owned.
#[derive(Clone, Debug)]
pub enum Pcm {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

impl Pcm {
    pub fn as_ref(&self) -> PcmRef<'_> {
        match self {
            Pcm::F32(x) => PcmRef::F32(x),
            Pcm::I16(x) => PcmRef::I16(x),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Pcm::F32(x) => x.len(),
            Pcm::I16(x) => x.len(),
        }
    }

    /// Samples `start..end` of every channel (interleaved, `channels` per
    /// sample).
    pub fn slice(&self, start: usize, end: usize, channels: usize) -> Pcm {
        match self {
            Pcm::F32(x) => Pcm::F32(x[start * channels..end * channels].to_vec()),
            Pcm::I16(x) => Pcm::I16(x[start * channels..end * channels].to_vec()),
        }
    }

    fn ptr(&self) -> *const c_void {
        match self {
            Pcm::F32(x) => x.as_ptr() as *const c_void,
            Pcm::I16(x) => x.as_ptr() as *const c_void,
        }
    }

    fn downmix(&self) -> DownmixFunc {
        match self {
            Pcm::F32(_) => downmix_float,
            Pcm::I16(_) => downmix_int,
        }
    }
}

/// The channels the analysis reads: libopus `c1` and `c2`.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub channels: usize,
    pub c1: usize,
    /// -1, -2, or a channel.
    pub c2: i32,
}

impl Layout {
    /// `OpusEncoder` with `channels` channels: `c1 = 0`, `c2 = -2`.
    pub const fn encoder(channels: usize) -> Self {
        Self { channels, c1: 0, c2: -2 }
    }

    pub fn second(&self) -> SecondChannel {
        match self.c2 {
            -1 => SecondChannel::None,
            -2 => SecondChannel::AllOthers,
            c => SecondChannel::Channel(c as usize),
        }
    }
}

/// The libopus analysis with its own state.
pub struct CAnalysis {
    pub st: Box<CState>,
    mode: *const c_void,
}

impl CAnalysis {
    pub fn new(fs: i32) -> Self {
        // SAFETY: CState is plain data; all zero bytes is a valid value,
        // and tonality_analysis_init then sets it up.
        let mut st: Box<CState> = unsafe { Box::new(std::mem::zeroed()) };
        unsafe { tonality_analysis_init(&mut *st, fs) };
        Self { st, mode: celt_mode() }
    }

    pub fn reset(&mut self) {
        unsafe { tonality_analysis_reset(&mut *self.st) };
    }

    /// `run_analysis` on `pcm`, which holds `analysis_frame_size` samples
    /// per channel (or more).
    pub fn run(
        &mut self,
        pcm: &Pcm,
        layout: Layout,
        analysis_frame_size: usize,
        frame_size: usize,
        lsb_depth: i32,
    ) -> CInfo {
        assert!(pcm.len() >= analysis_frame_size * layout.channels);
        let mut info = CInfo::default();
        let fs = self.st.fs;
        // SAFETY: the buffer holds what the call reads; the state is valid.
        unsafe {
            run_analysis(
                &mut *self.st,
                self.mode,
                pcm.ptr(),
                analysis_frame_size as c_int,
                frame_size as c_int,
                layout.c1 as c_int,
                layout.c2,
                layout.channels as c_int,
                fs,
                lsb_depth,
                pcm.downmix(),
                &mut info,
            )
        };
        info
    }

    pub fn get_info(&mut self, len: usize) -> CInfo {
        let mut info = CInfo::default();
        unsafe { tonality_get_info(&mut *self.st, &mut info, len as c_int) };
        info
    }
}

/// The opus1 analysis, called as `CAnalysis` is.
pub fn rs_run(
    st: &mut TonalityAnalysisState,
    pcm: &Pcm,
    layout: Layout,
    analysis_frame_size: usize,
    frame_size: usize,
    lsb_depth: i32,
) -> AnalysisInfo {
    st.run(
        AnalysisPcm {
            pcm: pcm.as_ref(),
            channels: layout.channels,
            c1: layout.c1,
            c2: layout.second(),
            frame_size: analysis_frame_size,
        },
        frame_size,
        lsb_depth,
    )
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

/// The bits of a float for the comparisons. Every NaN maps to one value:
/// the sign and payload of a NaN depend on the operand order inside the
/// FFT butterflies, and they never change a number or a decision of the
/// analysis (every comparison with a NaN is false, whatever its bits).
/// Every other value, the sign of zero included, must match exactly.
pub fn fbits(v: f32) -> u32 {
    if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|&v| fbits(v)).collect()
}

/// The first field of two results that differs, or `None`. Floats are
/// compared bit for bit.
pub fn info_difference(rs: &AnalysisInfo, c: &CInfo) -> Option<String> {
    macro_rules! f {
        ($name:ident) => {
            if fbits(rs.$name) != fbits(c.$name) {
                return Some(format!(
                    "{}: opus1 {:e} ({:#010x}), libopus {:e} ({:#010x})",
                    stringify!($name),
                    rs.$name,
                    rs.$name.to_bits(),
                    c.$name,
                    c.$name.to_bits()
                ));
            }
        };
    }
    if rs.valid != (c.valid != 0) {
        return Some(format!("valid: opus1 {}, libopus {}", rs.valid, c.valid));
    }
    f!(tonality);
    f!(tonality_slope);
    f!(noisiness);
    f!(activity);
    f!(music_prob);
    f!(music_prob_min);
    f!(music_prob_max);
    if rs.bandwidth != c.bandwidth {
        return Some(format!("bandwidth: opus1 {}, libopus {}", rs.bandwidth, c.bandwidth));
    }
    f!(activity_probability);
    f!(max_pitch_ratio);
    if rs.leak_boost != c.leak_boost {
        return Some(format!(
            "leak_boost: opus1 {:?}, libopus {:?}",
            rs.leak_boost, c.leak_boost
        ));
    }
    None
}

/// The first field of the two states that differs, or `None`.
pub fn state_difference(rs: &TonalityAnalysisState, c: &CState) -> Option<String> {
    macro_rules! arr {
        ($name:ident) => {
            let a = bits(rs.$name.as_flattened_or_self());
            let b = bits(c.$name.as_flattened_or_self());
            if a != b {
                let i = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
                return Some(format!(
                    "{}[{}]: opus1 {:e}, libopus {:e}",
                    stringify!($name),
                    i,
                    f32::from_bits(a[i]),
                    f32::from_bits(b[i])
                ));
            }
        };
    }
    macro_rules! scalar_f {
        ($name:ident) => {
            if fbits(rs.$name) != fbits(c.$name) {
                return Some(format!(
                    "{}: opus1 {:e}, libopus {:e}",
                    stringify!($name),
                    rs.$name,
                    c.$name
                ));
            }
        };
    }
    macro_rules! scalar_i {
        ($name:ident) => {
            if rs.$name as i64 != c.$name as i64 {
                return Some(format!(
                    "{}: opus1 {}, libopus {}",
                    stringify!($name),
                    rs.$name as i64,
                    c.$name as i64
                ));
            }
        };
    }
    if rs.fs != c.fs {
        return Some(format!("fs: opus1 {}, libopus {}", rs.fs, c.fs));
    }
    arr!(angle);
    arr!(d_angle);
    arr!(d2_angle);
    arr!(inmem);
    scalar_i!(mem_fill);
    arr!(prev_band_tonality);
    scalar_f!(prev_tonality);
    scalar_i!(prev_bandwidth);
    arr!(e);
    arr!(log_e);
    arr!(low_e);
    arr!(high_e);
    arr!(mean_e);
    arr!(mem);
    arr!(cmean);
    arr!(std);
    scalar_f!(e_tracker);
    scalar_f!(low_e_count);
    scalar_i!(e_count);
    scalar_i!(count);
    scalar_i!(analysis_offset);
    scalar_i!(write_pos);
    scalar_i!(read_pos);
    scalar_i!(read_subframe);
    scalar_f!(hp_ener_accum);
    if rs.initialized != (c.initialized != 0) {
        return Some(format!(
            "initialized: opus1 {}, libopus {}",
            rs.initialized, c.initialized
        ));
    }
    arr!(rnn_state);
    arr!(downmix_state);
    for (i, (a, b)) in rs.info.iter().zip(c.info.iter()).enumerate() {
        if let Some(d) = info_difference(a, b) {
            return Some(format!("info[{i}].{d}"));
        }
    }
    None
}

/// Flattens `[[f32; N]; M]` and passes `[f32; N]` through.
pub trait Flat {
    fn as_flattened_or_self(&self) -> &[f32];
}

impl<const N: usize> Flat for [f32; N] {
    fn as_flattened_or_self(&self) -> &[f32] {
        self
    }
}

impl<const N: usize, const M: usize> Flat for [[f32; N]; M] {
    fn as_flattened_or_self(&self) -> &[f32] {
        self.as_flattened()
    }
}

/// Runs both implementations side by side and checks after every call
/// that the result and the whole state are the same bits.
pub struct Pair {
    pub c: CAnalysis,
    pub rs: TonalityAnalysisState,
    /// Calls so far, for the messages.
    pub calls: usize,
    /// Valid results seen so far.
    pub valid: usize,
    pub label: String,
}

impl Pair {
    pub fn new(fs: i32, label: impl Into<String>) -> Self {
        let p = Self {
            c: CAnalysis::new(fs),
            rs: TonalityAnalysisState::new(fs),
            calls: 0,
            valid: 0,
            label: label.into(),
        };
        p.check_state("after init");
        p
    }

    pub fn check_state(&self, when: &str) {
        if let Some(d) = state_difference(&self.rs, &self.c.st) {
            panic!("{}: state differs {when} (call {}): {d}", self.label, self.calls);
        }
    }

    fn check_info(&self, rs: &AnalysisInfo, c: &CInfo, what: &str) {
        if let Some(d) = info_difference(rs, c) {
            panic!("{}: {what} differs at call {}: {d}", self.label, self.calls);
        }
    }

    /// One `run_analysis` on both; returns the opus1 result.
    pub fn run(
        &mut self,
        pcm: &Pcm,
        layout: Layout,
        analysis_frame_size: usize,
        frame_size: usize,
        lsb_depth: i32,
    ) -> AnalysisInfo {
        let ci = self.c.run(pcm, layout, analysis_frame_size, frame_size, lsb_depth);
        let ri = rs_run(&mut self.rs, pcm, layout, analysis_frame_size, frame_size, lsb_depth);
        self.calls += 1;
        self.check_info(&ri, &ci, "run_analysis result");
        self.check_state("after run_analysis");
        if ri.valid {
            self.valid += 1;
        }
        ri
    }

    /// One `tonality_get_info` on both.
    pub fn get_info(&mut self, len: usize) -> AnalysisInfo {
        let ci = self.c.get_info(len);
        let ri = self.rs.get_info(len);
        self.calls += 1;
        self.check_info(&ri, &ci, "tonality_get_info result");
        self.check_state("after tonality_get_info");
        ri
    }

    pub fn reset(&mut self) {
        self.c.reset();
        self.rs.reset();
        self.check_state("after reset");
    }

    /// Feeds `pcm` (interleaved, `layout.channels` channels) in calls of
    /// `frame_size` samples, as an encoder with the default frame duration
    /// does: each call's buffer is the coded frame.
    pub fn feed(&mut self, pcm: &Pcm, layout: Layout, frame_size: usize, lsb_depth: i32) {
        let n = pcm.len() / layout.channels;
        let mut pos = 0;
        while pos + frame_size <= n {
            let buf = pcm.slice(pos, pos + frame_size, layout.channels);
            self.run(&buf, layout, frame_size, frame_size, lsb_depth);
            pos += frame_size;
        }
    }
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// A small deterministic generator.
#[derive(Clone)]
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() as f64 / (1u64 << 31) as f64 - 1.0) as f32
    }

    pub fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

/// `fixtures/answer_16k.wav`: 18 s of mono speech, 16-bit, 16 kHz.
pub fn speech_16k() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            return data
                .as_chunks::<2>().0.iter()
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk in answer_16k.wav");
}

/// The speech fixture at `fs`, by linear interpolation from 16 kHz. The
/// quality of the resampling does not matter here: both implementations
/// get the same samples.
pub fn speech(fs: i32, seconds: f32) -> Vec<f32> {
    let src = speech_16k();
    let n = (seconds * fs as f32) as usize;
    let step = 16000.0 / fs as f64;
    (0..n)
        .map(|i| {
            let t = (i as f64 * step) % (src.len() - 1) as f64;
            let k = t as usize;
            let f = (t - k as f64) as f32;
            src[k] * (1.0 - f) + src[k + 1] * f
        })
        .collect()
}

/// Music-like: a chord of harmonic tones with vibrato that changes every
/// half second, plus a soft drum hit on each beat.
pub fn music(fs: i32, seconds: f32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let n = (seconds * fs as f32) as usize;
    let mut out = vec![0.0f32; n];
    let beat = fs as usize / 2;
    let mut notes = [220.0f64, 277.2, 329.6];
    let mut phase = [0.0f64; 3];
    for (i, o) in out.iter_mut().enumerate() {
        if i % beat == 0 {
            for f in notes.iter_mut() {
                *f = 110.0 * 2f64.powf(rng.below(36) as f64 / 12.0);
            }
        }
        let t = i as f64 / fs as f64;
        let mut v = 0.0f64;
        for (k, f) in notes.iter().enumerate() {
            let vib = 1.0 + 0.004 * (2.0 * std::f64::consts::PI * 5.5 * t).sin();
            phase[k] += 2.0 * std::f64::consts::PI * f * vib / fs as f64;
            for h in 1..6 {
                if f * h as f64 * 2.0 < fs as f64 {
                    v += 0.05 / h as f64 * (phase[k] * h as f64).sin();
                }
            }
        }
        let since = (i % beat) as f64 / fs as f64;
        v += 0.3 * (-since * 40.0).exp() * rng.uniform() as f64;
        *o = v as f32;
    }
    out
}

/// A sum of steady tones.
pub fn tones(fs: i32, seconds: f32, freqs: &[f64], amp: f32) -> Vec<f32> {
    let n = (seconds * fs as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs as f64;
            let v: f64 = freqs.iter().map(|f| (2.0 * std::f64::consts::PI * f * t).sin()).sum();
            amp * v as f32 / freqs.len() as f32
        })
        .collect()
}

/// White noise of the given peak amplitude.
pub fn noise(fs: i32, seconds: f32, amp: f32, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let n = (seconds * fs as f32) as usize;
    (0..n).map(|_| amp * rng.uniform()).collect()
}

/// A sine sweep from 50 Hz to just below half the rate.
pub fn chirp(fs: i32, seconds: f32, amp: f32) -> Vec<f32> {
    let n = (seconds * fs as f32) as usize;
    let f1 = fs as f64 * 0.48;
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let f = 50.0 + (f1 - 50.0) * i as f64 / n as f64;
            phase += 2.0 * std::f64::consts::PI * f / fs as f64;
            amp * phase.sin() as f32
        })
        .collect()
}

/// Clicks: one full-scale sample every `period` samples.
pub fn clicks(fs: i32, seconds: f32, period: usize) -> Vec<f32> {
    let n = (seconds * fs as f32) as usize;
    (0..n).map(|i| if i % period == 0 { 0.9 } else { 0.0 }).collect()
}

/// Concatenation.
pub fn concat(parts: &[Vec<f32>]) -> Vec<f32> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// `channels` interleaved channels: channel `c` is `src` scaled by
/// `gains[c]` and delayed by `c` samples, so that the channels differ.
pub fn interleave(src: &[f32], gains: &[f32]) -> Vec<f32> {
    let ch = gains.len();
    let mut out = vec![0.0f32; src.len() * ch];
    for i in 0..src.len() {
        for c in 0..ch {
            let v = if i >= c { src[i - c] } else { 0.0 };
            out[i * ch + c] = v * gains[c];
        }
    }
    out
}

/// libopus `FLOAT2INT16`-like conversion: scale, round, saturate.
pub fn to_i16(x: &[f32]) -> Vec<i16> {
    x.iter()
        .map(|v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}
