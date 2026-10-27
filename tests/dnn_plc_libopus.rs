// opus1 computes the 64-bit ARM bits of the neural network code on every
// target, and these tests compare with the bits of the linked libopus, so
// they are compiled for 64-bit ARM only (docs/guide/10, Platforms).
#![cfg(all(feature = "test-libopus-dnn", target_arch = "aarch64"))]
#![cfg(feature = "deep-plc")]
//! Deep PLC (`opus1::internals::dnn::plc`) against libopus 1.6.1's own
//! functions, whole models at a time: the feature analysis
//! (`lpcnet_compute_single_frame_features_float`), PitchDNN
//! (`compute_pitchdnn`), FARGAN (`fargan_cont`, `fargan_synthesize`,
//! `fargan_synthesize_int`), the state machine (`lpcnet_plc_init`,
//! `lpcnet_plc_reset`, `lpcnet_plc_update`, `lpcnet_plc_conceal`,
//! `lpcnet_plc_fec_add`, `lpcnet_plc_fec_clear`, which run
//! `compute_plc_pred` inside) and the CELT downsampler
//! (`update_plc_state`).
//!
//! The C states are `#[repr(C)]` copies of the libopus structs, so each
//! test also compares the states after every call. Every `DnnArch` this
//! processor runs is compared, in lockstep with one C state.
//!
//! The signals are `fixtures/answer_16k.wav` and the TSP speech corpus
//! (`corpus/tsp48k`, or `OPUS1_CORPUS`), resampled to 16 kHz. By default a
//! subset of the corpus runs; `OPUS1_DNN_PLC_ALL=1` runs all 1,444 files.
//!
//! The bit-exact reference is libopus built without loop vectorization
//! (`plan.md` section 1):
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref OPUS1_CORPUS=<repo>/corpus \
//!   cargo test --release --features test-libopus-dnn,deep-plc --test dnn_plc_libopus
//! ```
//!
//! With `OPUS1_DNN_REF_O2=1` and the `-O2` build
//! (`libopus-host-1.6.1-dnn`), the tests measure instead of asserting
//! equality: they print how far the outputs are apart and check only that
//! the concealed audio stays within the spread that different libopus
//! builds have among themselves (`deep_plc.md` section 8.4).
//!
//! `timing` (ignored by default) prints the cost of each call for opus1
//! and libopus: `-- --ignored --nocapture timing`.

mod dnn_gen;
mod dnn_plc_common;

use dnn_gen::{CFARGAN, CPLCModel, CPitchDNN};
use dnn_plc_common::{Lcg, fixture, short_name, tsp_16k, tsp_subset};
use opus1::internals::dnn::arch::{self, DnnArch};
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::models::DeepPlcModels;
use opus1::internals::dnn::plc::fargan::{
    self, FARGAN_CONT_SAMPLES, FARGAN_FRAME_SIZE, FarganState,
};
use opus1::internals::dnn::plc::features::{LpcnetEncState, compute_single_frame_features};
use opus1::internals::dnn::plc::lpcnet_plc::CELT_DECODE_BUFFER_SIZE;
use opus1::internals::dnn::plc::pitchdnn::{PitchDnnState, compute_pitchdnn};
use opus1::internals::dnn::plc::{
    CONT_VECTORS, DeepPlc, LPCNET_FRAME_SIZE, NB_FEATURES, NB_TOTAL_FEATURES, NB_XCORR_FEATURES,
    PITCH_IF_FEATURES, PLC_BUF_SIZE, PLC_MAX_FEC,
};
use std::os::raw::c_int;

const FRAME: usize = LPCNET_FRAME_SIZE;

// ---------------------------------------------------------------------------
// The C side
// ---------------------------------------------------------------------------

/// `#[repr(C)]` copies of the libopus states. Their sizes are checked
/// against the sizes measured on this Mac (`deep_plc.md`, Review) and
/// against `lpcnet_encoder_get_size()`.
mod c {
    use super::*;

    #[repr(C)]
    pub struct PitchDnnState {
        pub model: CPitchDNN,
        pub gru_state: [f32; 64],
        pub xcorr_mem1: [f32; 452],
        pub xcorr_mem2: [f32; 3616],
        pub xcorr_mem3: [f32; 3616],
    }

    #[repr(C)]
    pub struct EncState {
        pub pitchdnn: PitchDnnState,
        pub analysis_mem: [f32; 160],
        pub mem_preemph: f32,
        pub prev_if: [[f32; 2]; 30],
        pub if_features: [f32; 88],
        pub xcorr_features: [f32; 224],
        pub dnn_pitch: f32,
        pub pitch_mem: [f32; 16],
        pub pitch_filt: f32,
        pub exc_buf: [f32; 576],
        pub lp_buf: [f32; 576],
        pub lp_mem: [f32; 4],
        pub lpc: [f32; 16],
        pub features: [f32; 36],
        pub sig_mem: [f32; 16],
        pub burg_cepstrum: [f32; 36],
    }

    #[repr(C)]
    pub struct FarganState {
        pub model: CFARGAN,
        pub arch: c_int,
        pub cont_initialized: c_int,
        pub deemph_mem: f32,
        pub pitch_buf: [f32; 256],
        pub cond_conv1_state: [f32; 128],
        pub fwc0_mem: [f32; 328],
        pub gru1_state: [f32; 160],
        pub gru2_state: [f32; 128],
        pub gru3_state: [f32; 128],
        pub last_period: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct PlcNet {
        pub gru1_state: [f32; 192],
        pub gru2_state: [f32; 192],
    }

    #[repr(C)]
    pub struct PlcState {
        pub model: CPLCModel,
        pub fargan: FarganState,
        pub enc: EncState,
        pub loaded: c_int,
        pub arch: c_int,
        pub fec: [[f32; 20]; 104],
        pub analysis_gap: c_int,
        pub fec_read_pos: c_int,
        pub fec_fill_pos: c_int,
        pub fec_skip: c_int,
        pub analysis_pos: c_int,
        pub predict_pos: c_int,
        pub pcm: [f32; 2400],
        pub blend: c_int,
        pub features: [f32; 36],
        pub cont_features: [f32; 100],
        pub loss_count: c_int,
        pub plc_net: PlcNet,
        pub plc_bak: [PlcNet; 2],
    }

    unsafe extern "C" {
        pub fn opus_select_arch() -> c_int;
        pub fn lpcnet_encoder_get_size() -> c_int;
        pub fn lpcnet_encoder_init(st: *mut EncState) -> c_int;
        pub fn lpcnet_compute_single_frame_features_float(
            st: *mut EncState,
            pcm: *const f32,
            features: *mut f32,
            arch: c_int,
        ) -> c_int;
        pub fn pitchdnn_init(st: *mut PitchDnnState);
        pub fn compute_pitchdnn(
            st: *mut PitchDnnState,
            if_features: *const f32,
            xcorr_features: *const f32,
            arch: c_int,
        ) -> f32;
        pub fn fargan_init(st: *mut FarganState);
        pub fn fargan_cont(st: *mut FarganState, pcm0: *const f32, features0: *const f32);
        pub fn fargan_synthesize(st: *mut FarganState, pcm: *mut f32, features: *const f32);
        pub fn fargan_synthesize_int(st: *mut FarganState, pcm: *mut i16, features: *const f32);
        pub fn lpcnet_plc_init(st: *mut PlcState) -> c_int;
        pub fn lpcnet_plc_reset(st: *mut PlcState);
        pub fn lpcnet_plc_update(st: *mut PlcState, pcm: *mut i16) -> c_int;
        pub fn lpcnet_plc_conceal(st: *mut PlcState, pcm: *mut i16) -> c_int;
        pub fn lpcnet_plc_fec_add(st: *mut PlcState, features: *const f32);
        pub fn lpcnet_plc_fec_clear(st: *mut PlcState);
        pub fn update_plc_state(
            st: *mut PlcState,
            decode_mem: *mut *mut f32,
            plc_preemphasis_mem: *mut f32,
            cc: c_int,
        );
    }
}

/// A zeroed C state on the heap (the C structs hold only floats, integers
/// and pointers, so all zeros is a valid value).
fn zeroed<T>() -> Box<T> {
    // SAFETY: see above; the C init functions then fill the structs.
    unsafe { Box::<T>::new_zeroed().assume_init() }
}

fn c_arch() -> c_int {
    // Makes sure the static libopus is linked, then asks it for its arch.
    let _ = unsafe { audiopus_sys::opus_get_version_string() };
    unsafe { c::opus_select_arch() }
}

fn c_enc() -> Box<c::EncState> {
    let mut st = zeroed::<c::EncState>();
    unsafe { c::lpcnet_encoder_init(&mut *st) };
    st
}

fn c_fargan() -> Box<c::FarganState> {
    let mut st = zeroed::<c::FarganState>();
    unsafe { c::fargan_init(&mut *st) };
    st
}

fn c_plc() -> Box<c::PlcState> {
    let mut st = zeroed::<c::PlcState>();
    assert_eq!(unsafe { c::lpcnet_plc_init(&mut *st) }, 0);
    assert_eq!(st.loaded, 1);
    st
}

fn models() -> &'static DeepPlcModels<'static> {
    builtin::models().deep_plc.as_ref().expect("the built-in deep PLC models")
}

fn o2_mode() -> bool {
    std::env::var_os("OPUS1_DNN_REF_O2").is_some()
}

/// The variants to compare: all that this processor runs, or, when
/// measuring against the `-O2` build, the one opus1 selects.
fn archs() -> Vec<DnnArch> {
    if o2_mode() { vec![arch::detect()] } else { arch::available().to_vec() }
}

/// Panics with the first index where the bits differ.
#[track_caller]
fn same(what: &str, rust: &[f32], c: &[f32]) {
    assert_eq!(rust.len(), c.len(), "{what}: lengths");
    if let Some(i) = rust.iter().zip(c).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: index {i} differs: opus1 {} ({:#x}), libopus {} ({:#x})",
            rust[i], rust[i].to_bits(), c[i], c[i].to_bits());
    }
}

/// The speech signals: the fixture and every `step`-th corpus file (at
/// most `max`), at 16 kHz.
fn signals(step: usize, max: usize) -> Vec<(String, Vec<i16>)> {
    let mut v = vec![("answer_16k".to_string(), fixture())];
    for p in tsp_subset(step, max) {
        v.push((short_name(&p), tsp_16k(&p)));
    }
    v
}

fn frame_f32(s: &[i16]) -> [f32; FRAME] {
    let mut x = [0.0f32; FRAME];
    for (d, &v) in x.iter_mut().zip(s) {
        *d = v as f32;
    }
    x
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

#[test]
fn c_struct_layouts_match_libopus() {
    let _ = c_arch();
    assert_eq!(std::mem::size_of::<c::PitchDnnState>(), 31_440);
    assert_eq!(std::mem::size_of::<c::EncState>(), unsafe { c::lpcnet_encoder_get_size() } as usize);
    assert_eq!(std::mem::size_of::<c::EncState>(), 38_688);
    assert_eq!(std::mem::size_of::<c::FarganState>(), 5_808);
    assert_eq!(std::mem::size_of::<c::PlcState>(), 67_992);
}

// ---------------------------------------------------------------------------
// Feature analysis and PitchDNN
// ---------------------------------------------------------------------------

/// The 36 features of every frame, the PitchDNN inputs and PitchDNN's
/// state, with one state carried over the whole signal and a reset in the
/// middle.
#[test]
fn features_match_libopus() {
    let arch_c = c_arch();
    let m = models();
    let archs = archs();
    let o2 = o2_mode();
    let (mut frames, mut differ, mut max_diff) = (0usize, 0usize, 0.0f32);
    for (name, s) in signals(4, 400) {
        let mut cst = c_enc();
        let mut rst: Vec<LpcnetEncState> = archs.iter().map(|_| LpcnetEncState::new()).collect();
        let n = s.len() / FRAME;
        for k in 0..n {
            if k == n / 2 {
                unsafe { c::lpcnet_encoder_init(&mut *cst) };
                for r in rst.iter_mut() {
                    r.reset();
                }
            }
            let x = frame_f32(&s[k * FRAME..]);
            let mut cf = [0.0f32; NB_TOTAL_FEATURES];
            unsafe { c::lpcnet_compute_single_frame_features_float(&mut *cst, x.as_ptr(), cf.as_mut_ptr(), arch_c) };
            frames += 1;
            for (r, &a) in rst.iter_mut().zip(&archs) {
                let mut rf = [0.0f32; NB_TOTAL_FEATURES];
                compute_single_frame_features(r, &x, &mut rf, &m.pitchdnn, a);
                if o2 {
                    let d = rf.iter().zip(&cf).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                    if d > 0.0 {
                        differ += 1;
                        max_diff = max_diff.max(d);
                    }
                    continue;
                }
                let what = format!("{name} frame {k} {a:?}");
                same(&format!("{what} features"), &rf, &cf);
                same(&format!("{what} if_features"), r.if_features(), &cst.if_features);
                same(&format!("{what} xcorr_features"), r.xcorr_features(), &cst.xcorr_features);
                same(&format!("{what} pitchdnn gru"), r.pitchdnn().gru_state(), &cst.pitchdnn.gru_state);
                same(&format!("{what} xcorr_mem2"), r.pitchdnn().xcorr_mem2(), &cst.pitchdnn.xcorr_mem2[..1808]);
            }
        }
    }
    eprintln!("features: {frames} frames compared");
    if o2 {
        eprintln!("-O2 build: {differ} of {frames} frames differ, largest difference {max_diff}");
    }
}

/// PitchDNN alone, fed with the inputs of libopus's analysis, and with
/// inputs out of the usual range.
#[test]
fn pitchdnn_matches_libopus() {
    let arch_c = c_arch();
    let m = models();
    let archs = archs();
    let o2 = o2_mode();
    let mut cenc = c_enc();
    let mut cst = zeroed::<c::PitchDnnState>();
    unsafe { c::pitchdnn_init(&mut *cst) };
    let mut rst: Vec<PitchDnnState> = archs.iter().map(|_| PitchDnnState::new()).collect();
    let mut rng = Lcg(77);
    let mut count = 0;
    let mut differ = 0;
    for (name, s) in signals(16, 100) {
        for k in 0..s.len() / FRAME {
            let x = frame_f32(&s[k * FRAME..]);
            let mut cf = [0.0f32; NB_TOTAL_FEATURES];
            unsafe {
                c::lpcnet_compute_single_frame_features_float(&mut *cenc, x.as_ptr(), cf.as_mut_ptr(), arch_c)
            };
            let mut iff: [f32; PITCH_IF_FEATURES] = cenc.if_features;
            let mut xcf: [f32; NB_XCORR_FEATURES] = cenc.xcorr_features;
            // Every 50th frame, inputs out of the usual range.
            if k % 50 == 49 {
                for v in iff.iter_mut().chain(xcf.iter_mut()) {
                    *v = 4.0 * rng.uniform();
                }
            }
            let cp = unsafe { c::compute_pitchdnn(&mut *cst, iff.as_ptr(), xcf.as_ptr(), arch_c) };
            count += 1;
            for (r, &a) in rst.iter_mut().zip(&archs) {
                let rp = compute_pitchdnn(r, &iff, &xcf, &m.pitchdnn, a);
                if o2 {
                    differ += (rp.to_bits() != cp.to_bits()) as usize;
                    continue;
                }
                let what = format!("{name} frame {k} {a:?}");
                same(&format!("{what} pitch"), &[rp], &[cp]);
                same(&format!("{what} gru"), r.gru_state(), &cst.gru_state);
                same(&format!("{what} xcorr_mem1"), r.xcorr_mem1(), &cst.xcorr_mem1);
                same(&format!("{what} xcorr_mem2"), r.xcorr_mem2(), &cst.xcorr_mem2[..1808]);
            }
        }
    }
    eprintln!("pitchdnn: {count} frames compared");
    if o2 {
        eprintln!("-O2 build: {differ} of {count} pitch values differ");
    }
}

// ---------------------------------------------------------------------------
// FARGAN
// ---------------------------------------------------------------------------

fn same_fargan(what: &str, r: &FarganState, c: &c::FarganState) {
    same(&format!("{what} pitch_buf"), r.pitch_buf(), &c.pitch_buf);
    same(&format!("{what} cond_conv1_state"), r.cond_conv1_state(), &c.cond_conv1_state);
    same(&format!("{what} fwc0_mem"), r.fwc0_mem(), &c.fwc0_mem[..164]);
    let (g1, g2, g3) = r.gru_states();
    same(&format!("{what} gru1"), g1, &c.gru1_state);
    same(&format!("{what} gru2"), g2, &c.gru2_state);
    same(&format!("{what} gru3"), g3, &c.gru3_state);
    same(&format!("{what} deemph_mem"), &[r.deemph_mem()], &[c.deemph_mem]);
    assert_eq!(r.last_period(), c.last_period, "{what} last_period");
    assert_eq!(r.cont_initialized(), c.cont_initialized != 0, "{what} cont_initialized");
}

/// FARGAN over whole signals, on its own output: `fargan_cont` at the
/// start and every 150 frames (on the real past audio and features), and
/// `fargan_synthesize` (every third call `fargan_synthesize_int`) for
/// every frame in between, with the features of libopus's analysis.
#[test]
fn fargan_matches_libopus() {
    let arch_c = c_arch();
    let m = models();
    let archs = archs();
    let o2 = o2_mode();
    let mut calls = 0usize;
    let (mut sig, mut noise) = (0.0f64, 0.0f64);
    for (name, s) in signals(8, 200) {
        let mut cenc = c_enc();
        let mut feats: Vec<[f32; NB_TOTAL_FEATURES]> = Vec::new();
        for k in 0..s.len() / FRAME {
            let x = frame_f32(&s[k * FRAME..]);
            let mut cf = [0.0f32; NB_TOTAL_FEATURES];
            unsafe {
                c::lpcnet_compute_single_frame_features_float(&mut *cenc, x.as_ptr(), cf.as_mut_ptr(), arch_c)
            };
            feats.push(cf);
        }
        let mut cst = c_fargan();
        let mut rst: Vec<FarganState> = archs.iter().map(|_| FarganState::new()).collect();
        for k in 2..feats.len() {
            let what = format!("{name} frame {k}");
            if k >= CONT_VECTORS && (k - CONT_VECTORS).is_multiple_of(150) {
                // The last 20 ms of real audio and the last five vectors.
                let mut pcm0 = [0.0f32; FARGAN_CONT_SAMPLES];
                for (d, &v) in pcm0.iter_mut().zip(&s[k * FRAME - FARGAN_CONT_SAMPLES..]) {
                    *d = v as f32 / 32768.0;
                }
                let mut f0 = [0.0f32; CONT_VECTORS * NB_FEATURES];
                for (j, v) in f0.as_chunks_mut::<NB_FEATURES>().0.iter_mut().enumerate() {
                    v.copy_from_slice(&feats[k - CONT_VECTORS + j][..NB_FEATURES]);
                }
                unsafe { c::fargan_cont(&mut *cst, pcm0.as_ptr(), f0.as_ptr()) };
                for (r, &a) in rst.iter_mut().zip(&archs) {
                    fargan::fargan_cont(r, &pcm0, &f0, &m.fargan, a);
                    if !o2 {
                        same_fargan(&format!("{what} cont {a:?}"), r, &cst);
                    }
                }
            }
            if cst.cont_initialized == 0 {
                continue;
            }
            let mut f = [0.0f32; NB_FEATURES];
            f.copy_from_slice(&feats[k][..NB_FEATURES]);
            calls += 1;
            if k % 3 == 0 {
                let mut cp = [0i16; FRAME];
                unsafe { c::fargan_synthesize_int(&mut *cst, cp.as_mut_ptr(), f.as_ptr()) };
                for (r, &a) in rst.iter_mut().zip(&archs) {
                    let mut rp = [0i16; FRAME];
                    fargan::fargan_synthesize_int(r, &mut rp, &f, &m.fargan, a);
                    if o2 {
                        for (&x, &y) in cp.iter().zip(&rp) {
                            sig += (x as f64).powi(2);
                            noise += (x as f64 - y as f64).powi(2);
                        }
                        continue;
                    }
                    assert_eq!(rp, cp, "{what} {a:?} int output");
                    same_fargan(&format!("{what} {a:?}"), r, &cst);
                }
            } else {
                let mut cp = [0.0f32; FARGAN_FRAME_SIZE];
                unsafe { c::fargan_synthesize(&mut *cst, cp.as_mut_ptr(), f.as_ptr()) };
                for (r, &a) in rst.iter_mut().zip(&archs) {
                    let mut rp = [0.0f32; FARGAN_FRAME_SIZE];
                    fargan::fargan_synthesize(r, &mut rp, &f, &m.fargan, a);
                    if o2 {
                        for (&x, &y) in cp.iter().zip(&rp) {
                            sig += (x as f64 * 32768.0).powi(2);
                            noise += ((x as f64 - y as f64) * 32768.0).powi(2);
                        }
                        continue;
                    }
                    same(&format!("{what} {a:?} output"), &rp, &cp);
                    same_fargan(&format!("{what} {a:?}"), r, &cst);
                }
            }
        }
    }
    eprintln!("fargan: {calls} frames compared");
    if o2 {
        eprintln!("-O2 build: SNR over FARGAN's output {:.1} dB", 10.0 * (sig / noise.max(1e-30)).log10());
    }
}

// ---------------------------------------------------------------------------
// The state machine
// ---------------------------------------------------------------------------

/// One step of a test sequence.
#[derive(Clone, Copy, Debug)]
enum Ev {
    /// The next frame of the signal arrives.
    Good,
    /// The next frame of the signal is lost.
    Lost,
    /// `reset` (`OPUS_RESET_STATE`).
    Reset,
    /// `fec_add` with the features of the frame `back` frames before the
    /// current one (from libopus's analysis), and the pitch replaced by
    /// `pitch` when given.
    FecAdd { back: usize, pitch: Option<f32> },
    /// `fec_add(None)`.
    FecSkip,
    /// `fec_clear`.
    FecClear,
    /// `update_from_celt` with `channels` channels of the 48 kHz signal
    /// that ends at the current frame.
    Celt { channels: usize },
}

fn same_plc(what: &str, r: &DeepPlc, c: &c::PlcState) {
    let (apos, ppos, gap, loss) = r.positions();
    assert_eq!(
        (apos, ppos, gap, loss, r.blend()),
        (c.analysis_pos, c.predict_pos, c.analysis_gap != 0, c.loss_count, c.blend != 0),
        "{what}: positions"
    );
    assert_eq!(r.fec_cursor(), (c.fec_read_pos, c.fec_skip), "{what}: fec cursor");
    assert_eq!(r.fec_fill_pos(), c.fec_fill_pos, "{what}: fec_fill_pos");
    same(&format!("{what} history"), r.history(), &c.pcm);
    same(&format!("{what} features"), r.features(), &c.features);
    same(&format!("{what} cont_features"), r.cont_features(), &c.cont_features);
    let (net, bak) = r.plc_states();
    same(&format!("{what} plc_net gru1"), &net.gru1_state, &c.plc_net.gru1_state);
    same(&format!("{what} plc_net gru2"), &net.gru2_state, &c.plc_net.gru2_state);
    for (k, (rb, cb)) in bak.iter().zip(&c.plc_bak).enumerate() {
        same(&format!("{what} plc_bak[{k}] gru1"), &rb.gru1_state, &cb.gru1_state);
        same(&format!("{what} plc_bak[{k}] gru2"), &rb.gru2_state, &cb.gru2_state);
    }
    same(&format!("{what} analysis features"), r.analysis().features(), &c.enc.features);
    same_fargan(&format!("{what} fargan"), r.fargan(), &c.fargan);
}

/// What `run_sequence` saw.
#[derive(Default)]
struct Stats {
    /// Concealed frames.
    lost: usize,
    /// The energy of libopus's concealed frames.
    sig: f64,
    /// The energy of the difference between opus1 and libopus.
    noise: f64,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.lost += o.lost;
        self.sig += o.sig;
        self.noise += o.noise;
    }

    fn snr_db(&self) -> f64 {
        10.0 * (self.sig / self.noise.max(1e-30)).log10()
    }
}

/// Runs `events` on `s` through libopus and through opus1 (each variant),
/// and compares every concealed frame and, unless measuring, the states
/// after every step.
fn run_sequence(name: &str, s: &[i16], events: &[Ev], feats: &[[f32; NB_TOTAL_FEATURES]]) -> Stats {
    let m = models();
    let archs = archs();
    let o2 = o2_mode();
    let mut cst = c_plc();
    let mut rst: Vec<DeepPlc> = archs.iter().map(|_| DeepPlc::new()).collect();
    let frames = s.len() / FRAME;
    let mut k = 0usize;
    let mut st = Stats::default();
    for (step, ev) in events.iter().enumerate() {
        if k >= frames {
            break;
        }
        let what = format!("{name} step {step} frame {k} {ev:?}");
        match *ev {
            Ev::Good => {
                let mut pcm = [0i16; FRAME];
                pcm.copy_from_slice(&s[k * FRAME..(k + 1) * FRAME]);
                unsafe { c::lpcnet_plc_update(&mut *cst, pcm.as_mut_ptr()) };
                for r in rst.iter_mut() {
                    r.update(&pcm);
                }
                k += 1;
            }
            Ev::Lost => {
                let mut cp = [0i16; FRAME];
                unsafe { c::lpcnet_plc_conceal(&mut *cst, cp.as_mut_ptr()) };
                st.lost += 1;
                for (r, &a) in rst.iter_mut().zip(&archs) {
                    let mut rp = [0i16; FRAME];
                    r.conceal(m, a, &mut rp);
                    for (&x, &y) in cp.iter().zip(&rp) {
                        st.sig += (x as f64).powi(2);
                        st.noise += (x as f64 - y as f64).powi(2);
                    }
                    if !o2 {
                        assert_eq!(rp, cp, "{what} {a:?}: concealed frame");
                    }
                }
                k += 1;
            }
            Ev::Reset => {
                unsafe { c::lpcnet_plc_reset(&mut *cst) };
                for r in rst.iter_mut() {
                    r.reset();
                }
            }
            Ev::FecAdd { back, pitch } => {
                let mut f = [0.0f32; NB_FEATURES];
                f.copy_from_slice(&feats[k.saturating_sub(back)][..NB_FEATURES]);
                if let Some(p) = pitch {
                    f[18] = p;
                }
                if (cst.fec_fill_pos as usize) < PLC_MAX_FEC {
                    unsafe { c::lpcnet_plc_fec_add(&mut *cst, f.as_ptr()) };
                }
                for r in rst.iter_mut() {
                    r.fec_add(Some(&f));
                }
            }
            Ev::FecSkip => {
                unsafe { c::lpcnet_plc_fec_add(&mut *cst, std::ptr::null()) };
                for r in rst.iter_mut() {
                    r.fec_add(None);
                }
            }
            Ev::FecClear => {
                unsafe { c::lpcnet_plc_fec_clear(&mut *cst) };
                for r in rst.iter_mut() {
                    r.fec_clear();
                }
            }
            Ev::Celt { channels } => {
                // The 48 kHz history: the signal up to the current frame,
                // repeated 3 times per sample, the second channel a scaled
                // and shifted copy.
                let end = k * FRAME;
                let sample = |back: usize| if end > back { s[end - 1 - back] as f32 } else { 0.0 };
                let mem: Vec<Vec<f32>> = [0usize, 7]
                    .iter()
                    .zip([0.3f32, 0.18])
                    .map(|(&shift, gain)| {
                        (0..CELT_DECODE_BUFFER_SIZE)
                            .map(|i| gain * sample((CELT_DECODE_BUFFER_SIZE - 1 - i) / 3 + shift))
                            .collect()
                    })
                    .collect();
                let mut c_mem = mem.clone();
                let mut ptrs = [c_mem[0].as_mut_ptr(), c_mem[1].as_mut_ptr()];
                let mut c_pre = 0.0f32;
                unsafe { c::update_plc_state(&mut *cst, ptrs.as_mut_ptr(), &mut c_pre, channels as c_int) };
                for r in rst.iter_mut() {
                    let slices: Vec<&[f32]> = mem[..channels].iter().map(|v| v.as_slice()).collect();
                    let mut pre = 0.0f32;
                    r.update_from_celt(&slices, &mut pre);
                    if !o2 {
                        same(&format!("{what} preemphasis memory"), &[pre], &[c_pre]);
                    }
                }
            }
        }
        if !o2 {
            for (r, a) in rst.iter().zip(&archs) {
                same_plc(&format!("{what} {a:?}"), r, &cst);
            }
        }
    }
    st
}

/// The features of every frame, from libopus's analysis (for the DRED
/// queue).
fn c_features(s: &[i16]) -> Vec<[f32; NB_TOTAL_FEATURES]> {
    let arch_c = c_arch();
    let mut cenc = c_enc();
    (0..s.len() / FRAME)
        .map(|k| {
            let x = frame_f32(&s[k * FRAME..]);
            let mut cf = [0.0f32; NB_TOTAL_FEATURES];
            unsafe {
                c::lpcnet_compute_single_frame_features_float(&mut *cenc, x.as_ptr(), cf.as_mut_ptr(), arch_c)
            };
            cf
        })
        .collect()
}

fn push(ev: &mut Vec<Ev>, e: Ev, n: usize) {
    ev.extend(std::iter::repeat_n(e, n));
}

/// The scripted sequence of `deep_plc.md` section 8.3 item 5 and the
/// review: losses at the start, single losses after 1, 4, 14, 15, 16 and
/// 40 good frames, bursts of 2, 3, 6, 10, 11, 30 and 100 frames (1 s),
/// alternating losses, resets in the middle and right after a burst, the
/// DRED queue (with a pitch that gives FARGAN the period 0) and CELT
/// updates.
fn scripted() -> Vec<Ev> {
    use Ev::*;
    let mut ev = Vec::new();
    push(&mut ev, Lost, 3);
    for good in [1, 4, 14, 15, 16, 40] {
        push(&mut ev, Good, good);
        push(&mut ev, Lost, 1);
    }
    for burst in [2, 3, 6, 10, 11, 30, 100] {
        push(&mut ev, Good, 20);
        push(&mut ev, Lost, burst);
    }
    for _ in 0..20 {
        ev.push(Good);
        ev.push(Lost);
    }
    push(&mut ev, Good, 30);
    ev.push(Reset);
    push(&mut ev, Good, 5);
    push(&mut ev, Lost, 3);
    push(&mut ev, Good, 20);
    push(&mut ev, Lost, 5);
    ev.push(Reset);
    push(&mut ev, Lost, 2);
    push(&mut ev, Good, 20);
    push(&mut ev, Lost, 3);
    // The DRED queue.
    push(&mut ev, Good, 20);
    for back in [3, 2, 1] {
        ev.push(FecAdd { back, pitch: None });
    }
    ev.push(FecSkip);
    ev.push(FecSkip);
    ev.push(FecAdd { back: 1, pitch: None });
    push(&mut ev, Lost, 8);
    ev.push(FecClear);
    push(&mut ev, Good, 10);
    push(&mut ev, Lost, 2);
    // A DRED pitch of 8 gives the period 0 (libopus then reads past the
    // pitch buffer).
    push(&mut ev, Good, 16);
    ev.push(FecAdd { back: 1, pitch: Some(8.0) });
    ev.push(FecAdd { back: 1, pitch: Some(8.0) });
    ev.push(FecAdd { back: 1, pitch: Some(-3.0) });
    push(&mut ev, Lost, 4);
    ev.push(FecClear);
    // CELT updates: before a loss, and in the middle of one.
    push(&mut ev, Good, 20);
    ev.push(Celt { channels: 1 });
    push(&mut ev, Lost, 4);
    push(&mut ev, Good, 3);
    ev.push(Celt { channels: 2 });
    push(&mut ev, Lost, 3);
    ev.push(Celt { channels: 1 });
    push(&mut ev, Lost, 2);
    push(&mut ev, Good, 200);
    ev
}

/// A random sequence: losses in bursts (Gilbert model: a loss starts with
/// probability 8 %, continues with 60 %), now and then a reset, DRED
/// vectors or a CELT update.
fn random_events(rng: &mut Lcg, n: usize) -> Vec<Ev> {
    use Ev::*;
    let mut ev = Vec::with_capacity(n + 64);
    let mut lost = false;
    let mut frames = 0;
    while frames < n {
        let r = rng.below(1000);
        lost = if lost { r < 600 } else { r < 80 };
        match rng.below(1000) {
            0..=2 => ev.push(Reset),
            3..=6 if !lost => ev.push(Celt { channels: 1 + rng.below(2) as usize }),
            7..=9 => {
                ev.push(FecAdd { back: 1 + rng.below(5) as usize, pitch: None });
                ev.push(FecAdd { back: 1, pitch: None });
            }
            10 => ev.push(FecSkip),
            11..=13 => ev.push(FecClear),
            _ => {}
        }
        ev.push(if lost { Lost } else { Good });
        frames += 1;
    }
    ev
}

#[test]
fn plc_scripted_sequence_matches_libopus() {
    let s = fixture();
    let feats = c_features(&s);
    let st = run_sequence("answer_16k scripted", &s, &scripted(), &feats);
    eprintln!("plc scripted: {} concealed frames", st.lost);
    // The concealment makes up speech, not silence.
    assert!(st.sig > 1e6 * st.lost as f64, "energy {} over {} frames", st.sig, st.lost);
    if o2_mode() {
        eprintln!("-O2 build, scripted: SNR over concealed frames {:.1} dB", st.snr_db());
    }
}

#[test]
fn plc_random_sequences_match_libopus() {
    let mut rng = Lcg(2026);
    let mut total = Stats::default();
    let mut runs = 0;
    for (name, s) in signals(8, 200) {
        let feats = c_features(&s);
        let events = random_events(&mut rng, s.len() / FRAME);
        total.add(&run_sequence(&name, &s, &events, &feats));
        runs += 1;
    }
    eprintln!("plc random: {runs} signals, {} concealed frames", total.lost);
    if o2_mode() {
        let snr = total.snr_db();
        eprintln!("-O2 build, random: SNR over concealed frames {snr:.1} dB");
        assert!(snr >= 8.0, "SNR {snr:.1} dB against the -O2 build");
    }
}

/// `update_from_celt` alone, on noise at several levels (full scale and
/// beyond, where the 16-bit clamp acts), mono and stereo.
#[test]
fn update_from_celt_matches_libopus() {
    if o2_mode() {
        return;
    }
    let _ = c_arch();
    let mut rng = Lcg(5);
    let mut cst = c_plc();
    let mut r = DeepPlc::new();
    for round in 0..40 {
        let channels = 1 + round % 2;
        let level = [1.0f32, 3000.0, 40000.0, 1e6][round % 4];
        let mut mem = vec![vec![0.0f32; CELT_DECODE_BUFFER_SIZE]; 2];
        for ch in mem.iter_mut() {
            for v in ch.iter_mut() {
                *v = level * rng.uniform();
            }
        }
        let mut c_mem = mem.clone();
        let mut ptrs = [c_mem[0].as_mut_ptr(), c_mem[1].as_mut_ptr()];
        let mut c_pre = 0.0f32;
        unsafe { c::update_plc_state(&mut *cst, ptrs.as_mut_ptr(), &mut c_pre, channels as c_int) };
        let slices: Vec<&[f32]> = mem[..channels].iter().map(|v| v.as_slice()).collect();
        let mut pre = 0.0f32;
        r.update_from_celt(&slices, &mut pre);
        same(&format!("round {round} preemphasis memory"), &[pre], &[c_pre]);
        same_plc(&format!("round {round}"), &r, &cst);
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// The cost of one call of `r` and of `c`, in microseconds: the medians
/// over `batches` batches of `per_batch` calls, the batches of the two
/// sides taking turns, so both see the same processor state.
fn time_pair(batches: usize, per_batch: usize, mut r: impl FnMut(), mut c: impl FnMut()) -> (f64, f64) {
    let batch = |f: &mut dyn FnMut()| {
        let start = std::time::Instant::now();
        for _ in 0..per_batch {
            f();
        }
        start.elapsed().as_secs_f64() * 1e6 / per_batch as f64
    };
    // Warm up.
    batch(&mut r);
    batch(&mut c);
    let (mut tr, mut tc) = (Vec::new(), Vec::new());
    for _ in 0..batches {
        tr.push(batch(&mut r));
        tc.push(batch(&mut c));
    }
    (median(&mut tr), median(&mut tc))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "prints timings; run with --ignored --nocapture"]
fn timing() {
    let arch_c = c_arch();
    let a = arch::detect();
    let m = models();
    let s = fixture();
    let feats = c_features(&s);
    let mut pcm = [0i16; FRAME];
    pcm.copy_from_slice(&s[400 * FRAME..401 * FRAME]);
    println!("deep PLC per call, median, microseconds (opus1 {a:?}, libopus arch {arch_c})");
    println!("{:<46} {:>10} {:>10}", "call", "opus1", "libopus");
    let row = |name: &str, (r, c): (f64, f64)| println!("{name:<46} {r:>10.2} {c:>10.2}");

    // Feature analysis, on a stretch of speech.
    let mut renc = LpcnetEncState::new();
    let mut cenc = c_enc();
    let frames: Vec<[f32; FRAME]> = (0..64).map(|k| frame_f32(&s[(300 + k) * FRAME..])).collect();
    let (mut ir, mut ic) = (0usize, 0usize);
    let mut out = [0.0f32; NB_TOTAL_FEATURES];
    let mut out_c = [0.0f32; NB_TOTAL_FEATURES];
    row(
        "feature analysis (with PitchDNN)",
        time_pair(
            60,
            32,
            || {
                compute_single_frame_features(&mut renc, &frames[ir % 64], &mut out, &m.pitchdnn, a);
                ir += 1;
            },
            || {
                unsafe {
                    c::lpcnet_compute_single_frame_features_float(
                        &mut *cenc,
                        frames[ic % 64].as_ptr(),
                        out_c.as_mut_ptr(),
                        arch_c,
                    )
                };
                ic += 1;
            },
        ),
    );

    // PitchDNN alone.
    let mut rp = PitchDnnState::new();
    let mut cp = zeroed::<c::PitchDnnState>();
    unsafe { c::pitchdnn_init(&mut *cp) };
    let (iff, xcf) = (cenc.if_features, cenc.xcorr_features);
    row(
        "PitchDNN",
        time_pair(
            60,
            32,
            || {
                std::hint::black_box(compute_pitchdnn(&mut rp, &iff, &xcf, &m.pitchdnn, a));
            },
            || {
                std::hint::black_box(unsafe { c::compute_pitchdnn(&mut *cp, iff.as_ptr(), xcf.as_ptr(), arch_c) });
            },
        ),
    );

    // FARGAN.
    let mut rf = FarganState::new();
    let mut cf = c_fargan();
    let mut pcm0 = [0.0f32; FARGAN_CONT_SAMPLES];
    for (d, &v) in pcm0.iter_mut().zip(&s[400 * FRAME - FARGAN_CONT_SAMPLES..]) {
        *d = v as f32 / 32768.0;
    }
    let mut f0 = [0.0f32; CONT_VECTORS * NB_FEATURES];
    for (j, v) in f0.as_chunks_mut::<NB_FEATURES>().0.iter_mut().enumerate() {
        v.copy_from_slice(&feats[395 + j][..NB_FEATURES]);
    }
    row(
        "fargan_cont",
        time_pair(
            40,
            8,
            || fargan::fargan_cont(&mut rf, &pcm0, &f0, &m.fargan, a),
            || unsafe { c::fargan_cont(&mut *cf, pcm0.as_ptr(), f0.as_ptr()) },
        ),
    );
    let mut f = [0.0f32; NB_FEATURES];
    f.copy_from_slice(&feats[400][..NB_FEATURES]);
    let mut o = [0.0f32; FARGAN_FRAME_SIZE];
    let mut oc = [0.0f32; FARGAN_FRAME_SIZE];
    row(
        "fargan_synthesize (10 ms)",
        time_pair(
            40,
            8,
            || fargan::fargan_synthesize(&mut rf, &mut o, &f, &m.fargan, a),
            || unsafe { c::fargan_synthesize(&mut *cf, oc.as_mut_ptr(), f.as_ptr()) },
        ),
    );

    // The state machine.
    let mut rs = DeepPlc::new();
    let mut cs = c_plc();
    let mut pcm_c = pcm;
    row(
        "update (10 ms)",
        time_pair(
            60,
            200,
            || rs.update(&pcm),
            || unsafe {
                c::lpcnet_plc_update(&mut *cs, pcm_c.as_mut_ptr());
            },
        ),
    );
    // First and later lost frames, after 40 good frames.
    let (mut r1, mut c1, mut r2, mut c2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for round in 0..80 {
        for k in 0..40 {
            let mut p = [0i16; FRAME];
            p.copy_from_slice(&s[(round * 7 + k) * FRAME..(round * 7 + k + 1) * FRAME]);
            rs.update(&p);
            unsafe { c::lpcnet_plc_update(&mut *cs, p.as_mut_ptr()) };
        }
        for j in 0..5 {
            let mut o = [0i16; FRAME];
            let t = std::time::Instant::now();
            rs.conceal(m, a, &mut o);
            let tr = t.elapsed().as_secs_f64() * 1e6;
            let t = std::time::Instant::now();
            unsafe { c::lpcnet_plc_conceal(&mut *cs, o.as_mut_ptr()) };
            let tc = t.elapsed().as_secs_f64() * 1e6;
            if j == 0 {
                r1.push(tr);
                c1.push(tc);
            } else {
                r2.push(tr);
                c2.push(tc);
            }
        }
    }
    row("conceal, first lost frame after 40 good", (median(&mut r1), median(&mut c1)));
    row("conceal, each further lost frame", (median(&mut r2), median(&mut c2)));

    // The CELT downsampler.
    let mem = vec![vec![100.0f32; CELT_DECODE_BUFFER_SIZE]; 2];
    let mut c_mem = mem.clone();
    let mut ptrs = [c_mem[0].as_mut_ptr(), c_mem[1].as_mut_ptr()];
    let (mut pre, mut pre_c) = (0.0f32, 0.0f32);
    let slices: Vec<&[f32]> = vec![&mem[0], &mem[1]];
    row(
        "update_from_celt (stereo, 40 ms)",
        time_pair(
            60,
            16,
            || rs.update_from_celt(&slices, &mut pre),
            || unsafe { c::update_plc_state(&mut *cs, ptrs.as_mut_ptr(), &mut pre_c, 2) },
        ),
    );
    println!(
        "size_of::<DeepPlc>() = {} bytes; libopus LPCNetPLCState = {} bytes",
        std::mem::size_of::<DeepPlc>(),
        std::mem::size_of::<c::PlcState>()
    );
    let _ = PLC_BUF_SIZE;
}
