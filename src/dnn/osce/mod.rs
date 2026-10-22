//! LACE, NoLACE and the bandwidth extension (BWE): the Opus Speech Coding
//! Enhancement (OSCE) of libopus 1.6.1 (`dnn/osce.c`, `osce_features.c`,
//! `nndsp.c`), compiled with the `osce` feature (and `osce-bwe` for the
//! bandwidth extension).
//!
//! # What runs where
//!
//! OSCE runs inside the SILK decoder, once per SILK frame and per coded
//! channel (mid and side separately):
//!
//! - `osce_enhance_frame` post-filters a decoded 20 ms frame at 16 kHz with
//!   LACE (decoder complexity 6) or NoLACE (7 and above). libopus calls it
//!   in `silk_decode_frame` (`silk/decode_frame.c:109-114`), after the
//!   update of the SILK output history and before the PLC update, with the
//!   decoded samples, the parameters of the frame (`OsceFrame`) and the bits
//!   the frame used. The SILK history keeps the samples before
//!   enhancement.
//! - `osce_bwe` replaces the SILK resampler from 16 to 48 kHz in BBWE mode
//!   (`silk/dec_API.c:391-418`).
//!
//! # Reset and bypass rules (libopus 1.6.1)
//!
//! - Only frames with `fs_khz == 16` and `nb_subfr == 4` are enhanced. Any
//!   other frame calls `osce_reset` with the current method and is left
//!   as it is.
//! - `osce_reset` clears the features and the state of the new method, and
//!   sets the bypass counter to 2: the first eligible frame after it comes
//!   out unenhanced (but the features and the model still run, to build
//!   their history), and the second is cross-faded over its first 10 ms
//!   from unenhanced to enhanced (`osce_cross_fade_10ms`).
//! - The SILK decoder calls `osce_reset(channel, method)`:
//!   in `silk_reset_decoder` with `OSCE_DEFAULT_METHOD`; before decoding a
//!   frame when the channel's method differs from the method the decoder
//!   complexity selects; and after concealing a lost frame (with the
//!   channel's method).
//! - With method NONE, or when the layers of the method are not loaded, the
//!   output is the input, with -32768 clipped to -32767.
//! - The BWE state is not touched by `osce_reset` or by `silk_reset_decoder`
//!   (so it survives `OPUS_RESET_STATE`). `silk_init_decoder` clears it to
//!   zero (`OsceBweChannel::new`); `osce_bwe_reset` sets it to the state
//!   libopus starts a BBWE stretch with.
//!
//! # Numerics
//!
//! Everything computes the bits of libopus 1.6.1 on 64-bit ARM built at
//! `-O2 -fno-vectorize -fno-slp-vectorize` (`target/perf/next/plan.md`),
//! on every target, apart from the double-precision `exp`, `log`, `sin`
//! and `cos` of the platform's C library (or of the `libm` crate without
//! `std`).

#[cfg(feature = "osce-bwe")]
pub mod bbwenet;
pub mod features;
pub mod lace;
mod math;
pub mod nndsp;
pub mod nolace;

use self::features::{OSCE_FEATURE_DIM, OsceFeatureState, osce_calculate_features};
use self::lace::{LaceState, lace_process_20ms_frame};
use self::math::float2int;
use self::nolace::{NolaceState, nolace_process_20ms_frame};
use super::DnnArch;
use crate::in_place::write_fields;
use super::holder::DecoderDnn;
use super::models::lace::LaceLayers;
use super::models::nolace::NolaceLayers;

#[cfg(feature = "osce-bwe")]
use self::bbwenet::{BbwenetState, bbwenet_process_frames};
#[cfg(feature = "osce-bwe")]
use self::features::{OSCE_BWE_FEATURE_DIM, OsceBweFeatureState, osce_bwe_calculate_features};
#[cfg(feature = "osce-bwe")]
use super::models::bbwenet::BbwenetLayers;

pub use self::features::{osce_bwe_cross_fade_10ms, osce_cross_fade_10ms};

/// `OSCE_BWE_OUTPUT_DELAY`: the delay of `osce_bwe` in samples at 48 kHz.
pub const OSCE_BWE_OUTPUT_DELAY: usize = 21;

/// The enhancement method of a SILK channel (libopus `OSCE_METHOD_*`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum OsceMethod {
    /// No enhancement (`OSCE_METHOD_NONE`, 0).
    None = 0,
    /// LACE (`OSCE_METHOD_LACE`, 1), decoder complexity 6.
    Lace = 1,
    /// NoLACE (`OSCE_METHOD_NOLACE`, 2), decoder complexity 7 and above.
    Nolace = 2,
}

/// `OSCE_DEFAULT_METHOD`: NoLACE, the method of a channel after
/// `silk_reset_decoder`.
pub const OSCE_DEFAULT_METHOD: OsceMethod = OsceMethod::Nolace;

impl OsceMethod {
    /// The method libopus selects for a decoder complexity
    /// (`src/opus_decoder.c:443-451`): NONE below 6, LACE at 6, NoLACE from
    /// 7.
    pub fn for_complexity(complexity: i32) -> OsceMethod {
        if complexity >= 7 {
            OsceMethod::Nolace
        } else if complexity >= 6 {
            OsceMethod::Lace
        } else {
            OsceMethod::None
        }
    }
}

/// The extended mode of the SILK output stage (libopus `OSCE_MODE_*`,
/// `dnn/osce.h:46-49`), which decides between the SILK resampler and the
/// bandwidth extension. `Unset` is the value 0 of a new decoder, which is
/// none of the libopus constants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum OsceExtMode {
    /// 0: the value before the first SILK frame.
    Unset = 0,
    /// `OSCE_MODE_SILK_ONLY` (1000).
    SilkOnly = 1000,
    /// `OSCE_MODE_HYBRID` (1001).
    Hybrid = 1001,
    /// `OSCE_MODE_CELT_ONLY` (1002).
    CeltOnly = 1002,
    /// `OSCE_MODE_SILK_BBWE` (1003): the bandwidth extension replaces the
    /// resampler.
    SilkBbwe = 1003,
}

/// What `osce_enhance_frame` reads from the SILK decoder for one frame:
/// the fields of `silk_decoder_state` and `silk_decoder_control` that libopus
/// reads, and the bit count. The decoded samples are passed separately.
#[derive(Clone, Copy, Debug)]
pub struct OsceFrame<'a> {
    /// `psDec->fs_kHz`: the internal rate in kHz (8, 12 or 16).
    pub fs_khz: i32,
    /// `psDec->nb_subfr`: 4 for a 20 ms frame, 2 for 10 ms.
    pub nb_subfr: i32,
    /// `psDec->LPC_order`: 16 at 16 kHz.
    pub lpc_order: i32,
    /// `psDec->indices.signalType`: 0 inactive, 1 unvoiced, 2 voiced.
    pub signal_type: i32,
    /// `psDecCtrl->PredCoef_Q12`: the LPC coefficients of each half of the
    /// frame (Q12).
    pub pred_coef_q12: &'a [[i16; 16]; 2],
    /// `psDecCtrl->pitchL`: the pitch lag of each subframe, in samples.
    pub pitch_l: &'a [i32; 4],
    /// `psDecCtrl->LTPCoef_Q14`: the 5 LTP coefficients of each subframe
    /// (Q14).
    pub ltp_coef_q14: &'a [i16; 20],
    /// `psDecCtrl->Gains_Q16`: the gain of each subframe (Q16).
    pub gains_q16: &'a [i32; 4],
    /// The bits of the frame: the range decoder position after the frame
    /// minus the position before its indices (`ec_tell(psRangeDec) -
    /// ec_start` in `silk_decode_frame`).
    pub num_bits: i32,
}

/// The layers OSCE can use and the kernel variant, borrowed from the
/// decoder's `DecoderDnn` (or from the built-in models). A method whose
/// layers are `None` behaves as method NONE, as libopus does when its model
/// is not loaded.
#[derive(Clone, Copy)]
pub struct OsceModels<'a> {
    /// The kernel variant.
    pub arch: DnnArch,
    /// LACE.
    pub lace: Option<&'a LaceLayers<'a>>,
    /// NoLACE.
    pub nolace: Option<&'a NolaceLayers<'a>>,
    /// BBWENet.
    #[cfg(feature = "osce-bwe")]
    pub bbwenet: Option<&'a BbwenetLayers<'a>>,
}

impl<'a> OsceModels<'a> {
    /// The layers and the kernel variant of a decoder.
    pub fn from_decoder(dnn: &'a DecoderDnn) -> OsceModels<'a> {
        OsceModels {
            arch: dnn.arch(),
            lace: dnn.lace(),
            nolace: dnn.nolace(),
            #[cfg(feature = "osce-bwe")]
            bbwenet: dnn.bbwenet(),
        }
    }

    /// The built-in layers, with the given kernel variant.
    pub fn builtin(arch: DnnArch) -> OsceModels<'static> {
        let set = super::builtin::models();
        OsceModels {
            arch,
            lace: set.lace.as_ref(),
            nolace: set.nolace.as_ref(),
            #[cfg(feature = "osce-bwe")]
            bbwenet: set.bbwenet.as_ref(),
        }
    }
}

/// The model state of a channel: the state of its method (libopus keeps
/// the two in a union). It is stored inline, as all decoder state is.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum ModelState {
    None,
    Lace(LaceState),
    Nolace(NolaceState),
}

/// The OSCE state of one SILK channel (libopus `silk_OSCE_struct`): the
/// method, the feature state and the state of the method's model:
/// 14,344 bytes on 64-bit targets (libopus: 30,820).
#[derive(Clone)]
pub struct OsceChannel {
    method: OsceMethod,
    features: OsceFeatureState,
    state: ModelState,
}

impl OsceChannel {
    /// The state of a new SILK channel: `silk_init_decoder` clears it, then
    /// `silk_reset_decoder` resets it to `OSCE_DEFAULT_METHOD`.
    pub fn new() -> OsceChannel {
        let mut ch = core::mem::MaybeUninit::<OsceChannel>::uninit();
        // SAFETY: `write_new` initializes every field of `ch`.
        unsafe {
            OsceChannel::write_new(ch.as_mut_ptr());
            ch.assume_init()
        }
    }

    /// Writes the state of [`new`](Self::new) to `this`, which may be
    /// uninitialized memory, without building it on the stack first (see
    /// `crate::in_place`).
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    pub(crate) unsafe fn write_new(this: *mut OsceChannel) {
        // SAFETY: the caller guarantees that `this` is valid for writes;
        // once the fields are written, `*this` is a valid channel.
        unsafe {
            write_fields!(this, {
                method: OsceMethod::None,
                features: OsceFeatureState::default(),
                state: ModelState::None,
            });
            osce_reset(&mut *this, OSCE_DEFAULT_METHOD);
        }
    }

    /// The method of the channel.
    pub fn method(&self) -> OsceMethod {
        self.method
    }

    /// The bypass counter: 2 right after a reset, 1 when the next eligible
    /// frame is cross-faded, 0 otherwise.
    pub fn bypass_count(&self) -> i32 {
        self.features.reset
    }
}

impl Default for OsceChannel {
    fn default() -> OsceChannel {
        OsceChannel::new()
    }
}

/// libopus `osce_reset` (`dnn/osce.c:1381-1406`): clears the feature state
/// and the state of `method`, makes `method` the channel's method, and sets
/// the bypass counter to 2.
pub fn osce_reset(ch: &mut OsceChannel, method: OsceMethod) {
    ch.features = OsceFeatureState::default();
    ch.state = match method {
        OsceMethod::None => ModelState::None,
        OsceMethod::Lace => ModelState::Lace(LaceState::default()),
        OsceMethod::Nolace => ModelState::Nolace(NolaceState::default()),
    };
    ch.method = method;
    ch.features.reset = 2;
}

/// libopus `osce_enhance_frame` (`dnn/osce.c:1546-1681`): enhances one
/// decoded SILK frame in place.
///
/// `xq` holds the decoded samples (`frame.nb_subfr · 80` at 16 kHz; only
/// 20 ms frames at 16 kHz are enhanced, other frames reset the channel).
///
/// With method NONE the features are not computed: the output does not
/// depend on them, and only `osce_reset`, which clears them, leaves method
/// NONE. libopus computes them anyway; the output is the same
/// (`osce_enhance_frame_with_features` computes them, for tests).
pub fn osce_enhance_frame(ch: &mut OsceChannel, models: &OsceModels<'_>, frame: &OsceFrame<'_>, xq: &mut [i16]) {
    enhance_frame(ch, models, frame, xq, false);
}

/// `osce_enhance_frame` that computes the features also with method NONE,
/// exactly as libopus does. For tests.
#[doc(hidden)]
pub fn osce_enhance_frame_with_features(ch: &mut OsceChannel, models: &OsceModels<'_>, frame: &OsceFrame<'_>, xq: &mut [i16]) {
    enhance_frame(ch, models, frame, xq, true);
}

fn enhance_frame(ch: &mut OsceChannel, models: &OsceModels<'_>, frame: &OsceFrame<'_>, xq: &mut [i16], always_features: bool) {
    // Enhancement exists only for 20 ms frames at 16 kHz.
    if frame.fs_khz != 16 || frame.nb_subfr != 4 {
        let method = ch.method;
        osce_reset(ch, method);
        return;
    }
    let xq = &mut xq[..320];

    if ch.method == OsceMethod::None && !always_features {
        // The output is the input; only the clip of -32768 shows.
        step_bypass_counter(&mut ch.features);
        for v in xq.iter_mut() {
            *v = (*v).max(-32767);
        }
        return;
    }

    let mut features = [0.0f32; 4 * OSCE_FEATURE_DIM];
    let mut numbits = [0.0f32; 2];
    let mut periods = [0i32; 4];
    osce_calculate_features(&mut ch.features, frame, xq, &mut features, &mut numbits, &mut periods);

    let mut in_buffer = [0.0f32; 320];
    for (b, &v) in in_buffer.iter_mut().zip(xq.iter()) {
        *b = v as f32 * (1.0 / 32768.0);
    }
    let mut out_buffer = in_buffer;
    match (&mut ch.state, ch.method) {
        (ModelState::Lace(st), OsceMethod::Lace) => {
            if let Some(layers) = models.lace {
                lace_process_20ms_frame(layers, st, &mut out_buffer, &in_buffer, &features, &numbits, &periods, models.arch);
            }
        }
        (ModelState::Nolace(st), OsceMethod::Nolace) => {
            if let Some(layers) = models.nolace {
                nolace_process_20ms_frame(layers, st, &mut out_buffer, &in_buffer, &features, &numbits, &periods, models.arch);
            }
        }
        _ => {}
    }

    if ch.features.reset > 1 {
        out_buffer = in_buffer;
        ch.features.reset -= 1;
    } else if ch.features.reset != 0 {
        osce_cross_fade_10ms(&mut out_buffer, &in_buffer);
        ch.features.reset = 0;
    }

    for (v, &y) in xq.iter_mut().zip(out_buffer.iter()) {
        *v = float2int((32768.0 * y).clamp(-32767.0, 32767.0)) as i16;
    }
}

/// The bypass counter of a frame whose output is its input.
fn step_bypass_counter(features: &mut OsceFeatureState) {
    if features.reset > 1 {
        features.reset -= 1;
    } else {
        features.reset = 0;
    }
}

/// The bandwidth extension state of one SILK channel (libopus
/// `silk_OSCE_BWE_struct`): 8,220 bytes on 64-bit targets (libopus:
/// 18,368).
#[cfg(feature = "osce-bwe")]
#[derive(Clone, Default)]
pub struct OsceBweChannel {
    features: OsceBweFeatureState,
    state: BbwenetState,
}

#[cfg(feature = "osce-bwe")]
impl OsceBweChannel {
    /// The state of a new SILK channel: all zero, as `silk_init_decoder`
    /// leaves it (not the state of `osce_bwe_reset`).
    pub fn new() -> OsceBweChannel {
        OsceBweChannel::default()
    }

    /// Writes the state of [`new`](Self::new) to `this`, which may be
    /// uninitialized memory, without building it on the stack first (see
    /// `crate::in_place`).
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    pub(crate) unsafe fn write_new(this: *mut OsceBweChannel) {
        // SAFETY: the caller guarantees that `this` is valid for writes.
        unsafe {
            write_fields!(this, {
                features: OsceBweFeatureState::default(),
                state: BbwenetState::default(),
            });
        }
    }
}

/// libopus `osce_bwe_reset` (`dnn/osce.c:1410-1422`): clears the state, then
/// sets the real parts of the previous spectrum to 1e-9 (as the Python
/// model was trained).
#[cfg(feature = "osce-bwe")]
pub fn osce_bwe_reset(ch: &mut OsceBweChannel) {
    *ch = OsceBweChannel::default();
    for k in 0..=features::OSCE_BWE_MAX_INSTAFREQ_BIN {
        ch.features.last_spec[2 * k] = 1e-9;
    }
}

/// libopus `osce_bwe` (`dnn/osce.c:1478-1542`): extends 10 or 20 ms of
/// 16 kHz samples (`xq16`, 160 or 320 samples) to 48 kHz (`xq48`, three
/// times as many). The output is delayed by `OSCE_BWE_OUTPUT_DELAY` (21)
/// samples: the first 21 come from the previous call.
///
/// Panics when `xq16` does not hold 160 or 320 samples or `xq48` is shorter
/// than `3 · xq16.len()`.
#[cfg(feature = "osce-bwe")]
pub fn osce_bwe(ch: &mut OsceBweChannel, layers: &BbwenetLayers<'_>, arch: DnnArch, xq48: &mut [i16], xq16: &[i16]) {
    let len = xq16.len();
    assert!(len == 160 || len == 320, "osce_bwe takes 10 or 20 ms");
    let xq48 = &mut xq48[..3 * len];
    let num_frames = len / 160;

    let mut in_buffer = [0.0f32; 320];
    for (b, &v) in in_buffer.iter_mut().zip(xq16) {
        *b = v as f32 * (1.0 / 32768.0);
    }
    let mut feat = [0.0f32; 2 * OSCE_BWE_FEATURE_DIM];
    osce_bwe_calculate_features(&mut ch.features, &mut feat, xq16);

    let mut out_buffer = [0.0f32; 3 * 320];
    bbwenet_process_frames(layers, &mut ch.state, &mut out_buffer[..3 * len], &in_buffer[..len], &feat, num_frames, arch);

    let to_i16 = |y: f32| float2int((32768.0 * y).clamp(-32767.0, 32767.0)) as i16;
    let d = OSCE_BWE_OUTPUT_DELAY;
    xq48[..d].copy_from_slice(&ch.state.output_buffer);
    for (o, &y) in xq48[d..].iter_mut().zip(out_buffer.iter()) {
        *o = to_i16(y);
    }
    for (o, &y) in ch.state.output_buffer.iter_mut().zip(out_buffer[3 * len - d..3 * len].iter()) {
        *o = to_i16(y);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// A 20 ms frame at 16 kHz with plausible SILK parameters: a voiced
    /// tone with a fixed LPC filter.
    struct TestFrame {
        pred: [[i16; 16]; 2],
        pitch: [i32; 4],
        ltp: [i16; 20],
        gains: [i32; 4],
        xq: [i16; 320],
    }

    impl TestFrame {
        fn new(n: usize, amplitude: f32) -> TestFrame {
            let mut pred = [[0i16; 16]; 2];
            pred[0][0] = 3000;
            pred[0][1] = -1500;
            pred[1] = pred[0];
            TestFrame {
                pred,
                pitch: [100; 4],
                ltp: [1000, 2000, 8000, 2000, 1000, 1000, 2000, 8000, 2000, 1000, 1000, 2000, 8000, 2000, 1000, 1000, 2000, 8000, 2000, 1000],
                gains: [65536 * 300; 4],
                xq: core::array::from_fn(|i| {
                    let t = (n * 320 + i) as f32 / 16000.0;
                    (amplitude * (2.0 * core::f32::consts::PI * 160.0 * t).sin()) as i16
                }),
            }
        }
        fn frame(&self, fs_khz: i32, nb_subfr: i32) -> OsceFrame<'_> {
            OsceFrame {
                fs_khz,
                nb_subfr,
                lpc_order: 16,
                signal_type: 2,
                pred_coef_q12: &self.pred,
                pitch_l: &self.pitch,
                ltp_coef_q14: &self.ltp,
                gains_q16: &self.gains,
                num_bits: 300,
            }
        }
    }

    #[test]
    fn a_new_channel_uses_nolace_and_bypasses_two_frames() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let mut ch = OsceChannel::new();
        assert_eq!(ch.method(), OsceMethod::Nolace);
        let counts: Vec<i32> = (0..4)
            .map(|n| {
                let f = TestFrame::new(n, 8000.0);
                let mut x = f.xq;
                osce_enhance_frame(&mut ch, &models, &f.frame(16, 4), &mut x);
                if n == 0 {
                    // The first frame after a reset comes out unenhanced.
                    assert_eq!(x, f.xq);
                }
                ch.bypass_count()
            })
            .collect();
        assert_eq!(counts, [1, 0, 0, 0]);
    }

    #[test]
    fn an_ineligible_frame_resets_and_is_left_alone() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let mut ch = OsceChannel::new();
        osce_reset(&mut ch, OsceMethod::Lace);
        for n in 0..3 {
            let f = TestFrame::new(n, 8000.0);
            let mut x = f.xq;
            osce_enhance_frame(&mut ch, &models, &f.frame(16, 4), &mut x);
        }
        assert_eq!(ch.bypass_count(), 0);
        for (fs, subfr) in [(16, 2), (12, 4), (8, 4)] {
            let f = TestFrame::new(9, 8000.0);
            let mut x = f.xq;
            osce_enhance_frame(&mut ch, &models, &f.frame(fs, subfr), &mut x);
            assert_eq!(x, f.xq);
            assert_eq!(ch.method(), OsceMethod::Lace);
            assert_eq!(ch.bypass_count(), 2);
        }
    }

    #[test]
    fn method_none_clips_only_the_most_negative_sample() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let mut ch = OsceChannel::new();
        osce_reset(&mut ch, OsceMethod::None);
        let mut f = TestFrame::new(0, 32767.0);
        f.xq[5] = -32768;
        f.xq[6] = 32767;
        for _ in 0..3 {
            let mut x = f.xq;
            osce_enhance_frame(&mut ch, &models, &f.frame(16, 4), &mut x);
            let mut want = f.xq;
            want[5] = -32767;
            assert_eq!(x, want);
        }
    }

    /// Skipping the features with method NONE gives the output of computing
    /// them, also after switching to LACE or NoLACE (the switch resets the
    /// features).
    #[test]
    fn skipping_the_features_of_method_none_changes_nothing() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let mut a = OsceChannel::new();
        let mut b = OsceChannel::new();
        let methods = [OsceMethod::None, OsceMethod::Lace, OsceMethod::None, OsceMethod::Nolace];
        for n in 0..24 {
            let m = methods[n / 6];
            if a.method() != m {
                osce_reset(&mut a, m);
                osce_reset(&mut b, m);
            }
            let f = TestFrame::new(n, 12000.0);
            let (mut x, mut y) = (f.xq, f.xq);
            osce_enhance_frame(&mut a, &models, &f.frame(16, 4), &mut x);
            osce_enhance_frame_with_features(&mut b, &models, &f.frame(16, 4), &mut y);
            assert_eq!(x, y, "frame {n}");
        }
    }

    #[test]
    fn unloaded_layers_act_as_method_none_but_keep_the_features() {
        let loaded = OsceModels::builtin(crate::dnn::arch::detect());
        let unloaded = OsceModels { lace: None, nolace: None, ..loaded };
        let mut ch = OsceChannel::new();
        for n in 0..3 {
            let f = TestFrame::new(n, 8000.0);
            let mut x = f.xq;
            osce_enhance_frame(&mut ch, &unloaded, &f.frame(16, 4), &mut x);
            assert_eq!(x, f.xq);
        }
        assert_eq!(ch.method(), OsceMethod::Nolace);
        assert_ne!(ch.features.numbits_smooth, 0.0);
    }

    #[cfg(feature = "osce-bwe")]
    #[test]
    fn the_bandwidth_extension_is_delayed_by_21_samples() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let layers = models.bbwenet.unwrap();
        let mut ch = OsceBweChannel::new();
        let mut out = [1i16; 960];
        let silence = [0i16; 320];
        osce_bwe(&mut ch, layers, models.arch, &mut out, &silence);
        assert!(out.iter().all(|&v| v == 0));
        let f = TestFrame::new(1, 8000.0);
        osce_bwe(&mut ch, layers, models.arch, &mut out[..480], &f.xq[..160]);
        // The first 21 samples come from the silent call.
        assert!(out[..OSCE_BWE_OUTPUT_DELAY].iter().all(|&v| v == 0));
        assert!(out[OSCE_BWE_OUTPUT_DELAY..480].iter().any(|&v| v != 0));
    }

    #[cfg(feature = "osce-bwe")]
    #[test]
    #[should_panic(expected = "10 or 20 ms")]
    fn the_bandwidth_extension_takes_10_or_20_ms() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let mut ch = OsceBweChannel::new();
        let mut out = [0i16; 960];
        osce_bwe(&mut ch, models.bbwenet.unwrap(), models.arch, &mut out, &[0i16; 240]);
    }

    #[cfg(feature = "osce-bwe")]
    #[test]
    fn the_bwe_reset_sets_the_previous_spectrum_to_1e_9() {
        let mut ch = OsceBweChannel::new();
        assert!(ch.features.last_spec.iter().all(|&v| v == 0.0));
        osce_bwe_reset(&mut ch);
        for (k, &v) in ch.features.last_spec.iter().enumerate() {
            assert_eq!(v, if k % 2 == 0 { 1e-9 } else { 0.0 });
        }
    }

    /// The OSCE path, with its states on the heap as in a decoder, runs on
    /// a thread with 32 KB of stack (it needs 16 to 20 KB on 64-bit ARM
    /// macOS). Elsewhere the thread gets 64 KB: with 32 KB the test
    /// overflowed its stack on Android 14 (HONOR CLK-AN00, NDK 27) and on
    /// x86-64.
    /// Debug builds need more, so the test runs only in optimized builds.
    #[cfg(all(feature = "osce-bwe", not(debug_assertions)))]
    #[test]
    fn the_osce_path_runs_on_32_kb_of_stack() {
        let models = OsceModels::builtin(crate::dnn::arch::detect());
        let mut ch = std::boxed::Box::new(OsceChannel::new());
        let mut bwe = std::boxed::Box::new(OsceBweChannel::new());
        std::thread::Builder::new()
            .stack_size(if cfg!(all(target_arch = "aarch64", target_vendor = "apple")) { 32 } else { 64 } * 1024)
            .spawn(move || {
                for method in [OsceMethod::Nolace, OsceMethod::Lace] {
                    osce_reset(&mut ch, method);
                    for n in 0..3 {
                        let f = TestFrame::new(n, 8000.0);
                        let mut x = f.xq;
                        osce_enhance_frame(&mut ch, &models, &f.frame(16, 4), &mut x);
                        let mut out = [0i16; 960];
                        osce_bwe(&mut bwe, models.bbwenet.unwrap(), models.arch, &mut out, &x);
                    }
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
