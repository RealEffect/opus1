// opus1 computes the 64-bit ARM bits of the neural network code on every
// target, and these tests compare with the bits of the linked libopus, so
// they are compiled for 64-bit ARM only (docs/10, Platforms).
#![cfg(all(feature = "test-libopus-dnn", target_arch = "aarch64"))]
#![cfg(feature = "deep-plc")]
//! Every layer of the libopus 1.6.1 models through opus1 and through
//! libopus, with the real weights and inputs taken from speech.
//!
//! Each model (PitchDNN, the PLC model and FARGAN; with `osce` LACE and
//! NoLACE; with `osce-bwe` BBWENet) is run frame by frame as libopus runs
//! it: the same layer calls, with the same activations, input sizes,
//! convolution memories, GRU states and in-place GLUs. The code between the
//! layer calls (embeddings, gains, the temporal envelope of the shaping
//! layers, leaky ReLU, interpolation) is written here in plain Rust and is
//! the same for both sides. The signal filters of OSCE (comb filters,
//! adaptive convolutions) are left out: they do not feed the layers.
//!
//! The inputs come from the speech of `fixtures/answer_16k.wav`:
//!
//! - the deep PLC features, the PitchDNN inputs (instantaneous-frequency
//!   and cross-correlation features) and the pitch come from libopus's own
//!   feature extractor (`lpcnet_compute_single_frame_features_float`);
//! - the Burg cepstrum of the PLC input comes from opus1's
//!   `burg_cepstral_analysis` (bit-exact with libopus, `dnn_freq_burg.rs`);
//! - the BBWENet features come from libopus's `osce_bwe_calculate_features`;
//! - the LACE and NoLACE features are built from the same speech analysis
//!   (LPC spectrum, cepstrum, autocorrelation at the pitch lag, a voicing
//!   based LTP filter, the log gain), with a bit count that changes;
//! - FARGAN runs on its own output, as in a long loss, and is fed real
//!   speech again every 25 frames.
//!
//! Two checks run on these inputs:
//!
//! 1. Layer by layer: every call runs in libopus and in opus1 for every
//!    `DnnArch` this processor can run, on the same inputs and states. The
//!    outputs, states and memories must be equal bit for bit. libopus's
//!    result is then passed on, so a difference stays in the layer that
//!    made it.
//! 2. Whole models: the same models run once with only libopus layers and
//!    once with only opus1 layers (each `DnnArch`), each side feeding
//!    itself. Every layer output of the two runs must be equal.
//!
//! PitchDNN is also checked against libopus's own `compute_pitchdnn`,
//! which runs inside the feature extractor: the pitch computed here from
//! the layer outputs must equal the pitch it returns.
//!
//! The bit-exact reference is libopus built without loop vectorization
//! (`plan.md` section 1):
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,osce-bwe --test dnn_layers_libopus
//! ```
//!
//! With `OPUS1_DNN_REF_O2=1` and the plain `-O2` build
//! (`libopus-host-1.6.1-dnn`), the test measures instead of asserting
//! equality. It prints, for every layer that differs, the number of values
//! that differ and the largest difference, both for single calls (check 1)
//! and after the models ran on their own outputs (check 2). It then checks
//! only that single calls differ in float layers with one output, within
//! `2·M·2^-24·Σ|w·x|` (`nnet.md`, Review item 13). Run it with
//! `-- --nocapture --test-threads=1` to read the tables.

mod dnn_gen;

use dnn_gen::{CConv2dLayer, CLayer, CLinearLayer, CPitchDNN, MODELS};
use opus1::internals::dnn::arch::{self, DnnArch};
use opus1::internals::dnn::models::Layer;
use opus1::internals::dnn::nnet::{self, Conv2dLayer, LinearLayer};
use opus1::internals::dnn::{Activation, builtin, freq};
use std::collections::{BTreeMap, HashMap};
use std::os::raw::c_int;

// ---------------------------------------------------------------------------
// libopus functions
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn opus_select_arch() -> c_int;
    #[link_name = "compute_generic_dense"]
    fn c_generic_dense(l: *const CLinearLayer, out: *mut f32, input: *const f32, act: c_int, arch: c_int);
    #[link_name = "compute_generic_gru"]
    fn c_generic_gru(
        input_w: *const CLinearLayer,
        recurrent_w: *const CLinearLayer,
        state: *mut f32,
        input: *const f32,
        arch: c_int,
    );
    #[link_name = "compute_glu"]
    fn c_glu(l: *const CLinearLayer, out: *mut f32, input: *const f32, arch: c_int);
    #[link_name = "compute_generic_conv1d"]
    fn c_generic_conv1d(
        l: *const CLinearLayer,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        input_size: c_int,
        act: c_int,
        arch: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    fn compute_conv2d_c(
        c: *const CConv2dLayer,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        height: c_int,
        hstride: c_int,
        act: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    fn compute_conv2d_dotprod(
        c: *const CConv2dLayer,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        height: c_int,
        hstride: c_int,
        act: c_int,
    );
    fn compute_activation_c(out: *mut f32, input: *const f32, n: c_int, act: c_int);
    fn compute_activation_dotprod(out: *mut f32, input: *const f32, n: c_int, act: c_int);

    // The deep PLC feature extractor (`dnn/lpcnet_enc.c`).
    fn lpcnet_encoder_get_size() -> c_int;
    fn lpcnet_encoder_init(st: *mut u8) -> c_int;
    fn lpcnet_compute_single_frame_features_float(
        st: *mut u8,
        pcm: *const f32,
        features: *mut f32,
        arch: c_int,
    ) -> c_int;
}

#[cfg(feature = "osce-bwe")]
unsafe extern "C" {
    // The BBWENet features (`dnn/osce_features.c`).
    fn osce_bwe_calculate_features(state: *mut f32, features: *mut f32, xq: *const i16, num_samples: c_int);
}

/// The arch value of libopus for DotProd (`OPUS_ARCH_ARM_DOTPROD`).
const ARCH_DOTPROD: c_int = 4;

fn c_arch() -> c_int {
    // Makes sure the static libopus is linked, then asks it for its arch.
    let _ = unsafe { audiopus_sys::opus_get_version_string() };
    unsafe { opus_select_arch() }
}

fn o2_mode() -> bool {
    std::env::var_os("OPUS1_DNN_REF_O2").is_some()
}

/// The variants to compare: all that this processor runs, or, when
/// measuring against the `-O2` build, the one opus1 selects (all give the
/// same bits).
fn archs(measure: bool) -> Vec<DnnArch> {
    if measure { vec![arch::detect()] } else { arch::available().to_vec() }
}

// ---------------------------------------------------------------------------
// The layers of both sides, by field name
// ---------------------------------------------------------------------------

/// One model on both sides: the layers of libopus's `init_*` on its
/// built-in arrays, and the layers of opus1's built-in weights.
struct Net {
    c: HashMap<&'static str, CLayer>,
    rust: HashMap<&'static str, Layer<'static, 'static>>,
    names: Vec<&'static str>,
}

fn net(key: &str) -> Net {
    let m = MODELS.iter().find(|m| m.key == key).expect("model");
    let (ret, c_layers) = (m.init)((m.arrays)());
    assert_eq!(ret, 0, "init of {key} failed in libopus");
    let models = builtin::models();
    let rust: Vec<Layer<'static, 'static>> = match key {
        "pitchdnn" => models.deep_plc.as_ref().expect("deep PLC").pitchdnn.layers().to_vec(),
        "plc" => models.deep_plc.as_ref().expect("deep PLC").plc.layers().to_vec(),
        "fargan" => models.deep_plc.as_ref().expect("deep PLC").fargan.layers().to_vec(),
        #[cfg(feature = "osce")]
        "lace" => models.lace.as_ref().expect("LACE").layers().to_vec(),
        #[cfg(feature = "osce")]
        "nolace" => models.nolace.as_ref().expect("NoLACE").layers().to_vec(),
        #[cfg(feature = "osce-bwe")]
        "bbwenet" => models.bbwenet.as_ref().expect("BBWENet").layers().to_vec(),
        _ => panic!("model {key} is not in this build"),
    };
    assert_eq!(rust.len(), m.layers.len());
    assert_eq!(c_layers.len(), m.layers.len());
    let names: Vec<&'static str> = m.layers.iter().map(|s| s.name()).collect();
    Net {
        c: names.iter().copied().zip(c_layers).collect(),
        rust: names.iter().copied().zip(rust).collect(),
        names,
    }
}

impl Net {
    fn c_linear(&self, name: &str) -> &CLinearLayer {
        match self.c.get(name) {
            Some(CLayer::Linear(l)) => l,
            _ => panic!("{name} is not a linear layer"),
        }
    }

    fn linear(&self, name: &str) -> &'static LinearLayer<'static> {
        match self.rust.get(name) {
            Some(Layer::Linear(l)) => l,
            _ => panic!("{name} is not a linear layer"),
        }
    }

    fn c_conv2d(&self, name: &str) -> &CConv2dLayer {
        match self.c.get(name) {
            Some(CLayer::Conv2d(l)) => l,
            _ => panic!("{name} is not a conv2d layer"),
        }
    }

    fn conv2d(&self, name: &str) -> &'static Conv2dLayer<'static> {
        match self.rust.get(name) {
            Some(Layer::Conv2d(l)) => l,
            _ => panic!("{name} is not a conv2d layer"),
        }
    }

    /// The float weights that the model code reads as an embedding table.
    fn table(&self, name: &str) -> &'static [f32] {
        self.linear(name).float_weights.expect("an embedding table")
    }
}

// ---------------------------------------------------------------------------
// Running a layer on one side
// ---------------------------------------------------------------------------

/// Which implementation runs a layer.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    C,
    Rust(DnnArch),
}

/// Runs the layer calls of the models on one side. The slices are checked
/// before every C call, so a mistake in this test cannot make libopus read
/// out of bounds.
struct Exec<'n> {
    net: &'n Net,
    side: Side,
    c_arch: c_int,
}

impl Exec<'_> {
    fn dense(&self, name: &str, out: &mut [f32], input: &[f32], act: Activation) {
        let r = self.net.linear(name);
        assert!(input.len() >= r.nb_inputs && out.len() >= r.nb_outputs, "{name}: slice sizes");
        match self.side {
            Side::C => unsafe {
                c_generic_dense(self.net.c_linear(name), out.as_mut_ptr(), input.as_ptr(), act as c_int, self.c_arch)
            },
            Side::Rust(a) => nnet::compute_generic_dense(r, out, input, act, a),
        }
    }

    fn gru(&self, input_name: &str, recurrent_name: &str, state: &mut [f32], input: &[f32]) {
        let (ri, rr) = (self.net.linear(input_name), self.net.linear(recurrent_name));
        assert!(input.len() >= ri.nb_inputs && state.len() >= rr.nb_inputs, "{input_name}: slice sizes");
        match self.side {
            Side::C => unsafe {
                c_generic_gru(
                    self.net.c_linear(input_name),
                    self.net.c_linear(recurrent_name),
                    state.as_mut_ptr(),
                    input.as_ptr(),
                    self.c_arch,
                )
            },
            Side::Rust(a) => nnet::compute_generic_gru(ri, rr, state, input, a),
        }
    }

    fn glu(&self, name: &str, out: &mut [f32], input: &[f32]) {
        let r = self.net.linear(name);
        assert!(input.len() >= r.nb_inputs && out.len() >= r.nb_outputs, "{name}: slice sizes");
        match self.side {
            Side::C => unsafe { c_glu(self.net.c_linear(name), out.as_mut_ptr(), input.as_ptr(), self.c_arch) },
            Side::Rust(a) => nnet::compute_glu(r, out, input, a),
        }
    }

    /// `compute_glu(layer, x, x)`, as FARGAN calls it.
    fn glu_in_place(&self, name: &str, buf: &mut [f32]) {
        let r = self.net.linear(name);
        assert!(buf.len() >= r.nb_inputs, "{name}: slice sizes");
        match self.side {
            Side::C => unsafe { c_glu(self.net.c_linear(name), buf.as_mut_ptr(), buf.as_ptr(), self.c_arch) },
            Side::Rust(a) => nnet::compute_glu_in_place(r, buf, a),
        }
    }

    /// A conv1d whose input size is `input.len()`; an empty `mem` is
    /// passed to libopus as `NULL`.
    fn conv1d(&self, name: &str, out: &mut [f32], mem: &mut [f32], input: &[f32], act: Activation) {
        let r = self.net.linear(name);
        assert!(out.len() >= r.nb_outputs && mem.len() >= r.nb_inputs - input.len(), "{name}: slice sizes");
        match self.side {
            Side::C => unsafe {
                let mem_ptr = if mem.is_empty() { std::ptr::null_mut() } else { mem.as_mut_ptr() };
                c_generic_conv1d(
                    self.net.c_linear(name),
                    out.as_mut_ptr(),
                    mem_ptr,
                    input.as_ptr(),
                    input.len() as c_int,
                    act as c_int,
                    self.c_arch,
                )
            },
            Side::Rust(a) => nnet::compute_generic_conv1d(r, out, mem, input, act, a),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        name: &str,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        act: Activation,
    ) {
        let r = self.net.conv2d(name);
        let time_stride = r.in_channels * (height + r.kheight - 1);
        assert!(input.len() >= time_stride && mem.len() >= (r.ktime - 1) * time_stride);
        assert!(out.len() >= (r.out_channels - 1) * hstride + height);
        match self.side {
            Side::C => unsafe {
                let f = if self.c_arch >= ARCH_DOTPROD { compute_conv2d_dotprod } else { compute_conv2d_c };
                f(
                    self.net.c_conv2d(name),
                    out.as_mut_ptr(),
                    mem.as_mut_ptr(),
                    input.as_ptr(),
                    height as c_int,
                    hstride as c_int,
                    act as c_int,
                )
            },
            Side::Rust(a) => nnet::compute_conv2d(r, out, mem, input, height, hstride, act, a),
        }
    }

    /// `compute_activation(x, x, n, ACTIVATION_EXP)` of the shaping layers.
    #[cfg_attr(not(feature = "osce"), allow(dead_code))]
    fn exp(&self, buf: &mut [f32]) {
        match self.side {
            Side::C => unsafe {
                let f = if self.c_arch >= ARCH_DOTPROD { compute_activation_dotprod } else { compute_activation_c };
                f(buf.as_mut_ptr(), buf.as_ptr(), buf.len() as c_int, Activation::Exp as c_int)
            },
            Side::Rust(a) => nnet::compute_activation_in_place(buf, Activation::Exp, a),
        }
    }
}

// ---------------------------------------------------------------------------
// The two checks
// ---------------------------------------------------------------------------

/// The layer calls the model code makes. `Check` runs each call on both
/// sides and compares; `Single` runs one side and records every result.
trait Backend {
    fn dense(&mut self, name: &'static str, out: &mut [f32], input: &[f32], act: Activation);
    fn gru(&mut self, input_name: &'static str, recurrent_name: &'static str, state: &mut [f32], input: &[f32]);
    fn glu(&mut self, name: &'static str, out: &mut [f32], input: &[f32]);
    fn glu_in_place(&mut self, name: &'static str, buf: &mut [f32]);
    fn conv1d(&mut self, name: &'static str, out: &mut [f32], mem: &mut [f32], input: &[f32], act: Activation);
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &mut self,
        name: &'static str,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        act: Activation,
    );
    #[cfg_attr(not(feature = "osce"), allow(dead_code))]
    fn exp(&mut self, name: &'static str, buf: &mut [f32]);
}

/// What the measuring mode records for one layer.
#[derive(Default, Clone)]
struct Stat {
    calls: usize,
    values: usize,
    differ: usize,
    max_abs: f64,
    /// The largest difference divided by `Σ|w·x| + |b|` of that output
    /// (float layers only).
    max_rel: f64,
    /// The largest difference divided by the bound `2·M·2^-24·Σ|w·x|`.
    max_bound: f64,
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Check 1: every call on both sides, libopus's result passed on.
struct Check<'n> {
    c: Exec<'n>,
    rust: Vec<Exec<'n>>,
    measure: bool,
    stats: BTreeMap<&'static str, Stat>,
}

impl<'n> Check<'n> {
    fn new(net: &'n Net, measure: bool) -> Self {
        let c_arch = c_arch();
        Check {
            c: Exec { net, side: Side::C, c_arch },
            rust: archs(measure).into_iter().map(|a| Exec { net, side: Side::Rust(a), c_arch }).collect(),
            measure,
            stats: BTreeMap::new(),
        }
    }

    /// `Σ_j |w_ij·x_j| + |b_i|` for each output of a float layer, the scale
    /// of the rounding error of its sums.
    fn scale(&self, name: &str, input: &[f32]) -> Option<Vec<f64>> {
        let l = self.c.net.linear(name);
        if l.weights.is_some() {
            return None;
        }
        let w = l.float_weights?;
        let n = l.nb_outputs;
        Some(
            (0..n)
                .map(|i| {
                    let b = l.bias.map_or(0.0, |b| b[i].abs() as f64);
                    b + (0..l.nb_inputs).map(|j| (w[j * n + i] as f64 * input[j] as f64).abs()).sum::<f64>()
                })
                .collect(),
        )
    }

    fn compare(&mut self, name: &'static str, what: &str, side: Side, c: &[f32], r: &[f32], scale: Option<&[f64]>) {
        let nb_inputs = match self.c.net.rust.get(name) {
            Some(Layer::Linear(l)) => l.nb_inputs,
            _ => 0,
        };
        let st = self.stats.entry(name).or_default();
        st.values += c.len();
        for (i, (&a, &b)) in c.iter().zip(r).enumerate() {
            if same(a, b) {
                continue;
            }
            assert!(
                self.measure,
                "{name} ({what}), {side:?}, call {}: value {i} is {b:e} ({:#010x}), libopus has {a:e} ({:#010x})",
                st.calls,
                b.to_bits(),
                a.to_bits()
            );
            st.differ += 1;
            let d = (a as f64 - b as f64).abs();
            st.max_abs = st.max_abs.max(d);
            if let Some(s) = scale.and_then(|s| s.get(i)) {
                if *s > 0.0 {
                    st.max_rel = st.max_rel.max(d / s);
                    let bound = 2.0 * nb_inputs as f64 * f64::powi(2.0, -24) * s;
                    st.max_bound = st.max_bound.max(d / bound);
                }
            } else {
                // Not a float layer: no bound applies.
                st.max_bound = f64::INFINITY;
            }
        }
    }

    fn count_call(&mut self, name: &'static str) {
        self.stats.entry(name).or_default().calls += 1;
    }
}

impl Backend for Check<'_> {
    fn dense(&mut self, name: &'static str, out: &mut [f32], input: &[f32], act: Activation) {
        let n = self.c.net.linear(name).nb_outputs;
        let mut want = out.to_vec();
        self.c.dense(name, &mut want, input, act);
        let scale = if self.measure { self.scale(name, input) } else { None };
        for k in 0..self.rust.len() {
            let mut got = out.to_vec();
            self.rust[k].dense(name, &mut got, input, act);
            let side = self.rust[k].side;
            self.compare(name, "out", side, &want[..n], &got[..n], scale.as_deref());
            assert_eq!(want[n..], got[n..], "{name}: wrote past nb_outputs");
        }
        self.count_call(name);
        out.copy_from_slice(&want);
    }

    fn gru(&mut self, input_name: &'static str, recurrent_name: &'static str, state: &mut [f32], input: &[f32]) {
        let mut want = state.to_vec();
        self.c.gru(input_name, recurrent_name, &mut want, input);
        for k in 0..self.rust.len() {
            let mut got = state.to_vec();
            self.rust[k].gru(input_name, recurrent_name, &mut got, input);
            let side = self.rust[k].side;
            self.compare(input_name, "state", side, &want, &got, None);
        }
        self.count_call(input_name);
        self.count_call(recurrent_name);
        state.copy_from_slice(&want);
    }

    fn glu(&mut self, name: &'static str, out: &mut [f32], input: &[f32]) {
        let mut want = out.to_vec();
        self.c.glu(name, &mut want, input);
        for k in 0..self.rust.len() {
            let mut got = out.to_vec();
            self.rust[k].glu(name, &mut got, input);
            let side = self.rust[k].side;
            self.compare(name, "out", side, &want, &got, None);
        }
        self.count_call(name);
        out.copy_from_slice(&want);
    }

    fn glu_in_place(&mut self, name: &'static str, buf: &mut [f32]) {
        let mut want = buf.to_vec();
        self.c.glu_in_place(name, &mut want);
        for k in 0..self.rust.len() {
            let mut got = buf.to_vec();
            self.rust[k].glu_in_place(name, &mut got);
            let side = self.rust[k].side;
            self.compare(name, "out", side, &want, &got, None);
        }
        self.count_call(name);
        buf.copy_from_slice(&want);
    }

    fn conv1d(&mut self, name: &'static str, out: &mut [f32], mem: &mut [f32], input: &[f32], act: Activation) {
        let mut want = out.to_vec();
        let mut want_mem = mem.to_vec();
        self.c.conv1d(name, &mut want, &mut want_mem, input, act);
        let scale = if self.measure {
            // The layer's input is the memory followed by the new input.
            let l = self.c.net.linear(name);
            let mut x = mem[..l.nb_inputs - input.len()].to_vec();
            x.extend_from_slice(input);
            self.scale(name, &x)
        } else {
            None
        };
        for k in 0..self.rust.len() {
            let mut got = out.to_vec();
            let mut got_mem = mem.to_vec();
            self.rust[k].conv1d(name, &mut got, &mut got_mem, input, act);
            let side = self.rust[k].side;
            self.compare(name, "out", side, &want, &got, scale.as_deref());
            self.compare(name, "mem", side, &want_mem, &got_mem, None);
        }
        self.count_call(name);
        out.copy_from_slice(&want);
        mem.copy_from_slice(&want_mem);
    }

    fn conv2d(
        &mut self,
        name: &'static str,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        act: Activation,
    ) {
        let mut want = out.to_vec();
        let mut want_mem = mem.to_vec();
        self.c.conv2d(name, &mut want, &mut want_mem, input, height, hstride, act);
        for k in 0..self.rust.len() {
            let mut got = out.to_vec();
            let mut got_mem = mem.to_vec();
            self.rust[k].conv2d(name, &mut got, &mut got_mem, input, height, hstride, act);
            let side = self.rust[k].side;
            self.compare(name, "out", side, &want, &got, None);
            self.compare(name, "mem", side, &want_mem, &got_mem, None);
        }
        self.count_call(name);
        out.copy_from_slice(&want);
        mem.copy_from_slice(&want_mem);
    }

    fn exp(&mut self, name: &'static str, buf: &mut [f32]) {
        let mut want = buf.to_vec();
        self.c.exp(&mut want);
        for k in 0..self.rust.len() {
            let mut got = buf.to_vec();
            self.rust[k].exp(&mut got);
            let side = self.rust[k].side;
            self.compare(name, "exp", side, &want, &got, None);
        }
        buf.copy_from_slice(&want);
    }
}

/// Check 2: one side only. Every result is appended to `values`, and
/// `calls` says which layer wrote which part.
struct Single<'n> {
    exec: Exec<'n>,
    values: Vec<f32>,
    calls: Vec<(&'static str, usize)>,
}

impl<'n> Single<'n> {
    fn new(net: &'n Net, side: Side) -> Self {
        Single { exec: Exec { net, side, c_arch: c_arch() }, values: Vec::new(), calls: Vec::new() }
    }

    fn record(&mut self, name: &'static str, parts: &[&[f32]]) {
        self.calls.push((name, self.values.len()));
        for p in parts {
            self.values.extend_from_slice(p);
        }
    }
}

impl Backend for Single<'_> {
    fn dense(&mut self, name: &'static str, out: &mut [f32], input: &[f32], act: Activation) {
        self.exec.dense(name, out, input, act);
        let n = self.exec.net.linear(name).nb_outputs;
        self.record(name, &[&out[..n]]);
    }

    fn gru(&mut self, input_name: &'static str, recurrent_name: &'static str, state: &mut [f32], input: &[f32]) {
        self.exec.gru(input_name, recurrent_name, state, input);
        self.record(input_name, &[state]);
    }

    fn glu(&mut self, name: &'static str, out: &mut [f32], input: &[f32]) {
        self.exec.glu(name, out, input);
        self.record(name, &[out]);
    }

    fn glu_in_place(&mut self, name: &'static str, buf: &mut [f32]) {
        self.exec.glu_in_place(name, buf);
        self.record(name, &[buf]);
    }

    fn conv1d(&mut self, name: &'static str, out: &mut [f32], mem: &mut [f32], input: &[f32], act: Activation) {
        self.exec.conv1d(name, out, mem, input, act);
        let n = self.exec.net.linear(name).nb_outputs;
        self.record(name, &[&out[..n], mem]);
    }

    fn conv2d(
        &mut self,
        name: &'static str,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        act: Activation,
    ) {
        self.exec.conv2d(name, out, mem, input, height, hstride, act);
        self.record(name, &[out, mem]);
    }

    fn exp(&mut self, name: &'static str, buf: &mut [f32]) {
        self.exec.exp(buf);
        self.record(name, &[buf]);
    }
}

// ---------------------------------------------------------------------------
// Inputs from speech
// ---------------------------------------------------------------------------

const FRAME: usize = 160;
const NB_BANDS: usize = 18;
const NB_FEATURES: usize = 20;
const NB_TOTAL_FEATURES: usize = 36;
const IF_FEATURES: usize = 88;
const XCORR_FEATURES: usize = 224;

/// The 16-bit samples of the speech fixture (16 kHz, mono).
fn speech_samples() -> Vec<i16> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    let data = bytes.windows(4).position(|w| w == b"data").expect("a data chunk");
    let len = u32::from_le_bytes(bytes[data + 4..data + 8].try_into().unwrap()) as usize;
    let (samples, _) = bytes[data + 8..data + 8 + len].as_chunks::<2>();
    samples.iter().map(|&b| i16::from_le_bytes(b)).collect()
}

/// The analysis of every 10 ms frame of the speech.
struct Speech {
    pcm: Vec<i16>,
    /// The 36 features of libopus's extractor: 18 cepstral values, the
    /// pitch, the pitch correlation and 16 LPC coefficients.
    features: Vec<[f32; NB_TOTAL_FEATURES]>,
    /// PitchDNN's inputs for the frame, read from the extractor's state.
    if_features: Vec<[f32; IF_FEATURES]>,
    xcorr: Vec<[f32; XCORR_FEATURES]>,
    /// PitchDNN's result for the frame (`st->dnn_pitch`).
    dnn_pitch: Vec<f32>,
    /// The Burg cepstrum of the frame, as `lpcnet_plc_conceal` computes it.
    burg: Vec<[f32; 2 * NB_BANDS]>,
}

impl Speech {
    fn frames(&self) -> usize {
        self.features.len()
    }

    /// The pitch period of a frame in samples, as FARGAN computes it
    /// (`dnn/fargan.c:204`).
    fn period(&self, t: usize) -> usize {
        let p = self.features[t][NB_BANDS] as f64;
        (0.5 + 256.0 / 2f64.powf((1.0 / 60.0) * ((p + 1.5) * 60.0))).floor() as usize
    }

    /// The pitch correlation of a frame, in [0, 1].
    #[cfg_attr(not(feature = "osce"), allow(dead_code))]
    fn voicing(&self, t: usize) -> f32 {
        self.features[t][NB_BANDS + 1] + 0.5
    }

    /// `len` samples from `start`, scaled to [-1, 1).
    fn signal(&self, start: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| self.pcm[(start + i) % self.pcm.len()] as f32 / 32768.0).collect()
    }
}

// The layout of `LPCNetEncState` (`dnn/lpcnet_private.h:25-42`), to read
// PitchDNN's inputs. The size is checked against `lpcnet_encoder_get_size`.
const PITCHDNN_STATE_SIZE: usize = size_of::<CPitchDNN>() + 4 * (64 + 2 * 226 + 2 * 3616);
const ENC_IF_FEATURES: usize = PITCHDNN_STATE_SIZE + 4 * (160 + 1) + 8 * 30;
const ENC_XCORR: usize = ENC_IF_FEATURES + 4 * IF_FEATURES;
const ENC_DNN_PITCH: usize = ENC_XCORR + 4 * XCORR_FEATURES;
const ENC_SIZE: usize = ENC_DNN_PITCH + 4 * (1 + 16 + 1 + 2 * 576 + 4 + 16 + 36 + 16 + 36);

fn read_f32s<const N: usize>(st: &[u8], offset: usize) -> [f32; N] {
    std::array::from_fn(|i| f32::from_ne_bytes(st[offset + 4 * i..offset + 4 * i + 4].try_into().unwrap()))
}

fn analyse_speech() -> Speech {
    let pcm = speech_samples();
    let size = unsafe { lpcnet_encoder_get_size() } as usize;
    assert_eq!(size, ENC_SIZE.next_multiple_of(8), "the layout of LPCNetEncState changed");
    // u64 storage gives the 8-byte alignment of the C struct.
    let mut storage = vec![0u64; size / 8];
    let st = storage.as_mut_ptr() as *mut u8;
    unsafe { lpcnet_encoder_init(st) };
    let arch = c_arch();
    let mut s = Speech {
        pcm: pcm.clone(),
        features: Vec::new(),
        if_features: Vec::new(),
        xcorr: Vec::new(),
        dnn_pitch: Vec::new(),
        burg: Vec::new(),
    };
    for frame in pcm.as_chunks::<FRAME>().0 {
        let x: [f32; FRAME] = frame.map(|v| v as f32);
        let mut features = [0.0f32; NB_TOTAL_FEATURES];
        unsafe { lpcnet_compute_single_frame_features_float(st, x.as_ptr(), features.as_mut_ptr(), arch) };
        // SAFETY: `storage` holds `size` bytes and nothing else refers to it.
        let bytes = unsafe { std::slice::from_raw_parts(st as *const u8, size) };
        s.if_features.push(read_f32s(bytes, ENC_IF_FEATURES));
        s.xcorr.push(read_f32s(bytes, ENC_XCORR));
        let [pitch] = read_f32s::<1>(bytes, ENC_DNN_PITCH);
        assert_eq!(pitch.to_bits(), features[NB_BANDS].to_bits(), "the layout of LPCNetEncState changed");
        s.dnn_pitch.push(pitch);
        s.features.push(features);
        let mut burg = [0.0f32; 2 * NB_BANDS];
        freq::burg_cepstral_analysis(&mut burg, &x);
        s.burg.push(burg);
    }
    s
}

fn speech() -> &'static Speech {
    static SPEECH: std::sync::OnceLock<Speech> = std::sync::OnceLock::new();
    SPEECH.get_or_init(analyse_speech)
}

// ---------------------------------------------------------------------------
// The models, at the level of their layer calls
// ---------------------------------------------------------------------------

/// PitchDNN (`dnn/pitchdnn.c:12-55`), one call per frame. Returns the
/// pitch of each frame, computed from the output as `compute_pitchdnn`
/// does.
fn run_pitchdnn(b: &mut dyn Backend, s: &Speech, _net: &Net) -> Vec<f32> {
    let mut gru_state = [0.0f32; 64];
    let mut xcorr_mem1 = vec![0.0f32; 226 * 2];
    let mut xcorr_mem2 = vec![0.0f32; 226 * 2 * 8];
    let mut pitches = Vec::new();
    for t in 0..s.frames() {
        let mut if1_out = [0.0f32; 64];
        let mut downsampler_in = [0.0f32; XCORR_FEATURES + 64];
        let mut downsampler_out = [0.0f32; 64];
        let mut conv1_tmp1 = vec![0.0f32; 226 * 8];
        let mut conv1_tmp2 = vec![0.0f32; 226 * 8];
        let mut output = [0.0f32; 192];
        b.dense("dense_if_upsampler_1", &mut if1_out, &s.if_features[t], Activation::Tanh);
        b.dense("dense_if_upsampler_2", &mut downsampler_in[XCORR_FEATURES..], &if1_out, Activation::Tanh);
        conv1_tmp1[1..1 + XCORR_FEATURES].copy_from_slice(&s.xcorr[t]);
        b.conv2d("conv2d_1", &mut conv1_tmp2[1..], &mut xcorr_mem1, &conv1_tmp1, 224, 226, Activation::Tanh);
        b.conv2d("conv2d_2", &mut downsampler_in, &mut xcorr_mem2, &conv1_tmp2, 224, 224, Activation::Tanh);
        b.dense("dense_downsampler", &mut downsampler_out, &downsampler_in, Activation::Tanh);
        b.gru("gru_1_input", "gru_1_recurrent", &mut gru_state, &downsampler_out);
        b.dense("dense_final_upsampler", &mut output, &gru_state, Activation::Linear);
        let mut pos = 0;
        let mut maxval = -1.0f32;
        for (i, &v) in output[..180].iter().enumerate() {
            if v > maxval {
                pos = i;
                maxval = v;
            }
        }
        let (mut sum, mut count) = (0.0f32, 0.0f32);
        let first = pos.saturating_sub(2);
        for (i, &v) in output.iter().enumerate().take((pos + 2).min(179) + 1).skip(first) {
            let p = (v as f64).exp() as f32;
            // clang contracts `sum += p*i` into a fused multiply-add.
            sum = p.mul_add(i as f32, sum);
            count += p;
        }
        // `(1.f/60.f)*(sum/count) - 1.5`: a float product, then a double
        // subtraction, then the float return value.
        pitches.push((((1.0f32 / 60.0) * (sum / count)) as f64 - 1.5) as f32);
    }
    pitches
}

/// The PLC model (`compute_plc_pred`, `dnn/lpcnet_plc.c:105-115`), one
/// call per frame. Frames come in runs: analysed frames (Burg cepstrum,
/// features, flag 1), lost frames (all zeros) and frames recovered from
/// in-band FEC data (features, flag -1), as `lpcnet_plc_conceal` and
/// `get_fec_or_pred` build them.
fn run_plc(b: &mut dyn Backend, s: &Speech, _net: &Net) -> Vec<f32> {
    let mut gru1 = [0.0f32; 192];
    let mut gru2 = [0.0f32; 192];
    for t in 0..s.frames() {
        let mut input = [0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
        match (t / 7) % 6 {
            4 => {} // a loss
            5 if t % 7 < 2 => {
                input[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&s.features[t][..NB_FEATURES]);
                input[2 * NB_BANDS + NB_FEATURES] = -1.0;
            }
            _ => {
                input[..2 * NB_BANDS].copy_from_slice(&s.burg[t]);
                input[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&s.features[t][..NB_FEATURES]);
                input[2 * NB_BANDS + NB_FEATURES] = 1.0;
            }
        }
        let mut tmp = [0.0f32; 128];
        let mut out = [0.0f32; NB_FEATURES];
        b.dense("plc_dense_in", &mut tmp, &input, Activation::Tanh);
        b.gru("plc_gru1_input", "plc_gru1_recurrent", &mut gru1, &tmp);
        let gru1_copy = gru1;
        b.gru("plc_gru2_input", "plc_gru2_recurrent", &mut gru2, &gru1_copy);
        b.dense("plc_dense_out", &mut out, &gru2, Activation::Linear);
    }
    Vec::new()
}

/// FARGAN (`compute_fargan_cond` and `run_fargan_subframe`,
/// `dnn/fargan.c:44-138`), one frame of four subframes per 10 ms. The
/// excitation buffer holds FARGAN's own output, and real speech every 25
/// frames.
fn run_fargan(b: &mut dyn Backend, s: &Speech, net: &Net) -> Vec<f32> {
    const SUB: usize = 40;
    const COND: usize = 80;
    const PITCH_MAX: usize = 256;
    let pembed = net.table("cond_net_pembed");
    let mut cond_conv1_state = [0.0f32; 128];
    let mut fwc0_mem = [0.0f32; 164];
    let mut gru1_state = [0.0f32; 160];
    let mut gru2_state = [0.0f32; 128];
    let mut gru3_state = [0.0f32; 128];
    let mut pitch_buf = [0.0f32; PITCH_MAX];
    for t in 0..s.frames() {
        if t % 25 == 0 {
            // Pre-emphasized speech, as `fargan_cont` loads it.
            let x = s.signal(t * FRAME + FRAME, PITCH_MAX + 1);
            for i in 0..PITCH_MAX {
                pitch_buf[i] = x[i + 1] - 0.85 * x[i];
            }
        }
        let period = s.period(t);
        let mut dense_in = [0.0f32; NB_FEATURES + 12];
        dense_in[..NB_FEATURES].copy_from_slice(&s.features[t][..NB_FEATURES]);
        let row = period.saturating_sub(32).min(223);
        dense_in[NB_FEATURES..].copy_from_slice(&pembed[row * 12..row * 12 + 12]);
        let mut conv1_in = [0.0f32; 64];
        let mut fdense2_in = [0.0f32; 128];
        let mut cond = [0.0f32; 320];
        b.dense("cond_net_fdense1", &mut conv1_in, &dense_in, Activation::Tanh);
        b.conv1d("cond_net_fconv1", &mut fdense2_in, &mut cond_conv1_state, &conv1_in, Activation::Tanh);
        b.dense("cond_net_fdense2", &mut cond, &fdense2_in, Activation::Tanh);
        for sub in 0..4 {
            let cond = &cond[sub * COND..(sub + 1) * COND];
            let mut gain = [0.0f32; 1];
            b.dense("sig_net_cond_gain_dense", &mut gain, cond, Activation::Linear);
            let gain = (gain[0] as f64).exp() as f32;
            let gain_1 = 1.0 / (1e-5 + gain);
            let mut pred = [0.0f32; SUB + 4];
            let mut pos = PITCH_MAX as isize - period as isize - 2;
            for p in pred.iter_mut() {
                *p = (gain_1 * pitch_buf[pos.max(0) as usize]).clamp(-1.0, 1.0);
                pos += 1;
                if pos == PITCH_MAX as isize {
                    pos -= period as isize;
                }
            }
            let prev: [f32; SUB] =
                std::array::from_fn(|i| (gain_1 * pitch_buf[PITCH_MAX - SUB + i]).clamp(-1.0, 1.0));
            let mut fwc0_in = [0.0f32; COND + SUB + 4 + SUB];
            fwc0_in[..COND].copy_from_slice(cond);
            fwc0_in[COND..COND + SUB + 4].copy_from_slice(&pred);
            fwc0_in[COND + SUB + 4..].copy_from_slice(&prev);

            let mut gru1_in = [0.0f32; 192 + 2 * SUB];
            let mut gru2_in = [0.0f32; 160 + 2 * SUB];
            let mut gru3_in = [0.0f32; 128 + 2 * SUB];
            let mut skip_cat = [0.0f32; 688];
            let mut skip_out = [0.0f32; 128];
            let mut pitch_gate = [0.0f32; 4];
            let mut pcm = [0.0f32; SUB];
            b.conv1d("sig_net_fwc0_conv", &mut gru1_in[..192], &mut fwc0_mem, &fwc0_in, Activation::Tanh);
            b.glu_in_place("sig_net_fwc0_glu_gate", &mut gru1_in[..192]);
            b.dense("sig_net_gain_dense_out", &mut pitch_gate, &gru1_in, Activation::Sigmoid);
            for i in 0..SUB {
                gru1_in[192 + i] = pitch_gate[0] * pred[i + 2];
            }
            gru1_in[192 + SUB..].copy_from_slice(&prev);
            b.gru("sig_net_gru1_input", "sig_net_gru1_recurrent", &mut gru1_state, &gru1_in);
            b.glu("sig_net_gru1_glu_gate", &mut gru2_in[..160], &gru1_state);
            for i in 0..SUB {
                gru2_in[160 + i] = pitch_gate[1] * pred[i + 2];
            }
            gru2_in[160 + SUB..].copy_from_slice(&prev);
            b.gru("sig_net_gru2_input", "sig_net_gru2_recurrent", &mut gru2_state, &gru2_in);
            b.glu("sig_net_gru2_glu_gate", &mut gru3_in[..128], &gru2_state);
            for i in 0..SUB {
                gru3_in[128 + i] = pitch_gate[2] * pred[i + 2];
            }
            gru3_in[128 + SUB..].copy_from_slice(&prev);
            b.gru("sig_net_gru3_input", "sig_net_gru3_recurrent", &mut gru3_state, &gru3_in);
            b.glu("sig_net_gru3_glu_gate", &mut skip_cat[288..416], &gru3_state);
            skip_cat[..160].copy_from_slice(&gru2_in[..160]);
            skip_cat[160..288].copy_from_slice(&gru3_in[..128]);
            skip_cat[416..608].copy_from_slice(&gru1_in[..192]);
            for i in 0..SUB {
                skip_cat[608 + i] = pitch_gate[3] * pred[i + 2];
            }
            skip_cat[648..].copy_from_slice(&prev);
            b.dense("sig_net_skip_dense", &mut skip_out, &skip_cat, Activation::Tanh);
            b.glu_in_place("sig_net_skip_glu_gate", &mut skip_out);
            b.dense("sig_net_sig_dense_out", &mut pcm, &skip_out, Activation::Tanh);
            for v in pcm.iter_mut() {
                *v *= gain;
            }
            pitch_buf.copy_within(SUB.., 0);
            pitch_buf[PITCH_MAX - SUB..].copy_from_slice(&pcm);
        }
    }
    Vec::new()
}

/// The temporal envelope of the shaping layers (`adashape_process_frame`,
/// `dnn/nndsp.c:363-393`): `tenv_size` log averages, their mean removed,
/// then the mean.
#[cfg(feature = "osce")]
fn temporal_envelope(x: &[f32], avg_pool_k: usize) -> Vec<f32> {
    let n = x.len() / avg_pool_k;
    let f = 1.0 / avg_pool_k as f32;
    let mut tenv: Vec<f32> = (0..n)
        .map(|i| {
            let sum: f32 = x[i * avg_pool_k..(i + 1) * avg_pool_k].iter().map(|v| v.abs()).sum();
            (sum * f + 1.525_878_9e-5).ln()
        })
        .collect();
    let mean = tenv.iter().sum::<f32>() / n as f32;
    for v in tenv.iter_mut() {
        *v -= mean;
    }
    tenv.push(mean);
    tenv
}

/// The state of one shaping layer (`AdaShapeState`).
#[cfg(feature = "osce")]
struct Shape {
    alpha1f: Vec<f32>,
    alpha1t: Vec<f32>,
    alpha2: Vec<f32>,
    interpolate: f32,
}

#[cfg(feature = "osce")]
impl Shape {
    fn new(feature_dim: usize, tenv_size: usize, hidden_dim: usize) -> Self {
        Shape {
            alpha1f: vec![0.0; feature_dim],
            alpha1t: vec![0.0; tenv_size + 1],
            alpha2: vec![0.0; hidden_dim],
            interpolate: 0.0,
        }
    }

    /// The layer calls of `adashape_process_frame` (`dnn/nndsp.c:345-440`)
    /// for the signal `x`, with the layers `<prefix>_alpha1_f`,
    /// `<prefix>_alpha1_t` and `<prefix>_alpha2`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        b: &mut dyn Backend,
        names: [&'static str; 3],
        features: &[f32],
        x: &[f32],
        avg_pool_k: usize,
        interpolate_k: usize,
    ) {
        let frame_size = x.len();
        let hidden_dim = frame_size / interpolate_k;
        let tenv = temporal_envelope(x, avg_pool_k);
        let mut out = vec![0.0f32; frame_size];
        let mut tmp = vec![0.0f32; frame_size];
        b.conv1d(names[0], &mut out, &mut self.alpha1f, features, Activation::Linear);
        b.conv1d(names[1], &mut tmp, &mut self.alpha1t, &tenv, Activation::Linear);
        let hidden: Vec<f32> = (0..hidden_dim)
            .map(|i| {
                let v = out[i] + tmp[i];
                if v >= 0.0 { v } else { (0.2 * v as f64) as f32 }
            })
            .collect();
        b.conv1d(names[2], &mut tmp, &mut self.alpha2, &hidden, Activation::Linear);
        for i in 0..hidden_dim {
            for k in 0..interpolate_k {
                let alpha = (k + 1) as f32 / interpolate_k as f32;
                out[i * interpolate_k + k] = alpha * tmp[i] + (1.0 - alpha) * self.interpolate;
            }
            self.interpolate = tmp[i];
        }
        b.exp(names[2], &mut out);
    }
}

/// The OSCE features of one 20 ms frame (4 subframes of 93 values), the
/// bit counts and the pitch lags, built from the speech analysis in the
/// layout of `osce_calculate_features` (`dnn/osce_features.c:368-458`).
#[cfg(feature = "osce")]
struct OsceInput {
    features: Vec<f32>,
    numbits: [f32; 2],
    periods: [usize; 4],
}

#[cfg(feature = "osce")]
fn osce_inputs(s: &Speech) -> Vec<OsceInput> {
    let mut smooth = 0.0f32;
    (0..s.frames() / 2)
        .map(|f| {
            let num_bits = 120 + (f * 37) % 520;
            smooth = 0.9 * smooth + 0.1 * num_bits as f32;
            let mut features = vec![0.0f32; 4 * 93];
            let mut periods = [0usize; 4];
            for k in 0..4 {
                let t = 2 * f + k / 2;
                let p = &mut features[k * 93..(k + 1) * 93];
                // Clean spectrum: 0.3·ln of 1/|A| on 64 bands, from the LPC
                // coefficients of the frame.
                let lpc = &s.features[t][NB_BANDS + 2..];
                for (band, v) in p[..64].iter_mut().enumerate() {
                    let w = std::f64::consts::PI * (band as f64 + 0.5) / 64.0;
                    let (mut re, mut im) = (1.0f64, 0.0f64);
                    for (j, &a) in lpc.iter().enumerate() {
                        re += a as f64 * (w * (j + 1) as f64).cos();
                        im -= a as f64 * (w * (j + 1) as f64).sin();
                    }
                    *v = (0.3 * (1.0 / (re.hypot(im) + 1e-9) + 1e-9).ln()) as f32;
                }
                // Noisy cepstrum: the frame's cepstrum, from log10 of the
                // power to ln of the magnitude.
                for i in 0..NB_BANDS {
                    p[64 + i] = 1.151_293 * s.features[t][i];
                }
                let voiced = s.voicing(t) > 0.3;
                let lag = if voiced { s.period(t).clamp(32, 288) } else { 7 };
                periods[k] = lag;
                // Normalized autocorrelation at the lag, for lags - 2 to + 2.
                let x = s.signal((2 * f) * FRAME + k * 80 + 288 * 2, 80 + 300);
                let frame = &x[300..];
                for d in 0..5 {
                    let back = 300 - lag - 2 + d;
                    let (mut xx, mut yy, mut xy) = (0.0f64, 0.0f64, 0.0f64);
                    for n in 0..80 {
                        let (a, c) = (frame[n] as f64, x[back + n] as f64);
                        xx += a * a;
                        yy += c * c;
                        xy += a * c;
                    }
                    p[82 + d] = (xy / (xx * yy + 1e-9).sqrt()) as f32;
                }
                // LTP: a symmetric 5-tap filter scaled by the voicing.
                if voiced {
                    let v = s.voicing(t);
                    for (d, c) in [-0.05f32, 0.2, 0.6, 0.2, -0.05].iter().enumerate() {
                        p[87 + d] = c * v;
                    }
                }
                // Log gain: SILK's gain follows the level of the residual,
                // here taken as a fifth of the level of the subframe.
                let rms = (frame.iter().map(|v| (v * 32768.0) as f64 * (v * 32768.0) as f64).sum::<f64>() / 80.0).sqrt();
                p[92] = (0.2 * rms + 1.0).ln() as f32;
            }
            OsceInput { features, numbits: [num_bits as f32, smooth], periods }
        })
        .collect()
}

/// `compute_lace_numbits_embedding` and its NoLACE twin
/// (`dnn/osce.c:120-136`).
#[cfg(feature = "osce")]
fn numbits_embedding(numbits: f32, scales: &[f32; 8]) -> [f32; 8] {
    let (lo, hi) = ((50.0f64).ln() as f32, (650.0f64).ln() as f32);
    let x = ((numbits as f64).ln() as f32).clamp(lo, hi) - (hi + lo) / 2.0;
    std::array::from_fn(|k| ((x * scales[k] - 0.5) as f64).sin() as f32)
}

/// The feature network of LACE and NoLACE (`lace_feature_net`,
/// `dnn/osce.c:162-240`, and `nolace_feature_net`), with the layers
/// `names`: conv1, conv2, tconv, and the input and recurrent parts of the
/// GRU. Returns the 4 conditioning vectors of the frame.
#[cfg(feature = "osce")]
#[allow(clippy::too_many_arguments)]
fn osce_feature_net(
    b: &mut dyn Backend,
    names: [&'static str; 5],
    pitch_table: &[f32],
    scales: &[f32; 8],
    input: &OsceInput,
    hidden: usize,
    cond: usize,
    conv2_state: &mut [f32],
    gru_state: &mut [f32],
) -> Vec<f32> {
    let mut emb = [0.0f32; 16];
    emb[..8].copy_from_slice(&numbits_embedding(input.numbits[0], scales));
    emb[8..].copy_from_slice(&numbits_embedding(input.numbits[1], scales));
    let mut conv1_out = vec![0.0f32; 4 * hidden];
    for k in 0..4 {
        let mut x = Vec::with_capacity(93 + 64 + 16);
        x.extend_from_slice(&input.features[k * 93..(k + 1) * 93]);
        let period = input.periods[k];
        x.extend_from_slice(&pitch_table[period * 64..(period + 1) * 64]);
        x.extend_from_slice(&emb);
        b.conv1d(names[0], &mut conv1_out[k * hidden..(k + 1) * hidden], &mut [], &x, Activation::Tanh);
    }
    let mut conv2_out = vec![0.0f32; cond];
    b.conv1d(names[1], &mut conv2_out, conv2_state, &conv1_out, Activation::Tanh);
    let mut tconv_out = vec![0.0f32; 4 * cond];
    b.dense(names[2], &mut tconv_out, &conv2_out, Activation::Tanh);
    let mut out = vec![0.0f32; 4 * cond];
    for k in 0..4 {
        b.gru(names[3], names[4], gru_state, &tconv_out[k * cond..(k + 1) * cond]);
        out[k * cond..(k + 1) * cond].copy_from_slice(gru_state);
    }
    out
}

/// The layer calls of `adacomb_process_frame` (`dnn/nndsp.c:285-287`).
#[cfg(feature = "osce")]
fn adacomb(b: &mut dyn Backend, names: [&'static str; 3], features: &[f32]) {
    let mut kernel = [0.0f32; 16];
    let (mut gain, mut global_gain) = ([0.0f32; 1], [0.0f32; 1]);
    b.dense(names[0], &mut kernel, features, Activation::Linear);
    b.dense(names[1], &mut gain, features, Activation::Relu);
    b.dense(names[2], &mut global_gain, features, Activation::Tanh);
}

/// The layer calls of `adaconv_process_frame` (`dnn/nndsp.c:189-190`).
#[cfg(feature = "osce")]
fn adaconv(b: &mut dyn Backend, names: [&'static str; 2], features: &[f32], kernel_size: usize, gains: usize) {
    let mut kernel = vec![0.0f32; kernel_size];
    let mut gain = vec![0.0f32; gains];
    b.dense(names[0], &mut kernel, features, Activation::Linear);
    b.dense(names[1], &mut gain, features, Activation::Tanh);
}

/// LACE (`lace_process_20ms_frame`, `dnn/osce.c:243-340`).
#[cfg(feature = "osce")]
fn run_lace(b: &mut dyn Backend, s: &Speech, net: &Net) -> Vec<f32> {
    use opus1::internals::dnn::models::lace::*;
    let scales = [
        LACE_NUMBITS_SCALE_0,
        LACE_NUMBITS_SCALE_1,
        LACE_NUMBITS_SCALE_2,
        LACE_NUMBITS_SCALE_3,
        LACE_NUMBITS_SCALE_4,
        LACE_NUMBITS_SCALE_5,
        LACE_NUMBITS_SCALE_6,
        LACE_NUMBITS_SCALE_7,
    ];
    let table = net.table("lace_pitch_embedding");
    let mut conv2_state = vec![0.0f32; LACE_FNET_CONV2_STATE_SIZE];
    let mut gru_state = vec![0.0f32; LACE_COND_DIM];
    for input in osce_inputs(s) {
        let fb = osce_feature_net(
            b,
            ["lace_fnet_conv1", "lace_fnet_conv2", "lace_fnet_tconv", "lace_fnet_gru_input", "lace_fnet_gru_recurrent"],
            table,
            &scales,
            &input,
            LACE_HIDDEN_FEATURE_DIM,
            LACE_COND_DIM,
            &mut conv2_state,
            &mut gru_state,
        );
        let sub = |k: usize| &fb[k * LACE_COND_DIM..(k + 1) * LACE_COND_DIM];
        for k in 0..4 {
            adacomb(b, ["lace_cf1_kernel", "lace_cf1_gain", "lace_cf1_global_gain"], sub(k));
        }
        for k in 0..4 {
            adacomb(b, ["lace_cf2_kernel", "lace_cf2_gain", "lace_cf2_global_gain"], sub(k));
        }
        for k in 0..4 {
            adaconv(b, ["lace_af1_kernel", "lace_af1_gain"], sub(k), LACE_AF1_KERNEL_OUT_SIZE, LACE_AF1_GAIN_OUT_SIZE);
        }
    }
    Vec::new()
}

/// NoLACE (`nolace_process_20ms_frame`, `dnn/osce.c:509-845`).
#[cfg(feature = "osce")]
fn run_nolace(b: &mut dyn Backend, s: &Speech, net: &Net) -> Vec<f32> {
    use opus1::internals::dnn::models::nolace::*;
    let scales = [
        NOLACE_NUMBITS_SCALE_0,
        NOLACE_NUMBITS_SCALE_1,
        NOLACE_NUMBITS_SCALE_2,
        NOLACE_NUMBITS_SCALE_3,
        NOLACE_NUMBITS_SCALE_4,
        NOLACE_NUMBITS_SCALE_5,
        NOLACE_NUMBITS_SCALE_6,
        NOLACE_NUMBITS_SCALE_7,
    ];
    const C: usize = NOLACE_COND_DIM;
    let table = net.table("nolace_pitch_embedding");
    let mut conv2_state = vec![0.0f32; NOLACE_FNET_CONV2_STATE_SIZE];
    let mut gru_state = vec![0.0f32; C];
    let mut post: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; C]).collect();
    let mut shapes: Vec<Shape> = (0..3).map(|_| Shape::new(C, 20, 80)).collect();
    let post_names = ["nolace_post_cf1", "nolace_post_cf2", "nolace_post_af1", "nolace_post_af2", "nolace_post_af3"];
    let shape_names = [
        ["nolace_tdshape1_alpha1_f", "nolace_tdshape1_alpha1_t", "nolace_tdshape1_alpha2"],
        ["nolace_tdshape2_alpha1_f", "nolace_tdshape2_alpha1_t", "nolace_tdshape2_alpha2"],
        ["nolace_tdshape3_alpha1_f", "nolace_tdshape3_alpha1_t", "nolace_tdshape3_alpha2"],
    ];
    for (f, input) in osce_inputs(s).iter().enumerate() {
        let mut fb = osce_feature_net(
            b,
            [
                "nolace_fnet_conv1",
                "nolace_fnet_conv2",
                "nolace_fnet_tconv",
                "nolace_fnet_gru_input",
                "nolace_fnet_gru_recurrent",
            ],
            table,
            &scales,
            input,
            NOLACE_HIDDEN_FEATURE_DIM,
            C,
            &mut conv2_state,
            &mut gru_state,
        );
        // The 80-sample signal of subframe k that the shaping layers see.
        let x = |k: usize| s.signal(2 * f * FRAME + k * 80, 80);
        // Six stages over the 4 subframes: comb filters 1 and 2, adaptive
        // filter 1, then three rounds of shaping and adaptive filtering.
        // The first five end with a conv1d that transforms the features.
        for stage in 0..6 {
            let mut transformed = vec![0.0f32; 4 * C];
            for k in 0..4 {
                let feat = fb[k * C..(k + 1) * C].to_vec();
                match stage {
                    0 => adacomb(b, ["nolace_cf1_kernel", "nolace_cf1_gain", "nolace_cf1_global_gain"], &feat),
                    1 => adacomb(b, ["nolace_cf2_kernel", "nolace_cf2_gain", "nolace_cf2_global_gain"], &feat),
                    2 => adaconv(b, ["nolace_af1_kernel", "nolace_af1_gain"], &feat, 32, 2),
                    3 => {
                        shapes[0].run(b, shape_names[0], &feat, &x(k), 4, 1);
                        adaconv(b, ["nolace_af2_kernel", "nolace_af2_gain"], &feat, 64, 2);
                    }
                    4 => {
                        shapes[1].run(b, shape_names[1], &feat, &x(k), 4, 1);
                        adaconv(b, ["nolace_af3_kernel", "nolace_af3_gain"], &feat, 64, 2);
                    }
                    _ => {
                        shapes[2].run(b, shape_names[2], &feat, &x(k), 4, 1);
                        adaconv(b, ["nolace_af4_kernel", "nolace_af4_gain"], &feat, 32, 1);
                    }
                }
                if stage < 5 {
                    b.conv1d(
                        post_names[stage],
                        &mut transformed[k * C..(k + 1) * C],
                        &mut post[stage],
                        &feat,
                        Activation::Tanh,
                    );
                }
            }
            if stage < 5 {
                fb = transformed;
            }
        }
    }
    Vec::new()
}

/// BBWENet (`bbwe_feature_net` and `bbwenet_process_frames`,
/// `dnn/osce.c:855-1340`), on the features of `osce_bwe_calculate_features`
/// for 20 ms of speech.
#[cfg(feature = "osce-bwe")]
fn run_bbwenet(b: &mut dyn Backend, s: &Speech, _net: &Net) -> Vec<f32> {
    const C: usize = 128;
    let mut feature_state = vec![0.0f32; 160 + 82];
    let mut conv1_state = vec![0.0f32; 228];
    let mut conv2_state = vec![0.0f32; 256];
    let mut gru_state = vec![0.0f32; C];
    let mut shape1 = Shape::new(C, 20, 80);
    let mut shape2 = Shape::new(C, 20, 120);
    for f in 0..s.frames() / 2 {
        let xq = &s.pcm[2 * f * FRAME..2 * f * FRAME + 2 * FRAME];
        let mut features = vec![0.0f32; 2 * 114];
        unsafe { osce_bwe_calculate_features(feature_state.as_mut_ptr(), features.as_mut_ptr(), xq.as_ptr(), 320) };
        let mut out1 = vec![0.0f32; 2 * C];
        for i in 0..2 {
            b.conv1d(
                "bbwenet_fnet_conv1",
                &mut out1[i * C..(i + 1) * C],
                &mut conv1_state,
                &features[i * 114..(i + 1) * 114],
                Activation::Tanh,
            );
        }
        let mut out2 = vec![0.0f32; 2 * C];
        for i in 0..2 {
            b.conv1d("bbwenet_fnet_conv2", &mut out2[i * C..(i + 1) * C], &mut conv2_state, &out1[i * C..(i + 1) * C], Activation::Tanh);
        }
        let mut tconv = vec![0.0f32; 4 * C];
        for i in 0..2 {
            b.dense("bbwenet_fnet_tconv", &mut tconv[i * 2 * C..(i + 1) * 2 * C], &out2[i * C..(i + 1) * C], Activation::Tanh);
        }
        let mut latent = vec![0.0f32; 4 * C];
        for k in 0..4 {
            b.gru("bbwenet_fnet_gru_input", "bbwenet_fnet_gru_recurrent", &mut gru_state, &tconv[k * C..(k + 1) * C]);
            latent[k * C..(k + 1) * C].copy_from_slice(&gru_state);
        }
        let sub = |k: usize| &latent[k * C..(k + 1) * C];
        for k in 0..4 {
            adaconv(b, ["bbwenet_af1_kernel", "bbwenet_af1_gain"], sub(k), 48, 3);
        }
        for k in 0..4 {
            let x = s.signal(2 * f * FRAME + k * 80, 160);
            shape1.run(
                b,
                ["bbwenet_tdshape1_alpha1_f", "bbwenet_tdshape1_alpha1_t", "bbwenet_tdshape1_alpha2"],
                sub(k),
                &x,
                8,
                2,
            );
        }
        for k in 0..4 {
            adaconv(b, ["bbwenet_af2_kernel", "bbwenet_af2_gain"], sub(k), 288, 3);
        }
        for k in 0..4 {
            let x = s.signal(2 * f * FRAME + k * 80, 240);
            shape2.run(
                b,
                ["bbwenet_tdshape2_alpha1_f", "bbwenet_tdshape2_alpha1_t", "bbwenet_tdshape2_alpha2"],
                sub(k),
                &x,
                12,
                2,
            );
        }
        for k in 0..4 {
            adaconv(b, ["bbwenet_af3_kernel", "bbwenet_af3_gain"], sub(k), 48, 1);
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

/// The layers that the model code reads only as embedding tables.
const TABLES: [&str; 3] = ["cond_net_pembed", "lace_pitch_embedding", "nolace_pitch_embedding"];

/// Runs check 1 and check 2 on one model.
fn check_model(key: &str, run: fn(&mut dyn Backend, &Speech, &Net) -> Vec<f32>) -> Vec<f32> {
    let s = speech();
    let net = net(key);
    let measure = o2_mode();

    // Check 1: layer by layer.
    let mut check = Check::new(&net, measure);
    let result = run(&mut check, s, &net);
    for name in &net.names {
        assert!(
            check.stats.contains_key(name) || TABLES.contains(name),
            "{key}: layer {name} was never called"
        );
    }

    // Check 2: each side on its own outputs.
    let mut c = Single::new(&net, Side::C);
    run(&mut c, s, &net);
    let mut drift: BTreeMap<&'static str, Stat> = BTreeMap::new();
    for a in archs(measure) {
        let mut r = Single::new(&net, Side::Rust(a));
        run(&mut r, s, &net);
        assert_eq!(c.calls.len(), r.calls.len());
        assert_eq!(c.values.len(), r.values.len());
        for (k, &(name, start)) in c.calls.iter().enumerate() {
            let end = c.calls.get(k + 1).map_or(c.values.len(), |x| x.1);
            let st = drift.entry(name).or_default();
            st.calls += 1;
            for i in start..end {
                let (x, y) = (c.values[i], r.values[i]);
                st.values += 1;
                if same(x, y) {
                    continue;
                }
                assert!(
                    measure,
                    "{key}, {a:?}, running on its own outputs: {name}, call {k}: value {} is {y:e}, libopus has {x:e}",
                    i - start
                );
                st.differ += 1;
                let d = (x as f64 - y as f64).abs();
                st.max_abs = st.max_abs.max(d);
                st.max_rel = st.max_rel.max(d / (x.abs() as f64).max(1e-3));
            }
        }
    }

    let calls: usize = check.stats.values().map(|s| s.calls).sum();
    println!("{key}: {calls} layer calls on {} frames of speech", s.frames());
    if measure {
        println!("{key}, single calls against the -O2 build (layers that differ):");
        println!("  {:<32} {:>8} {:>10} {:>12} {:>12} {:>8}", "layer", "calls", "differ", "max |d|", "max d/Σ|wx|", "d/bound");
        for (name, st) in check.stats.iter().filter(|(_, st)| st.differ > 0) {
            println!(
                "  {name:<32} {:>8} {:>5}/{:<6} {:>12.3e} {:>12.3e} {:>8.3}",
                st.calls, st.differ, st.values, st.max_abs, st.max_rel, st.max_bound
            );
        }
        println!("{key}, each side on its own outputs, against the -O2 build (layers that differ):");
        println!("  {:<32} {:>8} {:>16} {:>12} {:>12}", "layer", "calls", "differ", "max |d|", "max d/|x|");
        for (name, st) in drift.iter().filter(|(_, st)| st.differ > 0) {
            println!(
                "  {name:<32} {:>8} {:>8}/{:<8} {:>12.3e} {:>12.3e}",
                st.calls, st.differ, st.values, st.max_abs, st.max_rel
            );
        }
        for (name, st) in &check.stats {
            assert!(
                st.max_bound <= 1.0,
                "{key}: {name} differs from the -O2 build beyond the bound of a float sum ({})",
                st.max_bound
            );
        }
    }
    result
}

#[test]
fn pitchdnn_layers_match_libopus() {
    let pitches = check_model("pitchdnn", run_pitchdnn);
    // libopus's own PitchDNN ran inside the feature extractor on the same
    // inputs. The pitch computed here from the layer outputs must be its
    // pitch: this checks that the calls above are those of
    // `compute_pitchdnn`.
    //
    // The `-O2` build computes the last step (`exp` of 5 outputs and their
    // weighted mean) with other roundings, so there the difference is
    // only measured.
    let s = speech();
    let mut differ = 0;
    let mut max_d = 0.0f32;
    for (t, (&p, &want)) in pitches.iter().zip(&s.dnn_pitch).enumerate() {
        if o2_mode() {
            differ += usize::from(p.to_bits() != want.to_bits());
            max_d = max_d.max((p - want).abs());
        } else {
            assert_eq!(p.to_bits(), want.to_bits(), "frame {t}: pitch {p}, compute_pitchdnn gave {want}");
        }
    }
    if o2_mode() {
        println!(
            "pitchdnn: the pitch of compute_pitchdnn differs in {differ} of {} frames, by at most {max_d:e}",
            pitches.len()
        );
    }
}

#[test]
fn plc_layers_match_libopus() {
    check_model("plc", run_plc);
}

#[test]
fn fargan_layers_match_libopus() {
    check_model("fargan", run_fargan);
}

#[cfg(feature = "osce")]
#[test]
fn lace_layers_match_libopus() {
    check_model("lace", run_lace);
}

#[cfg(feature = "osce")]
#[test]
fn nolace_layers_match_libopus() {
    check_model("nolace", run_nolace);
}

#[cfg(feature = "osce-bwe")]
#[test]
fn bbwenet_layers_match_libopus() {
    check_model("bbwenet", run_bbwenet);
}

