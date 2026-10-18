#![cfg(feature = "test-libopus-dnn")]
//! The inference kernels and layer functions against libopus 1.6.1 built
//! with its neural network code (`scripts/android/build-libopus.sh --dnn`).
//!
//! The test calls libopus's own kernel functions (`compute_linear_c`,
//! `compute_linear_dotprod`, `compute_activation_c`, ...) on random layers
//! and random and edge inputs, and compares every output and every memory
//! buffer with opus1's functions for every `DnnArch` this processor can
//! run. The comparison is bit for bit; two NaNs count as equal.
//!
//! The bit-exact reference is the build without loop vectorization
//! (`plan.md` section 1):
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,dnn --test dnn_kernels_libopus
//! ```
//!
//! Against the plain `-O2` build of Apple clang, the layers with one float
//! output differ (Apple clang vectorizes that sum, `nnet.md` section 6.3).
//! Set `OPUS1_DNN_REF_O2=1` to compare those layers within the bound
//! `2·M·2^-24·Σ|w·x|` (`nnet.md`, Review item 13) and print the largest
//! difference seen; every other output stays exact.

use opus1::internals::dnn::nnet::{
    Conv2dLayer, LinearLayer, compute_activation, compute_activation_in_place, compute_conv2d,
    compute_generic_conv1d, compute_generic_conv1d_dilation, compute_generic_dense,
    compute_generic_gru, compute_glu, compute_glu_in_place, compute_linear,
};
use opus1::internals::dnn::{Activation, arch};
use std::os::raw::c_int;
use std::ptr;

/// libopus `LinearLayer` (`dnn/nnet.h:66-76`).
#[repr(C)]
struct CLinear {
    bias: *const f32,
    subias: *const f32,
    weights: *const i8,
    float_weights: *const f32,
    weights_idx: *const c_int,
    diag: *const f32,
    scale: *const f32,
    nb_inputs: c_int,
    nb_outputs: c_int,
}

/// libopus `Conv2dLayer` (`dnn/nnet.h:79-86`).
#[repr(C)]
struct CConv2d {
    bias: *const f32,
    float_weights: *const f32,
    in_channels: c_int,
    out_channels: c_int,
    ktime: c_int,
    kheight: c_int,
}

unsafe extern "C" {
    fn opus_select_arch() -> c_int;
    fn compute_linear_c(l: *const CLinear, out: *mut f32, input: *const f32);
    fn compute_linear_dotprod(l: *const CLinear, out: *mut f32, input: *const f32);
    fn compute_activation_c(out: *mut f32, input: *const f32, n: c_int, act: c_int);
    fn compute_activation_dotprod(out: *mut f32, input: *const f32, n: c_int, act: c_int);
    #[allow(clippy::too_many_arguments)]
    fn compute_conv2d_c(
        c: *const CConv2d,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        height: c_int,
        hstride: c_int,
        act: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    fn compute_conv2d_dotprod(
        c: *const CConv2d,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        height: c_int,
        hstride: c_int,
        act: c_int,
    );
    #[link_name = "compute_generic_dense"]
    fn c_generic_dense(l: *const CLinear, out: *mut f32, input: *const f32, act: c_int, arch: c_int);
    #[link_name = "compute_generic_gru"]
    fn c_generic_gru(
        input_w: *const CLinear,
        recurrent_w: *const CLinear,
        state: *mut f32,
        input: *const f32,
        arch: c_int,
    );
    #[link_name = "compute_glu"]
    fn c_glu(l: *const CLinear, out: *mut f32, input: *const f32, arch: c_int);
    #[link_name = "compute_generic_conv1d"]
    fn c_generic_conv1d(
        l: *const CLinear,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        input_size: c_int,
        act: c_int,
        arch: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    #[link_name = "compute_generic_conv1d_dilation"]
    fn c_generic_conv1d_dilation(
        l: *const CLinear,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        input_size: c_int,
        dilation: c_int,
        act: c_int,
        arch: c_int,
    );
}

/// The arch value of libopus for DotProd (`OPUS_ARCH_ARM_DOTPROD`).
const ARCH_DOTPROD: c_int = 4;

fn c_arch() -> c_int {
    // Makes sure the static libopus is linked (as the other libopus tests
    // do), then asks it for its arch.
    let _ = unsafe { audiopus_sys::opus_get_version_string() };
    unsafe { opus_select_arch() }
}

fn o2_mode() -> bool {
    std::env::var_os("OPUS1_DNN_REF_O2").is_some()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn f(&mut self, a: f32) -> f32 {
        a * (2.0 * (self.next() as f32 / 4_294_967_296.0) - 1.0)
    }
    fn w8(&mut self) -> i8 {
        (self.below(255) as i32 - 127) as i8
    }
}

/// Edge inputs: signed zeros, values where 127·x lies on a half integer,
/// the wrap-around of the quantizer, the clamps of `exp`, the NaN limits
/// of TANH (2.31313e9) and SIGMOID (4.62627e9) on both sides, infinities
/// and subnormals.
const EDGE: [f32; 36] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5 / 127.0,
    1.5 / 127.0,
    2.5 / 127.0,
    -0.5 / 127.0,
    -2.5 / 127.0,
    1.003_937,
    1.007_874,
    1.5,
    -1.5,
    2.0,
    -2.0,
    3.0,
    88.0,
    -88.0,
    89.0,
    -89.0,
    1e10,
    -1e10,
    1e20,
    -1e20,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1e-40,
    -1e-40,
    7.3e9,
    5.4e7,
    2.313e9,
    2.3132e9,
    4.626e9,
    4.6263e9,
    -2.3132e9,
    1.3e8,
];

fn input(rng: &mut Rng, n: usize, trial: usize) -> Vec<f32> {
    (0..n)
        .map(|i| match trial {
            0..=2 => EDGE[(i * 7 + trial) % EDGE.len()],
            3..=19 => rng.f(1.0),
            20..=24 => rng.f(1.6),
            25..=27 => rng.f(4.0),
            _ => rng.f(30.0),
        })
        .collect()
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// The first difference, for the failure message.
fn first_diff(a: &[f32], b: &[f32]) -> String {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()) {
            return format!("[{i}]: opus1 {x:e} ({:08x}) libopus {y:e} ({:08x})", x.to_bits(), y.to_bits());
        }
    }
    String::from("lengths differ")
}

/// A random layer and its arrays.
struct Owned {
    bias: Vec<f32>,
    fw: Vec<f32>,
    w8: Vec<i8>,
    idx: Vec<i32>,
    scale: Vec<f32>,
    diag: Vec<f32>,
    kind: u8,
    nin: usize,
    nout: usize,
    with_bias: bool,
    with_diag: bool,
}

/// kind: 0 float, 1 int8, 2 int8 sparse, 3 float sparse.
fn make(rng: &mut Rng, nin: usize, nout: usize, kind: u8, with_bias: bool, with_diag: bool) -> Owned {
    let bias = (0..nout).map(|_| rng.f(0.5)).collect();
    let scale = (0..nout).map(|_| (1.0 / 128.0 / 127.0) * (0.5 + rng.below(1000) as f32 / 1000.0)).collect();
    let diag = (0..nout).map(|_| rng.f(0.3)).collect();
    let mut idx = Vec::new();
    let mut blocks = 0;
    if kind >= 2 {
        for _ in (0..nout).step_by(8) {
            let at = idx.len();
            idx.push(0);
            let mut count = 0;
            let mut cols: Vec<i32> = (0..nin.saturating_sub(3) as i32).step_by(4).filter(|_| rng.below(3) != 0).collect();
            // The table need not be in increasing order.
            if rng.below(4) == 0 {
                cols.reverse();
            }
            for c in cols {
                idx.push(c);
                count += 1;
            }
            idx[at] = count;
            blocks += count as usize;
        }
    }
    let nw = if kind >= 2 { 32 * blocks } else { nin * nout };
    let fw = (0..nw).map(|_| rng.f(0.3)).collect();
    let w8 = (0..nw).map(|_| rng.w8()).collect();
    Owned { bias, fw, w8, idx, scale, diag, kind, nin, nout, with_bias, with_diag }
}

impl Owned {
    fn layer(&self) -> LinearLayer<'_> {
        let mut l = LinearLayer { nb_inputs: self.nin, nb_outputs: self.nout, ..Default::default() };
        if self.with_bias {
            l.bias = Some(&self.bias);
        }
        if self.with_diag {
            l.diag = Some(&self.diag);
        }
        match self.kind {
            0 => l.float_weights = Some(&self.fw),
            1 => {
                l.weights = Some(&self.w8);
                l.scale = Some(&self.scale);
            }
            2 => {
                l.weights = Some(&self.w8);
                l.scale = Some(&self.scale);
                l.weights_idx = Some(&self.idx);
            }
            _ => {
                l.float_weights = Some(&self.fw);
                l.weights_idx = Some(&self.idx);
            }
        }
        l
    }

    /// The `#[repr(C)]` mirror of `layer()`, pointing at the same arrays.
    fn c(&self) -> CLinear {
        fn p<T>(s: Option<&[T]>) -> *const T {
            s.map_or(ptr::null(), |s| s.as_ptr())
        }
        let l = self.layer();
        CLinear {
            bias: p(l.bias),
            subias: ptr::null(),
            weights: p(l.weights),
            float_weights: p(l.float_weights),
            weights_idx: p(l.weights_idx),
            diag: p(l.diag),
            scale: p(l.scale),
            nb_inputs: l.nb_inputs as c_int,
            nb_outputs: l.nb_outputs as c_int,
        }
    }

    /// Whether Apple clang's `-O2` build sums this layer in another order
    /// (`nnet.md` section 3.3): one float output, more than 3 inputs.
    fn vectorized_by_o2(&self) -> bool {
        self.kind == 0 && self.nout == 1 && self.nin > 3
    }

    /// `Σ_j |w_j·x_j|` of a one-output float layer.
    fn magnitude(&self, x: &[f32]) -> f64 {
        self.fw.iter().zip(x).map(|(w, x)| (*w as f64 * *x as f64).abs()).sum()
    }
}

/// Every model layer shape of libopus 1.6.1 (`nnet.md` Appendix A) and
/// odd shapes: (inputs, outputs, kind).
const SHAPES: [(usize, usize, u8); 66] = [
    (88, 64, 1), (64, 64, 1), (288, 64, 1), (64, 192, 1), (57, 128, 0), (192, 20, 0),
    (128, 576, 1), (192, 576, 1), (224, 12, 0), (32, 64, 0), (80, 1, 0), (328, 192, 1),
    (688, 128, 1), (128, 40, 1), (192, 4, 0), (40, 64, 0), (544, 32, 1), (128, 56, 1),
    (544, 128, 2), (96, 64, 2), (480, 64, 2), (64, 96, 2), (26, 96, 0), (50, 128, 0),
    (576, 80, 2), (128, 320, 2), (301, 64, 0), (173, 96, 0), (768, 128, 1), (128, 512, 1),
    (128, 16, 1), (128, 1, 0), (160, 2, 0), (42, 80, 0), (160, 80, 0), (320, 160, 1),
    (342, 128, 0), (384, 128, 1), (256, 120, 1), (42, 120, 0), (240, 120, 0), (128, 3, 0),
    (128, 288, 1), (3, 1, 0), (4, 1, 0), (5, 1, 0), (7, 1, 0), (16, 1, 0), (17, 1, 0),
    (19, 1, 0), (20, 1, 0), (2, 8, 0), (32, 1, 0), (8, 48, 1), (1, 3, 0), (12, 12, 0),
    (64, 96, 3), (160, 24, 3), (2040, 8, 1), (8, 8, 1), (12, 40, 1), (20, 8, 1), (36, 21, 0),
    (9, 7, 0), (160, 640, 1), (256, 80, 1),
];

const ACTS: [Activation; 7] = [
    Activation::Linear,
    Activation::Sigmoid,
    Activation::Tanh,
    Activation::Relu,
    Activation::Softmax,
    Activation::Swish,
    Activation::Exp,
];

#[test]
fn linear_layers_match_libopus() {
    let arch_c = c_arch();
    let o2 = o2_mode();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (mut checks, mut o2_cases, mut o2_worst) = (0u64, 0u64, 0.0f64);
    for &(nin, nout, kind) in SHAPES.iter() {
        for trial in 0..30 {
            let with_diag = trial % 3 == 2 && nout == 3 * nin;
            let o = make(&mut rng, nin, nout, kind, trial % 4 != 3, with_diag);
            let cl = o.c();
            let x = input(&mut rng, nin, trial);
            let mut want = vec![0.0f32; nout];
            unsafe { compute_linear_c(&cl, want.as_mut_ptr(), x.as_ptr()) };
            let mut want_dp = vec![0.0f32; nout];
            if arch_c >= ARCH_DOTPROD {
                unsafe { compute_linear_dotprod(&cl, want_dp.as_mut_ptr(), x.as_ptr()) };
                if !(o2 && o.vectorized_by_o2()) {
                    assert!(same(&want_dp, &want), "libopus c and dotprod differ: {nin}x{nout}");
                }
            }
            let mut dense_c = vec![0.0f32; nout];
            unsafe {
                c_generic_dense(&cl, dense_c.as_mut_ptr(), x.as_ptr(), Activation::Sigmoid as c_int, arch_c)
            };
            for &a in arch::available() {
                let mut got = vec![0.0f32; nout];
                compute_linear(&o.layer(), &mut got, &x, a);
                if o2 && o.vectorized_by_o2() {
                    // The tolerance `2·M·2^-24·Σ|w·x|`.
                    let mag = o.magnitude(&x);
                    let bound = 2.0 * nin as f64 * mag / 16_777_216.0;
                    let diff = (got[0] as f64 - want[0] as f64).abs();
                    o2_cases += 1;
                    if mag > 0.0 && mag.is_finite() && diff.is_finite() {
                        o2_worst = o2_worst.max(diff / mag);
                    }
                    assert!(diff <= bound || (got[0].is_nan() && want[0].is_nan()) || got[0] == want[0], "{nin}x1: {diff:e} > {bound:e}");
                    continue;
                }
                assert!(same(&got, &want), "{a:?} {nin}x{nout} kind {kind} trial {trial}: {}", first_diff(&got, &want));
                checks += 1;
                let mut dense = vec![0.0f32; nout];
                compute_generic_dense(&o.layer(), &mut dense, &x, Activation::Sigmoid, a);
                assert!(same(&dense, &dense_c), "dense {a:?} {nin}x{nout} kind {kind} trial {trial}: {}", first_diff(&dense, &dense_c));
            }
        }
    }
    println!("linear: {checks} layer calls compared, all exact");
    if o2 {
        println!("-O2 reference: {o2_cases} one-output float calls within the bound; largest |diff| / Σ|w·x| = {o2_worst:e}");
    }
}

#[test]
fn activations_match_libopus() {
    let arch_c = c_arch();
    let mut rng = Rng(3);
    let mut checks = 0u64;
    for act in ACTS {
        for n in (1..=41).chain([576]) {
            for trial in 0..60 {
                let x = input(&mut rng, n, trial % 31);
                let mut want = vec![0.0f32; n];
                unsafe { compute_activation_c(want.as_mut_ptr(), x.as_ptr(), n as c_int, act as c_int) };
                if arch_c >= ARCH_DOTPROD {
                    let mut dp = vec![0.0f32; n];
                    unsafe { compute_activation_dotprod(dp.as_mut_ptr(), x.as_ptr(), n as c_int, act as c_int) };
                    assert!(same(&dp, &want));
                }
                for &a in arch::available() {
                    let mut got = vec![0.0f32; n];
                    compute_activation(&mut got, &x, act, a);
                    assert!(same(&got, &want), "{act:?} {a:?} n {n} trial {trial}: {}", first_diff(&got, &want));
                    let mut inplace = x.clone();
                    compute_activation_in_place(&mut inplace, act, a);
                    assert!(same(&inplace, &want), "in place {act:?} {a:?} n {n}");
                    checks += 1;
                }
            }
        }
    }
    // Every float bit pattern class through TANH and SIGMOID, in groups of
    // 4 and as tails: sign, exponent and the top fraction bits.
    let mut x = Vec::new();
    for sign in [0u32, 1] {
        for e in 0..256u32 {
            for f in [0u32, 1, 0x40_0000, 0x7f_ffff, rng.next() & 0x7f_ffff] {
                x.push(f32::from_bits(sign << 31 | e << 23 | f));
            }
        }
    }
    x.push(1.0);
    for act in [Activation::Tanh, Activation::Sigmoid, Activation::Exp, Activation::Swish] {
        for len in [x.len(), x.len() - 1, x.len() - 2, x.len() - 3] {
            let mut want = vec![0.0f32; len];
            unsafe { compute_activation_c(want.as_mut_ptr(), x.as_ptr(), len as c_int, act as c_int) };
            for &a in arch::available() {
                let mut got = vec![0.0f32; len];
                compute_activation(&mut got, &x[..len], act, a);
                assert!(same(&got, &want), "{act:?} {a:?} all classes: {}", first_diff(&got, &want));
                checks += 1;
            }
        }
    }
    println!("activations: {checks} calls compared, all exact");
}

#[test]
fn gru_glu_and_conv1d_match_libopus() {
    let arch_c = c_arch();
    let o2 = o2_mode();
    let mut rng = Rng(11);
    let mut checks = 0u64;
    // (inputs, state size, kind of the input layer)
    let grus = [(128usize, 192usize, 1u8), (192, 192, 1), (64, 64, 1), (272, 160, 1), (64, 32, 2), (160, 128, 0), (57, 4, 0), (8, 16, 1), (128, 128, 1), (96, 64, 2)];
    for &(nin, n, kind) in &grus {
        let li = make(&mut rng, nin, 3 * n, kind, true, false);
        // A float recurrent layer gets a diagonal, which no 1.6.1 model has.
        let lr = make(&mut rng, n, 3 * n, if kind == 2 { 1 } else { kind }, true, kind == 0);
        let (ci, cr) = (li.c(), lr.c());
        let mut want = vec![0.0f32; n];
        let mut states: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; n]).collect();
        for step in 0..50 {
            let x = input(&mut rng, nin, 3 + step % 28);
            unsafe { c_generic_gru(&ci, &cr, want.as_mut_ptr(), x.as_ptr(), arch_c) };
            for (s, &a) in states.iter_mut().zip(arch::available()) {
                compute_generic_gru(&li.layer(), &lr.layer(), s, &x, a);
                assert!(same(s, &want), "GRU {a:?} {nin}x{n} step {step}: {}", first_diff(s, &want));
                checks += 1;
            }
        }
    }
    for &(n, kind) in &[(192usize, 1u8), (160, 1), (128, 1), (64, 2), (24, 0), (20, 0)] {
        let l = make(&mut rng, n, n, kind, true, false);
        let cl = l.c();
        for trial in 0..20 {
            let x = input(&mut rng, n, trial);
            let mut want = vec![0.0f32; n];
            unsafe { c_glu(&cl, want.as_mut_ptr(), x.as_ptr(), arch_c) };
            let mut want_in_place = x.clone();
            unsafe { c_glu(&cl, want_in_place.as_mut_ptr(), want_in_place.as_ptr(), arch_c) };
            assert!(same(&want, &want_in_place));
            for &a in arch::available() {
                let mut got = vec![0.0f32; n];
                compute_glu(&l.layer(), &mut got, &x, a);
                assert!(same(&got, &want), "GLU {a:?} {n} trial {trial}: {}", first_diff(&got, &want));
                let mut inplace = x.clone();
                compute_glu_in_place(&l.layer(), &mut inplace, a);
                assert!(same(&inplace, &want), "GLU in place {a:?} {n}");
                checks += 2;
            }
        }
    }
    // conv1d: (input size, kernel size, outputs, kind, dilation), among
    // them the shapes of FARGAN, LACE, NoLACE, BBWENet and the RDOVAE
    // encoder (dilation 2).
    let convs = [
        (64usize, 3usize, 128usize, 1u8, 1usize),
        (164, 2, 192, 1, 1),
        (173, 1, 96, 0, 1),
        (384, 2, 128, 1, 1),
        (171, 2, 128, 0, 1),
        (64, 2, 64, 1, 2),
        (32, 4, 64, 2, 2),
        (96, 2, 64, 2, 2),
        (16, 3, 8, 0, 3),
        (40, 2, 16, 3, 1),
        (10, 2, 1, 0, 1),
    ];
    for &(isz, ksize, nout, kind, dil) in &convs {
        let l = make(&mut rng, isz * ksize, nout, kind, true, false);
        if o2 && l.vectorized_by_o2() {
            continue;
        }
        let cl = l.c();
        let memlen = isz * dil * (ksize - 1);
        let mut want_mem = vec![0.0f32; memlen.max(1)];
        let mut mems: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; memlen]).collect();
        for step in 0..30 {
            let x = input(&mut rng, isz, 3 + step % 28);
            let mut want = vec![0.0f32; nout];
            let act = ACTS[step % ACTS.len()];
            unsafe {
                let mem = if memlen == 0 { ptr::null_mut() } else { want_mem.as_mut_ptr() };
                if dil == 1 {
                    c_generic_conv1d(&cl, want.as_mut_ptr(), mem, x.as_ptr(), isz as c_int, act as c_int, arch_c);
                } else {
                    c_generic_conv1d_dilation(&cl, want.as_mut_ptr(), mem, x.as_ptr(), isz as c_int, dil as c_int, act as c_int, arch_c);
                }
            }
            for (m, &a) in mems.iter_mut().zip(arch::available()) {
                let mut got = vec![0.0f32; nout];
                if dil == 1 {
                    compute_generic_conv1d(&l.layer(), &mut got, m, &x, act, a);
                } else {
                    compute_generic_conv1d_dilation(&l.layer(), &mut got, m, &x, dil, act, a);
                }
                assert!(same(&got, &want), "conv1d {a:?} {isz}x{ksize} dil {dil} step {step}: {}", first_diff(&got, &want));
                assert!(same(m, &want_mem[..memlen]), "conv1d memory {a:?} {isz}x{ksize} dil {dil} step {step}");
                checks += 1;
            }
        }
    }
    println!("GRU, GLU, conv1d: {checks} calls compared, all exact");
}

#[test]
fn conv2d_matches_libopus() {
    let arch_c = c_arch();
    let mut rng = Rng(13);
    let mut checks = 0u64;
    // (in channels, out channels, ktime, kheight, height, hstride): the two
    // PitchDNN layers, then other kernel sizes (`conv2d_float`).
    let shapes = [(1usize, 4usize, 3usize, 3usize, 224usize, 226usize), (4, 1, 3, 3, 224, 224), (2, 3, 2, 5, 17, 20), (3, 2, 1, 1, 9, 9), (2, 2, 3, 3, 5, 8)];
    for &(cin, cout, kt, kh, height, hstride) in &shapes {
        let w: Vec<f32> = (0..cout * cin * kt * kh).map(|_| rng.f(0.5)).collect();
        let bias: Vec<f32> = (0..cout).map(|_| rng.f(0.5)).collect();
        let c = Conv2dLayer { bias: Some(&bias), float_weights: Some(&w), in_channels: cin, out_channels: cout, ktime: kt, kheight: kh };
        let cc = CConv2d {
            bias: bias.as_ptr(),
            float_weights: w.as_ptr(),
            in_channels: cin as c_int,
            out_channels: cout as c_int,
            ktime: kt as c_int,
            kheight: kh as c_int,
        };
        let ts = cin * (height + kh - 1);
        let memlen = (kt - 1) * ts;
        let mut want_mem = vec![0.0f32; memlen.max(1)];
        let mut want_mem_dp = want_mem.clone();
        let mut mems: Vec<Vec<f32>> = arch::available().iter().map(|_| want_mem.clone()).collect();
        for step in 0..20 {
            let x = input(&mut rng, ts, 3 + step % 28);
            let act = ACTS[step % ACTS.len()];
            let olen = (cout - 1) * hstride + height;
            let mut want = vec![0.0f32; olen];
            unsafe {
                compute_conv2d_c(&cc, want.as_mut_ptr(), want_mem.as_mut_ptr(), x.as_ptr(), height as c_int, hstride as c_int, act as c_int)
            };
            if arch_c >= ARCH_DOTPROD {
                let mut dp = vec![0.0f32; olen];
                unsafe {
                    compute_conv2d_dotprod(&cc, dp.as_mut_ptr(), want_mem_dp.as_mut_ptr(), x.as_ptr(), height as c_int, hstride as c_int, act as c_int)
                };
                assert!(same(&dp, &want));
            }
            for (m, &a) in mems.iter_mut().zip(arch::available()) {
                let mut got = vec![0.0f32; olen];
                compute_conv2d(&c, &mut got, m, &x, height, hstride, act, a);
                // The gaps between the output channels are not written.
                for i in 0..cout {
                    let r = i * hstride..i * hstride + height;
                    assert!(same(&got[r.clone()], &want[r.clone()]), "conv2d {a:?} {cin}->{cout} {kt}x{kh} step {step}: {}", first_diff(&got[r.clone()], &want[r]));
                }
                assert!(same(m, &want_mem), "conv2d memory {a:?} step {step}");
                checks += 1;
            }
        }
    }
    println!("conv2d: {checks} calls compared, all exact");
}

/// A `CLinear` for arrays that a test owns.
#[allow(clippy::too_many_arguments)]
fn c_layer(
    bias: Option<&[f32]>,
    weights: Option<&[i8]>,
    float_weights: Option<&[f32]>,
    idx: Option<&[i32]>,
    diag: Option<&[f32]>,
    scale: Option<&[f32]>,
    nin: usize,
    nout: usize,
) -> CLinear {
    fn p<T>(s: Option<&[T]>) -> *const T {
        s.map_or(ptr::null(), |s| s.as_ptr())
    }
    CLinear {
        bias: p(bias),
        subias: ptr::null(),
        weights: p(weights),
        float_weights: p(float_weights),
        weights_idx: p(idx),
        diag: p(diag),
        scale: p(scale),
        nb_inputs: nin as c_int,
        nb_outputs: nout as c_int,
    }
}

/// Inputs padded with 8 values: libopus quantizes 8 inputs at a time and
/// reads up to 4 values past `nb_inputs` when it is not a multiple of 8.
fn padded(x: &[f32]) -> Vec<f32> {
    let mut v = x.to_vec();
    v.extend([0.0; 8]);
    v
}

/// Edge cases against libopus: sparse tables with empty row blocks,
/// repeated and falling positions; int8 weights of ±127 with inputs that
/// quantize to -128 and 127, up to 2048 inputs; NaN, infinities and
/// subnormal numbers as layer inputs; GRU and GLU layers of odd sizes.
#[test]
fn edge_layers_match_libopus() {
    let arch_c = c_arch();
    let o2 = o2_mode();
    let mut rng = Rng(0xfeed_5eed);
    let mut checks = 0u64;
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-45,
        -1e-40,
        -0.0,
        2.147_483_6e9,
        -2_147_483_648.0,
        -1.003_937,
        1.007_874,
        1.5,
    ];
    let compare = |what: &str, l: &LinearLayer, cl: &CLinear, x: &[f32], checks: &mut u64| {
        let xp = padded(x);
        let mut want = vec![0.0f32; l.nb_outputs];
        unsafe { compute_linear_c(cl, want.as_mut_ptr(), xp.as_ptr()) };
        for &a in arch::available() {
            let mut got = vec![0.0f32; l.nb_outputs];
            compute_linear(l, &mut got, x, a);
            assert!(same(&got, &want), "{what} {a:?} {}x{}: {}", l.nb_inputs, l.nb_outputs, first_diff(&got, &want));
            *checks += 1;
        }
    };
    // Sparse tables of every kind, float and int8.
    for &(nin, nout) in &[(4usize, 48usize), (12, 48), (64, 96), (2048, 48)] {
        let groups = (nin / 4) as u32;
        let mut idx = Vec::new();
        let mut blocks = 0;
        for b in 0..nout / 8 {
            let positions: Vec<i32> = match b % 6 {
                0 => vec![],
                1 => vec![4 * (rng.next() % groups) as i32],
                2 => vec![0, 0, 0],
                3 => (0..groups as i32).rev().map(|g| 4 * g).collect(),
                4 => (0..groups as i32).map(|g| 4 * g).chain([4 * (groups as i32 - 1), 0]).collect(),
                _ => (0..5).map(|_| 4 * (rng.next() % groups) as i32).collect(),
            };
            idx.push(positions.len() as i32);
            blocks += positions.len();
            idx.extend(positions);
        }
        let fw: Vec<f32> = (0..32 * blocks).map(|_| rng.f(0.5)).collect();
        let w8: Vec<i8> = (0..32 * blocks).map(|_| rng.w8()).collect();
        let scale: Vec<f32> = (0..nout).map(|_| rng.f(0.01)).collect();
        let bias: Vec<f32> = (0..nout).map(|_| rng.f(0.5)).collect();
        for trial in 0..6 {
            let x: Vec<f32> = (0..nin).map(|j| if trial >= 3 && j % 5 == 0 { specials[(j + trial) % specials.len()] } else { rng.f(1.2) }).collect();
            let l = LinearLayer { float_weights: Some(&fw), weights_idx: Some(&idx), bias: Some(&bias), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
            let cl = c_layer(Some(&bias), None, Some(&fw), Some(&idx), None, None, nin, nout);
            compare("float sparse", &l, &cl, &x, &mut checks);
            let l = LinearLayer { weights: Some(&w8), scale: Some(&scale), weights_idx: Some(&idx), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
            let cl = c_layer(None, Some(&w8), None, Some(&idx), None, Some(&scale), nin, nout);
            compare("int8 sparse", &l, &cl, &x, &mut checks);
        }
    }
    // Dense int8 layers at the extremes.
    for &nin in &[4usize, 12, 2044, 2048] {
        let nout = 16;
        for pattern in 0..4 {
            let w: Vec<i8> = (0..nin * nout)
                .map(|i| match pattern {
                    0 => 127,
                    1 => -127,
                    2 => if i % 2 == 0 { 127 } else { -127 },
                    _ => if (i / 4) % 3 == 0 { -127 } else { 127 },
                })
                .collect();
            let scale = vec![1.0f32; nout];
            for x0 in [-1.003_937f32, 1.0, -1.0, 1.007_874, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut x = vec![x0; nin];
                x[0] = 1.0;
                let l = LinearLayer { weights: Some(&w), scale: Some(&scale), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
                let cl = c_layer(None, Some(&w), None, None, None, Some(&scale), nin, nout);
                compare("int8 extreme", &l, &cl, &x, &mut checks);
            }
        }
    }
    // Float and int8 dense layers with special inputs.
    for &(nin, nout, kind) in &[(16usize, 8usize, 1u8), (24, 16, 1), (17, 3, 0), (40, 16, 0), (3, 1, 0), (64, 1, 0)] {
        let o = make(&mut rng, nin, nout, kind, true, false);
        if o2 && o.vectorized_by_o2() {
            continue;
        }
        for shift in 0..specials.len() {
            let x: Vec<f32> = (0..nin).map(|j| if j % 3 == 0 { specials[(j + shift) % specials.len()] } else { rng.f(1.0) }).collect();
            compare("special inputs", &o.layer(), &o.c(), &x, &mut checks);
        }
    }
    // GRU and GLU layers of odd sizes (float weights, so any size runs),
    // with a diagonal on the recurrent weights.
    for &n in &[1usize, 3, 5, 7, 13, 191] {
        let nin = 2 * n + 1;
        let li = make(&mut rng, nin, 3 * n, 0, true, false);
        let lr = make(&mut rng, n, 3 * n, 0, true, true);
        let (ci, cr) = (li.c(), lr.c());
        let mut want = vec![0.0f32; n];
        let mut states: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; n]).collect();
        for step in 0..20 {
            let x: Vec<f32> = (0..nin).map(|_| rng.f(if step % 4 == 0 { 6.0 } else { 1.0 })).collect();
            unsafe { c_generic_gru(&ci, &cr, want.as_mut_ptr(), x.as_ptr(), arch_c) };
            for (s, &a) in states.iter_mut().zip(arch::available()) {
                compute_generic_gru(&li.layer(), &lr.layer(), s, &x, a);
                assert!(same(s, &want), "GRU {a:?} n {n} step {step}: {}", first_diff(s, &want));
                checks += 1;
            }
        }
        let lg = make(&mut rng, n, n, 0, true, false);
        if o2 && lg.vectorized_by_o2() {
            continue;
        }
        let x: Vec<f32> = (0..n).map(|_| rng.f(3.0)).collect();
        let mut want = vec![0.0f32; n];
        unsafe { c_glu(&lg.c(), want.as_mut_ptr(), x.as_ptr(), arch_c) };
        for &a in arch::available() {
            let mut got = vec![0.0f32; n];
            compute_glu(&lg.layer(), &mut got, &x, a);
            assert!(same(&got, &want), "GLU {a:?} n {n}: {}", first_diff(&got, &want));
            checks += 1;
        }
    }
    println!("edge layers: {checks} calls compared, all exact");
}
