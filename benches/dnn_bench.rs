//! Speed of the neural network kernels (`nnet.md` section 10.7): the layer
//! shapes of `nnet.md` section 10.1 for every `DnnArch` this processor can
//! run, the TANH and SIGMOID activations on 480 values, and one GRU step of
//! NoLACE and of FARGAN.
//!
//! The throughput is in multiply-accumulates: Criterion's "Gelem/s" is
//! GMAC/s (for the activations, billions of values per second).
//!
//! With the feature `test-libopus-dnn` and a libopus built with its neural
//! network code (`scripts/android/build-libopus.sh --dnn`), the same layers
//! also run through libopus's `compute_linear_c` and, on processors with
//! DotProd, `compute_linear_dotprod`:
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=target/dnn-ref cargo bench --features dnn,test-libopus-dnn --bench dnn_bench
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use opus1::internals::dnn::nnet::{
    LinearLayer, compute_activation, compute_activation_in_place, compute_generic_gru, compute_linear,
};
use opus1::internals::dnn::{Activation, DnnArch, arch};

fn configure() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_millis(400))
        .warm_up_time(Duration::from_millis(100))
}

/// The layer shapes of `nnet.md` section 10.1: (inputs, outputs, int8).
const SHAPES: [(usize, usize, bool); 14] = [
    (768, 128, true),
    (160, 480, true),
    (128, 384, true),
    (688, 128, true),
    (272, 480, true),
    (320, 160, true),
    (128, 16, true),
    (301, 64, false),
    (173, 96, false),
    (342, 128, false),
    (240, 120, false),
    (160, 80, false),
    (128, 1, false),
    (192, 20, false),
];

/// The arrays of one layer with fixed pseudo-random values.
struct Owned {
    w8: Vec<i8>,
    fw: Vec<f32>,
    scale: Vec<f32>,
    bias: Vec<f32>,
    nin: usize,
    nout: usize,
    int8: bool,
}

impl Owned {
    fn new(nin: usize, nout: usize, int8: bool, seed: u32) -> Owned {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let n = nin * nout;
        Owned {
            w8: (0..n).map(|_| ((next() % 255) as i32 - 127) as i8).collect(),
            fw: (0..n).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect(),
            scale: (0..nout).map(|_| 1.0 / (128.0 * 127.0)).collect(),
            bias: (0..nout).map(|_| (next() % 101) as f32 / 100.0 - 0.5).collect(),
            nin,
            nout,
            int8,
        }
    }

    fn layer(&self) -> LinearLayer<'_> {
        let mut l = LinearLayer {
            bias: Some(&self.bias),
            nb_inputs: self.nin,
            nb_outputs: self.nout,
            ..Default::default()
        };
        if self.int8 {
            l.weights = Some(&self.w8);
            l.scale = Some(&self.scale);
        } else {
            l.float_weights = Some(&self.fw);
        }
        l
    }
}

fn input(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect()
}

fn arch_name(a: DnnArch) -> String {
    format!("opus1-{a:?}")
}

#[cfg(feature = "test-libopus-dnn")]
mod libopus {
    //! The kernels of a DNN-enabled libopus.
    use super::Owned;
    use std::os::raw::c_int;
    use std::ptr;

    /// libopus `LinearLayer` (`dnn/nnet.h:66-76`).
    #[repr(C)]
    pub struct CLinear {
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

    unsafe extern "C" {
        fn opus_select_arch() -> c_int;
        pub fn compute_linear_c(l: *const CLinear, out: *mut f32, input: *const f32);
        pub fn compute_linear_dotprod(l: *const CLinear, out: *mut f32, input: *const f32);
        pub fn compute_activation_c(out: *mut f32, input: *const f32, n: c_int, act: c_int);
        #[link_name = "compute_generic_gru"]
        pub fn c_gru(
            input_w: *const CLinear,
            recurrent_w: *const CLinear,
            state: *mut f32,
            input: *const f32,
            arch: c_int,
        );
    }

    /// libopus's arch value; DotProd is 4.
    pub fn arch() -> c_int {
        let _ = unsafe { audiopus_sys::opus_get_version_string() };
        unsafe { opus_select_arch() }
    }

    pub fn layer(o: &Owned) -> CLinear {
        CLinear {
            bias: o.bias.as_ptr(),
            subias: ptr::null(),
            weights: if o.int8 { o.w8.as_ptr() } else { ptr::null() },
            float_weights: if o.int8 { ptr::null() } else { o.fw.as_ptr() },
            weights_idx: ptr::null(),
            diag: ptr::null(),
            scale: if o.int8 { o.scale.as_ptr() } else { ptr::null() },
            nb_inputs: o.nin as c_int,
            nb_outputs: o.nout as c_int,
        }
    }
}

fn bench_linear(c: &mut Criterion) {
    let mut g = c.benchmark_group("dnn_linear");
    for (k, &(nin, nout, int8)) in SHAPES.iter().enumerate() {
        let o = Owned::new(nin, nout, int8, 0x1234_5678 + k as u32);
        let l = o.layer();
        let x = input(nin);
        let mut out = vec![0.0f32; nout];
        let name = format!("{nin}x{nout} {}", if int8 { "int8" } else { "float" });
        g.throughput(Throughput::Elements((nin * nout) as u64));
        for &a in arch::available() {
            g.bench_function(BenchmarkId::new(&name, arch_name(a)), |b| {
                b.iter(|| compute_linear(black_box(&l), black_box(&mut out), black_box(&x), a))
            });
        }
        #[cfg(feature = "test-libopus-dnn")]
        {
            let cl = libopus::layer(&o);
            g.bench_function(BenchmarkId::new(&name, "libopus-c"), |b| {
                b.iter(|| unsafe { libopus::compute_linear_c(black_box(&cl), out.as_mut_ptr(), x.as_ptr()) })
            });
            if libopus::arch() >= 4 {
                g.bench_function(BenchmarkId::new(&name, "libopus-dotprod"), |b| {
                    b.iter(|| unsafe {
                        libopus::compute_linear_dotprod(black_box(&cl), out.as_mut_ptr(), x.as_ptr())
                    })
                });
            }
        }
    }
    g.finish();
}

fn bench_activations(c: &mut Criterion) {
    let mut g = c.benchmark_group("dnn_activation");
    let x = input(480);
    let mut buf = x.clone();
    g.throughput(Throughput::Elements(480));
    for (act, name) in [(Activation::Tanh, "tanh 480"), (Activation::Sigmoid, "sigmoid 480")] {
        // `compute_activation(out, input)`, as libopus's
        // `compute_activation_c`; it copies the input, then works in place.
        for &a in arch::available() {
            g.bench_function(BenchmarkId::new(name, arch_name(a)), |b| {
                b.iter(|| compute_activation(black_box(&mut buf), black_box(&x), act, a))
            });
        }
        // The in-place call of the layer functions.
        g.bench_function(BenchmarkId::new(name, format!("{} in place", arch_name(arch::detect()))), |b| {
            b.iter(|| compute_activation_in_place(black_box(&mut buf), act, arch::detect()))
        });
        #[cfg(feature = "test-libopus-dnn")]
        g.bench_function(BenchmarkId::new(name, "libopus-c"), |b| {
            b.iter(|| unsafe {
                libopus::compute_activation_c(buf.as_mut_ptr(), black_box(x.as_ptr()), 480, act as i32)
            })
        });
    }
    g.finish();
}

fn bench_gru(c: &mut Criterion) {
    let mut g = c.benchmark_group("dnn_gru");
    // NoLACE: 160 inputs, 160 states; FARGAN's first GRU: 272 inputs, 160
    // states. Both int8.
    for (nin, n, name) in [(160usize, 160usize, "nolace 160x480"), (272, 160, "fargan gru1 272x480")] {
        let oi = Owned::new(nin, 3 * n, true, 7);
        let or = Owned::new(n, 3 * n, true, 9);
        let (li, lr) = (oi.layer(), or.layer());
        let x = input(nin);
        let mut state = vec![0.1f32; n];
        g.throughput(Throughput::Elements((3 * n * (nin + n)) as u64));
        for &a in arch::available() {
            g.bench_function(BenchmarkId::new(name, arch_name(a)), |b| {
                b.iter(|| compute_generic_gru(black_box(&li), black_box(&lr), &mut state, black_box(&x), a))
            });
        }
        #[cfg(feature = "test-libopus-dnn")]
        {
            let (ci, cr) = (libopus::layer(&oi), libopus::layer(&or));
            let arch_c = libopus::arch();
            g.bench_function(BenchmarkId::new(name, "libopus"), |b| {
                b.iter(|| unsafe { libopus::c_gru(&ci, &cr, state.as_mut_ptr(), black_box(x.as_ptr()), arch_c) })
            });
        }
    }
    g.finish();
}

/// Layers that do not stay in the caches: 8 different layers, run one
/// after the other, as a model runs its layers. The weights of the 8 int8
/// layers of 160 × 480 take 614,400 bytes; those of the 8 float layers of
/// 160 × 80 take 409,600 bytes.
fn bench_cold(c: &mut Criterion) {
    let mut g = c.benchmark_group("dnn_cold");
    for (nin, nout, int8) in [(160usize, 480usize, true), (160, 80, false)] {
        let layers: Vec<Owned> = (0..8).map(|k| Owned::new(nin, nout, int8, 100 + k)).collect();
        let x = input(nin);
        let mut out = vec![0.0f32; nout];
        let name = format!("8 x {nin}x{nout} {}", if int8 { "int8" } else { "float" });
        g.throughput(Throughput::Elements((8 * nin * nout) as u64));
        for &a in arch::available() {
            g.bench_function(BenchmarkId::new(&name, arch_name(a)), |b| {
                b.iter(|| {
                    for o in &layers {
                        compute_linear(black_box(&o.layer()), black_box(&mut out), black_box(&x), a)
                    }
                })
            });
        }
        #[cfg(feature = "test-libopus-dnn")]
        {
            let cls: Vec<_> = layers.iter().map(libopus::layer).collect();
            g.bench_function(BenchmarkId::new(&name, "libopus-c"), |b| {
                b.iter(|| {
                    for cl in &cls {
                        unsafe { libopus::compute_linear_c(black_box(cl), out.as_mut_ptr(), x.as_ptr()) }
                    }
                })
            });
            if libopus::arch() >= 4 {
                g.bench_function(BenchmarkId::new(&name, "libopus-dotprod"), |b| {
                    b.iter(|| {
                        for cl in &cls {
                            unsafe { libopus::compute_linear_dotprod(black_box(cl), out.as_mut_ptr(), x.as_ptr()) }
                        }
                    })
                });
            }
        }
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_linear, bench_activations, bench_gru, bench_cold
}
criterion_main!(benches);
