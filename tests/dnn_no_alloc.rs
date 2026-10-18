//! The layer functions of the neural network code allocate nothing: their
//! scratch buffers are on the stack.
//!
//! This test binary has its own global allocator, which counts the calls a
//! thread makes while that thread has counting switched on (as
//! `multistream_dec_no_alloc.rs` does). The layers and inputs are built
//! first; then every layer function runs on every available `DnnArch`.
#![cfg(all(feature = "dnn", feature = "std"))]

use opus1::internals::dnn::nnet::{
    Conv2dLayer, LinearLayer, compute_activation, compute_activation_in_place, compute_conv2d,
    compute_generic_conv1d, compute_generic_conv1d_dilation, compute_generic_dense,
    compute_generic_gru, compute_glu, compute_glu_in_place, compute_linear,
};
use opus1::internals::dnn::{Activation, arch};
use std::alloc::{GlobalAlloc, Layout as MemoryLayout, System};
use std::cell::Cell;

thread_local! {
    /// Whether this thread counts its allocator calls.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Calls that take memory: `alloc`, `alloc_zeroed`, `realloc`.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count() {
    // `try_with`: the allocator also runs while a thread shuts down, when
    // its thread-local values are gone.
    if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: MemoryLayout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: MemoryLayout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: MemoryLayout, new_size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: MemoryLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` with counting on and returns the number of allocations.
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(|n| n.get())
}

#[test]
fn layer_functions_do_not_allocate() {
    let w8: Vec<i8> = (0..768 * 576).map(|i: i32| (i * 31 % 255 - 127) as i8).collect();
    let fw: Vec<f32> = (0..342 * 128).map(|i| ((i * 17 % 101) as f32 - 50.0) / 400.0).collect();
    let scale = vec![1e-3f32; 576];
    let bias = vec![0.01f32; 576];
    let int8 = |nin, nout| LinearLayer {
        weights: Some(&w8[..nin * nout]),
        scale: Some(&scale[..nout]),
        bias: Some(&bias[..nout]),
        nb_inputs: nin,
        nb_outputs: nout,
        ..Default::default()
    };
    let float = |nin, nout| LinearLayer {
        float_weights: Some(&fw[..nin * nout]),
        bias: Some(&bias[..nout]),
        nb_inputs: nin,
        nb_outputs: nout,
        ..Default::default()
    };
    let idx: Vec<i32> = (0..16).flat_map(|_| [2, 0, 8]).collect();
    let sparse = LinearLayer { weights_idx: Some(&idx), ..int8(16, 128) };
    let conv2d = Conv2dLayer {
        bias: Some(&bias[..4]),
        float_weights: Some(&fw[..36]),
        in_channels: 1,
        out_channels: 4,
        ktime: 3,
        kheight: 3,
    };
    let x: Vec<f32> = (0..2048).map(|i| ((i * 13 % 59) as f32 - 29.0) / 20.0).collect();
    let mut out = vec![0.0f32; 4096];
    let mut state = vec![0.0f32; 192];
    let mut mem = vec![0.0f32; 4096];
    // The first call of each variant may run the feature detection of
    // `std`; it does not allocate either, but runs before counting.
    for &a in arch::available() {
        compute_linear(&int8(768, 128), &mut out, &x, a);
    }
    for &a in arch::available() {
        let n = allocations(|| {
            compute_linear(&int8(768, 128), &mut out, &x, a);
            compute_linear(&float(342, 128), &mut out, &x, a);
            compute_linear(&float(128, 1), &mut out, &x, a);
            compute_linear(&sparse, &mut out, &x, a);
            compute_generic_dense(&int8(128, 384), &mut out, &x, Activation::Tanh, a);
            compute_generic_gru(&int8(128, 576), &int8(192, 576), &mut state, &x, a);
            compute_glu(&int8(192, 192), &mut out, &x, a);
            compute_glu_in_place(&int8(160, 160), &mut out, a);
            compute_generic_conv1d(&int8(328, 192), &mut out, &mut mem[..164], &x[..164], Activation::Tanh, a);
            compute_generic_conv1d_dilation(&int8(128, 64), &mut out, &mut mem[..128], &x[..64], 2, Activation::Tanh, a);
            compute_conv2d(&conv2d, &mut out, &mut mem, &x, 224, 226, Activation::Tanh, a);
            for act in [
                Activation::Linear,
                Activation::Sigmoid,
                Activation::Tanh,
                Activation::Relu,
                Activation::Softmax,
                Activation::Swish,
                Activation::Exp,
            ] {
                compute_activation(&mut out[..2048], &x[..2048], act, a);
                compute_activation_in_place(&mut out[..4096], act, a);
            }
        });
        assert_eq!(n, 0, "{a:?}: {n} allocations");
    }
}
