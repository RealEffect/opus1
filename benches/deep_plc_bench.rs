//! Speed of the deep PLC models (`src/dnn/plc/`) with the kernel variant
//! this processor selects: the feature analysis, PitchDNN, FARGAN, and the
//! calls of `DeepPlc` that a decoder makes (`update`, the first and a later
//! `conceal` of a loss, `update_from_celt`).
//!
//! ```text
//! cargo bench --features deep-plc --bench deep_plc_bench
//! ```
//!
//! `tests/dnn_plc_libopus.rs` has a test, `timing`, that measures the same
//! calls next to libopus's.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use opus1::internals::dnn::arch;
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::plc::fargan::{self, FARGAN_CONT_SAMPLES, FARGAN_FRAME_SIZE, FarganState};
use opus1::internals::dnn::plc::features::{LpcnetEncState, compute_single_frame_features};
use opus1::internals::dnn::plc::lpcnet_plc::CELT_DECODE_BUFFER_SIZE;
use opus1::internals::dnn::plc::pitchdnn::{PitchDnnState, compute_pitchdnn};
use opus1::internals::dnn::plc::{CONT_VECTORS, DeepPlc, LPCNET_FRAME_SIZE, NB_FEATURES, NB_TOTAL_FEATURES};

const FRAME: usize = LPCNET_FRAME_SIZE;

fn configure() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_millis(800))
        .warm_up_time(Duration::from_millis(200))
}

/// The 16 kHz speech fixture.
fn speech() -> Vec<i16> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    let data = bytes.windows(4).position(|w| w == b"data").expect("a data chunk");
    let len = u32::from_le_bytes(bytes[data + 4..data + 8].try_into().unwrap()) as usize;
    let (samples, _) = bytes[data + 8..data + 8 + len].as_chunks::<2>();
    samples.iter().map(|&b| i16::from_le_bytes(b)).collect()
}

fn frame_f32(s: &[i16]) -> [f32; FRAME] {
    let mut x = [0.0f32; FRAME];
    for (d, &v) in x.iter_mut().zip(s) {
        *d = v as f32;
    }
    x
}

fn frame_i16(s: &[i16]) -> [i16; FRAME] {
    let mut x = [0i16; FRAME];
    x.copy_from_slice(&s[..FRAME]);
    x
}

fn bench(c: &mut Criterion) {
    let a = arch::detect();
    let m = builtin::models().deep_plc.as_ref().expect("the built-in deep PLC models");
    let s = speech();
    let frames: Vec<[f32; FRAME]> = (300..364).map(|k| frame_f32(&s[k * FRAME..])).collect();
    let mut group = c.benchmark_group(format!("deep_plc_{a:?}"));

    // The feature analysis, frame after frame.
    let mut enc = LpcnetEncState::new();
    let mut feats = Vec::new();
    for x in &frames {
        let mut f = [0.0f32; NB_TOTAL_FEATURES];
        compute_single_frame_features(&mut enc, x, &mut f, &m.pitchdnn, a);
        feats.push(f);
    }
    let mut k = 0;
    let mut out = [0.0f32; NB_TOTAL_FEATURES];
    group.bench_function("features", |b| {
        b.iter(|| {
            compute_single_frame_features(&mut enc, &frames[k % frames.len()], &mut out, &m.pitchdnn, a);
            k += 1;
        })
    });

    // PitchDNN alone.
    let mut p = PitchDnnState::new();
    let (iff, xcf) = (*enc.if_features(), *enc.xcorr_features());
    group.bench_function("pitchdnn", |b| {
        b.iter(|| black_box(compute_pitchdnn(&mut p, &iff, &xcf, &m.pitchdnn, a)))
    });

    // FARGAN.
    let mut st = FarganState::new();
    let mut pcm0 = [0.0f32; FARGAN_CONT_SAMPLES];
    for (d, &v) in pcm0.iter_mut().zip(&s[310 * FRAME - FARGAN_CONT_SAMPLES..]) {
        *d = v as f32 / 32768.0;
    }
    let mut f0 = [0.0f32; CONT_VECTORS * NB_FEATURES];
    for (j, v) in f0.as_chunks_mut::<NB_FEATURES>().0.iter_mut().enumerate() {
        v.copy_from_slice(&feats[5 + j][..NB_FEATURES]);
    }
    group.bench_function("fargan_cont", |b| b.iter(|| fargan::fargan_cont(&mut st, &pcm0, &f0, &m.fargan, a)));
    let mut f = [0.0f32; NB_FEATURES];
    f.copy_from_slice(&feats[10][..NB_FEATURES]);
    let mut o = [0.0f32; FARGAN_FRAME_SIZE];
    group.bench_function("fargan_synthesize", |b| {
        b.iter(|| fargan::fargan_synthesize(&mut st, &mut o, &f, &m.fargan, a))
    });

    // The state machine: 40 good frames, then a loss.
    let mut ready = Box::new(DeepPlc::new());
    for k in 0..40 {
        ready.update(&frame_i16(&s[(300 + k) * FRAME..]));
    }
    let good = frame_i16(&s[400 * FRAME..]);
    let mut st = ready.clone();
    group.bench_function("update", |b| b.iter(|| st.update(black_box(&good))));
    group.bench_function("conceal_first", |b| {
        b.iter_batched_ref(
            || ready.clone(),
            |st| {
                let mut pcm = [0i16; FRAME];
                st.conceal(m, a, &mut pcm);
                pcm
            },
            BatchSize::SmallInput,
        )
    });
    let mut lossy = ready.clone();
    let mut pcm = [0i16; FRAME];
    lossy.conceal(m, a, &mut pcm);
    group.bench_function("conceal_next", |b| {
        b.iter(|| {
            lossy.conceal(m, a, &mut pcm);
            pcm
        })
    });

    // The CELT downsampler, stereo.
    let mem: Vec<Vec<f32>> = (0..2)
        .map(|c| (0..CELT_DECODE_BUFFER_SIZE).map(|i| s[i / 3 + c * 11] as f32).collect())
        .collect();
    let mut pre = 0.0f32;
    group.bench_function("update_from_celt", |b| b.iter(|| st.update_from_celt(&[&mem[0], &mem[1]], &mut pre)));
    group.finish();
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench
}
criterion_main!(benches);
