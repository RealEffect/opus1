#![no_main]
//! Weight blobs for the neural network models (`WeightList::parse`, the
//! model `init` functions and `ModelSet::load`).
//!
//! The first input byte picks how the rest is used:
//!
//! - as a blob of its own, placed 64-byte aligned or 1 to 3 bytes off;
//! - as changes to one of the built-in blobs: bytes set in the record
//!   headers or in the data, then the blob cut short, or not.
//!
//! What must hold: nothing panics. When a model loads, every layer has the
//! shape of its descriptor (`LAYERS`), and every array it reads lies inside
//! the blob. The same bytes, copied with `DnnBlob::from_bytes` and loaded
//! with `OpusDecoder::set_dnn_blob`, either load some model groups or
//! change nothing, and the decoder then still conceals a lost packet at
//! complexity 10.
//!
//! Needs the `dnn` feature of this package; `fuzz.sh` builds the target
//! with it.

use libfuzzer_sys::fuzz_target;
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::models::{
    Layer, LayerSpec, ModelSet, bbwenet, fargan, lace, nolace, pitchdnn, plc,
};
use opus1::internals::dnn::weights::WeightList;
use opus1::{DnnBlob, OpusDecoder};

/// The descriptors of every model, with the layers that `ModelSet` holds
/// for it once it is loaded.
fn loaded_layers<'a, 'w>(set: &'a ModelSet<'w>) -> Vec<(&'static [LayerSpec], Vec<Layer<'a, 'w>>)> {
    let mut out: Vec<(&'static [LayerSpec], Vec<Layer<'a, 'w>>)> = Vec::new();
    if let Some(m) = &set.deep_plc {
        out.push((&pitchdnn::LAYERS, m.pitchdnn.layers().to_vec()));
        out.push((&plc::LAYERS, m.plc.layers().to_vec()));
        out.push((&fargan::LAYERS, m.fargan.layers().to_vec()));
    }
    if let Some(m) = &set.lace {
        out.push((&lace::LAYERS, m.layers().to_vec()));
    }
    if let Some(m) = &set.nolace {
        out.push((&nolace::LAYERS, m.layers().to_vec()));
    }
    if let Some(m) = &set.bbwenet {
        out.push((&bbwenet::LAYERS, m.layers().to_vec()));
    }
    out
}

/// Checks that `s` lies inside `blob`.
fn inside<T>(blob: &[u8], s: Option<&[T]>) {
    if let Some(s) = s {
        let start = s.as_ptr() as usize;
        let end = start + std::mem::size_of_val(s);
        let (b0, b1) = (blob.as_ptr() as usize, blob.as_ptr() as usize + blob.len());
        assert!(b0 <= start && end <= b1, "an array lies outside the blob");
        assert_eq!(start % 4, 0, "an array is not 4-byte aligned");
    }
}

/// The public path: a copy of `blob` loaded into a decoder.
fn load_into_decoder(blob: &[u8]) {
    let Ok(b) = DnnBlob::from_bytes(blob) else { return };
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let before = dec.dnn_models();
    match dec.set_dnn_blob(&b) {
        Ok(m) => assert!(!m.is_empty() && dec.dnn_models().contains(m)),
        Err(_) => assert_eq!(dec.dnn_models(), before),
    }
    drop(b);
    dec.set_complexity(10).unwrap();
    let mut pcm = [0f32; 960];
    assert_eq!(dec.decode(&[], 960, &mut pcm), Ok(960));
}

fn exercise(blob: &[u8]) {
    load_into_decoder(blob);
    let Ok(list) = WeightList::parse(blob) else { return };
    for a in list.iter() {
        inside(blob, Some(a.data));
    }
    // Every layer of every model on its own, by its descriptor.
    for specs in [
        &pitchdnn::LAYERS[..],
        &plc::LAYERS[..],
        &fargan::LAYERS[..],
        &lace::LAYERS[..],
        &nolace::LAYERS[..],
        &bbwenet::LAYERS[..],
    ] {
        for spec in specs {
            let _ = spec.linear(&list);
            let _ = spec.conv2d(&list);
        }
    }
    let mut set = ModelSet::new();
    let loaded = set.load(&list);
    assert_eq!(loaded.is_ok(), !set.models().is_empty());
    for (specs, layers) in loaded_layers(&set) {
        assert_eq!(specs.len(), layers.len());
        for (spec, layer) in specs.iter().zip(&layers) {
            match (spec, layer) {
                (LayerSpec::Linear(s), Layer::Linear(l)) => {
                    assert_eq!((l.nb_inputs, l.nb_outputs), (s.nb_inputs, s.nb_outputs));
                    assert!(l.weights.is_some() || l.float_weights.is_some());
                    inside(blob, l.bias);
                    inside(blob, l.subias);
                    inside(blob, l.weights);
                    inside(blob, l.float_weights);
                    inside(blob, l.weights_idx);
                    inside(blob, l.diag);
                    inside(blob, l.scale);
                    if let Some(w) = l.weights {
                        assert!(!w.contains(&-128));
                        assert_eq!(w.len(), s.nb_inputs * s.nb_outputs);
                    }
                    if let Some(b) = l.bias {
                        assert_eq!(b.len(), s.nb_outputs);
                    }
                }
                (LayerSpec::Conv2d(s), Layer::Conv2d(c)) => {
                    assert_eq!(
                        (c.in_channels, c.out_channels, c.ktime, c.kheight),
                        (s.in_channels, s.out_channels, s.ktime, s.kheight)
                    );
                    inside(blob, c.bias);
                    inside(blob, c.float_weights);
                    let w = c.float_weights.expect("a loaded conv2d layer has weights");
                    assert_eq!(w.len(), s.in_channels * s.out_channels * s.ktime * s.kheight);
                }
                _ => panic!("a layer has another kind than its descriptor"),
            }
        }
    }
}

/// `bytes` copied to a 64-byte boundary plus `shift` bytes.
fn placed(bytes: &[u8], shift: usize) -> (Vec<u8>, usize) {
    let mut buf = vec![0u8; bytes.len() + 64 + shift];
    let off = buf.as_ptr().align_offset(64) + shift;
    buf[off..off + bytes.len()].copy_from_slice(bytes);
    (buf, off)
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else { return };
    if mode < 128 {
        // The input is the blob.
        let shift = usize::from(mode % 4);
        let (buf, off) = placed(rest, shift);
        exercise(&buf[off..off + rest.len()]);
        return;
    }
    // Changes to a built-in blob: a blob number, then groups of 4 bytes
    // (record number, where, value), then maybe a cut.
    let blobs = builtin::blobs();
    let Some((&pick, mut edits)) = rest.split_first() else { return };
    let mut blob = blobs[usize::from(pick) % blobs.len()].1.to_vec();
    let starts: Vec<usize> = {
        let (buf, off) = placed(&blob, 0);
        let list = WeightList::parse(&buf[off..off + blob.len()]).expect("built-in blobs parse");
        let base = buf[off..].as_ptr() as usize;
        list.iter().map(|a| a.data.as_ptr() as usize - base - 64).collect()
    };
    while let [rec, where_, value, extra, tail @ ..] = edits {
        let start = starts[usize::from(*rec) % starts.len()];
        let at = if where_ & 0x80 == 0 {
            // A byte of the header.
            start + usize::from(where_ % 64)
        } else {
            // A byte of the data (an int8 weight, a float byte, an index).
            start + 64 + ((usize::from(*where_ & 0x7f) << 8) | usize::from(*extra))
        };
        if at < blob.len() {
            blob[at] = *value;
        }
        edits = tail;
    }
    if mode & 1 == 1 && !edits.is_empty() {
        let cut = usize::from(edits[0]) * blob.len() / 256;
        blob.truncate(cut);
    }
    let (buf, off) = placed(&blob, 0);
    exercise(&buf[off..off + blob.len()]);
});
