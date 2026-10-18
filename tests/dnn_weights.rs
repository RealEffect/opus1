//! The weight blobs and their loader, without libopus.
//!
//! - The built-in blobs match `dnn-weights/SHA256SUMS`, start 64-byte
//!   aligned, read back record by record, and load every model of this
//!   build. Every layer has the arrays and the shape of its descriptor.
//! - Model groups load on their own: a group with a bad array stays as it
//!   was, and the others load.
//! - Malformed blobs and malformed layers give errors, never a panic:
//!   truncations, bad header fields, wrong sizes, missing arrays, bad
//!   sparse index tables, int8 weights equal to -128, and random changes.
//!
//! Run with a model feature, for example `--features osce-bwe`.

#![cfg(feature = "dnn")]

use opus1::internals::dnn::DnnModels;
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::models::{Layer, LayerSpec, ModelSet};
use opus1::internals::dnn::weights::{
    ERR_ARRAY_SIZE, ERR_INVALID_BLOB, ERR_LAYER_SHAPE, ERR_MISSING_ARRAY,
    ERR_NO_WEIGHTS, ERR_NOT_ALIGNED, ERR_SPARSE_INDEX, WeightArray, WeightList, conv2d_init,
    linear_init,
};
use opus1::internals::dnn::{Conv2dLayer, LinearLayer, MAX_INPUTS};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Bytes that start 64-byte aligned, as the built-in blobs do.
struct Aligned {
    buf: Vec<u8>,
    off: usize,
    len: usize,
}

impl Aligned {
    fn new(bytes: &[u8]) -> Aligned {
        let mut buf = vec![0u8; bytes.len() + 64];
        let off = buf.as_ptr().align_offset(64);
        buf[off..off + bytes.len()].copy_from_slice(bytes);
        Aligned { buf, off, len: bytes.len() }
    }
    fn bytes(&self) -> &[u8] {
        &self.buf[self.off..self.off + self.len]
    }
    #[cfg(feature = "osce")]
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.off..self.off + self.len]
    }
}

/// Appends one record in the format of libopus's `write_weights`.
fn write_record(out: &mut Vec<u8>, name: &[u8], kind: i32, data: &[u8]) {
    let block = data.len().div_ceil(64) * 64;
    out.extend_from_slice(b"DNNw");
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&(data.len() as i32).to_le_bytes());
    out.extend_from_slice(&(block as i32).to_le_bytes());
    let mut field = [0u8; 44];
    field[..name.len()].copy_from_slice(name);
    out.extend_from_slice(&field);
    out.extend_from_slice(data);
    out.resize(out.len() + block - data.len(), 0);
}

/// A blob of the arrays of `list`, in order, except those `skip` refuses.
fn rewrite(list: &WeightList<'_>, mut skip: impl FnMut(&WeightArray<'_>) -> bool) -> Vec<u8> {
    let mut out = Vec::new();
    for a in list.iter().filter(|a| !skip(a)) {
        write_record(&mut out, a.name, a.kind, a.data);
    }
    out
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The bytes of a slice of plain numbers.
fn bytes_of<T: Copy>(s: &[T]) -> &[u8] {
    // SAFETY: `T` is f32, i32 or i8 here: no padding, every byte initialized.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
}

/// The SHA-256 digest of `data`, as lowercase hexadecimal (FIPS 180-4).
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let [a, b, c, d, e, f, g, hh] = v;
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let t2 = s0.wrapping_add((a & b) ^ (a & c) ^ (b & c));
            v = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
        }
        for (x, y) in h.iter_mut().zip(v) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// The descriptors and built layers of every model of this build, from
/// `list`: (model, descriptors, init result as layer copies).
type ModelCheck<'w> = (&'static str, &'static [LayerSpec], Result<Vec<OwnedLayer<'w>>, &'static str>);

// A build with only `dnn` has no model to build.
#[cfg_attr(not(feature = "deep-plc"), allow(dead_code))]
#[derive(Clone, Copy)]
enum OwnedLayer<'w> {
    Linear(LinearLayer<'w>),
    Conv2d(Conv2dLayer<'w>),
}

#[cfg_attr(not(feature = "deep-plc"), allow(dead_code))]
fn owned<'w>(layers: &[Layer<'_, 'w>]) -> Vec<OwnedLayer<'w>> {
    layers
        .iter()
        .map(|l| match l {
            Layer::Linear(l) => OwnedLayer::Linear(**l),
            Layer::Conv2d(c) => OwnedLayer::Conv2d(**c),
        })
        .collect()
}

#[allow(unused_variables, unused_mut)]
fn init_all<'w>(list: &WeightList<'w>) -> Vec<ModelCheck<'w>> {
    let mut out: Vec<ModelCheck<'w>> = Vec::new();
    #[cfg(feature = "deep-plc")]
    {
        use opus1::internals::dnn::models::{fargan, pitchdnn, plc};
        out.push(("pitchdnn", &pitchdnn::LAYERS, pitchdnn::PitchDnn::init(list).map(|m| owned(&m.layers()))));
        out.push(("fargan", &fargan::LAYERS, fargan::Fargan::init(list).map(|m| owned(&m.layers()))));
        out.push(("plc", &plc::LAYERS, plc::PlcModel::init(list).map(|m| owned(&m.layers()))));
    }
    #[cfg(feature = "osce")]
    {
        use opus1::internals::dnn::models::{lace, nolace};
        out.push(("lace", &lace::LAYERS, lace::LaceLayers::init(list).map(|m| owned(&m.layers()))));
        out.push(("nolace", &nolace::LAYERS, nolace::NolaceLayers::init(list).map(|m| owned(&m.layers()))));
    }
    #[cfg(feature = "osce-bwe")]
    {
        use opus1::internals::dnn::models::bbwenet;
        out.push(("bbwenet", &bbwenet::LAYERS, bbwenet::BbwenetLayers::init(list).map(|m| owned(&m.layers()))));
    }
    out
}

/// Checks that a built layer reads the arrays its descriptor names, from
/// `list`, with the shape of the descriptor.
fn check_layer(list: &WeightList<'_>, spec: &LayerSpec, layer: &OwnedLayer<'_>) {
    /// The bytes of the named array in the list, when it is there.
    fn named<'w>(list: &WeightList<'w>, name: Option<&str>) -> Option<&'w [u8]> {
        list.find(name?).map(|a| a.data)
    }
    fn same<T: Copy>(what: &str, layer: &str, got: Option<&[T]>, want: Option<&[u8]>) {
        match (got, want) {
            (None, None) => {}
            (Some(g), Some(w)) => {
                assert_eq!(bytes_of(g).as_ptr(), w.as_ptr(), "{layer}.{what} points elsewhere");
                assert_eq!(bytes_of(g).len(), w.len(), "{layer}.{what} has another length");
            }
            (g, w) => panic!("{layer}.{what}: built {}, array {}", g.is_some(), w.is_some()),
        }
    }
    match (spec, layer) {
        (LayerSpec::Linear(s), OwnedLayer::Linear(l)) => {
            assert_eq!((l.nb_inputs, l.nb_outputs), (s.nb_inputs, s.nb_outputs), "{}", s.name);
            same("bias", s.name, l.bias, named(list, s.bias));
            same("subias", s.name, l.subias, named(list, s.subias));
            same("weights", s.name, l.weights, named(list, s.weights));
            same("float_weights", s.name, l.float_weights, named(list, s.float_weights));
            same("weights_idx", s.name, l.weights_idx, named(list, s.weights_idx));
            same("diag", s.name, l.diag, named(list, s.diag));
            let scale = if s.weights.is_some() { named(list, s.scale) } else { None };
            same("scale", s.name, l.scale, scale);
            assert!(l.weights.is_some() || l.float_weights.is_some(), "{} has no weights", s.name);
            // No 1.6.1 layer has a diagonal (nnet.md, Appendix A).
            assert!(l.diag.is_none());
        }
        (LayerSpec::Conv2d(s), OwnedLayer::Conv2d(c)) => {
            assert_eq!(
                (c.in_channels, c.out_channels, c.ktime, c.kheight),
                (s.in_channels, s.out_channels, s.ktime, s.kheight),
                "{}",
                s.name
            );
            same("bias", s.name, c.bias, named(list, s.bias));
            same("float_weights", s.name, c.float_weights, named(list, s.float_weights));
        }
        _ => panic!("{}: the layer kind differs from its descriptor", spec.name()),
    }
}

// ---------------------------------------------------------------------------
// Built-in blobs
// ---------------------------------------------------------------------------

#[test]
fn builtin_blobs_match_sha256sums_and_version() {
    let sums = include_str!("../dnn-weights/SHA256SUMS");
    let version = include_str!("../dnn-weights/VERSION");
    assert_eq!(version.lines().next(), Some(format!("libopus {}", builtin::LIBOPUS_VERSION).as_str()));
    for &(file, blob) in builtin::blobs() {
        let line = sums
            .lines()
            .find(|l| l.ends_with(&format!("  {file}")))
            .unwrap_or_else(|| panic!("{file} is not in SHA256SUMS"));
        assert_eq!(&line[..64], sha256_hex(blob), "{file}");
        assert_eq!(blob.as_ptr() as usize % 64, 0, "{file} is not 64-byte aligned");
    }
    let expected: &[&str] = &[
        #[cfg(feature = "deep-plc")]
        "pitchdnn.bin",
        #[cfg(feature = "deep-plc")]
        "fargan.bin",
        #[cfg(feature = "deep-plc")]
        "plc.bin",
        #[cfg(feature = "osce")]
        "lace.bin",
        #[cfg(feature = "osce")]
        "nolace.bin",
        #[cfg(feature = "osce-bwe")]
        "bbwenet.bin",
    ];
    let files: Vec<&str> = builtin::blobs().iter().map(|b| b.0).collect();
    assert_eq!(files, expected);
}

#[test]
fn sha256_of_known_strings() {
    assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn builtin_blobs_read_back_record_by_record() {
    for &(file, blob) in builtin::blobs() {
        let list = WeightList::parse(blob).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(!list.is_empty());
        // Writing the records again gives the same bytes.
        assert_eq!(rewrite(&list, |_| false), blob, "{file}");
        for a in list.iter() {
            assert_eq!(a.data.as_ptr() as usize % 64, 0, "{file}: every array starts 64-byte aligned");
            if a.kind == 3 {
                assert!(!a.data.contains(&0x80), "{file}: int8 -128");
            }
        }
    }
    // The list of all built-in blobs holds every array of each.
    let total: usize = builtin::blobs().iter().map(|b| WeightList::parse(b.1).unwrap().len()).sum();
    assert_eq!(builtin::weights().len(), total);
    assert_eq!(builtin::weights().blobs().len(), builtin::blobs().len());
}

#[test]
fn every_model_loads_with_the_shapes_of_its_descriptors() {
    // Each model loads from its own blob, and from no other blob.
    for &(file, blob) in builtin::blobs() {
        let list = WeightList::parse(blob).unwrap();
        for (model, specs, built) in init_all(&list) {
            if file != format!("{model}.bin") {
                assert!(built.is_err(), "{model} loads from {file}");
                continue;
            }
            let layers = built.unwrap_or_else(|e| panic!("{model}: {e}"));
            assert_eq!(layers.len(), specs.len(), "{model}");
            for (spec, layer) in specs.iter().zip(&layers) {
                check_layer(&list, spec, layer);
            }
        }
    }
    // Every model loads from the list of all built-in blobs.
    let all = init_all(builtin::weights());
    for (model, specs, built) in &all {
        let layers = built.as_ref().unwrap_or_else(|e| panic!("{model}: {e}"));
        for (spec, layer) in specs.iter().zip(layers) {
            check_layer(builtin::weights(), spec, layer);
        }
    }
    let counts: Vec<(&str, usize)> = all.iter().map(|m| (m.0, m.1.len())).collect();
    let want: &[(&str, usize)] = &[
        #[cfg(feature = "deep-plc")]
        ("pitchdnn", 8),
        #[cfg(feature = "deep-plc")]
        ("fargan", 20),
        #[cfg(feature = "deep-plc")]
        ("plc", 6),
        #[cfg(feature = "osce")]
        ("lace", 14),
        #[cfg(feature = "osce")]
        ("nolace", 34),
        #[cfg(feature = "osce-bwe")]
        ("bbwenet", 17),
    ];
    assert_eq!(counts, want);
}

/// The groups of this build.
fn all_groups() -> DnnModels {
    let mut m = DnnModels::NONE;
    if cfg!(feature = "deep-plc") {
        m = m.union(DnnModels::DEEP_PLC);
    }
    if cfg!(feature = "osce") {
        m = m.union(DnnModels::LACE).union(DnnModels::NOLACE);
    }
    if cfg!(feature = "osce-bwe") {
        m = m.union(DnnModels::BBWENET);
    }
    m
}

#[test]
fn builtin_models_load_every_group() {
    assert_eq!(builtin::models().models(), all_groups());
    let mut set = ModelSet::new();
    assert!(set.models().is_empty());
    let r = set.load(builtin::weights());
    if all_groups().is_empty() {
        assert!(r.is_err());
    } else {
        assert_eq!(r, Ok(all_groups()));
    }
    assert_eq!(set.models(), all_groups());
}

#[test]
fn concatenated_blobs_load_like_separate_blobs() {
    let joined: Vec<u8> = builtin::blobs().iter().flat_map(|b| b.1.iter().copied()).collect();
    let buf = Aligned::new(&joined);
    let one = WeightList::parse(buf.bytes()).unwrap();
    let many = builtin::weights();
    assert_eq!(one.len(), many.len());
    for (a, b) in one.iter().zip(many.iter()) {
        assert_eq!((a.name, a.kind, a.data), (b.name, b.kind, b.data));
    }
    let mut set = ModelSet::new();
    if !all_groups().is_empty() {
        assert_eq!(set.load(&one), Ok(all_groups()));
    }
}

// ---------------------------------------------------------------------------
// Loading model groups on their own
// ---------------------------------------------------------------------------

#[cfg(feature = "osce")]
#[test]
fn a_bad_group_keeps_what_it_had_and_the_others_load() {
    let all = builtin::weights();
    // Without one LACE array.
    let without = Aligned::new(&rewrite(all, |a| a.name == b"lace_cf1_gain_bias"));
    let list = WeightList::parse(without.bytes()).unwrap();
    let mut set = ModelSet::new();
    let want = all_groups().bits() & !DnnModels::LACE.bits();
    assert_eq!(set.load(&list).map(|m| m.bits()), Ok(want));
    assert!(set.lace.is_none());
    assert!(set.nolace.is_some());

    // A set that has LACE keeps its old LACE when a reload fails.
    let mut set = *builtin::models();
    let old = set.lace.unwrap().lace_fnet_conv2.weights.unwrap().as_ptr();
    assert_eq!(set.load(&list).map(|m| m.bits()), Ok(want));
    assert_eq!(set.lace.unwrap().lace_fnet_conv2.weights.unwrap().as_ptr(), old);
    // NoLACE was replaced by the arrays of the new list.
    let new = list.find("nolace_fnet_conv2_weights_int8").unwrap().data.as_ptr();
    assert_eq!(set.nolace.unwrap().nolace_fnet_conv2.weights.unwrap().as_ptr().cast(), new);

    // A NoLACE array of another size, and an int8 weight of -128 in LACE.
    let mut wrong = Vec::new();
    for a in all.iter() {
        let data = if a.name == b"nolace_af2_gain_bias" { &a.data[..4] } else { a.data };
        write_record(&mut wrong, a.name, a.kind, data);
    }
    let mut wrong = Aligned::new(&wrong);
    let at = {
        let list = WeightList::parse(wrong.bytes()).unwrap();
        let a = list.find("lace_fnet_tconv_weights_int8").unwrap();
        a.data.as_ptr() as usize - wrong.bytes().as_ptr() as usize + 17
    };
    wrong.bytes_mut()[at] = 0x80;
    let list = WeightList::parse(wrong.bytes()).unwrap();
    use opus1::internals::dnn::models::{lace::LaceLayers, nolace::NolaceLayers};
    assert_eq!(LaceLayers::init(&list).err(), Some(opus1::internals::dnn::weights::ERR_INT8_MINUS_128));
    assert_eq!(NolaceLayers::init(&list).err(), Some(ERR_ARRAY_SIZE));
    let mut set = ModelSet::new();
    let want = all_groups().bits() & !(DnnModels::LACE.bits() | DnnModels::NOLACE.bits());
    assert_eq!(set.load(&list).map(|m| m.bits()), Ok(want));
}

#[test]
fn a_blob_without_any_model_of_this_build_loads_nothing() {
    let mut blob = Vec::new();
    write_record(&mut blob, b"x_bias", 0, &floats(&[1.0; 8]));
    let buf = Aligned::new(&blob);
    let list = WeightList::parse(buf.bytes()).unwrap();
    let mut set = *builtin::models();
    assert!(set.load(&list).is_err());
    assert_eq!(set.models(), all_groups());
    assert!(ModelSet::new().load(&WeightList::empty()).is_err());
}

// ---------------------------------------------------------------------------
// Malformed blobs
// ---------------------------------------------------------------------------

/// A small blob with two records, the second at offset 128.
fn small_blob() -> Vec<u8> {
    let mut b = Vec::new();
    write_record(&mut b, b"a_bias", 0, &floats(&[0.5; 8]));
    write_record(&mut b, b"a_weights_int8", 3, &[1u8; 64]);
    b
}

#[test]
fn truncated_and_extended_blobs_are_refused() {
    let b = small_blob();
    for len in 0..=b.len() + 70 {
        let mut v = b.clone();
        v.resize(len, 0);
        let buf = Aligned::new(&v);
        let r = WeightList::parse(buf.bytes()).map(|l| l.len());
        let want = match len {
            0 => Ok(0),
            128 => Ok(1),
            256 => Ok(2),
            _ => Err(ERR_INVALID_BLOB),
        };
        assert_eq!(r, want, "length {len}");
    }
    // Every built-in blob, cut inside the header and inside the data of
    // each record, and at each record boundary.
    for &(file, blob) in builtin::blobs() {
        let list = WeightList::parse(blob).unwrap();
        for (k, a) in list.iter().enumerate() {
            let start = a.data.as_ptr() as usize - blob.as_ptr() as usize;
            for cut in [start - 63, start - 1, start, start + a.data.len() / 2] {
                assert_eq!(WeightList::parse(&blob[..cut]).err(), Some(ERR_INVALID_BLOB), "{file} cut at {cut}");
            }
            assert_eq!(WeightList::parse(&blob[..start - 64]).map(|l| l.len()), Ok(k), "{file}");
        }
    }
}

#[test]
fn bad_header_fields_are_refused() {
    let b = small_blob();
    let set = |at: usize, v: i32| {
        let mut v2 = b.clone();
        v2[at..at + 4].copy_from_slice(&v.to_le_bytes());
        v2
    };
    let parse = |v: &[u8]| WeightList::parse(Aligned::new(v).bytes()).map(|l| l.len());
    // size: 0, negative, above block_size; block_size: below size, past the
    // end, negative; the last name byte.
    for (at, v) in [(12, 0), (12, -1), (12, 65), (16, 16), (16, 128), (16, -64), (16, i32::MAX), (16, i32::MIN)] {
        assert_eq!(parse(&set(at, v)), Err(ERR_INVALID_BLOB), "field at {at} set to {v}");
        assert_eq!(parse(&set(128 + at, v)), Err(ERR_INVALID_BLOB), "second record, field at {at} set to {v}");
    }
    let mut v = b.clone();
    v[63] = 1;
    assert_eq!(parse(&v), Err(ERR_INVALID_BLOB));
    // A name that fills all 43 bytes is fine, and is found by its full name.
    let mut v = b.clone();
    v[20..63].fill(b'n');
    let buf = Aligned::new(&v);
    let list = WeightList::parse(buf.bytes()).unwrap();
    assert!(list.find(&"n".repeat(43)).is_some());
    // libopus accepts another head, version or type.
    assert_eq!(parse(&set(0, 0)), Ok(2));
    assert_eq!(parse(&set(4, 9)), Ok(2));
    assert_eq!(parse(&set(8, -5)), Ok(2));
    // A block_size that is not a multiple of 4 moves the next record's data
    // off its alignment; opus1 refuses that, unlike libopus.
    let mut v = Vec::new();
    write_record(&mut v, b"a", 3, &[1u8; 5]);
    v[16..20].copy_from_slice(&5i32.to_le_bytes());
    v.truncate(64 + 5);
    write_record(&mut v, b"b", 3, &[1u8; 4]);
    assert_eq!(parse(&v), Err(ERR_NOT_ALIGNED));
    // A blob that starts on an odd address.
    let mut odd = vec![0u8; b.len() + 65];
    let off = odd.as_ptr().align_offset(64) + 1;
    odd[off..off + b.len()].copy_from_slice(&b);
    assert_eq!(WeightList::parse(&odd[off..off + b.len()]).map(|l| l.len()), Err(ERR_NOT_ALIGNED));
}

// ---------------------------------------------------------------------------
// Malformed layers
// ---------------------------------------------------------------------------

/// A blob with the arrays of a sparse int8 layer of 16 inputs and 16
/// outputs (two row blocks), with the index table `idx`, and of some other
/// layers.
fn layer_blob(idx: &[i32], blocks: usize) -> Aligned {
    let mut b = Vec::new();
    write_record(&mut b, b"s_bias", 0, &floats(&[0.1; 16]));
    write_record(&mut b, b"s_scale", 0, &floats(&[0.01; 16]));
    write_record(&mut b, b"s_weights_idx", 1, &ints(idx));
    write_record(&mut b, b"s_weights_int8", 3, &vec![3u8; 32 * blocks.max(1)]);
    write_record(&mut b, b"s_weights_float", 0, &floats(&vec![0.5; 32 * blocks.max(1)]));
    write_record(&mut b, b"d_weights_int8", 3, &[1u8; 16 * 8]);
    write_record(&mut b, b"g_diag", 0, &floats(&[0.0; 24]));
    write_record(&mut b, b"g_weights_float", 0, &floats(&[0.0; 8 * 24]));
    write_record(&mut b, b"c_weights_float", 0, &floats(&[0.0; 18]));
    Aligned::new(&b)
}

/// Builds the sparse layer of `layer_blob`, with int8 or float weights.
fn sparse(list: &WeightList<'_>, float: bool) -> Result<(), &'static str> {
    linear_init(
        list,
        Some("s_bias"),
        None,
        if float { None } else { Some("s_weights_int8") },
        if float { Some("s_weights_float") } else { None },
        Some("s_weights_idx"),
        None,
        if float { None } else { Some("s_scale") },
        16,
        16,
    )
    .map(|_| ())
}

#[test]
fn sparse_index_tables() {
    // Valid: 2 blocks in rows 0-7 (columns 0 and 12), 1 block in rows 8-15.
    let good = [2, 0, 12, 1, 4];
    let buf = layer_blob(&good, 3);
    let list = WeightList::parse(buf.bytes()).unwrap();
    let l = linear_init(&list, Some("s_bias"), None, Some("s_weights_int8"), None, Some("s_weights_idx"), None, Some("s_scale"), 16, 16).unwrap();
    assert_eq!(l.weights_idx.unwrap(), &good);
    assert_eq!(l.weights.unwrap().len(), 96);
    assert!(sparse(&list, true).is_ok());
    // A trailing partial int is ignored, as in libopus.
    let mut b = Vec::new();
    write_record(&mut b, b"s_weights_idx", 1, &[ints(&good), vec![9, 9]].concat());
    write_record(&mut b, b"s_weights_int8", 3, &[3u8; 96]);
    write_record(&mut b, b"s_scale", 0, &floats(&[0.01; 16]));
    write_record(&mut b, b"s_bias", 0, &floats(&[0.1; 16]));
    let buf2 = Aligned::new(&b);
    assert!(sparse(&WeightList::parse(buf2.bytes()).unwrap(), false).is_ok());

    let bad: &[(&[i32], usize, &str)] = &[
        (&[-1, 0, 0], 1, ERR_SPARSE_INDEX),             // negative count
        (&[2, 0, 12, -3, 4], 3, ERR_SPARSE_INDEX),     // negative count in the second row block
        (&[1, -4, 1, 4], 2, ERR_SPARSE_INDEX),          // negative position
        (&[1, i32::MIN, 1, 4], 2, ERR_SPARSE_INDEX),    // the most negative position
        (&[1, 2, 1, 4], 2, ERR_SPARSE_INDEX),           // not a multiple of 4
        (&[1, 13, 1, 4], 2, ERR_SPARSE_INDEX),          // pos + 3 beyond the inputs, and odd
        (&[1, 16, 1, 4], 2, ERR_SPARSE_INDEX),          // pos + 3 >= nb_inputs
        (&[1, 0], 1, ERR_SPARSE_INDEX),                 // only one row block
        (&[1, 0, 1, 4, 0], 2, ERR_SPARSE_INDEX),        // three row blocks
        (&[3, 0, 4], 2, ERR_SPARSE_INDEX),              // a count past the end
        (&[i32::MAX, 0], 1, ERR_SPARSE_INDEX),          // a huge count
        (&[2, 0, 12, 1, 4], 2, ERR_ARRAY_SIZE),         // weights for 2 blocks, the table has 3
    ];
    for &(idx, blocks, err) in bad {
        let buf = layer_blob(idx, blocks);
        let list = WeightList::parse(buf.bytes()).unwrap();
        assert_eq!(sparse(&list, false).err(), Some(err), "int8, table {idx:?}");
        if err == ERR_SPARSE_INDEX {
            assert_eq!(sparse(&list, true).err(), Some(err), "float, table {idx:?}");
        }
    }
    // A missing table.
    let r = linear_init(&list, None, None, Some("s_weights_int8"), None, Some("x_weights_idx"), None, Some("s_scale"), 16, 16);
    assert_eq!(r.err(), Some(ERR_MISSING_ARRAY));
}

#[test]
fn layers_with_missing_or_wrong_arrays() {
    let buf = layer_blob(&[1, 0, 1, 0], 2);
    let list = WeightList::parse(buf.bytes()).unwrap();
    let dense = |bias, weights, float_weights, scale, nin, nout| {
        linear_init(&list, bias, None, weights, float_weights, None, None, scale, nin, nout).map(|_| ())
    };
    // int8 dense: 16 inputs, 8 outputs.
    assert_eq!(dense(None, Some("d_weights_int8"), None, Some("s_scale"), 16, 8), Err(ERR_ARRAY_SIZE));
    assert_eq!(dense(None, Some("d_weights_int8"), None, Some("s_bias"), 8, 16), Ok(()));
    assert_eq!(dense(None, Some("d_weights_int8"), None, None, 8, 16), Err(ERR_MISSING_ARRAY));
    assert_eq!(dense(None, Some("d_weights_int8"), None, Some("x_scale"), 8, 16), Err(ERR_MISSING_ARRAY));
    assert_eq!(dense(Some("x_bias"), Some("d_weights_int8"), None, Some("s_scale"), 8, 16), Err(ERR_MISSING_ARRAY));
    assert_eq!(dense(None, Some("x_weights_int8"), None, Some("s_scale"), 8, 16), Err(ERR_MISSING_ARRAY));
    // int8 shapes that the kernels cannot run: more than MAX_INPUTS inputs,
    // an input count that is not a multiple of 4.
    let big = vec![1u8; (MAX_INPUTS + 4) * 8];
    let mut b = Vec::new();
    write_record(&mut b, b"w", 3, &big);
    write_record(&mut b, b"s", 0, &floats(&[1.0; 8]));
    write_record(&mut b, b"w2", 3, &[1u8; 6 * 8]);
    let buf2 = Aligned::new(&b);
    let list2 = WeightList::parse(buf2.bytes()).unwrap();
    let l2 = |w, nin, nout| linear_init(&list2, None, None, Some(w), None, None, None, Some("s"), nin, nout).map(|_| ());
    assert_eq!(l2("w", MAX_INPUTS + 4, 8), Err(ERR_LAYER_SHAPE));
    assert_eq!(l2("w2", 6, 8), Err(ERR_LAYER_SHAPE));
    // A float layer: its weights are optional in libopus, required here.
    assert_eq!(dense(None, None, Some("x_weights_float"), None, 16, 16), Err(ERR_NO_WEIGHTS));
    assert_eq!(dense(None, None, None, None, 16, 16), Err(ERR_NO_WEIGHTS));
    assert_eq!(dense(None, None, Some("s_weights_float"), None, 4, 4), Err(ERR_ARRAY_SIZE));
    // diag: 3 · nb_inputs outputs.
    let gru = |nin, nout| {
        linear_init(&list, None, None, None, Some("g_weights_float"), None, Some("g_diag"), None, nin, nout).map(|_| ())
    };
    assert_eq!(gru(8, 24), Ok(()));
    let mut b = Vec::new();
    write_record(&mut b, b"g_diag", 0, &floats(&[0.0; 24]));
    write_record(&mut b, b"g_weights_float", 0, &floats(&[0.0; 6 * 24]));
    let buf3 = Aligned::new(&b);
    let list3 = WeightList::parse(buf3.bytes()).unwrap();
    let r = linear_init(&list3, None, None, None, Some("g_weights_float"), None, Some("g_diag"), None, 6, 24);
    assert_eq!(r.err(), Some(ERR_LAYER_SHAPE));
    // Byte counts that overflow.
    assert_eq!(dense(None, None, Some("s_weights_float"), None, usize::MAX, 3), Err(ERR_LAYER_SHAPE));
    assert_eq!(dense(Some("s_bias"), None, Some("s_weights_float"), None, 3, usize::MAX / 2), Err(ERR_LAYER_SHAPE));
    // conv2d: 2 input channels, 1 output channel, 3 × 3.
    let conv = |bias, w, cin| conv2d_init(&list, bias, w, cin, 1, 3, 3).map(|_| ());
    assert_eq!(conv(None, Some("c_weights_float"), 2), Ok(()));
    assert_eq!(conv(None, Some("c_weights_float"), 1), Err(ERR_ARRAY_SIZE));
    assert_eq!(conv(None, Some("x_weights_float"), 2), Err(ERR_NO_WEIGHTS));
    assert_eq!(conv(None, None, 2), Err(ERR_NO_WEIGHTS));
    assert_eq!(conv(Some("s_bias"), Some("c_weights_float"), 2), Err(ERR_ARRAY_SIZE));
    assert_eq!(conv2d_init(&list, None, Some("c"), usize::MAX, 2, 3, 3).err(), Some(ERR_LAYER_SHAPE));
}

// ---------------------------------------------------------------------------
// Random changes never panic
// ---------------------------------------------------------------------------

/// Parses `bytes` and, when that works, builds every model of this build
/// and every layer of their descriptors on their own.
fn exercise(bytes: &[u8]) {
    let Ok(list) = WeightList::parse(bytes) else { return };
    for (_, specs, built) in init_all(&list) {
        if let Ok(layers) = &built {
            for (spec, layer) in specs.iter().zip(layers) {
                check_layer(&list, spec, layer);
            }
        }
        for spec in specs {
            let _ = spec.linear(&list);
            let _ = spec.conv2d(&list);
        }
    }
    let mut set = ModelSet::new();
    let _ = set.load(&list);
}

/// The small built-in blobs, which the random tests change.
fn seeds() -> Vec<Vec<u8>> {
    let mut out = vec![small_blob()];
    for &(file, blob) in builtin::blobs() {
        if file == "pitchdnn.bin" || file == "lace.bin" {
            out.push(blob.to_vec());
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn random_header_changes_never_panic(
        seed in any::<prop::sample::Index>(),
        changes in prop::collection::vec((any::<prop::sample::Index>(), 0usize..64, any::<u8>()), 1..6),
        cut in any::<prop::sample::Index>(),
        do_cut in any::<bool>(),
    ) {
        let seeds = seeds();
        let mut blob = seeds[seed.index(seeds.len())].clone();
        // Change bytes inside record headers, where the parser looks.
        let starts: Vec<usize> = {
            let buf = Aligned::new(&blob);
            let list = WeightList::parse(buf.bytes()).unwrap();
            list.iter().map(|a| a.data.as_ptr() as usize - buf.bytes().as_ptr() as usize - 64).collect()
        };
        for (rec, at, byte) in changes {
            let pos = starts[rec.index(starts.len())] + at;
            blob[pos] = byte;
        }
        if do_cut {
            blob.truncate(cut.index(blob.len() + 1));
        }
        exercise(Aligned::new(&blob).bytes());
    }

    #[test]
    fn random_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..600)) {
        exercise(Aligned::new(&bytes).bytes());
        exercise(&bytes);
    }

    #[test]
    fn random_index_tables_never_panic(idx in prop::collection::vec(-40i32..40, 1..12), blocks in 0usize..5) {
        let buf = layer_blob(&idx, blocks);
        let list = WeightList::parse(buf.bytes()).unwrap();
        let _ = sparse(&list, false);
        let _ = sparse(&list, true);
    }
}
