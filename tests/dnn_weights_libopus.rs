#![cfg(feature = "test-libopus-dnn")]
//! The weights and their loader against libopus 1.6.1 built with its
//! neural network code.
//!
//! - opus1's blobs hold the arrays of libopus's `*_arrays` lists, in order,
//!   with the same names, types and bytes. Concatenated in the order of
//!   libopus's blob writer they give its blob (`dump_weights_blob`), and
//!   with BBWENet appended the complete blob.
//! - Every model built by opus1 has the layers that libopus's `init_*`
//!   functions build: the same shapes, the same arrays present or absent,
//!   and the same array contents. The C init functions also accept opus1's
//!   blobs, read with libopus's own `parse_weights`.
//! - The layers of the two DRED models, also in a build without the
//!   feature `dred`, load from their descriptors like `linear_init` loads
//!   them, sparse layers included.
//! - `WeightList::parse` accepts and refuses changed blobs as
//!   `parse_weights` does, and `linear_init` refuses bad index tables that
//!   libopus refuses.
//!
//! Run against the bit-exact reference build, with the model features:
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 \
//!   LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dnn-ref cargo test --release \
//!   --features test-libopus-dnn,osce-bwe --test dnn_weights_libopus
//! ```
//!
//! With `dred,osce-bwe` it runs the same way, against either the build
//! above or the one with DRED (`libopus-host-1.6.1-dred-novec`).
//!
//! The comparison with the blob files of the reference build looks in
//! `OPUS1_DNN_BLOB_DIR`, then in `target/android/dnn-blob-1.6.1` of the
//! repository (also from a worktree under `target/wt/`), and is skipped
//! with a message when neither exists.

#[cfg(not(feature = "dnn"))]
compile_error!("run the libopus weight tests with a model feature, for example --features test-libopus-dnn,osce-bwe");

mod dnn_gen;

use dnn_gen::{CLayer, CLinearLayer, CModel, CWeightArray, MODELS};
use opus1::internals::dnn::builtin;
#[cfg(feature = "deep-plc")]
use opus1::internals::dnn::models::Layer;
use opus1::internals::dnn::models::LayerSpec;
use opus1::internals::dnn::weights::{ERR_SPARSE_INDEX, WeightList, linear_init};
use opus1::internals::dnn::{Conv2dLayer, LinearLayer};
use opus1::Error;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::PathBuf;

unsafe extern "C" {
    /// `parse_weights` of `dnn/parse_lpcnet_weights.c`. The list is
    /// allocated with `malloc` (`opus_alloc`); it is `NULL` on failure.
    fn parse_weights(list: *mut *mut CWeightArray, data: *const c_void, len: c_int) -> c_int;
    #[link_name = "linear_init"]
    fn c_linear_init(
        layer: *mut CLinearLayer,
        arrays: *const CWeightArray,
        bias: *const c_char,
        subias: *const c_char,
        weights: *const c_char,
        float_weights: *const c_char,
        weights_idx: *const c_char,
        diag: *const c_char,
        scale: *const c_char,
        nb_inputs: c_int,
        nb_outputs: c_int,
    ) -> c_int;
    fn free(p: *mut c_void);
}

/// Makes sure the test binary links libopus: the functions above are found
/// in the static library that `audiopus_sys` brings along.
fn linked() {
    let version = unsafe { CStr::from_ptr(audiopus_sys::opus_get_version_string()) };
    assert!(version.to_string_lossy().contains("1.6.1"), "{version:?}");
}

/// Bytes that start 64-byte aligned.
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
}

/// One entry of a C list.
struct CEntry {
    name: Vec<u8>,
    kind: i32,
    data: &'static [u8],
}

/// The entries of a C list, up to its `NULL` name.
fn c_entries(mut p: *const CWeightArray) -> Vec<CEntry> {
    let mut out = Vec::new();
    // SAFETY: the list is a static array of libopus that ends with a NULL
    // name; each entry's data has `size` bytes.
    unsafe {
        while !(*p).name.is_null() {
            let e = &*p;
            let data = std::slice::from_raw_parts(e.data.cast::<u8>(), e.size as usize);
            out.push(CEntry { name: CStr::from_ptr(e.name).to_bytes().to_vec(), kind: e.kind, data });
            p = p.add(1);
        }
    }
    out
}

/// The blob that libopus's `write_weights` writes for a C list.
fn c_blob(model: &CModel) -> Vec<u8> {
    let mut out = Vec::new();
    for e in c_entries((model.arrays)()) {
        let block = e.data.len().div_ceil(64) * 64;
        out.extend_from_slice(b"DNNw");
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&e.kind.to_le_bytes());
        out.extend_from_slice(&(e.data.len() as i32).to_le_bytes());
        out.extend_from_slice(&(block as i32).to_le_bytes());
        let mut name = [0u8; 44];
        name[..e.name.len()].copy_from_slice(&e.name);
        out.extend_from_slice(&name);
        out.extend_from_slice(e.data);
        out.resize(out.len() + block - e.data.len(), 0);
    }
    out
}

/// opus1's built-in blob of a model, when this build includes it.
fn builtin_blob(key: &str) -> Option<&'static [u8]> {
    let file = format!("{key}.bin");
    builtin::blobs().iter().find(|b| b.0 == file).map(|b| b.1)
}

#[derive(Clone, Copy)]
enum RustLayer<'w> {
    Linear(LinearLayer<'w>),
    Conv2d(Conv2dLayer<'w>),
}

#[cfg(feature = "deep-plc")]
fn owned<'w>(layers: &[Layer<'_, 'w>]) -> Vec<RustLayer<'w>> {
    layers
        .iter()
        .map(|l| match l {
            Layer::Linear(l) => RustLayer::Linear(**l),
            Layer::Conv2d(c) => RustLayer::Conv2d(**c),
        })
        .collect()
}

/// A model's descriptors, and its layers as opus1 builds them.
type Built<'w> = (&'static [LayerSpec], Result<Vec<RustLayer<'w>>, Error>);

/// opus1's model `key` built from `list`, with its descriptors, when this
/// build includes the model.
#[allow(unused_variables)]
fn rust_model<'w>(key: &str, list: &WeightList<'w>) -> Option<Built<'w>> {
    #[cfg(feature = "deep-plc")]
    {
        use opus1::internals::dnn::models::{fargan, pitchdnn, plc};
        match key {
            "pitchdnn" => return Some((&pitchdnn::LAYERS, pitchdnn::PitchDnn::init(list).map(|m| owned(&m.layers())))),
            "fargan" => return Some((&fargan::LAYERS, fargan::Fargan::init(list).map(|m| owned(&m.layers())))),
            "plc" => return Some((&plc::LAYERS, plc::PlcModel::init(list).map(|m| owned(&m.layers())))),
            _ => {}
        }
    }
    #[cfg(feature = "dred")]
    {
        use opus1::internals::dnn::models::{dred_rdovae_dec as dec, dred_rdovae_enc as enc};
        match key {
            "dred_rdovae_enc" => return Some((&enc::LAYERS, enc::RdovaeEnc::init(list).map(|m| owned(&m.layers())))),
            "dred_rdovae_dec" => return Some((&dec::LAYERS, dec::RdovaeDec::init(list).map(|m| owned(&m.layers())))),
            _ => {}
        }
    }
    #[cfg(feature = "osce")]
    {
        use opus1::internals::dnn::models::{lace, nolace};
        match key {
            "lace" => return Some((&lace::LAYERS, lace::LaceLayers::init(list).map(|m| owned(&m.layers())))),
            "nolace" => return Some((&nolace::LAYERS, nolace::NolaceLayers::init(list).map(|m| owned(&m.layers())))),
            _ => {}
        }
    }
    #[cfg(feature = "osce-bwe")]
    {
        use opus1::internals::dnn::models::bbwenet;
        if key == "bbwenet" {
            return Some((&bbwenet::LAYERS, bbwenet::BbwenetLayers::init(list).map(|m| owned(&m.layers()))));
        }
    }
    None
}

/// Every layer of a descriptor list, built by opus1's `LayerSpec` methods.
fn rust_layers<'w>(specs: &[LayerSpec], list: &WeightList<'w>) -> Result<Vec<RustLayer<'w>>, Error> {
    specs
        .iter()
        .map(|s| match s {
            LayerSpec::Linear(_) => s.linear(list).map(RustLayer::Linear),
            LayerSpec::Conv2d(_) => s.conv2d(list).map(RustLayer::Conv2d),
        })
        .collect()
}

/// Compares an opus1 array with a C pointer: both absent, or both present
/// with the same bytes.
fn same_array<T: Copy>(what: &str, rust: Option<&[T]>, c: *const T) {
    match rust {
        None => assert!(c.is_null(), "{what}: libopus has the array, opus1 not"),
        Some(r) => {
            assert!(!c.is_null(), "{what}: opus1 has the array, libopus not");
            let bytes = std::mem::size_of_val(r);
            // SAFETY: libopus's array has at least the size that opus1
            // checked for the same layer shape (the comparison of the
            // shapes comes first, and the C code checks the same sizes).
            let c_bytes = unsafe { std::slice::from_raw_parts(c.cast::<u8>(), bytes) };
            // SAFETY: plain numbers, every byte initialized.
            let r_bytes = unsafe { std::slice::from_raw_parts(r.as_ptr().cast::<u8>(), bytes) };
            assert!(c_bytes == r_bytes, "{what}: the contents differ");
        }
    }
}

/// The number of `int` values of a C index table that covers `nb_outputs`
/// rows.
fn c_idx_len(idx: *const c_int, nb_outputs: usize) -> usize {
    let mut n = 0;
    for _ in 0..nb_outputs / 8 {
        // SAFETY: libopus accepted the table, so it holds these row blocks.
        let count = unsafe { *idx.add(n) } as usize;
        n += 1 + count;
    }
    n
}

/// Compares the layers that opus1 and libopus built, layer by layer.
fn compare_layers(model: &str, specs: &[LayerSpec], c: &[CLayer], rust: &[RustLayer<'_>]) {
    assert_eq!(c.len(), specs.len(), "{model}");
    assert_eq!(rust.len(), specs.len(), "{model}");
    for ((spec, c), r) in specs.iter().zip(c).zip(rust) {
        let name = format!("{model}.{}", spec.name());
        match (c, r) {
            (CLayer::Linear(c), RustLayer::Linear(r)) => {
                assert_eq!((c.nb_inputs as usize, c.nb_outputs as usize), (r.nb_inputs, r.nb_outputs), "{name}");
                let n = r.nb_outputs;
                for (what, len) in [("bias", r.bias.map(|b| b.len())), ("subias", r.subias.map(|b| b.len()))] {
                    assert!(len.is_none_or(|l| l == n), "{name}.{what}");
                }
                same_array(&format!("{name}.bias"), r.bias, c.bias);
                same_array(&format!("{name}.subias"), r.subias, c.subias);
                same_array(&format!("{name}.weights"), r.weights, c.weights);
                same_array(&format!("{name}.float_weights"), r.float_weights, c.float_weights);
                same_array(&format!("{name}.diag"), r.diag, c.diag);
                same_array(&format!("{name}.scale"), r.scale, c.scale);
                if let (Some(idx), false) = (r.weights_idx, c.weights_idx.is_null()) {
                    assert_eq!(c_idx_len(c.weights_idx, n), idx.len(), "{name}.weights_idx length");
                }
                same_array(&format!("{name}.weights_idx"), r.weights_idx, c.weights_idx);
            }
            (CLayer::Conv2d(c), RustLayer::Conv2d(r)) => {
                assert_eq!(
                    (c.in_channels as usize, c.out_channels as usize, c.ktime as usize, c.kheight as usize),
                    (r.in_channels, r.out_channels, r.ktime, r.kheight),
                    "{name}"
                );
                same_array(&format!("{name}.bias"), r.bias, c.bias);
                same_array(&format!("{name}.float_weights"), r.float_weights, c.float_weights);
            }
            _ => panic!("{name}: the layer kinds differ"),
        }
    }
}

/// A list parsed by libopus's `parse_weights`, freed on drop.
struct CList(*mut CWeightArray);

impl CList {
    /// `None` when `parse_weights` refuses the blob.
    fn parse(blob: &[u8]) -> (c_int, Option<CList>) {
        let mut list: *mut CWeightArray = std::ptr::null_mut();
        // SAFETY: `parse_weights` reads at most `len` bytes of `blob`.
        let ret = unsafe { parse_weights(&mut list, blob.as_ptr().cast(), blob.len() as c_int) };
        if ret < 0 {
            assert!(list.is_null());
            (ret, None)
        } else {
            (ret, Some(CList(list)))
        }
    }
}

impl Drop for CList {
    fn drop(&mut self) {
        // SAFETY: allocated by `parse_weights` with `malloc`.
        unsafe { free(self.0.cast()) };
    }
}

// ---------------------------------------------------------------------------
// The blobs
// ---------------------------------------------------------------------------

#[test]
fn opus1_blobs_hold_the_arrays_of_the_c_lists() {
    linked();
    let mut checked = 0;
    for model in &MODELS {
        let Some(blob) = builtin_blob(model.key) else { continue };
        let list = WeightList::parse(blob).unwrap();
        let ours: Vec<_> = list.iter().collect();
        let theirs = c_entries((model.arrays)());
        assert_eq!(ours.len(), theirs.len(), "{}", model.key);
        for (a, e) in ours.iter().zip(&theirs) {
            assert_eq!(a.name, e.name.as_slice(), "{}", model.key);
            assert_eq!(a.kind, e.kind, "{}", String::from_utf8_lossy(&e.name));
            assert!(a.data == e.data, "{}: the bytes differ", String::from_utf8_lossy(&e.name));
        }
        assert_eq!(c_blob(model), blob, "{}", model.key);
        checked += 1;
    }
    assert_eq!(checked, builtin::blobs().len());
}

/// The folder of the reference blob files, when it exists.
fn blob_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        std::env::var_os("OPUS1_DNN_BLOB_DIR").map(PathBuf::from),
        Some(manifest.join("target/android/dnn-blob-1.6.1")),
        // A worktree under target/wt/ of the repository.
        Some(manifest.join("../../android/dnn-blob-1.6.1")),
    ];
    candidates.into_iter().flatten().find(|p| p.join("weights_blob.bin").is_file())
}

#[test]
fn concatenated_blobs_equal_the_blob_of_libopus() {
    linked();
    // dump_weights_blob writes the first 7 models in the order of MODELS;
    // the complete blob adds BBWENet.
    let official: Vec<u8> = MODELS[..7].iter().flat_map(c_blob).collect();
    let mut complete = official.clone();
    complete.extend(c_blob(&MODELS[7]));
    assert_eq!(official.len(), 3_571_648);
    assert_eq!(complete.len(), 4_257_536);
    match blob_dir() {
        Some(dir) => {
            let file = std::fs::read(dir.join("weights_blob.bin")).unwrap();
            assert!(file == official, "weights_blob.bin differs from the C lists");
            if let Ok(file) = std::fs::read(dir.join("weights_blob_all.bin")) {
                assert!(file == complete, "weights_blob_all.bin differs from the C lists");
            }
        }
        None => eprintln!("no reference blob folder (set OPUS1_DNN_BLOB_DIR): compared with the C lists only"),
    }
    // opus1's blobs, where the writer covers the same models.
    let cat = |keys: &[&str]| -> Option<Vec<u8>> {
        let mut v = Vec::new();
        for k in keys {
            v.extend_from_slice(builtin_blob(k)?);
        }
        Some(v)
    };
    if let Some(plc) = cat(&["pitchdnn", "fargan", "plc"]) {
        assert!(official[..plc.len()] == plc[..], "deep PLC blobs");
        if let Some(dred) = cat(&["dred_rdovae_enc", "dred_rdovae_dec"]) {
            assert!(official[plc.len()..plc.len() + dred.len()] == dred[..], "DRED blobs");
        }
    }
    if let Some(osce) = cat(&["lace", "nolace"]) {
        assert!(official[official.len() - osce.len()..] == osce[..], "LACE and NoLACE blobs");
    }
    if let Some(bwe) = cat(&["bbwenet"]) {
        assert!(complete[official.len()..] == bwe[..], "BBWENet blob");
    }
}

// ---------------------------------------------------------------------------
// The layers
// ---------------------------------------------------------------------------

#[test]
fn every_model_builds_the_layers_of_libopus() {
    linked();
    let mut checked = 0;
    for model in &MODELS {
        let Some((specs, built)) = rust_model(model.key, builtin::weights()) else { continue };
        assert_eq!(specs, model.layers, "{}: the descriptors of src/ and tests/ differ", model.key);
        let rust = built.unwrap_or_else(|e| panic!("{}: {e}", model.key));
        // libopus with its built-in arrays.
        let (ret, c) = (model.init)((model.arrays)());
        assert_eq!(ret, 0, "{}", model.key);
        compare_layers(model.key, specs, &c, &rust);
        // libopus with opus1's blob, read by its own parser.
        let blob = builtin_blob(model.key).unwrap();
        let (n, list) = CList::parse(blob);
        assert_eq!(n as usize, WeightList::parse(blob).unwrap().len());
        let list = list.unwrap();
        let (ret, c) = (model.init)(list.0);
        assert_eq!(ret, 0, "{}", model.key);
        compare_layers(model.key, specs, &c, &rust);
        checked += 1;
    }
    assert_eq!(checked, builtin::blobs().len());
}

#[test]
fn dred_layers_load_like_libopus() {
    linked();
    for model in MODELS.iter().filter(|m| m.feature == "dred") {
        let blob = Aligned::new(&c_blob(model));
        let list = WeightList::parse(blob.bytes()).unwrap();
        let rust = rust_layers(model.layers, &list).unwrap_or_else(|e| panic!("{}: {e}", model.key));
        let (ret, c) = (model.init)((model.arrays)());
        assert_eq!(ret, 0);
        compare_layers(model.key, model.layers, &c, &rust);
        let sparse = model
            .layers
            .iter()
            .filter(|s| matches!(s, LayerSpec::Linear(l) if l.weights_idx.is_some()))
            .count();
        assert!(sparse > 0, "{} has sparse layers", model.key);
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// Whether libopus and opus1 accept `blob` alike, with the same number of
/// arrays.
fn parse_alike(blob: &[u8], what: &str) {
    let buf = Aligned::new(blob);
    let (n, _list) = CList::parse(buf.bytes());
    let ours = WeightList::parse(buf.bytes()).map(|l| l.len());
    match ours {
        Ok(len) => assert_eq!(n as usize, len, "{what}: libopus refuses or counts otherwise"),
        Err(e) => assert!(n < 0, "{what}: opus1 refuses ({e}), libopus accepts"),
    }
}

#[test]
fn parse_weights_accepts_and_refuses_alike() {
    linked();
    let model = &MODELS[0];
    let base = c_blob(model);
    let starts: Vec<usize> = {
        let buf = Aligned::new(&base);
        let list = WeightList::parse(buf.bytes()).unwrap();
        list.iter().map(|a| a.data.as_ptr() as usize - buf.bytes().as_ptr() as usize - 64).collect()
    };
    parse_alike(&base, "unchanged");
    parse_alike(&[], "empty");
    // Truncated or extended.
    for &s in starts.iter().take(6) {
        for cut in [s, s + 1, s + 63, s + 64, s + 65, s + 100] {
            parse_alike(&base[..cut.min(base.len())], &format!("cut at {cut}"));
        }
    }
    for extra in [1usize, 4, 63, 64, 65, 128] {
        let mut v = base.clone();
        v.resize(v.len() + extra, 0);
        parse_alike(&v, &format!("{extra} zero bytes appended"));
        let mut v = base.clone();
        v.extend(std::iter::repeat_n(0xff, extra));
        parse_alike(&v, &format!("{extra} bytes 0xff appended"));
    }
    // Header fields of the first, a middle and the last record. block_size
    // values stay multiples of 4, where both sides agree on alignment.
    let picks = [starts[0], starts[starts.len() / 2], *starts.last().unwrap()];
    let field = |v: &[u8], at: usize| i32::from_le_bytes(v[at..at + 4].try_into().unwrap());
    for &s in &picks {
        let size = field(&base, s + 12);
        let block = field(&base, s + 16);
        let rest = (base.len() - s - 64) as i32;
        let edits: Vec<(usize, i32)> = vec![
            (0, 0),
            (4, 1),
            (8, 7),
            (8, -1),
            (12, 0),
            (12, -1),
            (12, i32::MIN),
            (12, block),
            (12, block + 1),
            (12, size - 1),
            (16, size / 4 * 4),
            (16, block - 64),
            (16, block + 64),
            (16, rest),
            (16, rest + 4),
            (16, -4),
            (16, i32::MIN),
            (16, i32::MAX / 4 * 4),
        ];
        for (at, value) in edits {
            let mut v = base.clone();
            v[s + at..s + at + 4].copy_from_slice(&value.to_le_bytes());
            parse_alike(&v, &format!("record at {s}: field {at} = {value}"));
        }
        for (at, byte) in [(20usize, 0u8), (62, b'x'), (63, b'x'), (63, 0)] {
            let mut v = base.clone();
            v[s + at] = byte;
            parse_alike(&v, &format!("record at {s}: name byte {at} = {byte}"));
        }
    }
}

/// `linear_init` of libopus and of opus1 on the layer `name` of the RDOVAE
/// encoder, from a blob whose index table of that layer is `idx`: whether
/// each accepts it.
fn sparse_alike(idx: &[i32]) -> (bool, Result<(), Error>) {
    let model = MODELS.iter().find(|m| m.key == "dred_rdovae_enc").unwrap();
    let LayerSpec::Linear(spec) = *model.layers.iter().find(|s| s.name() == "gdense1").unwrap() else { unreachable!() };
    let mut blob = Vec::new();
    for e in c_entries((model.arrays)()) {
        let ints: Vec<u8>;
        let data = if e.name == b"gdense1_weights_idx" {
            ints = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
            &ints[..]
        } else {
            e.data
        };
        let block = data.len().div_ceil(64) * 64;
        blob.extend_from_slice(b"DNNw");
        blob.extend_from_slice(&0i32.to_le_bytes());
        blob.extend_from_slice(&e.kind.to_le_bytes());
        blob.extend_from_slice(&(data.len() as i32).to_le_bytes());
        blob.extend_from_slice(&(block as i32).to_le_bytes());
        let mut name = [0u8; 44];
        name[..e.name.len()].copy_from_slice(&e.name);
        blob.extend_from_slice(&name);
        blob.extend_from_slice(data);
        blob.resize(blob.len() + block - data.len(), 0);
    }
    let buf = Aligned::new(&blob);
    let list = WeightList::parse(buf.bytes()).unwrap();
    let ours = linear_init(
        &list,
        spec.bias,
        spec.subias,
        spec.weights,
        spec.float_weights,
        spec.weights_idx,
        spec.diag,
        spec.scale,
        spec.nb_inputs,
        spec.nb_outputs,
    )
    .map(|_| ());
    let (_, clist) = CList::parse(buf.bytes());
    let clist = clist.unwrap();
    let c = |s: Option<&str>| s.map(|s| CString::new(s).unwrap());
    let names = [spec.bias, spec.subias, spec.weights, spec.float_weights, spec.weights_idx, spec.diag, spec.scale].map(c);
    let p = |i: usize| names[i].as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    // SAFETY: zeroed is a valid layer; the C function reads the list, whose
    // index table here holds only counts that fit in the table, so it stays
    // in bounds.
    let mut layer: CLinearLayer = unsafe { std::mem::zeroed() };
    let ret = unsafe {
        c_linear_init(
            &mut layer,
            clist.0,
            p(0),
            p(1),
            p(2),
            p(3),
            p(4),
            p(5),
            p(6),
            spec.nb_inputs as c_int,
            spec.nb_outputs as c_int,
        )
    };
    (ret == 0, ours)
}

#[test]
fn sparse_index_tables_are_refused_like_libopus() {
    linked();
    let model = MODELS.iter().find(|m| m.key == "dred_rdovae_enc").unwrap();
    let good: Vec<i32> = {
        let e = c_entries((model.arrays)()).into_iter().find(|e| e.name == b"gdense1_weights_idx").unwrap();
        e.data.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect()
    };
    assert_eq!(sparse_alike(&good), (true, Ok(())));
    // The first row block: its count, then its first position.
    let count = good[0] as usize;
    assert!(count >= 2);
    let changed = |at: usize, v: i32| {
        let mut t = good.clone();
        t[at] = v;
        t
    };
    // Where each row block starts in the table.
    let mut starts = Vec::new();
    let mut p = 0;
    while p < good.len() {
        starts.push(p);
        p += 1 + good[p] as usize;
    }
    let last = *starts.last().unwrap();
    // Both refuse: a position that is not a multiple of 4, one past the
    // inputs, a table that misses its last row block or has one more, a
    // count past the end of the table.
    for (what, t) in [
        ("odd position", changed(1, 2)),
        ("position past the inputs", changed(1, 544)),
        ("last position past the inputs", changed(count, 1024)),
        ("one row block missing", good[..last].to_vec()),
        ("one row block more", [&good[..], &[0]].concat()),
        ("count past the end", good[..good.len() - 1].to_vec()),
    ] {
        let (c, ours) = sparse_alike(&t);
        assert!(!c, "{what}: libopus accepts");
        assert!(ours.is_err(), "{what}: opus1 accepts");
    }
    // libopus accepts a negative position, and its kernel then reads before
    // its input buffer; opus1 refuses it.
    let (c, ours) = sparse_alike(&changed(1, -4));
    assert!(c, "libopus refuses a negative position");
    assert_eq!(ours, Err(ERR_SPARSE_INDEX));
}
