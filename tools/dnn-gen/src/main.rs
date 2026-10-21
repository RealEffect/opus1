//! dnn-gen: writes the weights of the libopus neural network models in
//! libopus's blob format, and the matching Rust layer descriptors of opus1.
//!
//! It reads the data files of a libopus release (`dnn/*_data.c` and
//! `dnn/*_data.h`) and writes, under the opus1 repository:
//!
//! - `dnn-weights/<model>.bin`: one blob per model, the records that
//!   libopus's `dump_weights_blob` (`dnn/write_lpcnet_weights.c`) writes for
//!   that model;
//! - `dnn-weights/SHA256SUMS`, `dnn-weights/VERSION` and a copy of
//!   libopus's `COPYING` as `dnn-weights/COPYING`;
//! - `src/dnn/models/<model>.rs`: the layer struct, the header constants,
//!   the layer descriptors and `init`, translated from the C init function;
//! - `src/dnn/models/dred_rdovae_stats.rs`, with the DRED models: the
//!   constants of `dnn/dred_rdovae_constants.h` and the quantizer tables of
//!   `dnn/dred_rdovae_stats_data.c`;
//! - `tests/dnn_gen/mod.rs`: `#[repr(C)]` copies of the C model structs and
//!   the C symbols, for the tests against a DNN-enabled libopus.
//!
//! See README.md for the options.

mod blob;
mod codegen;
mod cparse;
mod sha256;

use cparse::{CType, DataFile, HeaderFile, InitCall};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// One model of libopus.
pub struct ModelDef {
    /// The name of the data files without `_data.c` and `_data.h`; also the
    /// name of the blob and of the Rust module.
    pub key: &'static str,
    /// The `WeightArray` list of the data file.
    pub list: &'static str,
    /// The init function of the data file.
    pub init: &'static str,
    /// The C struct of the layers.
    pub c_struct: &'static str,
    /// The Rust struct of the layers.
    pub rust_struct: &'static str,
    /// The opus1 feature that includes the model.
    pub feature: &'static str,
    /// The first line of the module documentation.
    pub summary: &'static str,
}

/// Every model with a `WeightArray` list in libopus 1.6.1, in the order of
/// `dump_weights_blob` (`dnn/write_lpcnet_weights.c:79-97`), then BBWENet,
/// which that program does not write. `lossgen_data.c` (a packet loss
/// generator for tests) is not part of the codec and is left out.
pub const MODELS: [ModelDef; 8] = [
    ModelDef {
        key: "pitchdnn",
        list: "pitchdnn_arrays",
        init: "init_pitchdnn",
        c_struct: "PitchDNN",
        rust_struct: "PitchDnn",
        feature: "deep-plc",
        summary: "PitchDNN, the pitch estimator of deep packet loss concealment (deep PLC)",
    },
    ModelDef {
        key: "fargan",
        list: "fargan_arrays",
        init: "init_fargan",
        c_struct: "FARGAN",
        rust_struct: "Fargan",
        feature: "deep-plc",
        summary: "FARGAN, the vocoder of deep packet loss concealment (deep PLC)",
    },
    ModelDef {
        key: "plc",
        list: "plcmodel_arrays",
        init: "init_plcmodel",
        c_struct: "PLCModel",
        rust_struct: "PlcModel",
        feature: "deep-plc",
        summary: "The PLC model, which predicts the features of lost frames in deep packet loss concealment (deep PLC)",
    },
    ModelDef {
        key: "dred_rdovae_enc",
        list: "rdovaeenc_arrays",
        init: "init_rdovaeenc",
        c_struct: "RDOVAEEnc",
        rust_struct: "RdovaeEnc",
        feature: "dred",
        summary: "The RDOVAE encoder of deep redundancy (DRED)",
    },
    ModelDef {
        key: "dred_rdovae_dec",
        list: "rdovaedec_arrays",
        init: "init_rdovaedec",
        c_struct: "RDOVAEDec",
        rust_struct: "RdovaeDec",
        feature: "dred",
        summary: "The RDOVAE decoder of deep redundancy (DRED)",
    },
    ModelDef {
        key: "lace",
        list: "lacelayers_arrays",
        init: "init_lacelayers",
        c_struct: "LACELayers",
        rust_struct: "LaceLayers",
        feature: "osce",
        summary: "LACE, the speech enhancer of SILK at decoder complexity 6",
    },
    ModelDef {
        key: "nolace",
        list: "nolacelayers_arrays",
        init: "init_nolacelayers",
        c_struct: "NOLACELayers",
        rust_struct: "NolaceLayers",
        feature: "osce",
        summary: "NoLACE, the speech enhancer of SILK at decoder complexity 7 and above",
    },
    ModelDef {
        key: "bbwenet",
        list: "bbwenetlayers_arrays",
        init: "init_bbwenetlayers",
        c_struct: "BBWENETLayers",
        rust_struct: "BbwenetLayers",
        feature: "osce-bwe",
        summary: "BBWENet, the network of the bandwidth extension (BWE) of SILK wideband to 48 kHz",
    },
];

/// How many models `dump_weights_blob` writes: the first 7 of `MODELS`.
const OFFICIAL_MODELS: usize = 7;

/// The release this generator was written for, and the SHA-256 of its
/// tarball `opus-1.6.1.tar.gz`.
const DEFAULT_VERSION: &str = "1.6.1";
const DEFAULT_TARBALL_SHA256: &str = "6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1";

/// A model read from its data files.
pub struct Model {
    pub def: &'static ModelDef,
    pub data: DataFile,
    pub header: HeaderFile,
    /// The model's blob.
    pub blob: Vec<u8>,
    /// SHA-256 of the two source files, for `VERSION`.
    pub source_sums: [(String, String); 2],
}

impl Model {
    /// Whether the list of a default build has an array of this name.
    pub fn has_array(&self, name: &str) -> bool {
        self.data.list.iter().any(|e| e.name == name)
    }

    fn array_size(&self, name: &str) -> Option<usize> {
        let e = self.data.list.iter().find(|e| e.name == name)?;
        Some(self.data.arrays[&e.array].data.len())
    }

    /// The bytes of each kind of array in the blob: (int8, float, int).
    fn bytes_by_type(&self) -> (usize, usize, usize) {
        let mut t = (0, 0, 0);
        for e in &self.data.list {
            let a = &self.data.arrays[&e.array];
            match a.ctype {
                CType::Int8 => t.0 += a.data.len(),
                CType::Float => t.1 += a.data.len(),
                CType::Int => t.2 += a.data.len(),
            }
        }
        t
    }
}

/// The constants and quantizer tables of DRED's RDOVAE, which are not part
/// of a model's data files.
pub struct DredStats {
    pub consts: Vec<cparse::HeaderConst>,
    pub tables: Vec<cparse::U8Table>,
    /// SHA-256 of the two source files, for `VERSION`.
    pub source_sums: [(String, String); 2],
}

/// The tables of `dnn/dred_rdovae_stats_data.h`, with their sizes:
/// `DRED_NUM_QUANTIZATION_LEVELS` times `DRED_LATENT_DIM` or `DRED_STATE_DIM`.
const DRED_STATS_TABLES: [(&str, usize); 8] = [
    ("dred_latent_quant_scales_q8", 400),
    ("dred_latent_dead_zone_q8", 400),
    ("dred_latent_r_q8", 400),
    ("dred_latent_p0_q8", 400),
    ("dred_state_quant_scales_q8", 800),
    ("dred_state_dead_zone_q8", 800),
    ("dred_state_r_q8", 800),
    ("dred_state_p0_q8", 800),
];

fn read_dred_stats(src: &Path) -> Result<DredStats, String> {
    let read = |name: &str| {
        let path = src.join(name);
        fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
    };
    let h_name = "dnn/dred_rdovae_constants.h";
    let c_name = "dnn/dred_rdovae_stats_data.c";
    let h_text = read(h_name)?;
    let c_text = read(c_name)?;
    let consts = cparse::parse_defines(h_name, &h_text)?;
    let tables = cparse::parse_u8_tables(c_name, &c_text)?;
    let names: Vec<(&str, usize)> = tables.iter().map(|t| (t.name.as_str(), t.data.len())).collect();
    if names != DRED_STATS_TABLES {
        return Err(format!("{c_name}: the tables {names:?} are not those of libopus 1.6.1 {DRED_STATS_TABLES:?}"));
    }
    let int = |name: &str| {
        consts.iter().find(|c| c.name == name).and_then(|c| match c.value {
            cparse::ConstValue::Int { value, .. } => Some(value as usize),
            _ => None,
        })
    };
    let (levels, latent, state) = (int("DRED_NUM_QUANTIZATION_LEVELS"), int("DRED_LATENT_DIM"), int("DRED_STATE_DIM"));
    if levels.zip(latent).map(|(l, d)| l * d) != Some(400) || levels.zip(state).map(|(l, d)| l * d) != Some(800) {
        return Err(format!("{h_name}: the dimensions do not match the sizes of the tables of {c_name}"));
    }
    let source_sums = [
        (sha256::hex(h_text.as_bytes()), h_name.to_string()),
        (sha256::hex(c_text.as_bytes()), c_name.to_string()),
    ];
    Ok(DredStats { consts, tables, source_sums })
}

fn read_model(src: &Path, def: &'static ModelDef) -> Result<Model, String> {
    let dnn = src.join("dnn");
    let c_path = dnn.join(format!("{}_data.c", def.key));
    let h_path = dnn.join(format!("{}_data.h", def.key));
    let c_text = fs::read_to_string(&c_path).map_err(|e| format!("{}: {e}", c_path.display()))?;
    let h_text = fs::read_to_string(&h_path).map_err(|e| format!("{}: {e}", h_path.display()))?;
    let c_name = format!("dnn/{}_data.c", def.key);
    let h_name = format!("dnn/{}_data.h", def.key);
    let data = cparse::parse_data_file(&c_name, &c_text, def.list, def.init)?;
    let header = cparse::parse_header(&h_name, &h_text, def.c_struct)?;
    let mut blob = Vec::new();
    for e in &data.list {
        blob::write_record(&mut blob, &e.name, e.kind, &data.arrays[&e.array].data)?;
    }
    let source_sums = [
        (sha256::hex(c_text.as_bytes()), c_name),
        (sha256::hex(h_text.as_bytes()), h_name),
    ];
    let m = Model { def, data, header, blob, source_sums };
    check_model(&m)?;
    Ok(m)
}

/// Checks what the Rust loader will rely on, so that a change in a later
/// libopus stops the generator instead of giving files that do not load.
fn check_model(m: &Model) -> Result<(), String> {
    let key = m.def.key;
    if m.data.init_struct != m.def.c_struct {
        return Err(format!("{key}: the init function fills `{}`, not `{}`", m.data.init_struct, m.def.c_struct));
    }
    // The struct and the init function have the same fields, of the same kinds.
    if m.header.fields.len() != m.data.init_calls.len() {
        return Err(format!("{key}: the struct has {} fields, the init function {} calls", m.header.fields.len(), m.data.init_calls.len()));
    }
    for (field, kind) in &m.header.fields {
        let call = m
            .data
            .init_calls
            .iter()
            .find(|c| c.field() == field)
            .ok_or_else(|| format!("{key}: no init call for the field `{field}`"))?;
        let ok = matches!(
            (kind, call),
            (cparse::FieldKind::Linear, InitCall::Linear { .. }) | (cparse::FieldKind::Conv2d, InitCall::Conv2d { .. })
        );
        if !ok {
            return Err(format!("{key}: field `{field}` and its init call differ in kind"));
        }
    }
    for c in &m.header.consts {
        if c.name == "LAYERS" || c.name == "BLOB_FILE" {
            return Err(format!("{key}: the header constant `{}` clashes with a generated name", c.name));
        }
    }
    for e in &m.data.list {
        let a = &m.data.arrays[&e.array];
        if a.minus_128 > 0 {
            return Err(format!("{key}: `{}` holds {} int8 weights equal to -128, which opus1 refuses", e.name, a.minus_128));
        }
    }
    // The blob reads back as the list, record by record.
    let records = blob::read(&m.blob).map_err(|e| format!("{key}: {e}"))?;
    let same = records.len() == m.data.list.len()
        && records.iter().zip(&m.data.list).all(|(r, e)| {
            r.name == e.name.as_bytes() && r.kind == e.kind && r.data == m.data.arrays[&e.array].data.as_slice()
        });
    if !same {
        return Err(format!("{key}: the blob does not read back as the list"));
    }
    // Every init call succeeds on the model's own list, with the sizes that
    // linear_init and conv2d_init require.
    for call in &m.data.init_calls {
        check_call(m, call).map_err(|e| format!("{key}: layer `{}`: {e}", call.field()))?;
    }
    Ok(())
}

fn check_call(m: &Model, call: &InitCall) -> Result<(), String> {
    let need = |name: &Option<String>, size: usize| -> Result<(), String> {
        if let Some(n) = name {
            match m.array_size(n) {
                Some(s) if s == size => Ok(()),
                Some(s) => Err(format!("`{n}` has {s} bytes, not {size}")),
                None => Err(format!("`{n}` is missing")),
            }
        } else {
            Ok(())
        }
    };
    let optional = |name: &Option<String>, size: usize| -> Result<bool, String> {
        match name.as_ref().and_then(|n| m.array_size(n).map(|s| (n, s))) {
            Some((_, s)) if s == size => Ok(true),
            Some((n, s)) => Err(format!("`{n}` has {s} bytes, not {size}")),
            None => Ok(false),
        }
    };
    match call {
        InitCall::Linear { bias, subias, weights, float_weights, weights_idx, diag, scale, nb_inputs, nb_outputs, .. } => {
            need(bias, 4 * nb_outputs)?;
            need(subias, 4 * nb_outputs)?;
            let values = if let Some(idx) = weights_idx {
                let e = m.data.list.iter().find(|e| &e.name == idx).ok_or_else(|| format!("`{idx}` is missing"))?;
                let a = &m.data.arrays[&e.array];
                if a.ctype != CType::Int {
                    return Err(format!("`{idx}` is not an int array"));
                }
                let ints: Vec<i32> = a.data.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
                let (mut p, mut rows, mut blocks) = (0usize, 0usize, 0usize);
                while p < ints.len() {
                    let n = usize::try_from(ints[p]).map_err(|_| "negative block count")?;
                    for &pos in ints.get(p + 1..p + 1 + n).ok_or("index table too short")? {
                        let pos = usize::try_from(pos).map_err(|_| "negative position")?;
                        if pos % 4 != 0 || pos + 3 >= *nb_inputs {
                            return Err(format!("bad position {pos}"));
                        }
                    }
                    p += 1 + n;
                    rows += 8;
                    blocks += n;
                }
                if rows != *nb_outputs {
                    return Err(format!("the index table covers {rows} rows, not {nb_outputs}"));
                }
                32 * blocks
            } else {
                nb_inputs * nb_outputs
            };
            need(weights, values)?;
            let has_float = optional(float_weights, 4 * values)?;
            need(diag, 4 * nb_outputs)?;
            if weights.is_some() {
                if scale.is_none() {
                    return Err("int8 weights without a scale".into());
                }
                need(scale, 4 * nb_outputs)?;
                if nb_outputs % 8 != 0 || nb_inputs % 4 != 0 {
                    return Err("int8 weights need a multiple of 8 outputs and of 4 inputs".into());
                }
            }
            if weights.is_none() && !has_float {
                return Err("no weights".into());
            }
        }
        InitCall::Conv2d { bias, float_weights, in_channels, out_channels, ktime, kheight, .. } => {
            need(bias, 4 * out_channels)?;
            if !optional(float_weights, 4 * in_channels * out_channels * ktime * kheight)? {
                return Err("no weights".into());
            }
        }
    }
    Ok(())
}

struct Args {
    src: PathBuf,
    repo: PathBuf,
    models: Vec<&'static ModelDef>,
    blobs_only: Option<PathBuf>,
    check_official: Option<PathBuf>,
    check_all: Option<PathBuf>,
    version: String,
    tarball_sha256: Option<String>,
}

const USAGE: &str = "\
usage: dnn-gen --src <libopus source dir> [options]

Writes dnn-weights/, src/dnn/models/<model>.rs and tests/dnn_gen/mod.rs
under the opus1 repository.

options:
  --src DIR               the libopus source tree (the folder with dnn/)
  --repo DIR              the opus1 repository (default: two levels above
                          this tool's Cargo.toml)
  --models a,b,...        the models to write (default: all eight,
                          pitchdnn,fargan,plc,dred_rdovae_enc,
                          dred_rdovae_dec,lace,nolace,bbwenet)
  --blobs-only DIR        write only the blobs of --models into DIR
  --check-official FILE   compare FILE with the blobs of the 7 models that
                          dump_weights_blob writes, concatenated in its order
  --check-all FILE        the same with BBWENet appended
  --version V             the libopus version that --src must hold (default 1.6.1)
  --tarball-sha256 HEX    the SHA-256 of the release tarball, for VERSION
                          (known for 1.6.1)
";

fn parse_args() -> Result<Args, String> {
    let mut src = None;
    let mut repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut models: Vec<&'static ModelDef> = MODELS.iter().collect();
    let mut blobs_only = None;
    let mut check_official = None;
    let mut check_all = None;
    let mut version = DEFAULT_VERSION.to_string();
    let mut tarball_sha256 = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--src" => src = Some(PathBuf::from(value()?)),
            "--repo" => repo = PathBuf::from(value()?),
            "--models" => {
                models = Vec::new();
                for k in value()?.split(',') {
                    let def = MODELS.iter().find(|m| m.key == k).ok_or_else(|| format!("unknown model `{k}`"))?;
                    models.push(def);
                }
            }
            "--blobs-only" => blobs_only = Some(PathBuf::from(value()?)),
            "--check-official" => check_official = Some(PathBuf::from(value()?)),
            "--check-all" => check_all = Some(PathBuf::from(value()?)),
            "--version" => version = value()?,
            "--tarball-sha256" => tarball_sha256 = Some(value()?),
            "-h" | "--help" => return Err(String::new()),
            _ => return Err(format!("unknown argument `{a}`")),
        }
    }
    let src = src.ok_or("--src is required")?;
    let repo = fs::canonicalize(&repo).map_err(|e| format!("{}: {e}", repo.display()))?;
    Ok(Args { src, repo, models, blobs_only, check_official, check_all, version, tarball_sha256 })
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

fn check_concat(file: &Path, models: &[Model], what: &str) -> Result<(), String> {
    let want = fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let got: Vec<u8> = models.iter().flat_map(|m| m.blob.iter().copied()).collect();
    if got == want {
        println!("{what}: {} equals the generated blobs concatenated ({} bytes)", file.display(), got.len());
        Ok(())
    } else {
        let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
        Err(format!(
            "{what}: {} differs from the generated blobs: {} bytes against {}, first difference at byte {first}",
            file.display(),
            want.len(),
            got.len()
        ))
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let pv_path = args.src.join("package_version");
    let pv = fs::read_to_string(&pv_path).map_err(|e| format!("{}: {e}", pv_path.display()))?;
    let found = pv.trim().strip_prefix("PACKAGE_VERSION=").map(|v| v.trim_matches('"')).unwrap_or("");
    if found != args.version {
        return Err(format!("{} holds libopus `{found}`, not `{}` (see --version)", args.src.display(), args.version));
    }
    let tarball = match (&args.tarball_sha256, args.version.as_str()) {
        (Some(s), _) => s.clone(),
        (None, DEFAULT_VERSION) => DEFAULT_TARBALL_SHA256.to_string(),
        (None, _) => return Err("give --tarball-sha256 for another version than 1.6.1".into()),
    };

    let mut all = Vec::new();
    for def in &MODELS {
        all.push(read_model(&args.src, def)?);
    }
    // The constants and tables of DRED, written with the DRED models.
    let dred_stats = if args.models.iter().any(|m| m.feature == "dred") { Some(read_dred_stats(&args.src)?) } else { None };
    let selected: Vec<&Model> = args.models.iter().map(|d| all.iter().find(|m| m.def.key == d.key).unwrap()).collect();

    if let Some(f) = &args.check_official {
        check_concat(f, &all[..OFFICIAL_MODELS], "official blob")?;
    }
    if let Some(f) = &args.check_all {
        check_concat(f, &all, "complete blob")?;
    }

    if let Some(dir) = &args.blobs_only {
        for m in &selected {
            write(&dir.join(format!("{}.bin", m.def.key)), &m.blob)?;
        }
    } else {
        let weights = args.repo.join("dnn-weights");
        let mut sums = String::new();
        let mut version = format!(
            "libopus {}\nopus-{}.tar.gz SHA-256 {tarball}\n\nThe blobs were written by tools/dnn-gen from these files of that release:\n\n",
            args.version, args.version
        );
        for m in &selected {
            let name = format!("{}.bin", m.def.key);
            write(&weights.join(&name), &m.blob)?;
            sums.push_str(&format!("{}  {name}\n", sha256::hex(&m.blob)));
            for (sum, file) in &m.source_sums {
                version.push_str(&format!("{sum}  {file}\n"));
            }
        }
        if let Some(st) = &dred_stats {
            for (sum, file) in &st.source_sums {
                version.push_str(&format!("{sum}  {file}\n"));
            }
        }
        write(&weights.join("SHA256SUMS"), sums.as_bytes())?;
        write(&weights.join("VERSION"), version.as_bytes())?;
        let copying = fs::read(args.src.join("COPYING")).map_err(|e| format!("COPYING: {e}"))?;
        write(&weights.join("COPYING"), &copying)?;
        for m in &selected {
            let path = args.repo.join("src/dnn/models").join(format!("{}.rs", m.def.key));
            write(&path, codegen::model_file(m, &args.version).as_bytes())?;
        }
        if let Some(st) = &dred_stats {
            let path = args.repo.join("src/dnn/models/dred_rdovae_stats.rs");
            write(&path, codegen::dred_stats_file(&st.consts, &st.tables, &args.version).as_bytes())?;
        }
        write(&args.repo.join("tests/dnn_gen/mod.rs"), codegen::ffi_file(&all, &args.version).as_bytes())?;
    }

    // Sizes, per model and per feature.
    println!();
    println!("{:<16} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10}", "model", "layers", "arrays", "int8", "float", "int", "blob");
    let mut features: Vec<(&str, usize)> = Vec::new();
    for m in &all {
        let (i8b, fb, ib) = m.bytes_by_type();
        println!(
            "{:<16} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10}",
            m.def.key,
            m.data.init_calls.len(),
            m.data.list.len(),
            i8b,
            fb,
            ib,
            m.blob.len()
        );
        match features.iter_mut().find(|f| f.0 == m.def.feature) {
            Some(f) => f.1 += m.blob.len(),
            None => features.push((m.def.feature, m.blob.len())),
        }
    }
    println!();
    for (f, n) in &features {
        println!("feature {f:<10} {n:>10} blob bytes (its own models only)");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.is_empty() => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("dnn-gen: {e}");
            eprint!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}
