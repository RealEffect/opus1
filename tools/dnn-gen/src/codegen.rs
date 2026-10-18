//! Writes the Rust files: one layer file per model for `src/dnn/models/`,
//! and the C interface file of the tests (`tests/dnn_gen/mod.rs`).

use crate::cparse::{ConstValue, FieldKind, InitCall};
use crate::{Model, ModelDef};
use std::fmt::Write as _;

/// The words that describe the weights a layer uses in a default build.
fn linear_kind(m: &Model, call: &InitCall) -> String {
    let InitCall::Linear { weights, float_weights, weights_idx, .. } = call else {
        return "conv2d, float weights".into();
    };
    let has = |n: &Option<String>| n.as_ref().is_some_and(|n| m.has_array(n));
    let sparse = if weights_idx.is_some() { " sparse" } else { "" };
    if has(float_weights) {
        format!("float{sparse}")
    } else if has(weights) {
        format!("int8{sparse}")
    } else {
        "no".into()
    }
}

fn opt(s: &Option<String>) -> String {
    match s {
        Some(s) => format!("Some(\"{s}\")"),
        None => "None".into(),
    }
}

/// The `LayerSpec` value of one init call, indented by `ind` spaces.
fn layer_spec(call: &InitCall, ind: usize) -> String {
    let p = " ".repeat(ind);
    let mut s = String::new();
    match call {
        InitCall::Linear {
            field,
            bias,
            subias,
            weights,
            float_weights,
            weights_idx,
            diag,
            scale,
            nb_inputs,
            nb_outputs,
        } => {
            let _ = writeln!(s, "{p}LayerSpec::Linear(LinearSpec {{");
            let _ = writeln!(s, "{p}    name: \"{field}\",");
            let _ = writeln!(s, "{p}    bias: {},", opt(bias));
            let _ = writeln!(s, "{p}    subias: {},", opt(subias));
            let _ = writeln!(s, "{p}    weights: {},", opt(weights));
            let _ = writeln!(s, "{p}    float_weights: {},", opt(float_weights));
            let _ = writeln!(s, "{p}    weights_idx: {},", opt(weights_idx));
            let _ = writeln!(s, "{p}    diag: {},", opt(diag));
            let _ = writeln!(s, "{p}    scale: {},", opt(scale));
            let _ = writeln!(s, "{p}    nb_inputs: {nb_inputs},");
            let _ = writeln!(s, "{p}    nb_outputs: {nb_outputs},");
            let _ = writeln!(s, "{p}}}),");
        }
        InitCall::Conv2d { field, bias, float_weights, in_channels, out_channels, ktime, kheight } => {
            let _ = writeln!(s, "{p}LayerSpec::Conv2d(Conv2dSpec {{");
            let _ = writeln!(s, "{p}    name: \"{field}\",");
            let _ = writeln!(s, "{p}    bias: {},", opt(bias));
            let _ = writeln!(s, "{p}    float_weights: {},", opt(float_weights));
            let _ = writeln!(s, "{p}    in_channels: {in_channels},");
            let _ = writeln!(s, "{p}    out_channels: {out_channels},");
            let _ = writeln!(s, "{p}    ktime: {ktime},");
            let _ = writeln!(s, "{p}    kheight: {kheight},");
            let _ = writeln!(s, "{p}}}),");
        }
    }
    s
}

/// Appends `text` as comment lines that start with `prefix` (for example
/// `"//! "` or `"    /// "`), wrapped at 80 columns.
fn wrap(out: &mut String, prefix: &str, text: &str) {
    let mut line = String::from(prefix);
    for word in text.split_whitespace() {
        if line.len() > prefix.len() && line.chars().count() + 1 + word.chars().count() > 80 {
            out.push_str(line.trim_end());
            out.push('\n');
            line = String::from(prefix);
        }
        if line.len() > prefix.len() {
            line.push(' ');
        }
        line.push_str(word);
    }
    out.push_str(line.trim_end());
    out.push('\n');
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The doc line of a struct field.
fn field_doc(m: &Model, call: &InitCall) -> String {
    match call {
        InitCall::Linear { field, nb_inputs, nb_outputs, .. } => format!(
            "`{field}`: {} weights, {}, {}.",
            linear_kind(m, call),
            plural(*nb_inputs, "input", "inputs"),
            plural(*nb_outputs, "output", "outputs")
        ),
        InitCall::Conv2d { field, in_channels, out_channels, ktime, kheight, .. } => format!(
            "`{field}`: conv2d, {} and {}, kernel {ktime} × {kheight} (time × height).",
            plural(*in_channels, "input channel", "input channels"),
            plural(*out_channels, "output channel", "output channels")
        ),
    }
}

/// `src/dnn/models/<key>.rs`.
pub fn model_file(m: &Model, version: &str) -> String {
    let d: &ModelDef = m.def;
    let calls = &m.data.init_calls;
    let n = calls.len();
    let has_conv = calls.iter().any(|c| matches!(c, InitCall::Conv2d { .. }));
    let has_linear = calls.iter().any(|c| matches!(c, InitCall::Linear { .. }));
    let has_float_const = m.header.consts.iter().any(|c| matches!(c.value, ConstValue::Float { .. }));
    let mut s = String::new();
    wrap(&mut s, "//! ", &format!("{}: its layers and the constants of its C header.", d.summary));
    let _ = writeln!(s, "//!");
    wrap(
        &mut s,
        "//! ",
        &format!(
            "Generated by `tools/dnn-gen` from libopus {version} `dnn/{0}_data.h` and `dnn/{0}_data.c` (`{1}`). \
             Do not edit: change the generator and run it again (`tools/dnn-gen/README.md`).",
            d.key, d.init
        ),
    );
    s.push('\n');
    if has_float_const {
        let _ = writeln!(s, "// The float constants keep every digit of the C header.");
        let _ = writeln!(s, "#![allow(clippy::excessive_precision)]");
        s.push('\n');
    }
    let mut spec_imports = vec!["Layer", "LayerSpec"];
    if has_conv {
        spec_imports.push("Conv2dSpec");
    }
    if has_linear {
        spec_imports.push("LinearSpec");
    }
    spec_imports.sort_unstable();
    let _ = writeln!(s, "use super::{{{}}};", spec_imports.join(", "));
    let mut layer_imports = Vec::new();
    if has_conv {
        layer_imports.push("Conv2dLayer");
    }
    if has_linear {
        layer_imports.push("LinearLayer");
    }
    if layer_imports.len() == 1 {
        let _ = writeln!(s, "use crate::dnn::nnet::{};", layer_imports[0]);
    } else {
        let _ = writeln!(s, "use crate::dnn::nnet::{{{}}};", layer_imports.join(", "));
    }
    let _ = writeln!(s, "use crate::dnn::weights::WeightList;");
    s.push('\n');
    let _ = writeln!(s, "/// The file of this model's weights in `dnn-weights/`.");
    let _ = writeln!(s, "pub const BLOB_FILE: &str = \"{}.bin\";", d.key);
    s.push('\n');
    let _ = writeln!(s, "// The constants of `dnn/{}_data.h`, in the same order.", d.key);
    for c in &m.header.consts {
        s.push('\n');
        match &c.value {
            ConstValue::Int { value, text } => {
                if text.parse::<u64>().is_ok() {
                    let _ = writeln!(s, "/// `{}`.", c.name);
                } else {
                    let _ = writeln!(s, "/// `{}`, `{}` in C.", c.name, text);
                }
                let _ = writeln!(s, "pub const {}: usize = {value};", c.name);
            }
            ConstValue::Float { digits, text } => {
                let _ = writeln!(s, "/// `{}`, `{}` in C.", c.name, text);
                let _ = writeln!(s, "pub const {}: f32 = {digits};", c.name);
            }
        }
    }
    s.push('\n');
    wrap(
        &mut s,
        "/// ",
        &format!(
            "The layers in the order of `{}`: the arrays that each one reads, and its shape. \
             The names are the array names of the blob.",
            d.init
        ),
    );
    let _ = writeln!(s, "pub const LAYERS: [LayerSpec; {n}] = [");
    for call in calls {
        s.push_str(&layer_spec(call, 4));
    }
    let _ = writeln!(s, "];");
    s.push('\n');
    wrap(
        &mut s,
        "/// ",
        &format!(
            "libopus `{}` (`dnn/{}_data.h`): the layers of the model, in the order of the C struct. \
             Each layer borrows its arrays from a weight list.",
            d.c_struct, d.key
        ),
    );
    let _ = writeln!(s, "#[derive(Clone, Copy, Default)]");
    let _ = writeln!(s, "pub struct {}<'w> {{", d.rust_struct);
    for (field, kind) in &m.header.fields {
        let call = calls.iter().find(|c| c.field() == field).expect("checked by the generator");
        wrap(&mut s, "    /// ", &field_doc(m, call));
        let ty = match kind {
            FieldKind::Linear => "LinearLayer",
            FieldKind::Conv2d => "Conv2dLayer",
        };
        let _ = writeln!(s, "    pub {field}: {ty}<'w>,");
    }
    let _ = writeln!(s, "}}");
    s.push('\n');
    let _ = writeln!(s, "impl<'w> {}<'w> {{", d.rust_struct);
    wrap(
        &mut s,
        "    /// ",
        &format!(
            "libopus `{}`: builds every layer from the arrays of `list`, in the order of `LAYERS`. \
             Fails at the first layer whose arrays are missing, have the wrong size or are malformed; \
             then nothing is built.",
            d.init
        ),
    );
    let _ = writeln!(s, "    pub fn init(list: &WeightList<'w>) -> Result<Self, &'static str> {{");
    let _ = writeln!(s, "        Ok({} {{", d.rust_struct);
    for (i, call) in calls.iter().enumerate() {
        let method = match call {
            InitCall::Linear { .. } => "linear",
            InitCall::Conv2d { .. } => "conv2d",
        };
        let _ = writeln!(s, "            {}: LAYERS[{i}].{method}(list)?,", call.field());
    }
    let _ = writeln!(s, "        }})");
    let _ = writeln!(s, "    }}");
    s.push('\n');
    let _ = writeln!(s, "    /// Every layer, in the order of `LAYERS`.");
    let _ = writeln!(s, "    pub fn layers(&self) -> [Layer<'_, 'w>; {n}] {{");
    let _ = writeln!(s, "        [");
    for call in calls {
        let v = match call {
            InitCall::Linear { .. } => "Linear",
            InitCall::Conv2d { .. } => "Conv2d",
        };
        let _ = writeln!(s, "            Layer::{v}(&self.{}),", call.field());
    }
    let _ = writeln!(s, "        ]");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "}}");
    s
}

/// `tests/dnn_gen/mod.rs`: the C side of every model, for the tests that
/// link a libopus built with its neural network code.
pub fn ffi_file(models: &[Model], version: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "//! The C side of the libopus {version} neural network models, for the tests that");
    let _ = writeln!(s, "//! link a libopus built with them (`scripts/android/build-libopus.sh --dnn`).");
    let _ = writeln!(s, "//!");
    let _ = writeln!(s, "//! Generated by `tools/dnn-gen` from `dnn/*_data.h` and `dnn/*_data.c`. Do not");
    let _ = writeln!(s, "//! edit: change the generator and run it again (`tools/dnn-gen/README.md`).");
    let _ = writeln!(s, "//!");
    let _ = writeln!(s, "//! It holds `#[repr(C)]` copies of the C layer types and of each model struct,");
    let _ = writeln!(s, "//! the C symbols of each model (its `WeightArray` list and its init function),");
    let _ = writeln!(s, "//! and `MODELS`: one entry per model, the DRED models included, with the layer");
    let _ = writeln!(s, "//! descriptors in the order of the C init function.");
    s.push('\n');
    let _ = writeln!(s, "// Each test file uses only a part of this module, and the structs keep");
    let _ = writeln!(s, "// the names of the C structs.");
    let _ = writeln!(s, "#![allow(dead_code, clippy::upper_case_acronyms)]");
    s.push('\n');
    let _ = writeln!(s, "use opus1::internals::dnn::models::{{Conv2dSpec, LayerSpec, LinearSpec}};");
    let _ = writeln!(s, "use std::os::raw::{{c_char, c_int, c_void}};");
    s.push('\n');
    s.push_str(FFI_COMMON);
    for m in models {
        let d = m.def;
        let calls = &m.data.init_calls;
        let cs = format!("C{}", d.c_struct);
        let upper = d.key.to_uppercase();
        s.push('\n');
        let _ = writeln!(s, "/// `{}` of `dnn/{}_data.h`, field by field.", d.c_struct, d.key);
        let _ = writeln!(s, "#[repr(C)]");
        let _ = writeln!(s, "#[derive(Clone, Copy)]");
        let _ = writeln!(s, "pub struct {cs} {{");
        for (field, kind) in &m.header.fields {
            let ty = match kind {
                FieldKind::Linear => "CLinearLayer",
                FieldKind::Conv2d => "CConv2dLayer",
            };
            let _ = writeln!(s, "    pub {field}: {ty},");
        }
        let _ = writeln!(s, "}}");
        s.push('\n');
        let _ = writeln!(s, "unsafe extern \"C\" {{");
        let _ = writeln!(s, "    /// The first entry of the list; the list ends with a `NULL` name.");
        let _ = writeln!(s, "    static {}: CWeightArray;", d.list);
        let _ = writeln!(s, "    fn {}(model: *mut {cs}, arrays: *const CWeightArray) -> c_int;", d.init);
        let _ = writeln!(s, "}}");
        s.push('\n');
        let _ = writeln!(s, "fn c_{}_arrays() -> *const CWeightArray {{", d.key);
        let _ = writeln!(s, "    &raw const {}", d.list);
        let _ = writeln!(s, "}}");
        s.push('\n');
        let _ = writeln!(s, "fn c_{}_init(arrays: *const CWeightArray) -> (c_int, Vec<CLayer>) {{", d.key);
        let _ = writeln!(s, "    // SAFETY: the struct holds only pointers and integers, so all zeros is a");
        let _ = writeln!(s, "    // valid value (null pointers, zero sizes). The init function reads the");
        let _ = writeln!(s, "    // list up to its end entry and writes only the struct.");
        let _ = writeln!(s, "    let mut m: {cs} = unsafe {{ std::mem::zeroed() }};");
        let _ = writeln!(s, "    let ret = unsafe {{ {}(&mut m, arrays) }};", d.init);
        let _ = writeln!(s, "    let layers = vec![");
        for call in calls {
            let v = match call {
                InitCall::Linear { .. } => "Linear",
                InitCall::Conv2d { .. } => "Conv2d",
            };
            let _ = writeln!(s, "        CLayer::{v}(m.{}),", call.field());
        }
        let _ = writeln!(s, "    ];");
        let _ = writeln!(s, "    (ret, layers)");
        let _ = writeln!(s, "}}");
        s.push('\n');
        let _ = writeln!(s, "const {upper}_LAYERS: [LayerSpec; {}] = [", calls.len());
        for call in calls {
            s.push_str(&layer_spec(call, 4));
        }
        let _ = writeln!(s, "];");
    }
    s.push('\n');
    let _ = writeln!(s, "/// Every model, in the order of the official blob writer (`dnn/write_lpcnet_weights.c`),");
    let _ = writeln!(s, "/// then BBWENet, which that writer leaves out.");
    let _ = writeln!(s, "pub const MODELS: [CModel; {}] = [", models.len());
    for m in models {
        let d = m.def;
        let _ = writeln!(s, "    CModel {{");
        let _ = writeln!(s, "        key: \"{}\",", d.key);
        let _ = writeln!(s, "        feature: \"{}\",", d.feature);
        let _ = writeln!(s, "        c_struct: \"{}\",", d.c_struct);
        let _ = writeln!(s, "        layers: &{}_LAYERS,", d.key.to_uppercase());
        let _ = writeln!(s, "        arrays: c_{}_arrays,", d.key);
        let _ = writeln!(s, "        init: c_{}_init,", d.key);
        let _ = writeln!(s, "    }},");
    }
    let _ = writeln!(s, "];");
    s
}

const FFI_COMMON: &str = r#"/// `LinearLayer` of `dnn/nnet.h:66-76`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CLinearLayer {
    pub bias: *const f32,
    pub subias: *const f32,
    pub weights: *const i8,
    pub float_weights: *const f32,
    pub weights_idx: *const c_int,
    pub diag: *const f32,
    pub scale: *const f32,
    pub nb_inputs: c_int,
    pub nb_outputs: c_int,
}

/// `Conv2dLayer` of `dnn/nnet.h:79-86`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CConv2dLayer {
    pub bias: *const f32,
    pub float_weights: *const f32,
    pub in_channels: c_int,
    pub out_channels: c_int,
    pub ktime: c_int,
    pub kheight: c_int,
}

/// `WeightArray` of `dnn/nnet.h:44-49`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CWeightArray {
    pub name: *const c_char,
    pub kind: c_int,
    pub size: c_int,
    pub data: *const c_void,
}

/// One layer of a C model struct.
#[derive(Clone, Copy, Debug)]
pub enum CLayer {
    Linear(CLinearLayer),
    Conv2d(CConv2dLayer),
}

/// One model of libopus.
pub struct CModel {
    /// The name of the data file without `_data.c`, which is also the name
    /// of the model's blob in `dnn-weights/`.
    pub key: &'static str,
    /// The opus1 feature that includes the model (`dred` for the two DRED
    /// models, which opus1 does not include yet).
    pub feature: &'static str,
    /// The C struct of the model.
    pub c_struct: &'static str,
    /// The layer descriptors, in the order of the C init function.
    pub layers: &'static [LayerSpec],
    /// The C list of arrays (`*_arrays`), which ends with a `NULL` name.
    pub arrays: fn() -> *const CWeightArray,
    /// Runs the C init function on a list: its return value, and the
    /// layers in the order of `layers`.
    pub init: fn(*const CWeightArray) -> (c_int, Vec<CLayer>),
}
"#;
