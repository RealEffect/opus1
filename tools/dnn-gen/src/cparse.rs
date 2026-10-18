//! Reads the generated data files of libopus (`dnn/*_data.c` and
//! `dnn/*_data.h`).
//!
//! These files are plain C written by libopus's training scripts. They use
//! only a small part of C, and this module understands exactly that part:
//!
//! - comments, and the preprocessor lines `#ifdef`, `#ifndef`, `#else`,
//!   `#endif`, `#define`, `#undef` and `#include` (which is ignored);
//! - arrays: `static const <type> <name>[<count>] = { <values> };`
//!   with the types `float`, `int` and `opus_int8`;
//! - the list of arrays: `const WeightArray <list>[] = { {"<name>",
//!   <type macro>, sizeof(<array>), <array>}, ..., {NULL,0,0,NULL} };`;
//! - the init function: `int <init>(<struct> *model, const WeightArray
//!   *arrays) { if (linear_init(&model-><field>, arrays, ...)) return 1;
//!   ... return 0; }`, with `conv2d_init` calls too;
//! - in the header: `#define` constants and the struct of layers.
//!
//! Anything else stops the generator with an error that names the file and
//! the line, so a change of format in a later libopus is noticed.

use std::collections::{HashMap, HashSet};

/// The macros that a default libopus build defines when it compiles a data
/// file. `DISABLE_DEBUG_FLOAT` is set by configure, CMake and meson unless
/// the debug option is given (`configure.ac:937-943`); it removes the float
/// copies of int8 weights in `bbwenet_data.c`. `USE_WEIGHTS_FILE` and
/// `DUMP_BINARY_WEIGHTS` are not defined, so the arrays and the init
/// function are both compiled.
pub const BUILD_DEFINES: &[&str] = &["DISABLE_DEBUG_FLOAT"];

/// An error with the file and line it comes from.
pub type Error = String;

fn err(file: &str, line: usize, msg: impl AsRef<str>) -> Error {
    format!("{file}:{line}: {}", msg.as_ref())
}

/// One `#define` that was active, in file order.
#[derive(Clone, Debug)]
pub struct Define {
    pub name: String,
    /// The text after the name, trimmed. Empty for `#define X`.
    pub value: String,
}

/// The result of preprocessing a file: the active lines that are not
/// preprocessor lines (with their line numbers), and the active defines.
struct Preprocessed {
    lines: Vec<(usize, String)>,
    defines: Vec<Define>,
}

/// Replaces every comment by spaces, keeping the line breaks, so that line
/// numbers stay right. String literals are kept as they are.
fn strip_comments(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            // Copy the string literal, with its escapes.
            out.push(b[i]);
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    out.push(b[i]);
                    i += 1;
                }
                out.push(b[i]);
                i += 1;
            }
            if i < b.len() {
                out.push(b[i]);
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            out.extend_from_slice(b"  ");
            while i < b.len() && !(b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/') {
                out.push(if b[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.extend_from_slice(b"  ");
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).expect("comment removal keeps UTF-8")
}

/// Runs the conditional part of the C preprocessor over `text`.
fn preprocess(file: &str, text: &str, predefined: &[&str]) -> Result<Preprocessed, Error> {
    let text = strip_comments(text);
    let mut defined: HashSet<String> = predefined.iter().map(|s| s.to_string()).collect();
    let mut defines = Vec::new();
    let mut lines = Vec::new();
    // One entry per open `#if...`: (this branch is active, the enclosing
    // region is active, an `#else` was seen).
    let mut stack: Vec<(bool, bool, bool)> = Vec::new();
    let mut active = true;
    for (n, raw) in text.lines().enumerate() {
        let line_no = n + 1;
        let t = raw.trim();
        if let Some(rest) = t.strip_prefix('#') {
            let rest = rest.trim_start();
            let (word, arg) = match rest.find(char::is_whitespace) {
                Some(p) => (&rest[..p], rest[p..].trim()),
                None => (rest, ""),
            };
            match word {
                "ifdef" | "ifndef" => {
                    let name = single_ident(file, line_no, arg)?;
                    let is_def = defined.contains(name);
                    let cond = if word == "ifdef" { is_def } else { !is_def };
                    stack.push((cond, active, false));
                    active = active && cond;
                }
                "else" => {
                    let top = stack
                        .last_mut()
                        .ok_or_else(|| err(file, line_no, "#else without #if"))?;
                    if top.2 {
                        return Err(err(file, line_no, "second #else"));
                    }
                    top.2 = true;
                    top.0 = !top.0;
                    active = top.1 && top.0;
                }
                "endif" => {
                    let top = stack
                        .pop()
                        .ok_or_else(|| err(file, line_no, "#endif without #if"))?;
                    active = top.1;
                }
                "define" if active => {
                    let (name, value) = match arg.find(char::is_whitespace) {
                        Some(p) => (&arg[..p], arg[p..].trim()),
                        None => (arg, ""),
                    };
                    if name.contains('(') {
                        return Err(err(file, line_no, "function-like macros are not supported"));
                    }
                    single_ident(file, line_no, name)?;
                    defined.insert(name.to_string());
                    defines.push(Define { name: name.to_string(), value: value.to_string() });
                }
                "undef" if active => {
                    defined.remove(single_ident(file, line_no, arg)?);
                }
                "include" => {}
                "define" | "undef" => {}
                _ => {
                    return Err(err(file, line_no, format!("unsupported preprocessor line `#{word}`")));
                }
            }
        } else if active && !t.is_empty() {
            lines.push((line_no, raw.to_string()));
        }
    }
    if !stack.is_empty() {
        return Err(err(file, text.lines().count(), "missing #endif"));
    }
    Ok(Preprocessed { lines, defines })
}

fn single_ident<'a>(file: &str, line: usize, s: &'a str) -> Result<&'a str, Error> {
    let ok = !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.starts_with(|c: char| c.is_ascii_digit());
    if ok { Ok(s) } else { Err(err(file, line, format!("expected one identifier, found `{s}`"))) }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Number(String),
    Str(String),
    Punct(char),
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: usize,
}

fn tokenize(file: &str, lines: &[(usize, String)]) -> Result<Vec<Token>, Error> {
    let mut out = Vec::new();
    for (line, text) in lines {
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c.is_ascii_whitespace() {
                i += 1;
            } else if c.is_ascii_alphabetic() || c == b'_' {
                let s = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(Token { tok: Tok::Ident(text[s..i].to_string()), line: *line });
            } else if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
                let s = i;
                while i < b.len() {
                    let d = b[i];
                    let exp_sign = (d == b'+' || d == b'-') && matches!(b[i - 1], b'e' | b'E');
                    if d.is_ascii_alphanumeric() || d == b'.' || exp_sign {
                        i += 1;
                    } else {
                        break;
                    }
                }
                out.push(Token { tok: Tok::Number(text[s..i].to_string()), line: *line });
            } else if c == b'"' {
                let s = i + 1;
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        return Err(err(file, *line, "escapes in string literals are not supported"));
                    }
                    i += 1;
                }
                if i >= b.len() {
                    return Err(err(file, *line, "unterminated string literal"));
                }
                out.push(Token { tok: Tok::Str(text[s..i].to_string()), line: *line });
                i += 1;
            } else if b"{}[]()=,;*&-+<>".contains(&c) {
                // `->` is read as `-` followed by `>`.
                out.push(Token { tok: Tok::Punct(c as char), line: *line });
                i += 1;
            } else {
                return Err(err(file, *line, format!("unexpected character `{}`", c as char)));
            }
        }
    }
    Ok(out)
}

/// A cursor over the tokens of one file.
struct Cursor<'a> {
    file: &'a str,
    toks: &'a [Token],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn line(&self) -> usize {
        self.toks
            .get(self.pos)
            .or_else(|| self.toks.last())
            .map_or(0, |t| t.line)
    }
    fn error(&self, msg: impl AsRef<str>) -> Error {
        err(self.file, self.line(), msg)
    }
    fn peek(&self) -> Option<&'a Tok> {
        self.toks.get(self.pos).map(|t| &t.tok)
    }
    fn peek_at(&self, k: usize) -> Option<&'a Tok> {
        self.toks.get(self.pos + k).map(|t| &t.tok)
    }
    fn at_end(&self) -> bool {
        self.pos >= self.toks.len()
    }
    fn next(&mut self) -> Result<&'a Tok, Error> {
        let t = self.toks.get(self.pos).map(|t| &t.tok).ok_or_else(|| self.error("unexpected end of file"))?;
        self.pos += 1;
        Ok(t)
    }
    fn punct(&mut self, c: char) -> Result<(), Error> {
        match self.next()? {
            Tok::Punct(p) if *p == c => Ok(()),
            t => Err(self.error(format!("expected `{c}`, found {t:?}"))),
        }
    }
    fn is_punct(&self, c: char) -> bool {
        matches!(self.peek(), Some(Tok::Punct(p)) if *p == c)
    }
    fn ident(&mut self) -> Result<&'a str, Error> {
        match self.next()? {
            Tok::Ident(s) => Ok(s),
            t => Err(self.error(format!("expected an identifier, found {t:?}"))),
        }
    }
    fn keyword(&mut self, k: &str) -> Result<(), Error> {
        let s = self.ident()?;
        if s == k { Ok(()) } else { Err(self.error(format!("expected `{k}`, found `{s}`"))) }
    }
    fn is_ident(&self, k: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s == k)
    }
    fn number(&mut self) -> Result<&'a str, Error> {
        match self.next()? {
            Tok::Number(s) => Ok(s),
            t => Err(self.error(format!("expected a number, found {t:?}"))),
        }
    }
    fn usize(&mut self) -> Result<usize, Error> {
        let s = self.number()?;
        s.parse::<usize>().map_err(|_| self.error(format!("expected a count, found `{s}`")))
    }
    /// A string literal or `NULL`.
    fn opt_str(&mut self) -> Result<Option<String>, Error> {
        match self.next()? {
            Tok::Str(s) => Ok(Some(s.clone())),
            Tok::Ident(s) if s == "NULL" => Ok(None),
            t => Err(self.error(format!("expected a string or NULL, found {t:?}"))),
        }
    }
}

/// The C element type of an array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CType {
    Float,
    Int,
    Int8,
}

impl CType {
    pub fn size(self) -> usize {
        match self {
            CType::Float | CType::Int => 4,
            CType::Int8 => 1,
        }
    }
    pub fn c_name(self) -> &'static str {
        match self {
            CType::Float => "float",
            CType::Int => "int",
            CType::Int8 => "opus_int8",
        }
    }
}

/// One `static const` array of a data file.
#[derive(Clone, Debug)]
pub struct CArray {
    pub ctype: CType,
    /// The values as little-endian bytes, as the C compiler lays them out.
    pub data: Vec<u8>,
    /// The number of int8 values equal to -128 (0 for other types).
    pub minus_128: usize,
    pub line: usize,
}

/// One entry of the `WeightArray` list.
#[derive(Clone, Debug)]
pub struct ListEntry {
    /// The name stored in the record.
    pub name: String,
    /// The `WEIGHT_TYPE_*` code.
    pub kind: i32,
    /// The array whose bytes the record holds.
    pub array: String,
}

/// One call of `linear_init` or `conv2d_init` in the init function.
#[derive(Clone, Debug, PartialEq)]
pub enum InitCall {
    Linear {
        field: String,
        bias: Option<String>,
        subias: Option<String>,
        weights: Option<String>,
        float_weights: Option<String>,
        weights_idx: Option<String>,
        diag: Option<String>,
        scale: Option<String>,
        nb_inputs: usize,
        nb_outputs: usize,
    },
    Conv2d {
        field: String,
        bias: Option<String>,
        float_weights: Option<String>,
        in_channels: usize,
        out_channels: usize,
        ktime: usize,
        kheight: usize,
    },
}

impl InitCall {
    pub fn field(&self) -> &str {
        match self {
            InitCall::Linear { field, .. } | InitCall::Conv2d { field, .. } => field,
        }
    }
}

/// What a `*_data.c` file holds.
#[derive(Clone, Debug)]
pub struct DataFile {
    pub arrays: HashMap<String, CArray>,
    /// The list named `list`, in order, with only the entries whose
    /// `WEIGHTS_*_DEFINED` macro is set in a default build.
    pub list: Vec<ListEntry>,
    /// The name of the struct that the init function fills.
    pub init_struct: String,
    pub init_calls: Vec<InitCall>,
    /// The line of the init function.
    pub init_line: usize,
}

/// The `WEIGHT_TYPE_*` codes of `dnn/nnet.h:51-54`.
fn weight_type_code(name: &str) -> Option<i32> {
    match name {
        "WEIGHT_TYPE_float" => Some(0),
        "WEIGHT_TYPE_int" => Some(1),
        "WEIGHT_TYPE_qweight" => Some(2),
        "WEIGHT_TYPE_int8" => Some(3),
        _ => None,
    }
}

/// Reads a `*_data.c` file. `list_name` and `init_name` name the
/// `WeightArray` list and the init function to read.
pub fn parse_data_file(file: &str, text: &str, list_name: &str, init_name: &str) -> Result<DataFile, Error> {
    let pp = preprocess(file, text, BUILD_DEFINES)?;
    let macros: HashMap<&str, &str> = pp.defines.iter().map(|d| (d.name.as_str(), d.value.as_str())).collect();
    let toks = tokenize(file, &pp.lines)?;
    let mut c = Cursor { file, toks: &toks, pos: 0 };
    let mut arrays = HashMap::new();
    let mut list = None;
    let mut init = None;
    while !c.at_end() {
        if c.is_ident("static") {
            let line = c.line();
            let (name, arr) = parse_array(&mut c, line)?;
            if arrays.insert(name.clone(), arr).is_some() {
                return Err(err(file, line, format!("array `{name}` defined twice")));
            }
        } else if c.is_ident("const") && matches!(c.peek_at(1), Some(Tok::Ident(s)) if s == "WeightArray") {
            let line = c.line();
            c.keyword("const")?;
            c.keyword("WeightArray")?;
            let name = c.ident()?.to_string();
            let entries = parse_list(&mut c, &macros)?;
            if name == list_name {
                if list.is_some() {
                    return Err(err(file, line, format!("list `{name}` defined twice")));
                }
                list = Some(entries);
            }
        } else if c.is_ident("int") && matches!(c.peek_at(1), Some(Tok::Ident(s)) if s == init_name) {
            let line = c.line();
            c.keyword("int")?;
            c.ident()?;
            let (st, calls) = parse_init(&mut c)?;
            init = Some((st, calls, line));
        } else {
            return Err(c.error(format!("unexpected {:?} at the top level", c.peek())));
        }
    }
    let list = list.ok_or_else(|| err(file, 0, format!("no list `{list_name}`")))?;
    let (init_struct, init_calls, init_line) = init.ok_or_else(|| err(file, 0, format!("no function `{init_name}`")))?;
    for e in &list {
        if !arrays.contains_key(&e.array) {
            return Err(err(file, 0, format!("list entry `{}` names the missing array `{}`", e.name, e.array)));
        }
    }
    Ok(DataFile { arrays, list, init_struct, init_calls, init_line })
}

fn parse_array(c: &mut Cursor, line: usize) -> Result<(String, CArray), Error> {
    c.keyword("static")?;
    c.keyword("const")?;
    let ctype = match c.ident()? {
        "float" => CType::Float,
        "int" => CType::Int,
        "opus_int8" => CType::Int8,
        t => return Err(c.error(format!("unsupported array type `{t}`"))),
    };
    let name = c.ident()?.to_string();
    c.punct('[')?;
    let count = c.usize()?;
    c.punct(']')?;
    c.punct('=')?;
    c.punct('{')?;
    let mut data = Vec::with_capacity(count * ctype.size());
    let mut n = 0usize;
    let mut minus_128 = 0usize;
    while !c.is_punct('}') {
        let neg = if c.is_punct('-') {
            c.punct('-')?;
            true
        } else {
            if c.is_punct('+') {
                c.punct('+')?;
            }
            false
        };
        let s = c.number()?;
        match ctype {
            CType::Float => {
                let v = parse_c_float(s).ok_or_else(|| c.error(format!("bad float `{s}`")))?;
                let v = if neg { -v } else { v };
                data.extend_from_slice(&v.to_le_bytes());
            }
            CType::Int | CType::Int8 => {
                let v: i64 = s.parse().map_err(|_| c.error(format!("bad integer `{s}`")))?;
                let v = if neg { -v } else { v };
                if ctype == CType::Int8 {
                    let b = i8::try_from(v).map_err(|_| c.error(format!("{v} does not fit in opus_int8")))?;
                    if b == -128 {
                        minus_128 += 1;
                    }
                    data.push(b as u8);
                } else {
                    let i = i32::try_from(v).map_err(|_| c.error(format!("{v} does not fit in int")))?;
                    data.extend_from_slice(&i.to_le_bytes());
                }
            }
        }
        n += 1;
        if c.is_punct(',') {
            c.punct(',')?;
        } else {
            break;
        }
    }
    c.punct('}')?;
    c.punct(';')?;
    if n > count {
        return Err(err(c.file, line, format!("array `{name}` has {n} values for {count} elements")));
    }
    // C fills the elements without a value with zeros.
    data.resize(count * ctype.size(), 0);
    Ok((name, CArray { ctype, data, minus_128, line }))
}

/// A float literal as C reads it: without a suffix it is a `double`,
/// converted to `float` with rounding to nearest (the data files write
/// their values this way); with an `f` suffix it is read as a `float`
/// directly.
fn parse_c_float(s: &str) -> Option<f32> {
    if let Some(body) = s.strip_suffix(['f', 'F']) {
        body.parse::<f32>().ok()
    } else {
        s.parse::<f64>().ok().map(|d| d as f32)
    }
}

fn parse_list(c: &mut Cursor, macros: &HashMap<&str, &str>) -> Result<Vec<ListEntry>, Error> {
    c.punct('[')?;
    c.punct(']')?;
    c.punct('=')?;
    c.punct('{')?;
    let mut out = Vec::new();
    loop {
        c.punct('{')?;
        let name = c.opt_str()?;
        c.punct(',')?;
        let Some(name) = name else {
            // The end entry: {NULL, 0, 0, NULL}.
            for _ in 0..2 {
                if c.number()? != "0" {
                    return Err(c.error("the end entry of a list must be {NULL,0,0,NULL}"));
                }
                c.punct(',')?;
            }
            c.keyword("NULL")?;
            c.punct('}')?;
            break;
        };
        let type_macro = c.ident()?;
        let type_name = macros.get(type_macro).copied().unwrap_or(type_macro);
        let kind = weight_type_code(type_name)
            .ok_or_else(|| c.error(format!("unknown weight type `{type_macro}`")))?;
        c.punct(',')?;
        c.keyword("sizeof")?;
        c.punct('(')?;
        let sized = c.ident()?.to_string();
        c.punct(')')?;
        c.punct(',')?;
        let array = c.ident()?.to_string();
        if sized != array {
            return Err(c.error(format!("entry `{name}`: sizeof({sized}) of another array than `{array}`")));
        }
        c.punct('}')?;
        c.punct(',')?;
        out.push(ListEntry { name, kind, array });
    }
    if c.is_punct(',') {
        c.punct(',')?;
    }
    c.punct('}')?;
    c.punct(';')?;
    Ok(out)
}

fn parse_init(c: &mut Cursor) -> Result<(String, Vec<InitCall>), Error> {
    c.punct('(')?;
    let st = c.ident()?.to_string();
    c.punct('*')?;
    c.keyword("model")?;
    c.punct(',')?;
    c.keyword("const")?;
    c.keyword("WeightArray")?;
    c.punct('*')?;
    c.keyword("arrays")?;
    c.punct(')')?;
    c.punct('{')?;
    let mut calls = Vec::new();
    loop {
        if c.is_ident("return") {
            c.keyword("return")?;
            if c.number()? != "0" {
                return Err(c.error("the init function must end with `return 0;`"));
            }
            c.punct(';')?;
            c.punct('}')?;
            break;
        }
        c.keyword("if")?;
        c.punct('(')?;
        let func = c.ident()?.to_string();
        c.punct('(')?;
        c.punct('&')?;
        c.keyword("model")?;
        c.punct('-')?;
        c.punct('>')?;
        let field = c.ident()?.to_string();
        c.punct(',')?;
        c.keyword("arrays")?;
        let call = match func.as_str() {
            "linear_init" => {
                let mut names = Vec::new();
                for _ in 0..7 {
                    c.punct(',')?;
                    names.push(c.opt_str()?);
                }
                c.punct(',')?;
                let nb_inputs = c.usize()?;
                c.punct(',')?;
                let nb_outputs = c.usize()?;
                let mut it = names.into_iter();
                let mut take = || it.next().unwrap();
                InitCall::Linear {
                    field,
                    bias: take(),
                    subias: take(),
                    weights: take(),
                    float_weights: take(),
                    weights_idx: take(),
                    diag: take(),
                    scale: take(),
                    nb_inputs,
                    nb_outputs,
                }
            }
            "conv2d_init" => {
                c.punct(',')?;
                let bias = c.opt_str()?;
                c.punct(',')?;
                let float_weights = c.opt_str()?;
                let mut dims = [0usize; 4];
                for d in &mut dims {
                    c.punct(',')?;
                    *d = c.usize()?;
                }
                InitCall::Conv2d {
                    field,
                    bias,
                    float_weights,
                    in_channels: dims[0],
                    out_channels: dims[1],
                    ktime: dims[2],
                    kheight: dims[3],
                }
            }
            f => return Err(c.error(format!("unsupported call `{f}` in the init function"))),
        };
        c.punct(')')?;
        c.punct(')')?;
        c.keyword("return")?;
        if c.number()? != "1" {
            return Err(c.error("a failed init call must `return 1;`"));
        }
        c.punct(';')?;
        calls.push(call);
    }
    Ok((st, calls))
}

/// The kind of a struct field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Linear,
    Conv2d,
}

/// A value of a header constant.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    /// An integer, or a product of integers (`(173 * (0))`). `text` is the
    /// C text.
    Int { value: u64, text: String },
    /// A float literal with the `f` suffix; `digits` is the literal without
    /// the suffix.
    Float { digits: String, text: String },
}

/// A constant of a header.
#[derive(Clone, Debug)]
pub struct HeaderConst {
    pub name: String,
    pub value: ConstValue,
}

/// What a `*_data.h` file holds.
#[derive(Clone, Debug)]
pub struct HeaderFile {
    pub consts: Vec<HeaderConst>,
    /// The struct of layers, in field order.
    pub fields: Vec<(String, FieldKind)>,
}

/// Reads a `*_data.h` file and the struct named `struct_name`.
pub fn parse_header(file: &str, text: &str, struct_name: &str) -> Result<HeaderFile, Error> {
    let pp = preprocess(file, text, &[])?;
    let mut consts = Vec::new();
    for d in &pp.defines {
        if d.value.is_empty() {
            // The include guard.
            continue;
        }
        let value = parse_const(&d.value).ok_or_else(|| err(file, 0, format!("unsupported value `{}` of `{}`", d.value, d.name)))?;
        consts.push(HeaderConst { name: d.name.clone(), value });
    }
    let toks = tokenize(file, &pp.lines)?;
    let mut c = Cursor { file, toks: &toks, pos: 0 };
    let mut fields = None;
    while !c.at_end() {
        // `typedef struct { ... } NAME;` or `struct NAME { ... };`
        if c.is_ident("typedef") && matches!(c.peek_at(1), Some(Tok::Ident(s)) if s == "struct") && matches!(c.peek_at(2), Some(Tok::Punct('{'))) {
            c.keyword("typedef")?;
            c.keyword("struct")?;
            let f = parse_fields(&mut c)?;
            let name = c.ident()?;
            c.punct(';')?;
            if name == struct_name {
                fields = Some(f);
            }
        } else if c.is_ident("struct") && matches!(c.peek_at(2), Some(Tok::Punct('{'))) {
            c.keyword("struct")?;
            let name = c.ident()?.to_string();
            let f = parse_fields(&mut c)?;
            c.punct(';')?;
            if name == struct_name {
                fields = Some(f);
            }
        } else {
            // Prototypes and other declarations: skip to the next `;`.
            while !c.is_punct(';') {
                c.next()?;
            }
            c.punct(';')?;
        }
    }
    let fields = fields.ok_or_else(|| err(file, 0, format!("no struct `{struct_name}`")))?;
    Ok(HeaderFile { consts, fields })
}

fn parse_fields(c: &mut Cursor) -> Result<Vec<(String, FieldKind)>, Error> {
    c.punct('{')?;
    let mut out = Vec::new();
    while !c.is_punct('}') {
        let kind = match c.ident()? {
            "LinearLayer" => FieldKind::Linear,
            "Conv2dLayer" => FieldKind::Conv2d,
            t => return Err(c.error(format!("unsupported field type `{t}`"))),
        };
        let name = c.ident()?.to_string();
        c.punct(';')?;
        out.push((name, kind));
    }
    c.punct('}')?;
    Ok(out)
}

/// `80`, `0.85f`, `(173 * (0))`.
fn parse_const(v: &str) -> Option<ConstValue> {
    let text = v.to_string();
    if let Some(digits) = v.strip_suffix('f') {
        let ok = digits.parse::<f32>().is_ok() && digits.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+');
        return ok.then(|| ConstValue::Float { digits: normalize_float_digits(digits), text });
    }
    if let Ok(value) = v.parse::<u64>() {
        return Some(ConstValue::Int { value, text });
    }
    // A product of integers, with parentheses.
    let inner: String = v.chars().filter(|c| *c != '(' && *c != ')' && !c.is_whitespace()).collect();
    let mut value = 1u64;
    for part in inner.split('*') {
        value = value.checked_mul(part.parse::<u64>().ok()?)?;
    }
    Some(ConstValue::Int { value, text })
}

/// Makes a C float literal a valid Rust literal (`.5` → `0.5`, `5.` → `5.0`).
fn normalize_float_digits(d: &str) -> String {
    let (sign, body) = match d.strip_prefix('-') {
        Some(b) => ("-", b),
        None => ("", d),
    };
    let mut s = String::from(sign);
    if body.starts_with('.') {
        s.push('0');
    }
    s.push_str(body);
    if s.ends_with('.') {
        s.push('0');
    }
    if !s.contains(['.', 'e', 'E']) {
        s.push_str(".0");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditionals_and_comments() {
        let text = "/* a */\n#ifndef USE_WEIGHTS_FILE\nA\n#ifndef DISABLE_DEBUG_FLOAT\nB\n#define X 1\n#endif /* x */\n#ifdef X\nC\n#else\nD\n#endif\n#endif\n";
        let pp = preprocess("t", text, BUILD_DEFINES).unwrap();
        let lines: Vec<&str> = pp.lines.iter().map(|(_, l)| l.trim()).collect();
        assert_eq!(lines, ["A", "D"]);
        assert!(pp.defines.is_empty());
    }

    #[test]
    fn consts() {
        assert_eq!(parse_const("80"), Some(ConstValue::Int { value: 80, text: "80".into() }));
        assert_eq!(parse_const("(173 * (0))"), Some(ConstValue::Int { value: 0, text: "(173 * (0))".into() }));
        assert_eq!(parse_const("(64 * (2))"), Some(ConstValue::Int { value: 128, text: "(64 * (2))".into() }));
        assert_eq!(
            parse_const("0.85f"),
            Some(ConstValue::Float { digits: "0.85".into(), text: "0.85f".into() })
        );
        assert_eq!(parse_const("x"), None);
    }

    #[test]
    fn c_floats_round_through_double() {
        // Without a suffix C reads a double, then rounds it to float.
        assert_eq!(parse_c_float("0.1"), Some(0.1f64 as f32));
        assert_eq!(parse_c_float("7.272227958310395e-05"), Some(7.272227958310395e-05f64 as f32));
        assert_eq!(parse_c_float("1.5f"), Some(1.5f32));
    }

    #[test]
    fn small_data_file() {
        let text = r#"
#ifndef USE_WEIGHTS_FILE
#define WEIGHTS_l_bias_DEFINED
#define WEIGHTS_l_bias_TYPE WEIGHT_TYPE_float
static const float l_bias[2] = {
0.5,-1e-3,
};
#endif
#ifndef USE_WEIGHTS_FILE
#ifndef DISABLE_DEBUG_FLOAT
#define WEIGHTS_l_weights_float_DEFINED
#define WEIGHTS_l_weights_float_TYPE WEIGHT_TYPE_float
static const float l_weights_float[1] = {1};
#endif /*DISABLE_DEBUG_FLOAT*/
#endif
#ifndef USE_WEIGHTS_FILE
const WeightArray m_arrays[] = {
#ifdef WEIGHTS_l_bias_DEFINED
{"l_bias", WEIGHTS_l_bias_TYPE,sizeof(l_bias),l_bias},
#endif
#ifdef WEIGHTS_l_weights_float_DEFINED
{"l_weights_float", WEIGHTS_l_weights_float_TYPE,sizeof(l_weights_float),l_weights_float},
#endif
{NULL,0,0,NULL}
};
#endif
#ifndef DUMP_BINARY_WEIGHTS
int init_m(M *model,const WeightArray *arrays) {
if (linear_init(&model->l,arrays,"l_bias",NULL,NULL,"l_weights_float",NULL,NULL,NULL,1,2)) return 1;
return 0;
}
#endif
"#;
        let d = parse_data_file("t", text, "m_arrays", "init_m").unwrap();
        assert_eq!(d.list.len(), 1);
        assert_eq!(d.list[0].name, "l_bias");
        assert_eq!(d.list[0].kind, 0);
        let mut want = Vec::new();
        want.extend_from_slice(&0.5f32.to_le_bytes());
        want.extend_from_slice(&(-1e-3f64 as f32).to_le_bytes());
        assert_eq!(d.arrays["l_bias"].data, want);
        assert!(!d.arrays.contains_key("l_weights_float"));
        assert_eq!(d.init_struct, "M");
        assert_eq!(d.init_calls.len(), 1);
        assert_eq!(d.init_calls[0].field(), "l");
    }
}
