//! Builds the fixed-point libopus 1.6.1 that the tests of opus1 compare
//! with, from the source of the official release.
//!
//! The tests also link the float build of libopus (through `audiopus_sys`).
//! Both builds define the same symbols (`opus_encode`, `silk_NSQ_c`, ...),
//! so every global symbol of the fixed-point build gets the prefix
//! `libopus_fixed_`:
//!
//! 1. Compile the C files once and list the global symbols they define
//!    (with `nm`, or `dumpbin` for MSVC).
//! 2. Write a header with one `#define name libopus_fixed_name` per symbol.
//! 3. Compile the C files again with that header included first, together
//!    with `shim.c`, whose functions (`fixed_ref_...`) the Rust side calls.
//! 4. Check that every global symbol now has one of the two prefixes, and
//!    put the objects in a static library.
//!
//! The compiler flags are those of the reference build of the earlier
//! comparisons: `-O2 -DOPUS_BUILD -DFIXED_POINT -DVAR_ARRAYS -DHAVE_LRINTF
//! -DHAVE_LRINT`, and nothing else (no `config.h`, no vector code). MSVC has
//! no variable-length arrays and gets `USE_ALLOCA` instead. That changes
//! where the scratch memory is, not the results.
//!
//! When the source is not found or the build fails, the crate is built
//! without the library: every function then panics with the reason. The
//! tests that need the library fail and say why, the other tests still run.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Prefix of every global symbol of the fixed-point build.
const PREFIX: &str = "libopus_fixed_";
/// Prefix of the functions of `shim.c`.
const SHIM_PREFIX: &str = "fixed_ref_";
/// Name of the static library.
const LIB: &str = "opus_fixed_161";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=shim.c");
    println!("cargo::rerun-if-env-changed=LIBOPUS_SRC_1_6_1");
    println!("cargo::rustc-check-cfg=cfg(libopus_fixed_missing)");
    if let Err(why) = build() {
        let why = why.replace(['\n', '\r'], " ");
        println!("cargo::warning=the fixed-point libopus 1.6.1 was not built: {why}");
        println!("cargo::rustc-cfg=libopus_fixed_missing");
        println!("cargo::rustc-env=LIBOPUS_FIXED_MISSING={why}");
    }
}

fn build() -> Result<(), String> {
    let src = source_dir()?;
    let out = PathBuf::from(env::var("OUT_DIR").map_err(|e| e.to_string())?);

    // The C files of the release's own build lists, without the vector
    // code and without the neural network code.
    let mut files = Vec::new();
    for (mk, list) in [
        ("opus_sources.mk", "OPUS_SOURCES"),
        ("opus_sources.mk", "OPUS_SOURCES_FLOAT"),
        ("celt_sources.mk", "CELT_SOURCES"),
        ("silk_sources.mk", "SILK_SOURCES"),
        ("silk_sources.mk", "SILK_SOURCES_FIXED"),
    ] {
        for file in mk_list(&src, mk, list)? {
            // silk/fixed/foo.c gives the object silk_fixed_foo.o.
            let stem = file.trim_end_matches(".c").replace('/', "_");
            files.push((stem, src.join(&file)));
        }
    }

    let mut cfg = cc::Build::new();
    cfg.opt_level(2)
        .debug(false)
        .warnings(false)
        .extra_warnings(false)
        .inherit_rustflags(false)
        .define("OPUS_BUILD", None)
        .define("FIXED_POINT", None)
        .define("HAVE_LRINTF", None)
        .define("HAVE_LRINT", None)
        .define("PACKAGE_VERSION", "\"1.6.1\"");
    for dir in ["include", "celt", "silk", "silk/fixed", "src", ""] {
        cfg.include(src.join(dir));
    }
    let msvc = cfg.try_get_compiler().map_err(|e| e.to_string())?.is_like_msvc();
    cfg.define(if msvc { "USE_ALLOCA" } else { "VAR_ARRAYS" }, None);
    let compiler = cfg.try_get_compiler().map_err(|e| e.to_string())?;

    // 1 and 2: the names to prefix.
    let first = compile(&compiler, &files, &out.join("names"), None)?;
    let names = defined_symbols(&compiler, &first)?;
    if names.len() < 300 {
        return Err(format!("only {} symbols found in the first build", names.len()));
    }
    let header = out.join("rename.h");
    let defines: String = names.iter().map(|n| format!("#define {n} {PREFIX}{n}\n")).collect();
    fs::write(&header, defines).map_err(|e| format!("{}: {e}", header.display()))?;

    // 3 and 4.
    let shim = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("shim.c");
    files.push(("shim".to_string(), shim));
    let objects = compile(&compiler, &files, &out.join("prefixed"), Some(&header))?;
    let unprefixed: Vec<String> = defined_symbols(&compiler, &objects)?
        .into_iter()
        .filter(|n| !n.starts_with(PREFIX) && !n.starts_with(SHIM_PREFIX))
        .collect();
    if !unprefixed.is_empty() {
        return Err(format!("symbols without the prefix: {}", unprefixed.join(" ")));
    }
    let library = out.join(if msvc { format!("{LIB}.lib") } else { format!("lib{LIB}.a") });
    let _ = fs::remove_file(&library);
    let mut archiver = cfg.try_get_archiver().map_err(|e| e.to_string())?;
    if msvc {
        archiver.arg("/NOLOGO").arg(format!("/OUT:{}", library.display()));
    } else {
        archiver.arg("crs").arg(&library);
    }
    run(archiver.args(&objects), "archiver")?;
    println!("cargo::rustc-link-search=native={}", out.display());
    println!("cargo::rustc-link-lib=static={LIB}");
    Ok(())
}

/// The libopus 1.6.1 source tree: `LIBOPUS_SRC_1_6_1` if it is set, else
/// the repository's git-ignored `opus-1.6.1/`, else
/// `target/perf/libopus-1.6.1-src/opus-1.6.1/`, the default place of
/// scripts/android/build-libopus.sh.
fn source_dir() -> Result<PathBuf, String> {
    let repo = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let candidates = match env::var_os("LIBOPUS_SRC_1_6_1") {
        Some(dir) => vec![PathBuf::from(dir)],
        None => vec![repo.join("opus-1.6.1"), repo.join("target/perf/libopus-1.6.1-src/opus-1.6.1")],
    };
    for dir in &candidates {
        let version = dir.join("package_version");
        let Ok(text) = fs::read_to_string(&version) else { continue };
        println!("cargo::rerun-if-changed={}", version.display());
        // The source that the `opus-head-sys` crate vendors is libopus's
        // main branch, which codes other packets.
        if text.trim() != "PACKAGE_VERSION=\"1.6.1\"" {
            return Err(format!("{} is not the libopus 1.6.1 release: {}", dir.display(), text.trim()));
        }
        return Ok(dir.clone());
    }
    let tried: Vec<String> = candidates.iter().map(|d| d.display().to_string()).collect();
    Err(format!(
        "no libopus 1.6.1 source in {}. Extract https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz \
         (SHA-256 6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1) to opus-1.6.1/ in the \
         repository, or set LIBOPUS_SRC_1_6_1 to its folder, then run `cargo clean -p libopus-fixed`",
        tried.join(" or ")
    ))
}

/// The C files of the list `name` in the build list `mk` of libopus, for
/// example `SILK_SOURCES` in `silk_sources.mk`.
fn mk_list(src: &Path, mk: &str, name: &str) -> Result<Vec<String>, String> {
    let path = src.join(mk);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut files, mut inside) = (Vec::new(), false);
    for line in text.lines() {
        let body = if inside {
            line
        } else if let Some(rest) = line.strip_prefix(name).and_then(|r| r.trim_start().strip_prefix('=')) {
            rest
        } else {
            continue;
        };
        files.extend(body.split_whitespace().filter(|w| w.ends_with(".c")).map(String::from));
        // A line that ends with a backslash continues the list.
        inside = body.trim_end().ends_with('\\');
        if !inside {
            break;
        }
    }
    if files.is_empty() {
        return Err(format!("no list {name} in {}", path.display()));
    }
    Ok(files)
}

/// Compiles each `(object name, C file)` of `files` into the folder `dir`,
/// as many at once as there are processors, with `header` included before
/// anything else. Returns the object files.
fn compile(
    compiler: &cc::Tool,
    files: &[(String, PathBuf)],
    dir: &Path,
    header: Option<&Path>,
) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let msvc = compiler.is_like_msvc();
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (mut objects, mut running) = (Vec::new(), Vec::<(PathBuf, Child)>::new());
    for (stem, file) in files {
        let object = dir.join(format!("{stem}.o"));
        let mut cmd = compiler.to_command();
        match (header, msvc) {
            (Some(h), true) => {
                cmd.arg(format!("/FI{}", h.display()));
            }
            (Some(h), false) => {
                cmd.arg("-include").arg(h);
            }
            (None, _) => {}
        }
        if msvc {
            cmd.arg("/c").arg(format!("/Fo{}", object.display())).arg(file);
        } else {
            cmd.arg("-c").arg(file).arg("-o").arg(&object);
        }
        // MSVC prints its errors to stdout, the others to stderr.
        let child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", compiler.path().display()))?;
        running.push((file.clone(), child));
        objects.push(object);
        if running.len() >= jobs {
            finish(running.remove(0))?;
        }
    }
    for job in running {
        finish(job)?;
    }
    Ok(objects)
}

/// Waits for one compiler run of `compile`.
fn finish((file, child): (PathBuf, Child)) -> Result<(), String> {
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if output.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&output.stderr) + String::from_utf8_lossy(&output.stdout);
    Err(format!("{} does not compile: {}", file.display(), err.chars().take(2000).collect::<String>()))
}

/// Runs `cmd`, which `what` names in an error.
fn run(cmd: &mut Command, what: &str) -> Result<String, String> {
    let output = cmd.output().map_err(|e| format!("cannot start the {what}: {e}"))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("the {what} failed: {}", err.chars().take(2000).collect::<String>()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The global symbols that `objects` define, sorted, without the
/// underscore that Apple targets put in front of C names.
fn defined_symbols(compiler: &cc::Tool, objects: &[PathBuf]) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    if compiler.is_like_msvc() {
        // Lines like `00A 00000000 SECT4  notype ()    External     | silk_NSQ_c`.
        let target = env::var("TARGET").unwrap();
        let mut dumpbin = cc::windows_registry::find(&target, "dumpbin.exe").ok_or("dumpbin.exe not found")?;
        let listing = run(dumpbin.arg("/NOLOGO").arg("/SYMBOLS").args(objects), "dumpbin")?;
        for line in listing.lines() {
            if !line.contains(" External ") || line.contains(" UNDEF ") {
                continue;
            }
            if let Some(name) = line.split("| ").nth(1).and_then(|n| n.split_whitespace().next()) {
                names.push(name.to_string());
            }
        }
    } else {
        // `NM`, else the `llvm-nm` next to the compiler (cross compilers
        // such as the Android NDK's), else `nm`.
        let nm = env::var_os("NM").map(PathBuf::from).unwrap_or_else(|| {
            let llvm_nm = compiler.path().with_file_name("llvm-nm");
            if llvm_nm.exists() { llvm_nm } else { PathBuf::from("nm") }
        });
        let apple = env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple");
        // POSIX format: one `name type value size` line per symbol, and a
        // `file.o:` line before the symbols of each file.
        let listing = run(Command::new(&nm).arg("-g").arg("-P").args(objects), "symbol lister (nm)")?;
        for line in listing.lines() {
            let mut words = line.split_whitespace();
            let (Some(name), Some(kind)) = (words.next(), words.next()) else { continue };
            // `U` is a symbol that the file uses but does not define.
            if name.ends_with(':') || !matches!(kind, "T" | "D" | "B" | "R" | "S" | "C" | "V" | "W") {
                continue;
            }
            let name = if apple { name.strip_prefix('_').unwrap_or(name) } else { name };
            names.push(name.to_string());
        }
    }
    // Keep the names of the C source. MSVC also makes global symbols of
    // its own: string literals (`??_C@...`), floating-point constants
    // (`__real@...`) and `__isa_available_default`. The source never names
    // them, so the header cannot and need not rename them; the copies in
    // the two libraries have equal contents and the linker keeps one.
    // Names that begin with two underscores belong to the compiler, and
    // libopus defines none.
    let is_c_name = |n: &str| {
        let mut chars = n.chars();
        chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            && !n.starts_with("__")
    };
    names.retain(|n| is_c_name(n));
    names.sort();
    names.dedup();
    Ok(names)
}
