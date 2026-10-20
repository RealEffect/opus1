//! Signals for the deep PLC tests: the 16 kHz speech fixture, the TSP
//! speech corpus resampled to 16 kHz, and a fixed generator.

// Each test file uses a part of this module.
#![allow(dead_code)]

use opus1::internals::silk::SilkResampler;
use std::path::{Path, PathBuf};

/// The 16-bit samples of a mono WAV file, and its sampling rate.
pub fn read_wav(path: &Path) -> (Vec<i16>, u32) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..4], b"RIFF", "{}", path.display());
    let mut pos = 12;
    let mut rate = 0;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            let channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            let bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            assert_eq!((channels, bits), (1, 16), "{}: mono 16-bit", path.display());
            rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
        } else if id == b"data" {
            let end = (body + len).min(bytes.len());
            let (samples, _) = bytes[body..end].as_chunks::<2>();
            return (samples.iter().map(|&b| i16::from_le_bytes(b)).collect(), rate);
        }
        pos = body + len + (len & 1);
    }
    panic!("{}: no data chunk", path.display());
}

/// `fixtures/answer_16k.wav`: 18 s of speech at 16 kHz.
pub fn fixture() -> Vec<i16> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/answer_16k.wav");
    let (s, rate) = read_wav(&path);
    assert_eq!(rate, 16000);
    s
}

/// The corpus root: `OPUS1_CORPUS`, else `<repo>/corpus`.
pub fn corpus_dir() -> PathBuf {
    match std::env::var_os("OPUS1_CORPUS") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus"),
    }
}

/// The WAV files of the TSP corpus (`corpus/tsp48k/48k/*/*.wav`), sorted,
/// or an empty list when the corpus is absent.
pub fn tsp_files() -> Vec<PathBuf> {
    let root = corpus_dir().join("tsp48k/48k");
    let Ok(dirs) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    for d in dirs.flatten() {
        if let Ok(entries) = std::fs::read_dir(d.path()) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "wav") {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    files
}

/// Every `step`-th file of the TSP corpus (all with `OPUS1_DNN_PLC_ALL=1`),
/// at most `max` of them. Prints a note when the corpus is absent.
pub fn tsp_subset(step: usize, max: usize) -> Vec<PathBuf> {
    let files = tsp_files();
    if files.is_empty() {
        eprintln!(
            "note: no TSP corpus under {} (set OPUS1_CORPUS); only the fixture is used",
            corpus_dir().display()
        );
        return files;
    }
    if std::env::var("OPUS1_DNN_PLC_ALL").is_ok_and(|v| v == "1") {
        return files;
    }
    files.into_iter().step_by(step.max(1)).take(max).collect()
}

/// 48 kHz samples resampled to 16 kHz with SILK's encoder resampler (any
/// fixed resampler would do: both sides of a test get the same samples).
pub fn to_16k(x48: &[i16]) -> Vec<i16> {
    let mut rs = SilkResampler::default();
    assert_eq!(rs.init_for_enc(48000, 16000), 0);
    let mut out = Vec::with_capacity(x48.len() / 3);
    let mut buf = [0i16; 160];
    for chunk in x48.as_chunks::<480>().0 {
        rs.process(&mut buf, chunk, 480);
        out.extend_from_slice(&buf);
    }
    out
}

/// A corpus file at 16 kHz.
pub fn tsp_16k(path: &Path) -> Vec<i16> {
    let (s, rate) = read_wav(path);
    assert_eq!(rate, 48000, "{}", path.display());
    to_16k(&s)
}

/// A fixed generator of numbers.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// Uniform in [0, n).
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next_u32() as u64 * n as u64) >> 32) as u32
    }
}

/// The name of a file for messages: `CA/CA01_01.wav`.
pub fn short_name(p: &Path) -> String {
    let parent = p.parent().and_then(|d| d.file_name()).map(|s| s.to_string_lossy().into_owned());
    let name = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    match parent {
        Some(d) => format!("{d}/{name}"),
        None => name,
    }
}
