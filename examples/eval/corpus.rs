//! Items and corpora: where an item comes from, how it is loaded as a
//! 48 kHz 16-bit reference, set files, the tuning and hold-out split, and
//! the `corpus-index` command.

use crate::signals::{self, REF_RATE};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// What kind of audio an item holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    Speech,
    Music,
    Mixed,
    Synthetic,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Speech => "speech",
            Class::Music => "music",
            Class::Mixed => "mixed",
            Class::Synthetic => "synthetic",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "speech" => Some(Class::Speech),
            "music" => Some(Class::Music),
            "mixed" => Some(Class::Mixed),
            "synthetic" => Some(Class::Synthetic),
            _ => None,
        }
    }
}

/// Tuning work may look at the `Tune` items only. The `Holdout` items are
/// reported separately and serve to check a change afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Tune,
    Holdout,
}

impl Split {
    pub fn name(self) -> &'static str {
        match self {
            Split::Tune => "tune",
            Split::Holdout => "holdout",
        }
    }
}

/// 64-bit FNV-1a (Fowler-Noll-Vo) hash.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// An item is in the hold-out split when the FNV-1a hash of its name is a
/// multiple of 4 (about a quarter of the items).
pub fn split_of(name: &str) -> Split {
    if fnv1a64(name.as_bytes()).is_multiple_of(4) { Split::Holdout } else { Split::Tune }
}

/// Builds the samples of a generated item: the channel count and the
/// interleaved 16-bit samples at 48 kHz.
pub type Make = Box<dyn Fn() -> Result<(usize, Vec<i16>), String> + Send + Sync>;

pub enum Source {
    /// Part of a WAV file. `duration` 0 means up to the end.
    File { path: PathBuf, start: f64, duration: f64 },
    Generated(Make),
}

/// How to build one item.
pub struct ItemSpec {
    pub name: String,
    pub class: Class,
    pub source: Source,
}

impl ItemSpec {
    pub fn file(name: String, class: Class, path: PathBuf, start: f64, duration: f64) -> Self {
        Self { name, class, source: Source::File { path, start, duration } }
    }

    /// A synthetic item from channels of f64 samples at 48 kHz, scaled to a
    /// peak of -6 dBFS.
    pub fn synth(name: &str, class: Class, make: impl Fn() -> Vec<Vec<f64>> + Send + Sync + 'static) -> Self {
        Self {
            name: name.to_string(),
            class,
            source: Source::Generated(Box::new(move || {
                let channels = make();
                Ok((channels.len(), signals::finish(channels)))
            })),
        }
    }

    pub fn generated(name: &str, class: Class, make: Make) -> Self {
        Self { name: name.to_string(), class, source: Source::Generated(make) }
    }
}

/// A loaded item: the reference signal of the comparisons.
pub struct Item {
    pub name: String,
    pub class: Class,
    pub split: Split,
    pub channels: usize,
    /// Interleaved 16-bit samples at 48 kHz. The length per channel is a
    /// multiple of 12, so that it divides into whole samples at every rate
    /// `opus_compare` accepts (48000 / R is 1, 2, 3, 4 or 6).
    pub reference: Vec<i16>,
    moc_reference: OnceLock<Vec<f32>>,
}

impl Item {
    pub fn new(name: String, class: Class, channels: usize, mut reference: Vec<i16>) -> Self {
        let frames = reference.len() / channels / 12 * 12;
        reference.truncate(frames * channels);
        Self { split: split_of(&name), name, class, channels, reference, moc_reference: OnceLock::new() }
    }

    /// Samples per channel at 48 kHz.
    pub fn frames(&self) -> usize {
        self.reference.len() / self.channels
    }

    pub fn seconds(&self) -> f64 {
        self.frames() as f64 / REF_RATE as f64
    }

    /// The reference for MOC: the mono mix `(L+R)/2` resampled to 16 kHz,
    /// computed once.
    pub fn moc_reference(&self) -> &[f32] {
        self.moc_reference.get_or_init(|| {
            let mono = self.mono_f64();
            signals::Resampler::new(REF_RATE, 16_000).run(&mono).iter().map(|&v| v as f32).collect()
        })
    }

    /// One channel at 48 kHz, as f64 in [-1, 1): the mix `(L+R)/2` for a
    /// stereo item. The values are exact.
    pub fn mono_f64(&self) -> Vec<f64> {
        if self.channels == 1 {
            self.reference.iter().map(|&s| s as f64 / 32768.0).collect()
        } else {
            self.reference
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| 0.5 * (p[0] as f64 + p[1] as f64) / 32768.0)
                .collect()
        }
    }
}

/// Load one item. A file at another rate is resampled to 48 kHz.
pub fn load(spec: &ItemSpec) -> Result<Item, String> {
    let (channels, samples) = match &spec.source {
        Source::Generated(make) => make()?,
        Source::File { path, start, duration } => {
            let wav = signals::read_wav(path)?;
            let mut samples = if wav.rate == REF_RATE {
                wav.samples
            } else {
                signals::resample_interleaved(&wav.samples, wav.channels, wav.rate, REF_RATE)
            };
            let frames = samples.len() / wav.channels;
            let first = ((start * REF_RATE as f64).round() as usize).min(frames);
            let last = if *duration > 0.0 {
                (first + (duration * REF_RATE as f64).round() as usize).min(frames)
            } else {
                frames
            };
            samples.truncate(last * wav.channels);
            samples.drain(..first * wav.channels);
            (wav.channels, samples.iter().map(|&v| signals::to_i16(v)).collect())
        }
    };
    if !(1..=2).contains(&channels) {
        return Err(format!("{}: {channels} channels", spec.name));
    }
    let item = Item::new(spec.name.clone(), spec.class, channels, samples);
    if item.frames() < 4800 {
        return Err(format!("{}: shorter than 0.1 s", spec.name));
    }
    Ok(item)
}

/// Load all items, in parallel. The order of the result is the order of
/// `specs`.
pub fn load_all(specs: &[ItemSpec], threads: usize) -> Result<Vec<Item>, String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<Item, String>>>> =
        specs.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1).min(specs.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let item = load(&specs[i]);
                    *slots[i].lock().unwrap() = Some(item);
                }
            });
        }
    });
    slots.into_iter().map(|s| s.into_inner().unwrap().expect("every item loaded")).collect()
}

/// The corpus root: `OPUS1_CORPUS`, else `<repository>/corpus`.
pub fn corpus_root() -> PathBuf {
    match std::env::var_os("OPUS1_CORPUS") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus"),
    }
}

/// The directory of the committed fixtures.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Read a set file: one item per line, tab-separated
/// `path class start_s duration_s`, where the last two are optional
/// (default: the whole file). `#` starts a comment. A relative path is
/// relative to `root`.
pub fn read_set_file(path: &Path, root: &Path) -> Result<Vec<ItemSpec>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_set(&text, root).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse_set(text: &str, root: &Path) -> Result<Vec<ItemSpec>, String> {
    let mut specs = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
        let bad = |what: &str| format!("line {}: {what}", n + 1);
        let name = fields[0];
        let class = fields.get(1).map_or(Some(Class::Speech), |c| Class::parse(c)).ok_or_else(|| bad("unknown class"))?;
        let number = |i: usize| -> Result<f64, String> {
            fields.get(i).map_or(Ok(0.0), |v| v.parse::<f64>().map_err(|_| bad("bad number")))
        };
        let (start, duration) = (number(2)?, number(3)?);
        let path = if Path::new(name).is_absolute() { PathBuf::from(name) } else { root.join(name) };
        specs.push(ItemSpec::file(name.to_string(), class, path, start, duration));
    }
    Ok(specs)
}

/// Every `.wav` file under `dir`, sorted.
fn wav_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            wav_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")) {
            out.push(path);
        }
    }
    Ok(())
}

/// The text of a set file for the WAV files under `dir`: grouped by their
/// directory (one talker per directory in the TSP corpus), sorted by name,
/// `per_group` files from each group, evenly spaced through the group.
/// Paths are written relative to `root` when they lie under it.
pub fn corpus_index(dir: &Path, root: &Path, per_group: usize, class: Class) -> Result<String, String> {
    let mut files = Vec::new();
    wav_files(dir, &mut files)?;
    let mut groups: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    for f in files {
        groups.entry(f.parent().map(Path::to_path_buf).unwrap_or_default()).or_default().push(f);
    }
    let mut text = format!(
        "# Written by `cargo run --release --example eval -- corpus-index`.\n\
         # {} group(s) under {}, {per_group} file(s) per group, evenly spaced.\n\
         # path\tclass\tstart_s\tduration_s\n",
        groups.len(),
        dir.strip_prefix(root).unwrap_or(dir).display()
    );
    for files in groups.values() {
        let take = per_group.min(files.len());
        for k in 0..take {
            let f = &files[k * files.len() / take];
            let shown = f.strip_prefix(root).unwrap_or(f);
            text.push_str(&format!("{}\t{}\t0\t0\n", shown.display(), class.name()));
        }
    }
    Ok(text)
}

/// FNV-1a hash of the item names and their samples, so that two runs can
/// tell whether they used the same items.
pub fn items_hash(items: &[Item]) -> u64 {
    let mut bytes = Vec::new();
    for item in items {
        bytes.extend_from_slice(item.name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&fnv1a64(&item.reference.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>()).to_le_bytes());
    }
    fnv1a64(&bytes)
}
