//! The built-in sets: groups of conditions, their gates, and their items.
//!
//! - `quick`: about 30 conditions over 11 items that are in the repository
//!   (four 2.5 s TSP excerpts in `fixtures/tsp/` and synthetic signals).
//! - `selftest`: 6 conditions over 4 items, for the degradation self-test.
//! - `tsp`: TSP speech from the corpus (the items of
//!   `examples/eval/sets/tsp_speech.txt`), VoIP mono at 48 kHz, 6 to
//!   128 kb/s, 20 and 60 ms frames, complexity 5 and 10.
//! - `tsp_rates`: the items of `tsp`, VoIP mono at 16 and 24 kHz, 6 to
//!   64 kb/s, 20 and 60 ms frames, complexity 5 and 10.
//! - `music`: synthetic music-like items in stereo at 64 to 160 kb/s.
//! - `baseline`: `tsp` and `music` together.

use crate::corpus::{self, Class, Item, ItemSpec};
use crate::pipeline::{App, Condition, Mode};
use crate::signals;
use crate::stats::{Gate, GateKind};
use std::path::Path;

/// A group of conditions: the item classes it codes and its gates.
pub struct Group {
    pub name: String,
    pub classes: Vec<Class>,
    pub gates: Vec<Gate>,
}

pub struct EvalSet {
    pub name: String,
    pub groups: Vec<Group>,
    pub conditions: Vec<Condition>,
    pub items: Vec<ItemSpec>,
}

impl EvalSet {
    pub fn group(&self, name: &str) -> Option<&Group> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Every (condition, item) pair to run: each condition codes the items
    /// whose class its group lists.
    pub fn jobs(&self, items: &[Item]) -> Vec<(usize, usize)> {
        let mut jobs = Vec::new();
        for (c, cond) in self.conditions.iter().enumerate() {
            let Some(group) = self.group(&cond.group) else { continue };
            for (i, item) in items.iter().enumerate() {
                if group.classes.contains(&item.class) {
                    jobs.push((c, i));
                }
            }
        }
        jobs
    }

    /// Keep the conditions whose name or group contains one of the
    /// comma-separated parts of `filter`.
    pub fn filter_conditions(&mut self, filter: &str) {
        self.conditions.retain(|c| filter.split(',').any(|part| c.name.contains(part) || c.group.contains(part)));
    }

    fn merge(mut self, other: EvalSet, name: &str) -> EvalSet {
        self.name = name.to_string();
        self.groups.extend(other.groups);
        self.conditions.extend(other.conditions);
        self.items.extend(other.items);
        self
    }
}

pub const SET_NAMES: [&str; 6] = ["quick", "selftest", "tsp", "tsp_rates", "music", "baseline"];

/// A built-in set by name.
pub fn builtin(name: &str, corpus_root: &Path) -> Result<EvalSet, String> {
    match name {
        "quick" => Ok(quick()),
        "selftest" => Ok(selftest()),
        "tsp" => tsp(corpus_root),
        "tsp_rates" => tsp_rates(corpus_root),
        "music" => Ok(music()),
        "baseline" => Ok(tsp(corpus_root)?.merge(music(), "baseline")),
        _ => Err(format!("unknown set {name}; the sets are {}", SET_NAMES.join(", "))),
    }
}

/// The gates of a group: the defaults of [`default_gates`], with the
/// limits of [`CALIBRATED`] in place of the defaults they cover.
pub fn gates(group: &str) -> Vec<Gate> {
    let mut gates = default_gates(group);
    for &(g, kind, bitrate, limit) in CALIBRATED {
        if g != group {
            continue;
        }
        match gates.iter_mut().find(|x| x.kind == kind && x.bitrate == bitrate) {
            Some(x) => x.limit = limit,
            None => gates.push(Gate { kind, limit, bitrate }),
        }
    }
    gates
}

/// The default gates. The error-ratio limits start from the values of the
/// research notes: a geometric mean of at most 1.20 and a largest ratio of
/// at most 1.50 for CELT at high bitrates, 1.35 and 2.00 for speech. The
/// bytes limits follow section 4.4 of the design note: at most 2 % more
/// than libopus for CELT, and 10 % for frames of 40 ms and more, where
/// opus1 deliberately differs (docs/10). The MOC limits are those of the
/// OSCE draft's relative test (-0.1 per clip, -0.025 on average).
///
/// Every group also gets the gates of `--baseline` (opus1 now against a
/// stored run of opus1): a geometric mean of the error ratio of at most
/// 1.02, a largest ratio of at most 1.10, no change on the libopus side,
/// and, for groups with a `DeltaSnr` gate, a mean loss of signal-to-noise
/// ratio of at most 0.5 dB.
pub fn default_gates(group: &str) -> Vec<Gate> {
    use GateKind::*;
    let err = |mean: f64, max: f64| vec![Gate::new(GeomeanErr, mean), Gate::new(MaxErr, max)];
    let bytes = |max: f64, min: f64| {
        vec![Gate::new(BytesMax, max), Gate::new(BytesMaxLong, 1.10), Gate::new(BytesMin, min), Gate::new(BytesCbr, 0.0)]
    };
    let moc = || vec![Gate::new(MocClip, -0.1), Gate::new(MocMean, -0.025)];
    let waveform = || vec![Gate::new(Align, 0.0), Gate::new(DeltaSnr, -1.0)];
    let mut gates = match group {
        "silk_low" | "tsp_low" => [err(1.35, 2.0), bytes(1.10, 0.90), moc()].concat(),
        "speech_mid" | "tsp_mid" => [err(1.35, 2.0), bytes(1.05, 0.90), moc()].concat(),
        // The alignment gate is left out for speech in VoIP. There the best
        // alignment is often 1 or 2 samples early, for both encoders: the
        // high-pass filter of VoIP shifts the phase of the low frequencies
        // (CELT-only streams too), and SILK output at 24 and 48 kHz comes
        // a sample or two early with any application. CELT-only streams
        // of the other applications are aligned at 0.
        "tsp_high" => [err(1.20, 1.50), bytes(1.02, 0.95), moc(), vec![Gate::new(DeltaSnr, -1.0)]].concat(),
        "celt_mid" | "celt_high" | "music_high" | "lowdelay" => [err(1.20, 1.50), bytes(1.02, 0.95), waveform()].concat(),
        "switch" => [err(1.20, 1.50), bytes(1.02, 0.95)].concat(),
        _ => Vec::new(),
    };
    let snr = gates.iter().any(|g| g.kind == DeltaSnr);
    gates.extend([Gate::new(BaselineGeomean, 1.02), Gate::new(BaselineMax, 1.10), Gate::new(LibopusChanged, 0.0)]);
    if snr {
        gates.push(Gate::new(BaselineSnr, -0.5));
    }
    gates
}

/// Limits set from runs of the current code (U4: start from the research
/// notes' values, then calibrate). Where opus1 at commit d541821 already
/// exceeds a default, the limit lies just above the measured value, so
/// that the gate warns when the gap grows:
///
/// - `GeomeanErr`: 5 % above, rounded up to 0.05;
/// - `MaxErr`: 10 % above, rounded up to 0.1;
/// - `BytesMax` and `BytesMaxLong`: 0.01 above, rounded up to 0.01;
/// - `BytesMin`: 0.01 below, rounded down to 0.01;
/// - `MocClip`: 0.05 below; `MocMean`: 0.01 below.
///
/// The measured values come from the `quick`, `selftest`, `tsp` and
/// `music` sets on the Mac with Homebrew libopus 1.6.1. Each entry: group,
/// rule, bitrate (for `GeomeanErr`), limit.
pub const CALIBRATED: &[(&str, GateKind, Option<i32>, f64)] = {
    use GateKind::*;
    // Measured at c7744f6 (speech and music analysis, no low-frequency
    // burst at 48 kHz VoIP). Only gates whose measured value does not pass
    // the default are listed; each limit leaves about 10 % of room.
    &[
        // The quick and selftest sets.
        ("celt_mid", BytesMax, None, 1.10),             // measured 1.0451 (quiet speech, 48 kb/s mono, c9)
        ("silk_low", BytesMin, None, 0.86),             // measured 0.8874
        ("silk_low", MaxErr, None, 2.4),                // measured 2.1498
        ("silk_low", MocClip, None, -0.15),             // measured -0.1063 (SILK wideband at 48 kHz)
        ("speech_mid", MocClip, None, -0.16),           // measured -0.1208
        // The tsp set.
        ("tsp_low", BytesMaxLong, None, 1.15),          // measured 1.1307 (60 ms, no per-frame limit)
        ("tsp_low", MaxErr, None, 5.2),                 // measured 4.6538
        ("tsp_low", MocClip, None, -0.32),              // measured -0.2914 (FD22_04, 6 kb/s, c5, 60 ms)
        ("tsp_mid", BytesMaxLong, None, 1.15),          // measured 1.1252 (60 ms, no per-frame limit)
        ("tsp_mid", MaxErr, None, 3.6),                 // measured 3.2147
        ("tsp_high", MaxErr, None, 5.5),                // measured 4.9346
        ("tsp_high", MocClip, None, -0.28),             // measured -0.2436
    ]
};

fn group(name: &str, classes: &[Class]) -> Group {
    Group { name: name.to_string(), classes: classes.to_vec(), gates: gates(name) }
}

const SPEECH: &[Class] = &[Class::Speech];
const ALL: &[Class] = &[Class::Speech, Class::Music, Class::Mixed, Class::Synthetic];
const NOT_SPEECH: &[Class] = &[Class::Music, Class::Mixed, Class::Synthetic];

/// The TSP excerpts in `fixtures/tsp/`.
const FIXTURES: [&str; 4] = ["FA01_05", "MD19_04", "FJ55_02", "MH43_04"];

fn fixture(name: &str) -> std::path::PathBuf {
    corpus::fixtures_dir().join("tsp").join(format!("{name}.wav"))
}

fn fixture_item(name: &str) -> ItemSpec {
    ItemSpec::file(format!("tsp/{name}"), Class::Speech, fixture(name), 0.0, 0.0)
}

/// Two excerpts in a row, with the last 0.4 s of every second replaced by
/// digital silence.
fn pauses_item() -> ItemSpec {
    ItemSpec::generated(
        "tsp/pauses",
        Class::Speech,
        Box::new(|| {
            let mut samples = Vec::new();
            for name in &FIXTURES[..2] {
                let wav = signals::read_wav(&fixture(name))?;
                samples.extend(wav.samples.iter().map(|&v| signals::to_i16(v)));
            }
            let second = signals::REF_RATE as usize;
            for (i, s) in samples.iter_mut().enumerate() {
                if i % second >= second * 6 / 10 {
                    *s = 0;
                }
            }
            Ok((1, samples))
        }),
    )
}

/// An excerpt scaled so that its peak lies at -40 dBFS.
fn quiet_item() -> ItemSpec {
    ItemSpec::generated(
        "tsp/quiet",
        Class::Speech,
        Box::new(|| {
            let wav = signals::read_wav(&fixture(FIXTURES[2]))?;
            let peak = wav.samples.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let gain = 0.01 / peak.max(1e-9);
            Ok((1, wav.samples.iter().map(|&v| signals::to_i16(v * gain)).collect()))
        }),
    )
}

fn quick_items() -> Vec<ItemSpec> {
    let mut items: Vec<ItemSpec> = FIXTURES.iter().map(|n| fixture_item(n)).collect();
    items.push(pauses_item());
    items.push(quiet_item());
    items.push(ItemSpec::synth("synth/music_0", Class::Music, || signals::music(0, 6.0)));
    items.push(ItemSpec::synth("synth/castanets", Class::Synthetic, || signals::castanets(4.0)));
    items.push(ItemSpec::synth("synth/tones", Class::Synthetic, || signals::tones(4.0)));
    items.push(ItemSpec::synth("synth/pink_noise", Class::Synthetic, || signals::pink_noise(4.0)));
    items.push(ItemSpec::synth("synth/vowels", Class::Synthetic, || signals::vowels(4.0)));
    items
}

/// The built-in `quick` set. It runs without any download.
pub fn quick() -> EvalSet {
    use App::*;
    let k = |b: i32| b * 1000;
    let mut conditions = vec![
        // SILK, low bitrates.
        Condition::new("silk_low", Voip, 16_000, 1, k(9), 9, 20_000),
        Condition::new("silk_low", Voip, 16_000, 1, k(12), 9, 20_000),
        Condition::new("silk_low", Voip, 16_000, 1, k(16), 9, 20_000),
        Condition::new("silk_low", Voip, 8_000, 1, k(8), 9, 20_000),
        Condition::new("silk_low", Voip, 16_000, 1, k(12), 0, 20_000),
        Condition::new("silk_low", Voip, 16_000, 1, k(12), 5, 20_000),
        Condition::new("silk_low", Voip, 48_000, 1, k(16), 5, 20_000).forced(Mode::Silk).bandwidth(opus1::Bandwidth::Wideband),
        // Speech at middle bitrates. The complexity 5 variants run without
        // libopus's analysis, so they test the rest of the encoder.
        Condition::new("speech_mid", Voip, 16_000, 1, k(24), 9, 20_000),
        Condition::new("speech_mid", Voip, 48_000, 1, k(24), 9, 20_000),
        Condition::new("speech_mid", Voip, 48_000, 1, k(32), 9, 20_000),
        Condition::new("speech_mid", Voip, 48_000, 1, k(24), 5, 20_000),
        Condition::new("speech_mid", Voip, 48_000, 1, k(32), 5, 20_000),
        Condition::new("speech_mid", Voip, 48_000, 1, k(20), 5, 20_000).forced(Mode::Hybrid),
        Condition::new("speech_mid", Voip, 48_000, 2, k(40), 9, 20_000),
        Condition::new("speech_mid", Voip, 16_000, 1, k(24), 9, 20_000).fec(10, 10),
        // CELT.
        Condition::new("celt_mid", Audio, 48_000, 1, k(48), 9, 20_000),
        Condition::new("celt_mid", Audio, 48_000, 1, k(64), 9, 20_000),
        Condition::new("celt_mid", Audio, 48_000, 2, k(64), 9, 20_000),
        Condition::new("celt_mid", Audio, 48_000, 2, k(64), 9, 20_000).cbr(),
        Condition::new("celt_mid", Audio, 48_000, 1, k(32), 9, 20_000).forced(Mode::Celt),
        Condition::new("celt_high", Audio, 48_000, 2, k(96), 9, 20_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(128), 9, 20_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(160), 9, 20_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(128), 0, 20_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(128), 5, 20_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(128), 10, 20_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(128), 9, 10_000),
        Condition::new("celt_high", Audio, 48_000, 2, k(128), 9, 2_500),
        Condition::new("lowdelay", LowDelay, 48_000, 2, k(128), 9, 10_000),
    ];
    conditions.push(
        Condition::new("switch", Audio, 48_000, 2, k(16), 9, 20_000)
            .schedule(vec![(0.0, k(16)), (1.5, k(64)), (3.0, k(24)), (4.5, k(128))]),
    );
    EvalSet {
        name: "quick".into(),
        groups: vec![
            group("silk_low", SPEECH),
            group("speech_mid", SPEECH),
            group("celt_mid", ALL),
            group("celt_high", ALL),
            group("lowdelay", NOT_SPEECH),
            group("switch", &[Class::Music]),
        ],
        conditions,
        items: quick_items(),
    }
}

/// The reduced set of the self-test: 6 conditions, 4 items.
pub fn selftest() -> EvalSet {
    use App::*;
    let k = |b: i32| b * 1000;
    EvalSet {
        name: "selftest".into(),
        groups: vec![
            group("silk_low", SPEECH),
            group("speech_mid", SPEECH),
            group("celt_mid", ALL),
            group("celt_high", ALL),
        ],
        conditions: vec![
            Condition::new("silk_low", Voip, 16_000, 1, k(12), 5, 20_000),
            Condition::new("speech_mid", Voip, 48_000, 1, k(24), 5, 20_000),
            Condition::new("speech_mid", Voip, 48_000, 1, k(32), 5, 20_000),
            Condition::new("celt_mid", Audio, 48_000, 2, k(64), 9, 20_000).cbr(),
            Condition::new("celt_high", Audio, 48_000, 2, k(96), 5, 20_000),
            Condition::new("celt_high", Audio, 48_000, 2, k(128), 9, 20_000),
        ],
        items: vec![
            fixture_item(FIXTURES[0]),
            fixture_item(FIXTURES[1]),
            ItemSpec::synth("synth/music_0_4s", Class::Music, || signals::music(0, 4.0)),
            ItemSpec::synth("synth/castanets", Class::Synthetic, || signals::castanets(4.0)),
        ],
    }
}

/// The committed list of TSP items: 9 files per talker directory, evenly
/// spaced (6 from `CB`, which has 6), 222 in all.
pub fn tsp_set_file() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/eval/sets/tsp_speech.txt")
}

/// The bitrates of the speech baseline, in kb/s.
pub const SPEECH_BITRATES: [i32; 10] = [6, 9, 12, 16, 24, 32, 48, 64, 96, 128];

/// TSP speech: VoIP, mono, 48 kHz.
pub fn tsp(corpus_root: &Path) -> Result<EvalSet, String> {
    tsp_at("tsp", &[48_000], &SPEECH_BITRATES, corpus_root)
}

/// TSP speech: VoIP, mono, 16 and 24 kHz, up to 64 kb/s. These rates are
/// the control for changes that should only affect 48 kHz.
pub fn tsp_rates(corpus_root: &Path) -> Result<EvalSet, String> {
    tsp_at("tsp_rates", &[16_000, 24_000], &SPEECH_BITRATES[..8], corpus_root)
}

/// The items of `tsp_speech.txt` in VoIP mono at each of `rates` and
/// `bitrates` (kb/s), with 20 and 60 ms frames at complexity 5 and 10.
fn tsp_at(name: &str, rates: &[u32], bitrates: &[i32], corpus_root: &Path) -> Result<EvalSet, String> {
    let items = corpus::read_set_file(&tsp_set_file(), corpus_root)?;
    let mut conditions = Vec::new();
    for &sample_rate in rates {
        for &frame in &[20_000, 60_000] {
            for &complexity in &[5, 10] {
                for &rate in bitrates {
                    let group = match rate {
                        0..=16 => "tsp_low",
                        17..=48 => "tsp_mid",
                        _ => "tsp_high",
                    };
                    conditions.push(Condition::new(group, App::Voip, sample_rate, 1, rate * 1000, complexity, frame));
                }
            }
        }
    }
    Ok(EvalSet {
        name: name.into(),
        groups: vec![group("tsp_low", SPEECH), group("tsp_mid", SPEECH), group("tsp_high", SPEECH)],
        conditions,
        items,
    })
}

/// The bitrates of the music baseline, in kb/s.
pub const MUSIC_BITRATES: [i32; 4] = [64, 96, 128, 160];

/// Synthetic music-like items in stereo: 8 seeds of the music generator,
/// castanets, tones and pink noise, 6 s each. Audio application, 48 kHz.
pub fn music() -> EvalSet {
    let mut items = Vec::new();
    for seed in 1..=8u64 {
        items.push(ItemSpec::synth(&format!("synth/music_{seed}"), Class::Music, move || signals::music(seed, 6.0)));
    }
    items.push(ItemSpec::synth("synth/castanets_6s", Class::Synthetic, || signals::castanets(6.0)));
    items.push(ItemSpec::synth("synth/tones_6s", Class::Synthetic, || signals::tones(6.0)));
    items.push(ItemSpec::synth("synth/pink_noise_6s", Class::Synthetic, || signals::pink_noise(6.0)));
    let mut conditions = Vec::new();
    for &frame in &[20_000, 60_000] {
        for &complexity in &[5, 10] {
            for &rate in &MUSIC_BITRATES {
                conditions.push(Condition::new("music_high", App::Audio, 48_000, 2, rate * 1000, complexity, frame));
            }
        }
    }
    EvalSet { name: "music".into(), groups: vec![group("music_high", NOT_SPEECH)], conditions, items }
}
