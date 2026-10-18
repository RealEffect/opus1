//! One job of the paired comparison: encode an item under one condition
//! with opus1 and with libopus, decode both packet streams with the same
//! decoder, align, and measure.
//!
//! The steps of one side:
//!
//! 1. The encoder input is the item at the condition's rate and channel
//!    count ([`encoder_input`]). At 48 kHz it is the 16-bit reference
//!    itself, scaled by 1/32768 (exact). A mono condition on a stereo item
//!    codes `(L+R)/2`, which is also what `opus_compare` compares with.
//! 2. The encoder codes the input followed by enough frames of zeros that
//!    the decoder outputs the whole item after the encoder's look-ahead.
//! 3. The `opus_demo -d` loop of `opus1::internals::eval::demo` decodes the
//!    packets, with libopus unless `--decoder opus1` is given. Packets of a
//!    condition with a decode loss are dropped first, with a fixed seed,
//!    and recovered with FEC or concealed as `opus_demo` does. The loop
//!    checks the encoder's final range after every packet it decodes
//!    normally, so a packet that libopus reads differently from how opus1
//!    wrote it stops the job.
//! 4. The first `lookahead` samples are dropped; the rest is cut to the
//!    length of the input.
//! 5. Measures: `opus_compare` against the 48 kHz reference, the
//!    signal-to-noise ratio against the encoder input, the best time shift
//!    within 1 ms, and MOC for speech in mono. MOC is defined at 16 kHz:
//!    its reference is the 48 kHz reference resampled to 16 kHz, and a
//!    decoded signal at another rate is resampled to 16 kHz first.

use crate::corpus::{Class, Item};
use crate::degrade::Degrade;
use crate::libopus::{self, CDecoder, CEncoder};
use crate::signals::{self, REF_RATE, XorShift};
use audiopus_sys as ffi;
use opus1::internals::eval::compare::{Reference, opus_compare};
use opus1::internals::eval::demo::{BitPacket, decode_bitstream};
use opus1::internals::eval::moc::{moc, moc_relative};
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusDecoder, OpusEncoder};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Room for one packet: the largest Opus packet is 1275 bytes per 20 ms
/// frame, and the longest condition here is 120 ms.
const MAX_PACKET: usize = 1276 * 6;

/// The seed of the packets dropped before decoding.
const LOSS_SEED: u64 = 0x105E_0000_DEC0_DE01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum App {
    Voip,
    Audio,
    LowDelay,
}

impl App {
    pub fn name(self) -> &'static str {
        match self {
            App::Voip => "voip",
            App::Audio => "audio",
            App::LowDelay => "lowdelay",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Silk,
    Hybrid,
    Celt,
}

/// The short name of a forced bandwidth in a condition's name.
fn bandwidth_name(bw: Bandwidth) -> &'static str {
    match bw {
        Bandwidth::Auto => "auto",
        Bandwidth::Narrowband => "nb",
        Bandwidth::Mediumband => "mb",
        Bandwidth::Wideband => "wb",
        Bandwidth::Superwideband => "swb",
        Bandwidth::Fullband => "fb",
    }
}

/// Which codec fills a slot of the pair. The identity checks of the
/// self-test put the same codec in both slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Opus1,
    Libopus,
}

impl Codec {
    pub fn name(self) -> &'static str {
        match self {
            Codec::Opus1 => "opus1",
            Codec::Libopus => "libopus",
        }
    }
}

/// One set of encoder and decoder settings.
#[derive(Clone, Debug)]
pub struct Condition {
    pub name: String,
    pub group: String,
    pub app: App,
    pub rate: u32,
    pub channels: usize,
    /// Bits per second.
    pub bitrate: i32,
    pub cbr: bool,
    pub complexity: i32,
    pub frame_us: u32,
    pub fec: bool,
    pub loss_perc: i32,
    /// Packets dropped before decoding, in percent (fixed seed). The first
    /// packet and the last two are never dropped.
    pub decode_loss: u32,
    /// A forced coding mode, or `None` to let each encoder choose.
    pub mode: Option<Mode>,
    /// A forced audio bandwidth, or `None` to let each encoder choose.
    /// libopus codes a forced SILK mode as Hybrid when the bandwidth is
    /// wider than wideband, so SILK at 24 and 48 kHz needs both.
    pub bandwidth: Option<Bandwidth>,
    /// Bitrate changes: from second `.0` on, the bitrate is `.1`. Empty for
    /// a fixed bitrate.
    pub schedule: Vec<(f64, i32)>,
    pub tags: Vec<&'static str>,
}

impl Condition {
    pub fn new(group: &str, app: App, rate: u32, channels: usize, bitrate: i32, complexity: i32, frame_us: u32) -> Self {
        let mut c = Self {
            name: String::new(),
            group: group.to_string(),
            app,
            rate,
            channels,
            bitrate,
            cbr: false,
            complexity,
            frame_us,
            fec: false,
            loss_perc: 0,
            decode_loss: 0,
            mode: None,
            bandwidth: None,
            schedule: Vec::new(),
            tags: Vec::new(),
        };
        c.rename();
        c
    }

    pub fn cbr(mut self) -> Self {
        self.cbr = true;
        self.rename();
        self
    }

    /// In-band FEC with `loss_perc` percent expected loss, and `decode_loss`
    /// percent of the packets dropped before decoding.
    pub fn fec(mut self, loss_perc: i32, decode_loss: u32) -> Self {
        self.fec = true;
        self.loss_perc = loss_perc;
        self.decode_loss = decode_loss;
        self.rename();
        self
    }

    pub fn forced(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self.rename();
        self
    }

    pub fn bandwidth(mut self, bandwidth: Bandwidth) -> Self {
        self.bandwidth = Some(bandwidth);
        self.rename();
        self
    }

    pub fn schedule(mut self, schedule: Vec<(f64, i32)>) -> Self {
        self.schedule = schedule;
        self.rename();
        self
    }

    /// Samples per channel in one frame.
    pub fn frame(&self) -> usize {
        (self.rate as u64 * self.frame_us as u64 / 1_000_000) as usize
    }

    /// The bitrate at second `t`.
    pub fn bitrate_at(&self, t: f64) -> i32 {
        self.schedule.iter().rev().find(|(start, _)| *start <= t).map_or(self.bitrate, |&(_, b)| b)
    }

    /// libopus's float build runs its signal analysis (`analysis.c`) at
    /// complexity 7 and above when the rate is at least 16 kHz; opus1 does
    /// not have it yet.
    pub fn runs_analysis(&self) -> bool {
        self.complexity >= 7 && self.rate >= 16_000 && self.app != App::LowDelay
    }

    fn rename(&mut self) {
        let frame = if self.frame_us.is_multiple_of(1000) {
            format!("{}ms", self.frame_us / 1000)
        } else {
            format!("{}ms", self.frame_us as f64 / 1000.0)
        };
        let mut name = format!(
            "{}_{}k_{}_{}k_{}_c{}_{}",
            self.app.name(),
            self.rate / 1000,
            if self.channels == 2 { "stereo" } else { "mono" },
            self.bitrate as f64 / 1000.0,
            if self.cbr { "cbr" } else { "vbr" },
            self.complexity,
            frame
        );
        if self.fec {
            name += &format!("_fec{}", self.loss_perc);
        }
        if self.decode_loss > 0 {
            name += &format!("_loss{}", self.decode_loss);
        }
        if let Some(mode) = self.mode {
            name += match mode {
                Mode::Silk => "_silk",
                Mode::Hybrid => "_hybrid",
                Mode::Celt => "_celt",
            };
        }
        if let Some(bw) = self.bandwidth {
            name += "_";
            name += bandwidth_name(bw);
        }
        if !self.schedule.is_empty() {
            name += "_switch";
        }
        self.name = name;
        self.tags = if self.runs_analysis() { vec!["analysis"] } else { Vec::new() };
    }
}

/// A packet stream with the encoder's final range after each packet.
#[derive(Default)]
pub struct Packets {
    data: Vec<u8>,
    spans: Vec<(usize, usize)>,
    ranges: Vec<u32>,
    pub lookahead: usize,
}

impl Packets {
    fn push(&mut self, packet: &[u8], range: u32) {
        self.spans.push((self.data.len(), packet.len()));
        self.data.extend_from_slice(packet);
        self.ranges.push(range);
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn get(&self, i: usize) -> &[u8] {
        let (offset, len) = self.spans[i];
        &self.data[offset..offset + len]
    }

    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    /// How often each configuration occurs, as
    /// "mode/bandwidth/frame/channels xN" from the first byte of each packet.
    pub fn toc_label(&self) -> String {
        let mut histogram: BTreeMap<u8, usize> = BTreeMap::new();
        for i in 0..self.len() {
            let key = self.get(i).first().map_or(0xFF, |toc| toc >> 2);
            *histogram.entry(key).or_insert(0) += 1;
        }
        let parts: Vec<String> = histogram.iter().map(|(&k, &n)| format!("{} x{n}", toc_name(k))).collect();
        parts.join(", ")
    }
}

/// "mode/bandwidth/duration/channels" for the upper six bits of a TOC byte.
fn toc_name(key: u8) -> String {
    if key == 0xFF {
        return "empty".to_string();
    }
    let config = (key >> 1) as usize;
    let channels = if key & 1 == 1 { "stereo" } else { "mono" };
    const SILK_MS: [&str; 4] = ["10ms", "20ms", "40ms", "60ms"];
    const CELT_MS: [&str; 4] = ["2.5ms", "5ms", "10ms", "20ms"];
    let (mode, bandwidth, duration) = match config {
        0..=3 => ("silk", "nb", SILK_MS[config & 3]),
        4..=7 => ("silk", "mb", SILK_MS[config & 3]),
        8..=11 => ("silk", "wb", SILK_MS[config & 3]),
        12..=13 => ("hybrid", "swb", SILK_MS[config & 1]),
        14..=15 => ("hybrid", "fb", SILK_MS[config & 1]),
        16..=19 => ("celt", "nb", CELT_MS[config & 3]),
        20..=23 => ("celt", "wb", CELT_MS[config & 3]),
        24..=27 => ("celt", "swb", CELT_MS[config & 3]),
        _ => ("celt", "fb", CELT_MS[config & 3]),
    };
    format!("{mode}/{bandwidth}/{duration}/{channels}")
}

/// The encoder input of an item for a rate and channel count, interleaved,
/// scaled to [-1, 1).
pub fn encoder_input(item: &Item, rate: u32, channels: usize) -> Vec<f32> {
    let parts: Vec<Vec<f64>> = match (item.channels, channels) {
        (2, 1) => vec![item.mono_f64()],
        (1, 2) => {
            let m = item.mono_f64();
            vec![m.clone(), m]
        }
        _ => {
            let all: Vec<f64> = item.reference.iter().map(|&s| s as f64 / 32768.0).collect();
            signals::deinterleave(&all, item.channels)
        }
    };
    let parts: Vec<Vec<f64>> = if rate == REF_RATE {
        parts
    } else {
        let resampler = signals::Resampler::new(REF_RATE, rate);
        parts.iter().map(|p| resampler.run(p)).collect()
    };
    signals::interleave(&parts).iter().map(|&v| v as f32).collect()
}

/// The encoder inputs of a run, computed once per (item, rate, channels).
pub struct Inputs {
    map: HashMap<(usize, u32, usize), OnceLock<Vec<f32>>>,
}

impl Inputs {
    pub fn new(keys: impl IntoIterator<Item = (usize, u32, usize)>) -> Self {
        Self { map: keys.into_iter().map(|k| (k, OnceLock::new())).collect() }
    }

    pub fn get(&self, items: &[Item], item: usize, rate: u32, channels: usize) -> &[f32] {
        self.map
            .get(&(item, rate, channels))
            .expect("input registered")
            .get_or_init(|| encoder_input(&items[item], rate, channels))
    }
}

/// The encoder of one side.
enum Encoder {
    Opus1(Box<OpusEncoder>),
    Libopus(CEncoder),
}

impl Encoder {
    fn new(codec: Codec, cond: &Condition, degrade: Option<&Degrade>) -> Result<Self, String> {
        match codec {
            Codec::Opus1 => {
                let application = match cond.app {
                    App::Voip => Application::Voip,
                    App::Audio => Application::Audio,
                    App::LowDelay => Application::RestrictedLowDelay,
                };
                let mut e = OpusEncoder::new(cond.rate as i32, cond.channels, application)
                    .map_err(|e| format!("OpusEncoder::new: {e}"))?;
                e.set_complexity(cond.complexity).map_err(|e| format!("set_complexity: {e}"))?;
                e.set_vbr(!cond.cbr);
                e.set_inband_fec(cond.fec);
                e.set_packet_loss_perc(cond.loss_perc).map_err(|e| format!("set_packet_loss_perc: {e}"))?;
                if let Some(mode) = cond.mode {
                    e.set_force_mode(Some(match mode {
                        Mode::Silk => ForcedMode::SilkOnly,
                        Mode::Hybrid => ForcedMode::Hybrid,
                        Mode::Celt => ForcedMode::CeltOnly,
                    }));
                }
                if let Some(bw) = cond.bandwidth {
                    e.set_bandwidth(bw);
                }
                if let Some(mask) = degrade.and_then(|d| d.energy_mask(cond)) {
                    e.set_energy_mask(Some(&mask));
                }
                Ok(Encoder::Opus1(Box::new(e)))
            }
            Codec::Libopus => {
                if degrade.is_some_and(|d| d.energy_mask(cond).is_some()) {
                    return Err("the energy mask degradation applies to opus1 only".into());
                }
                let mut e = CEncoder::new(cond.rate, cond.channels, libopus_application(cond.app))?;
                e.ctl(ffi::OPUS_SET_COMPLEXITY_REQUEST, cond.complexity)?;
                e.ctl(ffi::OPUS_SET_VBR_REQUEST, i32::from(!cond.cbr))?;
                e.ctl(ffi::OPUS_SET_INBAND_FEC_REQUEST, i32::from(cond.fec))?;
                e.ctl(ffi::OPUS_SET_PACKET_LOSS_PERC_REQUEST, cond.loss_perc)?;
                if let Some(mode) = cond.mode {
                    e.set_force_mode(match mode {
                        Mode::Silk => libopus::MODE_SILK_ONLY,
                        Mode::Hybrid => libopus::MODE_HYBRID,
                        Mode::Celt => libopus::MODE_CELT_ONLY,
                    })?;
                }
                if let Some(bw) = cond.bandwidth {
                    // opus1's values are libopus's `OPUS_BANDWIDTH_*`.
                    e.ctl(ffi::OPUS_SET_BANDWIDTH_REQUEST, bw as i32)?;
                }
                Ok(Encoder::Libopus(e))
            }
        }
    }

    fn set_bitrate(&mut self, bitrate: i32) -> Result<(), String> {
        match self {
            Encoder::Opus1(e) => e.set_bitrate(Bitrate::BitsPerSecond(bitrate)).map_err(|e| format!("set_bitrate: {e}")),
            Encoder::Libopus(e) => e.ctl(ffi::OPUS_SET_BITRATE_REQUEST, bitrate),
        }
    }

    fn lookahead(&self) -> Result<usize, String> {
        match self {
            Encoder::Opus1(e) => Ok(e.lookahead()),
            Encoder::Libopus(e) => Ok(e.get(ffi::OPUS_GET_LOOKAHEAD_REQUEST)? as usize),
        }
    }

    fn encode(&mut self, pcm: &[f32], frame: usize, out: &mut [u8]) -> Result<usize, String> {
        match self {
            Encoder::Opus1(e) => e.encode(pcm, frame, out).map_err(|e| format!("encode: {e}")),
            Encoder::Libopus(e) => e.encode(pcm, frame, out),
        }
    }

    fn final_range(&self) -> u32 {
        match self {
            Encoder::Opus1(e) => e.final_range(),
            Encoder::Libopus(e) => e.final_range(),
        }
    }

    /// The settings as the encoder reports them.
    fn settings(&self, cond: &Condition) -> Result<Settings, String> {
        Ok(match self {
            Encoder::Opus1(e) => Settings {
                // opus1 has no getter for the application; it is the one
                // `Encoder::new` passed.
                application: libopus_application(cond.app),
                sample_rate: e.sampling_rate(),
                channels: e.channels() as i32,
                bitrate: e.bitrate(),
                vbr: e.vbr(),
                vbr_constraint: e.vbr_constraint(),
                complexity: e.complexity(),
                inband_fec: i32::from(e.inband_fec()),
                packet_loss_perc: e.packet_loss_perc(),
                force_channels: e.force_channels().map_or(OPUS_AUTO, |c| c as i32),
                expert_frame_duration: e.expert_frame_duration() as i32,
                prediction_disabled: e.prediction_disabled(),
                lookahead: e.lookahead() as i32,
                // opus1 has no control for these. It always behaves as
                // libopus does with its defaults: no limit below fullband,
                // no signal hint, 24-bit depth for float input, no
                // discontinuous transmission, phase inversion on.
                max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
                signal: OPUS_AUTO,
                lsb_depth: 24,
                dtx: false,
                phase_inversion_disabled: false,
            },
            Encoder::Libopus(e) => Settings {
                application: e.get(ffi::OPUS_GET_APPLICATION_REQUEST)?,
                sample_rate: e.get(ffi::OPUS_GET_SAMPLE_RATE_REQUEST)?,
                channels: e.channels() as i32,
                bitrate: e.get(ffi::OPUS_GET_BITRATE_REQUEST)?,
                vbr: e.get(ffi::OPUS_GET_VBR_REQUEST)? != 0,
                vbr_constraint: e.get(ffi::OPUS_GET_VBR_CONSTRAINT_REQUEST)? != 0,
                complexity: e.get(ffi::OPUS_GET_COMPLEXITY_REQUEST)?,
                inband_fec: e.get(ffi::OPUS_GET_INBAND_FEC_REQUEST)?,
                packet_loss_perc: e.get(ffi::OPUS_GET_PACKET_LOSS_PERC_REQUEST)?,
                force_channels: e.get(ffi::OPUS_GET_FORCE_CHANNELS_REQUEST)?,
                expert_frame_duration: e.get(ffi::OPUS_GET_EXPERT_FRAME_DURATION_REQUEST)?,
                prediction_disabled: e.get(ffi::OPUS_GET_PREDICTION_DISABLED_REQUEST)? != 0,
                lookahead: e.get(ffi::OPUS_GET_LOOKAHEAD_REQUEST)?,
                max_bandwidth: e.get(ffi::OPUS_GET_MAX_BANDWIDTH_REQUEST)?,
                signal: e.get(ffi::OPUS_GET_SIGNAL_REQUEST)?,
                lsb_depth: e.get(ffi::OPUS_GET_LSB_DEPTH_REQUEST)?,
                dtx: e.get(ffi::OPUS_GET_DTX_REQUEST)? != 0,
                phase_inversion_disabled: e.get(ffi::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST)? != 0,
            },
        })
    }
}

/// libopus `OPUS_AUTO`.
const OPUS_AUTO: i32 = -1000;
/// libopus `OPUS_BANDWIDTH_FULLBAND`.
const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

fn libopus_application(app: App) -> i32 {
    match app {
        App::Voip => ffi::OPUS_APPLICATION_VOIP,
        App::Audio => ffi::OPUS_APPLICATION_AUDIO,
        App::LowDelay => ffi::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    }
}

/// The encoder settings that the two sides of a pair must share, as each
/// encoder reports them. The forced mode and the forced bandwidth are not
/// in it: libopus has no request that reads them back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    pub application: i32,
    pub sample_rate: i32,
    pub channels: i32,
    /// `OPUS_GET_BITRATE` and `OpusEncoder::bitrate`.
    pub bitrate: i32,
    pub vbr: bool,
    pub vbr_constraint: bool,
    pub complexity: i32,
    pub inband_fec: i32,
    pub packet_loss_perc: i32,
    /// -1000 (`OPUS_AUTO`) when not forced.
    pub force_channels: i32,
    pub expert_frame_duration: i32,
    pub prediction_disabled: bool,
    pub lookahead: i32,
    pub max_bandwidth: i32,
    pub signal: i32,
    pub lsb_depth: i32,
    pub dtx: bool,
    pub phase_inversion_disabled: bool,
}

/// The settings of the encoder of one side, set up for `cond` as
/// [`encode`] sets it up for its first frame (without a degradation).
pub fn encoder_settings(codec: Codec, cond: &Condition) -> Result<Settings, String> {
    let mut encoder = Encoder::new(codec, cond, None)?;
    encoder.set_bitrate(cond.bitrate_at(0.0))?;
    encoder.settings(cond)
}

/// Check that opus1's and libopus's encoders report the same settings for
/// each condition, before any job runs.
pub fn check_settings(conditions: &[Condition]) -> Result<(), String> {
    for cond in conditions {
        let opus1 = encoder_settings(Codec::Opus1, cond)?;
        let libopus = encoder_settings(Codec::Libopus, cond)?;
        if opus1 != libopus {
            return Err(format!("{}: the encoders differ in their settings: opus1 {opus1:?}, libopus {libopus:?}", cond.name));
        }
    }
    Ok(())
}

/// Encode `input` (interleaved, at the condition's rate and channel count)
/// followed by zeros, so that the decoded output covers the input after the
/// look-ahead.
pub fn encode(codec: Codec, cond: &Condition, input: &[f32], degrade: Option<&Degrade>) -> Result<Packets, String> {
    let mut encoder = Encoder::new(codec, cond, degrade)?;
    let scale = degrade.map_or(1.0, |d| d.bitrate_scale());
    let frame = cond.frame();
    let c = cond.channels;
    let n = input.len() / c;
    let lookahead = encoder.lookahead()?;
    let frames = (n + lookahead).div_ceil(frame) + 1;
    let mut padded = input.to_vec();
    padded.resize(frames * frame * c, 0.0);
    let mut out = vec![0u8; MAX_PACKET];
    let mut packets = Packets { lookahead, ..Packets::default() };
    let mut current = None;
    for (k, pcm) in padded.chunks_exact(frame * c).enumerate() {
        let t = (k * frame) as f64 / cond.rate as f64;
        let bitrate = (cond.bitrate_at(t) as f64 * scale).round() as i32;
        if current != Some(bitrate) {
            encoder.set_bitrate(bitrate)?;
            current = Some(bitrate);
        }
        let len = encoder.encode(pcm, frame, &mut out)?;
        packets.push(&out[..len], encoder.final_range());
    }
    Ok(packets)
}

/// Statistics of one decoded packet stream.
#[derive(Clone, Debug)]
pub struct Stream {
    pub bytes: usize,
    pub packets: usize,
    pub kbps: f64,
    pub toc: String,
    /// Packets dropped before decoding, and FEC decodes that recovered one.
    pub lost: usize,
    pub fec_calls: usize,
    /// Packets whose final range the decoding loop compared with the
    /// encoder's.
    pub range_checks: usize,
}

/// Which packets a condition drops before decoding.
pub fn loss_pattern(cond: &Condition, packets: usize) -> Vec<bool> {
    let mut rng = XorShift(LOSS_SEED);
    (0..packets)
        .map(|i| {
            let draw = rng.below(100) < cond.decode_loss as u64;
            draw && i > 0 && i + 2 < packets
        })
        .collect()
}

/// Decode with the `opus_demo -d` loop. Returns the interleaved 16-bit
/// output after the look-ahead, cut to `n` samples per channel.
pub fn decode(decoder: Codec, cond: &Condition, packets: &Packets, n: usize) -> Result<(Vec<i16>, Stream), String> {
    let lost = loss_pattern(cond, packets.len());
    let records: Vec<BitPacket<'_>> = (0..packets.len())
        .map(|i| {
            if lost[i] {
                BitPacket { data: &[], enc_final_range: 0 }
            } else {
                BitPacket { data: packets.get(i), enc_final_range: packets.ranges[i] }
            }
        })
        .collect();
    let output = match decoder {
        Codec::Libopus => decode_bitstream(&mut CDecoder::new(cond.rate, cond.channels)?, &records),
        Codec::Opus1 => {
            let mut d = OpusDecoder::new(cond.rate as i32, cond.channels).map_err(|e| format!("OpusDecoder::new: {e}"))?;
            decode_bitstream(&mut d, &records)
        }
    };
    if let Some(m) = output.range_mismatch {
        return Err(format!(
            "final range mismatch at packet {}: encoder {:08x}, decoder {:08x}",
            m.packet, m.encoder, m.decoder
        ));
    }
    if output.decode_errors > 0 {
        return Err(format!("{} decode error(s)", output.decode_errors));
    }
    let c = cond.channels;
    let first = packets.lookahead * c;
    let last = first + n * c;
    if output.pcm.len() < last {
        return Err(format!("decoded {} samples, need {}", output.pcm.len() / c, last / c));
    }
    let seconds = packets.len() as f64 * cond.frame_us as f64 / 1e6;
    let stream = Stream {
        bytes: packets.bytes(),
        packets: packets.len(),
        kbps: packets.bytes() as f64 * 8.0 / seconds.max(1e-9) / 1000.0,
        toc: packets.toc_label(),
        lost: output.lost,
        fec_calls: output.fec_calls,
        range_checks: output.range_checks,
    };
    Ok((output.pcm[first..last].to_vec(), stream))
}

/// The results of one side of a pair.
#[derive(Clone, Debug)]
pub struct Side {
    pub stream: Stream,
    pub lookahead: usize,
    /// `opus_compare`: internal weighted error, quality, worst frame.
    pub err: f64,
    pub q: f32,
    pub worst_frame: usize,
    /// Signal-to-noise ratio in dB against the encoder input, with no time
    /// shift.
    pub snr_db: f64,
    /// The time shift (samples, positive when the output is late) within
    /// ±1 ms that correlates best with the encoder input. 0 when aligned.
    pub best_shift: i32,
    /// MOC against the 16 kHz reference, without and with the activity
    /// mask. Only for speech in mono.
    pub moc: Option<f64>,
    pub moc_vad: Option<f64>,
    /// The decoded signal, when the run keeps it.
    pub pcm: Option<Vec<i16>>,
}

/// Measure a decoded signal.
pub fn measure(
    cond: &Condition,
    item: &Item,
    input: &[f32],
    pcm: Vec<i16>,
    stream: Stream,
    lookahead: usize,
    keep_pcm: bool,
) -> Result<Side, String> {
    let c = cond.channels;
    let reference = if item.channels == 2 { Reference::Stereo(&item.reference) } else { Reference::Mono(&item.reference) };
    let cmp = opus_compare(reference, &pcm, c, cond.rate).map_err(|e| format!("opus_compare: {e}"))?;
    let snr_db = snr_db(input, &pcm);
    let best_shift = best_shift(input, &pcm, c, cond.rate);
    let (mut moc_plain, mut moc_vad) = (None, None);
    if item.class == Class::Speech && c == 1 {
        let reference = item.moc_reference();
        let test: Vec<f64> = pcm.iter().map(|&s| s as f64 / 32768.0).collect();
        let test16: Vec<f32> = if cond.rate == 16_000 {
            test.iter().map(|&v| v as f32).collect()
        } else {
            signals::Resampler::new(cond.rate, 16_000).run(&test).iter().map(|&v| v as f32).collect()
        };
        moc_plain = moc(reference, &test16, false).ok();
        moc_vad = moc(reference, &test16, true).ok();
    }
    Ok(Side {
        stream,
        lookahead,
        err: cmp.err,
        q: cmp.quality,
        worst_frame: cmp.worst_frame,
        snr_db,
        best_shift,
        moc: moc_plain,
        moc_vad,
        pcm: keep_pcm.then_some(pcm),
    })
}

/// Signal-to-noise ratio in dB of 16-bit `test` against `reference`. 999
/// means identical.
pub fn snr_db(reference: &[f32], test: &[i16]) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (&r, &t) in reference.iter().zip(test) {
        let r = r as f64;
        let d = r - t as f64 / 32768.0;
        signal += r * r;
        noise += d * d;
    }
    if noise == 0.0 { 999.0 } else { 10.0 * (signal.max(1e-30) / noise).log10() }
}

/// The shift within ±1 ms at which `test` correlates best with
/// `reference`, over the loudest 250 ms of the first channel. A tie keeps
/// the smaller shift. 0 when the item is too short.
pub fn best_shift(reference: &[f32], test: &[i16], channels: usize, rate: u32) -> i32 {
    let max = (rate / 1000) as usize;
    let window = (rate / 4) as usize;
    let frames = reference.len().min(test.len()) / channels;
    if frames < window + 2 * max + 1 {
        return 0;
    }
    let energy = |start: usize| -> f64 { (start..start + window).map(|i| (reference[i * channels] as f64).powi(2)).sum() };
    let mut start = max;
    let mut loudest = f64::MIN;
    let mut s = max;
    while s + window + max <= frames {
        let e = energy(s);
        if e > loudest {
            loudest = e;
            start = s;
        }
        s += window / 2;
    }
    let correlation = |delay: isize| -> f64 {
        (start..start + window)
            .map(|i| reference[i * channels] as f64 * test[(i as isize + delay) as usize * channels] as f64)
            .sum()
    };
    let mut best = (0isize, correlation(0));
    for d in 1..=max as isize {
        for delay in [-d, d] {
            let v = correlation(delay);
            if v > best.1 {
                best = (delay, v);
            }
        }
    }
    best.0 as i32
}

/// Run one side: encode, decode, degrade (opus1 side only), measure.
pub fn run_side(
    codec: Codec,
    decoder: Codec,
    cond: &Condition,
    item: &Item,
    input: &[f32],
    degrade: Option<&Degrade>,
    keep_pcm: bool,
) -> Result<Side, String> {
    let packets = encode(codec, cond, input, degrade)?;
    let n = input.len() / cond.channels;
    let (mut pcm, stream) = decode(decoder, cond, &packets, n)?;
    if let Some(d) = degrade {
        d.apply_post(&mut pcm, cond.rate, cond.channels);
    }
    measure(cond, item, input, pcm, stream, packets.lookahead, keep_pcm)
}

/// Apply a degradation that acts on the decoded signal to a side that kept
/// its signal, without coding again.
pub fn degrade_side(side: &Side, d: &Degrade, cond: &Condition, item: &Item, input: &[f32]) -> Result<Side, String> {
    let mut pcm = side.pcm.clone().ok_or("the side did not keep its signal")?;
    d.apply_post(&mut pcm, cond.rate, cond.channels);
    measure(cond, item, input, pcm, side.stream.clone(), side.lookahead, true)
}

/// The paired values of one job.
#[derive(Clone, Debug)]
pub struct Pair {
    /// The tested side (opus1 unless an identity check runs).
    pub opus1: Side,
    /// The reference side (libopus unless an identity check runs).
    pub libopus: Side,
    /// `(err_opus1 + 1e-6) / (err_libopus + 1e-6)`.
    pub ratio_err: f64,
    pub delta_snr_db: f64,
    pub ratio_bytes: f64,
    /// D of the OSCE draft, with libopus as the reference decoder.
    pub moc_d: Option<f64>,
}

pub fn pair(opus1: Side, libopus: Side) -> Result<Pair, String> {
    if opus1.lookahead != libopus.lookahead {
        return Err(format!("look-ahead differs: opus1 {}, libopus {}", opus1.lookahead, libopus.lookahead));
    }
    let ratio_err = (opus1.err + 1e-6) / (libopus.err + 1e-6);
    let delta_snr_db = opus1.snr_db - libopus.snr_db;
    let ratio_bytes = opus1.stream.bytes as f64 / libopus.stream.bytes.max(1) as f64;
    let moc_d = match (libopus.moc, opus1.moc) {
        (Some(r), Some(t)) => Some(moc_relative(r, t)),
        _ => None,
    };
    Ok(Pair { opus1, libopus, ratio_err, delta_snr_db, ratio_bytes, moc_d })
}

/// One job's result.
#[derive(Clone, Debug)]
pub struct Row {
    pub cond: usize,
    pub item: usize,
    pub result: Result<Pair, String>,
}

/// Run `f` for every index below `count` on `threads` threads, and return
/// the results in index order. Prints progress to stderr when `label` is
/// not empty.
pub fn parallel_map<T: Send>(count: usize, threads: usize, label: &str, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<T>>> = (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1).min(count.max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    let value = f(i);
                    *slots[i].lock().unwrap() = Some(value);
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if !label.is_empty() && count >= 20 && d.is_multiple_of((count / 10).max(1)) {
                        eprintln!("{label}: {d}/{count} jobs, {:.1} s", start.elapsed().as_secs_f64());
                    }
                }
            });
        }
    });
    slots.into_iter().map(|s| s.into_inner().unwrap().expect("every job ran")).collect()
}
