//! Discontinuous transmission (DTX) of `OpusEncoder` against libopus
//! (`OPUS_SET_DTX`).
//!
//! libopus has two kinds of DTX:
//!
//! - **SILK's DTX** (`silk_mode.useDTX`), when the speech and music
//!   analysis has no valid result and the packet is not digital silence.
//!   SILK counts frames without speech. After 10 of them it returns no
//!   bytes for up to 20 frames, and the packet is its TOC byte.
//! - **The DTX of the Opus layer** (`decide_dtx_mode`) otherwise. It counts
//!   the time without activity, from the analysis or from digital silence,
//!   and replaces up to 400 ms of frames by their TOC byte after 200 ms.
//!
//! What is checked:
//!
//! 1. CELT-only streams (forced, and the `RestrictedLowDelay` application)
//!    with DTX are libopus's byte for byte, at every complexity and input
//!    rate, with 2.5 to 60 ms frames. Their `in_dtx` is libopus's
//!    `OPUS_GET_IN_DTX` after every packet.
//! 2. In streams with SILK and Hybrid frames, the frames that DTX leaves
//!    out, `in_dtx`, and with a constant bitrate the packet sizes are
//!    libopus's wherever DTX decides on the same inputs:
//!    - the DTX of the Opus layer (complexity 7 to 10 at 16 to 48 kHz)
//!      decides on the analysis, the energy of the input and digital
//!      silence, which are libopus's;
//!    - SILK's DTX decides on SILK's voice activity detector. The detector
//!      is the same integer code in both builds of libopus. Except with the
//!      `Voip` application, opus1's SILK reads the samples libopus's float
//!      build gives it.
//!
//!    With `Voip`, opus1's SILK reads the output of the integer high-pass
//!    filter of the fixed-point build (see `OpusEncoder::encode_impl`), so
//!    its detector can decide another way in a few frames. Those streams
//!    are only reported here. `core_fixed_point.rs` compares them with the
//!    fixed-point build, whose SILK reads the same samples: there the
//!    packets, DTX ones included, are the same byte for byte.
//! 3. Packets of several frames (40 to 120 ms) count their DTX frames as
//!    libopus does, and are padded with a constant bitrate unless every
//!    frame was left out.
//! 4. `reset_state` keeps the DTX setting and starts the count again;
//!    DTX and the complexity can change in a running stream. A switch to
//!    CELT-only inside a DTX pause, announced by a frame that DTX leaves
//!    out, gives the CELT-only packets of libopus.
//! 5. The opus1 decoder and the libopus decoder give the same samples for
//!    opus1's DTX streams (comfort noise and loss concealment).
//!
//! The packets depend on the rounding of the analysis, which is libopus's
//! on 64-bit ARM. Elsewhere the tests only print what they find.
#![cfg(feature = "analysis")]

mod core_common;
mod k24_common;

use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech, speech_fullband, to_stereo};
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusDecoder, OpusEncoder, SampleRate};
use opus1::packet::Framing;

/// Whether the packets must be libopus's. The speech and music analysis
/// rounds as the libopus the tests link only on 64-bit ARM.
const EXACT: bool = cfg!(target_arch = "aarch64");
const MAX_PACKET: usize = 1276 * 6;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rc {
    Cbr,
    Vbr,
    Cvbr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct App {
    name: &'static str,
    c: i32,
    rs: Application,
}

const AUDIO: App = App { name: "audio", c: APP_AUDIO, rs: Application::Audio };
const VOIP: App = App { name: "voip", c: APP_VOIP, rs: Application::Voip };
const LOWDELAY: App = App { name: "lowdelay", c: APP_LOWDELAY, rs: Application::RestrictedLowDelay };

/// The settings of one stream, for both encoders. DTX is on.
#[derive(Clone, Debug)]
struct Setup {
    fs: usize,
    channels: usize,
    app: App,
    complexity: i32,
    bitrate: i32,
    rc: Rc,
    /// Samples per channel of each encode call.
    frame: usize,
    force: Option<ForcedMode>,
    /// 16-bit input (`opus_encode`, `encode_i16`).
    i16_input: bool,
}

impl Setup {
    fn new(fs: usize, channels: usize, app: App, frame_ms_x2: usize, bitrate: i32) -> Self {
        Setup {
            fs,
            channels,
            app,
            complexity: 9,
            bitrate,
            rc: Rc::Vbr,
            frame: fs * frame_ms_x2 / 2000,
            force: None,
            i16_input: false,
        }
    }

    fn label(&self, signal: &str) -> String {
        format!(
            "{} kHz {} ch {} {}/2 ms {} b/s c{} {:?}{}{} {signal}",
            self.fs / 1000,
            self.channels,
            self.app.name,
            self.frame * 2000 / self.fs,
            self.bitrate,
            self.complexity,
            self.rc,
            match self.force {
                Some(mode) => format!(" {mode:?}"),
                None => String::new(),
            },
            if self.i16_input { " i16" } else { "" },
        )
    }

    /// Whether DTX decides on inputs that are libopus's (see the module
    /// documentation): always, except with SILK's DTX in the `Voip`
    /// application.
    fn same_dtx_inputs(&self) -> bool {
        let analysis = self.complexity >= 7 && self.fs >= 16_000;
        self.app != VOIP || analysis || self.force == Some(ForcedMode::CeltOnly)
    }

    fn rs_encoder(&self) -> OpusEncoder {
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(self.fs as i32).unwrap(),
            Channels::try_from(self.channels).unwrap(),
            self.app.rs,
        )
        .unwrap();
        enc.set_complexity(self.complexity).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(self.bitrate)).unwrap();
        enc.set_vbr(self.rc != Rc::Cbr);
        enc.set_vbr_constraint(self.rc == Rc::Cvbr);
        enc.set_force_mode(self.force);
        enc.set_dtx(true);
        enc
    }

    fn c_encoder(&self) -> CEncoder {
        let mut enc = CEncoder::new(self.fs, self.channels, self.app.c);
        enc.complexity(self.complexity).bitrate(self.bitrate).vbr(self.rc != Rc::Cbr);
        enc.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, (self.rc == Rc::Cvbr) as i32);
        if let Some(mode) = self.force {
            enc.force_mode(c_mode(mode));
        }
        enc.set(sys::OPUS_SET_DTX_REQUEST, 1);
        enc
    }
}

fn c_mode(mode: ForcedMode) -> i32 {
    match mode {
        ForcedMode::SilkOnly => MODE_SILK,
        ForcedMode::Hybrid => MODE_HYBRID,
        ForcedMode::CeltOnly => MODE_CELT,
    }
}

/// The input of one stream: float, and its 16-bit version.
struct Input {
    f32: Vec<f32>,
    i16: Vec<i16>,
}

impl Input {
    fn new(f32: Vec<f32>) -> Self {
        let i16 = f32
            .iter()
            .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect();
        Input { f32, i16 }
    }

    /// The samples of encode call `k` of `n` samples (all channels).
    fn call(&self, k: usize, n: usize) -> Pcm<'_> {
        Pcm { f32: &self.f32[k * n..(k + 1) * n], i16: &self.i16[k * n..(k + 1) * n] }
    }
}

/// The samples of one encode call, in both forms.
struct Pcm<'a> {
    f32: &'a [f32],
    i16: &'a [i16],
}

/// White noise of amplitude `level`.
fn noise(n: usize, level: f32, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            level * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
        })
        .collect()
}

/// Speech with pauses of every kind DTX treats differently: digital
/// silence, quiet noise (-60 dBFS), noise below the 16-bit step (which
/// 16-bit input turns into digital silence), and a short gap.
fn speech_with_pauses(fs: usize) -> Vec<f32> {
    let n = |s: f64| (s * fs as f64) as usize;
    let talk = speech_fullband(fs, 6.0);
    let mut x = talk[..n(1.4)].to_vec();
    x.extend(std::iter::repeat_n(0.0, n(0.9)));
    x.extend_from_slice(&talk[n(1.4)..n(2.6)]);
    x.extend(noise(n(1.3), 1e-3, 7));
    x.extend_from_slice(&talk[n(2.6)..n(3.6)]);
    x.extend(noise(n(0.9), 1e-5, 11));
    x.extend_from_slice(&talk[n(3.6)..n(4.4)]);
    x.extend(std::iter::repeat_n(0.0, n(0.12)));
    x.extend_from_slice(&talk[n(4.4)..n(5.2)]);
    x
}

/// Music with a long pause of quiet noise, then digital silence, then
/// music that fades in.
fn music_with_pauses(fs: usize) -> Vec<f32> {
    let n = |s: f64| (s * fs as f64) as usize;
    let mut x = multitone(fs, n(1.2));
    x.extend(noise(n(1.0), 3e-4, 3));
    x.extend(std::iter::repeat_n(0.0, n(1.5)));
    let m = multitone(fs, n(1.2));
    let len = m.len();
    x.extend(m.iter().enumerate().map(|(i, v)| v * i as f32 / len as f32));
    x
}

/// The test signals at `fs` Hz with `channels` channels, interleaved. The
/// third is the speech fixture as it is, with the pauses of a talker.
fn signals(fs: usize, channels: usize) -> Vec<(&'static str, Input)> {
    let mono = [
        ("speech with pauses", speech_with_pauses(fs)),
        ("music with pauses", music_with_pauses(fs)),
        ("speech", speech(fs, 5.0)),
    ];
    mono.into_iter()
        .map(|(name, x)| {
            let x = if channels == 2 { to_stereo(&x, fs) } else { x };
            (name, Input::new(x))
        })
        .collect()
}

/// What both encoders produced for one stream: the packets, and
/// `OPUS_GET_IN_DTX` after each.
#[derive(Default)]
struct Run {
    c: Vec<Vec<u8>>,
    rs: Vec<Vec<u8>>,
    c_in_dtx: Vec<bool>,
    rs_in_dtx: Vec<bool>,
}

impl Run {
    fn push(&mut self, c: &mut CEncoder, rs: &OpusEncoder, cp: Vec<u8>, rp: &[u8]) {
        self.c.push(cp);
        self.rs.push(rp.to_vec());
        self.c_in_dtx.push(c.get(sys::OPUS_GET_IN_DTX_REQUEST) != 0);
        self.rs_in_dtx.push(rs.in_dtx());
    }
}

/// Encode call `k` with both encoders and record the result.
fn encode_call(
    s: &Setup,
    c: &mut CEncoder,
    rs: &mut OpusEncoder,
    pcm: Pcm<'_>,
    out: &mut [u8],
    run: &mut Run,
) {
    let (cp, len) = if s.i16_input {
        (c.try_encode_i16(pcm.i16, s.frame, MAX_PACKET), rs.encode_i16(pcm.i16, s.frame, out))
    } else {
        (c.try_encode(pcm.f32, s.frame, MAX_PACKET), rs.encode(pcm.f32, s.frame, out))
    };
    let len = len.expect("opus1 encode");
    run.push(c, rs, cp.expect("libopus encode"), &out[..len]);
}

/// Encode `input` with both encoders, one call per `s.frame` samples per
/// channel.
fn encode_both(s: &Setup, input: &Input) -> Run {
    let mut c = s.c_encoder();
    let mut rs = s.rs_encoder();
    let n = s.frame * s.channels;
    let mut run = Run::default();
    let mut out = vec![0u8; MAX_PACKET];
    for k in 0..input.f32.len() / n {
        encode_call(s, &mut c, &mut rs, input.call(k, n), &mut out, &mut run);
    }
    run
}

/// For each frame of each packet: whether DTX left it out, that is,
/// whether it has no bytes. A packet of several frames counts each.
fn dtx_frames(packets: &[Vec<u8>]) -> Vec<bool> {
    let mut out = Vec::new();
    for p in packets {
        let parsed = opus1::packet::parse(p, Framing::Normal).expect("valid packet");
        out.extend(parsed.frames(p).map(|f| f.is_empty()));
    }
    out
}

fn count(flags: &[bool]) -> usize {
    flags.iter().filter(|&&b| b).count()
}

/// How one stream of opus1 compares with libopus's.
struct Comparison {
    /// Frames DTX left out, libopus and opus1.
    dtx: (usize, usize),
    /// Frames that one encoder left out and the other coded.
    dtx_differ: usize,
    /// Packets after which `in_dtx` differs.
    in_dtx_differ: usize,
    /// Packets of another size, with a constant bitrate (0 otherwise).
    size_differ: usize,
    /// Packets with the same first byte, the frame count code left out:
    /// the mode, the bandwidth, the frame duration and the stereo flag.
    toc_same: usize,
    packets: usize,
}

impl Comparison {
    fn new(s: &Setup, run: &Run) -> Self {
        let (cd, rd) = (dtx_frames(&run.c), dtx_frames(&run.rs));
        let pairs = || run.c.iter().zip(&run.rs);
        Comparison {
            dtx: (count(&cd), count(&rd)),
            dtx_differ: cd.iter().zip(&rd).filter(|(a, b)| a != b).count() + cd.len().abs_diff(rd.len()),
            in_dtx_differ: run.c_in_dtx.iter().zip(&run.rs_in_dtx).filter(|(a, b)| a != b).count(),
            size_differ: if s.rc == Rc::Cbr { pairs().filter(|(a, b)| a.len() != b.len()).count() } else { 0 },
            toc_same: pairs().filter(|(a, b)| a[0] & 0xfc == b[0] & 0xfc).count(),
            packets: run.c.len(),
        }
    }

    fn same(&self) -> bool {
        self.dtx_differ == 0 && self.in_dtx_differ == 0 && self.size_differ == 0
    }

    fn line(&self, label: &str) -> String {
        format!(
            "{label}: DTX frames {} / {} (libopus / opus1), {} differ; in_dtx differs after {} packets; \
             {} sizes differ; first byte same in {} of {}",
            self.dtx.0, self.dtx.1, self.dtx_differ, self.in_dtx_differ, self.size_differ, self.toc_same, self.packets
        )
    }
}

/// Streams that are not libopus's, with the first packet that differs.
#[derive(Default)]
struct Tally {
    streams: usize,
    packets: usize,
    dtx: usize,
    failures: Vec<String>,
}

impl Tally {
    /// Every packet and every `in_dtx` must be libopus's.
    fn check_exact(&mut self, what: String, run: &Run) {
        self.streams += 1;
        self.packets += run.c.len();
        self.dtx += count(&dtx_frames(&run.c));
        if let Some(k) = run.c.iter().zip(&run.rs).position(|(a, b)| a != b) {
            let same = run.c.iter().zip(&run.rs).filter(|(a, b)| a == b).count();
            self.failures.push(format!("{what}: packet {k} differs, {same} of {} same", run.c.len()));
        } else if let Some(k) = run.c_in_dtx.iter().zip(&run.rs_in_dtx).position(|(a, b)| a != b) {
            self.failures.push(format!("{what}: in_dtx differs after packet {k}"));
        }
    }

    fn finish(self, name: &str) {
        println!("{name}: {} streams, {} packets, {} DTX frames", self.streams, self.packets, self.dtx);
        for f in self.failures.iter().take(30) {
            println!("  {f}");
        }
        if EXACT {
            assert!(
                self.failures.is_empty(),
                "{name}: {} of {} streams differ from libopus",
                self.failures.len(),
                self.streams
            );
        }
    }
}

/// A small random number generator (64-bit linear congruential).
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// Bitrates for all channels together.
const BITRATES: [i32; 7] = [6_000, 12_000, 20_000, 32_000, 64_000, 128_000, 256_000];

/// CELT-only streams with DTX at `fs`, every packet libopus's. Each row
/// runs once below complexity 7 (no analysis: only digital silence is
/// left out) and once from 7 on (the analysis decides at 16 to 48 kHz).
fn check_celt_only(fs: usize, seed: u64) {
    let mut tally = Tally::default();
    let mut rng = Lcg(seed);
    for channels in [1, 2] {
        let sigs = signals(fs, channels);
        for app in [AUDIO, VOIP, LOWDELAY] {
            for frame_ms_x2 in [5, 10, 20, 40, 80, 120] {
                for rc in [Rc::Cbr, Rc::Vbr, Rc::Cvbr] {
                    // Packets of 40 and 60 ms are compared with CBR only:
                    // with VBR opus1 leaves out libopus's limit of each
                    // frame to its share of the bitrate on purpose.
                    if frame_ms_x2 > 40 && rc != Rc::Cbr {
                        continue;
                    }
                    for complexity in [rng.below(7) as i32, 7 + rng.below(4) as i32] {
                        let mut s = Setup::new(fs, channels, app, frame_ms_x2, rng.pick(&BITRATES));
                        s.rc = rc;
                        s.force = Some(ForcedMode::CeltOnly);
                        s.complexity = complexity;
                        s.i16_input = rng.below(3) == 0;
                        let (name, input) = &sigs[rng.below(2)];
                        let run = encode_both(&s, input);
                        tally.check_exact(s.label(name), &run);
                    }
                }
            }
        }
    }
    assert!(tally.dtx > 1000, "too few DTX frames to show anything: {}", tally.dtx);
    tally.finish(&format!("CELT-only with DTX at {} kHz", fs / 1000));
}

#[test]
fn celt_only_streams_are_libopus_at_8_and_12_khz() {
    check_celt_only(8_000, 1);
    check_celt_only(12_000, 2);
}

#[test]
fn celt_only_streams_are_libopus_at_16_khz() {
    check_celt_only(16_000, 3);
}

#[test]
fn celt_only_streams_are_libopus_at_24_khz() {
    check_celt_only(24_000, 4);
}

#[test]
fn celt_only_streams_are_libopus_at_48_khz() {
    check_celt_only(48_000, 5);
}

/// Every complexity from 0 to 10, 20 ms, 48 kHz mono and stereo, CELT-only:
/// below 7 only digital silence is left out, from 7 on also the quiet
/// noise the analysis finds inactive.
#[test]
fn celt_only_streams_are_libopus_at_every_complexity() {
    let mut tally = Tally::default();
    let mut dtx_by_complexity = [0usize; 11];
    for channels in [1, 2] {
        let sigs = signals(48_000, channels);
        for complexity in 0..=10 {
            for (name, input) in &sigs {
                let mut s = Setup::new(48_000, channels, AUDIO, 40, 48_000);
                s.force = Some(ForcedMode::CeltOnly);
                s.complexity = complexity;
                let run = encode_both(&s, input);
                dtx_by_complexity[complexity as usize] += count(&dtx_frames(&run.rs));
                tally.check_exact(s.label(name), &run);
            }
        }
    }
    println!("DTX frames by complexity: {dtx_by_complexity:?}");
    // The analysis finds the quiet noise inactive, digital silence alone
    // leaves fewer frames out.
    assert!(dtx_by_complexity[7] > dtx_by_complexity[6] && dtx_by_complexity[6] > 0);
    tally.finish("CELT-only at every complexity");
}

/// Streams in which the encoder picks the mode (and forced SILK-only and
/// Hybrid streams): the frames DTX leaves out are libopus's wherever it
/// decides on the same inputs. `frames_ms_x2` are the frame durations in
/// half milliseconds.
fn check_mixed(fs: usize, frames_ms_x2: &[usize], seed: u64) {
    let mut rng = Lcg(seed);
    let (mut streams, mut dtx, mut toc_same, mut packets) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    let mut voip_report = Vec::new();
    let (mut voip_frames, mut voip_differ) = (0, 0);
    for channels in [1, 2] {
        let sigs = signals(fs, channels);
        for app in [VOIP, AUDIO] {
            for &frame_ms_x2 in frames_ms_x2 {
                for complexity in [rng.below(7) as i32, 7 + rng.below(4) as i32] {
                    let (name, input) = &sigs[rng.below(sigs.len())];
                    let mut s = Setup::new(fs, channels, app, frame_ms_x2, rng.pick(&BITRATES[..5]));
                    s.complexity = complexity;
                    s.rc = rng.pick(&[Rc::Vbr, Rc::Cbr, Rc::Cvbr]);
                    s.force = rng.pick(&[None, None, Some(ForcedMode::SilkOnly), Some(ForcedMode::Hybrid)]);
                    s.i16_input = rng.below(2) == 0;
                    let run = encode_both(&s, input);
                    let cmp = Comparison::new(&s, &run);
                    let line = cmp.line(&s.label(name));
                    if s.same_dtx_inputs() {
                        streams += 1;
                        dtx += cmp.dtx.0;
                        toc_same += cmp.toc_same;
                        packets += cmp.packets;
                        if !cmp.same() {
                            failures.push(line);
                        }
                    } else {
                        voip_frames += dtx_frames(&run.c).len();
                        voip_differ += cmp.dtx_differ;
                        voip_report.push(line);
                    }
                }
            }
        }
    }
    println!(
        "{} kHz: {streams} streams with the same DTX inputs, {dtx} DTX frames, \
         first byte same in {toc_same} of {packets} packets",
        fs / 1000
    );
    for f in failures.iter().take(30) {
        println!("  {f}");
    }
    println!("SILK's DTX with Voip: {voip_differ} of {voip_frames} frames differ");
    for line in &voip_report {
        println!("  {line}");
    }
    assert!(dtx > 200, "too few DTX frames to show anything: {dtx}");
    // Measured: 16 of 9202 frames at 8 and 12 kHz, none at 16 to 48 kHz.
    assert!(voip_differ * 100 <= voip_frames, "SILK's DTX with Voip: {voip_differ} of {voip_frames} frames differ");
    if EXACT {
        assert!(failures.is_empty(), "{} of {streams} streams leave out other frames", failures.len());
    }
}

#[test]
fn dtx_frames_of_mixed_streams_are_libopus_at_8_and_12_khz() {
    check_mixed(8_000, &[20, 40, 80, 120], 11);
    check_mixed(12_000, &[20, 40, 80, 120], 12);
}

#[test]
fn dtx_frames_of_mixed_streams_are_libopus_at_16_khz() {
    check_mixed(16_000, &[20, 40, 80, 120], 13);
}

#[test]
fn dtx_frames_of_mixed_streams_are_libopus_at_24_khz() {
    check_mixed(24_000, &[20, 40, 80, 120], 14);
}

#[test]
fn dtx_frames_of_mixed_streams_are_libopus_at_48_khz() {
    check_mixed(48_000, &[20, 40, 80, 120], 15);
}

/// Packets of 40 to 120 ms: SILK-only packets of 80 and 120 ms are two
/// frames of 40 and 60 ms, the others frames of 20 ms. Each frame is left
/// out on its own. With a constant bitrate the packet keeps its size unless
/// every frame was left out (C `dtx_count`).
#[test]
fn packets_of_several_frames_count_their_dtx_frames() {
    let mut rng = Lcg(21);
    let (mut streams, mut mixed, mut all_dtx_packets) = (0, 0, 0);
    let mut failures = Vec::new();
    for fs in [16_000, 48_000] {
        for channels in [1, 2] {
            let sigs = signals(fs, channels);
            for frame_ms_x2 in [80, 120, 160, 200, 240] {
                for force in [None, Some(ForcedMode::SilkOnly), Some(ForcedMode::CeltOnly)] {
                    let mut s = Setup::new(fs, channels, AUDIO, frame_ms_x2, rng.pick(&[12_000, 24_000, 48_000]));
                    s.rc = Rc::Cbr;
                    s.force = force;
                    s.complexity = rng.pick(&[3, 9]);
                    let (name, input) = &sigs[rng.below(2)];
                    let run = encode_both(&s, input);
                    let cmp = Comparison::new(&s, &run);
                    streams += 1;
                    for p in &run.c {
                        let flags = dtx_frames(std::slice::from_ref(p));
                        let n = count(&flags);
                        if n == flags.len() {
                            all_dtx_packets += 1;
                        } else if n > 0 {
                            mixed += 1;
                        }
                    }
                    let exact = force == Some(ForcedMode::CeltOnly);
                    let differs = if exact { run.c != run.rs || run.c_in_dtx != run.rs_in_dtx } else { !cmp.same() };
                    if differs {
                        failures.push(cmp.line(&s.label(name)));
                    }
                }
            }
        }
    }
    println!(
        "{streams} streams; libopus packets with some frames left out: {mixed}, with all: {all_dtx_packets}"
    );
    for f in &failures {
        println!("  {f}");
    }
    assert!(mixed > 10 && all_dtx_packets > 50, "the streams do not show both kinds of packet");
    if EXACT {
        assert!(failures.is_empty(), "{} of {streams} streams differ", failures.len());
    }
}

/// A change of a setting in a running stream.
#[derive(Clone, Copy)]
enum Step {
    Complexity(i32),
    Dtx(bool),
    Reset,
}

/// The changes of `settings_change_and_reset_in_a_running_stream`, one
/// every 23 frames.
const STEPS: [Step; 8] = [
    Step::Complexity(9),
    Step::Complexity(5),
    Step::Dtx(false),
    Step::Dtx(true),
    Step::Reset,
    Step::Complexity(10),
    Step::Complexity(3),
    Step::Complexity(8),
];

/// DTX goes on and off, the complexity crosses 7 (where the analysis takes
/// over from SILK's DTX), and `reset_state` comes in the middle of a pause.
/// The reset keeps DTX on and starts the count of inactive frames again.
#[test]
fn settings_change_and_reset_in_a_running_stream() {
    let mut failures = Vec::new();
    let mut streams = 0;
    for (fs, channels) in [(48_000, 1), (16_000, 2), (24_000, 1)] {
        let sigs = signals(fs, channels);
        for force in [Some(ForcedMode::CeltOnly), None] {
            for (name, input) in &sigs[..2] {
                let mut s = Setup::new(fs, channels, AUDIO, 40, 24_000);
                s.force = force;
                let mut c = s.c_encoder();
                let mut rs = s.rs_encoder();
                let n = s.frame * channels;
                let mut run = Run::default();
                let mut out = vec![0u8; MAX_PACKET];
                for k in 0..input.f32.len() / n {
                    if k % 23 == 0 {
                        match STEPS[(k / 23) % STEPS.len()] {
                            Step::Complexity(complexity) => {
                                c.complexity(complexity);
                                rs.set_complexity(complexity).unwrap();
                            }
                            Step::Dtx(on) => {
                                c.set(sys::OPUS_SET_DTX_REQUEST, on as i32);
                                rs.set_dtx(on);
                            }
                            Step::Reset => {
                                c.reset();
                                rs.reset_state();
                            }
                        }
                        assert_eq!(c.get(sys::OPUS_GET_DTX_REQUEST) != 0, rs.dtx());
                    }
                    encode_call(&s, &mut c, &mut rs, input.call(k, n), &mut out, &mut run);
                }
                streams += 1;
                let label = s.label(name);
                if force.is_some() {
                    if let Some(k) = run.c.iter().zip(&run.rs).position(|(a, b)| a != b) {
                        failures.push(format!("{label}: packet {k} differs"));
                    } else if run.c_in_dtx != run.rs_in_dtx {
                        failures.push(format!("{label}: in_dtx differs"));
                    }
                } else {
                    let cmp = Comparison::new(&s, &run);
                    if !cmp.same() {
                        failures.push(cmp.line(&label));
                    }
                }
            }
        }
    }
    for f in &failures {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} of {streams} streams differ", failures.len());
    }
}

/// The controls, and that `reset_state` keeps DTX on.
#[test]
fn dtx_controls() {
    let mut enc =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
    assert!(!enc.dtx());
    assert!(!enc.in_dtx());
    enc.set_dtx(true);
    assert!(enc.dtx());
    enc.reset_state();
    assert!(enc.dtx());
    // Digital silence: 10 frames (200 ms) are coded, the next 20 are not,
    // then one is coded again.
    let silence = vec![0.0f32; 960 * 2];
    let mut out = [0u8; 1500];
    let sizes: Vec<usize> = (0..32).map(|_| enc.encode(&silence, 960, &mut out).unwrap()).collect();
    assert!(sizes[..10].iter().all(|&n| n > 1), "{sizes:?}");
    assert!(sizes[10..30].iter().all(|&n| n == 1), "{sizes:?}");
    assert!(sizes[30] > 1 && sizes[31] == 1, "{sizes:?}");
    assert!(enc.in_dtx());
    enc.set_dtx(false);
    assert_eq!(enc.encode(&silence, 960, &mut out).map(|n| n > 1), Ok(true));
    assert!(!enc.in_dtx());
}

/// opus1's DTX streams decode to the same samples in the opus1 decoder and
/// in libopus's: the 1-byte packets are concealed, with comfort noise after
/// SILK frames.
///
/// Every packet decodes to libopus's samples bit for bit, the frames DTX
/// left out included. Before the cross-fade of the decoder (`smooth_fade`)
/// fused one of its products as clang does, a few Hybrid packets that
/// announce a switch to CELT-only decoded with a difference in the last
/// bit of some samples of their last 2.5 ms (4 of 480 samples of packet 21
/// of `24 kHz 1 ch voip 40/2 ms 48000 b/s c9 Cbr speech with pauses`).
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn opus1_dtx_streams_decode_like_libopus() {
    let mut rng = Lcg(41);
    let (mut streams, mut dtx_checked, mut not_identical, mut total) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for fs in [8_000, 16_000, 24_000, 48_000] {
        for channels in [1, 2] {
            let sigs = signals(fs, channels);
            for (app, force) in [
                (VOIP, None),
                (VOIP, Some(ForcedMode::SilkOnly)),
                (AUDIO, Some(ForcedMode::Hybrid)),
                (AUDIO, Some(ForcedMode::CeltOnly)),
                (AUDIO, None),
            ] {
                let frame_ms_x2 = rng.pick(&[10, 20, 40, 80, 120, 240]);
                let mut s = Setup::new(fs, channels, app, frame_ms_x2, rng.pick(&[12_000, 24_000, 48_000]));
                s.force = force;
                s.complexity = rng.pick(&[2, 6, 9]);
                s.rc = rng.pick(&[Rc::Vbr, Rc::Cbr]);
                let (name, input) = &sigs[rng.below(2)];
                let label = s.label(name);
                let mut rs = s.rs_encoder();
                let n = s.frame * channels;
                let mut out = vec![0u8; MAX_PACKET];
                let packets: Vec<Vec<u8>> = (0..input.f32.len() / n)
                    .map(|k| {
                        let len = rs.encode(&input.f32[k * n..(k + 1) * n], s.frame, &mut out).unwrap();
                        out[..len].to_vec()
                    })
                    .collect();
                streams += 1;
                let mut rs_dec = OpusDecoder::new(
                    SampleRate::try_from(fs as i32).unwrap(),
                    Channels::try_from(channels).unwrap(),
                )
                .unwrap();
                let mut c_dec = CDecoder::new(fs, channels);
                let max = fs * 120 / 1000;
                let mut pcm = vec![0f32; max * channels];
                // All packets so far decoded to the same samples.
                let mut in_step = true;
                for (k, p) in packets.iter().enumerate() {
                    let len = rs_dec.decode(p, max, &mut pcm).unwrap();
                    let got = &pcm[..len * channels];
                    let want = c_dec.decode(Some(p), max, false).unwrap();
                    let flags = dtx_frames(std::slice::from_ref(p));
                    total += 1;
                    if got == &want[..] {
                        dtx_checked += (in_step && flags.contains(&true)) as usize;
                        continue;
                    }
                    not_identical += 1;
                    // The frame of the packet where the samples first differ.
                    let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                    let frame = first / (got.len() / flags.len()).max(1);
                    let dtx_frame = flags.get(frame).copied().unwrap_or(false);
                    let agreement = snr(&want, got);
                    failures.push(format!(
                        "{label}: packet {k} ({} bytes) decodes otherwise from frame {frame} \
                         (left out: {dtx_frame}), {agreement:.1} dB",
                        p.len()
                    ));
                    in_step = false;
                }
            }
        }
    }
    println!(
        "{streams} streams: {dtx_checked} packets with DTX frames identical; \
         {not_identical} of {total} packets not identical"
    );
    for f in &failures {
        println!("  {f}");
    }
    assert!(dtx_checked > 500, "too few DTX packets checked: {dtx_checked}");
    assert!(failures.is_empty(), "{} packets decode otherwise", failures.len());
}


/// A switch to CELT-only inside a DTX pause: speech coded with SILK, then
/// quiet noise, and in the noise the caller forces CELT-only. The frame
/// that announces the switch codes a redundant 5 ms CELT frame at its end,
/// and the CELT-only frames continue from it. The CELT-only packets after
/// the switch are libopus's byte for byte:
///
/// - Complexity 9: the DTX of the Opus layer leaves the announcing frame
///   out, after libopus has coded its redundant frame (opus_encoder.c
///   2513-2538 come before 2564-2577).
/// - Complexity 5: SILK's DTX leaves the announcing frame out, before
///   anything but SILK ran, so the mode stays SILK and the switch is
///   announced again by the next frame, until SILK codes one. The delay
///   buffer, which the CELT frames read, does not move in the frames SILK
///   left out (opus_encoder.c:2242-2249).
#[test]
fn a_switch_to_celt_only_inside_a_dtx_pause_is_libopus() {
    let mut failures = Vec::new();
    let mut announced_in_dtx = 0;
    for fs in [16_000, 24_000, 48_000] {
        let n = |s: f64| (s * fs as f64) as usize;
        let mut x = speech_fullband(fs, 1.0);
        x.extend(noise(n(2.0), 1e-3, 5));
        x.extend(speech_fullband(fs, 1.0));
        let input = Input::new(x);
        for complexity in [5, 9] {
            // The noise starts with frame 50; DTX starts with frame 60.
            for switch_at in [65, 75, 78] {
                let mut s = Setup::new(fs, 1, AUDIO, 40, 32_000);
                s.complexity = complexity;
                s.force = Some(ForcedMode::SilkOnly);
                let label = format!("{} switch at frame {switch_at}", s.label("speech, noise, speech"));
                let mut c = s.c_encoder();
                let mut rs = s.rs_encoder();
                c.bandwidth(BW_WB);
                rs.set_bandwidth(opus1::Bandwidth::Wideband);
                let mut run = Run::default();
                let mut out = vec![0u8; MAX_PACKET];
                for k in 0..input.f32.len() / s.frame {
                    if k == switch_at {
                        c.force_mode(MODE_CELT);
                        rs.set_force_mode(Some(ForcedMode::CeltOnly));
                    }
                    encode_call(&s, &mut c, &mut rs, input.call(k, s.frame), &mut out, &mut run);
                }
                announced_in_dtx += (run.c[switch_at].len() == 1) as usize;
                if dtx_frames(&run.c) != dtx_frames(&run.rs) {
                    failures.push(format!("{label}: other frames left out"));
                }
                // The CELT-only packets after the switch, DTX ones included.
                let celt: Vec<usize> = (switch_at..run.c.len()).filter(|&k| run.c[k][0] >= 0x80).collect();
                if celt.len() < 50 {
                    failures.push(format!("{label}: only {} CELT-only packets", celt.len()));
                }
                if let Some(&k) = celt.iter().find(|&&k| run.c[k] != run.rs[k]) {
                    failures.push(format!("{label}: CELT-only packet {k} differs"));
                }
            }
        }
    }
    for f in &failures {
        println!("  {f}");
    }
    assert!(announced_in_dtx >= 12, "the switch was announced in only {announced_in_dtx} left-out frames");
    if EXACT {
        assert!(failures.is_empty(), "{} streams differ", failures.len());
    }
}
