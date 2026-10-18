//! `OpusEncoder` with the speech and music analysis against libopus
//! (`opus_encode_float` and `opus_encode`) at complexity 7 to 10, where
//! the float build of libopus runs its analysis (src/analysis.c).
//!
//! What is checked:
//!
//! 1. CELT-only packets (forced, or the `RestrictedLowDelay` application)
//!    are libopus's byte for byte: 16, 24 and 48 kHz; mono and stereo;
//!    2.5 to 60 ms; CBR, VBR and constrained VBR; the three applications;
//!    the three signal types; 6 to 510 kb/s; float and 16-bit input.
//!    Packets of 40 and 60 ms (two or three CELT frames) are compared with
//!    CBR only: with VBR opus1 leaves out libopus's limit of each frame to
//!    its share of the bitrate on purpose (see `encode_multiframe`).
//! 2. The `AnalysisInfo` each frame is coded with is what libopus's
//!    `run_analysis` and `tonality_get_info` give for the same calls, in
//!    every mode (SILK and Hybrid included), and with packets of several
//!    frames.
//! 3. The analysis reads the caller's whole buffer with an expert frame
//!    duration, it starts again when the complexity drops below 7, and
//!    `OPUS_RESET_STATE` keeps the speech probability (`voice_ratio`).
//! 4. Where the encoder picks the mode itself, the mode, bandwidth and
//!    channels of its packets (the first byte without the frame count code)
//!    are libopus's up to and including the first SILK or Hybrid packet,
//!    and streams that stay CELT-only are libopus's byte for byte. The
//!    frame count code is left out because a SILK packet with CBR is padded
//!    when SILK codes fewer bytes than the packet holds, and opus1's SILK
//!    codes other bytes than libopus's.
//!
//! The SILK encoder of opus1 is the fixed-point one, so SILK and Hybrid
//! packets cannot be libopus's (float build). `snr_report` prints how far
//! the decoded streams are from libopus's, with and without the analysis.
//!
//! The packets depend on the rounding of the analysis, which is libopus's
//! on 64-bit ARM. Elsewhere the tests only print what they find.
#![cfg(feature = "analysis")]

mod analysis_common;
mod core_common;
mod k24_common;

use analysis_common::{CAnalysis, Layout, Pcm, info_difference};
use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech, speech_fullband, to_stereo};
use opus1::internals::analysis::{AnalysisGate, last_analysis_info, set_analysis_gate};
use opus1::{Application, Bitrate, ForcedMode, FrameDuration, InbandFec, OpusEncoder, Signal};

/// Whether the packets must be libopus's: the analysis rounds as the
/// libopus the tests link only on 64-bit ARM.
const EXACT: bool = cfg!(target_arch = "aarch64");
const MAX_PACKET: usize = 1276 * 3;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rc {
    Cbr,
    Vbr,
    Cvbr,
}

#[derive(Clone, Copy, Debug)]
struct App {
    name: &'static str,
    c: i32,
    rs: Application,
}

const AUDIO: App = App { name: "audio", c: APP_AUDIO, rs: Application::Audio };
const VOIP: App = App { name: "voip", c: APP_VOIP, rs: Application::Voip };
const LOWDELAY: App = App { name: "lowdelay", c: APP_LOWDELAY, rs: Application::RestrictedLowDelay };

/// The settings of one stream, for both encoders.
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
    force_celt: bool,
    signal: Signal,
    /// 16-bit input (`opus_encode`, `encode_i16`).
    i16_input: bool,
    lsb_depth: i32,
    fec: InbandFec,
    loss: i32,
    expert: FrameDuration,
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
            force_celt: false,
            signal: Signal::Auto,
            i16_input: false,
            lsb_depth: 24,
            fec: InbandFec::Off,
            loss: 0,
            expert: FrameDuration::Argument,
        }
    }

    fn label(&self, signal: &str) -> String {
        format!(
            "{} kHz {} ch {} {}/2 ms {} b/s c{} {:?}{}{} {:?}{}{} {signal}",
            self.fs / 1000,
            self.channels,
            self.app.name,
            self.frame * 2000 / self.fs,
            self.bitrate,
            self.complexity,
            self.rc,
            if self.force_celt { " CELT" } else { "" },
            if self.i16_input { " i16" } else { "" },
            self.signal,
            if self.lsb_depth != 24 { format!(" lsb {}", self.lsb_depth) } else { String::new() },
            if self.fec != InbandFec::Off { format!(" {:?} {}%", self.fec, self.loss) } else { String::new() },
        )
    }

    fn rs_encoder(&self) -> OpusEncoder {
        let mut enc = OpusEncoder::new(self.fs as i32, self.channels, self.app.rs).unwrap();
        enc.set_complexity(self.complexity).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(self.bitrate)).unwrap();
        enc.set_vbr(self.rc != Rc::Cbr);
        enc.set_vbr_constraint(self.rc == Rc::Cvbr);
        if self.force_celt {
            enc.set_force_mode(Some(ForcedMode::CeltOnly));
        }
        enc.set_signal(self.signal);
        enc.set_lsb_depth(self.lsb_depth).unwrap();
        enc.set_inband_fec_mode(self.fec);
        enc.set_packet_loss_perc(self.loss).unwrap();
        enc.set_expert_frame_duration(self.expert);
        enc
    }

    fn c_encoder(&self) -> CEncoder {
        let mut enc = CEncoder::new(self.fs, self.channels, self.app.c);
        enc.complexity(self.complexity).bitrate(self.bitrate).vbr(self.rc != Rc::Cbr);
        enc.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, (self.rc == Rc::Cvbr) as i32);
        if self.force_celt {
            enc.force_mode(MODE_CELT);
        }
        enc.set(sys::OPUS_SET_SIGNAL_REQUEST, c_signal(self.signal));
        enc.set(sys::OPUS_SET_LSB_DEPTH_REQUEST, self.lsb_depth);
        enc.set(sys::OPUS_SET_INBAND_FEC_REQUEST, c_fec(self.fec));
        enc.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, self.loss);
        enc.set(sys::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, self.expert.to_raw());
        enc
    }
}

fn c_signal(signal: Signal) -> i32 {
    match signal {
        Signal::Auto => sys::OPUS_AUTO,
        Signal::Voice => sys::OPUS_SIGNAL_VOICE,
        Signal::Music => sys::OPUS_SIGNAL_MUSIC,
    }
}

fn c_fec(fec: InbandFec) -> i32 {
    match fec {
        InbandFec::Off => 0,
        InbandFec::On => 1,
        InbandFec::OnUnlessMusic => 2,
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
}

/// A signal in which the analysis has something to find: music, then
/// digital silence, speech, a quiet noise floor and music again.
fn mixed(fs: usize) -> Vec<f32> {
    let n = |s: f64| (s * fs as f64) as usize;
    let mut x = multitone(fs, n(0.6));
    x.extend(std::iter::repeat_n(0.0, n(0.3)));
    x.extend(speech_fullband(fs, 0.8));
    let mut seed = 99u32;
    x.extend((0..n(0.3)).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        3e-4 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
    }));
    x.extend(multitone(fs, n(0.5)).iter().map(|v| 0.5 * v));
    x
}

/// The test signals at `fs` Hz with `channels` channels, interleaved.
fn signals(fs: usize, channels: usize) -> Vec<(&'static str, Input)> {
    let mono = [
        ("speech", speech_fullband(fs, 2.0)),
        ("music", multitone(fs, 2 * fs)),
        ("mixed", mixed(fs)),
    ];
    mono.into_iter()
        .map(|(name, x)| {
            let x = if channels == 2 { to_stereo(&x, fs) } else { x };
            (name, Input::new(x))
        })
        .collect()
}

/// The packets of both encoders for `input`, one encode call per `frame`
/// samples per channel. Also the `AnalysisInfo` opus1 coded the last frame
/// of each packet with.
fn encode_both(
    s: &Setup,
    input: &Input,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut c = s.c_encoder();
    let mut rs = s.rs_encoder();
    let n = s.frame * s.channels;
    let calls = input.f32.len() / n;
    let mut c_packets = Vec::with_capacity(calls);
    let mut rs_packets = Vec::with_capacity(calls);
    let mut out = vec![0u8; MAX_PACKET];
    for k in 0..calls {
        let (cp, len) = if s.i16_input {
            let pcm = &input.i16[k * n..(k + 1) * n];
            (c.try_encode_i16(pcm, s.frame, MAX_PACKET), rs.encode_i16(pcm, s.frame, &mut out))
        } else {
            let pcm = &input.f32[k * n..(k + 1) * n];
            (c.try_encode(pcm, s.frame, MAX_PACKET), rs.encode(pcm, s.frame, &mut out))
        };
        c_packets.push(cp.expect("libopus encode"));
        rs_packets.push(out[..len.expect("opus1 encode")].to_vec());
    }
    (c_packets, rs_packets)
}

/// The first byte of each packet without the frame count code: the mode,
/// the bandwidth, the frame duration and the stereo flag.
fn first_bytes(packets: &[Vec<u8>]) -> Vec<Vec<u8>> {
    packets.iter().map(|p| vec![p[0] & 0xfc]).collect()
}

/// A packet that holds no audio: libopus's answer to a budget too small to
/// code anything. Its frames are empty.
fn holds_no_audio(packet: &[u8]) -> bool {
    match opus1::packet::parse(packet, false) {
        Ok(parsed) => parsed.frames(packet).all(|f| f.is_empty()),
        Err(_) => false,
    }
}

/// Streams that are not libopus's, with the first packet that differs.
#[derive(Default)]
struct Tally {
    streams: usize,
    packets: usize,
    failures: Vec<String>,
}

impl Tally {
    fn check(&mut self, what: String, c: &[Vec<u8>], rs: &[Vec<u8>]) {
        self.streams += 1;
        self.packets += c.len();
        if let Some(k) = c.iter().zip(rs).position(|(a, b)| a != b) {
            let same = c.iter().zip(rs).filter(|(a, b)| a == b).count();
            self.failures.push(format!("{what}: packet {k} differs, {same} of {} same", c.len()));
        }
    }

    fn finish(self, name: &str) {
        println!("{name}: {} streams, {} packets", self.streams, self.packets);
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

/// Bitrates for all channels together, from 6 kb/s to 510 kb/s.
const BITRATES: [i32; 8] = [6_000, 12_000, 20_000, 32_000, 48_000, 80_000, 160_000, 510_000];

/// CELT-only streams at `fs`, every packet libopus's. The matrix runs over
/// the channels, applications, frame durations and rate controls; each row
/// takes three bitrates, and a complexity, a signal type, an input type and
/// a signal at random.
fn check_celt_only(fs: usize, seed: u64) {
    let mut tally = Tally::default();
    let mut rng = Lcg(seed);
    for channels in [1, 2] {
        let sigs = signals(fs, channels);
        for app in [AUDIO, VOIP, LOWDELAY] {
            for frame_ms_x2 in [5, 10, 20, 40, 80, 120] {
                for rc in [Rc::Cbr, Rc::Vbr, Rc::Cvbr] {
                    if frame_ms_x2 > 40 && rc != Rc::Cbr {
                        continue;
                    }
                    for _ in 0..3 {
                        let mut s = Setup::new(fs, channels, app, frame_ms_x2, rng.pick(&BITRATES));
                        s.rc = rc;
                        s.force_celt = true;
                        s.complexity = 7 + rng.below(4) as i32;
                        s.signal = rng.pick(&[Signal::Auto, Signal::Voice, Signal::Music]);
                        s.i16_input = rng.below(3) == 0;
                        let (name, input) = &sigs[rng.below(sigs.len())];
                        let (c, rs) = encode_both(&s, input);
                        tally.check(s.label(name), &c, &rs);
                    }
                }
            }
        }
    }
    tally.finish(&format!("CELT-only at {} kHz", fs / 1000));
}

#[test]
fn celt_only_packets_are_libopus_at_16_khz() {
    check_celt_only(16_000, 1);
}

#[test]
fn celt_only_packets_are_libopus_at_24_khz() {
    check_celt_only(24_000, 2);
}

#[test]
fn celt_only_packets_are_libopus_at_48_khz() {
    check_celt_only(48_000, 3);
}

/// Every bitrate of the matrix at every complexity from 7 to 10, 20 ms,
/// 48 kHz stereo and mono: CELT-only packets are libopus's.
#[test]
fn celt_only_packets_are_libopus_at_every_bitrate_and_complexity() {
    let mut tally = Tally::default();
    for channels in [1, 2] {
        let sigs = signals(48_000, channels);
        for complexity in 7..=10 {
            for bitrate in BITRATES {
                for (name, input) in &sigs {
                    let mut s = Setup::new(48_000, channels, AUDIO, 40, bitrate);
                    s.force_celt = true;
                    s.complexity = complexity;
                    let (c, rs) = encode_both(&s, input);
                    tally.check(s.label(name), &c, &rs);
                }
            }
        }
    }
    tally.finish("every bitrate and complexity");
}

/// The analysis opus1 codes each frame with is what libopus's analysis
/// gives for the same calls, in every mode. A copy of the libopus analysis
/// is called as `opus_encode_native` calls it: on the caller's buffer, all
/// channels, with the input's bit depth, and for a packet of several frames
/// once per frame from the read position before the packet.
#[test]
fn the_analysis_of_each_frame_is_libopus() {
    let mut failures = Vec::new();
    let mut frames = 0;
    let mut rng = Lcg(17);
    for fs in [16_000, 24_000, 48_000] {
        for channels in [1, 2] {
            let sigs = signals(fs, channels);
            for app in [AUDIO, VOIP, LOWDELAY] {
                for frame_ms_x2 in [5, 20, 40, 80, 120, 200, 240] {
                    let mut s = Setup::new(fs, channels, app, frame_ms_x2, rng.pick(&BITRATES));
                    s.complexity = 7 + rng.below(4) as i32;
                    s.i16_input = rng.below(2) == 0;
                    s.lsb_depth = rng.pick(&[8, 16, 24]);
                    let (name, input) = &sigs[rng.below(sigs.len())];
                    let label = s.label(name);
                    let mut rs = s.rs_encoder();
                    let mut c = CAnalysis::new(fs as i32);
                    let n = s.frame * channels;
                    let lsb = if s.i16_input { s.lsb_depth.min(16) } else { s.lsb_depth };
                    let mut out = vec![0u8; MAX_PACKET];
                    for k in 0..input.f32.len() / n {
                        let range = k * n..(k + 1) * n;
                        let (pcm, len) = if s.i16_input {
                            let len = rs.encode_i16(&input.i16[range.clone()], s.frame, &mut out).unwrap();
                            (Pcm::I16(input.i16[range].to_vec()), len)
                        } else {
                            let len = rs.encode(&input.f32[range.clone()], s.frame, &mut out).unwrap();
                            (Pcm::F32(input.f32[range].to_vec()), len)
                        };
                        // The frames of the packet: SILK-only packets of 80
                        // and 120 ms have two frames of 40 and 60 ms, the
                        // rest frames of 20 ms. The first byte tells.
                        let toc = out[0];
                        let silk = toc < 0x60;
                        let part = if s.frame <= fs / 50 {
                            s.frame
                        } else if silk && s.frame == 4 * fs / 50 {
                            2 * fs / 50
                        } else if silk && s.frame == 6 * fs / 50 {
                            3 * fs / 50
                        } else if silk && s.frame <= 3 * fs / 50 {
                            s.frame
                        } else {
                            fs / 50
                        };
                        let cursor = (c.st.read_pos, c.st.read_subframe);
                        let mut info = c.run(&pcm, Layout::encoder(channels), s.frame, s.frame, lsb);
                        if part < s.frame {
                            (c.st.read_pos, c.st.read_subframe) = cursor;
                            for _ in 0..s.frame / part {
                                info = c.get_info(part);
                            }
                        }
                        // A packet without audio codes no frame, and the
                        // last frame opus1 coded is an earlier one.
                        if holds_no_audio(&out[..len]) {
                            continue;
                        }
                        frames += 1;
                        let got = last_analysis_info(&rs);
                        if let Some(d) = info_difference(&got, &info) {
                            failures.push(format!("{label}: call {k}: {d}"));
                            break;
                        }
                    }
                }
            }
        }
    }
    println!("{frames} calls compared");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} streams with another analysis", failures.len());
    }
}

/// With an expert frame duration the caller passes more samples than the
/// frame. libopus's analysis reads them all and takes the rest as the start
/// of the next call; opus1 does the same, so the CELT-only packets stay
/// libopus's when the caller passes non-overlapping buffers (the analysis
/// then sees a signal that does not fit) and when it passes overlapping
/// ones.
#[test]
fn expert_frame_duration_reads_the_whole_buffer() {
    let mut tally = Tally::default();
    for (channels, fs) in [(1, 48_000), (2, 16_000), (2, 24_000)] {
        let sigs = signals(fs, channels);
        for (expert, caller_ms_x2) in [
            (FrameDuration::Ms10, 40),
            (FrameDuration::Ms20, 40),
            (FrameDuration::Ms20, 120),
            (FrameDuration::Ms5, 20),
        ] {
            for overlapping in [false, true] {
                for (name, input) in &sigs {
                    let mut s = Setup::new(fs, channels, AUDIO, caller_ms_x2, 64_000);
                    s.force_celt = true;
                    s.expert = expert;
                    let coded = expert.samples(fs as i32).unwrap();
                    let step = if overlapping { coded } else { s.frame } * channels;
                    let n = s.frame * channels;
                    let mut c = s.c_encoder();
                    let mut rs = s.rs_encoder();
                    let mut out = vec![0u8; MAX_PACKET];
                    let (mut cp, mut rp) = (Vec::new(), Vec::new());
                    let mut at = 0;
                    while at + n <= input.f32.len() {
                        let pcm = &input.f32[at..at + n];
                        cp.push(c.try_encode(pcm, s.frame, MAX_PACKET).unwrap());
                        let len = rs.encode(pcm, s.frame, &mut out).unwrap();
                        rp.push(out[..len].to_vec());
                        at += step;
                    }
                    let what = format!("{expert:?} of {}/2 ms, overlapping {overlapping}, {}", caller_ms_x2, s.label(name));
                    tally.check(what, &cp, &rp);
                }
            }
        }
    }
    tally.finish("expert frame duration");
}

/// The complexity goes between 6 (no analysis) and 7 to 10 in a running
/// stream, and `OPUS_RESET_STATE` comes in the middle. libopus resets the
/// analysis when it stops running, and keeps `voice_ratio` across the reset.
/// At a stereo bitrate near the threshold between mono and stereo coding
/// (17 to 19 kb/s, which `voice_ratio` moves), the packets show both.
#[test]
fn complexity_changes_and_reset() {
    let mut tally = Tally::default();
    for (channels, bitrate) in [(2, 18_000), (2, 24_000), (1, 32_000), (2, 96_000)] {
        let sigs = signals(48_000, channels);
        for (name, input) in &sigs {
            for force_celt in [true, false] {
                let mut s = Setup::new(48_000, channels, AUDIO, 40, bitrate);
                s.force_celt = force_celt;
                let mut c = s.c_encoder();
                let mut rs = s.rs_encoder();
                let n = s.frame * channels;
                let (mut cp, mut rp) = (Vec::new(), Vec::new());
                let mut out = vec![0u8; MAX_PACKET];
                for k in 0..input.f32.len() / n {
                    // 7 .. 9 .. 6 .. 10 .. reset .. 8, in steps of 15 frames.
                    if k % 15 == 0 {
                        let complexity = [7, 9, 6, 10, 10, 8, 6, 9][(k / 15) % 8];
                        c.complexity(complexity);
                        rs.set_complexity(complexity).unwrap();
                        if (k / 15) % 8 == 4 {
                            c.reset();
                            rs.reset_state();
                        }
                    }
                    let pcm = &input.f32[k * n..(k + 1) * n];
                    cp.push(c.encode_max(pcm, MAX_PACKET));
                    let len = rs.encode(pcm, s.frame, &mut out).unwrap();
                    rp.push(out[..len].to_vec());
                }
                if force_celt {
                    tally.check(s.label(name), &cp, &rp);
                } else {
                    // The first bytes, while no SILK or Hybrid packet was
                    // coded.
                    let celt = cp.iter().take_while(|p| p[0] >= 0x80).count();
                    let n = (celt + 1).min(cp.len());
                    tally.check(
                        format!("first bytes of {} CELT packets, {}", celt, s.label(name)),
                        &first_bytes(&cp[..n]),
                        &first_bytes(&rp[..n]),
                    );
                }
            }
        }
    }
    tally.finish("complexity changes and reset");
}

/// `OPUS_RESET_STATE` keeps the speech probability of the last packet that
/// was not digital silence (C `voice_ratio`). After a reset, the frames of
/// digital silence go by the probability the analysis found in the music
/// before the reset: the mode, bandwidth and channels of their packets are
/// libopus's, while the `Voip` application on its own would ask for speech.
#[test]
fn reset_keeps_the_speech_probability() {
    let mut tally = Tally::default();
    let fs = 48_000;
    let mut changed = 0;
    for channels in [1, 2] {
        for app in [VOIP, AUDIO] {
            for bitrate in [16_000, 24_000, 32_000, 40_000, 48_000, 64_000] {
                let music = multitone(fs, fs);
                let music = if channels == 2 { to_stereo(&music, fs) } else { music };
                let silence = vec![0.0f32; fs / 5 * channels];
                let s = Setup::new(fs, channels, app, 40, bitrate * channels as i32);
                let mut c = s.c_encoder();
                let mut rs = s.rs_encoder();
                let n = s.frame * channels;
                let mut out = vec![0u8; MAX_PACKET];
                for f in music.chunks_exact(n) {
                    c.encode_max(f, MAX_PACKET);
                    rs.encode(f, s.frame, &mut out).unwrap();
                }
                c.reset();
                rs.reset_state();
                let mut fresh = s.c_encoder();
                let (mut cp, mut rp, mut fp) = (Vec::new(), Vec::new(), Vec::new());
                for f in silence.chunks_exact(n) {
                    cp.push(c.encode_max(f, MAX_PACKET));
                    fp.push(fresh.encode_max(f, MAX_PACKET));
                    let len = rs.encode(f, s.frame, &mut out).unwrap();
                    rp.push(out[..len].to_vec());
                }
                // Whether the kept probability changed libopus's decision,
                // against a new encoder.
                changed += (first_bytes(&cp) != first_bytes(&fp)) as usize;
                tally.check(format!("after the reset, {}", s.label("music")), &first_bytes(&cp), &first_bytes(&rp));
            }
        }
    }
    println!("{changed} streams where the kept probability changes libopus's packets");
    assert!(changed > 0, "no stream shows the kept speech probability");
    tally.finish("reset keeps the speech probability");
}

/// The encoder picks the mode itself. The first byte of every packet
/// (mode, bandwidth, channels) is libopus's as long as only CELT-only
/// packets were coded: the speech probability, the detected bandwidth and
/// the stereo decision follow libopus. For music at the higher bitrates the
/// whole stream is CELT-only, and its packets are libopus's byte for byte.
#[test]
fn automatic_mode_follows_libopus_until_the_first_silk_packet() {
    let mut tally = Tally::default();
    let mut whole = 0;
    let mut rng = Lcg(23);
    for fs in [16_000, 24_000, 48_000] {
        for channels in [1, 2] {
            let sigs = signals(fs, channels);
            for app in [AUDIO, VOIP] {
                for bitrate in [12_000, 24_000, 32_000, 64_000, 128_000] {
                    for (name, input) in &sigs {
                        let mut s = Setup::new(fs, channels, app, 40, bitrate);
                        s.complexity = 7 + rng.below(4) as i32;
                        s.rc = rng.pick(&[Rc::Cbr, Rc::Vbr, Rc::Cvbr]);
                        s.fec = rng.pick(&[InbandFec::Off, InbandFec::On, InbandFec::OnUnlessMusic]);
                        s.loss = if s.fec == InbandFec::Off { 0 } else { rng.pick(&[5, 20]) };
                        let (c, rs) = encode_both(&s, input);
                        let celt = c.iter().take_while(|p| p[0] >= 0x80).count();
                        if celt == c.len() {
                            whole += 1;
                            tally.check(s.label(name), &c, &rs);
                        } else {
                            tally.check(
                                format!("first bytes of {celt} CELT packets, {}", s.label(name)),
                                &first_bytes(&c[..=celt]),
                                &first_bytes(&rs[..=celt]),
                            );
                        }
                    }
                }
            }
        }
    }
    println!("{whole} streams CELT-only throughout");
    tally.finish("automatic mode");
}

/// The signal types and in-band FEC mode 2 decide the mode: CELT-only for
/// music where FEC would otherwise move the stream to SILK. The first
/// bytes are libopus's until the first SILK packet, and the share of SILK
/// packets is the same in both encoders.
#[test]
fn signal_type_and_fec_mode_2_choose_the_mode_as_libopus() {
    let mut tally = Tally::default();
    for signal in [Signal::Auto, Signal::Voice, Signal::Music] {
        for fec in [InbandFec::On, InbandFec::OnUnlessMusic] {
            for (name, input) in &signals(48_000, 1) {
                let mut s = Setup::new(48_000, 1, AUDIO, 40, 64_000);
                s.signal = signal;
                s.fec = fec;
                s.loss = 10;
                let (c, rs) = encode_both(&s, input);
                let celt = c.iter().take_while(|p| p[0] >= 0x80).count();
                let n = (celt + 1).min(c.len());
                let silk_c = c.iter().filter(|p| p[0] < 0x80).count();
                let silk_rs = rs.iter().filter(|p| p[0] < 0x80).count();
                println!("{}: {silk_c} / {silk_rs} SILK or Hybrid packets", s.label(name));
                tally.check(
                    format!("first bytes, {}", s.label(name)),
                    &first_bytes(&c[..n]),
                    &first_bytes(&rs[..n]),
                );
                if signal == Signal::Music || (fec == InbandFec::OnUnlessMusic && *name == "music") {
                    assert_eq!(silk_c == 0, silk_rs == 0, "{}", s.label(name));
                }
            }
        }
    }
    tally.finish("signal type and FEC mode 2");
}

/// The new controls keep their values, refuse what libopus refuses, and
/// survive `reset_state` as in libopus.
#[test]
fn controls() {
    let mut enc = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
    assert_eq!(enc.signal(), Signal::Auto);
    assert_eq!(enc.lsb_depth(), 24);
    assert_eq!(enc.inband_fec_mode(), InbandFec::Off);
    assert!(enc.set_lsb_depth(7).is_err());
    assert!(enc.set_lsb_depth(25).is_err());
    assert_eq!(enc.lsb_depth(), 24);
    enc.set_lsb_depth(8).unwrap();
    enc.set_signal(Signal::Music);
    enc.set_inband_fec_mode(InbandFec::OnUnlessMusic);
    assert!(enc.inband_fec());
    set_analysis_gate(&mut enc, AnalysisGate::FixedPointBuild);
    enc.reset_state();
    assert_eq!(enc.lsb_depth(), 8);
    assert_eq!(enc.signal(), Signal::Music);
    assert_eq!(enc.inband_fec_mode(), InbandFec::OnUnlessMusic);
    assert_eq!(opus1::internals::analysis::analysis_gate(&enc), AnalysisGate::FixedPointBuild);
    enc.set_inband_fec(false);
    assert_eq!(enc.inband_fec_mode(), InbandFec::Off);
    enc.set_inband_fec(true);
    assert_eq!(enc.inband_fec_mode(), InbandFec::On);
}

/// The analysis gate: `Off` runs no analysis at any complexity, and
/// `FixedPointBuild` only at 10. Without the analysis the CELT-only packets
/// at complexity 9 are those libopus codes at complexity 6 with the same
/// settings but CELT's own complexity: there is no C build to compare
/// with, so the check is that the analysis never ran.
#[test]
fn analysis_gate() {
    let input = &signals(48_000, 1)[1].1;
    for (gate, complexity, runs) in [
        (AnalysisGate::FloatBuild, 7, true),
        (AnalysisGate::FloatBuild, 6, false),
        (AnalysisGate::FixedPointBuild, 9, false),
        (AnalysisGate::FixedPointBuild, 10, true),
        (AnalysisGate::Off, 10, false),
    ] {
        let mut enc = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
        enc.set_complexity(complexity).unwrap();
        set_analysis_gate(&mut enc, gate);
        let mut out = vec![0u8; MAX_PACKET];
        let mut valid = 0;
        for frame in input.f32.as_chunks::<960>().0 {
            enc.encode(frame, 960, &mut out).unwrap();
            valid += last_analysis_info(&enc).valid as usize;
        }
        assert_eq!(valid > 0, runs, "{gate:?} at complexity {complexity}");
    }
    // 8 and 12 kHz input never runs it.
    for fs in [8_000, 12_000] {
        let mut enc = OpusEncoder::new(fs, 1, Application::Audio).unwrap();
        enc.set_complexity(10).unwrap();
        let x = speech(fs as usize, 1.0);
        let mut out = vec![0u8; MAX_PACKET];
        for frame in x.chunks_exact(fs as usize / 50) {
            enc.encode(frame, fs as usize / 50, &mut out).unwrap();
            assert!(!last_analysis_info(&enc).valid);
        }
    }
}

/// Report: the decoded SNR of opus1 and of libopus for streams with SILK
/// and Hybrid packets, with opus1's analysis off (as before the port) and
/// on. SILK in opus1 is the fixed-point encoder, so these streams cannot be
/// libopus's byte for byte.
///
/// `cargo test --release --test core_analysis_libopus snr_report -- --ignored --nocapture`
#[test]
#[ignore = "report, prints a table"]
fn snr_report() {
    println!(
        "{:<56} {:>7} {:>7} {:>7} {:>7} {:>9}",
        "stream", "C dB", "off dB", "on dB", "on-C", "same TOC"
    );
    let mut sum = [0.0f64; 3];
    let mut rows = 0;
    let (mut same_toc, mut packets) = (0, 0);
    for fs in [16_000, 24_000, 48_000] {
        for channels in [1, 2] {
            let sigs = signals(fs, channels);
            for app in [VOIP, AUDIO] {
                for bitrate in [10_000, 16_000, 24_000, 32_000, 48_000] {
                    let bitrate = bitrate * channels as i32;
                    for (name, input) in &sigs {
                        let s = Setup::new(fs, channels, app, 40, bitrate);
                        let (c, on) = encode_both(&s, input);
                        let mut off_enc = s.rs_encoder();
                        set_analysis_gate(&mut off_enc, AnalysisGate::Off);
                        let mut out = vec![0u8; MAX_PACKET];
                        let off: Vec<Vec<u8>> = input
                            .f32
                            .chunks_exact(s.frame * channels)
                            .map(|f| {
                                let len = off_enc.encode(f, s.frame, &mut out).unwrap();
                                out[..len].to_vec()
                            })
                            .collect();
                        // The decoded signal lags the input by the
                        // encoder's look-ahead; the first 100 ms are left
                        // out.
                        let lag = off_enc.lookahead() * channels;
                        let snr = |p: &[Vec<u8>]| {
                            let dec = k24_common::c_decode_all(p, fs, channels);
                            let (mut sig, mut err) = (0f64, 0f64);
                            let n = input.f32.len().min(dec.len().saturating_sub(lag));
                            for i in fs / 10 * channels..n {
                                let (a, b) = (input.f32[i] as f64, dec[i + lag] as f64);
                                sig += a * a;
                                err += (a - b) * (a - b);
                            }
                            10.0 * (sig / err.max(1e-30)).log10()
                        };
                        let (sc, soff, son) = (snr(&c), snr(&off), snr(&on));
                        let same = c.iter().zip(&on).filter(|(a, b)| a[0] == b[0]).count();
                        println!(
                            "{:<56} {:>7.2} {:>7.2} {:>7.2} {:>+7.2} {:>4}/{:<4}",
                            s.label(name),
                            sc,
                            soff,
                            son,
                            son - sc,
                            same,
                            c.len()
                        );
                        sum[0] += sc;
                        sum[1] += soff;
                        sum[2] += son;
                        rows += 1;
                        same_toc += same;
                        packets += c.len();
                    }
                }
            }
        }
    }
    let r = rows as f64;
    println!("first byte of libopus in {same_toc} of {packets} packets");
    println!(
        "mean over {rows} streams: C {:.2} dB, opus1 without the analysis {:.2} dB ({:+.2}), with it {:.2} dB ({:+.2})",
        sum[0] / r,
        sum[1] / r,
        (sum[1] - sum[0]) / r,
        sum[2] / r,
        (sum[2] - sum[0]) / r
    );
}
