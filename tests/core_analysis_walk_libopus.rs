//! `OpusEncoder` against libopus while the caller changes the settings in
//! the middle of a stream, with the speech and music analysis running
//! whenever the complexity is 7 or more.
//!
//! The other analysis tests keep the settings of a stream fixed. Here a
//! random walk changes one setting every few packets: the bitrate (with
//! `OPUS_AUTO` and `OPUS_BITRATE_MAX`), the complexity (across the gate of
//! the analysis at 7), the signal type, the bandwidth and the maximum
//! bandwidth, CBR, VBR and constrained VBR, the LSB depth, the packet loss
//! and in-band FEC, the prediction switch, the forced channel count, DTX,
//! the expert frame duration, the caller's frame size (2.5 to 120 ms), the
//! input type (float or 16-bit), output buffers too small to code anything,
//! and `OPUS_RESET_STATE`. Some float buffers hold a NaN, an infinity or a
//! huge value, which libopus's silence test, energy tracker, downmix and
//! input check each treat in their own way.
//!
//! 1. CELT-only streams (forced, or `RestrictedLowDelay`): every packet, its
//!    final range and `OPUS_GET_IN_DTX` are libopus's, at 8 to 48 kHz.
//! 2. Forced switches between CELT-only, SILK-only and Hybrid (mono): the
//!    CELT-only packets are libopus's, and so is the redundant 5 ms CELT
//!    frame at the end of a packet that leaves CELT-only. That frame is
//!    coded with the analysis of the packet; the CELT encoder forgets the
//!    analysis at each of its resets.
//!
//! Packets of more than 20 ms with a variable bitrate are left out on
//! purpose: opus1 does not limit each of their frames to its share of the
//! bitrate (see `encode_multiframe`). The packets depend on the rounding of
//! the analysis, which is libopus's on 64-bit ARM; elsewhere the tests only
//! print what they find.
#![cfg(feature = "analysis")]

mod core_common;
mod k24_common;

use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech_fullband, to_stereo};
use opus1::{
    Application, Bandwidth, Bitrate, Channels, ForcedMode, FrameDuration, InbandFec, OpusEncoder,
    SampleRate, Signal,
};

const EXACT: bool = cfg!(target_arch = "aarch64");
/// Room for six 20 ms CELT frames of 1275 bytes.
const MAX_PACKET: usize = 1276 * 6;
const OPUS_SET_EXPERT_FRAME_DURATION_REQUEST: i32 = 4040;

/// A small random number generator (64-bit linear congruential).
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// About 9 s of mono input with something for every part of the analysis
/// and of DTX: music, digital silence, speech, quiet noise, a tone below the
/// 16-bit silence threshold, a signal beyond full scale, a DC offset, and a
/// pause long enough for DTX.
fn mono_signal(fs: usize) -> Vec<f32> {
    let n = |s: f64| (s * fs as f64) as usize;
    let mut seed = 7u32;
    let mut noise = |len: usize, level: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                level * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
            })
            .collect()
    };
    let mut x = multitone(fs, n(0.6));
    x.extend(std::iter::repeat_n(0.0, n(0.3)));
    x.extend(speech_fullband(fs, 1.0));
    x.extend(noise(n(0.3), 3e-4));
    x.extend(multitone(fs, n(0.5)).iter().map(|v| 0.5 * v));
    // 1e-5 is below 2^-16 (digital silence for 16-bit input) and above
    // 2^-24.
    x.extend((0..n(0.3)).map(|i| 1e-5 * (i as f32 * 0.05).sin()));
    x.extend(multitone(fs, n(0.2)).iter().map(|v| 2.5 * v));
    x.extend(speech_fullband(fs, 1.0));
    x.extend(speech_fullband(fs, 0.4).iter().map(|v| 0.25 + 0.3 * v));
    x.extend(std::iter::repeat_n(0.0, n(0.7)));
    x.extend(noise(n(0.8), 1e-3));
    x.extend(multitone(fs, n(1.0)));
    x.extend(speech_fullband(fs, 1.0).iter().map(|v| 0.1 * v));
    x.extend(std::iter::repeat_n(0.0, n(0.9)));
    x
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rc {
    Cbr,
    Vbr,
    Cvbr,
}

const BANDWIDTHS: [(Bandwidth, i32); 5] = [
    (Bandwidth::Narrowband, BW_NB),
    (Bandwidth::Mediumband, BW_MB),
    (Bandwidth::Wideband, BW_WB),
    (Bandwidth::Superwideband, BW_SWB),
    (Bandwidth::Fullband, BW_FB),
];

const DURATIONS: [FrameDuration; 9] = [
    FrameDuration::Ms2_5,
    FrameDuration::Ms5,
    FrameDuration::Ms10,
    FrameDuration::Ms20,
    FrameDuration::Ms40,
    FrameDuration::Ms60,
    FrameDuration::Ms80,
    FrameDuration::Ms100,
    FrameDuration::Ms120,
];

/// Both encoders, set the same way.
struct Pair {
    fs: usize,
    channels: usize,
    rs: OpusEncoder,
    c: CEncoder,
    rc: Rc,
    /// The caller's frame size, samples per channel.
    frame: usize,
    expert: FrameDuration,
    i16_input: bool,
    /// Output buffer size, `MAX_PACKET` unless a small one is tried.
    out_len: usize,
    /// Packets left with the small output buffer.
    small_left: usize,
    /// Packets left whose float input gets a NaN, an infinity or a huge
    /// value.
    poison_left: usize,
    /// What was changed, for the failure message.
    log: Vec<String>,
    resets: usize,
}

impl Pair {
    fn new(fs: usize, channels: usize, app: Application, c_app: i32, force_celt: bool) -> Self {
        let mut rs = OpusEncoder::new(
            SampleRate::try_from(fs as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
            app,
        )
        .unwrap();
        let mut c = CEncoder::new(fs, channels, c_app);
        if force_celt {
            rs.set_force_mode(Some(ForcedMode::CeltOnly));
            c.force_mode(MODE_CELT);
        }
        Pair {
            fs,
            channels,
            rs,
            c,
            rc: Rc::Vbr,
            frame: fs / 50,
            expert: FrameDuration::Argument,
            i16_input: false,
            out_len: MAX_PACKET,
            small_left: 0,
            poison_left: 0,
            log: Vec::new(),
            resets: 0,
        }
    }

    /// The sampling rate of the pair.
    fn rate(&self) -> SampleRate {
        SampleRate::try_from(self.fs as u32).unwrap()
    }

    /// The frame the next call codes, in samples per channel.
    fn coded(&self) -> usize {
        self.expert.samples(self.rate()).unwrap_or(self.frame)
    }

    /// Whether the coded frame is a packet of several frames with a
    /// variable bitrate, which opus1 codes differently on purpose.
    fn deliberate_difference(&self) -> bool {
        self.rc != Rc::Cbr && self.coded() > self.fs / 50
    }

    fn set_rc(&mut self, rc: Rc) {
        self.rc = rc;
        self.rs.set_vbr(rc != Rc::Cbr);
        self.rs.set_vbr_constraint(rc == Rc::Cvbr);
        self.c.vbr(rc != Rc::Cbr);
        self.c.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, (rc == Rc::Cvbr) as i32);
    }

    /// One random change of the settings.
    fn change(&mut self, rng: &mut Lcg) {
        let what = rng.below(18);
        let note = match what {
            0 => {
                let (rs, c) = rng.pick(&[
                    (Bitrate::BitsPerSecond(6_000), 6_000),
                    (Bitrate::BitsPerSecond(9_000), 9_000),
                    (Bitrate::BitsPerSecond(16_000), 16_000),
                    (Bitrate::BitsPerSecond(24_000), 24_000),
                    (Bitrate::BitsPerSecond(32_000), 32_000),
                    (Bitrate::BitsPerSecond(48_000), 48_000),
                    (Bitrate::BitsPerSecond(64_000), 64_000),
                    (Bitrate::BitsPerSecond(96_000), 96_000),
                    (Bitrate::BitsPerSecond(160_000), 160_000),
                    (Bitrate::BitsPerSecond(320_000), 320_000),
                    (Bitrate::Auto, sys::OPUS_AUTO),
                    (Bitrate::Max, sys::OPUS_BITRATE_MAX),
                ]);
                self.rs.set_bitrate(rs).unwrap();
                self.c.bitrate(c);
                format!("bitrate {rs:?}")
            }
            1 | 2 => {
                let complexity = rng.below(11) as i32;
                self.rs.set_complexity(complexity).unwrap();
                self.c.complexity(complexity);
                format!("complexity {complexity}")
            }
            3 => {
                let (rs, c) = rng.pick(&[
                    (Signal::Auto, sys::OPUS_AUTO),
                    (Signal::Voice, sys::OPUS_SIGNAL_VOICE),
                    (Signal::Music, sys::OPUS_SIGNAL_MUSIC),
                ]);
                self.rs.set_signal(rs);
                self.c.set(sys::OPUS_SET_SIGNAL_REQUEST, c);
                format!("signal {rs:?}")
            }
            4 => {
                let (rs, c) = if rng.below(3) == 0 {
                    (Bandwidth::Auto, sys::OPUS_AUTO)
                } else {
                    rng.pick(&BANDWIDTHS)
                };
                self.rs.set_bandwidth(rs);
                self.c.bandwidth(c);
                format!("bandwidth {rs:?}")
            }
            5 => {
                let (rs, c) = rng.pick(&BANDWIDTHS);
                self.rs.set_max_bandwidth(rs).unwrap();
                self.c.max_bandwidth(c);
                format!("max bandwidth {rs:?}")
            }
            6 => {
                let rc = rng.pick(&[Rc::Cbr, Rc::Vbr, Rc::Cvbr]);
                self.set_rc(rc);
                format!("{rc:?}")
            }
            7 => {
                let depth = 8 + rng.below(17) as i32;
                self.rs.set_lsb_depth(depth).unwrap();
                self.c.set(sys::OPUS_SET_LSB_DEPTH_REQUEST, depth);
                format!("lsb depth {depth}")
            }
            8 => {
                let loss = rng.pick(&[0, 1, 3, 5, 9, 20]);
                let (rs, c) = rng.pick(&[(InbandFec::Off, 0), (InbandFec::On, 1), (InbandFec::OnUnlessMusic, 2)]);
                self.rs.set_packet_loss_perc(loss).unwrap();
                self.rs.set_inband_fec_mode(rs);
                self.c.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, loss);
                self.c.set(sys::OPUS_SET_INBAND_FEC_REQUEST, c);
                format!("loss {loss} fec {rs:?}")
            }
            9 => {
                let off = rng.below(4) == 0;
                self.rs.set_prediction_disabled(off);
                self.c.set(sys::OPUS_SET_PREDICTION_DISABLED_REQUEST, off as i32);
                format!("prediction disabled {off}")
            }
            10 if self.channels == 2 => {
                let (rs, c) =
                    rng.pick(&[(None, sys::OPUS_AUTO), (Some(Channels::Mono), 1), (Some(Channels::Stereo), 2)]);
                self.rs.set_force_channels(rs);
                self.c.force_channels(c);
                format!("force channels {rs:?}")
            }
            11 => {
                let on = rng.below(2) == 0;
                self.rs.set_dtx(on);
                self.c.set(sys::OPUS_SET_DTX_REQUEST, on as i32);
                format!("dtx {on}")
            }
            12 => {
                self.resets += 1;
                self.rs.reset_state();
                self.c.reset();
                "reset".to_string()
            }
            13 => {
                // The caller's frame size.
                let d = rng.pick(&DURATIONS);
                self.frame = d.samples(self.rate()).unwrap();
                if self.expert != FrameDuration::Argument && self.coded() > self.frame {
                    self.expert = FrameDuration::Argument;
                    self.rs.set_expert_frame_duration(self.expert);
                    self.c.set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, self.expert.to_raw());
                }
                format!("frame {} samples", self.frame)
            }
            14 => {
                // An expert frame duration no longer than the caller's frame.
                let fits: Vec<FrameDuration> = DURATIONS
                    .iter()
                    .copied()
                    .filter(|d| d.samples(self.rate()).unwrap() <= self.frame)
                    .collect();
                self.expert = if rng.below(3) == 0 { FrameDuration::Argument } else { rng.pick(&fits) };
                self.rs.set_expert_frame_duration(self.expert);
                self.c.set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, self.expert.to_raw());
                format!("expert {:?}", self.expert)
            }
            15 => {
                self.i16_input = !self.i16_input;
                format!("16-bit input {}", self.i16_input)
            }
            16 => {
                self.poison_left = 1 + rng.below(3);
                format!("{} buffers with odd values", self.poison_left)
            }
            _ => {
                // A few packets into a buffer too small for audio.
                self.out_len = rng.pick(&[1, 2, 3, 4, 7, 12]);
                self.small_left = 1 + rng.below(4);
                format!("output {} bytes", self.out_len)
            }
        };
        // A packet of several frames with a variable bitrate is a deliberate
        // difference: go back to 20 ms frames.
        if self.deliberate_difference() {
            self.frame = self.fs / 50;
            self.expert = FrameDuration::Argument;
            self.rs.set_expert_frame_duration(self.expert);
            self.c.set(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, self.expert.to_raw());
        }
        self.log.push(note);
    }
}

/// Both packets of one encode call: the bytes, or the error.
#[derive(Debug, PartialEq)]
enum Outcome {
    Packet(Vec<u8>),
    Error,
}

/// What the walks went through, to show that they reach the paths they are
/// meant to test.
#[derive(Default, Debug)]
struct Seen {
    packets: usize,
    /// Packets coded with a valid analysis.
    analysed: usize,
    /// Packets of 1 or 2 bytes (DTX, or a buffer too small for audio).
    tiny: usize,
    errors: usize,
    in_dtx: usize,
    resets: usize,
    /// Float buffers with a NaN, an infinity or a huge value.
    poisoned: usize,
}

/// One walk of `packets` encode calls. Returns the first difference, or
/// `None`.
fn walk(
    pair: &mut Pair,
    input: &[f32],
    packets: usize,
    rng: &mut Lcg,
    change_every: usize,
    seen: &mut Seen,
) -> Option<String> {
    let ch = pair.channels;
    let input16: Vec<i16> = input
        .iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let total = input.len() / ch;
    let mut at = 0usize;
    let mut out = vec![0u8; MAX_PACKET];
    for k in 0..packets {
        if rng.below(change_every) == 0 {
            pair.change(rng);
        }
        if pair.frame + at > total {
            at = 0;
        }
        let range = at * ch..(at + pair.frame) * ch;
        let out_len = pair.out_len;
        let (c, rs) = if pair.i16_input {
            let pcm = &input16[range];
            (pair.c.try_encode_i16(pcm, pair.frame, out_len), pair.rs.encode_i16(pcm, pair.frame, &mut out[..out_len]))
        } else if pair.poison_left > 0 {
            // One to three odd values anywhere in the caller's buffer: the
            // first or last sample, or one at random.
            pair.poison_left -= 1;
            seen.poisoned += 1;
            let mut pcm = input[range].to_vec();
            for _ in 0..1 + rng.below(3) {
                let at = match rng.below(3) {
                    0 => 0,
                    1 => pcm.len() - 1,
                    _ => rng.below(pcm.len()),
                };
                pcm[at] = rng.pick(&[f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -3e38, 2.5]);
            }
            (pair.c.try_encode(&pcm, pair.frame, out_len), pair.rs.encode(&pcm, pair.frame, &mut out[..out_len]))
        } else {
            let pcm = &input[range];
            (pair.c.try_encode(pcm, pair.frame, out_len), pair.rs.encode(pcm, pair.frame, &mut out[..out_len]))
        };
        let c = match c {
            Ok(p) => Outcome::Packet(p),
            Err(_) => Outcome::Error,
        };
        let rs = match rs {
            Ok(n) => Outcome::Packet(out[..n].to_vec()),
            Err(_) => Outcome::Error,
        };
        seen.packets += 1;
        match &rs {
            Outcome::Packet(p) => {
                seen.tiny += (p.len() <= 2) as usize;
                seen.analysed += opus1::internals::analysis::last_analysis_info(&pair.rs).valid as usize;
                seen.in_dtx += pair.rs.in_dtx() as usize;
            }
            Outcome::Error => seen.errors += 1,
        }
        let recent = pair.log.iter().rev().take(6).rev().cloned().collect::<Vec<_>>().join(", ");
        if c != rs {
            let show = |o: &Outcome| match o {
                Outcome::Packet(p) => format!("{} bytes {:02x?}", p.len(), &p[..p.len().min(8)]),
                Outcome::Error => "error".to_string(),
            };
            return Some(format!("packet {k}: libopus {}, opus1 {}; last changes: {recent}", show(&c), show(&rs)));
        }
        if let Outcome::Packet(_) = c {
            let (cr, rr) = (pair.c.final_range(), pair.rs.final_range());
            if cr != rr {
                return Some(format!("packet {k}: final range {cr:08x} against {rr:08x}; last changes: {recent}"));
            }
            let c_in_dtx = pair.c.get(sys::OPUS_GET_IN_DTX_REQUEST) != 0;
            if c_in_dtx != pair.rs.in_dtx() {
                return Some(format!("packet {k}: in DTX {c_in_dtx} against {}; last changes: {recent}", pair.rs.in_dtx()));
            }
        }
        // The caller passes again what was not coded, as libopus's
        // documentation of the expert frame duration asks.
        at += pair.coded();
        if pair.small_left > 0 {
            pair.small_left -= 1;
            if pair.small_left == 0 {
                pair.out_len = MAX_PACKET;
            }
        }
    }
    None
}

/// CELT-only streams with every setting changing in the middle of the
/// stream: every packet is libopus's.
#[test]
fn celt_only_streams_with_changing_settings_are_libopus() {
    let mut failures = Vec::new();
    let mut streams = 0;
    let mut seen = Seen::default();
    let mut rng = Lcg(2024);
    // 8 and 12 kHz run no analysis: DTX goes by the frame energy there.
    for fs in [8_000, 12_000, 16_000, 24_000, 48_000] {
        let mono = mono_signal(fs);
        let stereo = to_stereo(&mono, fs);
        for channels in [1, 2] {
            let input = if channels == 2 { &stereo } else { &mono };
            for (app, c_app) in [
                (Application::Audio, APP_AUDIO),
                (Application::Voip, APP_VOIP),
                (Application::RestrictedLowDelay, APP_LOWDELAY),
            ] {
                for change_every in [2, 6, 20] {
                    let mut pair = Pair::new(fs, channels, app, c_app, true);
                    let start = 7 + rng.below(4) as i32;
                    pair.rs.set_complexity(start).unwrap();
                    pair.c.complexity(start);
                    streams += 1;
                    if let Some(d) = walk(&mut pair, input, 600, &mut rng, change_every, &mut seen) {
                        failures.push(format!("{} kHz {channels} ch {app:?} change every {change_every}: {d}", fs / 1000));
                    }
                    seen.resets += pair.resets;
                }
            }
        }
    }
    println!("{streams} streams of 600 packets: {seen:?}");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} of {streams} streams differ from libopus", failures.len());
    }
}

/// libopus `compute_redundancy_bytes` (opus_encoder.c:1142): the size of
/// the redundant CELT frame of a mode switch, before the limit of what the
/// packet has left.
fn redundancy_bytes(max_data_bytes: i32, bitrate: i32, frame_rate: i32, channels: i32) -> usize {
    let base_bits = 40 * channels + 20;
    let rate = 3 * (bitrate + base_bits * (200 - frame_rate)) / 2;
    let available = max_data_bytes * 8 - 2 * base_bits;
    let cap = (available * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    let bytes = (rate / 1600).min(cap);
    if bytes > 4 + 8 * channels { bytes.min(257) as usize } else { 0 }
}

/// Forced switches between CELT-only, SILK-only and Hybrid in a mono `Audio`
/// stream, complexity 7 to 10, with a variable bitrate:
///
/// - every CELT-only packet is libopus's. After a switch to CELT-only the
///   CELT encoder goes on from the redundant frame, which starts from a
///   reset; so the SILK and Hybrid packets before leave nothing behind
///   that differs.
/// - the redundant CELT frame at the end of each packet that leaves or
///   enters CELT-only is libopus's. The frame that leaves CELT-only is coded
///   with the analysis of its packet. The one that enters CELT-only starts
///   from a reset, which drops the analysis.
///
/// The SILK bytes cannot be libopus's: opus1's SILK is the fixed-point one.
///
/// This found that the gain of the Hybrid high band (C `HB_gain`) used the
/// exact `exp` where libopus uses the `celt_exp2` polynomial of
/// `FLOAT_APPROX`: some redundant frames of a switch from Hybrid to
/// CELT-only, and the CELT-only packets after them, differed.
#[test]
fn forced_mode_switches_keep_the_celt_frames_of_libopus() {
    let mut failures = Vec::new();
    let (mut celt_packets, mut redundant_frames, mut streams) = (0, 0, 0);
    let mut rng = Lcg(77);
    for fs in [16_000usize, 24_000, 48_000] {
        let input = mono_signal(fs);
        for run in 0..24 {
            let complexity = 7 + rng.below(4) as i32;
            // Below about 32 kb/s Hybrid attenuates its high band (C
            // `HB_gain`), which the redundant frame of a switch to CELT-only
            // codes.
            let bitrate = rng.pick(&[16_000, 20_000, 24_000, 32_000, 48_000, 64_000, 96_000, 128_000]);
            let cvbr = rng.below(2) == 0;
            let frame = if rng.below(3) == 0 { fs / 100 } else { fs / 50 };
            let frame_rate = (fs / frame) as i32;
            let mut rs = OpusEncoder::new(
                SampleRate::try_from(fs as i32).unwrap(),
                Channels::Mono,
                Application::Audio,
            )
            .unwrap();
            let mut c = CEncoder::new(fs, 1, APP_AUDIO);
            rs.set_complexity(complexity).unwrap();
            c.complexity(complexity);
            rs.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            c.bitrate(bitrate);
            rs.set_vbr_constraint(cvbr);
            c.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, cvbr as i32);
            if run % 3 == 2 {
                rs.set_signal(Signal::Music);
                c.set(sys::OPUS_SET_SIGNAL_REQUEST, sys::OPUS_SIGNAL_MUSIC);
            }
            let (mut cp, mut rp) = (Vec::new(), Vec::new());
            let mut out = vec![0u8; 1276];
            let mut left = 0;
            for f in input.chunks_exact(frame).take(400) {
                if left == 0 {
                    left = 2 + rng.below(12);
                    // CELT-only, SILK-only (wideband) or Hybrid (fullband).
                    let modes: &[usize] = if fs > 16_000 { &[0, 1, 2] } else { &[0, 1] };
                    match rng.pick(modes) {
                        0 => {
                            rs.set_force_mode(Some(ForcedMode::CeltOnly));
                            rs.set_bandwidth(Bandwidth::Auto);
                            c.force_mode(MODE_CELT).bandwidth(sys::OPUS_AUTO);
                        }
                        1 => {
                            rs.set_force_mode(Some(ForcedMode::SilkOnly));
                            rs.set_bandwidth(Bandwidth::Wideband);
                            c.force_mode(MODE_SILK).bandwidth(BW_WB);
                        }
                        _ => {
                            rs.set_force_mode(Some(ForcedMode::Hybrid));
                            rs.set_bandwidth(Bandwidth::Fullband);
                            c.force_mode(MODE_HYBRID).bandwidth(BW_FB);
                        }
                    }
                }
                left -= 1;
                cp.push(c.encode_max(f, 1276));
                let n = rs.encode(f, frame, &mut out).unwrap();
                rp.push(out[..n].to_vec());
            }
            streams += 1;
            let label = format!("{} kHz c{complexity} {bitrate} b/s cvbr {cvbr} frame {frame} run {run}", fs / 1000);
            let celt = |p: &[u8]| p[0] >= 0x80;
            let r = redundancy_bytes(1276, bitrate, frame_rate, 1);
            for k in 0..cp.len() {
                let (a, b) = (&cp[k], &rp[k]);
                if celt(a) != celt(b) {
                    failures.push(format!("{label}: packet {k} mode {:02x} against {:02x}", a[0], b[0]));
                    break;
                }
                if celt(a) {
                    celt_packets += 1;
                    if a != b {
                        failures.push(format!("{label}: CELT-only packet {k} differs"));
                        break;
                    }
                    continue;
                }
                // A packet that leaves or enters CELT-only ends with the
                // redundant frame.
                let leaves = k > 0 && celt(&cp[k - 1]);
                let enters = k + 1 < cp.len() && celt(&cp[k + 1]);
                if (leaves || enters) && r > 0 {
                    redundant_frames += 1;
                    if a.len() <= r || b.len() <= r || a[a.len() - r..] != b[b.len() - r..] {
                        failures.push(format!(
                            "{label}: packet {k} ({}) has another redundant frame of {r} bytes",
                            if leaves { "leaves CELT" } else { "enters CELT" }
                        ));
                        break;
                    }
                }
            }
        }
    }
    println!("{streams} streams: {celt_packets} CELT-only packets, {redundant_frames} redundant frames");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    assert!(redundant_frames > 100, "too few switches");
    if EXACT {
        assert!(failures.is_empty(), "{} of {streams} streams differ from libopus", failures.len());
    }
}

/// Mono `Audio` streams where the encoder picks the mode itself, and the
/// analysis moves it between CELT-only and SILK or Hybrid. opus1's SILK is
/// the fixed-point one, so its packets are not libopus's, and SILK's
/// feedback can change later decisions. As long as the first byte of every
/// packet so far is libopus's:
///
/// - every CELT-only packet is libopus's, also after a return from SILK or
///   Hybrid (the CELT encoder then goes on from the redundant frame, which
///   starts from a reset);
/// - the redundant CELT frame at the end of a packet that leaves or enters
///   CELT-only is libopus's.
#[test]
fn automatic_mode_switches_keep_the_celt_frames_of_libopus() {
    let mut failures = Vec::new();
    let (mut celt_packets, mut redundant_frames, mut agreeing, mut total, mut switches) = (0, 0, 0, 0, 0);
    let mut rng = Lcg(91);
    for fs in [16_000usize, 24_000, 48_000] {
        let input = mono_signal(fs);
        for _ in 0..16 {
            let complexity = 7 + rng.below(4) as i32;
            let bitrate = rng.pick(&[12_000, 16_000, 20_000, 24_000, 32_000, 40_000, 48_000, 64_000]);
            let cvbr = rng.below(2) == 0;
            let frame = rng.pick(&[fs / 100, fs / 50]);
            let frame_rate = (fs / frame) as i32;
            let mut rs = OpusEncoder::new(
                SampleRate::try_from(fs as i32).unwrap(),
                Channels::Mono,
                Application::Audio,
            )
            .unwrap();
            let mut c = CEncoder::new(fs, 1, APP_AUDIO);
            rs.set_complexity(complexity).unwrap();
            c.complexity(complexity);
            rs.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            c.bitrate(bitrate);
            rs.set_vbr_constraint(cvbr);
            c.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, cvbr as i32);
            let mut out = vec![0u8; 1276];
            let (mut cp, mut rp) = (Vec::new(), Vec::new());
            for f in input.chunks_exact(frame) {
                cp.push(c.encode_max(f, 1276));
                let n = rs.encode(f, frame, &mut out).unwrap();
                rp.push(out[..n].to_vec());
            }
            let label = format!("{} kHz c{complexity} {bitrate} b/s cvbr {cvbr} frame {frame}", fs / 1000);
            let celt = |p: &[u8]| p[0] >= 0x80;
            let r = redundancy_bytes(1276, bitrate, frame_rate, 1);
            // The packets whose first bytes, and those of all packets before,
            // are libopus's.
            let same = cp.iter().zip(&rp).take_while(|(a, b)| a[0] == b[0]).count();
            agreeing += same;
            total += cp.len();
            for k in 0..same {
                let (a, b) = (&cp[k], &rp[k]);
                if k > 0 && celt(a) != celt(&cp[k - 1]) {
                    switches += 1;
                }
                if celt(a) {
                    celt_packets += 1;
                    if a != b {
                        failures.push(format!("{label}: CELT-only packet {k} differs"));
                        break;
                    }
                    continue;
                }
                let leaves = k > 0 && celt(&cp[k - 1]);
                let enters = k + 1 < same && celt(&cp[k + 1]);
                if (leaves || enters) && r > 0 {
                    redundant_frames += 1;
                    if a.len() <= r || b.len() <= r || a[a.len() - r..] != b[b.len() - r..] {
                        failures.push(format!(
                            "{label}: packet {k} ({}) has another redundant frame",
                            if leaves { "leaves CELT" } else { "enters CELT" }
                        ));
                        break;
                    }
                }
            }
        }
    }
    println!(
        "{agreeing} of {total} packets with libopus's first bytes so far, {switches} switches, \
         {celt_packets} CELT-only packets, {redundant_frames} redundant frames"
    );
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} streams differ from libopus", failures.len());
    }
}
