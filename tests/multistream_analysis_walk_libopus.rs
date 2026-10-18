//! `MultistreamEncoder` against libopus while the caller changes the
//! settings in the middle of a stream, every stream CELT-only, with the
//! speech and music analysis of each stream running whenever the complexity
//! is 7 or more.
//!
//! A random walk changes one setting every few packets: the bitrate, the
//! complexity (across the gate of the analysis at 7), the signal type, the
//! LSB depth, DTX, CBR, VBR and constrained VBR, the bandwidth, the packet
//! loss, the expert frame duration, the caller's frame size, the input type
//! (float or 16-bit), and `OPUS_RESET_STATE`. Every packet and its final
//! range are libopus's, for every layout of `multistream_enc_common::CASES`
//! (surround with its LFE stream, ambisonics, family 255, generic layouts).
//!
//! Two deliberate differences are left out: packets of more than 20 ms with
//! a variable bitrate (see `OpusEncoder::encode_multiframe`), and the
//! automatic and the maximum bitrate in surround mode, where opus1 chooses
//! the bandwidth from the allocated bitrate (docs/10, D1). The packets depend on the
//! rounding of the analysis, which is libopus's on 64-bit ARM; elsewhere the
//! test only prints what it finds.
#![cfg(all(feature = "std", feature = "analysis"))]

mod k24_common;
mod ms_support;
mod multistream_enc_common;

use audiopus_sys as sys;
use ms_support::*;
use multistream_enc_common::*;
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, Bandwidth, ForcedMode, FrameDuration, Signal};

const EXACT: bool = cfg!(target_arch = "aarch64");
const OPUS_SET_EXPERT_FRAME_DURATION_REQUEST: i32 = 4040;

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

/// `channels` channels of about 5 s: speech, music, digital silence, quiet
/// noise, and a pause long enough for DTX, each channel delayed and scaled
/// differently.
fn signal(fs: usize, channels: usize) -> Vec<f32> {
    let n = |s: f64| (s * fs as f64) as usize;
    let speech = k24_common::speech_fullband(fs, 2.2);
    let music = k24_common::multitone(fs, n(2.2));
    let mut seed = 3u32;
    let mut mono: Vec<Vec<f32>> = Vec::new();
    for c in 0..channels {
        let src = if c % 2 == 0 { &speech } else { &music };
        let gain = 1.0 - 0.08 * c as f32;
        let delay = 37 * c;
        let mut x: Vec<f32> = src[delay..delay + n(1.5)].iter().map(|v| gain * v).collect();
        x.extend(std::iter::repeat_n(0.0, n(0.8)));
        x.extend((0..n(0.6)).map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            1e-4 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
        }));
        x.extend(src[delay..delay + n(1.5)].iter().map(|v| 0.5 * gain * v));
        x.extend(std::iter::repeat_n(0.0, n(0.7)));
        mono.push(x);
    }
    let len = mono[0].len();
    let mut pcm = vec![0.0f32; len * channels];
    for (c, x) in mono.iter().enumerate() {
        for i in 0..len {
            pcm[i * channels + c] = x[i];
        }
    }
    pcm
}

const DURATIONS: [FrameDuration; 6] = [
    FrameDuration::Ms2_5,
    FrameDuration::Ms5,
    FrameDuration::Ms10,
    FrameDuration::Ms20,
    FrameDuration::Ms40,
    FrameDuration::Ms60,
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rc {
    Cbr,
    Vbr,
    Cvbr,
}

struct Walk {
    fs: usize,
    channels: usize,
    /// Surround mode (mapping family 1 with more than two channels).
    surround: bool,
    rc: Rc,
    frame: usize,
    expert: FrameDuration,
    i16_input: bool,
    log: Vec<String>,
}

impl Walk {
    fn coded(&self) -> usize {
        self.expert.samples(self.fs as i32).unwrap_or(self.frame)
    }

    fn change(&mut self, rust: &mut MultistreamEncoder, c: &mut CMsEncoder, rng: &mut Lcg) {
        let note = match rng.below(12) {
            0 => {
                let per_channel = rng.pick(&[6_000, 12_000, 16_000, 24_000, 32_000, 64_000, 128_000]);
                let value = per_channel * self.channels as i32;
                let value = if self.surround { value } else { rng.pick(&[value, OPUS_AUTO, OPUS_BITRATE_MAX]) };
                rust.set_bitrate(rust_bitrate(value)).unwrap();
                c.set_bitrate(value);
                format!("bitrate {value}")
            }
            1 | 2 => {
                let complexity = rng.below(11) as i32;
                rust.set_complexity(complexity).unwrap();
                c.set_complexity(complexity);
                format!("complexity {complexity}")
            }
            3 => {
                let (rs, cv) = rng.pick(&[
                    (Signal::Auto, OPUS_AUTO),
                    (Signal::Voice, sys::OPUS_SIGNAL_VOICE),
                    (Signal::Music, sys::OPUS_SIGNAL_MUSIC),
                ]);
                rust.set_signal(rs);
                c.ctl(sys::OPUS_SET_SIGNAL_REQUEST, cv);
                format!("signal {rs:?}")
            }
            4 => {
                let depth = 8 + rng.below(17) as i32;
                rust.set_lsb_depth(depth).unwrap();
                c.ctl(sys::OPUS_SET_LSB_DEPTH_REQUEST, depth);
                format!("lsb depth {depth}")
            }
            5 => {
                let on = rng.below(2) == 0;
                rust.set_dtx(on);
                c.ctl(sys::OPUS_SET_DTX_REQUEST, on as i32);
                format!("dtx {on}")
            }
            6 => {
                self.rc = rng.pick(&[Rc::Cbr, Rc::Vbr, Rc::Cvbr]);
                rust.set_vbr(self.rc != Rc::Cbr);
                rust.set_vbr_constraint(self.rc == Rc::Cvbr);
                c.set_vbr(self.rc != Rc::Cbr);
                c.ctl(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, (self.rc == Rc::Cvbr) as i32);
                format!("{:?}", self.rc)
            }
            7 => {
                rust.reset_state();
                c.ctl(sys::OPUS_RESET_STATE, 0);
                "reset".to_string()
            }
            8 => {
                let (rs, cv) = rng.pick(&[
                    (Bandwidth::Auto, OPUS_AUTO),
                    (Bandwidth::Narrowband, sys::OPUS_BANDWIDTH_NARROWBAND),
                    (Bandwidth::Wideband, sys::OPUS_BANDWIDTH_WIDEBAND),
                    (Bandwidth::Superwideband, sys::OPUS_BANDWIDTH_SUPERWIDEBAND),
                    (Bandwidth::Fullband, sys::OPUS_BANDWIDTH_FULLBAND),
                ]);
                rust.set_bandwidth(rs);
                c.ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, cv);
                format!("bandwidth {rs:?}")
            }
            9 => {
                let loss = rng.pick(&[0, 2, 5, 15]);
                rust.set_packet_loss_perc(loss).unwrap();
                c.set_packet_loss_perc(loss);
                format!("loss {loss}")
            }
            10 => {
                self.frame = rng.pick(&DURATIONS).samples(self.fs as i32).unwrap();
                let fits: Vec<FrameDuration> = DURATIONS
                    .iter()
                    .copied()
                    .filter(|d| d.samples(self.fs as i32).unwrap() <= self.frame)
                    .collect();
                self.expert = if rng.below(2) == 0 { FrameDuration::Argument } else { rng.pick(&fits) };
                rust.set_expert_frame_duration(self.expert);
                c.ctl(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, self.expert.to_raw());
                format!("frame {} expert {:?}", self.frame, self.expert)
            }
            _ => {
                self.i16_input = !self.i16_input;
                format!("16-bit input {}", self.i16_input)
            }
        };
        if self.rc != Rc::Cbr && self.coded() > self.fs / 50 {
            self.frame = self.fs / 50;
            self.expert = FrameDuration::Argument;
            rust.set_expert_frame_duration(self.expert);
            c.ctl(OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, self.expert.to_raw());
        }
        self.log.push(note);
    }
}

/// Every layout, all streams CELT-only, settings changing every few
/// packets: every packet and its final range are libopus's.
#[test]
fn celt_only_multistream_with_changing_settings_is_libopus() {
    let mut rng = Lcg(4242);
    let (mut streams, mut packets, mut failures) = (0, 0, Vec::new());
    for fs in [16_000usize, 24_000, 48_000] {
        for case in CASES {
            let pcm = signal(fs, case.channels);
            let pcm16: Vec<i16> = pcm.iter().map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
            let application = rng.pick(&[Application::Audio, Application::Voip]);
            let (mut rust, mut c) = encoders(case, fs as i32, application);
            configure(&mut rust, &mut c, 32_000 * case.channels as i32, true);
            let complexity = 7 + rng.below(4) as i32;
            rust.set_complexity(complexity).unwrap();
            c.set_complexity(complexity);
            rust.set_force_mode(Some(ForcedMode::CeltOnly));
            c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
            let mut walk = Walk {
                fs,
                channels: case.channels,
                surround: SURROUND_CASES.iter().any(|s| s.name == case.name),
                rc: Rc::Vbr,
                frame: fs / 50,
                expert: FrameDuration::Argument,
                i16_input: false,
                log: Vec::new(),
            };
            let max_bytes = max_packet_bytes(rust.layout().streams());
            let mut out = vec![0u8; max_bytes];
            let total = pcm.len() / case.channels;
            let mut at = 0;
            streams += 1;
            for k in 0..300 {
                if rng.below(3) == 0 {
                    walk.change(&mut rust, &mut c, &mut rng);
                }
                if at + walk.frame > total {
                    at = 0;
                }
                let range = at * case.channels..(at + walk.frame) * case.channels;
                let (cp, n) = if walk.i16_input {
                    (c.encode_i16(&pcm16[range.clone()], walk.frame, max_bytes), rust.encode_i16(&pcm16[range], walk.frame, &mut out))
                } else {
                    (c.encode_float(&pcm[range.clone()], walk.frame, max_bytes), rust.encode(&pcm[range], walk.frame, &mut out))
                };
                packets += 1;
                let recent = walk.log.iter().rev().take(5).rev().cloned().collect::<Vec<_>>().join(", ");
                let what = format!("{} {fs} Hz {application:?}: packet {k}", case.name);
                match (cp, n) {
                    (Ok(cp), Ok(n)) => {
                        if cp != out[..n] {
                            failures.push(format!("{what} differs; last changes: {recent}"));
                            break;
                        }
                        if c.final_range() != rust.final_range() {
                            failures.push(format!("{what}: final range; last changes: {recent}"));
                            break;
                        }
                    }
                    (Err(_), Err(_)) => {}
                    (cp, n) => {
                        failures.push(format!("{what}: libopus {:?}, opus1 {:?}; last changes: {recent}", cp.map(|p| p.len()), n));
                        break;
                    }
                }
                at += walk.coded();
            }
        }
    }
    println!("{streams} streams, {packets} packets");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} of {streams} streams are not libopus's", failures.len());
    }
}
