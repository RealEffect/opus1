//! Performance and bit-exactness harness: opus-rs against C libopus.
//!
//! ```text
//! cargo run --release --example perf_harness -- <command> [options]
//! ```
//!
//! Commands:
//!
//! * `list` prints every scenario with its settings.
//! * `bench` times four operations per scenario: Rust encode, C encode,
//!   Rust decode and C decode.
//! * `run` loops one operation so that a sampling profiler can attach.
//! * `golden` writes or checks hashes of the codec output. It tells whether
//!   a change to the codec altered what the codec produces.
//! * `modes` compares the Rust and the C packet streams and the two
//!   decoders.
//! * `settings` prints the encoder settings both libraries report after
//!   encoding (libopus: read back with its `OPUS_GET_*` requests).
//! * `signals` prints what the test signals contain.
//! * `resampler` checks the resampler behind the speech signals with sine
//!   tones.
//! * `spread` compares several `bench --json` files.
//!
//! The C side is libopus, linked through `audiopus_sys`. Both encoders are
//! forced to the coding mode, the audio bandwidth and the channel count
//! that the scenario names (`OPUS_SET_FORCE_MODE`, `OPUS_SET_BANDWIDTH` and
//! `OPUS_SET_FORCE_CHANNELS`; in opus-rs `set_force_mode`, `set_bandwidth`
//! and `set_force_channels`). Both sides then do the same kind of work. The
//! mode-switching streams of `golden` and `modes` are the exception: there
//! only the channel count is forced.
//!
//! The two encoders still do not do identical work. Their packets are almost
//! never byte-identical, and:
//!
//! * The C library is a float build. Its SILK encoder is the floating-point
//!   one (`silk/float`); the SILK encoder of opus-rs is a port of the
//!   fixed-point one.
//! * libopus 1.6 runs two encoder steps that opus-rs does not have. They
//!   are listed at `Scenario::c_only_steps`, and `bench` names them per
//!   scenario in the column `only C runs`.
//!
//! `bench` and `modes` report every scenario in which the two sides differ
//! in encoder settings, packet configuration, bit rate or decoded audio.
//!
//! Timings also depend on where the linker places the code; see "Code
//! placement" below.

use std::collections::BTreeMap;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::hint::black_box;
use std::io::Write as _;
use std::os::raw::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use audiopus_sys as ffi;
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::silk::decode_frame::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST};
use opus_rs::{Application, ForcedMode, OpusDecoder, OpusEncoder};

/// Output buffer handed to both encoders. It is larger than any packet, so
/// it never limits the bit rate.
const MAX_PACKET: usize = 4000;
/// Clip length used by `golden` and `modes`. It is fixed so that a golden
/// file never depends on a command line option.
const GOLDEN_SECONDS: f64 = 5.0;
/// `golden` replaces every packet whose index is 9, 19, 29, ... by a lost
/// packet.
const LOSS_PERIOD: usize = 10;
/// For the burst-loss decode, `golden` also drops packets 12, 13 and 14 out
/// of every 25.
const BURST_PERIOD: usize = 25;
const BURST_START: usize = 12;
const BURST_LENGTH: usize = 3;
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
/// Peak level of every test signal: 0.5 is -6 dB relative to full scale.
const PEAK: f64 = 0.5;

/// Private libopus request (`opus_private.h`): force the coding mode.
const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
const MODE_SILK_ONLY: c_int = 1000;
const MODE_HYBRID: c_int = 1001;
const MODE_CELT_ONLY: c_int = 1002;

// ---------------------------------------------------------------------------
// Code placement
// ---------------------------------------------------------------------------
//
// The speed of some codec functions depends on the address at which the
// linker places them. Measured on an Apple M4 Pro: the same machine code of
// `silk_warped_autocorrelation_fix` has a fast and a slow state, about 1.7
// times apart, and the placement decides which one a build gets. That
// changes the SILK encode time at complexity 5 by 11 to 17 %. No simple rule
// for the slow placements was found. Any change to this program or to the
// library moves the code, so a timing difference of that size between two
// builds may have nothing to do with the change.
//
// Example: two builds of this program that differed in a comment and one
// local variable gave 35.7 and 31.1 microseconds per frame for the Rust
// encoder of `silk_nb_8k_mono_12k_vbr_c5_20ms`, and 64.3 and 55.3 for
// `silk_wb_16k_mono_20k_vbr_c5_20ms`. No other bench row moved by more than
// the usual noise.
//
// To tell the two apart, build this program with the environment variable
// `PERF_HARNESS_PAD=<bytes>` (for example 0, 20, 40 and 60). The padding
// below then moves all library code by about that many bytes, and `bench`
// can be repeated at several placements. The `spread` command compares the
// results. The padding exists on macOS only.

/// Parse a decimal number at compile time. Anything else gives 0.
const fn parse_decimal(text: Option<&str>) -> usize {
    let Some(text) = text else {
        return 0;
    };
    let bytes = text.as_bytes();
    let mut value = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || value > 1 << 20 {
            return 0;
        }
        value = value * 10 + (bytes[i] - b'0') as usize;
        i += 1;
    }
    value
}

/// Bytes of padding in the code section, from `PERF_HARNESS_PAD` at build
/// time.
const LAYOUT_PAD: usize = parse_decimal(option_env!("PERF_HARNESS_PAD"));

/// The padding itself. It is never read or executed. The object file of this
/// program is linked before the library, so the padding moves the library
/// code.
#[cfg(target_os = "macos")]
#[used]
#[unsafe(link_section = "__TEXT,__text")]
static LAYOUT_PADDING: [u8; LAYOUT_PAD] = [0; LAYOUT_PAD];

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Silk,
    Hybrid,
    Celt,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum App {
    Voip,
    Audio,
    LowDelay,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Signal {
    Speech,
    Music,
}

/// Audio bandwidth. A scenario codes the widest one its input sample rate
/// can carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

impl Bandwidth {
    /// The libopus constant for this bandwidth.
    fn c_value(self) -> i32 {
        match self {
            Bandwidth::Narrow => ffi::OPUS_BANDWIDTH_NARROWBAND,
            Bandwidth::Medium => ffi::OPUS_BANDWIDTH_MEDIUMBAND,
            Bandwidth::Wide => ffi::OPUS_BANDWIDTH_WIDEBAND,
            Bandwidth::SuperWide => ffi::OPUS_BANDWIDTH_SUPERWIDEBAND,
            Bandwidth::Full => ffi::OPUS_BANDWIDTH_FULLBAND,
        }
    }

    /// The opus-rs value for this bandwidth.
    fn rust_value(self) -> opus_rs::Bandwidth {
        match self {
            Bandwidth::Narrow => opus_rs::Bandwidth::Narrowband,
            Bandwidth::Medium => opus_rs::Bandwidth::Mediumband,
            Bandwidth::Wide => opus_rs::Bandwidth::Wideband,
            Bandwidth::SuperWide => opus_rs::Bandwidth::Superwideband,
            Bandwidth::Full => opus_rs::Bandwidth::Fullband,
        }
    }
}

#[derive(Clone, Debug)]
struct Scenario {
    /// Coding mode. Both encoders are forced to it.
    mode: Mode,
    /// Encoder input sample rate in Hz.
    rate: u32,
    channels: usize,
    app: App,
    /// Target bit rate in bits per second.
    bitrate: i32,
    /// Frame duration in microseconds (2500 to 60000).
    frame_us: u32,
    complexity: i32,
    cbr: bool,
    /// In-band forward error correction (SILK redundancy data).
    fec: bool,
    /// Expected packet loss in percent, as told to the encoder.
    loss_perc: i32,
    signal: Signal,
    /// Decoder output sample rate in Hz.
    dec_rate: u32,
    /// Insert digital silence into the signal: 0.4 s out of every second.
    /// This reaches the silence and inactivity handling of the codec.
    pauses: bool,
    /// Member of the smaller set that `bench` times by default.
    bench: bool,
}

impl Scenario {
    fn new(mode: Mode, rate: u32, channels: usize, bitrate: i32, complexity: i32) -> Self {
        Self {
            mode,
            rate,
            channels,
            app: Self::default_app(mode),
            bitrate,
            frame_us: 20_000,
            complexity,
            cbr: false,
            fec: false,
            loss_perc: 0,
            signal: Self::default_signal(mode),
            dec_rate: rate,
            pauses: false,
            bench: false,
        }
    }

    fn default_app(mode: Mode) -> App {
        match mode {
            Mode::Celt => App::Audio,
            Mode::Silk | Mode::Hybrid => App::Voip,
        }
    }

    fn default_signal(mode: Mode) -> Signal {
        match mode {
            Mode::Celt => Signal::Music,
            Mode::Silk | Mode::Hybrid => Signal::Speech,
        }
    }

    fn frame(mut self, frame_us: u32) -> Self {
        self.frame_us = frame_us;
        self
    }

    fn cbr(mut self) -> Self {
        self.cbr = true;
        self
    }

    fn fec(mut self, loss_perc: i32) -> Self {
        self.fec = true;
        self.loss_perc = loss_perc;
        self
    }

    fn app(mut self, app: App) -> Self {
        self.app = app;
        self
    }

    fn signal(mut self, signal: Signal) -> Self {
        self.signal = signal;
        self
    }

    fn dec(mut self, dec_rate: u32) -> Self {
        self.dec_rate = dec_rate;
        self
    }

    fn pauses(mut self) -> Self {
        self.pauses = true;
        self
    }

    fn bench(mut self) -> Self {
        self.bench = true;
        self
    }

    fn bandwidth(&self) -> Bandwidth {
        match self.rate {
            8000 => Bandwidth::Narrow,
            12000 => Bandwidth::Medium,
            16000 => Bandwidth::Wide,
            24000 => Bandwidth::SuperWide,
            _ => Bandwidth::Full,
        }
    }

    /// The bandwidth that ends up in the packets. CELT has no medium band:
    /// both libraries code 12 kHz input as wide band.
    fn coded_bandwidth(&self) -> Bandwidth {
        match (self.mode, self.bandwidth()) {
            (Mode::Celt, Bandwidth::Medium) => Bandwidth::Wide,
            (_, bandwidth) => bandwidth,
        }
    }

    /// Encoder steps that libopus 1.6 (float build) runs for these settings
    /// and opus-rs does not have. Checked against the sources of both:
    ///
    /// * `analysis`: the tonality analysis, `run_analysis` in `analysis.c`.
    ///   libopus runs it at complexity 7 and above for input from 16 kHz
    ///   (`opus_encoder.c`). opus-rs never fills its `AnalysisInfo`.
    /// * `stereo-angle`: for stereo CELT bands at complexity 8 and above,
    ///   libopus quantises each band twice and keeps the better stereo angle
    ///   (`theta_rdo` in `bands.c`). opus-rs keeps `theta_round` at 0.
    ///
    /// The CELT pitch pre-filter on stereo input is no longer listed: opus-rs
    /// runs `run_prefilter` for every input channel, as libopus does
    /// (`src/celt.rs`, the `cc` channels of `in_buf`).
    ///
    /// The C encoder time of a scenario that lists a step includes that
    /// step, so its Rust/C encode ratio flatters opus-rs.
    ///
    /// The list holds the large steps only. `opus_encoder.c` also has small
    /// per-frame steps without a counterpart in opus-rs: the digital silence
    /// check and the frame energy tracker, which serve discontinuous
    /// transmission. (The stereo width estimate and the gain reduction of
    /// the upper band in Hybrid are ported.)
    fn c_only_steps(&self) -> Vec<&'static str> {
        let mut steps = Vec::new();
        if self.complexity >= 7 && self.rate >= 16000 {
            steps.push("analysis");
        }
        if self.channels == 2 && self.mode != Mode::Silk && self.complexity >= 8 {
            steps.push("stereo-angle");
        }
        steps
    }

    fn name(&self) -> String {
        let mode = match self.mode {
            Mode::Silk => "silk",
            Mode::Hybrid => "hybrid",
            Mode::Celt => "celt",
        };
        let bandwidth = match self.bandwidth() {
            Bandwidth::Narrow => "nb",
            Bandwidth::Medium => "mb",
            Bandwidth::Wide => "wb",
            Bandwidth::SuperWide => "swb",
            Bandwidth::Full => "fb",
        };
        let mut name = format!(
            "{mode}_{bandwidth}_{}k_{}_{}k_{}_c{}_{}",
            self.rate / 1000,
            if self.channels == 2 { "stereo" } else { "mono" },
            self.bitrate / 1000,
            if self.cbr { "cbr" } else { "vbr" },
            self.complexity,
            frame_label(self.frame_us),
        );
        if self.fec {
            let _ = write!(name, "_fec{}", self.loss_perc);
        }
        if self.app != Self::default_app(self.mode) {
            name.push_str(match self.app {
                App::Voip => "_voip",
                App::Audio => "_audio",
                App::LowDelay => "_lowdelay",
            });
        }
        if self.signal != Self::default_signal(self.mode) {
            name.push_str(match self.signal {
                Signal::Speech => "_speech",
                Signal::Music => "_music",
            });
        }
        if self.pauses {
            name.push_str("_pauses");
        }
        if self.dec_rate != self.rate {
            let _ = write!(name, "_dec{}k", self.dec_rate / 1000);
        }
        name
    }

    /// Samples per channel in one encoder frame.
    fn frame_samples(&self) -> usize {
        (self.rate as u64 * self.frame_us as u64 / 1_000_000) as usize
    }

    /// Samples per channel that the decoder returns for one packet.
    fn dec_frame_samples(&self) -> usize {
        (self.dec_rate as u64 * self.frame_us as u64 / 1_000_000) as usize
    }

    /// The packet configuration this scenario should produce: the upper six
    /// bits of the first packet byte (5 bits configuration, 1 bit stereo).
    /// It follows the libopus rules.
    fn expected_toc_key(&self) -> u8 {
        let bandwidth = self.bandwidth();
        let config = match self.mode {
            Mode::Silk => {
                let bw = match bandwidth {
                    Bandwidth::Narrow => 0,
                    Bandwidth::Medium => 1,
                    _ => 2,
                };
                let duration = match self.frame_us {
                    10_000 => 0,
                    20_000 => 1,
                    40_000 => 2,
                    _ => 3,
                };
                bw * 4 + duration
            }
            Mode::Hybrid => {
                let bw = if bandwidth == Bandwidth::Full { 2 } else { 0 };
                // 40 and 60 ms packets carry 20 ms frames.
                12 + bw + u8::from(self.frame_us >= 20_000)
            }
            Mode::Celt => {
                // CELT has no medium band: libopus codes it as wide band.
                let bw = match bandwidth {
                    Bandwidth::Narrow => 0,
                    Bandwidth::Medium | Bandwidth::Wide => 1,
                    Bandwidth::SuperWide => 2,
                    Bandwidth::Full => 3,
                };
                let duration = match self.frame_us {
                    2_500 => 0,
                    5_000 => 1,
                    10_000 => 2,
                    _ => 3,
                };
                16 + bw * 4 + duration
            }
        };
        (config << 1) | u8::from(self.channels == 2)
    }
}

fn frame_label(frame_us: u32) -> String {
    if frame_us.is_multiple_of(1000) {
        format!("{}ms", frame_us / 1000)
    } else {
        format!("{}p{}ms", frame_us / 1000, (frame_us % 1000) / 100)
    }
}

/// Every scenario. Those marked `.bench()` form the bench set; all of them
/// together form the golden set.
///
/// Both encoders are forced to the mode a scenario names, so the bit rate
/// need not be one at which an encoder would pick that mode by itself. Most
/// are: SILK and Hybrid at speech rates, CELT at music rates.
fn scenarios() -> Vec<Scenario> {
    use Mode::{Celt, Hybrid, Silk};
    let s = Scenario::new;
    let mut list = Vec::new();

    // ---- bench set ------------------------------------------------------
    for complexity in [0, 5, 10] {
        list.push(s(Silk, 8000, 1, 12_000, complexity).bench());
    }
    for complexity in [0, 5, 10] {
        list.push(s(Silk, 16000, 1, 20_000, complexity).bench());
    }
    list.push(s(Silk, 16000, 1, 20_000, 9).frame(10_000).bench());
    list.push(s(Silk, 16000, 2, 32_000, 9).bench());
    list.push(s(Silk, 16000, 1, 24_000, 9).fec(10).bench());
    // Hybrid is timed at constant bit rate: at variable bit rate opus-rs
    // spends far more bits than libopus in this mode, so the two encoders
    // would not do the same amount of work. The variable bit rate versions
    // are in the golden set below.
    list.push(s(Hybrid, 24000, 1, 24_000, 9).cbr().bench());
    list.push(s(Hybrid, 24000, 2, 36_000, 9).cbr().bench());
    list.push(s(Hybrid, 48000, 1, 32_000, 9).cbr().bench());
    list.push(s(Hybrid, 48000, 2, 40_000, 9).cbr().bench());
    for complexity in [0, 5, 10] {
        list.push(s(Celt, 48000, 1, 64_000, complexity).bench());
    }
    for complexity in [0, 5, 10] {
        list.push(s(Celt, 48000, 2, 128_000, complexity).bench());
    }
    for frame_us in [2_500, 5_000, 10_000] {
        list.push(s(Celt, 48000, 2, 128_000, 9).frame(frame_us).bench());
    }
    list.push(s(Celt, 24000, 1, 48_000, 9).bench());
    list.push(s(Celt, 24000, 2, 96_000, 9).bench());
    list.push(s(Celt, 48000, 1, 24_000, 9).bench());
    list.push(s(Celt, 48000, 2, 256_000, 9).bench());
    // Hybrid on the music signal. The speech fixture has almost no energy
    // above 4 kHz, so only the music scenarios give the CELT part of Hybrid
    // (8 to 20 kHz) real content.
    list.push(s(Hybrid, 48000, 1, 32_000, 9).cbr().signal(Signal::Music).bench());
    list.push(s(Hybrid, 48000, 2, 40_000, 9).cbr().signal(Signal::Music).bench());

    // ---- golden set only ------------------------------------------------
    // Medium band (12 kHz input).
    list.push(s(Silk, 12000, 1, 16_000, 9));
    list.push(s(Silk, 12000, 1, 16_000, 9).cbr());
    // 40 and 60 ms frames.
    list.push(s(Silk, 16000, 1, 20_000, 9).frame(40_000));
    list.push(s(Silk, 16000, 1, 20_000, 9).frame(60_000));
    list.push(s(Silk, 8000, 1, 12_000, 9).frame(60_000));
    list.push(s(Celt, 48000, 2, 128_000, 9).frame(40_000));
    list.push(s(Celt, 48000, 1, 64_000, 9).frame(60_000));
    list.push(s(Hybrid, 48000, 1, 32_000, 9).frame(40_000));
    list.push(s(Hybrid, 24000, 1, 24_000, 9).frame(60_000));
    list.push(s(Hybrid, 48000, 1, 32_000, 9).frame(10_000));
    // Hybrid at variable bit rate.
    list.push(s(Hybrid, 24000, 1, 24_000, 9));
    list.push(s(Hybrid, 24000, 2, 36_000, 9));
    list.push(s(Hybrid, 48000, 1, 32_000, 9));
    list.push(s(Hybrid, 48000, 2, 40_000, 9));
    // Constant bit rate.
    list.push(s(Silk, 8000, 1, 12_000, 9).cbr());
    list.push(s(Silk, 16000, 1, 20_000, 9).cbr());
    list.push(s(Silk, 16000, 2, 32_000, 9).cbr());
    list.push(s(Celt, 48000, 1, 64_000, 9).cbr());
    list.push(s(Celt, 48000, 2, 128_000, 9).cbr());
    list.push(s(Celt, 48000, 2, 128_000, 9).frame(2_500).cbr());
    list.push(s(Celt, 24000, 1, 48_000, 9).cbr());
    // More complexity settings.
    list.push(s(Silk, 16000, 1, 20_000, 2));
    list.push(s(Silk, 16000, 1, 20_000, 7));
    list.push(s(Celt, 48000, 2, 128_000, 2));
    list.push(s(Celt, 48000, 2, 128_000, 7));
    // Application "restricted low delay": CELT only, no look-ahead.
    list.push(s(Celt, 48000, 1, 64_000, 9).frame(10_000).app(App::LowDelay));
    list.push(s(Celt, 48000, 2, 128_000, 9).frame(5_000).app(App::LowDelay));
    list.push(s(Celt, 48000, 2, 128_000, 9).frame(2_500).app(App::LowDelay));
    list.push(s(Celt, 24000, 1, 48_000, 9).app(App::LowDelay));
    list.push(s(Celt, 16000, 1, 32_000, 9).app(App::LowDelay));
    list.push(s(Celt, 8000, 1, 24_000, 9).app(App::LowDelay));
    // Bit rates from 6 to 510 kb/s.
    list.push(s(Silk, 8000, 1, 6_000, 9));
    list.push(s(Silk, 16000, 1, 40_000, 9));
    list.push(s(Celt, 48000, 2, 64_000, 9));
    list.push(s(Celt, 48000, 2, 510_000, 9));
    // 24 kHz input, more cases.
    list.push(s(Celt, 24000, 1, 48_000, 9).frame(10_000));
    list.push(s(Celt, 24000, 1, 48_000, 9).frame(2_500));
    list.push(s(Celt, 24000, 2, 96_000, 9).frame(5_000));
    list.push(s(Celt, 24000, 2, 96_000, 9).cbr());
    list.push(s(Hybrid, 24000, 1, 24_000, 9).frame(10_000));
    list.push(s(Hybrid, 24000, 2, 36_000, 9).frame(40_000));
    // CELT mono at the short frame sizes, and CELT on 12 kHz input (coded
    // as wide band).
    list.push(s(Celt, 48000, 1, 64_000, 9).frame(2_500));
    list.push(s(Celt, 48000, 1, 64_000, 9).frame(5_000));
    list.push(s(Celt, 48000, 1, 64_000, 9).frame(10_000));
    list.push(s(Celt, 12000, 1, 24_000, 9).app(App::LowDelay));
    // Hybrid stereo at 10 ms, and Hybrid at low complexity.
    list.push(s(Hybrid, 48000, 2, 40_000, 9).frame(10_000).cbr());
    list.push(s(Hybrid, 48000, 1, 32_000, 0).cbr());
    list.push(s(Hybrid, 48000, 1, 32_000, 5).cbr());
    // SILK at the other frame sizes and bandwidths, mono and stereo.
    list.push(s(Silk, 8000, 1, 12_000, 9).frame(10_000));
    list.push(s(Silk, 8000, 1, 12_000, 9).frame(40_000));
    list.push(s(Silk, 12000, 1, 16_000, 9).frame(10_000));
    list.push(s(Silk, 12000, 1, 16_000, 9).frame(60_000));
    list.push(s(Silk, 12000, 2, 24_000, 9));
    list.push(s(Silk, 16000, 2, 32_000, 9).frame(10_000));
    list.push(s(Silk, 16000, 2, 32_000, 9).frame(40_000));
    list.push(s(Silk, 16000, 2, 32_000, 9).frame(60_000));
    list.push(s(Silk, 16000, 2, 32_000, 0));
    // Decoder output at every supported rate.
    for dec_rate in [8000, 12000, 16000, 24000] {
        list.push(s(Celt, 48000, 2, 128_000, 9).dec(dec_rate));
    }
    for dec_rate in [8000, 12000, 24000, 48000] {
        list.push(s(Silk, 16000, 1, 20_000, 9).dec(dec_rate));
    }
    for dec_rate in [16000, 24000] {
        list.push(s(Hybrid, 48000, 1, 32_000, 9).dec(dec_rate));
    }
    list.push(s(Celt, 24000, 1, 48_000, 9).dec(48000));
    list.push(s(Hybrid, 24000, 1, 24_000, 9).dec(48000));
    for dec_rate in [8000, 12000, 16000, 24000] {
        list.push(s(Celt, 48000, 1, 64_000, 9).dec(dec_rate));
    }
    list.push(s(Celt, 48000, 2, 128_000, 9).frame(2_500).dec(24000));
    list.push(s(Celt, 24000, 2, 96_000, 9).dec(48000));
    list.push(s(Celt, 24000, 2, 96_000, 9).dec(16000));
    for dec_rate in [16000, 48000] {
        list.push(s(Silk, 8000, 1, 12_000, 9).dec(dec_rate));
    }
    for dec_rate in [8000, 48000] {
        list.push(s(Silk, 12000, 1, 16_000, 9).dec(dec_rate));
    }
    for dec_rate in [8000, 48000] {
        list.push(s(Silk, 16000, 2, 32_000, 9).dec(dec_rate));
    }
    for dec_rate in [8000, 12000] {
        list.push(s(Hybrid, 48000, 1, 32_000, 9).cbr().dec(dec_rate));
    }
    list.push(s(Hybrid, 48000, 2, 40_000, 9).cbr().dec(24000));
    list.push(s(Hybrid, 24000, 2, 36_000, 9).cbr().dec(48000));
    // In-band forward error correction.
    list.push(s(Silk, 16000, 1, 24_000, 9).fec(20));
    list.push(s(Silk, 8000, 1, 20_000, 9).fec(10));
    list.push(s(Hybrid, 48000, 1, 32_000, 9).fec(10));
    list.push(s(Silk, 12000, 1, 16_000, 9).fec(10));
    list.push(s(Silk, 16000, 1, 24_000, 9).frame(10_000).fec(10));
    list.push(s(Silk, 16000, 1, 24_000, 9).frame(60_000).fec(10));
    list.push(s(Silk, 16000, 2, 32_000, 9).fec(10));
    list.push(s(Hybrid, 48000, 2, 40_000, 9).cbr().fec(10));
    list.push(s(Hybrid, 24000, 1, 24_000, 9).cbr().fec(10));
    // Other applications and signals.
    list.push(s(Silk, 16000, 1, 20_000, 9).app(App::Audio));
    list.push(s(Celt, 48000, 1, 96_000, 9).app(App::Voip).signal(Signal::Speech));
    list.push(s(Silk, 8000, 2, 20_000, 9));
    // Signals with pauses of digital silence.
    list.push(s(Silk, 16000, 1, 20_000, 9).pauses());
    list.push(s(Hybrid, 48000, 1, 32_000, 9).cbr().pauses());
    list.push(s(Celt, 48000, 2, 128_000, 9).pauses());

    list
}

fn find_scenario(name: &str) -> Result<Scenario, String> {
    let all = scenarios();
    if let Some(sc) = all.iter().find(|sc| sc.name() == name) {
        return Ok(sc.clone());
    }
    let partial: Vec<&Scenario> = all.iter().filter(|sc| sc.name().contains(name)).collect();
    match partial.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(format!("no scenario named '{name}' (see the `list` command)")),
        many => {
            let names: Vec<String> = many.iter().map(|sc| sc.name()).collect();
            Err(format!("'{name}' matches several scenarios: {}", names.join(", ")))
        }
    }
}

/// `filter` is a comma-separated list of substrings; one match is enough.
fn name_matches(name: &str, filter: Option<&str>) -> bool {
    filter.is_none_or(|f| f.split(',').any(|part| name.contains(part)))
}

// ---------------------------------------------------------------------------
// Test signals
// ---------------------------------------------------------------------------

/// xorshift64 generator with a fixed seed: the signals never change.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform value in [-1, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
    }
}

/// White noise through three one-pole low-pass filters (Paul Kellet's
/// "economy" filter). The spectrum falls by roughly 3 dB per octave.
struct PinkNoise {
    rng: XorShift,
    state: [f64; 3],
}

impl PinkNoise {
    fn new(seed: u64) -> Self {
        Self {
            rng: XorShift(seed),
            state: [0.0; 3],
        }
    }

    fn next(&mut self) -> f64 {
        let white = self.rng.uniform();
        self.state[0] = 0.99765 * self.state[0] + white * 0.099_046;
        self.state[1] = 0.96300 * self.state[1] + white * 0.296_516_4;
        self.state[2] = 0.57000 * self.state[2] + white * 1.052_691_3;
        self.state[0] + self.state[1] + self.state[2] + white * 0.1848
    }
}

fn peak_of(channels: &[Vec<f64>]) -> f64 {
    channels
        .iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()))
}

fn normalize(channels: &mut [Vec<f64>]) {
    let peak = peak_of(channels);
    if peak > 0.0 {
        let gain = PEAK / peak;
        for v in channels.iter_mut().flat_map(|ch| ch.iter_mut()) {
            *v *= gain;
        }
    }
}

/// Add pink noise to one channel. `rms` is the root-mean-square level of the
/// added noise.
fn add_pink_noise(channel: &mut [f64], seed: u64, rms: f64) {
    let mut generator = PinkNoise::new(seed);
    let noise: Vec<f64> = (0..channel.len()).map(|_| generator.next()).collect();
    let power = noise.iter().map(|v| v * v).sum::<f64>() / noise.len().max(1) as f64;
    let gain = if power > 0.0 { rms / power.sqrt() } else { 0.0 };
    for (dst, n) in channel.iter_mut().zip(&noise) {
        *dst += gain * n;
    }
}

/// Modified Bessel function of the first kind, order zero (for the Kaiser
/// window).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..200 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// Polyphase resampler with a Kaiser-windowed sinc low-pass filter.
/// The output rate is the input rate times `up / down`.
///
/// The filter has 32 zero crossings on each side and a Kaiser window with
/// beta 9 (stop band about 90 dB down). The cut-off is 91.5 % of the lower of
/// the two Nyquist frequencies, so the stop band starts at that Nyquist
/// frequency.
fn resample(input: &[f64], up: usize, down: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    const ZERO_CROSSINGS: f64 = 32.0;
    const KAISER_BETA: f64 = 9.0;
    const ROLL_OFF: f64 = 0.915;

    // Cut-off in cycles per input sample.
    let cutoff = 0.5 * ROLL_OFF * (up as f64 / down as f64).min(1.0);
    // Filter half-length in input samples.
    let half = (ZERO_CROSSINGS / (2.0 * cutoff)).ceil() as isize;
    let taps = (2 * half) as usize;
    let i0_beta = bessel_i0(KAISER_BETA);

    // One row of coefficients per output phase.
    let mut table = vec![0.0f64; up * taps];
    for (phase, row) in table.chunks_exact_mut(taps).enumerate() {
        let fraction = phase as f64 / up as f64;
        for (j, coef) in row.iter_mut().enumerate() {
            // Distance from input sample `floor(position) + j - half + 1`
            // to the output instant, in input samples.
            let t = fraction - (j as isize - half + 1) as f64;
            let u = t / half as f64;
            let window = if u.abs() < 1.0 {
                bessel_i0(KAISER_BETA * (1.0 - u * u).sqrt()) / i0_beta
            } else {
                0.0
            };
            let x = PI * 2.0 * cutoff * t;
            let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
            *coef = 2.0 * cutoff * sinc * window;
        }
        // Unity gain at 0 Hz for every phase.
        let sum: f64 = row.iter().sum();
        for coef in row.iter_mut() {
            *coef /= sum;
        }
    }

    let out_len = input.len() * up / down;
    let mut out = Vec::with_capacity(out_len);
    for n in 0..out_len {
        let position = n * down;
        let first = (position / up) as isize - half + 1;
        let row = &table[(position % up) * taps..][..taps];
        let mut acc = 0.0;
        for (j, &coef) in row.iter().enumerate() {
            let index = first + j as isize;
            if index >= 0 && (index as usize) < input.len() {
                acc += coef * input[index as usize];
            }
        }
        out.push(acc);
    }
    out
}

/// Read a mono 16-bit PCM WAV file. Samples are scaled to [-1, 1).
fn read_wav_mono_pcm16(path: &str, expected_rate: u32) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{path}: not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let mut format_ok = false;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            let format = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            if format != 1 || channels != 1 || rate != expected_rate || bits != 16 {
                return Err(format!(
                    "{path}: expected mono 16-bit PCM at {expected_rate} Hz, found format {format}, \
                     {channels} channel(s), {rate} Hz, {bits} bits"
                ));
            }
            format_ok = true;
        } else if id == b"data" {
            if !format_ok {
                return Err(format!("{path}: data chunk before fmt chunk"));
            }
            let (pairs, _) = body.as_chunks::<2>();
            return Ok(pairs
                .iter()
                .map(|&pair| i16::from_le_bytes(pair) as f64 / 32768.0)
                .collect());
        }
        pos += 8 + size + (size & 1);
    }
    Err(format!("{path}: no data chunk"))
}

/// Speech at `rate`: the 16 kHz fixture, resampled.
///
/// The stereo version places the talker left of centre: the right channel is
/// 0.3 ms later and 4.4 dB quieter. Each channel also gets its own low-level
/// noise, so left and right differ in level, in time and in noise.
fn make_speech(speech_16k: &[f64], rate: u32, channels: usize) -> Vec<Vec<f64>> {
    let mono = match rate {
        8000 => resample(speech_16k, 1, 2),
        12000 => resample(speech_16k, 3, 4),
        24000 => resample(speech_16k, 3, 2),
        48000 => resample(speech_16k, 3, 1),
        _ => speech_16k.to_vec(),
    };
    let mut out = if channels == 2 {
        let n = mono.len();
        let delay = (rate as f64 * 0.0003).round() as usize;
        let right: Vec<f64> = (0..n).map(|i| 0.6 * mono[(i + n - delay) % n]).collect();
        let mut both = vec![mono, right];
        let noise_rms = 0.003 * peak_of(&both);
        add_pink_noise(&mut both[0], 0x5EED_0001_0000_0001, noise_rms);
        add_pink_noise(&mut both[1], 0x5EED_0002_0000_0003, noise_rms);
        both
    } else {
        vec![mono]
    };
    normalize(&mut out);
    out
}

/// Constant-power pan: 0 is hard left, 1 is hard right.
fn pan_gains(pan: f64) -> (f64, f64) {
    let angle = pan * std::f64::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

/// Synthetic music, 8 seconds long, made to repeat without a gap.
///
/// * Five voices play harmonic notes (up to 64 harmonics each, limited to
///   20 kHz or 45 % of the sample rate). The chord changes every 250 ms.
/// * Percussion: a bass drum, a snare made of decaying noise, and decaying
///   high-pass noise every 125 ms.
/// * A pink noise floor, different in each channel.
///
/// Every voice has its own position between left and right, and all noise is
/// generated separately per channel. The mono version is the mean of left
/// and right.
fn make_music(rate: u32, channels: usize) -> Vec<Vec<f64>> {
    use std::f64::consts::TAU;
    const STEPS: usize = 32;
    /// MIDI note numbers: four chord notes per step.
    const CHORDS: [[u8; 4]; 8] = [
        [45, 52, 60, 64],
        [41, 53, 57, 60],
        [48, 55, 60, 64],
        [43, 50, 59, 62],
        [50, 57, 62, 65],
        [45, 52, 57, 64],
        [40, 52, 56, 59],
        [45, 57, 60, 69],
    ];
    /// Per voice: position (0 left, 1 right), harmonic roll-off exponent,
    /// decay time in seconds, gain. The fifth voice is the melody.
    const VOICES: [(f64, f64, f64, f64); 5] = [
        (0.50, 1.0, 0.50, 1.00),
        (0.08, 1.2, 0.35, 0.70),
        (0.92, 1.2, 0.35, 0.70),
        (0.25, 1.0, 0.30, 0.60),
        (0.80, 0.8, 0.12, 0.60),
    ];
    const ATTACK: f64 = 0.004;
    const RELEASE: f64 = 0.010;

    let fs = rate as f64;
    let step_len = (rate / 4) as usize;
    let step_seconds = step_len as f64 / fs;
    let n = step_len * STEPS;
    let max_frequency = (0.45 * fs).min(20_000.0);
    let mut left = vec![0.0f64; n];
    let mut right = vec![0.0f64; n];

    // Fade in over ATTACK, fade out over RELEASE before the next step.
    let edges = |t: f64| (t / ATTACK).min(1.0) * ((step_seconds - t) / RELEASE).clamp(0.0, 1.0);

    let mut melody = XorShift(0x9E37_79B9_7F4A_7C15);
    for step in 0..STEPS {
        let chord = CHORDS[step % CHORDS.len()];
        let melody_note = chord[(melody.next_u64() % 4) as usize] + 24 + 12 * (melody.next_u64() % 2) as u8;
        let melody_rests = melody.next_u64().is_multiple_of(4);
        let notes = [chord[0], chord[1], chord[2], chord[3], melody_note];
        let base = step * step_len;
        for (voice, &(pan, roll_off, decay, gain)) in VOICES.iter().enumerate() {
            if voice == 4 && melody_rests {
                continue;
            }
            let f0 = 440.0 * 2f64.powf((notes[voice] as f64 - 69.0) / 12.0);
            let harmonics = ((max_frequency / f0) as usize).min(64);
            if harmonics == 0 {
                continue;
            }
            let weights: Vec<f64> = (1..=harmonics).map(|h| (h as f64).powf(-roll_off)).collect();
            let (gain_left, gain_right) = pan_gains(pan);
            let omega = TAU * f0 / fs;
            for i in 0..step_len {
                let t = i as f64 / fs;
                // sin(h x) for h = 1, 2, ... by the recurrence
                // sin((h + 1) x) = 2 cos(x) sin(h x) - sin((h - 1) x).
                let x = omega * i as f64;
                let two_cos = 2.0 * x.cos();
                let mut previous = 0.0;
                let mut current = x.sin();
                let mut sum = weights[0] * current;
                for &weight in &weights[1..] {
                    let next = two_cos * current - previous;
                    sum += weight * next;
                    previous = current;
                    current = next;
                }
                let y = gain * edges(t) * (-t / decay).exp() * sum;
                left[base + i] += gain_left * y;
                right[base + i] += gain_right * y;
            }
        }
    }

    // Percussion. Each channel has its own noise generator.
    let mut noise_left = XorShift(0x0123_4567_89AB_CDEF);
    let mut noise_right = XorShift(0xFEDC_BA98_7654_3211);
    let eighth = step_len / 2;
    for hit in 0..STEPS * 2 {
        let base = hit * eighth;
        let step = hit / 2;
        let on_beat = hit.is_multiple_of(2);

        // High-pass noise burst every 125 ms, alternating sides.
        let (decay, pan) = if on_beat { (0.012, 0.15) } else { (0.035, 0.85) };
        let (gain_left, gain_right) = pan_gains(pan);
        let (mut last_left, mut last_right) = (0.0, 0.0);
        for i in 0..eighth {
            let t = i as f64 / fs;
            let envelope = 0.30 * (-t / decay).exp();
            let (white_left, white_right) = (noise_left.uniform(), noise_right.uniform());
            left[base + i] += gain_left * envelope * (white_left - last_left);
            right[base + i] += gain_right * envelope * (white_right - last_right);
            last_left = white_left;
            last_right = white_right;
        }

        if on_beat && step % 4 == 0 {
            // Bass drum: a sine that falls from 150 Hz to 50 Hz.
            for i in 0..step_len {
                let t = i as f64 / fs;
                let phase = TAU * (50.0 * t + 100.0 * 0.03 * (1.0 - (-t / 0.03).exp()));
                let y = 0.9 * edges(t) * (-t / 0.12).exp() * phase.sin();
                left[base + i] += std::f64::consts::FRAC_1_SQRT_2 * y;
                right[base + i] += std::f64::consts::FRAC_1_SQRT_2 * y;
            }
        }
        if on_beat && step % 4 == 2 {
            // Snare: decaying white noise plus a short 180 Hz tone.
            for i in 0..step_len {
                let t = i as f64 / fs;
                let fade = edges(t);
                let noise = 0.5 * fade * (-t / 0.05).exp();
                let tone = 0.4 * fade * (-t / 0.03).exp() * (TAU * 180.0 * t).sin();
                left[base + i] += 0.8 * noise * noise_left.uniform() + 0.75 * tone;
                right[base + i] += 0.6 * noise * noise_right.uniform() + 0.5 * tone;
            }
        }
    }

    let mut out = vec![left, right];
    let noise_rms = 0.006 * peak_of(&out);
    add_pink_noise(&mut out[0], 0xA5A5_5A5A_1234_5678, noise_rms);
    add_pink_noise(&mut out[1], 0x0F0F_F0F0_8765_4321, noise_rms);

    if channels == 1 {
        let mono: Vec<f64> = out[0].iter().zip(&out[1]).map(|(l, r)| 0.5 * (l + r)).collect();
        out = vec![mono];
    }
    normalize(&mut out);
    out
}

/// One test clip, cut into whole encoder frames.
struct Clip {
    /// Interleaved samples, `frames * step` of them. Every value is a
    /// multiple of 1/32768, so `pcm_i16` holds exactly the same audio.
    pcm: Vec<f32>,
    pcm_i16: Vec<i16>,
    /// Samples per channel in one frame.
    frame: usize,
    /// Interleaved samples in one frame (`frame * channels`).
    step: usize,
    frames: usize,
    channels: usize,
}

/// Signal generator with a cache: each (signal, rate, channel count)
/// combination is built once.
struct Signals {
    speech_16k: Vec<f64>,
    cache: BTreeMap<(u8, u32, usize), Vec<Vec<f64>>>,
}

impl Signals {
    fn load() -> Result<Self, String> {
        Ok(Self {
            speech_16k: read_wav_mono_pcm16(FIXTURE, 16000)?,
            cache: BTreeMap::new(),
        })
    }

    /// Build the clip for a scenario. The source signal repeats when it is
    /// shorter than `seconds`.
    ///
    /// `perturb_lsb` is for the self-test of the golden check: a non-zero
    /// value is added (in units of 1/32768) to one sample in the middle of
    /// the clip.
    fn clip(&mut self, sc: &Scenario, seconds: f64, perturb_lsb: i32) -> Clip {
        let speech_16k = &self.speech_16k;
        let (signal, rate, channels) = (sc.signal, sc.rate, sc.channels);
        let source = self
            .cache
            .entry((signal as u8, rate, channels))
            .or_insert_with(|| match signal {
                Signal::Speech => make_speech(speech_16k, rate, channels),
                Signal::Music => make_music(rate, channels),
            });

        let frame = sc.frame_samples();
        let frames = ((seconds * 1e6 / sc.frame_us as f64).floor() as usize).max(1);
        let step = frame * channels;
        let source_len = source[0].len();
        // With pauses: each second is 0.6 s of signal, then 0.4 s of exact
        // zeros, with 10 ms linear fades.
        let gate = |i: usize| -> f64 {
            if !sc.pauses {
                return 1.0;
            }
            let t = (i % rate as usize) as f64 / rate as f64;
            (t / 0.010).min(1.0) * ((0.6 - t) / 0.010).clamp(0.0, 1.0)
        };
        let mut pcm_i16 = Vec::with_capacity(frames * step);
        for i in 0..frames * frame {
            for channel in source.iter() {
                let value = (gate(i) * channel[i % source_len] * 32768.0).round();
                pcm_i16.push(value.clamp(-32768.0, 32767.0) as i16);
            }
        }
        if perturb_lsb != 0 {
            let index = (frames * frame / 2) * channels;
            let changed = (pcm_i16[index] as i32 + perturb_lsb).clamp(-32768, 32767) as i16;
            pcm_i16[index] = changed;
        }
        let pcm = pcm_i16.iter().map(|&v| v as f32 / 32768.0).collect();
        Clip {
            pcm,
            pcm_i16,
            frame,
            step,
            frames,
            channels,
        }
    }
}

// ---------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------

/// A packet stream stored in one buffer.
#[derive(Default)]
struct Packets {
    data: Vec<u8>,
    /// Offset and length of each packet in `data`.
    spans: Vec<(usize, usize)>,
}

impl Packets {
    fn push(&mut self, packet: &[u8]) {
        self.spans.push((self.data.len(), packet.len()));
        self.data.extend_from_slice(packet);
    }

    fn get(&self, index: usize) -> &[u8] {
        let (offset, len) = self.spans[index];
        &self.data[offset..offset + len]
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn iter(&self) -> impl Iterator<Item = &[u8]> {
        self.spans.iter().map(|&(offset, len)| &self.data[offset..offset + len])
    }

    fn kbps(&self, sc: &Scenario) -> f64 {
        let seconds = self.len() as f64 * sc.frame_us as f64 / 1e6;
        self.data.len() as f64 * 8.0 / seconds.max(1e-9) / 1000.0
    }

    /// 64-bit FNV-1a hash of every packet length followed by its bytes.
    fn hash(&self) -> u64 {
        let mut h = Fnv::new();
        for packet in self.iter() {
            h.u32(packet.len() as u32);
            h.bytes(packet);
        }
        h.0
    }
}

/// How often each packet configuration occurs. The key is the upper six bits
/// of the first packet byte; 0xFF stands for an empty packet.
type Histogram = BTreeMap<u8, usize>;

fn toc_histogram(packets: &Packets) -> Histogram {
    let mut histogram = Histogram::new();
    for packet in packets.iter() {
        let key = packet.first().map_or(0xFF, |toc| toc >> 2);
        *histogram.entry(key).or_insert(0) += 1;
    }
    histogram
}

/// "mode/bandwidth/frame duration/channels" for a histogram key.
fn toc_label(key: u8) -> String {
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

fn histogram_label(histogram: &Histogram) -> String {
    let parts: Vec<String> = histogram
        .iter()
        .map(|(&key, &count)| format!("{} x{count}", toc_label(key)))
        .collect();
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}

/// 64-bit FNV-1a hash.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, data: &[u8]) {
        for &byte in data {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn f32_bits(&mut self, samples: &[f32]) {
        for sample in samples {
            self.u32(sample.to_bits());
        }
    }

    fn i16s(&mut self, samples: &[i16]) {
        for sample in samples {
            self.bytes(&sample.to_le_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// The two codecs
// ---------------------------------------------------------------------------

/// `forced` pins the coding mode, the bandwidth and the channel count to
/// what the scenario names, as `CEncoder::new` does for libopus. Without it
/// opus-rs decides by itself.
fn rust_encoder(sc: &Scenario, forced: bool) -> Result<OpusEncoder, String> {
    let application = match sc.app {
        App::Voip => Application::Voip,
        App::Audio => Application::Audio,
        App::LowDelay => Application::RestrictedLowDelay,
    };
    let mut encoder = OpusEncoder::new(sc.rate as i32, sc.channels, application)
        .map_err(|e| format!("OpusEncoder::new: {e}"))?;
    encoder.bitrate_bps = sc.bitrate;
    encoder.complexity = sc.complexity;
    encoder.use_cbr = sc.cbr;
    encoder.use_inband_fec = sc.fec;
    encoder.packet_loss_perc = sc.loss_perc;
    if forced {
        encoder.set_force_mode(Some(match sc.mode {
            Mode::Silk => ForcedMode::SilkOnly,
            Mode::Hybrid => ForcedMode::Hybrid,
            Mode::Celt => ForcedMode::CeltOnly,
        }));
        encoder.set_bandwidth(sc.bandwidth().rust_value());
        if sc.channels == 2 {
            encoder
                .set_force_channels(Some(2))
                .map_err(|e| format!("set_force_channels: {e}"))?;
        }
    }
    Ok(encoder)
}

fn rust_decoder(sc: &Scenario) -> Result<OpusDecoder, String> {
    OpusDecoder::new(sc.dec_rate as i32, sc.channels).map_err(|e| format!("OpusDecoder::new: {e}"))
}

/// Encoder settings read back from libopus with its `OPUS_GET_*` requests.
/// There is no such request for the forced coding mode; the packets show it.
#[derive(Clone, Copy, Debug)]
struct CSettings {
    application: i32,
    sample_rate: i32,
    bitrate: i32,
    vbr: i32,
    vbr_constraint: i32,
    complexity: i32,
    inband_fec: i32,
    packet_loss_perc: i32,
    force_channels: i32,
    /// Audio bandwidth of the last coded frame.
    bandwidth: i32,
    max_bandwidth: i32,
    signal: i32,
    dtx: i32,
    lsb_depth: i32,
    frame_duration: i32,
    prediction_disabled: i32,
    phase_inversion_disabled: i32,
    /// Encoder look-ahead in samples. Informative; nothing is expected.
    lookahead: i32,
}

impl CSettings {
    /// Every setting that is not what the scenario asks for, as
    /// "name: expected X, libopus has Y".
    ///
    /// Settings the scenario does not name are expected at the libopus
    /// default, because opus-rs behaves like that default: constrained
    /// variable bit rate for CELT, 24-bit input depth, automatic signal
    /// type, no discontinuous transmission, the frame size given per call.
    fn mismatches(&self, sc: &Scenario) -> Vec<String> {
        let application = match sc.app {
            App::Voip => ffi::OPUS_APPLICATION_VOIP,
            App::Audio => ffi::OPUS_APPLICATION_AUDIO,
            App::LowDelay => ffi::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        };
        let checks = [
            ("application", application, self.application),
            ("sample rate", sc.rate as i32, self.sample_rate),
            ("bit rate", sc.bitrate, self.bitrate),
            ("variable bit rate", i32::from(!sc.cbr), self.vbr),
            ("constrained variable bit rate", 1, self.vbr_constraint),
            ("complexity", sc.complexity, self.complexity),
            ("forward error correction", i32::from(sc.fec), self.inband_fec),
            ("expected loss", sc.loss_perc, self.packet_loss_perc),
            (
                "forced channels",
                if sc.channels == 2 { 2 } else { ffi::OPUS_AUTO },
                self.force_channels,
            ),
            ("bandwidth", sc.coded_bandwidth().c_value(), self.bandwidth),
            ("maximum bandwidth", ffi::OPUS_BANDWIDTH_FULLBAND, self.max_bandwidth),
            ("signal type", ffi::OPUS_AUTO, self.signal),
            ("discontinuous transmission", 0, self.dtx),
            ("input depth in bits", 24, self.lsb_depth),
            ("frame duration", ffi::OPUS_FRAMESIZE_ARG, self.frame_duration),
            ("prediction disabled", 0, self.prediction_disabled),
            ("phase inversion disabled", 0, self.phase_inversion_disabled),
        ];
        checks
            .iter()
            .filter(|(_, expected, actual)| expected != actual)
            .map(|(name, expected, actual)| format!("{name}: expected {expected}, libopus has {actual}"))
            .collect()
    }
}

/// The settings opus-rs exposes as public fields, read back after encoding:
/// bit rate, complexity, constant bit rate, forward error correction,
/// expected loss in percent. The encoder clamps them on every call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RustSettings {
    bitrate: i32,
    complexity: i32,
    cbr: bool,
    fec: bool,
    loss_perc: i32,
}

impl RustSettings {
    fn of(encoder: &OpusEncoder) -> Self {
        Self {
            bitrate: encoder.bitrate_bps,
            complexity: encoder.complexity,
            cbr: encoder.use_cbr,
            fec: encoder.use_inband_fec,
            loss_perc: encoder.packet_loss_perc,
        }
    }

    fn wanted(sc: &Scenario) -> Self {
        Self {
            bitrate: sc.bitrate,
            complexity: sc.complexity,
            cbr: sc.cbr,
            fec: sc.fec,
            loss_perc: sc.loss_perc,
        }
    }
}

/// C libopus encoder.
struct CEncoder {
    state: *mut ffi::OpusEncoder,
    channels: usize,
}

impl CEncoder {
    /// `forced` pins the coding mode, the bandwidth and the channel count to
    /// what the scenario names. Without it libopus decides by itself.
    fn new(sc: &Scenario, forced: bool) -> Result<Self, String> {
        let application = match sc.app {
            App::Voip => ffi::OPUS_APPLICATION_VOIP,
            App::Audio => ffi::OPUS_APPLICATION_AUDIO,
            App::LowDelay => ffi::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        };
        let mut error: c_int = 0;
        // SAFETY: plain C constructor; `error` is a valid out pointer.
        let state = unsafe {
            ffi::opus_encoder_create(sc.rate as i32, sc.channels as c_int, application, &mut error)
        };
        if state.is_null() || error != ffi::OPUS_OK {
            return Err(format!("opus_encoder_create failed with {error}"));
        }
        let encoder = Self {
            state,
            channels: sc.channels,
        };
        encoder.ctl(ffi::OPUS_SET_BITRATE_REQUEST, sc.bitrate)?;
        encoder.ctl(ffi::OPUS_SET_VBR_REQUEST, c_int::from(!sc.cbr))?;
        encoder.ctl(ffi::OPUS_SET_COMPLEXITY_REQUEST, sc.complexity)?;
        encoder.ctl(ffi::OPUS_SET_INBAND_FEC_REQUEST, c_int::from(sc.fec))?;
        encoder.ctl(ffi::OPUS_SET_PACKET_LOSS_PERC_REQUEST, sc.loss_perc)?;
        if forced {
            let mode = match sc.mode {
                Mode::Silk => MODE_SILK_ONLY,
                Mode::Hybrid => MODE_HYBRID,
                Mode::Celt => MODE_CELT_ONLY,
            };
            encoder.ctl(OPUS_SET_FORCE_MODE_REQUEST, mode)?;
            encoder.ctl(ffi::OPUS_SET_BANDWIDTH_REQUEST, sc.bandwidth().c_value())?;
            if sc.channels == 2 {
                encoder.ctl(ffi::OPUS_SET_FORCE_CHANNELS_REQUEST, 2)?;
            }
        }
        Ok(encoder)
    }

    fn ctl(&self, request: c_int, value: c_int) -> Result<(), String> {
        // SAFETY: `state` is a live encoder; every request used here takes
        // one `opus_int32` argument.
        let ret = unsafe { ffi::opus_encoder_ctl(self.state, request, value) };
        if ret == ffi::OPUS_OK {
            Ok(())
        } else {
            Err(format!("opus_encoder_ctl({request}, {value}) failed with {ret}"))
        }
    }

    /// Read one setting back from libopus.
    fn get(&self, request: c_int) -> Result<i32, String> {
        let mut value: i32 = i32::MIN;
        // SAFETY: `state` is a live encoder; every request used here writes
        // one `opus_int32` through the pointer.
        let ret = unsafe { ffi::opus_encoder_ctl(self.state, request, &mut value as *mut i32) };
        if ret == ffi::OPUS_OK {
            Ok(value)
        } else {
            Err(format!("opus_encoder_ctl({request}) failed with {ret}"))
        }
    }

    /// Read every setting back that libopus has an `OPUS_GET_*` request for.
    /// `bandwidth` is the bandwidth of the last coded frame, so this is
    /// meant to be called after encoding.
    fn settings(&self) -> Result<CSettings, String> {
        Ok(CSettings {
            application: self.get(ffi::OPUS_GET_APPLICATION_REQUEST)?,
            sample_rate: self.get(ffi::OPUS_GET_SAMPLE_RATE_REQUEST)?,
            bitrate: self.get(ffi::OPUS_GET_BITRATE_REQUEST)?,
            vbr: self.get(ffi::OPUS_GET_VBR_REQUEST)?,
            vbr_constraint: self.get(ffi::OPUS_GET_VBR_CONSTRAINT_REQUEST)?,
            complexity: self.get(ffi::OPUS_GET_COMPLEXITY_REQUEST)?,
            inband_fec: self.get(ffi::OPUS_GET_INBAND_FEC_REQUEST)?,
            packet_loss_perc: self.get(ffi::OPUS_GET_PACKET_LOSS_PERC_REQUEST)?,
            force_channels: self.get(ffi::OPUS_GET_FORCE_CHANNELS_REQUEST)?,
            bandwidth: self.get(ffi::OPUS_GET_BANDWIDTH_REQUEST)?,
            max_bandwidth: self.get(ffi::OPUS_GET_MAX_BANDWIDTH_REQUEST)?,
            signal: self.get(ffi::OPUS_GET_SIGNAL_REQUEST)?,
            dtx: self.get(ffi::OPUS_GET_DTX_REQUEST)?,
            lsb_depth: self.get(ffi::OPUS_GET_LSB_DEPTH_REQUEST)?,
            frame_duration: self.get(ffi::OPUS_GET_EXPERT_FRAME_DURATION_REQUEST)?,
            prediction_disabled: self.get(ffi::OPUS_GET_PREDICTION_DISABLED_REQUEST)?,
            phase_inversion_disabled: self.get(ffi::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST)?,
            lookahead: self.get(ffi::OPUS_GET_LOOKAHEAD_REQUEST)?,
        })
    }

    /// Returns the packet length, or a negative libopus error code.
    #[inline]
    fn encode(&mut self, pcm: &[f32], frame: usize, out: &mut [u8]) -> i32 {
        assert!(pcm.len() >= frame * self.channels);
        // SAFETY: `pcm` holds `frame * channels` samples (checked above) and
        // `out` is writable for `out.len()` bytes.
        unsafe {
            ffi::opus_encode_float(
                self.state,
                pcm.as_ptr(),
                frame as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        }
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        // SAFETY: `state` came from `opus_encoder_create` and is freed once.
        unsafe { ffi::opus_encoder_destroy(self.state) }
    }
}

/// C libopus decoder.
struct CDecoder {
    state: *mut ffi::OpusDecoder,
    channels: usize,
}

impl CDecoder {
    fn new(sc: &Scenario) -> Result<Self, String> {
        let mut error: c_int = 0;
        // SAFETY: plain C constructor; `error` is a valid out pointer.
        let state =
            unsafe { ffi::opus_decoder_create(sc.dec_rate as i32, sc.channels as c_int, &mut error) };
        if state.is_null() || error != ffi::OPUS_OK {
            return Err(format!("opus_decoder_create failed with {error}"));
        }
        Ok(Self {
            state,
            channels: sc.channels,
        })
    }

    /// Returns the samples per channel, or a negative libopus error code.
    #[inline]
    fn decode(&mut self, packet: &[u8], frame: usize, out: &mut [f32]) -> i32 {
        assert!(out.len() >= frame * self.channels);
        // SAFETY: `out` has room for `frame * channels` samples (checked
        // above); `packet` is readable for its length.
        unsafe {
            ffi::opus_decode_float(
                self.state,
                packet.as_ptr(),
                packet.len() as i32,
                out.as_mut_ptr(),
                frame as c_int,
                0,
            )
        }
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        // SAFETY: `state` came from `opus_decoder_create` and is freed once.
        unsafe { ffi::opus_decoder_destroy(self.state) }
    }
}

fn libopus_version() -> String {
    // SAFETY: returns a pointer to a static NUL-terminated string.
    unsafe { CStr::from_ptr(ffi::opus_get_version_string()) }
        .to_string_lossy()
        .into_owned()
}

fn rust_encode_all(sc: &Scenario, clip: &Clip) -> Result<Packets, String> {
    rust_encode_all_with_settings(sc, clip).map(|(packets, _)| packets)
}

/// Encode the clip with opus-rs. Also returns the public encoder settings as
/// they are after the last frame.
fn rust_encode_all_with_settings(
    sc: &Scenario,
    clip: &Clip,
) -> Result<(Packets, RustSettings), String> {
    let mut encoder = rust_encoder(sc, true)?;
    let mut out = [0u8; MAX_PACKET];
    let mut packets = Packets::default();
    for (index, frame) in clip.pcm.chunks_exact(clip.step).enumerate() {
        let len = encoder
            .encode(frame, clip.frame, &mut out)
            .map_err(|e| format!("Rust encode failed at frame {index}: {e}"))?;
        packets.push(&out[..len]);
    }
    Ok((packets, RustSettings::of(&encoder)))
}

/// Same as `rust_encode_all`, through the 16-bit integer entry point.
fn rust_encode_all_i16(sc: &Scenario, clip: &Clip) -> Result<Packets, String> {
    let mut encoder = rust_encoder(sc, true)?;
    let mut out = [0u8; MAX_PACKET];
    let mut packets = Packets::default();
    for (index, frame) in clip.pcm_i16.chunks_exact(clip.step).enumerate() {
        let len = encoder
            .encode_i16(frame, clip.frame, &mut out)
            .map_err(|e| format!("Rust encode_i16 failed at frame {index}: {e}"))?;
        packets.push(&out[..len]);
    }
    Ok(packets)
}

fn c_encode_all(sc: &Scenario, clip: &Clip, forced: bool) -> Result<Packets, String> {
    c_encode_all_with_settings(sc, clip, forced).map(|(packets, _)| packets)
}

/// Encode the clip with libopus. Also returns the settings libopus reports
/// after the last frame.
fn c_encode_all_with_settings(
    sc: &Scenario,
    clip: &Clip,
    forced: bool,
) -> Result<(Packets, CSettings), String> {
    let mut encoder = CEncoder::new(sc, forced)?;
    let mut out = [0u8; MAX_PACKET];
    let mut packets = Packets::default();
    for (index, frame) in clip.pcm.chunks_exact(clip.step).enumerate() {
        let len = encoder.encode(frame, clip.frame, &mut out);
        if len < 0 {
            return Err(format!("C encode failed at frame {index} with {len}"));
        }
        packets.push(&out[..len as usize]);
    }
    let settings = encoder.settings()?;
    Ok((packets, settings))
}

/// What a decoder produced for a whole packet stream.
struct Decoded {
    /// Interleaved output, one full frame per packet.
    pcm: Vec<f32>,
    /// Hash of the sample count and the sample bit patterns of every packet.
    hash: u64,
    /// Packets the decoder rejected.
    errors: usize,
    first_error: Option<String>,
    /// Packets for which the decoder returned another number of samples
    /// than one scenario frame.
    wrong_size: usize,
}

impl Decoded {
    fn with_capacity(samples: usize) -> Self {
        Self {
            pcm: Vec::with_capacity(samples),
            hash: 0,
            errors: 0,
            first_error: None,
            wrong_size: 0,
        }
    }
}

/// Which packets a decode run treats as lost.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Loss {
    None,
    /// Every packet whose index is `LOSS_PERIOD - 1` modulo `LOSS_PERIOD`:
    /// single losses, each followed by nine good packets.
    Single,
    /// The single losses of `Single`, and in addition three packets in a row
    /// out of every `BURST_PERIOD`. A decoder conceals the second and third
    /// lost packet of a row differently from the first.
    Bursts,
}

impl Loss {
    fn is_lost(self, index: usize) -> bool {
        let single = index % LOSS_PERIOD == LOSS_PERIOD - 1;
        let in_burst = (BURST_START..BURST_START + BURST_LENGTH).contains(&(index % BURST_PERIOD));
        match self {
            Loss::None => false,
            Loss::Single => single,
            Loss::Bursts => single || in_burst,
        }
    }
}

/// Decode with opus-rs. A packet that `loss` marks as lost is replaced by an
/// empty packet, which makes the decoder conceal one frame.
fn rust_decode_all(sc: &Scenario, packets: &Packets, loss: Loss) -> Result<Decoded, String> {
    let mut decoder = rust_decoder(sc)?;
    let frame = sc.dec_frame_samples();
    let step = frame * sc.channels;
    let mut out = vec![0.0f32; step];
    let mut decoded = Decoded::with_capacity(step * packets.len());
    let mut hash = Fnv::new();
    for (index, packet) in packets.iter().enumerate() {
        let input: &[u8] = if loss.is_lost(index) { &[] } else { packet };
        out.fill(0.0);
        match decoder.decode(input, frame, &mut out) {
            Ok(samples) => {
                hash.u32(samples as u32);
                hash.f32_bits(&out[..(samples * sc.channels).min(step)]);
                decoded.wrong_size += usize::from(samples != frame);
            }
            Err(e) => {
                hash.bytes(b"error:");
                hash.bytes(e.as_bytes());
                decoded.errors += 1;
                decoded
                    .first_error
                    .get_or_insert_with(|| format!("packet {index}: {e}"));
            }
        }
        decoded.pcm.extend_from_slice(&out);
    }
    decoded.hash = hash.0;
    Ok(decoded)
}

fn c_decode_all(sc: &Scenario, packets: &Packets) -> Result<Decoded, String> {
    let mut decoder = CDecoder::new(sc)?;
    let frame = sc.dec_frame_samples();
    let step = frame * sc.channels;
    let mut out = vec![0.0f32; step];
    let mut decoded = Decoded::with_capacity(step * packets.len());
    for (index, packet) in packets.iter().enumerate() {
        out.fill(0.0);
        let samples = decoder.decode(packet, frame, &mut out);
        if samples < 0 {
            decoded.errors += 1;
            decoded
                .first_error
                .get_or_insert_with(|| format!("packet {index}: libopus error {samples}"));
        } else {
            decoded.wrong_size += usize::from(samples as usize != frame);
        }
        decoded.pcm.extend_from_slice(&out);
    }
    Ok(decoded)
}

/// The first frame inside a packet, found by libopus's packet parser.
fn first_frame(packet: &[u8]) -> Option<&[u8]> {
    if packet.is_empty() {
        return None;
    }
    let mut frames = [std::ptr::null::<u8>(); 48];
    let mut sizes = [0i16; 48];
    // SAFETY: `frames` and `sizes` have the 48 entries the parser may write;
    // the two optional out pointers may be null.
    let count = unsafe {
        ffi::opus_packet_parse(
            packet.as_ptr(),
            packet.len() as i32,
            std::ptr::null_mut(),
            frames.as_mut_ptr(),
            sizes.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if count < 1 || frames[0].is_null() || sizes[0] <= 0 {
        return None;
    }
    // SAFETY: the parser returns pointers into `packet`.
    let offset = unsafe { frames[0].offset_from(packet.as_ptr()) } as usize;
    packet.get(offset..offset + sizes[0] as usize)
}

/// Decode a SILK stream, mono or stereo, with one packet in `LOSS_PERIOD`
/// lost, and recover each lost packet from the redundancy data (LBRR, "low
/// bit rate redundancy") carried by the packet after it.
///
/// The SILK decoder of the crate is driven directly, which also shows how
/// many frames had redundancy data. Its output is at the SILK internal
/// rate, one channel after the other. (`OpusDecoder::decode_fec` is the
/// public entry point for the same thing.) Returns the output hash and the
/// number of frames recovered from redundancy data: for stereo, frames of
/// the first channel. `None` when the scenario is not SILK with forward
/// error correction.
fn silk_fec_decode(sc: &Scenario, packets: &Packets) -> Option<(u64, usize)> {
    if !(sc.fec && sc.mode == Mode::Silk) {
        return None;
    }
    let internal_rate = match sc.bandwidth() {
        Bandwidth::Narrow => 8000,
        Bandwidth::Medium => 12000,
        _ => 16000,
    };
    let frame_ms = (sc.frame_us / 1000) as i32;
    let silk_frames = if frame_ms > 20 { (frame_ms / 20) as usize } else { 1 };
    let mut decoder = SilkDecoder::new();
    decoder.init(internal_rate, sc.channels as i32);
    decoder.channel_state[0].fs_api_hz = internal_rate;

    // One SILK frame: 20 ms at 16 kHz, for two channels.
    let mut pcm = [0i16; 2 * 320];
    let mut hash = Fnv::new();
    let mut recovered = 0usize;
    for index in 0..packets.len() {
        let lost = index % LOSS_PERIOD == LOSS_PERIOD - 1;
        let (payload, flag) = if !lost {
            (first_frame(packets.get(index)), FLAG_DECODE_NORMAL)
        } else if index + 1 < packets.len() {
            (first_frame(packets.get(index + 1)), FLAG_DECODE_LBRR)
        } else {
            (None, FLAG_PACKET_LOST)
        };
        let (payload, flag) = match payload {
            Some(payload) => (payload, flag),
            None => (&[][..], FLAG_PACKET_LOST),
        };
        let mut range_decoder = RangeCoder::new_decoder(payload);
        for silk_frame in 0..silk_frames {
            let samples = decoder.decode(
                &mut range_decoder,
                &mut pcm,
                flag,
                silk_frame == 0,
                frame_ms,
                internal_rate,
            );
            hash.u32(samples as u32);
            if samples > 0 {
                hash.i16s(&pcm[..(samples as usize * sc.channels).min(pcm.len())]);
            }
            if flag == FLAG_DECODE_LBRR && decoder.channel_state[0].lbrr_flags[silk_frame] == 1 {
                recovered += 1;
            }
        }
    }
    Some((hash.0, recovered))
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Ask macOS to run this thread on a performance core.
#[cfg(target_os = "macos")]
fn raise_thread_priority() -> bool {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    // SAFETY: changes the scheduling class of the calling thread only.
    unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 }
}

#[cfg(not(target_os = "macos"))]
fn raise_thread_priority() -> bool {
    false
}

// The four timed functions below create a fresh codec state, then time one
// pass over the clip. Between the two clock reads there is only the loop
// over the frames, the codec call and an addition per frame. The input goes
// through `black_box` so that the compiler cannot specialise a call for it,
// and the output goes through `black_box` so that it counts as used.
//
// A call that fails would be faster than a real one, so the functions count
// failures and stop the program after the clock was read.

#[inline(never)]
fn time_rust_encode(sc: &Scenario, clip: &Clip) -> Duration {
    let mut encoder = rust_encoder(sc, true).expect("Rust encoder");
    let mut out = [0u8; MAX_PACKET];
    let (mut bytes, mut failures) = (0usize, 0usize);
    let start = Instant::now();
    for frame in clip.pcm.chunks_exact(clip.step) {
        match encoder.encode(black_box(frame), clip.frame, &mut out) {
            Ok(len) => bytes += len,
            Err(_) => failures += 1,
        }
    }
    let elapsed = start.elapsed();
    black_box((bytes, &out));
    assert!(failures == 0, "Rust encode failed {failures} times while it was timed");
    elapsed
}

#[inline(never)]
fn time_c_encode(sc: &Scenario, clip: &Clip) -> Duration {
    let mut encoder = CEncoder::new(sc, true).expect("C encoder");
    let mut out = [0u8; MAX_PACKET];
    let (mut bytes, mut failures) = (0i64, 0usize);
    let start = Instant::now();
    for frame in clip.pcm.chunks_exact(clip.step) {
        let len = encoder.encode(black_box(frame), clip.frame, &mut out);
        bytes += len as i64;
        failures += usize::from(len < 0);
    }
    let elapsed = start.elapsed();
    black_box((bytes, &out));
    assert!(failures == 0, "C encode failed {failures} times while it was timed");
    elapsed
}

#[inline(never)]
fn time_rust_decode(sc: &Scenario, packets: &Packets, out: &mut [f32]) -> Duration {
    let mut decoder = rust_decoder(sc).expect("Rust decoder");
    let frame = sc.dec_frame_samples();
    let (mut samples, mut failures) = (0usize, 0usize);
    let start = Instant::now();
    for &(offset, len) in &packets.spans {
        match decoder.decode(black_box(&packets.data[offset..offset + len]), frame, out) {
            Ok(n) => samples += n,
            Err(_) => failures += 1,
        }
    }
    let elapsed = start.elapsed();
    black_box((samples, &*out));
    assert!(failures == 0, "Rust decode failed {failures} times while it was timed");
    assert!(
        samples == frame * packets.len(),
        "Rust decode returned {samples} samples for {} packets of {frame}",
        packets.len()
    );
    elapsed
}

#[inline(never)]
fn time_c_decode(sc: &Scenario, packets: &Packets, out: &mut [f32]) -> Duration {
    let mut decoder = CDecoder::new(sc).expect("C decoder");
    let frame = sc.dec_frame_samples();
    let (mut samples, mut failures) = (0i64, 0usize);
    let start = Instant::now();
    for &(offset, len) in &packets.spans {
        let n = decoder.decode(black_box(&packets.data[offset..offset + len]), frame, out);
        samples += n as i64;
        failures += usize::from(n < 0);
    }
    let elapsed = start.elapsed();
    black_box((samples, &*out));
    assert!(failures == 0, "C decode failed {failures} times while it was timed");
    assert!(
        samples == (frame * packets.len()) as i64,
        "C decode returned {samples} samples for {} packets of {frame}",
        packets.len()
    );
    elapsed
}

/// Minimum and median time per frame, in nanoseconds.
#[derive(Clone, Copy)]
struct Stat {
    min_ns: f64,
    median_ns: f64,
}

fn stat(times: &[Duration], frames: usize) -> Stat {
    let mut ns: Vec<f64> = times
        .iter()
        .map(|t| t.as_nanos() as f64 / frames as f64)
        .collect();
    ns.sort_by(f64::total_cmp);
    let mid = ns.len() / 2;
    let median_ns = if ns.len().is_multiple_of(2) {
        0.5 * (ns[mid - 1] + ns[mid])
    } else {
        ns[mid]
    };
    Stat {
        min_ns: ns[0],
        median_ns,
    }
}

struct BenchRow {
    name: String,
    frames: usize,
    frame_us: u32,
    /// How many rounds were measured (see `bench_one`).
    rounds: usize,
    rust_enc: Stat,
    c_enc: Stat,
    rust_dec: Stat,
    c_dec: Stat,
    rust_kbps: f64,
    c_kbps: f64,
    rust_toc: String,
    c_toc: String,
    /// The problems found by `compare`. Empty when the Rust and the C side
    /// produced equivalent streams.
    notes: Vec<String>,
    /// Encoder steps that only libopus runs for this scenario
    /// (`Scenario::c_only_steps`).
    c_only: Vec<&'static str>,
    /// One entry per operation whose median was more than 3 % above its
    /// minimum in every round. For such an operation the minimum is a rare
    /// fast repetition, and the median is the number to use.
    unsteady: Vec<String>,
}

impl BenchRow {
    fn frame_ns(&self) -> f64 {
        self.frame_us as f64 * 1000.0
    }
}

/// The four operations, in the order used for all timing arrays.
const OPERATIONS: [&str; 4] = ["rust_enc", "c_enc", "rust_dec", "c_dec"];
const OPERATION_LABELS: [&str; 4] = ["Rust enc", "C enc", "Rust dec", "C dec"];

/// Time the four operations `reps` times each, after one warm-up repetition
/// that is dropped. Rust and C alternate, and the order within each pair
/// flips on every repetition. Returns the time of every repetition (one pass
/// over the clip), per operation.
fn measure(sc: &Scenario, clip: &Clip, c_packets: &Packets, reps: usize) -> [Vec<Duration>; 4] {
    let mut out = vec![0.0f32; sc.dec_frame_samples() * sc.channels];
    let mut times: [Vec<Duration>; 4] = Default::default();
    for rep in 0..=reps {
        let (re, ce, rd, cd);
        if rep % 2 == 0 {
            re = time_rust_encode(sc, clip);
            ce = time_c_encode(sc, clip);
            rd = time_rust_decode(sc, c_packets, &mut out);
            cd = time_c_decode(sc, c_packets, &mut out);
        } else {
            ce = time_c_encode(sc, clip);
            re = time_rust_encode(sc, clip);
            cd = time_c_decode(sc, c_packets, &mut out);
            rd = time_rust_decode(sc, c_packets, &mut out);
        }
        if rep > 0 {
            times[0].push(re);
            times[1].push(ce);
            times[2].push(rd);
            times[3].push(cd);
        }
    }
    times
}

/// Hidden command behind `bench`: measure one scenario in this process and
/// print the time of every repetition in nanoseconds, one line per
/// operation.
fn cmd_bench_worker(opts: &Opts) -> Result<i32, String> {
    let sc = find_scenario(opts.get("--scenario").ok_or("bench-worker needs --scenario")?)?;
    let reps: usize = opts.number("--reps", 5)?;
    let seconds: f64 = opts.number("--seconds", 6.0)?;
    let mut signals = Signals::load()?;
    let clip = signals.clip(&sc, seconds, 0);
    let c_packets = c_encode_all(&sc, &clip, true)?;
    let performance_core = raise_thread_priority();
    let times = measure(&sc, &clip, &c_packets, reps);
    for (label, values) in OPERATIONS.iter().zip(&times) {
        let values: Vec<String> = values.iter().map(|t| t.as_nanos().to_string()).collect();
        println!("{label} {}", values.join(" "));
    }
    println!("performance_core {}", u8::from(performance_core));
    Ok(0)
}

/// Measure one scenario in `procs` fresh processes, one after the other, and
/// collect the repetition times of all of them.
///
/// A fresh process per scenario makes the result independent of which
/// scenarios ran before. Several processes per scenario cover effects that
/// are fixed for the life of one process, such as the random addresses the
/// operating system gives to code, stack and heap. Both were seen to move a
/// result by up to 10 %.
///
/// Also returns whether every process could raise its thread priority.
fn measure_in_processes(
    sc: &Scenario,
    seconds: f64,
    reps_per_proc: usize,
    procs: usize,
) -> Result<([Vec<Duration>; 4], bool), String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this program: {e}"))?;
    let mut times: [Vec<Duration>; 4] = Default::default();
    let mut performance_core = true;
    for _ in 0..procs {
        let output = std::process::Command::new(&exe)
            .args(["bench-worker", "--scenario", &sc.name()])
            .args(["--reps", &reps_per_proc.to_string()])
            .args(["--seconds", &seconds.to_string()])
            .output()
            .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
        if !output.status.success() {
            return Err(format!(
                "measuring process failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let mut tokens = line.split_whitespace();
            let label = tokens.next().unwrap_or_default();
            if label == "performance_core" {
                performance_core &= tokens.next() == Some("1");
            } else if let Some(index) = OPERATIONS.iter().position(|op| *op == label) {
                times[index].extend(
                    tokens
                        .filter_map(|t| t.parse::<u64>().ok())
                        .map(Duration::from_nanos),
                );
            }
        }
    }
    if times.iter().any(|t| t.len() != reps_per_proc * procs) {
        return Err("measuring process printed an unexpected number of results".to_string());
    }
    Ok((times, performance_core))
}

/// A round is `procs` processes with `reps_per_proc` repetitions each.
#[derive(Clone, Copy)]
struct Rounds {
    seconds: f64,
    reps_per_proc: usize,
    procs: usize,
    /// Upper limit of rounds per scenario.
    max_rounds: usize,
}

/// In a quiet round the median of every operation is at most this far above
/// its minimum.
const QUIET_MEDIAN_OVER_MIN: f64 = 1.03;

/// Measure one scenario.
///
/// Other programs on the machine can slow down a whole round. Such a round
/// shows a median well above the minimum. The scenario is then measured
/// again, until a round is quiet or `max_rounds` is reached. The result is,
/// per operation, the lowest minimum of all rounds and the lowest median of
/// any single round.
fn bench_one(sc: &Scenario, clip: &Clip, plan: Rounds) -> Result<(BenchRow, bool), String> {
    let comparison = compare(sc, clip)?;
    let mut performance_core = true;
    let mut best: Option<[Stat; 4]> = None;
    // Per operation: the smallest "median over minimum" of any one round.
    let mut steadiest = [f64::INFINITY; 4];
    let mut rounds = 0;
    loop {
        rounds += 1;
        let (times, raised) =
            measure_in_processes(sc, plan.seconds, plan.reps_per_proc, plan.procs)?;
        performance_core &= raised;
        let round: [Stat; 4] = std::array::from_fn(|i| stat(&times[i], clip.frames));
        for (lowest, s) in steadiest.iter_mut().zip(&round) {
            *lowest = lowest.min(s.median_ns / s.min_ns);
        }
        let quiet = round
            .iter()
            .all(|s| s.median_ns <= QUIET_MEDIAN_OVER_MIN * s.min_ns);
        best = Some(match best {
            None => round,
            Some(previous) => std::array::from_fn(|i| Stat {
                min_ns: previous[i].min_ns.min(round[i].min_ns),
                median_ns: previous[i].median_ns.min(round[i].median_ns),
            }),
        });
        if quiet || rounds >= plan.max_rounds {
            break;
        }
    }
    let best = best.unwrap_or(
        [Stat {
            min_ns: 0.0,
            median_ns: 0.0,
        }; 4],
    );
    let unsteady = OPERATION_LABELS
        .iter()
        .zip(steadiest)
        .filter(|(_, ratio)| *ratio > QUIET_MEDIAN_OVER_MIN)
        .map(|(label, ratio)| {
            format!(
                "{label}: in each of the {rounds} rounds the median was at least {:.1} % above the minimum, so the minimum is a rare fast repetition; use the median",
                (ratio - 1.0) * 100.0
            )
        })
        .collect();
    let [rust_enc, c_enc, rust_dec, c_dec] = best;
    let row = BenchRow {
        name: sc.name(),
        frames: clip.frames,
        frame_us: sc.frame_us,
        rounds,
        rust_enc,
        c_enc,
        rust_dec,
        c_dec,
        rust_kbps: comparison.rust_kbps,
        c_kbps: comparison.c_kbps,
        rust_toc: comparison.rust_toc,
        c_toc: comparison.c_toc,
        notes: comparison.problems,
        c_only: sc.c_only_steps(),
        unsteady,
    };
    Ok((row, performance_core))
}

const BENCH_HEADER: &str = "| scenario | frames | rounds | Rust enc min | Rust enc med | C enc min | C enc med | enc Rust/C min | enc Rust/C med | Rust enc xRT | Rust dec min | Rust dec med | C dec min | C dec med | dec Rust/C min | dec Rust/C med | Rust dec xRT | Rust kb/s | C kb/s | streams | timing | only C runs |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|";

fn bench_row_markdown(row: &BenchRow) -> String {
    format!(
        "| {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.3} | {:.3} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.3} | {:.3} | {:.0} | {:.1} | {:.1} | {} | {} | {} |",
        row.name,
        row.frames,
        row.rounds,
        row.rust_enc.min_ns,
        row.rust_enc.median_ns,
        row.c_enc.min_ns,
        row.c_enc.median_ns,
        row.rust_enc.min_ns / row.c_enc.min_ns,
        row.rust_enc.median_ns / row.c_enc.median_ns,
        row.frame_ns() / row.rust_enc.min_ns,
        row.rust_dec.min_ns,
        row.rust_dec.median_ns,
        row.c_dec.min_ns,
        row.c_dec.median_ns,
        row.rust_dec.min_ns / row.c_dec.min_ns,
        row.rust_dec.median_ns / row.c_dec.median_ns,
        row.frame_ns() / row.rust_dec.min_ns,
        row.rust_kbps,
        row.c_kbps,
        if row.notes.is_empty() { "same" } else { "CHECK" },
        if row.unsteady.is_empty() { "steady" } else { "UNSTEADY" },
        if row.c_only.is_empty() { "-".to_string() } else { row.c_only.join(", ") },
    )
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One JSON object per line, so that `--compare` can read the file back
/// without a JSON library.
fn bench_row_json(row: &BenchRow) -> String {
    let list = |items: &mut dyn Iterator<Item = &str>| items.map(json_string).collect::<Vec<_>>().join(",");
    let notes = list(&mut row.notes.iter().map(String::as_str));
    let unsteady = list(&mut row.unsteady.iter().map(String::as_str));
    let c_only = list(&mut row.c_only.iter().copied());
    format!(
        "{{\"name\":{},\"frames\":{},\"frame_us\":{},\"rounds\":{},\
         \"rust_enc_min_ns\":{:.1},\"rust_enc_med_ns\":{:.1},\"c_enc_min_ns\":{:.1},\"c_enc_med_ns\":{:.1},\
         \"enc_ratio\":{:.4},\"enc_ratio_med\":{:.4},\"rust_enc_xrt\":{:.1},\
         \"rust_dec_min_ns\":{:.1},\"rust_dec_med_ns\":{:.1},\"c_dec_min_ns\":{:.1},\"c_dec_med_ns\":{:.1},\
         \"dec_ratio\":{:.4},\"dec_ratio_med\":{:.4},\"rust_dec_xrt\":{:.1},\
         \"rust_kbps\":{:.2},\"c_kbps\":{:.2},\"rust_toc\":{},\"c_toc\":{},\"notes\":[{}],\"unsteady\":[{}],\"only_c_runs\":[{}]}}",
        json_string(&row.name),
        row.frames,
        row.frame_us,
        row.rounds,
        row.rust_enc.min_ns,
        row.rust_enc.median_ns,
        row.c_enc.min_ns,
        row.c_enc.median_ns,
        row.rust_enc.min_ns / row.c_enc.min_ns,
        row.rust_enc.median_ns / row.c_enc.median_ns,
        row.frame_ns() / row.rust_enc.min_ns,
        row.rust_dec.min_ns,
        row.rust_dec.median_ns,
        row.c_dec.min_ns,
        row.c_dec.median_ns,
        row.rust_dec.min_ns / row.c_dec.min_ns,
        row.rust_dec.median_ns / row.c_dec.median_ns,
        row.frame_ns() / row.rust_dec.min_ns,
        row.rust_kbps,
        row.c_kbps,
        json_string(&row.rust_toc),
        json_string(&row.c_toc),
        notes,
        unsteady,
        c_only,
    )
}

/// Value of a `"key":"text"` field in one line of a bench JSON file.
fn json_text_field(line: &str, key: &str) -> Option<String> {
    let rest = &line[line.find(&format!("\"{key}\":\""))? + key.len() + 4..];
    Some(rest[..rest.find('"')?].to_string())
}

/// Value of a `"key":number` field in one line of a bench JSON file.
fn json_number_field(line: &str, key: &str) -> Option<f64> {
    let rest = &line[line.find(&format!("\"{key}\":"))? + key.len() + 3..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn git_revision() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR")])
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match run(&["rev-parse", "--short", "HEAD"]) {
        Some(head) => {
            let dirty = run(&["status", "--porcelain", "--untracked-files=no"])
                .is_some_and(|s| !s.is_empty());
            if dirty { format!("{head}+changes") } else { head }
        }
        None => "unknown".to_string(),
    }
}

fn write_file(path: &str, text: &str) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
}

fn cmd_bench(opts: &Opts) -> Result<i32, String> {
    let seconds: f64 = opts.number("--seconds", 6.0)?;
    let requested_reps: usize = opts.number("--reps", 15)?;
    let procs: usize = opts.number("--procs", 3)?;
    if requested_reps == 0 || procs == 0 || seconds <= 0.0 {
        return Err("--reps, --procs and --seconds must be positive".to_string());
    }
    let max_rounds: usize = opts.number("--rounds", 3)?;
    if max_rounds == 0 {
        return Err("--rounds must be positive".to_string());
    }
    // The repetitions are split over the processes, rounded up.
    let reps_per_proc = requested_reps.div_ceil(procs);
    let reps = reps_per_proc * procs;
    let plan = Rounds {
        seconds,
        reps_per_proc,
        procs,
        max_rounds,
    };
    let filter = opts.get("--filter");
    let all = opts.flag("--all");
    let mut performance_core = true;
    let mut signals = Signals::load()?;

    let unix_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut header = String::new();
    let _ = writeln!(header, "# perf_harness bench");
    let _ = writeln!(header);
    let _ = writeln!(
        header,
        "- opus-rs {} at git {}, C reference: {}",
        env!("CARGO_PKG_VERSION"),
        git_revision(),
        libopus_version()
    );
    let _ = writeln!(
        header,
        "- clip {seconds} s; one round = {reps} repetitions per operation: {procs} fresh processes per scenario, each with {reps_per_proc} repetitions after 1 warm-up; Rust and C interleaved; unix time {unix_time}"
    );
    let _ = writeln!(
        header,
        "- a round in which a median is more than 3 % above its minimum was disturbed by other programs; the scenario is then measured again, up to {max_rounds} rounds (column `rounds`)"
    );
    let _ = writeln!(
        header,
        "- measuring threads ask for QOS_CLASS_USER_INTERACTIVE (performance core); code padding (PERF_HARNESS_PAD): {LAYOUT_PAD} bytes"
    );
    let _ = writeln!(
        header,
        "- times are nanoseconds per frame (one encode or decode call); min = lowest repetition of all rounds, med = lowest median of any one round"
    );
    let _ = writeln!(
        header,
        "- xRT = frame duration / minimum time; Rust/C min and Rust/C med = ratio of the two minimums and of the two medians (below 1 means Rust is faster)"
    );
    let _ = writeln!(
        header,
        "- on a machine that other programs are using, the medians and their ratio repeat better than the minimums, most of all for decode"
    );
    let _ = writeln!(
        header,
        "- both decoders decode the C-encoded stream; `streams` is `same` when both encoders have the settings of the scenario and the two sides produced equivalent streams, and `CHECK` when a note below the table says what differs"
    );
    let _ = writeln!(
        header,
        "- `timing` is `UNSTEADY` when the median of an operation stayed more than 3 % above its minimum in every round; a note below the table names the operation, and its median is the number to use"
    );
    let _ = writeln!(
        header,
        "- `only C runs` lists encoder steps that libopus runs for the scenario and opus-rs does not have: `analysis` (tonality analysis), `stereo-angle` (second quantisation pass per stereo band); the C encode time includes them, so the encode ratio of such a row flatters opus-rs"
    );
    let _ = writeln!(
        header,
        "- the C SILK encoder is the floating-point one of libopus (`silk/float`), the opus-rs SILK encoder is a port of the fixed-point one"
    );
    let _ = writeln!(
        header,
        "- the Rust encode time of the SILK complexity-5 rows depends on where the linker places the code: two states about 16 % apart were measured, and any change to the library or to this program can switch between them; compare such a row over several PERF_HARNESS_PAD builds with the `spread` command before trusting a change"
    );
    println!("{header}");
    println!("{BENCH_HEADER}");

    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for sc in scenarios() {
        let name = sc.name();
        if !(sc.bench || all) || !name_matches(&name, filter) {
            continue;
        }
        let clip = signals.clip(&sc, seconds, 0);
        match bench_one(&sc, &clip, plan) {
            Ok((row, raised)) => {
                performance_core &= raised;
                println!("{}", bench_row_markdown(&row));
                let _ = std::io::stdout().flush();
                rows.push(row);
            }
            Err(e) => {
                eprintln!("{name}: FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if rows.is_empty() && failures.is_empty() {
        return Err("no scenario matches the filter".to_string());
    }

    let mut footer = String::new();
    for row in &rows {
        for note in row.notes.iter().chain(&row.unsteady) {
            let _ = writeln!(footer, "- {}: {note}", row.name);
        }
    }
    for failure in &failures {
        let _ = writeln!(footer, "- FAILED {failure}");
    }
    if !footer.is_empty() {
        footer = format!("\nScenarios that need attention:\n\n{footer}");
    }
    if !performance_core {
        footer.push_str(
            "\nNote: a measuring thread could not raise its priority, so it may have run on an efficiency core.\n",
        );
    }
    print!("{footer}");

    let mut comparison = String::new();
    if let Some(path) = opts.get("--compare") {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let baseline: BTreeMap<String, &str> = text
            .lines()
            .filter_map(|line| Some((json_text_field(line, "name")?, line)))
            .collect();
        let _ = writeln!(comparison, "\nChange against {path}: minimum / median");
        let _ = writeln!(
            comparison,
            "(negative = faster now. The C library did not change, so the C columns show how much the machine itself moved between the two runs. \
             `Rust against C` is the change of the Rust time divided by the change of the C time: a slowdown or speed-up of the whole machine cancels out there, \
             so that is the column to read for a change in opus-rs.)\n"
        );
        let _ = writeln!(
            comparison,
            "| scenario | Rust enc | C enc | enc: Rust against C | Rust dec | C dec | dec: Rust against C |\n|---|---:|---:|---:|---:|---:|---:|"
        );
        for row in &rows {
            let Some(line) = baseline.get(&row.name) else {
                let _ = writeln!(comparison, "| {} | not in baseline | | | | | |", row.name);
                continue;
            };
            // Now divided by before, for the minimum and for the median.
            let factors = |operation: &str, now: Stat| -> Option<(f64, f64)> {
                let before_min = json_number_field(line, &format!("{operation}_min_ns")).filter(|v| *v > 0.0)?;
                let before_med = json_number_field(line, &format!("{operation}_med_ns")).filter(|v| *v > 0.0)?;
                Some((now.min_ns / before_min, now.median_ns / before_med))
            };
            let percent = |factor: f64| format!("{:+.1}%", (factor - 1.0) * 100.0);
            let cell = |pair: Option<(f64, f64)>| match pair {
                Some((min, med)) => format!("{} / {}", percent(min), percent(med)),
                None => "?".to_string(),
            };
            let relative = |rust: Option<(f64, f64)>, c: Option<(f64, f64)>| match (rust, c) {
                (Some(rust), Some(c)) => Some((rust.0 / c.0, rust.1 / c.1)),
                _ => None,
            };
            let (rust_enc, c_enc) = (factors("rust_enc", row.rust_enc), factors("c_enc", row.c_enc));
            let (rust_dec, c_dec) = (factors("rust_dec", row.rust_dec), factors("c_dec", row.c_dec));
            let _ = writeln!(
                comparison,
                "| {} | {} | {} | {} | {} | {} | {} |",
                row.name,
                cell(rust_enc),
                cell(c_enc),
                cell(relative(rust_enc, c_enc)),
                cell(rust_dec),
                cell(c_dec),
                cell(relative(rust_dec, c_dec)),
            );
        }
        print!("{comparison}");
    }

    if let Some(path) = opts.get("--md") {
        let mut text = format!("{header}\n{BENCH_HEADER}\n");
        for row in &rows {
            let _ = writeln!(text, "{}", bench_row_markdown(row));
        }
        text.push_str(&footer);
        text.push_str(&comparison);
        write_file(path, &text)?;
    }
    if let Some(path) = opts.get("--json") {
        let mut text = String::new();
        let _ = writeln!(
            text,
            "{{\"meta\":{{\"opus_rs\":{},\"git\":{},\"libopus\":{},\"clip_seconds\":{seconds},\"reps\":{reps},\"procs\":{procs},\"reps_per_proc\":{reps_per_proc},\"max_rounds\":{max_rounds},\"unix_time\":{unix_time},\"performance_core_requested\":{performance_core},\"layout_pad\":{LAYOUT_PAD}}},",
            json_string(env!("CARGO_PKG_VERSION")),
            json_string(&git_revision()),
            json_string(&libopus_version()),
        );
        let _ = writeln!(text, "\"results\":[");
        let lines: Vec<String> = rows.iter().map(bench_row_json).collect();
        let _ = writeln!(text, "{}", lines.join(",\n"));
        let _ = writeln!(text, "]}}");
        write_file(path, &text)?;
    }
    Ok(if failures.is_empty() { 0 } else { 1 })
}

/// Compare several `bench --json` files. For each scenario and each of the
/// four operations, print the lowest minimum, and how far the highest
/// minimum and the highest median are above the lowest ones.
///
/// Use it on repeated runs of one build to see the measurement noise, or on
/// builds with different `PERF_HARNESS_PAD` to see how much the timings
/// depend on where the code is placed.
fn cmd_spread(paths: &[String]) -> Result<i32, String> {
    if paths.len() < 2 {
        return Err("spread needs at least two bench JSON files".to_string());
    }
    // What one scenario has in all files: for each operation the minimums
    // and the medians, and the Rust/C ratios of the medians for encode and
    // for decode.
    #[derive(Default)]
    struct Values {
        times: [(Vec<f64>, Vec<f64>); 4],
        ratios: [Vec<f64>; 2],
    }
    let mut table: Vec<(String, Values)> = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        for line in text.lines() {
            let Some(name) = json_text_field(line, "name") else {
                continue;
            };
            let index = match table.iter().position(|(n, _)| *n == name) {
                Some(index) => index,
                None => {
                    table.push((name, Values::default()));
                    table.len() - 1
                }
            };
            let values = &mut table[index].1;
            for ((minimums, medians), operation) in values.times.iter_mut().zip(OPERATIONS) {
                minimums.extend(json_number_field(line, &format!("{operation}_min_ns")));
                medians.extend(json_number_field(line, &format!("{operation}_med_ns")));
            }
            for (ratios, key) in values.ratios.iter_mut().zip(["enc_ratio_med", "dec_ratio_med"]) {
                ratios.extend(json_number_field(line, key));
            }
        }
    }
    let lowest = |values: &[f64]| values.iter().copied().fold(f64::INFINITY, f64::min);
    let highest = |values: &[f64]| values.iter().copied().fold(0.0, f64::max);
    // Highest value above the lowest, in percent.
    let spread = |values: &[f64]| (highest(values) / lowest(values) - 1.0) * 100.0;
    println!(
        "{} files. Per operation: lowest minimum (ns per frame), spread of the minimums, spread of the medians.",
        paths.len()
    );
    println!("Spread = how far the highest value is above the lowest.");
    println!(
        "Rust/C columns: lowest and highest ratio of the two medians. A slowdown of the whole machine moves the times but not this ratio.\n"
    );
    println!(
        "| scenario | Rust enc | min | med | C enc | min | med | Rust dec | min | med | C dec | min | med | enc Rust/C | dec Rust/C |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    );
    let mut worst = [(0.0f64, 0.0f64); 4];
    let mut worst_ratio = [0.0f64; 2];
    for (name, values) in &table {
        let mut row = format!("| {name} |");
        for (column, (minimums, medians)) in values.times.iter().enumerate() {
            if minimums.is_empty() || medians.is_empty() {
                row.push_str(" - | - | - |");
                continue;
            }
            let (of_min, of_med) = (spread(minimums), spread(medians));
            worst[column] = (worst[column].0.max(of_min), worst[column].1.max(of_med));
            let _ = write!(row, " {:.0} | {of_min:.1}% | {of_med:.1}% |", lowest(minimums));
        }
        for (column, ratios) in values.ratios.iter().enumerate() {
            if ratios.is_empty() {
                row.push_str(" - |");
                continue;
            }
            worst_ratio[column] = worst_ratio[column].max(spread(ratios));
            let _ = write!(row, " {:.3} to {:.3} |", lowest(ratios), highest(ratios));
        }
        println!("{row}");
    }
    println!("\nLargest spread (minimums / medians):");
    for (operation, (of_min, of_med)) in OPERATION_LABELS.iter().zip(worst) {
        println!("- {operation}: {of_min:.1}% / {of_med:.1}%");
    }
    println!(
        "- Rust/C ratio of the medians: encode {:.1}%, decode {:.1}%",
        worst_ratio[0], worst_ratio[1]
    );
    Ok(0)
}

// ---------------------------------------------------------------------------
// run: a loop for a sampling profiler
// ---------------------------------------------------------------------------

/// Loop one operation for `seconds`. The codec state lives for the whole
/// loop, and the clock is read once per pass over the clip. Returns the
/// number of frames processed and the time spent.
#[inline(never)]
fn run_loop(mut pass: impl FnMut(), frames_per_pass: usize, seconds: f64) -> (u64, Duration) {
    let limit = Duration::from_secs_f64(seconds);
    let mut frames = 0u64;
    let start = Instant::now();
    loop {
        pass();
        frames += frames_per_pass as u64;
        let elapsed = start.elapsed();
        if elapsed >= limit {
            return (frames, elapsed);
        }
    }
}

fn cmd_run(opts: &Opts) -> Result<i32, String> {
    let sc = find_scenario(opts.get("--scenario").ok_or("run needs --scenario NAME")?)?;
    let op = opts.get("--op").ok_or("run needs --op enc|dec")?;
    let implementation = opts.get("--impl").unwrap_or("rust");
    let seconds: f64 = opts.number("--seconds", 10.0)?;
    let clip_seconds: f64 = opts.number("--clip-seconds", 6.0)?;

    // Everything that is not the measured operation happens before the loop.
    let mut signals = Signals::load()?;
    let clip = signals.clip(&sc, clip_seconds, 0);
    let packets = c_encode_all(&sc, &clip, true)?;
    let mut pcm_out = vec![0.0f32; sc.dec_frame_samples() * sc.channels];
    let mut packet_out = [0u8; MAX_PACKET];
    let dec_frame = sc.dec_frame_samples();
    raise_thread_priority();

    println!("pid {}", std::process::id());
    println!("scenario {} op {op} impl {implementation} for {seconds} s", sc.name());
    let _ = std::io::stdout().flush();

    let (frames, elapsed) = match (op, implementation) {
        ("enc", "rust") => {
            let mut encoder = rust_encoder(&sc, true)?;
            run_loop(
                || {
                    for frame in clip.pcm.chunks_exact(clip.step) {
                        let _ = black_box(encoder.encode(frame, clip.frame, &mut packet_out));
                    }
                },
                clip.frames,
                seconds,
            )
        }
        ("enc", "c") => {
            let mut encoder = CEncoder::new(&sc, true)?;
            run_loop(
                || {
                    for frame in clip.pcm.chunks_exact(clip.step) {
                        black_box(encoder.encode(frame, clip.frame, &mut packet_out));
                    }
                },
                clip.frames,
                seconds,
            )
        }
        ("dec", "rust") => {
            let mut decoder = rust_decoder(&sc)?;
            run_loop(
                || {
                    for packet in packets.iter() {
                        let _ = black_box(decoder.decode(packet, dec_frame, &mut pcm_out));
                    }
                },
                clip.frames,
                seconds,
            )
        }
        ("dec", "c") => {
            let mut decoder = CDecoder::new(&sc)?;
            run_loop(
                || {
                    for packet in packets.iter() {
                        black_box(decoder.decode(packet, dec_frame, &mut pcm_out));
                    }
                },
                clip.frames,
                seconds,
            )
        }
        _ => return Err("--op must be enc or dec, --impl must be rust or c".to_string()),
    };
    println!(
        "done: {frames} frames in {:.3} s, {:.0} ns per frame",
        elapsed.as_secs_f64(),
        elapsed.as_nanos() as f64 / frames as f64
    );
    Ok(0)
}

// ---------------------------------------------------------------------------
// golden: hashes of the codec output
// ---------------------------------------------------------------------------

/// The hash fields of one scenario, in file order.
///
/// * `enc`: Rust packets from `encode` (lengths and bytes).
/// * `enc_i16`: Rust packets from `encode_i16` on the same audio.
/// * `dec_rust`: Rust decoder output for the Rust packets.
/// * `c_pk`: C packets. It changes only when libopus or the input changes.
/// * `dec_c`: Rust decoder output for the C packets.
/// * `dec_c_plc`: the same with one packet in ten lost.
/// * `dec_c_burst`: the same with, in addition, three packets in a row lost
///   out of every 25.
/// * `fec_silk`: SILK decoder output with lost packets recovered from the
///   redundancy data (`-` where that does not apply).
/// * `fec_frames`: how many frames that recovery used.
/// * `errs`: packets the Rust decoder rejected in the four decodes.
///
/// A step that fails gives `ERR(reason)` in its field. Each step runs on its
/// own, so a step that panics does not hide the results of the others.
fn golden_fields(sc: &Scenario, clip: &Clip) -> Vec<(&'static str, String)> {
    let mut fields = Vec::new();
    let mut errors = 0usize;

    let rust_packets = guarded(|| rust_encode_all(sc, clip));
    fields.push(("enc", packets_field(&rust_packets)));
    fields.push(("enc_i16", packets_field(&guarded(|| rust_encode_all_i16(sc, clip)))));
    fields.push(("dec_rust", decode_field(sc, &rust_packets, Loss::None, &mut errors)));
    let c_packets = guarded(|| c_encode_all(sc, clip, true));
    fields.push(("c_pk", packets_field(&c_packets)));
    fields.push(("dec_c", decode_field(sc, &c_packets, Loss::None, &mut errors)));
    fields.push(("dec_c_plc", decode_field(sc, &c_packets, Loss::Single, &mut errors)));
    fields.push(("dec_c_burst", decode_field(sc, &c_packets, Loss::Bursts, &mut errors)));
    let fec = match &c_packets {
        Ok(packets) => guarded(|| Ok(silk_fec_decode(sc, packets))),
        Err(_) => Ok(None),
    };
    let (fec_hash, fec_frames) = match fec {
        Ok(Some((hash, frames))) => (format!("{hash:016x}"), frames.to_string()),
        Ok(None) => ("-".to_string(), "-".to_string()),
        Err(e) => (error_field(&e), "-".to_string()),
    };
    fields.push(("fec_silk", fec_hash));
    fields.push(("fec_frames", fec_frames));
    fields.push(("errs", errors.to_string()));
    fields
}

/// Run one step of `golden`. A panic becomes an error.
fn guarded<T>(step: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(step)).unwrap_or_else(|_| Err("panic".to_string()))
}

fn error_field(error: &str) -> String {
    format!("ERR({})", error.replace(' ', "_"))
}

/// Golden field for a packet stream: its hash, or the error.
fn packets_field(packets: &Result<Packets, String>) -> String {
    match packets {
        Ok(packets) => format!("{:016x}", packets.hash()),
        Err(e) => error_field(e),
    }
}

/// Golden field for the Rust decoder output on `packets`: its hash, the
/// error, or `-` when there are no packets. Adds the rejected packets to
/// `errors`.
fn decode_field(
    sc: &Scenario,
    packets: &Result<Packets, String>,
    loss: Loss,
    errors: &mut usize,
) -> String {
    let Ok(packets) = packets else {
        return "-".to_string();
    };
    match guarded(|| rust_decode_all(sc, packets, loss)) {
        Ok(decoded) => {
            *errors += decoded.errors;
            format!("{:016x}", decoded.hash)
        }
        Err(e) => error_field(&e),
    }
}

fn golden_line(sc: &Scenario, clip: &Clip) -> String {
    let mut line = sc.name();
    for (key, value) in golden_fields(sc, clip) {
        let _ = write!(line, " {key}={value}");
    }
    line
}

// Streams that change the coding mode while they run.
//
// A scenario keeps one mode for the whole clip. The code that handles a
// change of mode, of bandwidth or of the SILK sample rate inside a stream is
// therefore only reached by the streams below. Their bit rate follows
// `SWITCH_BITRATES`, one step every 200 ms, and neither mode nor bandwidth
// is forced:
//
// * libopus then moves between SILK, Hybrid and CELT and between the
//   bandwidths. The Rust decoder decodes that stream.
// * opus-rs moves between SILK or Hybrid and CELT as well. It has no
//   speech/music detection, so it does not switch on the same packets.
//
// For stereo input both encoders also change between stereo and mono
// packets. The stereo streams have both encoders forced to stereo, except
// the one named `..._free`, in which the Rust decoder gets mono packets in a
// stereo stream.

/// Bit rates in bits per second, one per 200 ms, repeated.
const SWITCH_BITRATES: [i32; 10] = [
    8_000, 16_000, 24_000, 40_000, 96_000, 12_000, 64_000, 20_000, 128_000, 32_000,
];

/// One mode-switching stream.
struct SwitchCase {
    name: String,
    /// Carries the sample rate, channel count, application, signal and frame
    /// size. Its `mode`, `bitrate` and name are not used.
    sc: Scenario,
    /// Let libopus choose between stereo and mono packets for stereo input.
    free_channels: bool,
}

fn switch_cases() -> Vec<SwitchCase> {
    let case = |rate: u32, channels: usize, app: App, signal: Signal, frame_us: u32, free_channels: bool| {
        let mut sc = Scenario::new(Mode::Celt, rate, channels, SWITCH_BITRATES[0], 9).frame(frame_us);
        sc.app = app;
        sc.signal = signal;
        let name = format!(
            "switch_{}k_{}_{}_{}_{}{}",
            rate / 1000,
            if channels == 2 { "stereo" } else { "mono" },
            if app == App::Voip { "voip" } else { "audio" },
            if signal == Signal::Speech { "speech" } else { "music" },
            frame_label(frame_us),
            if free_channels { "_free" } else { "" },
        );
        SwitchCase {
            name,
            sc,
            free_channels,
        }
    };
    vec![
        case(48000, 1, App::Voip, Signal::Speech, 20_000, false),
        case(48000, 2, App::Audio, Signal::Music, 20_000, false),
        case(48000, 2, App::Voip, Signal::Speech, 20_000, false),
        case(48000, 2, App::Audio, Signal::Music, 20_000, true),
        case(48000, 1, App::Audio, Signal::Music, 10_000, false),
        case(24000, 1, App::Voip, Signal::Speech, 20_000, false),
        case(24000, 2, App::Audio, Signal::Music, 20_000, false),
        case(16000, 1, App::Voip, Signal::Speech, 20_000, false),
    ]
}

/// The bit rate for the frame with this index.
fn switch_bitrate(sc: &Scenario, index: usize) -> i32 {
    let frames_per_step = (200_000 / sc.frame_us).max(1) as usize;
    SWITCH_BITRATES[(index / frames_per_step) % SWITCH_BITRATES.len()]
}

fn rust_encode_switching(case: &SwitchCase, clip: &Clip) -> Result<Packets, String> {
    let sc = &case.sc;
    let mut encoder = rust_encoder(sc, false)?;
    if sc.channels == 2 && !case.free_channels {
        encoder
            .set_force_channels(Some(2))
            .map_err(|e| format!("set_force_channels: {e}"))?;
    }
    let mut out = [0u8; MAX_PACKET];
    let mut packets = Packets::default();
    for (index, frame) in clip.pcm.chunks_exact(clip.step).enumerate() {
        encoder.bitrate_bps = switch_bitrate(sc, index);
        let len = encoder
            .encode(frame, clip.frame, &mut out)
            .map_err(|e| format!("Rust encode failed at frame {index}: {e}"))?;
        packets.push(&out[..len]);
    }
    Ok(packets)
}

fn c_encode_switching(case: &SwitchCase, clip: &Clip) -> Result<Packets, String> {
    let sc = &case.sc;
    let mut encoder = CEncoder::new(sc, false)?;
    if sc.channels == 2 && !case.free_channels {
        encoder.ctl(ffi::OPUS_SET_FORCE_CHANNELS_REQUEST, 2)?;
    }
    let mut out = [0u8; MAX_PACKET];
    let mut packets = Packets::default();
    for (index, frame) in clip.pcm.chunks_exact(clip.step).enumerate() {
        encoder.ctl(ffi::OPUS_SET_BITRATE_REQUEST, switch_bitrate(sc, index))?;
        let len = encoder.encode(frame, clip.frame, &mut out);
        if len < 0 {
            return Err(format!("C encode failed at frame {index} with {len}"));
        }
        packets.push(&out[..len as usize]);
    }
    Ok(packets)
}

/// "distinct packet configurations / changes from one packet to the next".
fn switch_count(packets: &Packets) -> String {
    let keys: Vec<u8> = packets.iter().map(|p| p.first().map_or(0xFF, |toc| toc >> 2)).collect();
    let changes = keys.windows(2).filter(|pair| pair[0] != pair[1]).count();
    format!("{}/{changes}", toc_histogram(packets).len())
}

/// Golden line of one mode-switching stream.
///
/// * `enc`, `dec_rust`: Rust packets and the Rust decoder output for them.
/// * `c_pk`, `dec_c`, `dec_c_plc`, `dec_c_burst`: C packets and the Rust
///   decoder output for them, without loss and with the two loss patterns.
/// * `rust_cfg`, `c_cfg`: how many packet configurations each stream uses
///   and how often the configuration changes. They show that the streams
///   really switch.
/// * `errs`: packets the Rust decoder rejected in the four decodes.
fn golden_switch_line(case: &SwitchCase, clip: &Clip) -> String {
    let sc = &case.sc;
    let mut errors = 0usize;
    let count = |packets: &Result<Packets, String>| packets.as_ref().map_or("-".to_string(), switch_count);
    let rust_packets = guarded(|| rust_encode_switching(case, clip));
    let c_packets = guarded(|| c_encode_switching(case, clip));
    let fields = [
        ("enc", packets_field(&rust_packets)),
        ("dec_rust", decode_field(sc, &rust_packets, Loss::None, &mut errors)),
        ("c_pk", packets_field(&c_packets)),
        ("dec_c", decode_field(sc, &c_packets, Loss::None, &mut errors)),
        ("dec_c_plc", decode_field(sc, &c_packets, Loss::Single, &mut errors)),
        ("dec_c_burst", decode_field(sc, &c_packets, Loss::Bursts, &mut errors)),
        ("rust_cfg", count(&rust_packets)),
        ("c_cfg", count(&c_packets)),
        ("errs", errors.to_string()),
    ];
    let mut line = case.name.clone();
    for (key, value) in fields {
        let _ = write!(line, " {key}={value}");
    }
    line
}

fn golden_header() -> String {
    format!(
        "# perf_harness golden v2 clip_seconds={GOLDEN_SECONDS} libopus={}",
        libopus_version().replace(' ', "_")
    )
}

fn cmd_golden(opts: &Opts) -> Result<i32, String> {
    let filter = opts.get("--filter");
    let perturb: i32 = opts.number("--perturb", 0)?;
    let (write_path, check_path) = (opts.get("--write"), opts.get("--check"));
    if write_path.is_some() == check_path.is_some() {
        return Err("golden needs exactly one of --write PATH or --check PATH".to_string());
    }
    if perturb != 0 {
        eprintln!("self-test: one input sample is changed by {perturb} / 32768");
    }
    let start = Instant::now();
    let mut signals = Signals::load()?;
    let mut lines = Vec::new();
    for sc in scenarios() {
        if !name_matches(&sc.name(), filter) {
            continue;
        }
        let clip = signals.clip(&sc, GOLDEN_SECONDS, perturb);
        lines.push(golden_line(&sc, &clip));
    }
    for case in switch_cases() {
        if !name_matches(&case.name, filter) {
            continue;
        }
        let clip = signals.clip(&case.sc, GOLDEN_SECONDS, perturb);
        lines.push(golden_switch_line(&case, &clip));
    }
    if lines.is_empty() {
        return Err("no scenario matches the filter".to_string());
    }
    let problems: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("ERR(") || !l.ends_with("errs=0"))
        .collect();

    if let Some(path) = write_path {
        let mut text = golden_header();
        text.push('\n');
        for line in &lines {
            text.push_str(line);
            text.push('\n');
        }
        write_file(path, &text)?;
        println!(
            "wrote {} scenarios to {path} in {:.1} s",
            lines.len(),
            start.elapsed().as_secs_f64()
        );
        for line in &problems {
            println!("note: this scenario has errors: {line}");
        }
        return Ok(0);
    }

    let path = check_path.unwrap_or_default();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut stored: BTreeMap<&str, BTreeMap<&str, &str>> = BTreeMap::new();
    let mut stored_header = "";
    for line in text.lines() {
        if line.starts_with('#') {
            stored_header = line;
            continue;
        }
        let mut tokens = line.split_whitespace();
        if let Some(name) = tokens.next() {
            stored.insert(name, tokens.filter_map(|t| t.split_once('=')).collect());
        }
    }
    if stored_header != golden_header() {
        println!("note: file header is `{stored_header}`, this build gives `{}`", golden_header());
    }

    let mut differing = 0usize;
    let mut seen = Vec::new();
    for line in &lines {
        let mut tokens = line.split_whitespace();
        let name = tokens.next().unwrap_or_default();
        seen.push(name);
        let current: Vec<(&str, &str)> = tokens.filter_map(|t| t.split_once('=')).collect();
        let Some(before) = stored.get(name) else {
            println!("DIFF {name}: not in the golden file");
            differing += 1;
            continue;
        };
        let mut changed = Vec::new();
        for &(key, value) in &current {
            match before.get(key) {
                Some(&old) if old == value => {}
                Some(&old) => changed.push(format!("{key} {old} -> {value}")),
                None => changed.push(format!("{key} (missing) -> {value}")),
            }
        }
        for key in before.keys() {
            if !current.iter().any(|(k, _)| k == key) {
                changed.push(format!("{key} {} -> (missing)", before[key]));
            }
        }
        if !changed.is_empty() {
            differing += 1;
            println!("DIFF {name}: {}", changed.join(", "));
            if changed.iter().any(|c| c.starts_with("c_pk ")) {
                println!(
                    "     note: c_pk changed, so the C reference stream itself differs (other libopus build or other input)"
                );
            }
        }
    }
    if filter.is_none() {
        for name in stored.keys().filter(|name| !seen.contains(name)) {
            println!("DIFF {name}: in the golden file, but no longer a scenario");
            differing += 1;
        }
    }
    println!(
        "checked {} scenarios against {path} in {:.1} s: {differing} differ",
        lines.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(if differing == 0 { 0 } else { 1 })
}

// ---------------------------------------------------------------------------
// modes: compare the two packet streams and the two decoders
// ---------------------------------------------------------------------------

/// Signal-to-noise ratio in dB of `test` against `reference`. 999 means the
/// two are identical.
fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (&r, &t) in reference.iter().zip(test) {
        signal += r as f64 * r as f64;
        noise += (r as f64 - t as f64) * (r as f64 - t as f64);
    }
    if noise == 0.0 {
        999.0
    } else {
        10.0 * (signal.max(1e-30) / noise).log10()
    }
}

/// Find the delay of `test` behind `reference` (in sample frames, from
/// `-max_delay` to `max_delay`) that gives the highest correlation, and the
/// signal-to-noise ratio in dB at that delay.
///
/// The correlation is taken over the loudest quarter second of the first
/// channel of `reference`. Returns `None` when the clip is too short.
fn aligned_snr(
    reference: &[f32],
    test: &[f32],
    channels: usize,
    rate: u32,
    max_delay: usize,
    allow_negative: bool,
) -> Option<(f64, isize)> {
    let frames = reference.len().min(test.len()) / channels;
    let window = (rate / 4) as usize;
    if frames < 2 * window + 2 * max_delay {
        return None;
    }
    let energy = |start: usize| -> f64 {
        (start..start + window)
            .map(|i| (reference[i * channels] as f64).powi(2))
            .sum()
    };
    let start = (max_delay..frames - window - max_delay)
        .step_by(window)
        .max_by(|&a, &b| energy(a).total_cmp(&energy(b)))?;
    let correlation = |delay: isize| -> f64 {
        (start..start + window)
            .map(|i| {
                let j = (i as isize + delay) as usize;
                reference[i * channels] as f64 * test[j * channels] as f64
            })
            .sum()
    };
    let lowest = if allow_negative { -(max_delay as isize) } else { 0 };
    let delay = (lowest..=max_delay as isize)
        .max_by(|&a, &b| correlation(a).total_cmp(&correlation(b)))?;
    let shift = delay.unsigned_abs() * channels;
    let count = (frames - delay.unsigned_abs()) * channels;
    let snr = if delay >= 0 {
        snr_db(&reference[..count], &test[shift..shift + count])
    } else {
        snr_db(&reference[shift..shift + count], &test[..count])
    };
    Some((snr, delay))
}

/// Signal-to-noise ratio in dB of a decoded clip against the encoder input,
/// and the codec delay in samples (0 to 40 ms) at which it was measured.
///
/// A waveform measure like this says little about how a codec sounds. It is
/// only meant to show an encoder or decoder that is far worse than its
/// counterpart on the same input.
fn chain_snr(input: &[f32], decoded: &[f32], channels: usize, rate: u32) -> Option<(f64, isize)> {
    aligned_snr(input, decoded, channels, rate, (rate / 25) as usize, false)
}

/// How well two decoders agree on the same packets.
struct Agreement {
    /// Signal-to-noise ratio in dB with no time shift. 999 means identical.
    snr: f64,
    /// When `snr` is poor: the ratio at the best time shift within 10 ms,
    /// and that shift in samples (positive = the Rust output is later).
    shifted: Option<(f64, isize)>,
}

impl Agreement {
    /// Below this the two outputs are reported as different.
    const POOR_DB: f64 = 40.0;

    fn measure(c_output: &[f32], rust_output: &[f32], channels: usize, rate: u32) -> Self {
        let snr = snr_db(c_output, rust_output);
        let shifted = if snr < Self::POOR_DB {
            aligned_snr(c_output, rust_output, channels, rate, (rate / 100) as usize, true)
        } else {
            None
        };
        Self { snr, shifted }
    }

    fn label(&self) -> String {
        match (self.snr, self.shifted) {
            (snr, _) if snr >= 999.0 => "exact".to_string(),
            (snr, Some((best, shift))) if shift != 0 => format!("{snr:.1} ({best:.1} @{shift:+})"),
            (snr, _) => format!("{snr:.1}"),
        }
    }

    fn problem(&self, which: &str) -> Option<String> {
        if self.snr >= Self::POOR_DB {
            return None;
        }
        Some(match self.shifted {
            Some((best, shift)) if shift != 0 => format!(
                "Rust and C decoders differ on the {which} packets: {:.1} dB; {best:.1} dB when the Rust output is shifted by {shift:+} samples",
                self.snr
            ),
            _ => format!(
                "Rust and C decoders differ on the {which} packets: {:.1} dB",
                self.snr
            ),
        })
    }
}

/// Everything `modes` and `bench` want to know about how the Rust and the C
/// side behave on one clip.
struct Comparison {
    expected: String,
    rust_toc: String,
    c_toc: String,
    rust_kbps: f64,
    c_kbps: f64,
    identical_packets: usize,
    /// Rust decoder against C decoder on the C packets, then on the Rust
    /// packets.
    agree_c_packets: Agreement,
    agree_rust_packets: Agreement,
    /// Decoded audio against the encoder input: Rust encoder with C decoder,
    /// C encoder with C decoder, C encoder with Rust decoder.
    chain_rust_enc: Option<(f64, isize)>,
    chain_c: Option<(f64, isize)>,
    chain_rust_dec: Option<(f64, isize)>,
    /// Empty when the two sides did equivalent work: the settings of the
    /// scenario on both sides, same packet configuration, bit rates within
    /// 5 %, no rejected packet, and decoded audio of similar accuracy.
    problems: Vec<String>,
}

fn compare(sc: &Scenario, clip: &Clip) -> Result<Comparison, String> {
    let (rust_packets, rust_settings) = rust_encode_all_with_settings(sc, clip)?;
    let (c_packets, c_settings) = c_encode_all_with_settings(sc, clip, true)?;
    let rust_histogram = toc_histogram(&rust_packets);
    let c_histogram = toc_histogram(&c_packets);
    let expected: Histogram = [(sc.expected_toc_key(), clip.frames)].into_iter().collect();

    let mut problems = Vec::new();
    for mismatch in c_settings.mismatches(sc) {
        problems.push(format!("C encoder setting differs from the scenario: {mismatch}"));
    }
    if rust_settings != RustSettings::wanted(sc) {
        problems.push(format!(
            "Rust encoder settings differ from the scenario: {rust_settings:?}"
        ));
    }
    if rust_histogram != c_histogram {
        problems.push("Rust and forced C streams have different packet configurations".to_string());
    }
    if rust_histogram != expected {
        problems.push("Rust stream is not the expected configuration".to_string());
    }
    let (rust_kbps, c_kbps) = (rust_packets.kbps(sc), c_packets.kbps(sc));
    if (rust_kbps - c_kbps).abs() > 0.05 * c_kbps {
        problems.push(format!(
            "bit rate differs: Rust {rust_kbps:.1} kb/s, C {c_kbps:.1} kb/s"
        ));
    }

    let rust_on_c = rust_decode_all(sc, &c_packets, Loss::None)?;
    let c_on_c = c_decode_all(sc, &c_packets)?;
    let rust_on_rust = rust_decode_all(sc, &rust_packets, Loss::None)?;
    let c_on_rust = c_decode_all(sc, &rust_packets)?;
    let checks = [
        ("Rust decoder", "C packets", &rust_on_c),
        ("C decoder", "C packets", &c_on_c),
        ("Rust decoder", "Rust packets", &rust_on_rust),
        ("C decoder", "Rust packets", &c_on_rust),
    ];
    for (who, what, decoded) in checks {
        if decoded.errors > 0 {
            problems.push(format!(
                "{who} rejected {what}: {} of {} ({})",
                decoded.errors,
                clip.frames,
                decoded.first_error.as_deref().unwrap_or("?")
            ));
        }
        if decoded.wrong_size > 0 {
            problems.push(format!(
                "{who} returned another number of samples than one frame ({}) for {} of {} {what}",
                sc.dec_frame_samples(),
                decoded.wrong_size,
                clip.frames
            ));
        }
    }
    let agree_c_packets = Agreement::measure(&c_on_c.pcm, &rust_on_c.pcm, sc.channels, sc.dec_rate);
    let agree_rust_packets =
        Agreement::measure(&c_on_rust.pcm, &rust_on_rust.pcm, sc.channels, sc.dec_rate);
    problems.extend(agree_c_packets.problem("C"));
    problems.extend(agree_rust_packets.problem("Rust"));

    let chain = |decoded: &Decoded| {
        if sc.dec_rate == sc.rate {
            chain_snr(&clip.pcm, &decoded.pcm, clip.channels, sc.rate)
        } else {
            None
        }
    };
    let (chain_rust_enc, chain_c, chain_rust_dec) = (chain(&c_on_rust), chain(&c_on_c), chain(&rust_on_c));
    // A delay difference below 1/8000 s is within the accuracy of the
    // correlation peak.
    let delay_tolerance = (sc.rate / 8000) as isize;
    if let (Some((rust_snr, rust_delay)), Some((c_snr, c_delay))) = (chain_rust_enc, chain_c) {
        if rust_snr < c_snr - 3.0 {
            problems.push(format!(
                "Rust encoder is {:.1} dB below the C encoder against the input (both decoded by the C decoder)",
                c_snr - rust_snr
            ));
        }
        if (rust_delay - c_delay).abs() > delay_tolerance {
            problems.push(format!(
                "encoder delay differs: Rust {rust_delay} samples, C {c_delay} samples"
            ));
        }
    }
    if let (Some((rust_snr, rust_delay)), Some((c_snr, c_delay))) = (chain_rust_dec, chain_c) {
        if rust_snr < c_snr - 3.0 {
            problems.push(format!(
                "Rust decoder is {:.1} dB below the C decoder against the input (both on the C packets)",
                c_snr - rust_snr
            ));
        }
        if (rust_delay - c_delay).abs() > delay_tolerance {
            problems.push(format!(
                "decoder delay differs: Rust {rust_delay} samples, C {c_delay} samples"
            ));
        }
    }

    let identical_packets = rust_packets
        .iter()
        .zip(c_packets.iter())
        .filter(|(a, b)| a == b)
        .count();
    Ok(Comparison {
        expected: toc_label(sc.expected_toc_key()),
        rust_toc: histogram_label(&rust_histogram),
        c_toc: histogram_label(&c_histogram),
        rust_kbps,
        c_kbps,
        identical_packets,
        agree_c_packets,
        agree_rust_packets,
        chain_rust_enc,
        chain_c,
        chain_rust_dec,
        problems,
    })
}

/// One table row of `modes`: the comparison plus what libopus does when
/// nothing is forced.
fn modes_one(sc: &Scenario, clip: &Clip) -> Result<(Comparison, String), String> {
    let comparison = compare(sc, clip)?;
    let c_free_packets = c_encode_all(sc, clip, false)?;
    Ok((comparison, histogram_label(&toc_histogram(&c_free_packets))))
}

fn cmd_modes(opts: &Opts) -> Result<i32, String> {
    let filter = opts.get("--filter");
    let bench_only = opts.flag("--bench-set");
    let mut signals = Signals::load()?;
    let chain = |value: Option<(f64, isize)>| {
        value.map_or("-".to_string(), |(snr, delay)| format!("{snr:.1} @{delay}"))
    };

    let mut text = String::new();
    let _ = writeln!(text, "# perf_harness modes\n");
    let _ = writeln!(text, "- C reference: {}, clip {GOLDEN_SECONDS} s", libopus_version());
    let _ = writeln!(
        text,
        "- `C forced`: libopus with the mode, bandwidth and channel count of the scenario forced. `C free`: libopus deciding by itself."
    );
    let _ = writeln!(
        text,
        "- `same pk`: packets that are byte-identical in the Rust and the forced C stream."
    );
    let _ = writeln!(
        text,
        "- `dec agree`: Rust decoder output against C decoder output for the same packets, in dB (`exact` = identical samples). A value in brackets is the result after shifting the Rust output by the given number of samples."
    );
    let _ = writeln!(
        text,
        "- `chain`: decoded audio against the encoder input, in dB, at the best delay in samples. `R>C` = Rust encoder and C decoder, `C>C` = C encoder and C decoder, `C>R` = C encoder and Rust decoder.\n"
    );
    let _ = writeln!(
        text,
        "| scenario | Rust stream | C forced | C free | Rust kb/s | C kb/s | same pk | dec agree (C pk) | dec agree (Rust pk) | chain R>C | chain C>C | chain C>R | ok |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    );
    print!("{text}");

    let mut problems = String::new();
    let mut count = 0usize;
    for sc in scenarios() {
        let name = sc.name();
        if (bench_only && !sc.bench) || !name_matches(&name, filter) {
            continue;
        }
        count += 1;
        let clip = signals.clip(&sc, GOLDEN_SECONDS, 0);
        let result = catch_unwind(AssertUnwindSafe(|| modes_one(&sc, &clip)))
            .unwrap_or_else(|_| Err("panic".to_string()));
        let line = match result {
            Ok((row, c_free_toc)) => {
                for problem in &row.problems {
                    let _ = write!(problems, "- {name}: {problem}");
                    if problem.contains("configuration") {
                        let _ = write!(
                            problems,
                            " (expected {}; Rust: {}; C forced: {})",
                            row.expected, row.rust_toc, row.c_toc
                        );
                    }
                    problems.push('\n');
                }
                format!(
                    "| {name} | {} | {} | {c_free_toc} | {:.1} | {:.1} | {}/{} | {} | {} | {} | {} | {} | {} |",
                    row.rust_toc,
                    row.c_toc,
                    row.rust_kbps,
                    row.c_kbps,
                    row.identical_packets,
                    clip.frames,
                    row.agree_c_packets.label(),
                    row.agree_rust_packets.label(),
                    chain(row.chain_rust_enc),
                    chain(row.chain_c),
                    chain(row.chain_rust_dec),
                    if row.problems.is_empty() { "ok" } else { "CHECK" },
                )
            }
            Err(e) => {
                let _ = writeln!(problems, "- {name}: FAILED: {e}");
                format!("| {name} | FAILED: {e} | | | | | | | | | | | CHECK |")
            }
        };
        println!("{line}");
        let _ = std::io::stdout().flush();
        let _ = writeln!(text, "{line}");
    }

    // The mode-switching streams of `golden`.
    let mut switch_table = String::new();
    for case in switch_cases() {
        if bench_only || !name_matches(&case.name, filter) {
            continue;
        }
        count += 1;
        let clip = signals.clip(&case.sc, GOLDEN_SECONDS, 0);
        let line = match guarded(|| modes_switch_row(&case, &clip)) {
            Ok((line, row_problems)) => {
                for problem in row_problems {
                    let _ = writeln!(problems, "- {}: {problem}", case.name);
                }
                line
            }
            Err(e) => {
                let _ = writeln!(problems, "- {}: FAILED: {e}", case.name);
                format!("| {} | FAILED: {e} | | | | | CHECK |", case.name)
            }
        };
        let _ = writeln!(switch_table, "{line}");
    }
    if !switch_table.is_empty() {
        let section = format!(
            "\nMode-switching streams: the bit rate changes every 200 ms and nothing is forced, except stereo for stereo input unless the name ends in `_free`. \
             Both decoders decode each stream; `rejected` counts the packets the Rust decoder refused.\n\n\
             | stream | Rust stream | C stream | rejected (C pk) | dec agree (C pk) | dec agree (Rust pk) | ok |\n|---|---|---|---:|---:|---:|---|\n{switch_table}"
        );
        print!("{section}");
        text.push_str(&section);
    }

    if count == 0 {
        return Err("no scenario matches the filter".to_string());
    }
    let tail = if problems.is_empty() {
        "\nNo scenario needs attention.\n".to_string()
    } else {
        format!("\nScenarios that need attention:\n\n{problems}")
    };
    print!("{tail}");
    text.push_str(&tail);
    if let Some(path) = opts.get("--md") {
        write_file(path, &text)?;
    }
    Ok(0)
}

/// One table row of `modes` for a mode-switching stream, and its problems.
fn modes_switch_row(case: &SwitchCase, clip: &Clip) -> Result<(String, Vec<String>), String> {
    let sc = &case.sc;
    let rust_packets = rust_encode_switching(case, clip)?;
    let c_packets = c_encode_switching(case, clip)?;
    let rust_on_c = rust_decode_all(sc, &c_packets, Loss::None)?;
    let c_on_c = c_decode_all(sc, &c_packets)?;
    let rust_on_rust = rust_decode_all(sc, &rust_packets, Loss::None)?;
    let c_on_rust = c_decode_all(sc, &rust_packets)?;
    let agree_c = Agreement::measure(&c_on_c.pcm, &rust_on_c.pcm, sc.channels, sc.dec_rate);
    let agree_rust = Agreement::measure(&c_on_rust.pcm, &rust_on_rust.pcm, sc.channels, sc.dec_rate);

    let mut problems = Vec::new();
    for (who, what, decoded) in [
        ("Rust decoder", "C packets", &rust_on_c),
        ("C decoder", "C packets", &c_on_c),
        ("Rust decoder", "Rust packets", &rust_on_rust),
        ("C decoder", "Rust packets", &c_on_rust),
    ] {
        if decoded.errors > 0 {
            problems.push(format!(
                "{who} rejected {what}: {} of {} ({})",
                decoded.errors,
                clip.frames,
                decoded.first_error.as_deref().unwrap_or("?")
            ));
        }
    }
    problems.extend(agree_c.problem("C"));
    problems.extend(agree_rust.problem("Rust"));
    let line = format!(
        "| {} | {} | {} | {}/{} | {} | {} | {} |",
        case.name,
        histogram_label(&toc_histogram(&rust_packets)),
        histogram_label(&toc_histogram(&c_packets)),
        rust_on_c.errors,
        clip.frames,
        agree_c.label(),
        agree_rust.label(),
        if problems.is_empty() { "ok" } else { "CHECK" },
    );
    Ok((line, problems))
}

// ---------------------------------------------------------------------------
// settings: what the two encoders report
// ---------------------------------------------------------------------------

/// Name of a libopus constant where there is one, the number otherwise.
fn c_constant_label(value: i32) -> String {
    match value {
        ffi::OPUS_AUTO => "auto",
        ffi::OPUS_APPLICATION_VOIP => "voip",
        ffi::OPUS_APPLICATION_AUDIO => "audio",
        ffi::OPUS_APPLICATION_RESTRICTED_LOWDELAY => "low delay",
        ffi::OPUS_BANDWIDTH_NARROWBAND => "nb",
        ffi::OPUS_BANDWIDTH_MEDIUMBAND => "mb",
        ffi::OPUS_BANDWIDTH_WIDEBAND => "wb",
        ffi::OPUS_BANDWIDTH_SUPERWIDEBAND => "swb",
        ffi::OPUS_BANDWIDTH_FULLBAND => "fb",
        ffi::OPUS_SIGNAL_VOICE => "voice",
        ffi::OPUS_SIGNAL_MUSIC => "music",
        ffi::OPUS_FRAMESIZE_ARG => "per call",
        other => return other.to_string(),
    }
    .to_string()
}

/// Encode the clip of every scenario with both libraries and print the
/// settings each encoder reports afterwards.
///
/// The C values come from the `OPUS_GET_*` requests of libopus. opus-rs has
/// no such requests: its bit rate, complexity, constant-bit-rate switch,
/// forward error correction switch and expected loss are public fields, and
/// the application, sample rate and channel count only go into the
/// constructor.
fn cmd_settings(opts: &Opts) -> Result<i32, String> {
    let filter = opts.get("--filter");
    let bench_only = opts.flag("--bench-set");
    let mut signals = Signals::load()?;

    let mut text = String::new();
    let _ = writeln!(text, "# perf_harness settings\n");
    let _ = writeln!(text, "- C reference: {}, clip {GOLDEN_SECONDS} s", libopus_version());
    let _ = writeln!(
        text,
        "- C columns: read back from libopus with `OPUS_GET_*` after the clip was encoded with the mode, bandwidth and channel count forced. `bandwidth` is that of the last frame."
    );
    let _ = writeln!(
        text,
        "- `Rust`: bit rate / complexity / rate control / forward error correction / expected loss, read from the encoder after the clip."
    );
    let _ = writeln!(
        text,
        "- `check` is `as asked` when every value is what the scenario asks for (settings that a scenario does not name must be at the libopus default).\n"
    );
    let _ = writeln!(
        text,
        "| scenario | application | Hz | forced ch | bit rate | VBR | constrained | complexity | FEC | loss % | bandwidth | max bw | signal | DTX | depth | frame | no prediction | no phase inv. | look-ahead | Rust | check |\n|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|---|---|"
    );
    print!("{text}");

    let mut problems = String::new();
    let mut count = 0usize;
    for sc in scenarios() {
        let name = sc.name();
        if (bench_only && !sc.bench) || !name_matches(&name, filter) {
            continue;
        }
        count += 1;
        let clip = signals.clip(&sc, GOLDEN_SECONDS, 0);
        let c = guarded(|| c_encode_all_with_settings(&sc, &clip, true));
        let rust = guarded(|| rust_encode_all_with_settings(&sc, &clip));
        let mut mismatches = Vec::new();
        let c_cells = match &c {
            Ok((_, c)) => {
                mismatches.extend(c.mismatches(&sc));
                format!(
                    "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                    c_constant_label(c.application),
                    c.sample_rate,
                    c_constant_label(c.force_channels),
                    c.bitrate,
                    c.vbr,
                    c.vbr_constraint,
                    c.complexity,
                    c.inband_fec,
                    c.packet_loss_perc,
                    c_constant_label(c.bandwidth),
                    c_constant_label(c.max_bandwidth),
                    c_constant_label(c.signal),
                    c.dtx,
                    c.lsb_depth,
                    c_constant_label(c.frame_duration),
                    c.prediction_disabled,
                    c.phase_inversion_disabled,
                    c.lookahead,
                )
            }
            Err(e) => {
                mismatches.push(format!("C encoder failed: {e}"));
                format!("FAILED{}", " |".repeat(17))
            }
        };
        let rust_cell = match &rust {
            Ok((_, rust)) => {
                if *rust != RustSettings::wanted(&sc) {
                    mismatches.push(format!("Rust encoder has {rust:?}"));
                }
                format!(
                    "{} / {} / {} / {} / {}",
                    rust.bitrate,
                    rust.complexity,
                    if rust.cbr { "CBR" } else { "VBR" },
                    if rust.fec { "FEC" } else { "no FEC" },
                    rust.loss_perc,
                )
            }
            Err(e) => {
                mismatches.push(format!("Rust encoder failed: {e}"));
                "FAILED".to_string()
            }
        };
        for mismatch in &mismatches {
            let _ = writeln!(problems, "- {name}: {mismatch}");
        }
        let line = format!(
            "| {name} | {c_cells} | {rust_cell} | {} |",
            if mismatches.is_empty() { "as asked" } else { "DIFFERS" },
        );
        println!("{line}");
        let _ = writeln!(text, "{line}");
    }
    if count == 0 {
        return Err("no scenario matches the filter".to_string());
    }
    let tail = if problems.is_empty() {
        format!("\nAll {count} scenarios: both encoders have the settings the scenario asks for.\n")
    } else {
        format!("\nSettings that differ:\n\n{problems}")
    };
    print!("{tail}");
    text.push_str(&tail);
    if let Some(path) = opts.get("--md") {
        write_file(path, &text)?;
    }
    Ok(i32::from(!problems.is_empty()))
}

// ---------------------------------------------------------------------------
// signals: what the test signals contain
// ---------------------------------------------------------------------------

/// In-place radix-2 fast Fourier transform. `re.len()` must be a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -std::f64::consts::TAU / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (sin, cos) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cos - im[b] * sin, re[b] * sin + im[b] * cos);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// Share of the signal power in each 4 kHz wide band, in dB relative to the
/// total power. The last entry is everything from 20 kHz up.
fn band_levels_db(samples: &[f32], rate: u32) -> [f64; 6] {
    const N: usize = 4096;
    let mut power = [0.0f64; 6];
    for block in samples.as_chunks::<N>().0.iter().step_by(2) {
        let mut re: Vec<f64> = block
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let hann = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos();
                v as f64 * hann
            })
            .collect();
        let mut im = vec![0.0; N];
        fft(&mut re, &mut im);
        for bin in 1..N / 2 {
            let frequency = bin as f64 * rate as f64 / N as f64;
            let band = ((frequency / 4000.0) as usize).min(5);
            power[band] += re[bin] * re[bin] + im[bin] * im[bin];
        }
    }
    let total: f64 = power.iter().sum();
    power.map(|p| 10.0 * (p / total.max(1e-300)).max(1e-30).log10())
}

fn write_wav(path: &str, rate: u32, channels: usize, samples: &[i16]) -> Result<(), String> {
    let data_len = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(channels as u16).to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(&((channels * 2) as u16).to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")))
        .and_then(|()| std::fs::write(path, bytes))
        .map_err(|e| format!("{path}: {e}"))
}

/// Print level, stereo difference and spectrum of every test signal that a
/// scenario uses, and optionally write them as WAV files.
fn cmd_signals(opts: &Opts) -> Result<i32, String> {
    let seconds: f64 = opts.number("--seconds", GOLDEN_SECONDS)?;
    let dump = opts.get("--dump");
    let mut signals = Signals::load()?;
    let mut seen = Vec::new();
    println!(
        "| signal | rate | ch | peak dBFS | RMS dBFS | frames under -50 dBFS | left/right correlation | 0-4 kHz | 4-8 | 8-12 | 12-16 | 16-20 | 20+ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    for sc in scenarios() {
        let key = (sc.signal, sc.rate, sc.channels, sc.pauses);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        // 20 ms frames, whatever the scenario uses.
        let clip = signals.clip(&sc.clone().frame(20_000), seconds, 0);
        let peak = clip.pcm.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
        let power = clip.pcm.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / clip.pcm.len() as f64;
        let quiet_frames = clip
            .pcm
            .chunks_exact(clip.step)
            .filter(|frame| {
                let p = frame.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / frame.len() as f64;
                p < 1e-5
            })
            .count();
        let first: Vec<f32> = clip.pcm.iter().step_by(clip.channels).copied().collect();
        let correlation = if clip.channels == 2 {
            let (mut ll, mut rr, mut lr) = (0.0f64, 0.0f64, 0.0f64);
            for &[l, r] in clip.pcm.as_chunks::<2>().0 {
                let (l, r) = (l as f64, r as f64);
                ll += l * l;
                rr += r * r;
                lr += l * r;
            }
            format!("{:.3}", lr / (ll * rr).sqrt().max(1e-30))
        } else {
            "-".to_string()
        };
        let bands = band_levels_db(&first, sc.rate);
        let nyquist_band = ((sc.rate / 2 - 1) / 4000).min(5) as usize;
        let band_text: Vec<String> = bands
            .iter()
            .enumerate()
            .map(|(i, db)| if i <= nyquist_band { format!("{db:.1}") } else { "-".to_string() })
            .collect();
        let signal = match (sc.signal, sc.pauses) {
            (Signal::Speech, false) => "speech",
            (Signal::Speech, true) => "speech_pauses",
            (Signal::Music, false) => "music",
            (Signal::Music, true) => "music_pauses",
        };
        println!(
            "| {signal} | {} | {} | {:.2} | {:.2} | {}/{} | {correlation} | {} |",
            sc.rate,
            sc.channels,
            20.0 * peak.max(1e-30).log10(),
            10.0 * power.max(1e-30).log10(),
            quiet_frames,
            clip.frames,
            band_text.join(" | "),
        );
        if let Some(dir) = dump {
            let path = format!("{dir}/{signal}_{}k_{}ch.wav", sc.rate / 1000, sc.channels);
            write_wav(&path, sc.rate, sc.channels, &clip.pcm_i16)?;
        }
    }
    println!("\nBand columns: share of the power of the first channel in that band, in dB relative to the whole signal.");
    Ok(0)
}

// ---------------------------------------------------------------------------
// resampler: check the resampler behind the speech signals
// ---------------------------------------------------------------------------

/// The pass band of `resample` ends at this share of the lower of the two
/// Nyquist frequencies. Above it the filter rolls off, and from the Nyquist
/// frequency on it blocks.
const RESAMPLER_PASS_BAND: f64 = 0.82;
/// Limit in dB, relative to the input tone, for everything in the output
/// that is not the tone itself.
const RESAMPLER_SPURIOUS_LIMIT_DB: f64 = -60.0;

/// Resample sine tones from 16 kHz to every rate the speech signals use and
/// measure what comes out.
///
/// The output is split, by a least-squares fit, into the tone at its own
/// frequency and the rest. The rest holds every image and alias component
/// and any other distortion. All levels are in dB relative to the input
/// tone, and the output is measured before it is rounded to 16 bits.
///
/// A row passes when the rest is below -60 dB and, for a tone in the pass
/// band, the tone comes out within 0.1 dB and without a time shift. A tone
/// above the output Nyquist frequency must disappear, so for such a row the
/// whole output is "the rest".
fn cmd_resampler() -> Result<i32, String> {
    use std::f64::consts::TAU;
    const INPUT_RATE: f64 = 16000.0;
    const INPUT_LEN: usize = 16000;
    const PHASE: f64 = 0.4;
    const TONES: [f64; 14] = [
        100.0, 1000.0, 3000.0, 3250.0, 3900.0, 4100.0, 4900.0, 5900.0, 6100.0, 6500.0, 7000.0,
        7500.0, 7900.0, 7990.0,
    ];
    let db = |amplitude: f64| 20.0 * amplitude.max(1e-15).log10();

    println!(
        "Resampler check: sine tones at 16 kHz, amplitude 1, resampled to each output rate.\n\
         Levels in dB relative to the input tone. `rest` = the output minus the tone at its own frequency\n\
         (all image and alias components and other distortion). Limit for `rest`: {RESAMPLER_SPURIOUS_LIMIT_DB} dB.\n"
    );
    println!(
        "| output Hz | tone Hz | tone is in the | tone level dB | time shift (samples) | rest dB | result |\n|---:|---:|---|---:|---:|---:|---|"
    );
    let mut failures = 0usize;
    for (output_rate, up, down) in [(48000.0, 3, 1), (24000.0, 3, 2), (12000.0, 3, 4), (8000.0, 1, 2)] {
        let lower_nyquist = 0.5 * f64::min(INPUT_RATE, output_rate);
        let mut worst_rest = f64::NEG_INFINITY;
        for tone in TONES {
            let input: Vec<f64> = (0..INPUT_LEN)
                .map(|n| (TAU * tone * n as f64 / INPUT_RATE + PHASE).sin())
                .collect();
            let output = resample(&input, up, down);
            // Leave out 0.1 s at each end, where the filter runs over the
            // edge of the input.
            let margin = (0.1 * output_rate) as usize;
            let middle = &output[margin..output.len() - margin];
            let omega = TAU * tone / output_rate;
            let aliased = tone >= 0.5 * output_rate;

            // Least-squares fit of a sin(omega m + PHASE) + b cos(omega m + PHASE).
            // A perfect resampler gives a = 1 and b = 0.
            let (mut ss, mut sc, mut cc, mut ys, mut yc, mut yy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for (i, &y) in middle.iter().enumerate() {
                let (sin, cos) = (omega * (i + margin) as f64 + PHASE).sin_cos();
                ss += sin * sin;
                sc += sin * cos;
                cc += cos * cos;
                ys += y * sin;
                yc += y * cos;
                yy += y * y;
            }
            let count = middle.len() as f64;
            let (tone_level, shift, rest_power) = if aliased {
                (None, None, yy / count)
            } else {
                let determinant = ss * cc - sc * sc;
                let a = (ys * cc - yc * sc) / determinant;
                let b = (yc * ss - ys * sc) / determinant;
                let fitted_power = (a * ys + b * yc) / count;
                // a sin(x) + b cos(x) = A sin(x + atan2(b, a)): a positive
                // angle means the output is early.
                let shift = -b.atan2(a) / omega;
                (Some(a.hypot(b)), Some(shift), (yy / count - fitted_power).max(0.0))
            };
            // The power of a tone of amplitude 1 is 1/2.
            let rest_db = 10.0 * (2.0 * rest_power).max(1e-30).log10();
            worst_rest = worst_rest.max(rest_db);

            let band = if aliased {
                "stop band (above the output Nyquist)"
            } else if tone >= lower_nyquist {
                "stop band"
            } else if tone <= RESAMPLER_PASS_BAND * lower_nyquist {
                "pass band"
            } else {
                "roll-off"
            };
            let mut ok = rest_db < RESAMPLER_SPURIOUS_LIMIT_DB;
            if let (Some(level), Some(shift)) = (tone_level, shift) {
                match band {
                    "pass band" => ok &= db(level).abs() < 0.1 && shift.abs() < 0.01,
                    "stop band" => ok &= db(level) < RESAMPLER_SPURIOUS_LIMIT_DB,
                    _ => {}
                }
            }
            failures += usize::from(!ok);
            // The shift of a tone that is almost gone means nothing.
            let audible = tone_level.is_some_and(|level| db(level) > -40.0);
            println!(
                "| {output_rate:.0} | {tone:.0} | {band} | {} | {} | {} | {} |",
                tone_level.map_or("-".to_string(), |level| format!("{:.2}", db(level))),
                shift
                    .filter(|_| audible)
                    .map_or("-".to_string(), |shift| format!("{shift:+.4}")),
                // Below about -150 dB the number is rounding error of the fit.
                if rest_db < -150.0 { "below -150".to_string() } else { format!("{rest_db:.1}") },
                if ok { "ok" } else { "FAIL" },
            );
        }
        println!("| {output_rate:.0} | all | | | | worst {worst_rest:.1} | |");
    }
    println!(
        "\n{}",
        if failures == 0 {
            "All rows pass.".to_string()
        } else {
            format!("{failures} rows FAIL.")
        }
    );
    Ok(i32::from(failures > 0))
}

// ---------------------------------------------------------------------------
// list and the command line
// ---------------------------------------------------------------------------

fn cmd_list() -> Result<i32, String> {
    let all = scenarios();
    let mut names: Vec<String> = all.iter().map(Scenario::name).collect();
    names.sort();
    if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(format!("duplicate scenario name {}", pair[0]));
    }
    println!(
        "| scenario | set | rate | ch | application | kb/s | frame | complexity | rate control | FEC | signal | decoder rate | expected packets |\n|---|---|---:|---:|---|---:|---|---:|---|---|---|---:|---|"
    );
    for sc in &all {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            sc.name(),
            if sc.bench { "bench+golden" } else { "golden" },
            sc.rate,
            sc.channels,
            match sc.app {
                App::Voip => "voip",
                App::Audio => "audio",
                App::LowDelay => "restricted low delay",
            },
            sc.bitrate / 1000,
            frame_label(sc.frame_us).replace('p', "."),
            sc.complexity,
            if sc.cbr { "CBR" } else { "VBR" },
            if sc.fec {
                format!("on, loss {}%", sc.loss_perc)
            } else {
                "off".to_string()
            },
            match (sc.signal, sc.pauses) {
                (Signal::Speech, false) => "speech",
                (Signal::Speech, true) => "speech with pauses",
                (Signal::Music, false) => "music",
                (Signal::Music, true) => "music with pauses",
            },
            sc.dec_rate,
            toc_label(sc.expected_toc_key()),
        );
    }
    println!(
        "\n{} scenarios, {} of them in the bench set",
        all.len(),
        all.iter().filter(|sc| sc.bench).count()
    );
    let switching: Vec<String> = switch_cases().into_iter().map(|case| case.name).collect();
    println!(
        "\n`golden` and `modes` also cover {} mode-switching streams (the bit rate changes every 200 ms):\n{}",
        switching.len(),
        switching.join("\n")
    );
    Ok(0)
}

/// Parsed command line options.
struct Opts {
    values: BTreeMap<String, String>,
    flags: Vec<String>,
}

impl Opts {
    fn parse(args: &[String], with_value: &[&str], flags: &[&str]) -> Result<Self, String> {
        let mut opts = Self {
            values: BTreeMap::new(),
            flags: Vec::new(),
        };
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if with_value.contains(&arg.as_str()) {
                let value = iter.next().ok_or(format!("{arg} needs a value"))?;
                opts.values.insert(arg.clone(), value.clone());
            } else if flags.contains(&arg.as_str()) {
                opts.flags.push(arg.clone());
            } else {
                return Err(format!("unknown option {arg}"));
            }
        }
        Ok(opts)
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn flag(&self, key: &str) -> bool {
        self.flags.iter().any(|f| f == key)
    }

    fn number<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.get(key) {
            Some(text) => text.parse().map_err(|_| format!("{key}: '{text}' is not a valid number")),
            None => Ok(default),
        }
    }
}

const USAGE: &str = "usage: perf_harness <command> [options]

  list
      Print every scenario.

  bench [--filter SUBSTR] [--reps N] [--procs P] [--rounds R] [--seconds S]
        [--md PATH] [--json PATH] [--compare JSON] [--all]
      Time Rust encode, C encode, Rust decode and C decode per scenario.
      Defaults: 15 repetitions, 6 second clips, the bench set.
      --procs     fresh processes per scenario (default 3); the repetitions
                  are split over them, each process adds 1 warm-up
      --rounds    a scenario is measured again when other programs disturbed
                  it (median more than 3 % above minimum), at most this many
                  times in total (default 3)
      --all       time every scenario, not only the bench set
      --compare   also print the change against an earlier --json file

  run --scenario NAME --op enc|dec [--impl rust|c] [--seconds S]
      [--clip-seconds S]
      Print the process id, then loop one operation (default 10 s) so that a
      sampling profiler can attach, for example `sample <pid> 5`.

  golden --write PATH
  golden --check PATH [--filter SUBSTR]
      Write, or compare against, hashes of the codec output for every
      scenario and for the mode-switching streams. --check exits with
      status 1 when anything differs.
      --perturb N  self-test: add N/32768 to one input sample

  modes [--filter SUBSTR] [--bench-set] [--md PATH]
      Compare the Rust and C packet streams and the two decoders.

  settings [--filter SUBSTR] [--bench-set] [--md PATH]
      Encode every scenario and print the settings both encoders report
      afterwards (libopus: read back with OPUS_GET_*). Exits with status 1
      when a value is not what the scenario asks for.

  signals [--seconds S] [--dump DIR]
      Print level and spectrum of every test signal; --dump writes them as
      WAV files.

  resampler
      Check the resampler that makes the speech signals: resample sine tones
      and print the level of the tone and of everything else (image and
      alias components). Exits with status 1 when a row fails.

  spread JSON JSON...
      Compare several bench --json files: the lowest minimum per scenario,
      and how far the minimums and the medians of the files are apart.

Code placement: the speed of some functions depends on where the linker puts
them (up to 15 % on a whole scenario). Build with PERF_HARNESS_PAD=<bytes>
to move the library code (macOS only), for example:
  for pad in 0 20 40 60; do
    PERF_HARNESS_PAD=$pad cargo run --release --example perf_harness -- \
      bench --json target/perf/pad_$pad.json
  done
  cargo run --release --example perf_harness -- spread target/perf/pad_*.json

--filter takes a comma-separated list of substrings of scenario names.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let result = match command.as_str() {
        "list" => Opts::parse(rest, &[], &[]).and_then(|_| cmd_list()),
        "bench" => Opts::parse(
            rest,
            &[
                "--filter",
                "--reps",
                "--procs",
                "--rounds",
                "--seconds",
                "--md",
                "--json",
                "--compare",
            ],
            &["--all"],
        )
        .and_then(|opts| cmd_bench(&opts)),
        "bench-worker" => Opts::parse(rest, &["--scenario", "--reps", "--seconds"], &[])
            .and_then(|opts| cmd_bench_worker(&opts)),
        "run" => Opts::parse(
            rest,
            &["--scenario", "--op", "--impl", "--seconds", "--clip-seconds"],
            &[],
        )
        .and_then(|opts| cmd_run(&opts)),
        "golden" => Opts::parse(rest, &["--write", "--check", "--filter", "--perturb"], &[])
            .and_then(|opts| cmd_golden(&opts)),
        "modes" => Opts::parse(rest, &["--filter", "--md"], &["--bench-set"])
            .and_then(|opts| cmd_modes(&opts)),
        "settings" => Opts::parse(rest, &["--filter", "--md"], &["--bench-set"])
            .and_then(|opts| cmd_settings(&opts)),
        "signals" => Opts::parse(rest, &["--seconds", "--dump"], &[]).and_then(|opts| cmd_signals(&opts)),
        "resampler" => Opts::parse(rest, &[], &[]).and_then(|_| cmd_resampler()),
        "spread" => cmd_spread(rest),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(0)
        }
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(message) => {
            eprintln!("error: {message}");
            std::process::exit(2);
        }
    }
}
