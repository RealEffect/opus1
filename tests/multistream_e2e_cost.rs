//! The time one multistream encode call takes, against C libopus under the
//! same settings: 5.1 and 7.1 at 48 kHz with 20 ms frames, with surround
//! masking, the LFE tuning and the forced modes all active.
//!
//! Both encoders code the same 2 seconds. A round times opus1 and then
//! libopus, or the other way round in every second round, so that a change
//! in the load of the machine hits both alike. The smallest time over all
//! rounds counts: it is the run that was disturbed least.
//!
//! Measured on an Apple M4 Pro (aarch64), libopus 1.6.1 from Homebrew,
//! variable bitrate, microseconds per frame, opus1 / libopus:
//!
//! - 5.1 at 256 kb/s, complexity 9: 124 / 272, ratio 0.46;
//! - 5.1 at 96 kb/s, complexity 9: 92 / 241, ratio 0.38;
//! - 7.1 at 320 kb/s, complexity 9: 157 / 347, ratio 0.45;
//! - 7.1 at 128 kb/s, complexity 9: 122 / 307, ratio 0.40;
//! - 5.1 at 256 kb/s, complexity 6: 125 / 230, ratio 0.54;
//! - 7.1 at 320 kb/s, complexity 6: 159 / 297, ratio 0.54.
//!
//! A 20 ms frame of 7.1 thus takes 0.8 % of real time. The two encoders do
//! not do the same work. From complexity 7 on libopus runs its analysis of
//! speech against music in every stream, which opus1 does not have; the
//! rows at complexity 6 are without it. The single-stream encoder of
//! opus1 also leaves out other steps of libopus (see the reports on the
//! CELT encoder). So the numbers say what a call costs, not that one
//! implementation of the same steps is faster.
//!
//! The limit is what the name of the test says: opus1 must not take
//! longer than libopus. Measured, it takes 0.38 to 0.54 times as long, so
//! the limit is reached when opus1 gets about twice as slow. Both
//! encoders are timed in turns and the fastest round counts, so load on a
//! shared machine moves the ratio little. The limit applies to optimized
//! builds only.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_e2e_common;
mod multistream_enc_common;

use ms_support::*;
use multistream_e2e_common::*;
use multistream_enc_common::*;
use opus1::Application;
use std::hint::black_box;
use std::time::{Duration, Instant};

const FS: i32 = 48_000;
/// 20 ms.
const FRAME: usize = 960;
/// Frames per run: 2 seconds.
const FRAMES: usize = 100;
const ROUNDS: usize = 12;

/// Codes the signal once with a new opus1 encoder. Returns the time of
/// the encode calls and the bytes written.
fn time_rust(
    case: &Case,
    bitrate: i32,
    complexity: i32,
    pcm: &[f32],
    out: &mut [u8],
) -> (Duration, usize) {
    let mut enc = rust_encoder(case, FS, Application::Audio);
    enc.set_bitrate(rust_bitrate(bitrate));
    enc.set_complexity(complexity as u8).unwrap();
    let mut bytes = 0;
    let start = Instant::now();
    for frame in pcm.chunks_exact(FRAME * case.channels) {
        bytes += enc.encode(black_box(frame), FRAME, out).unwrap();
    }
    (start.elapsed(), black_box(bytes))
}

/// The same with a new C encoder.
fn time_c(
    case: &Case,
    bitrate: i32,
    complexity: i32,
    pcm: &[f32],
    out: &mut [u8],
) -> (Duration, usize) {
    let mut enc = c_encoder(case, FS, Application::Audio);
    enc.set_bitrate(bitrate);
    enc.set_complexity(complexity);
    let mut bytes = 0;
    let start = Instant::now();
    for frame in pcm.chunks_exact(FRAME * case.channels) {
        // The wrapper allocates the packet it returns. That is part of the
        // wrapper, not of libopus, and takes well under a microsecond.
        bytes += enc.encode_float(black_box(frame), FRAME, out.len()).unwrap().len();
    }
    (start.elapsed(), black_box(bytes))
}

/// Microseconds per frame of the fastest round: opus1, libopus.
fn measure_cost(case: &Case, bitrate: i32, complexity: i32) -> (f64, f64) {
    let pcm = program(FS, case, FRAMES * FRAME);
    let mut out = vec![0u8; max_packet_bytes(8)];
    let (mut rust, mut c) = (Duration::MAX, Duration::MAX);
    let (mut rust_bytes, mut c_bytes) = (0, 0);
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            let r = time_rust(case, bitrate, complexity, &pcm, &mut out);
            let l = time_c(case, bitrate, complexity, &pcm, &mut out);
            (rust, c) = (rust.min(r.0), c.min(l.0));
            (rust_bytes, c_bytes) = (r.1, l.1);
        } else {
            let l = time_c(case, bitrate, complexity, &pcm, &mut out);
            let r = time_rust(case, bitrate, complexity, &pcm, &mut out);
            (rust, c) = (rust.min(r.0), c.min(l.0));
            (rust_bytes, c_bytes) = (r.1, l.1);
        }
    }
    // Both did the work that was asked for: within 6 % of the request.
    let request = bitrate as f64 * FRAMES as f64 / 50.0 / 8.0;
    for (who, bytes) in [("opus1", rust_bytes), ("libopus", c_bytes)] {
        assert!(
            (bytes as f64 / request - 1.0).abs() < 0.06,
            "{}, {bitrate} b/s: {who} wrote {bytes} bytes, the request is {request:.0}",
            case.name
        );
    }
    let per_frame = |d: Duration| d.as_secs_f64() * 1e6 / FRAMES as f64;
    (per_frame(rust), per_frame(c))
}

#[test]
fn one_encode_call_does_not_cost_more_than_in_libopus() {
    // Layout, bitrate, complexity. From complexity 7 on libopus runs its
    // analysis of speech against music in every stream.
    let cases = [
        (Case::family("5.1", 6, 1), 256_000, 9),
        (Case::family("5.1", 6, 1), 96_000, 9),
        (Case::family("7.1", 8, 1), 320_000, 9),
        (Case::family("7.1", 8, 1), 128_000, 9),
        (Case::family("5.1", 6, 1), 256_000, 6),
        (Case::family("7.1", 8, 1), 320_000, 6),
    ];
    for (case, bitrate, complexity) in cases {
        let (rust, c) = measure_cost(&case, bitrate, complexity);
        let label = format!("{} at {bitrate} b/s, complexity {complexity}", case.name);
        println!(
            "{label}: opus1 {rust:.0} us per frame, libopus {c:.0} us, ratio {:.2}; {:.2} % of \
             real time",
            rust / c,
            rust / 20_000.0 * 100.0
        );
        if cfg!(debug_assertions) {
            continue;
        }
        assert!(
            rust < c,
            "{label}: opus1 {rust:.0} us per frame, libopus {c:.0} us"
        );
    }
}
