//! Timing of the range decoder set-up. Not a correctness test: it prints
//! numbers and is ignored by default.
//!
//! ```text
//! cargo test --release --test perfk_rc_timing -- --ignored --nocapture
//! ```
//!
//! `RangeCoder::new_decoder` copies the packet into the coder's own buffer.
//! A decoder that borrowed the packet would save that copy. The test
//! measures what the copy costs: the time of `new_decoder` for packets of
//! several lengths, against the time for an empty packet (same code, no
//! bytes to copy) and against a plain copy of the same bytes.

use opus1::internals::range_coder::RangeCoder;
use std::hint::black_box;
use std::time::Instant;

/// Lowest time per call in nanoseconds over `rounds` rounds of `calls`.
fn time_ns(rounds: usize, calls: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..calls {
            f();
        }
        best = best.min(start.elapsed().as_nanos() as f64 / calls as f64);
    }
    best
}

#[test]
#[ignore = "timing only; run with --ignored --nocapture"]
fn packet_copy_of_the_range_decoder() {
    let packet: Vec<u8> = (0..1275u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
    let mut sink = [0u8; 1275];
    println!("| packet bytes | new_decoder + 1 bit, ns | minus empty packet, ns | plain copy, ns |");
    println!("|---:|---:|---:|---:|");
    let mut empty_ns = 0.0;
    for len in [0usize, 8, 40, 80, 160, 320, 640, 1275] {
        let data = &packet[..len];
        let decoder_ns = time_ns(9, 200_000, || {
            let mut rc = RangeCoder::new_decoder(black_box(data));
            black_box(rc.decode_bit_logp(1));
        });
        let copy_ns = time_ns(9, 200_000, || {
            sink[..len].copy_from_slice(black_box(data));
            black_box(&mut sink);
        });
        if len == 0 {
            empty_ns = decoder_ns;
        }
        println!(
            "| {len} | {decoder_ns:.1} | {:.1} | {copy_ns:.1} |",
            decoder_ns - empty_ns
        );
    }
}
