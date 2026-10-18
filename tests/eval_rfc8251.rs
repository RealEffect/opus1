//! The conformance test of RFC 6716 as updated by RFC 8251, as libopus's
//! `tests/run_vectors.sh` runs it, with the ports of `opus_demo -d` and
//! `opus_compare` in `opus1::internals::eval`.
//!
//! For each of the 12 test vectors, each output rate the RFC allows
//! (48, 24, 16, 12 and 8 kHz) and mono and stereo output (120 runs):
//!
//! 1. decode `testvectorNN.bit` with a new `OpusDecoder`, as
//!    `opus_demo -d <rate> <channels> -ignore_extensions` does. The final
//!    range of the decoder must equal the one in the bitstream after every
//!    packet the loop checks.
//! 2. compare the output with `testvectorNN.dec` and `testvectorNNm.dec`
//!    (the outputs decoded without the CELT phase inversion of RFC 8251)
//!    with `opus_compare -r <rate>`. The run passes when either comparison
//!    passes.
//!
//! The test prints the quality `Q` of every run and the mean quality per
//! configuration as `run_vectors.sh` computes it.
//!
//! The vectors are not in the repository. The test looks for
//! `testvector01.bit` in `OPUS1_TESTVECTORS`, in `$OPUS1_CORPUS/rfc8251`
//! (`OPUS1_CORPUS` defaults to `<repo>/corpus`), and in their direct
//! subdirectories (the archive extracts to `opus_newvectors/`). Without
//! them it prints a message and passes. Once the directory is found, a
//! missing file fails the test.
//!
//! Run it with `cargo test --release --test eval_rfc8251 -- --nocapture`.

#![cfg(feature = "std")]

mod eval_common;

use eval_common::corpus_dir;
use eval_common::libopus_demo::LibopusDecoder;
use opus1::OpusDecoder;
use opus1::internals::eval::compare::{Comparison, Reference, TestSignal, opus_compare_many};
use opus1::internals::eval::demo::{DemoDecoder, DemoOutput, decode_bitstream, parse_bitstream};
use std::path::{Path, PathBuf};
use std::time::Instant;

const RATES: [u32; 5] = [48000, 24000, 16000, 12000, 8000];
const VECTORS: usize = 12;

/// The directory that holds `testvector01.bit`, if any.
fn find_vectors() -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os("OPUS1_TESTVECTORS").filter(|d| !d.is_empty()) {
        roots.push(PathBuf::from(dir));
    }
    roots.push(corpus_dir().join("rfc8251"));
    for root in roots {
        if root.join("testvector01.bit").is_file() {
            return Some(root);
        }
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut subdirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        subdirs.sort();
        if let Some(dir) = subdirs.into_iter().find(|d| d.join("testvector01.bit").is_file()) {
            return Some(dir);
        }
    }
    None
}

/// 16-bit little-endian samples of a file.
fn read_s16(path: &Path) -> Result<Vec<i16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect())
}

/// The two comparisons of one run: against `.dec` and against `m.dec`.
struct Run {
    channels: usize,
    rate: u32,
    output: DemoOutput,
    dec: Result<Comparison, String>,
    mdec: Result<Comparison, String>,
}

impl Run {
    fn passes(&self) -> bool {
        self.output.range_mismatch.is_none()
            && (self.dec.as_ref().is_ok_and(|c| c.passes()) || self.mdec.as_ref().is_ok_and(|c| c.passes()))
    }
}

struct Vector {
    name: String,
    seconds: f64,
    packets: usize,
    runs: Vec<Run>,
}

fn decode<D: DemoDecoder>(decoder: &mut D, bits: &[u8]) -> DemoOutput {
    let parsed = parse_bitstream(bits);
    assert_eq!(parsed.error, None, "the bitstream is cut or invalid");
    decode_bitstream(decoder, &parsed.packets)
}

fn run_vector(dir: &Path, nn: usize) -> Result<Vector, String> {
    let name = format!("testvector{nn:02}");
    let bits = std::fs::read(dir.join(format!("{name}.bit"))).map_err(|e| format!("{name}.bit: {e}"))?;
    let dec = read_s16(&dir.join(format!("{name}.dec")))?;
    let mdec = read_s16(&dir.join(format!("{name}m.dec")))?;
    let parsed = parse_bitstream(&bits);
    if let Some(e) = parsed.error {
        return Err(format!("{name}.bit: {e}"));
    }
    let mut runs = Vec::new();
    for channels in [1, 2] {
        let outputs: Vec<DemoOutput> = RATES
            .iter()
            .map(|&rate| decode_bitstream(&mut OpusDecoder::new(rate as i32, channels).unwrap(), &parsed.packets))
            .collect();
        let tests: Vec<TestSignal<'_>> =
            outputs.iter().zip(RATES).map(|(o, rate)| TestSignal { samples: &o.pcm, rate }).collect();
        let mut compared = opus_compare_many(&[Reference::Stereo(&dec), Reference::Stereo(&mdec)], &tests, channels);
        let mdec_results = compared.pop().unwrap();
        let dec_results = compared.pop().unwrap();
        for (((output, rate), d), m) in outputs.into_iter().zip(RATES).zip(dec_results).zip(mdec_results) {
            runs.push(Run {
                channels,
                rate,
                output,
                dec: d.map_err(|e| e.to_string()),
                mdec: m.map_err(|e| e.to_string()),
            });
        }
    }
    Ok(Vector { name, seconds: dec.len() as f64 / 2.0 / 48000.0, packets: parsed.packets.len(), runs })
}

/// `Q` as the table shows it.
fn cell(c: &Result<Comparison, String>) -> String {
    match c {
        Ok(c) if c.passes() => format!("{:6.1}", c.quality),
        Ok(c) => format!("{:>6}", format!("F{:.2}", c.err)),
        Err(_) => "   ERR".to_string(),
    }
}

#[test]
fn rfc8251_test_vectors() {
    let Some(dir) = find_vectors() else {
        println!("skipped: RFC 8251 test vectors not found (set OPUS1_TESTVECTORS or OPUS1_CORPUS)");
        return;
    };
    let start = Instant::now();
    let results: Vec<Result<Vector, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (1..=VECTORS).map(|nn| {
            let dir = &dir;
            s.spawn(move || run_vector(dir, nn))
        }).collect();
        handles.into_iter().map(|h| h.join().expect("vector thread")).collect()
    });
    let elapsed = start.elapsed();

    let mut failures = Vec::new();
    let vectors: Vec<Vector> = results
        .into_iter()
        .filter_map(|r| r.map_err(|e| failures.push(e)).ok())
        .collect();
    println!("RFC 8251 test vectors in {}", dir.display());
    for (label, pick) in [("testvectorNN.dec", 0), ("testvectorNNm.dec", 1)] {
        println!("\nQ against {label} (F<err>: fails, with its internal weighted error):");
        print!("{:14}", "vector");
        for ch in ["mono", "stereo"] {
            for rate in RATES {
                print!(" {:>6}", format!("{}{}", rate / 1000, if ch == "mono" { "k m" } else { "k s" }));
            }
        }
        println!();
        for v in &vectors {
            print!("{:14}", v.name);
            for run in &v.runs {
                print!(" {}", cell(if pick == 0 { &run.dec } else { &run.mdec }));
            }
            println!();
        }
    }
    println!();
    for v in &vectors {
        let checks: usize = v.runs.iter().map(|r| r.output.range_checks).sum();
        let fec: usize = v.runs.iter().map(|r| r.output.fec_calls).sum();
        let lost: usize = v.runs.iter().map(|r| r.output.lost).sum::<usize>() / v.runs.len();
        println!(
            "{}: {:.1} s, {} packets ({lost} lost), final range checked {checks} times over 10 runs, {fec} FEC calls",
            v.name, v.seconds, v.packets
        );
        for run in &v.runs {
            if let Some(m) = run.output.range_mismatch {
                failures.push(format!(
                    "{} {} ch {} Hz: range mismatch at packet {}: encoder {:08x}, decoder {:08x}",
                    v.name, run.channels, run.rate, m.packet, m.encoder, m.decoder
                ));
            }
            if run.output.decode_errors != 0 {
                failures.push(format!("{} {} ch {} Hz: {} decode errors", v.name, run.channels, run.rate, run.output.decode_errors));
            }
            if !run.passes() {
                failures.push(format!(
                    "{} {} ch {} Hz: fails (.dec {}, m.dec {})",
                    v.name, run.channels, run.rate, cell(&run.dec).trim(), cell(&run.mdec).trim()
                ));
            }
        }
    }

    // The mean quality of run_vectors.sh: per reference set, the mean of
    // the printed (one decimal) values when all 12 vectors pass against that
    // set, else 0; the larger of the two.
    println!("\nmean quality per configuration (as run_vectors.sh):");
    for channels in [1, 2] {
        for rate in RATES {
            let mean = |pick: usize| -> f64 {
                let qs: Vec<f64> = vectors
                    .iter()
                    .filter_map(|v| v.runs.iter().find(|r| r.channels == channels && r.rate == rate))
                    .filter_map(|r| match if pick == 0 { &r.dec } else { &r.mdec } {
                        Ok(c) if c.passes() => Some(format!("{:.1}", c.quality).parse::<f64>().unwrap()),
                        _ => None,
                    })
                    .collect();
                if qs.len() == VECTORS { qs.iter().sum::<f64>() / VECTORS as f64 } else { 0.0 }
            };
            let (a, b) = (mean(0), mean(1));
            println!(
                "  {} {:5} Hz: {:.4} % (.dec {:.4}, m.dec {:.4})",
                if channels == 1 { "mono  " } else { "stereo" },
                rate,
                a.max(b),
                a,
                b
            );
        }
    }
    println!("\n{} runs in {:.1} s", vectors.len() * 10, elapsed.as_secs_f64());
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
    assert_eq!(vectors.len(), VECTORS);
}

/// libopus decodes the vectors through the same loop; its output must be
/// opus1's, bit for bit.
#[test]
fn rfc8251_vectors_decode_as_libopus() {
    let Some(dir) = find_vectors() else {
        println!("skipped: RFC 8251 test vectors not found (set OPUS1_TESTVECTORS or OPUS1_CORPUS)");
        return;
    };
    let results: Vec<(String, Vec<String>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (1..=VECTORS)
            .map(|nn| {
                let dir = &dir;
                s.spawn(move || {
                    let name = format!("testvector{nn:02}");
                    let bits = std::fs::read(dir.join(format!("{name}.bit"))).expect("bitstream");
                    let mut differences = Vec::new();
                    for channels in [1, 2] {
                        for rate in RATES {
                            let a = decode(&mut OpusDecoder::new(rate as i32, channels).unwrap(), &bits);
                            let b = decode(&mut LibopusDecoder::new(rate, channels), &bits);
                            assert_eq!(a.pcm.len(), b.pcm.len(), "{name} {channels} ch {rate} Hz: lengths");
                            let differ = a.pcm.iter().zip(&b.pcm).filter(|(x, y)| x != y).count();
                            if differ != 0 || a.range_mismatch != b.range_mismatch {
                                let (mut sig, mut err) = (0f64, 0f64);
                                for (&x, &y) in b.pcm.iter().zip(&a.pcm) {
                                    sig += f64::from(x).powi(2);
                                    err += (f64::from(x) - f64::from(y)).powi(2);
                                }
                                differences.push(format!(
                                    "{channels} ch {rate} Hz: {differ} of {} samples differ, SNR {:.1} dB",
                                    a.pcm.len(),
                                    10.0 * (sig / err.max(1e-30)).log10()
                                ));
                            }
                        }
                    }
                    (name, differences)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("vector thread")).collect()
    });
    let mut identical = 0;
    let mut report = Vec::new();
    for (name, differences) in &results {
        identical += 10 - differences.len();
        for d in differences {
            report.push(format!("{name} {d}"));
        }
    }
    println!("{identical} of {} outputs identical to libopus", results.len() * 10);
    assert!(report.is_empty(), "outputs that differ from libopus:\n{}", report.join("\n"));
}
