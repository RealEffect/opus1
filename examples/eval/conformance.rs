//! `eval conformance`: the RFC 8251 check of libopus's
//! `tests/run_vectors.sh`, with opus1's decoder, as a table. It follows
//! the steps of `tests/eval_rfc8251.rs`: decode each `testvectorNN.bit`
//! with the `opus_demo -d` loop at 48, 24, 16, 12 and 8 kHz in mono and
//! stereo, then compare with `testvectorNN.dec` and `testvectorNNm.dec`.
//! A run passes when either comparison passes. As in the test, a cut or
//! invalid bitstream, a final range mismatch or a failed decode call makes
//! the run fail.

use crate::corpus::corpus_root;
use crate::pipeline::parallel_map;
use opus1::OpusDecoder;
use opus1::internals::eval::compare::{Reference, TestSignal, opus_compare_many};
use opus1::internals::eval::demo::{decode_bitstream, parse_bitstream};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const RATES: [u32; 5] = [48000, 24000, 16000, 12000, 8000];

/// The directory that holds `testvector01.bit`: `dir` if given, else
/// `OPUS1_TESTVECTORS`, else `$OPUS1_CORPUS/rfc8251`, each also one level
/// down.
pub fn find_vectors(dir: Option<&str>) -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Some(d) = dir {
        roots.push(PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("OPUS1_TESTVECTORS").filter(|d| !d.is_empty()) {
        roots.push(PathBuf::from(d));
    }
    roots.push(corpus_root().join("rfc8251"));
    for root in roots {
        if root.join("testvector01.bit").is_file() {
            return Some(root);
        }
        let mut subdirs: Vec<PathBuf> =
            std::fs::read_dir(&root).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        subdirs.sort();
        if let Some(d) = subdirs.into_iter().find(|d| d.join("testvector01.bit").is_file()) {
            return Some(d);
        }
    }
    None
}

fn read_s16(path: &Path) -> Result<Vec<i16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b)).collect())
}

/// Q against `.dec` and `m.dec` for each rate of one vector and channel
/// count; `None` for a failed comparison (its err is given instead).
type Cell = (Result<(bool, f32, f64), String>, Result<(bool, f32, f64), String>);

fn run_vector(dir: &Path, n: usize, channels: usize) -> Result<Vec<Cell>, String> {
    let bits = std::fs::read(dir.join(format!("testvector{n:02}.bit"))).map_err(|e| e.to_string())?;
    let dec = read_s16(&dir.join(format!("testvector{n:02}.dec")))?;
    let mdec = read_s16(&dir.join(format!("testvector{n:02}m.dec")))?;
    let stream = parse_bitstream(&bits);
    if let Some(e) = stream.error {
        return Err(format!("testvector{n:02}.bit: {e}"));
    }
    let mut outputs = Vec::new();
    for rate in RATES {
        let mut decoder = OpusDecoder::new(rate as i32, channels).map_err(|e| e.to_string())?;
        let out = decode_bitstream(&mut decoder, &stream.packets);
        if let Some(m) = out.range_mismatch {
            return Err(format!("testvector{n:02} at {rate} Hz: range mismatch at packet {}", m.packet));
        }
        // run_vectors.sh does not see these (opus_demo goes on), but the
        // vectors decode without one, as tests/eval_rfc8251.rs requires.
        if out.decode_errors > 0 {
            return Err(format!("testvector{n:02} at {rate} Hz: {} decode errors", out.decode_errors));
        }
        outputs.push(out.pcm);
    }
    let tests: Vec<TestSignal<'_>> = outputs.iter().zip(RATES).map(|(pcm, rate)| TestSignal { samples: pcm, rate }).collect();
    let results = opus_compare_many(&[Reference::Stereo(&dec), Reference::Stereo(&mdec)], &tests, channels);
    let cell = |r: &Result<opus1::internals::eval::compare::Comparison, _>| {
        r.as_ref().map(|c| (c.passes(), c.quality, c.err)).map_err(|e: &opus1::internals::eval::compare::CompareError| e.to_string())
    };
    Ok((0..RATES.len()).map(|t| (cell(&results[0][t]), cell(&results[1][t]))).collect())
}

/// Run the check. Returns the table and whether every run passed.
pub fn conformance(dir: &Path, threads: usize) -> (String, bool) {
    let jobs: Vec<(usize, usize)> = (1..=12).flat_map(|n| [(n, 1), (n, 2)]).collect();
    let results = parallel_map(jobs.len(), threads, "", |j| run_vector(dir, jobs[j].0, jobs[j].1));
    let mut text = String::new();
    let mut all_pass = true;
    // Means as run_vectors.sh computes them: per reference set, the mean
    // over the 12 vectors only when all 12 passed against that set.
    let mut sums = [[[0.0f64; 5]; 2]; 2];
    let mut all = [[[true; 5]; 2]; 2];
    for (set, name) in ["", "m"].iter().enumerate() {
        let _ = writeln!(text, "Q against testvectorNN{name}.dec (F = fails; mono 48 24 16 12 8 kHz | stereo):");
        for n in 1..=12 {
            let mut line = format!("{n:02}:");
            for (ch, channels) in [1usize, 2].iter().enumerate() {
                let j = jobs.iter().position(|&x| x == (n, *channels)).unwrap();
                for r in 0..RATES.len() {
                    let cell = match &results[j] {
                        Ok(cells) => {
                            let c = if set == 0 { &cells[r].0 } else { &cells[r].1 };
                            match c {
                                Ok((true, q, _)) => {
                                    // run_vectors.sh averages the printed
                                    // values ("%.1f").
                                    sums[set][ch][r] += format!("{:.1}", *q as f64).parse::<f64>().unwrap_or(0.0);
                                    format!("{q:6.1}")
                                }
                                Ok((false, _, err)) => {
                                    all[set][ch][r] = false;
                                    format!("F{err:5.2}")
                                }
                                Err(_) => {
                                    all[set][ch][r] = false;
                                    "   err".into()
                                }
                            }
                        }
                        Err(_) => {
                            all[set][ch][r] = false;
                            "   err".into()
                        }
                    };
                    line += &format!(" {cell}");
                }
                if ch == 0 {
                    line += " |";
                }
            }
            let _ = writeln!(text, "{line}");
        }
    }
    for (j, r) in results.iter().enumerate() {
        match r {
            Err(e) => {
                all_pass = false;
                let _ = writeln!(text, "ERROR {e}");
            }
            Ok(cells) => {
                for (k, (a, b)) in cells.iter().enumerate() {
                    let pass = matches!(a, Ok((true, _, _))) || matches!(b, Ok((true, _, _)));
                    if !pass {
                        all_pass = false;
                        let _ = writeln!(text, "FAIL testvector{:02} {} ch at {} Hz", jobs[j].0, jobs[j].1, RATES[k]);
                    }
                }
            }
        }
    }
    for (ch, name) in ["mono", "stereo"].iter().enumerate() {
        let means: Vec<String> = (0..RATES.len())
            .map(|r| {
                let m = (0..2).map(|s| if all[s][ch][r] { sums[s][ch][r] / 12.0 } else { 0.0 }).fold(0.0, f64::max);
                format!("{m:.4}")
            })
            .collect();
        let _ = writeln!(text, "mean Q {name} (48 24 16 12 8 kHz): {}", means.join(" "));
    }
    let _ = writeln!(text, "{}", if all_pass { "all 120 runs pass" } else { "some runs FAIL" });
    (text, all_pass)
}
