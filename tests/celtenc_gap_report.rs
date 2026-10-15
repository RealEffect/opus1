//! Report: the opus-rs CELT encoder against the CELT encoder of C libopus.
//!
//! Both encoders get the same samples and settings (`c_celt::Stream`), and
//! C libopus decodes both streams. The table shows the signal-to-noise
//! ratio of each and how many packets are the same byte for byte.
//!
//! Run it with
//! `cargo test --release --test celtenc_gap_report -- --ignored --nocapture`.
//! `CELTENC_REPORT` narrows the table: a list of words separated by commas,
//! and a row is printed when its label contains all of them, for example
//! `CELTENC_REPORT=stereo,48k,c5`.

mod celtenc_common;

use celtenc_common::c_celt::Stream;
use celtenc_common::*;

/// The test signals for `channels` channels at `fs` Hz, interleaved.
pub fn signals(fs: usize, channels: usize) -> Vec<(&'static str, Vec<f32>)> {
    let seconds = 6;
    let s = speech(fs, seconds);
    let m0 = music(fs, seconds, 0);
    if channels == 1 {
        return vec![("speech", s), ("music", m0)];
    }
    // The same talker in both channels, the right one quieter and later.
    let lag = 12 * fs / 48_000;
    let late: Vec<f32> = (0..s.len())
        .map(|i| if i >= lag { 0.6 * s[i - lag] } else { 0.0 })
        .collect();
    let m1 = music(fs, seconds, 1);
    vec![
        ("speech", stereo_of(&s, &late, 0.0)),
        ("music", stereo_of(&m0, &m1, 0.3)),
    ]
}

#[test]
#[ignore = "report, prints a table"]
fn celt_encoder_against_libopus() {
    let filter = std::env::var("CELTENC_REPORT").unwrap_or_default();
    let words: Vec<&str> = filter.split(',').filter(|w| !w.is_empty()).collect();
    println!(
        "{:<44} {:>7} {:>7} {:>6}  {:>9} {:>6}",
        "stream", "rs dB", "C dB", "diff", "same", "first"
    );
    for channels in [1usize, 2] {
        for fs in [48_000usize, 16_000] {
            let sigs = signals(fs, channels);
            for frame_ms_x2 in [5usize, 10, 20, 40] {
                for per_channel in [24_000i32, 32_000, 48_000, 64_000, 96_000] {
                    for complexity in [4, 5, 9] {
                        for vbr in [false, true] {
                            for (name, pcm) in &sigs {
                                let label = format!(
                                    "{} {} {}k {}ms {}kb c{} {}",
                                    if channels == 1 { "mono" } else { "stereo" },
                                    name,
                                    fs / 1000,
                                    frame_ms_x2 as f32 / 2.0,
                                    per_channel / 1000,
                                    complexity,
                                    if vbr { "vbr" } else { "cbr" },
                                );
                                if !words.iter().all(|w| label.contains(w)) {
                                    continue;
                                }
                                let mut st = Stream::new(
                                    fs,
                                    channels,
                                    frame_ms_x2,
                                    per_channel * channels as i32,
                                );
                                st.complexity = complexity;
                                st.vbr = vbr;
                                let c = st.enc_c(pcm);
                                let rs = st.enc_rs(pcm);
                                let (snr_rs, snr_c) = (st.snr_db(pcm, &rs), st.snr_db(pcm, &c));
                                println!(
                                    "{:<44} {:>7.2} {:>7.2} {:>+6.2}  {:>4}/{:<4} {:>6}",
                                    label,
                                    snr_rs,
                                    snr_c,
                                    snr_rs - snr_c,
                                    Stream::same_frames(&c, &rs),
                                    c.len(),
                                    Stream::same_prefix(&c, &rs),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The stream that `CELTENC_DIFF` describes, and its input signal.
fn stream_from_env(spec: &str) -> (Stream, Vec<f32>) {
    let f: Vec<&str> = spec.split(',').collect();
    let num = |i: usize| f[i].parse::<usize>().expect("number");
    let (channels, fs) = (num(0), num(1));
    let mut st = Stream::new(fs, channels, num(2), (num(3) * channels) as i32);
    st.complexity = num(4) as i32;
    st.vbr = f[5] == "vbr";
    // `CELTENC_PRED=1` switches the pitch pre-filter off in both encoders.
    if let Ok(p) = std::env::var("CELTENC_PRED") {
        st.prediction = p.parse().expect("prediction");
    }
    let sigs = signals(fs, channels);
    let pcm = sigs
        .into_iter()
        .find(|(name, _)| *name == f[6])
        .expect("signal")
        .1;
    (st, pcm)
}

/// Print, for each field of the header and of the allocation, in how many
/// frames the two encoders wrote the same value. `CELTENC_DIFF` picks the
/// stream, as for `first_difference`.
#[test]
#[ignore = "report, prints how often each field agrees"]
fn field_agreement() {
    let Ok(spec) = std::env::var("CELTENC_DIFF") else {
        return;
    };
    let (st, pcm) = stream_from_env(&spec);
    let c = st.enc_c(&pcm);
    let rs = st.enc_rs(&pcm);
    let lm = st.format().lm();
    let names = [
        "size",
        "silence",
        "postfilter on/off",
        "postfilter",
        "transient",
        "intra",
        "coarse",
        "tf",
        "spread",
        "boost",
        "trim",
        "intensity",
        "dual stereo",
        "coded bands",
        "pulses",
        "fine bits",
        "fine",
        "all bytes",
    ];
    let mut same = vec![0usize; names.len()];
    let mut first = vec![usize::MAX; names.len()];
    for (k, (pc, pr)) in c.iter().zip(&rs).enumerate() {
        let (hc, ac) = parse_frame(pc, st.stream_channels, st.end_band, lm);
        let (hr, ar) = parse_frame(pr, st.stream_channels, st.end_band, lm);
        let (ac, ar) = (ac.unwrap_or_default(), ar.unwrap_or_default());
        let eq = [
            pc.len() == pr.len(),
            hc.silence == hr.silence,
            hc.postfilter.is_some() == hr.postfilter.is_some(),
            hc.postfilter == hr.postfilter,
            hc.transient == hr.transient,
            hc.intra == hr.intra,
            hc.coarse == hr.coarse,
            hc.tf == hr.tf,
            hc.spread == hr.spread,
            hc.boost == hr.boost,
            hc.trim == hr.trim,
            ac.intensity == ar.intensity,
            ac.dual_stereo == ar.dual_stereo,
            ac.coded_bands == ar.coded_bands,
            ac.pulses == ar.pulses,
            ac.fine_bits == ar.fine_bits,
            ac.fine == ar.fine,
            pc == pr,
        ];
        if std::env::var("CELTENC_INTENSITY").is_ok() && ac.intensity != ar.intensity {
            println!(
                "frame {k}: intensity C {} rs {}, coded bands C {} rs {}",
                ac.intensity, ar.intensity, ac.coded_bands, ar.coded_bands
            );
        }
        for (i, &e) in eq.iter().enumerate() {
            if e {
                same[i] += 1;
            } else if first[i] == usize::MAX {
                first[i] = k;
            }
        }
    }
    println!("{spec}: {} frames", c.len());
    for (i, name) in names.iter().enumerate() {
        let at = if first[i] == usize::MAX {
            String::new()
        } else {
            format!("first difference in frame {}", first[i])
        };
        println!("  {name:<18} {:>5}  {at}", same[i]);
    }
}

/// Print the headers of the first frames in which the two encoders differ.
///
/// `CELTENC_DIFF` holds channels, sampling rate, frame duration in half
/// milliseconds, bitrate per channel, complexity, `vbr` or `cbr`, and the
/// signal name, separated by commas: `2,48000,40,32000,5,cbr,speech`.
/// An optional eighth field is the number of differing frames to print.
#[test]
#[ignore = "report, prints the first differing frames"]
fn first_difference() {
    let Ok(spec) = std::env::var("CELTENC_DIFF") else {
        return;
    };
    let f: Vec<&str> = spec.split(',').collect();
    let show = if f.len() > 7 {
        f[7].parse::<usize>().expect("number")
    } else {
        3
    };
    let (st, pcm) = stream_from_env(&spec);
    let channels = st.stream_channels;
    let pcm = &pcm;
    let c = st.enc_c(pcm);
    let rs = st.enc_rs(pcm);
    let lm = st.format().lm();
    let mut shown = 0;
    for (k, (pc, pr)) in c.iter().zip(&rs).enumerate() {
        if pc == pr {
            continue;
        }
        let (hc, ac) = parse_frame(pc, channels, st.end_band, lm);
        let (hr, ar) = parse_frame(pr, channels, st.end_band, lm);
        if hc == hr && ac == ar && std::env::var("CELTENC_HDR").is_ok() {
            continue;
        }
        if pc.len() == pr.len() && std::env::var("CELTENC_SIZE").is_ok() {
            continue;
        }
        println!("frame {k}: C {} bytes, rs {} bytes", pc.len(), pr.len());
        if hc != hr {
            println!("  C  {hc:?}");
            println!("  rs {hr:?}");
        } else if ac != ar {
            println!("  same header");
            println!("  C  {ac:?}");
            println!("  rs {ar:?}");
        } else if std::env::var("CELTENC_HDR").is_ok() {
            // Only frames whose header or allocation differ are shown.
            continue;
        } else {
            let at = pc.iter().zip(pr).take_while(|(a, b)| a == b).count();
            println!("  same header and allocation, bytes differ from {at}");
        }
        shown += 1;
        if shown == show {
            break;
        }
    }
    println!(
        "{} of {} frames the same",
        Stream::same_frames(&c, &rs),
        c.len()
    );
}
