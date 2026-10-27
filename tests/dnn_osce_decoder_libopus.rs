// opus1 computes the 64-bit ARM bits of the neural network code on every
// target, and these tests compare with the bits of the linked libopus, so
// they are compiled for 64-bit ARM only (docs/guide/10, Platforms).
#![cfg(all(feature = "test-libopus-dnn", feature = "osce-bwe", target_arch = "aarch64"))]
//! The Opus Speech Coding Enhancement (OSCE) inside the decoder: LACE
//! (complexity 6), NoLACE (7 and above) and the bandwidth extension (BWE,
//! from complexity 4 with `set_osce_bwe(true)` at 48 kHz output), against
//! libopus 1.6.1 built with them and without loop vectorization
//! (`scripts/android/build-libopus.sh --dnn`, the "novec" build). Every
//! decode call of `OpusDecoder` and `MultistreamDecoder` must give
//! libopus's bits: the float output, the 16-bit output and the final
//! range.
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,osce-bwe --test dnn_osce_decoder_libopus
//! ```
//!
//! The streams: SILK wideband (WB) of 10 to 60 ms, at low bitrates (stereo
//! with mid-only stretches), with FEC and DTX; SILK narrowband (NB) and
//! mediumband (MB), which OSCE does not touch; Hybrid; VoIP streams that
//! switch between SILK and Hybrid; music-mode streams that switch between
//! SILK, Hybrid and CELT; and spliced streams: SILK WB with Hybrid, CELT,
//! NB, 10 ms SILK and stereo SILK (a mono to stereo switch). Each is
//! decoded at 48 kHz and at 16 kHz (with `OPUS1_DNN_OSCE_ALL=1` at every
//! output rate), into one and two channels, at complexities 0, 4, 6, 7
//! and 10, with BWE on and off, and with these events: no loss, single
//! losses, bursts, losses concealed in pieces, FEC, resets, losses at the
//! start, and switches of the complexity and of BWE in the middle of the
//! stream.
//!
//! libopus 1.6.1 does not decode CELT-only frames that follow frames in
//! BWE mode: it outputs silence until the next SILK frame (fixed in
//! libopus main, commit 228a0f85). opus1 decodes them. When a run reaches
//! such a frame, the test checks that libopus's output is the silence of
//! that defect and stops the run there; `dnn_osce_decoder_props.rs`
//! checks what opus1 does instead.
//!
//! `OPUS1_DNN_OSCE_ALL=1` adds the TSP corpus as the signal and every
//! output rate, complexity and event. `OPUS1_DNN_OSCE_STREAMS=<text>`
//! keeps only the streams whose name contains the text.

mod core_common;
mod dnn_dec_common;
mod dnn_plc_common;
mod k24_common;
mod ms_support;

use std::num::NonZeroU32;
use core_common::*;
use dnn_dec_common::{CDec, CMsDec, Coding, c_encode, first_difference, rs_encode, splice, toc_mode};
use opus1::multistream::MultistreamDecoder;
use opus1::{Application, Bitrate, Channels, InbandFec, OpusDecoder, SampleRate};

fn all() -> bool {
    std::env::var_os("OPUS1_DNN_OSCE_ALL").is_some_and(|v| v != "0")
}

/// Whether the stream named `label` is decoded (`OPUS1_DNN_OSCE_STREAMS`).
fn wanted(label: &str) -> bool {
    std::env::var("OPUS1_DNN_OSCE_STREAMS").map_or(true, |s| label.contains(&s))
}

fn input(rate: usize, channels: usize, seconds: f64, seed: usize) -> Vec<f32> {
    dnn_dec_common::input(rate, channels, seconds, seed, all())
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

/// One stream to decode.
struct Stream {
    label: String,
    packets: Vec<Vec<u8>>,
}

fn streams(seconds: f64) -> Vec<Stream> {
    let mut codings = Vec::new();
    for channels in [1usize, 2] {
        let ch = channels as i32;
        let c = |label: &str, rate, mode, bw| Coding::new(&format!("{label} {channels}ch"), rate, channels, mode, bw);
        // Low bitrates: stereo SILK codes the side channel only now and
        // then (mid-only frames).
        codings.push(c("SILK WB 20 ms", 16000, Some(MODE_SILK), Some(BW_WB)).bitrates(&[9000 * ch, 12000, 20000 * ch]));
        codings.push(c("SILK WB 10 ms", 48000, Some(MODE_SILK), Some(BW_WB)).durations(&[4]).bitrates(&[16000 * ch]));
        codings.push(c("SILK WB 40/60 ms", 16000, Some(MODE_SILK), Some(BW_WB)).durations(&[16, 24]));
        codings.push(c("SILK WB 20 ms FEC", 48000, Some(MODE_SILK), Some(BW_WB)).bitrates(&[16000 * ch]).fec());
        codings.push(c("SILK WB 20 ms DTX", 16000, Some(MODE_SILK), Some(BW_WB)).bitrates(&[14000 * ch]).dtx());
        codings.push(c("SILK NB 20 ms", 8000, Some(MODE_SILK), Some(BW_NB)).bitrates(&[10000 * ch]));
        codings.push(c("SILK MB 20 ms", 12000, Some(MODE_SILK), Some(BW_MB)).bitrates(&[12000 * ch]));
        codings.push(c("Hybrid SWB 20 ms", 24000, Some(MODE_HYBRID), Some(BW_SWB)).bitrates(&[24000 * ch, 32000 * ch]));
        codings.push(c("Hybrid FB 10 ms FEC", 48000, Some(MODE_HYBRID), Some(BW_FB)).durations(&[4]).fec());
        // The encoder chooses: SILK WB at the low bitrates, Hybrid at the
        // high ones, with the switch frames it makes.
        codings.push(
            c("VoIP SILK and Hybrid", 48000, None, None)
                .durations(&[8, 8, 8, 4])
                .bitrates(&[12000 * ch, 40000 * ch, 14000 * ch, 32000 * ch]),
        );
        codings.push(
            c("mode switches", 48000, None, None)
                .application(APP_AUDIO)
                .durations(&[8, 8, 4, 8])
                .bitrates(&[10000 * ch, 64000 * ch, 16000 * ch, 40000 * ch, 12000 * ch]),
        );
    }
    let mut out = Vec::new();
    for (i, coding) in codings.iter().enumerate() {
        if !wanted(&coding.label) {
            continue;
        }
        let pcm = input(coding.rate, coding.channels, seconds, i);
        out.push(Stream { label: format!("{} (libopus)", coding.label), packets: c_encode(coding, &pcm) });
        if !coding.dtx {
            out.push(Stream { label: format!("{} (opus1)", coding.label), packets: rs_encode(coding, &pcm) });
        }
    }
    // Switches without redundancy frames, from packets of separate
    // encoders in turns.
    for channels in [1usize, 2] {
        let ch = channels as i32;
        let pcm = input(48000, channels, seconds, 5);
        let enc = |mode, bw, d: &[usize], bitrate| {
            c_encode(&Coding::new("", 48000, channels, Some(mode), Some(bw)).durations(d).bitrates(&[bitrate]), &pcm)
        };
        let silk = enc(MODE_SILK, BW_WB, &[8], 16000 * ch);
        let silk10 = enc(MODE_SILK, BW_WB, &[4], 16000 * ch);
        let silk_nb = enc(MODE_SILK, BW_NB, &[8], 10000 * ch);
        let hybrid = enc(MODE_HYBRID, BW_FB, &[8], 32000 * ch);
        let celt = enc(MODE_CELT, BW_FB, &[8], 40000 * ch);
        out.push(Stream { label: format!("SILK WB and Hybrid spliced {channels}ch"), packets: splice(&silk, &hybrid, 9) });
        out.push(Stream { label: format!("SILK WB and CELT spliced {channels}ch"), packets: splice(&silk, &celt, 13) });
        out.push(Stream { label: format!("Hybrid and CELT spliced {channels}ch"), packets: splice(&hybrid, &celt, 11) });
        // The SILK rate changes: no fade out of BWE.
        out.push(Stream { label: format!("SILK WB and NB spliced {channels}ch"), packets: splice(&silk, &silk_nb, 7) });
        // 10 ms frames reset LACE and NoLACE; BWE takes them.
        out.push(Stream { label: format!("SILK WB 20 and 10 ms spliced {channels}ch"), packets: splice(&silk, &silk10, 5) });
    }
    // Mono and stereo SILK in turns: the second channel starts afresh on
    // each switch to stereo, its BWE state at zero.
    let mono = c_encode(
        &Coding::new("", 48000, 1, Some(MODE_SILK), Some(BW_WB)).bitrates(&[16000]),
        &input(48000, 1, seconds, 6),
    );
    let stereo = c_encode(
        &Coding::new("", 48000, 2, Some(MODE_SILK), Some(BW_WB)).bitrates(&[32000]),
        &input(48000, 2, seconds, 6),
    );
    out.push(Stream { label: "SILK WB mono and stereo spliced".to_string(), packets: splice(&mono, &stereo, 8) });
    out.retain(|s| wanted(&s.label));
    out
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// One step of a decoding run.
#[derive(Clone, Copy, Debug)]
enum Call {
    /// Decode packet `k`.
    Packet(usize),
    /// Conceal this many samples per channel.
    Conceal(usize),
    /// Rebuild this many samples per channel from the FEC data of packet
    /// `k`.
    Fec(usize, usize),
    /// `reset_state` / `OPUS_RESET_STATE`.
    Reset,
    /// Set the complexity.
    Complexity(i32),
    /// Switch the bandwidth extension on or off.
    Bwe(bool),
}

/// What happens to the packets of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Events {
    None,
    /// About one packet in eight is lost.
    Single,
    /// Three packets in a row out of every 19.
    Burst,
    /// About one packet in six, concealed in pieces of 2.5 ms, 7.5 ms and
    /// the rest.
    Pieces,
    /// About one packet in eight, rebuilt from the FEC data of the next one.
    Fec,
    /// Single losses, and a decoder reset every 37 packets.
    Reset,
    /// The first 5 packets are lost, then single losses.
    Start,
    /// The complexity goes through 7, 6, 4, 10, 0, 6, 5 and 7, changing
    /// every 11 packets, BWE switches every 17 packets, and some packets
    /// are lost.
    Switch,
}

const EVENTS: [Events; 8] =
    [Events::None, Events::Single, Events::Burst, Events::Pieces, Events::Fec, Events::Reset, Events::Start, Events::Switch];

/// The calls that decode `packets` at `rate` with `events`.
fn calls(packets: &[Vec<u8>], rate: usize, events: Events, seed: u32) -> Vec<Call> {
    let duration = |k: usize| opus1::packet::nb_samples(&packets[k], SampleRate::try_from(rate as u32).unwrap()).unwrap();
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(17);
    let mut calls = Vec::new();
    for k in 0..packets.len() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = state >> 24;
        let n = duration(k);
        if events == Events::Switch {
            if k > 0 && k % 11 == 0 {
                calls.push(Call::Complexity([7, 6, 4, 10, 0, 6, 5, 7][k / 11 % 8]));
            }
            if k > 0 && k % 17 == 0 {
                calls.push(Call::Bwe(k / 17 % 2 == 0));
            }
        }
        match events {
            Events::Single | Events::Switch if k > 2 && r < 32 => calls.push(Call::Conceal(n)),
            Events::Burst if k > 2 && k % 19 >= 16 => calls.push(Call::Conceal(n)),
            Events::Pieces if k > 2 && r < 42 => {
                let unit = rate / 400;
                let mut left = n;
                for piece in [unit, 3 * unit] {
                    if left > piece {
                        calls.push(Call::Conceal(piece));
                        left -= piece;
                    }
                }
                calls.push(Call::Conceal(left));
            }
            Events::Fec if k > 2 && r < 32 && k + 1 < packets.len() => calls.push(Call::Fec(k + 1, n)),
            Events::Reset if k > 2 && r < 28 => calls.push(Call::Conceal(n)),
            Events::Reset if k % 37 == 20 => {
                calls.push(Call::Reset);
                calls.push(Call::Packet(k));
            }
            Events::Start if k < 5 || r < 28 => calls.push(Call::Conceal(n)),
            _ => calls.push(Call::Packet(k)),
        }
    }
    calls
}

/// The settings of a run.
#[derive(Clone, Copy, Debug)]
struct Setting {
    rate: usize,
    channels: usize,
    complexity: i32,
    bwe: bool,
    events: Events,
}

/// What one decoding run did.
#[derive(Default, Clone, Copy)]
struct Count {
    calls: usize,
    concealed: usize,
    /// The run stopped at a CELT-only frame after BWE frames, which
    /// libopus 1.6.1 does not decode.
    cut: bool,
}

/// Whether a call may decode or conceal a CELT-only frame: a CELT-only
/// packet, or a concealment next to one. `last` is the last packet
/// decoded before the call. A concealment is in CELT mode after a
/// CELT-only packet, and after a packet that ends with the redundancy
/// frame of a switch to CELT-only.
fn may_be_celt_only(call: Call, packets: &[Vec<u8>], last: Option<usize>) -> bool {
    let celt = |k: usize| packets.get(k).is_some_and(|p| toc_mode(p) == MODE_CELT);
    match call {
        Call::Packet(k) | Call::Fec(k, _) => celt(k),
        Call::Conceal(_) => last.is_some_and(|k| celt(k) || celt(k + 1)),
        _ => false,
    }
}

/// Decodes `packets` with libopus and opus1 in float and in 16 bits.
/// Returns the first difference, or what was decoded.
fn compare(packets: &[Vec<u8>], s: Setting, seed: u32) -> Result<Count, String> {
    let (rate, channels) = (s.rate, s.channels);
    let mut c = CDec::new(rate, channels);
    let mut c16 = CDec::new(rate, channels);
    let mut d = OpusDecoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    let mut d16 = OpusDecoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    let complexity = |c: &mut CDec, c16: &mut CDec, d: &mut OpusDecoder, d16: &mut OpusDecoder, value| {
        c.set_complexity(value);
        c16.set_complexity(value);
        d.set_complexity(value as u8).unwrap();
        d16.set_complexity(value as u8).unwrap();
    };
    let bwe = |c: &mut CDec, c16: &mut CDec, d: &mut OpusDecoder, d16: &mut OpusDecoder, on| {
        c.set_osce_bwe(on);
        c16.set_osce_bwe(on);
        d.set_osce_bwe(on);
        d16.set_osce_bwe(on);
    };
    complexity(&mut c, &mut c16, &mut d, &mut d16, s.complexity);
    bwe(&mut c, &mut c16, &mut d, &mut d16, s.bwe);
    let (mut now_complexity, mut now_bwe) = (s.complexity, s.bwe);
    // Whether BWE has run in this decoder (it may then hit the defect of
    // libopus 1.6.1).
    let mut bwe_ran = false;
    let mut last = None;
    let mut out = vec![0f32; rate * channels];
    let mut out16 = vec![0i16; rate * channels];
    let mut count = Count::default();
    for (index, call) in calls(packets, rate, s.events, seed).into_iter().enumerate() {
        let (want, want16, got, got16) = match call {
            Call::Packet(k) => {
                let p = &packets[k];
                let n = opus1::packet::nb_samples(p, SampleRate::try_from(rate as u32).unwrap()).unwrap();
                (
                    c.decode(Some(p), n, false),
                    c16.decode_i16(Some(p), n, false),
                    d.decode(p, n, &mut out),
                    d16.decode_i16(p, n, &mut out16),
                )
            }
            Call::Conceal(n) => {
                count.concealed += 1;
                (c.decode(None, n, false), c16.decode_i16(None, n, false), d.decode(&[], n, &mut out), d16.decode_i16(&[], n, &mut out16))
            }
            Call::Fec(k, n) => {
                count.concealed += 1;
                let p = &packets[k];
                (c.decode(Some(p), n, true), c16.decode_i16(Some(p), n, true), d.decode_fec(p, n, &mut out), d16.decode_fec_i16(p, n, &mut out16))
            }
            Call::Reset => {
                c.reset();
                c16.reset();
                d.reset_state();
                d16.reset_state();
                continue;
            }
            Call::Complexity(value) => {
                complexity(&mut c, &mut c16, &mut d, &mut d16, value);
                now_complexity = value;
                continue;
            }
            Call::Bwe(on) => {
                bwe(&mut c, &mut c16, &mut d, &mut d16, on);
                now_bwe = on;
                continue;
            }
        };
        count.calls += 1;
        let at = format!("call {index} ({call:?})");
        let (want, got) = match (want, got) {
            (Ok(w), Ok(n)) => (w, &out[..n * channels]),
            (w, g) => return Err(format!("{at}: libopus {:?}, opus1 {:?}", w.map(|v| v.len()), g)),
        };
        let (want16, got16) = match (want16, got16) {
            (Ok(w), Ok(n)) => (w, &out16[..n * channels]),
            (w, g) => return Err(format!("{at}: 16 bits: libopus {:?}, opus1 {:?}", w.map(|v| v.len()), g)),
        };
        let diff = first_difference(&want, got, |a: f32, b: f32| a.to_bits() == b.to_bits())
            .map(|w| format!("float output, {w}"))
            .or_else(|| first_difference(&want16, got16, |a: i16, b: i16| a == b).map(|w| format!("16-bit output, {w}")))
            .or_else(|| {
                ((c.final_range(), c16.final_range()) != (d.final_range(), d16.final_range())).then(|| "final range".to_string())
            });
        bwe_ran |= now_bwe && now_complexity >= 4 && rate == 48000;
        if let Some(what) = diff {
            // libopus 1.6.1 skips CELT-only frames after BWE frames: its
            // output is zero, after the first 5 ms of a packet, which may
            // hold the cross-fade of a mode switch.
            let from = if matches!(call, Call::Conceal(_)) { 0 } else { rate / 200 * channels };
            let silent =
                want.len() > from && want[from..].iter().all(|&v| v == 0.0) && got[from..].iter().any(|&v| v != 0.0);
            if bwe_ran && may_be_celt_only(call, packets, last) && silent {
                count.cut = true;
                return Ok(count);
            }
            return Err(format!("{at}: {what}"));
        }
        if let Call::Packet(k) | Call::Fec(k, _) = call {
            last = Some(k);
        }
    }
    Ok(count)
}

/// The settings of the runs of one stream at one rate and channel count.
fn plan(rate: usize, channels: usize) -> Vec<Setting> {
    let mut out = Vec::new();
    let mut add = |complexity, bwe, events: &[Events]| {
        for &events in events {
            out.push(Setting { rate, channels, complexity, bwe, events });
        }
    };
    if all() {
        for complexity in 0..=10 {
            add(complexity, true, &EVENTS);
            add(complexity, false, &EVENTS);
        }
        return out;
    }
    use Events::*;
    add(7, true, &EVENTS);
    add(10, true, &[None, Single, Burst, Pieces, Fec, Reset]);
    add(6, true, &[None, Single, Burst, Reset, Switch]);
    add(4, true, &[None, Single, Pieces, Fec]);
    add(0, true, &[Single, Switch]);
    add(7, false, &[None, Single, Fec, Reset]);
    add(6, false, &[None, Single, Burst]);
    out
}

/// Decodes every stream at the output rates and channel counts with the
/// settings of `plan`, on all cores. Panics with the list of runs that
/// differ.
fn decode_streams(streams: &[Stream], rates: &[usize]) {
    let start = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let results: Vec<(usize, Count, usize, Vec<String>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let (mut runs, mut total, mut cut) = (0, Count::default(), 0);
                    let mut failures = Vec::new();
                    for stream in streams.iter().skip(t).step_by(threads) {
                        // The events depend on the stream's name, not on its
                        // place in the list, so a run with
                        // `OPUS1_DNN_OSCE_STREAMS` repeats them.
                        let name = stream.label.bytes().fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
                        for &rate in rates {
                            for channels in [1, 2] {
                                for s in plan(rate, channels) {
                                    runs += 1;
                                    let seed = name ^ (rate + channels * 31 + s.complexity as usize * 131) as u32;
                                    match compare(&stream.packets, s, seed) {
                                        Ok(count) => {
                                            total.calls += count.calls;
                                            total.concealed += count.concealed;
                                            cut += count.cut as usize;
                                        }
                                        Err(e) => failures.push(format!("{}, {s:?}: {e}", stream.label)),
                                    }
                                }
                            }
                        }
                    }
                    (runs, total, cut, failures)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    let runs: usize = results.iter().map(|r| r.0).sum();
    let calls: usize = results.iter().map(|r| r.1.calls).sum();
    let concealed: usize = results.iter().map(|r| r.1.concealed).sum();
    let cut: usize = results.iter().map(|r| r.2).sum();
    let failures: Vec<&String> = results.iter().flat_map(|r| &r.3).collect();
    println!(
        "{} streams, {runs} runs, {calls} decode calls ({concealed} concealed or rebuilt), {cut} runs stopped at a \
         CELT-only frame after BWE (libopus 1.6.1 defect), {} differ, {:.1} s",
        streams.len(),
        failures.len(),
        start.elapsed().as_secs_f64()
    );
    for f in failures.iter().take(40) {
        println!("DIFF {f}");
    }
    assert!(failures.is_empty(), "{} of {runs} runs differ from libopus", failures.len());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Every stream at 48 and 16 kHz output, or with `OPUS1_DNN_OSCE_ALL` at
/// all five output rates.
#[test]
fn streams_match_libopus() {
    let streams = streams(if all() { 8.0 } else { 4.0 });
    let rates: &[usize] = if all() { &[8000, 12000, 16000, 24000, 48000] } else { &[48000, 16000] };
    decode_streams(&streams, rates);
}

/// The other output rates, 8, 12 and 24 kHz, where BWE does not run, for
/// a few streams.
#[test]
fn other_output_rates_match_libopus() {
    if all() {
        return; // `streams_match_libopus` covers them.
    }
    let mut streams = streams(2.0);
    streams.retain(|s| s.label.contains("SILK WB 20 ms") || s.label.contains("VoIP") || s.label.contains("mono and stereo"));
    decode_streams(&streams, &[8000, 12000, 24000]);
}

/// The tests above would pass if OSCE never ran: in libopus, LACE,
/// NoLACE and BWE each change the output of SILK wideband speech, and
/// libopus 1.6.1 outputs silence for CELT-only frames after BWE frames
/// while opus1 decodes them.
#[test]
fn osce_changes_the_output_of_libopus() {
    let pcm = input(48000, 1, 3.0, 0);
    let silk = c_encode(&Coding::new("", 48000, 1, Some(MODE_SILK), Some(BW_WB)), &pcm);
    let decode = |complexity, bwe| {
        let mut c = CDec::new(48000, 1);
        c.set_complexity(complexity);
        c.set_osce_bwe(bwe);
        silk.iter().flat_map(|p| c.decode(Some(p), 960, false).unwrap()).collect::<Vec<f32>>()
    };
    let plain = decode(5, false);
    for (label, complexity, bwe) in [("LACE", 6, false), ("NoLACE", 7, false), ("BWE", 4, true)] {
        let other = decode(complexity, bwe);
        let differ = plain.iter().zip(&other).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("{label}: {differ} of {} samples differ from complexity 5", plain.len());
        assert!(differ > plain.len() / 2, "{label} does not change libopus's output");
    }

    // SILK wideband, then CELT-only wideband, with a redundancy frame
    // between them (a music-mode encoder that is told to switch; with
    // another bandwidth it would switch through a Hybrid frame).
    let mut enc = CEncoder::new(48000, 1, APP_AUDIO);
    enc.bitrate(16000).bandwidth(BW_WB).force_mode(MODE_SILK);
    let mut packets: Vec<Vec<u8>> = pcm[..48000].as_chunks::<960>().0.iter().map(|f| enc.encode(f)).collect();
    enc.bitrate(64000).force_mode(MODE_CELT);
    packets.extend(pcm[48000..96000].as_chunks::<960>().0.iter().map(|f| enc.encode(f)));
    let mut c = CDec::new(48000, 1);
    c.set_complexity(7);
    c.set_osce_bwe(true);
    let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    d.set_complexity(7).unwrap();
    d.set_osce_bwe(true);
    let mut out = vec![0f32; 960];
    let (mut silent, mut celt) = (0, 0);
    for p in &packets {
        let want = c.decode(Some(p), 960, false).unwrap();
        let n = d.decode(p, 960, &mut out).unwrap();
        if toc_mode(p) == MODE_CELT {
            celt += 1;
            silent += want.iter().all(|&v| v == 0.0) as usize;
            assert!(out[..n].iter().any(|&v| v != 0.0), "opus1 skipped a CELT-only frame");
        }
    }
    println!("libopus 1.6.1 output silence for {silent} of {celt} CELT-only frames after BWE frames");
    assert_eq!(silent, celt, "the linked libopus decodes CELT-only frames after BWE frames");
}

/// A 5.1 multistream decoder (4 streams, 2 of them coupled) against
/// libopus at complexities 6, 7 and 10, BWE on and off, with losses, FEC
/// and resets, for packets of libopus's and opus1's multistream encoders.
/// libopus's multistream decoder does not forward `OPUS_SET_OSCE_BWE`, so
/// the test sets it on each libopus stream decoder; opus1 forwards it.
#[test]
fn multistream_matches_libopus() {
    let channels = 6;
    let mut pcm = Vec::new();
    let voices: Vec<Vec<f32>> = (0..channels).map(|c| input(48000, 1, 4.0, c)).collect();
    for i in 0..voices[0].len() {
        for v in &voices {
            pcm.push(v[i]);
        }
    }
    let mut packet_sets = Vec::new();
    for (bitrate, fec) in [(64000, false), (96000, true)] {
        let mut enc = ms_support::CMsEncoder::surround(48000, channels, 1, APP_VOIP).unwrap();
        enc.set_bitrate(bitrate);
        if fec {
            enc.set_inband_fec(true).set_packet_loss_perc(20);
        }
        let layout = (enc.streams, enc.coupled, enc.mapping.clone());
        packet_sets.push((format!("libopus {bitrate} b/s"), enc.encode_all(&pcm, 960), layout));
        let mut ours =
            opus1::multistream::MultistreamEncoder::new_surround(SampleRate::Hz48000, channels, 1, Application::Voip).unwrap();
        ours.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
        if fec {
            ours.set_inband_fec(InbandFec::On);
            ours.set_packet_loss_perc(20).unwrap();
        }
        let layout = ours.layout();
        let layout = (layout.streams(), layout.coupled_streams(), layout.mapping().to_vec());
        let mut buf = vec![0u8; 8000];
        let packets = pcm
            .chunks_exact(960 * channels)
            .map(|frame| {
                let n = ours.encode(frame, 960, &mut buf).unwrap();
                buf[..n].to_vec()
            })
            .collect::<Vec<_>>();
        packet_sets.push((format!("opus1 {bitrate} b/s"), packets, layout));
    }
    let mut failures = Vec::new();
    let mut runs = 0;
    for (label, packets, (streams, coupled, mapping)) in &packet_sets {
        for rate in [48000usize, 16000] {
            for (complexity, bwe) in [(6, false), (7, true), (10, true), (4, true)] {
                for events in [Events::Single, Events::Burst, Events::Fec, Events::Reset] {
                    runs += 1;
                    let mut c = CMsDec::new(rate, channels, *streams, *coupled, mapping);
                    let mut c16 = CMsDec::new(rate, channels, *streams, *coupled, mapping);
                    let mut d = MultistreamDecoder::new(
                        SampleRate::try_from(rate as i32).unwrap(),
                        channels,
                        *streams,
                        *coupled,
                        mapping,
                    )
                    .unwrap();
                    let mut d16 = MultistreamDecoder::new(
                        SampleRate::try_from(rate as i32).unwrap(),
                        channels,
                        *streams,
                        *coupled,
                        mapping,
                    )
                    .unwrap();
                    c.set_complexity(complexity);
                    c16.set_complexity(complexity);
                    c.set_osce_bwe(bwe);
                    c16.set_osce_bwe(bwe);
                    d.set_complexity(complexity as u8).unwrap();
                    d16.set_complexity(complexity as u8).unwrap();
                    d.set_osce_bwe(bwe);
                    d16.set_osce_bwe(bwe);
                    let mut out = vec![0f32; 5760 * channels];
                    let mut out16 = vec![0i16; 5760 * channels];
                    for (index, call) in calls(packets, rate, events, 3).into_iter().enumerate() {
                        let (want, want16, got, got16) = match call {
                            Call::Packet(k) => (
                                c.decode(Some(&packets[k]), 5760, false),
                                c16.decode_i16(Some(&packets[k]), 5760, false),
                                d.decode(&packets[k], 5760, &mut out),
                                d16.decode_i16(&packets[k], 5760, &mut out16),
                            ),
                            Call::Conceal(n) => (
                                c.decode(None, n, false),
                                c16.decode_i16(None, n, false),
                                d.decode(&[], n, &mut out),
                                d16.decode_i16(&[], n, &mut out16),
                            ),
                            Call::Fec(k, n) => (
                                c.decode(Some(&packets[k]), n, true),
                                c16.decode_i16(Some(&packets[k]), n, true),
                                d.decode_fec(&packets[k], n, &mut out),
                                d16.decode_fec_i16(&packets[k], n, &mut out16),
                            ),
                            Call::Reset => {
                                c.reset();
                                c16.reset();
                                d.reset_state();
                                d16.reset_state();
                                continue;
                            }
                            Call::Complexity(_) | Call::Bwe(_) => unreachable!(),
                        };
                        let (want, want16, n, n16) = (want.unwrap(), want16.unwrap(), got.unwrap(), got16.unwrap());
                        let diff = first_difference(&want, &out[..n * channels], |a: f32, b: f32| a.to_bits() == b.to_bits())
                            .or_else(|| first_difference(&want16, &out16[..n16 * channels], |a: i16, b: i16| a == b))
                            .or_else(|| {
                                (matches!(call, Call::Packet(_)) && c.final_range() != d.final_range())
                                    .then(|| "final range".to_string())
                            });
                        if let Some(what) = diff {
                            failures.push(format!(
                                "{label}, {rate} Hz, complexity {complexity}, BWE {bwe}, {events:?}, call {index} ({call:?}): {what}"
                            ));
                            break;
                        }
                    }
                }
            }
        }
    }
    println!("multistream: {runs} runs, {} differ", failures.len());
    for f in &failures {
        println!("DIFF {f}");
    }
    assert!(failures.is_empty());
}

/// The time of a decode call of a 20 ms SILK wideband packet (20 kb/s,
/// 10 s of speech, no loss) for opus1 and for the linked libopus, mono
/// 48 kHz output, at complexity 5 (no enhancement), 6 (LACE), 7
/// (NoLACE), 4 with BWE and 7 with BWE. Medians over 5 runs, in
/// microseconds. Run it with `--ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "prints timings"]
fn timing() {
    let pcm = input(48000, 1, 10.0, 0);
    let packets = c_encode(&Coding::new("", 48000, 1, Some(MODE_SILK), Some(BW_WB)).bitrates(&[20000]), &pcm);
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    println!("| Setting | opus1 (µs) | libopus (µs) |");
    println!("|---|---:|---:|");
    for (label, complexity, bwe) in
        [("complexity 5", 5, false), ("LACE (6)", 6, false), ("NoLACE (7)", 7, false), ("BWE (4)", 4, true), ("NoLACE and BWE (7)", 7, true)]
    {
        let (mut ours, mut theirs) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
            d.set_complexity(complexity).unwrap();
            d.set_osce_bwe(bwe);
            let mut c = CDec::new(48000, 1);
            c.set_complexity(i32::from(complexity));
            c.set_osce_bwe(bwe);
            let mut out = vec![0f32; 960];
            let (mut t_ours, mut t_theirs) = (Vec::new(), Vec::new());
            for p in &packets {
                let start = std::time::Instant::now();
                d.decode(p, 960, &mut out).unwrap();
                t_ours.push(start.elapsed().as_secs_f64() * 1e6);
                let start = std::time::Instant::now();
                c.decode(Some(p), 960, false).unwrap();
                t_theirs.push(start.elapsed().as_secs_f64() * 1e6);
            }
            ours.push(median(t_ours));
            theirs.push(median(t_theirs));
        }
        println!("| {label} | {:.1} | {:.1} |", median(ours), median(theirs));
    }
}

/// With `OPUS1_DNN_REF_O2=1`, against a libopus built at `-O2` with loop
/// vectorization (`target/android/libopus-host-1.6.1-dnn`), whose OSCE
/// rounds differently: measures the difference of the decoded output, as
/// a signal-to-noise ratio (SNR), for every stream without CELT-only
/// packets, mono and stereo at 48 kHz, at complexity 6 (LACE), 7 (NoLACE)
/// and 4 with BWE, without loss. Without the variable the test does
/// nothing.
#[test]
fn distance_to_an_o2_build() {
    if std::env::var_os("OPUS1_DNN_REF_O2").is_none_or(|v| v == "0") {
        return;
    }
    let mut streams = streams(if all() { 8.0 } else { 4.0 });
    streams.retain(|s| s.packets.iter().all(|p| toc_mode(p) != MODE_CELT));
    println!("| Setting | Calls that differ | Largest difference (16 bits) | SNR of the difference |");
    println!("|---|---:|---:|---:|");
    for (label, complexity, bwe) in [("LACE", 6, false), ("NoLACE", 7, false), ("BWE (complexity 4)", 4, true), ("NoLACE and BWE", 7, true)] {
        let (mut sig, mut err, mut largest, mut calls, mut differ) = (0f64, 0f64, 0i32, 0usize, 0usize);
        for stream in &streams {
            for channels in [1, 2] {
                let mut c = CDec::new(48000, channels);
                c.set_complexity(complexity);
                c.set_osce_bwe(bwe);
                let mut d = OpusDecoder::new(
                    SampleRate::Hz48000,
                    Channels::try_from(channels).unwrap(),
                )
                .unwrap();
                d.set_complexity(complexity as u8).unwrap();
                d.set_osce_bwe(bwe);
                let mut out = vec![0i16; 5760 * channels];
                for p in &stream.packets {
                    let want = c.decode_i16(Some(p), 5760, false).unwrap();
                    let n = d.decode_i16(p, 5760, &mut out).unwrap();
                    let got = &out[..n * channels];
                    calls += 1;
                    let mut call_differs = false;
                    for (&a, &b) in want.iter().zip(got) {
                        sig += (a as f64).powi(2);
                        err += (a as f64 - b as f64).powi(2);
                        largest = largest.max((a as i32 - b as i32).abs());
                        call_differs |= a != b;
                    }
                    differ += call_differs as usize;
                }
            }
        }
        let snr = if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() };
        println!("| {label} | {differ} of {calls} | {largest} | {snr:.1} dB |");
        assert!(snr > 40.0, "{label}: {snr:.1} dB from the -O2 build");
    }
}

