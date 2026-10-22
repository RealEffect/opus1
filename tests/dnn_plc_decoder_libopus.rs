// opus1 computes the 64-bit ARM bits of the neural network code on every
// target, and these tests compare with the bits of the linked libopus, so
// they are compiled for 64-bit ARM only (docs/10, Platforms).
#![cfg(all(feature = "test-libopus-dnn", feature = "deep-plc", target_arch = "aarch64"))]
//! Deep packet loss concealment (deep PLC) inside the decoder, against
//! libopus 1.6.1 built with it and without loop vectorization
//! (`scripts/android/build-libopus.sh --dnn`, the "novec" build). Every
//! decode call of `OpusDecoder` and `MultistreamDecoder` must give
//! libopus's bits: the float output, the 16-bit output and the final
//! range.
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
//!   CARGO_TARGET_DIR=<worktree>/target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,osce-bwe --test dnn_plc_decoder_libopus
//! ```
//!
//! The streams are coded by libopus and by opus1: SILK at 8, 12 and 16 kHz,
//! Hybrid, CELT, mono and stereo, frames of 2.5 to 60 ms, in-band FEC, mode
//! switches with and without the redundancy frames, and packets of two
//! streams spliced together (a switch without redundancy: the decoder
//! conceals 5 ms in the old mode on a packet that arrived, so deep PLC also
//! runs without loss). Each stream is decoded at 8 to 48 kHz, into one and
//! two channels, at several decoder complexities, with these events: no
//! loss, single losses, bursts, bursts longer than 200 ms, losses
//! concealed in pieces, FEC, a decoder reset in the middle, complexity
//! changes inside losses, and losses at the start.
//!
//! The reference build also has OSCE (LACE at complexity 6, NoLACE from
//! 7), which enhances SILK frames of 20 ms at 16 kHz, and so does opus1
//! with the feature `osce`: complexities from 6 compare deep PLC and the
//! enhancement together. The bandwidth extension stays off here;
//! `dnn_osce_decoder_libopus.rs` tests it.
//!
//! `OPUS1_DNN_PLC_ALL=1` adds the TSP corpus as the signal and decodes
//! every stream at every complexity from 0 to 10, at every output rate,
//! with every loss pattern. The default run takes about 45 s on an Apple
//! M4 Pro.
//! `OPUS1_DNN_PLC_STREAMS=<text>` keeps only the streams whose name
//! contains the text.

mod core_common;
mod dnn_dec_common;
mod dnn_plc_common;
mod k24_common;
mod ms_support;

use std::num::NonZeroU32;
use core_common::*;
use dnn_dec_common::{CDec, CMsDec, Coding, c_encode, first_difference, rs_encode, splice};
use opus1::multistream::MultistreamDecoder;
use opus1::{Application, Bitrate, Channels, InbandFec, OpusDecoder, SampleRate};

fn all() -> bool {
    std::env::var_os("OPUS1_DNN_PLC_ALL").is_some_and(|v| v != "0")
}

/// Whether the stream named `label` is decoded (`OPUS1_DNN_PLC_STREAMS`).
fn wanted(label: &str) -> bool {
    std::env::var("OPUS1_DNN_PLC_STREAMS").map_or(true, |s| label.contains(&s))
}

/// `seconds` of input for an encoder at `rate` with `channels` (see
/// `dnn_dec_common::input`), with the TSP corpus under
/// `OPUS1_DNN_PLC_ALL`.
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

/// The complexities to decode at.
fn complexities() -> Vec<i32> {
    if all() { (0..=10).collect() } else { vec![0, 4, 5, 6, 10] }
}

/// The streams of the tests: each coding by libopus and by opus1, and the
/// spliced streams.
fn streams(seconds: f64) -> Vec<Stream> {
    let mut codings = Vec::new();
    for channels in [1usize, 2] {
        let ch = channels as i32;
        let c = |label: &str, rate, mode, bw| Coding::new(&format!("{label} {channels}ch"), rate, channels, mode, bw);
        codings.push(c("SILK NB 20 ms", 8000, Some(MODE_SILK), Some(BW_NB)).bitrates(&[8000 * ch, 12000 * ch]));
        codings.push(c("SILK MB 20 ms", 12000, Some(MODE_SILK), Some(BW_MB)).bitrates(&[10000 * ch, 14000 * ch]));
        codings.push(c("SILK WB 20 ms", 16000, Some(MODE_SILK), Some(BW_WB)).bitrates(&[12000 * ch, 20000 * ch]));
        codings.push(c("SILK WB 10 ms", 16000, Some(MODE_SILK), Some(BW_WB)).durations(&[4]));
        codings.push(c("SILK WB 40/60 ms", 16000, Some(MODE_SILK), Some(BW_WB)).durations(&[16, 24]));
        codings.push(c("SILK WB 20 ms FEC", 16000, Some(MODE_SILK), Some(BW_WB)).bitrates(&[16000 * ch]).fec());
        codings.push(c("SILK WB 20 ms DTX", 16000, Some(MODE_SILK), Some(BW_WB)).bitrates(&[14000 * ch]).dtx());
        codings.push(c("Hybrid SWB 20 ms", 24000, Some(MODE_HYBRID), Some(BW_SWB)).bitrates(&[24000 * ch, 32000 * ch]));
        codings.push(c("Hybrid FB 10 ms FEC", 48000, Some(MODE_HYBRID), Some(BW_FB)).durations(&[4]).fec());
        codings.push(c("CELT FB 20 ms", 48000, Some(MODE_CELT), Some(BW_FB)).bitrates(&[32000 * ch, 48000 * ch]));
        codings.push(c("CELT FB 10 ms", 48000, Some(MODE_CELT), Some(BW_FB)).durations(&[4]).bitrates(&[48000 * ch]));
        codings.push(c("CELT FB 5/2.5 ms", 48000, Some(MODE_CELT), Some(BW_FB)).durations(&[2, 1]).bitrates(&[64000 * ch]));
        codings.push(c("CELT WB 20 ms", 16000, Some(MODE_CELT), Some(BW_WB)).bitrates(&[24000 * ch]));
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
    // Switches without redundancy: SILK or Hybrid packets and CELT packets
    // of separate encoders, in turns.
    for channels in [1usize, 2] {
        let ch = channels as i32;
        let pcm = input(48000, channels, seconds, 3);
        let silk = c_encode(&Coding::new("", 48000, channels, Some(MODE_SILK), Some(BW_WB)).bitrates(&[16000 * ch]), &pcm);
        let hybrid = c_encode(&Coding::new("", 48000, channels, Some(MODE_HYBRID), Some(BW_FB)).bitrates(&[32000 * ch]), &pcm);
        let celt = rs_encode(&Coding::new("", 48000, channels, Some(MODE_CELT), Some(BW_FB)).bitrates(&[40000 * ch]), &pcm);
        let celt10 = c_encode(
            &Coding::new("", 48000, channels, Some(MODE_CELT), Some(BW_FB)).durations(&[4]).bitrates(&[48000 * ch]),
            &pcm,
        );
        let silk10 =
            c_encode(&Coding::new("", 48000, channels, Some(MODE_SILK), Some(BW_WB)).durations(&[4]).bitrates(&[16000 * ch]), &pcm);
        out.push(Stream { label: format!("SILK WB and CELT spliced {channels}ch"), packets: splice(&silk, &celt, 13) });
        out.push(Stream { label: format!("Hybrid and CELT spliced {channels}ch"), packets: splice(&hybrid, &celt, 11) });
        out.push(Stream { label: format!("SILK WB and CELT 10 ms spliced {channels}ch"), packets: splice(&silk10, &celt10, 17) });
        // The SILK rate changes for more than 15 frames: no history is
        // appended at 8 kHz.
        let silk_nb = c_encode(&Coding::new("", 48000, channels, Some(MODE_SILK), Some(BW_NB)).bitrates(&[10000 * ch]), &pcm);
        out.push(Stream { label: format!("SILK WB and NB spliced {channels}ch"), packets: splice(&silk, &silk_nb, 21) });
    }
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
}

/// How packets are lost.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Loss {
    None,
    /// About one packet in eight.
    Single,
    /// Three packets in a row out of every 19.
    Burst,
    /// 14 packets in a row out of every 50, concealed one packet at a time:
    /// longer than 200 ms with 20 ms packets, where CELT goes from deep
    /// PLC to noise.
    Long,
    /// About one packet in six, concealed in pieces of 2.5 ms, 7.5 ms and
    /// the rest.
    Pieces,
    /// About one packet in eight, rebuilt from the FEC data of the next one
    /// (or concealed when it has none).
    Fec,
    /// Single losses and bursts, with a decoder reset every 61 packets,
    /// once right after a burst.
    Reset,
    /// Bursts during which the complexity goes from 5 to 4 and back.
    Flip,
    /// The first 6 packets are lost, then single losses.
    Start,
    /// Packets 30 to 139 are lost (2.2 s of 20 ms packets), concealed one
    /// packet at a time.
    Huge,
}

const LOSSES: [Loss; 10] = [
    Loss::None,
    Loss::Single,
    Loss::Burst,
    Loss::Long,
    Loss::Pieces,
    Loss::Fec,
    Loss::Reset,
    Loss::Flip,
    Loss::Start,
    Loss::Huge,
];

/// The calls that decode `packets` at `rate` with `loss`.
fn calls(packets: &[Vec<u8>], rate: usize, loss: Loss, seed: u32) -> Vec<Call> {
    let duration = |k: usize| opus1::packet::nb_samples(&packets[k], SampleRate::try_from(rate as u32).unwrap()).unwrap();
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(17);
    let mut calls = Vec::new();
    for k in 0..packets.len() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = state >> 24;
        let n = duration(k);
        match loss {
            Loss::Single if k > 2 && r < 32 => calls.push(Call::Conceal(n)),
            Loss::Burst if k > 2 && k % 19 >= 16 => calls.push(Call::Conceal(n)),
            Loss::Long if k > 2 && k % 50 >= 36 => calls.push(Call::Conceal(n)),
            Loss::Pieces if k > 2 && r < 42 => {
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
            Loss::Fec if k > 2 && r < 32 && k + 1 < packets.len() => calls.push(Call::Fec(k + 1, n)),
            Loss::Reset if k > 2 && (r < 24 || k % 61 >= 57) => {
                calls.push(Call::Conceal(n));
                if k % 61 == 60 {
                    calls.push(Call::Reset);
                }
            }
            Loss::Reset if k % 61 == 30 => {
                calls.push(Call::Reset);
                calls.push(Call::Packet(k));
            }
            Loss::Flip if k > 2 && k % 23 >= 17 => {
                if k % 23 == 19 {
                    calls.push(Call::Complexity(4));
                }
                if k % 23 == 21 {
                    calls.push(Call::Complexity(5));
                }
                calls.push(Call::Conceal(n));
            }
            Loss::Start if k < 6 || r < 28 => calls.push(Call::Conceal(n)),
            Loss::Huge if (30..140).contains(&k) => calls.push(Call::Conceal(n)),
            _ => calls.push(Call::Packet(k)),
        }
    }
    calls
}

/// What one decoding run did.
#[derive(Default, Clone, Copy)]
struct Count {
    calls: usize,
    concealed: usize,
}

/// Decodes `packets` with libopus and opus1 in float and in 16 bits.
/// Returns the first difference, or what was decoded.
fn compare(packets: &[Vec<u8>], rate: usize, channels: usize, complexity: i32, loss: Loss, seed: u32) -> Result<Count, String> {
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
    let set = |c: &mut CDec, c16: &mut CDec, d: &mut OpusDecoder, d16: &mut OpusDecoder, value| {
        c.set_complexity(value);
        c16.set_complexity(value);
        d.set_complexity(value as u8).unwrap();
        d16.set_complexity(value as u8).unwrap();
    };
    set(&mut c, &mut c16, &mut d, &mut d16, complexity);
    let mut out = vec![0f32; rate * channels];
    let mut out16 = vec![0i16; rate * channels];
    let mut count = Count::default();
    for (index, call) in calls(packets, rate, loss, seed).into_iter().enumerate() {
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
                set(&mut c, &mut c16, &mut d, &mut d16, value);
                continue;
            }
        };
        count.calls += 1;
        let at = format!("call {index} ({call:?})");
        let (want, got) = match (want, got) {
            (Ok(w), Ok(n)) => (w, &out[..n * channels]),
            (w, g) => return Err(format!("{at}: libopus {:?}, opus1 {:?}", w.map(|v| v.len()), g)),
        };
        if let Some(what) = first_difference(&want, got, |a: f32, b: f32| a.to_bits() == b.to_bits()) {
            return Err(format!("{at}: float output, {what}"));
        }
        let (want16, got16) = match (want16, got16) {
            (Ok(w), Ok(n)) => (w, &out16[..n * channels]),
            (w, g) => return Err(format!("{at}: 16 bits: libopus {:?}, opus1 {:?}", w.map(|v| v.len()), g)),
        };
        if let Some(what) = first_difference(&want16, got16, |a: i16, b: i16| a == b) {
            return Err(format!("{at}: 16-bit output, {what}"));
        }
        if (c.final_range(), c16.final_range()) != (d.final_range(), d16.final_range()) {
            return Err(format!("{at}: final range"));
        }
    }
    Ok(count)
}

/// The complexities and losses of one decoding run: every loss at
/// complexity 5, a few at the others. With `OPUS1_DNN_PLC_ALL`, every loss
/// at every complexity.
fn plan(complexities: &[i32], losses: &[Loss]) -> Vec<(i32, Loss)> {
    let mut out = Vec::new();
    for &c in complexities {
        for &loss in losses {
            let wanted = all()
                || c == 5
                || matches!((c, loss), (0, Loss::Single | Loss::Burst | Loss::Fec) | (4, Loss::Single) | (6 | 10, Loss::Single | Loss::Long));
            if wanted {
                out.push((c, loss));
            }
        }
    }
    out
}

/// Decodes every stream at the output rates, channel counts,
/// complexities and losses, on all cores. Panics with the list of runs
/// that differ.
fn decode_streams(streams: &[Stream], rates: &[usize], losses: &[Loss]) {
    let start = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let results: Vec<(usize, Count, Vec<String>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let (mut runs, mut total) = (0, Count::default());
                    let mut failures = Vec::new();
                    for stream in streams.iter().skip(t).step_by(threads) {
                        // The losses depend on the stream's name, not on its
                        // place in the list, so a run with
                        // `OPUS1_DNN_PLC_STREAMS` repeats them.
                        let name = stream.label.bytes().fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
                        for &rate in rates {
                            for channels in [1, 2] {
                                for (complexity, loss) in plan(&complexities(), losses) {
                                    runs += 1;
                                    let seed = name ^ (rate + channels * 31 + complexity as usize * 131) as u32;
                                    match compare(&stream.packets, rate, channels, complexity, loss, seed) {
                                        Ok(count) => {
                                            total.calls += count.calls;
                                            total.concealed += count.concealed;
                                        }
                                        Err(e) => failures.push(format!(
                                            "{}, {rate} Hz, {channels} ch, complexity {complexity}, {loss:?}: {e}",
                                            stream.label
                                        )),
                                    }
                                }
                            }
                        }
                    }
                    (runs, total, failures)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    let runs: usize = results.iter().map(|r| r.0).sum();
    let calls: usize = results.iter().map(|r| r.1.calls).sum();
    let concealed: usize = results.iter().map(|r| r.1.concealed).sum();
    let failures: Vec<&String> = results.iter().flat_map(|r| &r.2).collect();
    println!(
        "{} streams, {runs} runs, {calls} decode calls ({concealed} concealed or rebuilt), {} differ, {:.1} s",
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

/// Every stream at 16 and 48 kHz output with every loss pattern, or with
/// `OPUS1_DNN_PLC_ALL` at all five output rates.
#[test]
fn streams_match_libopus() {
    let streams = streams(if all() { 8.0 } else { 4.0 });
    let rates: &[usize] = if all() { &[8000, 12000, 16000, 24000, 48000] } else { &[16000, 48000] };
    decode_streams(&streams, rates, &LOSSES);
}

/// The other output rates, 8, 12 and 24 kHz, with fewer loss patterns.
#[test]
fn other_output_rates_match_libopus() {
    if all() {
        return; // `streams_match_libopus` covers them.
    }
    let streams = streams(3.0);
    decode_streams(&streams, &[8000, 12000, 24000], &[Loss::Single, Loss::Long, Loss::Fec]);
}

/// The test above would pass if deep PLC never ran: libopus at complexity
/// 5 must differ from libopus at complexity 0 on lost SILK wideband and
/// CELT frames, and on a switch without redundancy without loss.
#[test]
fn deep_plc_changes_the_output_of_libopus() {
    let pcm = input(48000, 1, 3.0, 0);
    let silk = c_encode(&Coding::new("", 48000, 1, Some(MODE_SILK), Some(BW_WB)), &pcm);
    let celt = c_encode(&Coding::new("", 48000, 1, Some(MODE_CELT), Some(BW_FB)), &pcm);
    let spliced = splice(&silk, &celt, 13);
    for (label, packets, loss) in [("SILK", &silk, Loss::Single), ("CELT", &celt, Loss::Single), ("switch", &spliced, Loss::None)] {
        let decode = |complexity| {
            let mut c = CDec::new(48000, 1);
            c.set_complexity(complexity);
            let mut out = Vec::new();
            for call in calls(packets, 48000, loss, 1) {
                out.extend(match call {
                    Call::Packet(k) => c.decode(Some(&packets[k]), 5760, false).unwrap(),
                    Call::Conceal(n) => c.decode(None, n, false).unwrap(),
                    _ => unreachable!(),
                });
            }
            out
        };
        let (plain, deep) = (decode(0), decode(5));
        let differ = plain.iter().zip(&deep).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("{label}: {differ} of {} samples differ between complexity 0 and 5", plain.len());
        assert!(differ > 1000, "{label}: deep PLC does not change libopus's output");
    }
}

/// A 5.1 multistream decoder (4 streams, 2 of them coupled) against
/// libopus, with losses, FEC and resets, at complexity 0 and 5, for
/// packets of libopus's and opus1's multistream encoders.
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
    for (bitrate, fec) in [(96000, false), (160000, true)] {
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
        for rate in [16000usize, 48000] {
            for complexity in [0, 5] {
                for loss in [Loss::Single, Loss::Burst, Loss::Fec, Loss::Reset] {
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
                    d.set_complexity(complexity as u8).unwrap();
                    d16.set_complexity(complexity as u8).unwrap();
                    let mut out = vec![0f32; 5760 * channels];
                    let mut out16 = vec![0i16; 5760 * channels];
                    for (index, call) in calls(packets, rate, loss, 3).into_iter().enumerate() {
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
                            Call::Complexity(_) => unreachable!(),
                        };
                        let (want, want16, n, n16) = (want.unwrap(), want16.unwrap(), got.unwrap(), got16.unwrap());
                        let diff = first_difference(&want, &out[..n * channels], |a: f32, b: f32| a.to_bits() == b.to_bits())
                            .or_else(|| first_difference(&want16, &out16[..n16 * channels], |a: i16, b: i16| a == b))
                            .or_else(|| {
                                (matches!(call, Call::Packet(_)) && c.final_range() != d.final_range())
                                    .then(|| "final range".to_string())
                            });
                        if let Some(what) = diff {
                            failures.push(format!("{label}, {rate} Hz, complexity {complexity}, {loss:?}, call {index} ({call:?}): {what}"));
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

/// The time of decode calls at complexity 5 for opus1 and for the linked
/// libopus, mono 48 kHz output: the first lost packet of a loss after 22
/// good ones (deep PLC analyses its whole history first), the next two
/// lost packets, and the packets that arrive. Medians over 5 runs of 10 s
/// of speech, in microseconds. Run it with `--ignored --nocapture
/// --test-threads=1`.
#[test]
#[ignore = "prints timings"]
fn timing() {
    let pcm = input(48000, 1, 10.0, 0);
    let codings = [
        ("SILK WB 20 ms", Coding::new("", 48000, 1, Some(MODE_SILK), Some(BW_WB)).bitrates(&[20000])),
        ("CELT FB 20 ms", Coding::new("", 48000, 1, Some(MODE_CELT), Some(BW_FB)).bitrates(&[48000])),
        ("CELT FB 10 ms", Coding::new("", 48000, 1, Some(MODE_CELT), Some(BW_FB)).durations(&[4]).bitrates(&[48000])),
    ];
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    println!("| Stream | Call | opus1 (µs) | libopus (µs) |");
    println!("|---|---|---:|---:|");
    for (label, coding) in codings {
        let packets = c_encode(&coding, &pcm);
        let lost = |k: usize| k % 25 >= 22;
        // (first lost, further lost, good) per run.
        let mut ours = [Vec::new(), Vec::new(), Vec::new()];
        let mut theirs = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..5 {
            let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
            d.set_complexity(5).unwrap();
            let mut c = CDec::new(48000, 1);
            c.set_complexity(5);
            let mut out = vec![0f32; 960];
            let mut t_ours = [Vec::new(), Vec::new(), Vec::new()];
            let mut t_theirs = [Vec::new(), Vec::new(), Vec::new()];
            for (k, p) in packets.iter().enumerate() {
                let n = opus1::packet::nb_samples(p, SampleRate::Hz48000).unwrap();
                let kind = if !lost(k) { 2 } else if k % 25 == 22 { 0 } else { 1 };
                let start = std::time::Instant::now();
                if kind == 2 {
                    d.decode(p, n, &mut out).unwrap();
                } else {
                    d.decode(&[], n, &mut out).unwrap();
                }
                t_ours[kind].push(start.elapsed().as_secs_f64() * 1e6);
                let start = std::time::Instant::now();
                c.decode(if kind == 2 { Some(p) } else { None }, n, false).unwrap();
                t_theirs[kind].push(start.elapsed().as_secs_f64() * 1e6);
            }
            for i in 0..3 {
                ours[i].push(median(t_ours[i].clone()));
                theirs[i].push(median(t_theirs[i].clone()));
            }
        }
        for (i, call) in ["first lost packet", "further lost packet", "packet that arrived"].iter().enumerate() {
            println!("| {label} | {call} | {:.1} | {:.1} |", median(ours[i].clone()), median(theirs[i].clone()));
        }
    }
}

/// With `OPUS1_DNN_REF_O2=1`, against a libopus built at `-O2` with loop
/// vectorization (`target/android/libopus-host-1.6.1-dnn`), whose deep PLC
/// rounds differently: measures how far the concealed audio, and the
/// audio decoded after it, are from libopus's, as a signal-to-noise ratio
/// (SNR) over all streams, mono at 48 kHz and complexity 5, with single
/// losses, bursts and long losses. It requires at least 8 dB over the
/// concealed audio, about the distance between libopus builds, and that
/// in each run the loudest concealed call is at most 6 dB louder than
/// libopus's loudest (no explosion). Single calls are not compared by
/// loudness: the two concealments go their own ways, and one can be
/// quiet where the other is not.
/// Without the variable the test does nothing.
#[test]
fn distance_to_an_o2_build() {
    if std::env::var_os("OPUS1_DNN_REF_O2").is_none_or(|v| v == "0") {
        return;
    }
    let energy = |x: &[f32]| x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    let error = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>();
    let db = |sig: f64, err: f64| if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() };
    let streams = streams(if all() { 8.0 } else { 4.0 });
    // Signal and error energy over concealed calls and over the packets
    // that arrived; the SNR of each run with concealed audio.
    let (mut lost_sig, mut lost_err, mut good_sig, mut good_err) = (0f64, 0f64, 0f64, 0f64);
    let mut run_snrs = Vec::new();
    let (mut calls_differ, mut concealed_calls, mut loudest) = (0, 0, f64::NEG_INFINITY);
    for stream in &streams {
        for loss in [Loss::Single, Loss::Burst, Loss::Long] {
            let mut c = CDec::new(48000, 1);
            c.set_complexity(5);
            let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
            d.set_complexity(5).unwrap();
            let mut out = vec![0f32; 48000];
            let (mut run_sig, mut run_err) = (0f64, 0f64);
            let (mut run_loudest_want, mut run_loudest_got) = (0f64, 0f64);
            for call in calls(&stream.packets, 48000, loss, 7) {
                let (want, n, lost) = match call {
                    Call::Packet(k) => {
                        let p = &stream.packets[k];
                        let n = opus1::packet::nb_samples(p, SampleRate::Hz48000).unwrap();
                        (c.decode(Some(p), n, false).unwrap(), d.decode(p, n, &mut out).unwrap(), false)
                    }
                    Call::Conceal(n) => (c.decode(None, n, false).unwrap(), d.decode(&[], n, &mut out).unwrap(), true),
                    _ => unreachable!(),
                };
                let got = &out[..n];
                let (sig, err) = (energy(&want), error(&want, got));
                if lost {
                    concealed_calls += 1;
                    calls_differ += (err > 0.0) as usize;
                    run_sig += sig;
                    run_err += err;
                    // Mean energy per sample of the call.
                    run_loudest_want = run_loudest_want.max(sig / n as f64);
                    run_loudest_got = run_loudest_got.max(energy(got) / n as f64);
                } else {
                    good_sig += sig;
                    good_err += err;
                }
            }
            lost_sig += run_sig;
            lost_err += run_err;
            // Ignore runs whose concealment stays below -60 dB of full scale.
            let floor = 1e-6;
            loudest = f64::max(loudest, 10.0 * ((run_loudest_got + floor) / (run_loudest_want + floor)).log10());
            if run_err > 0.0 {
                run_snrs.push(db(run_sig, run_err));
            }
        }
    }
    run_snrs.sort_by(|a, b| a.total_cmp(b));
    let lost_snr = db(lost_sig, lost_err);
    println!("{} streams, {} runs that differ, {calls_differ} of {concealed_calls} concealed calls differ", streams.len(), run_snrs.len());
    println!("SNR over concealed audio: {lost_snr:.1} dB; over the packets that arrived: {:.1} dB", db(good_sig, good_err));
    if !run_snrs.is_empty() {
        println!(
            "SNR of single runs: lowest {:.1} dB, median {:.1} dB, highest {:.1} dB",
            run_snrs[0],
            run_snrs[run_snrs.len() / 2],
            run_snrs[run_snrs.len() - 1]
        );
    }
    println!("loudest concealed call of a run against libopus's loudest: at most {loudest:+.1} dB");
    assert!(lost_snr >= 8.0, "concealed audio {lost_snr:.1} dB from the -O2 build");
    assert!(loudest <= 6.0, "a run's loudest concealed call is {loudest:.1} dB above the -O2 build's");
}
