//! Packets whose channel count differs from the decoder's, against C
//! libopus.
//!
//! libopus decodes any stream with any decoder channel count
//! (`opus_decode_frame` in opus_decoder.c): a mono packet is written to both
//! channels of a stereo decoder, and a stereo packet is mixed down by a mono
//! decoder. The count may change inside a stream: libopus's own encoder codes
//! stereo input as mono at low bitrates.
//!
//! The rules under test:
//!
//! - CELT is told the packet's channel count before every frame
//!   (`CELT_SET_CHANNELS(st->stream_channels)`, opus_decoder.c:581).
//! - SILK gets the decoder's count and the packet's count with every frame
//!   (`nChannelsAPI`, `nChannelsInternal`, opus_decoder.c:424-442).
//! - A lost frame is concealed by SILK with the channel count and the rate
//!   of the last frame that carried data, whatever a placeholder packet in
//!   between said. CELT conceals in the decoder's own channel count.
//!
//! Every call is compared with libopus: the samples and the final range.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{chirp, full_scale, multitone, speech_fullband, to_stereo, toc};
use opus1::OpusDecoder;

/// Output rates a decoder can have.
const DEC_RATES: [usize; 5] = [8000, 12000, 16000, 24000, 48000];
/// The input rate of every encoder here. Any bandwidth can be coded from it.
const ENC_RATE: usize = 48000;

/// A libopus decoder and an opus1 decoder that get the same calls.
struct Pair {
    c: CDecoder,
    rs: OpusDecoder,
    rate: usize,
    channels: usize,
    /// Lowest SNR of any call, in dB, over the calls with a signal.
    worst: f64,
    /// Calls after which the two final ranges differ.
    range_mismatches: usize,
    calls: usize,
}

impl Pair {
    fn new(rate: usize, channels: usize) -> Self {
        Self {
            c: CDecoder::new(rate, channels),
            rs: OpusDecoder::new(rate as i32, channels).unwrap(),
            rate,
            channels,
            worst: 200.0,
            range_mismatches: 0,
            calls: 0,
        }
    }

    /// Samples per channel of `tenth_ms` tenths of a millisecond.
    fn samples(&self, tenth_ms: u32) -> usize {
        self.rate * tenth_ms as usize / 10_000
    }

    fn compare(&mut self, what: &str, expected: Result<Vec<f32>, i32>, got: Result<usize, &'static str>, buf: &[f32]) -> Vec<f32> {
        let expected = expected.unwrap_or_else(|e| panic!("{what}: libopus error {e}"));
        let n = got.unwrap_or_else(|e| panic!("{what}: opus1 error: {e}"));
        assert_eq!(n * self.channels, expected.len(), "{what}: number of samples");
        let ours = &buf[..n * self.channels];
        // A call that libopus leaves silent has no SNR. It must be silent
        // here too.
        if expected.iter().all(|v| *v == 0.0) {
            assert!(ours.iter().all(|v| *v == 0.0), "{what}: libopus is silent, opus1 is not");
        } else {
            self.worst = self.worst.min(snr(&expected, ours));
        }
        self.range_mismatches += (self.c.final_range() != self.rs.final_range()) as usize;
        self.calls += 1;
        expected
    }

    /// Decode a packet. Returns libopus's output.
    fn decode(&mut self, packet: &[u8]) -> Vec<f32> {
        // Room for the longest packet, 120 ms.
        let room = self.samples(1200);
        let mut buf = vec![0f32; room * self.channels];
        let expected = self.c.decode(Some(packet), room, false);
        let got = self.rs.decode(packet, room, &mut buf);
        self.compare("decode", expected, got, &buf)
    }

    /// Conceal a loss of `tenth_ms` tenths of a millisecond.
    fn lose(&mut self, tenth_ms: u32) -> Vec<f32> {
        let n = self.samples(tenth_ms);
        let mut buf = vec![0f32; n * self.channels];
        let expected = self.c.decode(None, n, false);
        let got = self.rs.decode(&[], n, &mut buf);
        self.compare("loss", expected, got, &buf)
    }

    /// Recover a loss of `tenth_ms` from the packet that followed it.
    fn recover(&mut self, next: &[u8], tenth_ms: u32) -> Vec<f32> {
        let n = self.samples(tenth_ms);
        let mut buf = vec![0f32; n * self.channels];
        let expected = self.c.decode(Some(next), n, true);
        let got = self.rs.decode_fec(next, n, &mut buf);
        self.compare("FEC", expected, got, &buf)
    }
}

/// Stereo test input: speech, and a delayed, quieter copy on the right.
fn stereo_speech(seconds: f64) -> Vec<f32> {
    to_stereo(&speech_fullband(ENC_RATE, seconds), ENC_RATE)
}

/// How many packets of a stream are stereo, and how often the channel count
/// changes from one packet to the next.
fn channel_counts(packets: &[Vec<u8>]) -> (usize, usize, usize) {
    let stereo: Vec<bool> = packets.iter().map(|p| toc(p[0]).stereo).collect();
    let changes = stereo.windows(2).filter(|w| w[0] != w[1]).count();
    let n_stereo = stereo.iter().filter(|s| **s).count();
    (packets.len() - n_stereo, n_stereo, changes)
}

/// One kind of stream: the mode and bandwidth the C encoder is forced to.
#[derive(Clone, Copy)]
struct Kind {
    name: &'static str,
    mode: i32,
    bandwidth: i32,
    /// Frame duration in tenths of a millisecond.
    frame_tenth_ms: u32,
    /// Bitrate per coded channel.
    bitrate: i32,
    /// The lowest SNR a call may have against libopus, in dB. SILK-only
    /// output is the same integer samples in both decoders. With a CELT
    /// layer the two float decoders agree to rounding.
    limit_db: f64,
}

const SILK_EXACT_DB: f64 = 150.0;
const CELT_DB: f64 = 100.0;

const KINDS: [Kind; 12] = [
    Kind { name: "SILK-NB 20 ms", mode: MODE_SILK, bandwidth: BW_NB, frame_tenth_ms: 200, bitrate: 14000, limit_db: SILK_EXACT_DB },
    Kind { name: "SILK-MB 10 ms", mode: MODE_SILK, bandwidth: BW_MB, frame_tenth_ms: 100, bitrate: 20000, limit_db: SILK_EXACT_DB },
    Kind { name: "SILK-WB 20 ms", mode: MODE_SILK, bandwidth: BW_WB, frame_tenth_ms: 200, bitrate: 24000, limit_db: SILK_EXACT_DB },
    Kind { name: "SILK-WB 40 ms", mode: MODE_SILK, bandwidth: BW_WB, frame_tenth_ms: 400, bitrate: 24000, limit_db: SILK_EXACT_DB },
    Kind { name: "SILK-NB 60 ms", mode: MODE_SILK, bandwidth: BW_NB, frame_tenth_ms: 600, bitrate: 14000, limit_db: SILK_EXACT_DB },
    Kind { name: "Hybrid-SWB 20 ms", mode: MODE_HYBRID, bandwidth: BW_SWB, frame_tenth_ms: 200, bitrate: 32000, limit_db: CELT_DB },
    Kind { name: "Hybrid-FB 10 ms", mode: MODE_HYBRID, bandwidth: BW_FB, frame_tenth_ms: 100, bitrate: 40000, limit_db: CELT_DB },
    Kind { name: "CELT-NB 20 ms", mode: MODE_CELT, bandwidth: BW_NB, frame_tenth_ms: 200, bitrate: 24000, limit_db: CELT_DB },
    Kind { name: "CELT-WB 10 ms", mode: MODE_CELT, bandwidth: BW_WB, frame_tenth_ms: 100, bitrate: 40000, limit_db: CELT_DB },
    Kind { name: "CELT-SWB 5 ms", mode: MODE_CELT, bandwidth: BW_SWB, frame_tenth_ms: 50, bitrate: 56000, limit_db: CELT_DB },
    Kind { name: "CELT-FB 2.5 ms", mode: MODE_CELT, bandwidth: BW_FB, frame_tenth_ms: 25, bitrate: 80000, limit_db: CELT_DB },
    Kind { name: "CELT-FB 20 ms", mode: MODE_CELT, bandwidth: BW_FB, frame_tenth_ms: 200, bitrate: 64000, limit_db: CELT_DB },
];

impl Kind {
    fn frame(&self) -> usize {
        ENC_RATE * self.frame_tenth_ms as usize / 10_000
    }

    /// A C encoder for stereo input, forced to this mode and bandwidth.
    fn encoder(&self, application: i32) -> CEncoder {
        let mut enc = CEncoder::new(ENC_RATE, 2, application);
        enc.force_mode(self.mode).bandwidth(self.bandwidth);
        enc
    }
}

/// Streams that keep one channel count, in decoders of either count, at
/// every output rate.
#[test]
fn mono_and_stereo_streams_decode_in_either_decoder() {
    let pcm = stereo_speech(0.8);
    let mut failures = Vec::new();
    let mut streams = 0;
    for kind in &KINDS {
        let mut lowest = 200f64;
        for stream_ch in [1usize, 2] {
            let mut enc = kind.encoder(APP_AUDIO);
            enc.force_channels(stream_ch as i32).bitrate(kind.bitrate * stream_ch as i32);
            let packets = enc.encode_all(&pcm, kind.frame());
            let (mono, stereo, _) = channel_counts(&packets);
            // The first packets of a stream that is forced to mono can still
            // be stereo, never the other way round.
            assert_eq!(if stream_ch == 1 { stereo } else { mono }, 0, "{}: {mono} mono and {stereo} stereo packets", kind.name);
            assert!(packets.iter().all(|p| p.len() > 2), "{}: empty packets", kind.name);
            for dec_ch in [1usize, 2] {
                for rate in DEC_RATES {
                    let mut pair = Pair::new(rate, dec_ch);
                    for p in &packets {
                        pair.decode(p);
                    }
                    streams += 1;
                    lowest = lowest.min(pair.worst);
                    if pair.worst < kind.limit_db || pair.range_mismatches != 0 {
                        failures.push(format!(
                            "{}, {stream_ch}-channel stream, {dec_ch}-channel decoder at {rate} Hz: worst call {:.1} dB, final range differs after {} of {} calls",
                            kind.name, pair.worst, pair.range_mismatches, pair.calls
                        ));
                    }
                }
            }
        }
        println!("{}: worst call of 20 streams {lowest:.1} dB against libopus", kind.name);
    }
    assert!(failures.is_empty(), "{} of {streams} streams fail:\n{}", failures.len(), failures.join("\n"));
}

/// Encode stereo input with the forced channel count changing every
/// `period` packets.
fn alternating_stream(kind: &Kind, pcm: &[f32], period: usize, fec: bool) -> Vec<Vec<u8>> {
    let mut enc = kind.encoder(APP_VOIP);
    enc.fec(fec, if fec { 20 } else { 0 });
    let mut packets = Vec::new();
    for (i, frame) in pcm.chunks_exact(kind.frame() * 2).enumerate() {
        let ch = if (i / period) % 2 == 0 { 2 } else { 1 };
        enc.force_channels(ch).bitrate(kind.bitrate * ch);
        packets.push(enc.encode(frame));
    }
    packets
}

/// Streams of one mode whose channel count changes again and again.
#[test]
fn the_channel_count_may_change_inside_a_stream() {
    let pcm = stereo_speech(1.6);
    let mut failures = Vec::new();
    let mut streams = 0;
    for kind in &KINDS {
        // A change about every 140 ms.
        let period = (1400 / kind.frame_tenth_ms).max(2) as usize;
        let packets = alternating_stream(kind, &pcm, period, false);
        let (mono, stereo, changes) = channel_counts(&packets);
        assert!(mono >= 3 && stereo >= 3 && changes >= 4, "{}: {mono} mono, {stereo} stereo packets, {changes} changes", kind.name);
        let mut lowest = 200f64;
        for dec_ch in [1usize, 2] {
            for rate in DEC_RATES {
                let mut pair = Pair::new(rate, dec_ch);
                for p in &packets {
                    pair.decode(p);
                }
                streams += 1;
                lowest = lowest.min(pair.worst);
                if pair.worst < kind.limit_db || pair.range_mismatches != 0 {
                    failures.push(format!(
                        "{}, {changes} changes, {dec_ch}-channel decoder at {rate} Hz: worst call {:.1} dB, final range differs after {} of {} calls",
                        kind.name, pair.worst, pair.range_mismatches, pair.calls
                    ));
                }
            }
        }
        println!("{}: {mono} mono and {stereo} stereo packets, {changes} changes: worst call of 10 streams {lowest:.1} dB against libopus", kind.name);
    }
    assert!(failures.is_empty(), "{} of {streams} streams fail:\n{}", failures.len(), failures.join("\n"));
}

/// Bitrates in bits per second, one for every 200 ms. The low ones make
/// libopus code stereo input as mono.
const BITRATES: [i32; 12] = [
    8_000, 96_000, 12_000, 40_000, 10_000, 24_000, 64_000, 14_000, 32_000, 9_000, 128_000, 18_000,
];

/// libopus left to itself: it chooses mode, bandwidth and channel count
/// from the bitrate, which changes every 200 ms.
#[test]
fn streams_in_which_libopus_chooses_the_channel_count() {
    let pcm = stereo_speech(4.8);
    let mut failures = Vec::new();
    for (application, app_name) in [(APP_VOIP, "VoIP"), (APP_AUDIO, "audio")] {
        for frame_tenth_ms in [200u32, 100] {
            let frame = ENC_RATE * frame_tenth_ms as usize / 10_000;
            let mut enc = CEncoder::new(ENC_RATE, 2, application);
            let mut packets = Vec::new();
            for (i, f) in pcm.chunks_exact(frame * 2).enumerate() {
                let step = i * frame_tenth_ms as usize / 2000;
                enc.bitrate(BITRATES[step % BITRATES.len()]);
                packets.push(enc.encode(f));
            }
            let (mono, stereo, changes) = channel_counts(&packets);
            let mut modes: Vec<&str> = packets.iter().map(|p| toc(p[0]).mode).collect();
            modes.sort_unstable();
            modes.dedup();
            let name = format!("{app_name}, {} ms: {mono} mono and {stereo} stereo packets, {changes} changes, modes {modes:?}", frame_tenth_ms / 10);
            assert!(mono >= 20 && stereo >= 20 && changes >= 6, "{name}");
            assert!(modes.len() >= 2, "{name}");
            let mut lowest = 200f64;
            for dec_ch in [1usize, 2] {
                for rate in DEC_RATES {
                    let mut pair = Pair::new(rate, dec_ch);
                    for p in &packets {
                        pair.decode(p);
                    }
                    lowest = lowest.min(pair.worst);
                    if pair.worst < CELT_DB || pair.range_mismatches != 0 {
                        failures.push(format!(
                            "{name}; {dec_ch}-channel decoder at {rate} Hz: worst call {:.1} dB, final range differs after {} of {} calls",
                            pair.worst, pair.range_mismatches, pair.calls
                        ));
                    }
                }
            }
            println!("{name}: worst call of 10 streams {lowest:.1} dB against libopus");
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Losses in streams whose channel count changes. One loss in six is
/// concealed, the next one recovered from the FEC data of the packet after
/// it, and so on. SILK-only and Hybrid streams, where both decoders conceal
/// alike.
#[test]
fn losses_and_fec_while_the_channel_count_changes() {
    let pcm = stereo_speech(2.4);
    let mut failures = Vec::new();
    let mut streams = 0;
    for kind in KINDS.iter().filter(|k| k.mode != MODE_CELT) {
        // Changes fall on other packets than the losses do.
        let period = (1000 / kind.frame_tenth_ms).max(2) as usize;
        let packets = alternating_stream(kind, &pcm, period, true);
        let (mono, stereo, changes) = channel_counts(&packets);
        assert!(mono >= 3 && stereo >= 3 && changes >= 4, "{}: {mono} mono, {stereo} stereo packets, {changes} changes", kind.name);
        let mut lowest = 200f64;
        for dec_ch in [1usize, 2] {
            for rate in DEC_RATES {
                // Every phase of the loss pattern, so that losses fall
                // before, on and after the changes of the channel count.
                for phase in 0..6 {
                    let mut pair = Pair::new(rate, dec_ch);
                    let (mut concealed, mut recovered) = (0, 0);
                    for (i, p) in packets.iter().enumerate() {
                        let lost = i >= 3 && i % 6 == phase && i + 1 < packets.len();
                        if !lost {
                            pair.decode(p);
                        } else if (i / 6) % 2 == 0 {
                            pair.lose(kind.frame_tenth_ms);
                            concealed += 1;
                        } else {
                            pair.recover(&packets[i + 1], kind.frame_tenth_ms);
                            recovered += 1;
                        }
                    }
                    assert!(concealed >= 2 && recovered >= 2, "{}: {concealed} concealed, {recovered} recovered", kind.name);
                    streams += 1;
                    lowest = lowest.min(pair.worst);
                    if pair.worst < kind.limit_db || pair.range_mismatches != 0 {
                        failures.push(format!(
                            "{}, losses at {phase} of 6, {dec_ch}-channel decoder at {rate} Hz: worst call {:.1} dB, final range differs after {} of {} calls",
                            kind.name, pair.worst, pair.range_mismatches, pair.calls
                        ));
                    }
                }
            }
        }
        println!("{}: {changes} changes of the channel count: worst call of 60 streams {lowest:.1} dB against libopus", kind.name);
    }
    assert!(failures.is_empty(), "{} of {streams} streams fail:\n{}", failures.len(), failures.join("\n"));
}

/// The first byte of a packet with one frame: SILK, Hybrid or CELT at a
/// bandwidth, 20 ms (10 ms where `ten_ms`), mono or stereo.
fn first_byte(mode: i32, bandwidth: i32, ten_ms: bool, stereo: bool) -> u8 {
    let config = match (mode, bandwidth) {
        (MODE_SILK, BW_NB) => 1 - ten_ms as u8,
        (MODE_SILK, BW_MB) => 5 - ten_ms as u8,
        (MODE_SILK, _) => 9 - ten_ms as u8,
        (MODE_HYBRID, BW_SWB) => 13 - ten_ms as u8,
        (MODE_HYBRID, _) => 15 - ten_ms as u8,
        (_, BW_NB) => 19 - ten_ms as u8,
        (_, BW_WB) => 23 - ten_ms as u8,
        (_, BW_SWB) => 27 - ten_ms as u8,
        _ => 31 - ten_ms as u8,
    };
    (config << 3) | ((stereo as u8) << 2)
}

/// A packet that is only its first byte stands for a lost frame, as does a
/// frame of one byte. libopus still takes the mode, the bandwidth, the
/// frame size and the channel count from that first byte
/// (`opus_decode_native`), and then:
///
/// - SILK conceals with the channel count and rate of the last frame that
///   carried data (opus_decoder.c:424-442 are skipped without data).
/// - The concealment runs in the mode of the last frame, not in the mode
///   the placeholder names.
/// - FEC decoding goes by the placeholder's mode: after a CELT-only
///   placeholder the next FEC call conceals (opus_decoder.c:814).
/// - A piece of concealment is at most as long as a frame of the
///   placeholder (opus_decoder.c:320).
///
/// Each placeholder here says something else than the stream: the other
/// channel count, another mode, another bandwidth, another frame size.
#[test]
fn placeholder_packets_do_not_change_what_silk_conceals_with() {
    let pcm = stereo_speech(1.5);
    let mut failures = Vec::new();
    let mut streams = 0;
    let stream_kinds = [KINDS[0], KINDS[2], KINDS[5], KINDS[6]];
    for kind in &stream_kinds {
        let mut lowest = 200f64;
        for stream_ch in [1usize, 2] {
            let mut enc = kind.encoder(APP_VOIP);
            enc.force_channels(stream_ch as i32).bitrate(kind.bitrate * stream_ch as i32).fec(true, 20);
            let packets = enc.encode_all(&pcm, kind.frame());
            let other = stream_ch == 1;
            // (what the placeholder says, its first byte)
            let placeholders = [
                ("the other channel count", first_byte(kind.mode, kind.bandwidth, kind.frame_tenth_ms == 100, other)),
                ("SILK-NB 10 ms, other channel count", first_byte(MODE_SILK, BW_NB, true, other)),
                ("SILK-WB 20 ms, other channel count", first_byte(MODE_SILK, BW_WB, false, other)),
                ("Hybrid-FB 20 ms, other channel count", first_byte(MODE_HYBRID, BW_FB, false, other)),
                ("Hybrid-SWB 10 ms, same channel count", first_byte(MODE_HYBRID, BW_SWB, true, !other)),
                ("CELT-FB 20 ms, other channel count", first_byte(MODE_CELT, BW_FB, false, other)),
                ("CELT-NB 10 ms, same channel count", first_byte(MODE_CELT, BW_NB, true, !other)),
            ];
            for (what, byte) in placeholders {
                let placeholder_tenth_ms = toc(byte).frame_tenth_ms;
                for dec_ch in [1usize, 2] {
                    for rate in [8000usize, 16000, 48000] {
                        let mut pair = Pair::new(rate, dec_ch);
                        for (i, p) in packets.iter().enumerate() {
                            match i % 12 {
                                // The placeholder, as a packet of one byte.
                                5 => {
                                    pair.decode(&[byte]);
                                }
                                // A loss right after it, longer than its
                                // frame, then FEC for the packet after.
                                6 => {
                                    pair.lose(400);
                                }
                                7 if i + 1 < packets.len() => {
                                    pair.recover(&packets[i + 1], kind.frame_tenth_ms);
                                }
                                // The placeholder as a frame of one byte in a
                                // packet of two frames (code 2: the size of
                                // the first frame, then both frames).
                                10 => {
                                    pair.decode(&[byte | 2, 1, 0x55]);
                                }
                                _ => {
                                    pair.decode(p);
                                }
                            }
                        }
                        streams += 1;
                        lowest = lowest.min(pair.worst);
                        if pair.worst < kind.limit_db || pair.range_mismatches != 0 {
                            failures.push(format!(
                                "{}, {stream_ch}-channel stream, placeholder says {what} ({} ms), {dec_ch}-channel decoder at {rate} Hz: worst call {:.1} dB, final range differs after {} of {} calls",
                                kind.name, placeholder_tenth_ms / 10, pair.worst, pair.range_mismatches, pair.calls
                            ));
                        }
                    }
                }
            }
        }
        println!("{}: worst call of 84 streams {lowest:.1} dB against libopus", kind.name);
    }
    assert!(failures.is_empty(), "{} of {streams} streams fail:\n{}", failures.len(), failures.join("\n"));
}

/// A small deterministic random number generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

/// Packets of every mode, bandwidth, frame size and channel count, whole
/// and damaged, mixed with losses and FEC calls of any duration, through a
/// libopus decoder and an opus1 decoder of either channel count.
///
/// - No call panics.
/// - Both decoders accept and reject the same calls, and give the same
///   number of samples.
/// - The output is finite.
/// - The final range is libopus's after nearly every call. The exception is
///   known: a damaged CELT frame can be read differently
///   (`RangeCoder::dec_uint` lacks a limit of libopus, see
///   tests/celtdec_robust.rs). Without it, about 1 call in 1000 differs.
#[test]
fn damaged_packets_losses_and_fec_in_any_order_follow_libopus() {
    let pcm = stereo_speech(0.6);
    // Twelve packets of every kind, mono and stereo.
    let mut pool: Vec<Vec<u8>> = Vec::new();
    for kind in &KINDS {
        for stream_ch in [1usize, 2] {
            let mut enc = kind.encoder(APP_VOIP);
            enc.force_channels(stream_ch as i32).bitrate(kind.bitrate * stream_ch as i32).fec(true, 10);
            pool.extend(enc.encode_all(&pcm, kind.frame()).into_iter().skip(2).take(12));
        }
    }
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (mut accepted, mut rejected, mut range_mismatches) = (0usize, 0usize, 0usize);
    for round in 0..300 {
        let rate = DEC_RATES[rng.below(DEC_RATES.len())];
        let channels = 1 + rng.below(2);
        let mut c_dec = CDecoder::new(rate, channels);
        let mut rs_dec = OpusDecoder::new(rate as i32, channels).unwrap();
        let room = rate * 120 / 1000;
        let mut buf = vec![0f32; room * channels];
        for step in 0..40 {
            let mut packet = pool[rng.below(pool.len())].clone();
            // A lost duration: 2.5 to 120 ms.
            let lost = rate / 400 * (1 + rng.below(48));
            let what = rng.below(12);
            match what {
                // Flipped bits.
                2 => {
                    for _ in 0..1 + rng.below(4) {
                        let at = rng.below(packet.len());
                        packet[at] ^= 1 << rng.below(8);
                    }
                }
                // Cut short.
                3 => packet.truncate(1 + rng.below(packet.len())),
                // Another payload, another first byte, random bytes of any
                // length.
                4 => packet.iter_mut().skip(1).for_each(|b| *b = rng.next() as u8),
                5 => packet[0] = rng.next() as u8,
                6 => packet = (0..1 + rng.below(1500)).map(|_| rng.next() as u8).collect(),
                // One to three bytes.
                7 => packet.truncate(1 + rng.below(3)),
                // A payload of ones or of zeros.
                8 => packet.iter_mut().skip(1).for_each(|b| *b = 0xFF),
                9 => packet.iter_mut().skip(1).for_each(|b| *b = 0),
                _ => {}
            }
            let (expected, got) = match what {
                10 => (c_dec.decode(None, lost, false), rs_dec.decode(&[], lost, &mut buf)),
                11 => (c_dec.decode(Some(&packet), lost, true), rs_dec.decode_fec(&packet, lost, &mut buf)),
                _ => (c_dec.decode(Some(&packet), room, false), rs_dec.decode(&packet, room, &mut buf)),
            };
            let call = format!("round {round}, step {step}, kind {what}, {channels}-channel decoder at {rate} Hz, first byte {:#04x}, {} bytes", packet[0], packet.len());
            match (expected, got) {
                (Ok(expected), Ok(n)) => {
                    assert_eq!(n * channels, expected.len(), "{call}: number of samples");
                    assert!(buf[..n * channels].iter().all(|v| v.is_finite()), "{call}: not finite");
                    range_mismatches += (c_dec.final_range() != rs_dec.final_range()) as usize;
                    accepted += 1;
                }
                (Err(_), Err(_)) => rejected += 1,
                (expected, got) => panic!("{call}: libopus {:?}, opus1 {got:?}", expected.map(|v| v.len())),
            }
        }
    }
    println!("{accepted} calls accepted and {rejected} rejected by both; the final range differs after {range_mismatches}");
    assert!(accepted > 8000 && rejected > 800, "{accepted} accepted, {rejected} rejected");
    assert!(range_mismatches * 200 <= accepted, "the final range differs after {range_mismatches} of {accepted} calls");
}

/// Stereo input that is hard on the conversion between channel counts.
fn hard_stereo_signals(seconds: f64) -> Vec<(&'static str, Vec<f32>)> {
    let n = (ENC_RATE as f64 * seconds) as usize;
    let speech = speech_fullband(ENC_RATE, seconds);
    let (tones, sweep, loud) = (multitone(ENC_RATE, n), chirp(ENC_RATE, n), full_scale(ENC_RATE, n));
    vec![
        // The mono mix of these two channels is silence.
        ("speech in opposite phase", speech.iter().flat_map(|&v| [v, -v]).collect()),
        ("tones in the left channel only", tones.iter().flat_map(|&v| [v, 0.0]).collect()),
        ("a sweep in the right channel only", sweep.iter().flat_map(|&v| [0.0, v]).collect()),
        ("full scale left, a sweep right", loud.iter().zip(&sweep).flat_map(|(&a, &b)| [a, b]).collect()),
        // Beyond full scale: the decoded peaks exceed 1.0, and the 16-bit
        // output goes through the soft clipper.
        ("overdriven tones in opposite phase", tones.iter().flat_map(|&v| [3.0 * v, -2.5 * v]).collect()),
    ]
}

/// The largest difference between two sample buffers.
fn largest_difference<T: Copy + Into<f64>>(a: &[T], b: &[T]) -> f64 {
    a.iter().zip(b).fold(0f64, |m, (&x, &y)| m.max((x.into() - y.into()).abs()))
}

/// Channel conversion on signals other than speech: channels in opposite
/// phase, one silent channel, unrelated channels at full scale, and input
/// beyond full scale. The channel count of the stream changes every 150 ms.
/// SILK-only and Hybrid streams also lose one packet in seven, which is
/// concealed or recovered from FEC data in turn. Both outputs are compared:
/// float, and 16 bits (which is soft-clipped).
///
/// A signal-to-noise ratio says little here: the mono mix of two opposite
/// channels is near silence. So every call is held to an absolute limit:
///
/// - SILK-only: the samples are libopus's exactly;
/// - with a CELT layer: floats within 0.00003 of full scale, or of the peak
///   of the call when that is larger. libopus's encoder overshoots a long
///   way when it is forced from stereo to mono on full-scale input in
///   2.5 ms frames (peaks of 10000 times full scale), and the two decoders
///   follow it alike;
/// - 16-bit samples within 1. Where the call reaches the limits of 16 bits
///   the soft clipper is at work, which turns a small float difference into
///   several steps: within 8 there;
/// - the final range is libopus's after every call.
#[test]
fn channel_conversion_of_hard_stereo_signals_matches_libopus() {
    let kinds = [KINDS[0], KINDS[2], KINDS[4], KINDS[5], KINDS[6], KINDS[7], KINDS[8], KINDS[10], KINDS[11]];
    let mut failures = Vec::new();
    let mut calls = 0usize;
    let (mut largest_float, mut largest_i16, mut largest_ratio) = (0f64, 0f64, 0f64);
    for (signal, pcm) in hard_stereo_signals(1.2) {
        for kind in &kinds {
            let celt_only = kind.mode == MODE_CELT;
            let period = (1500 / kind.frame_tenth_ms).max(2) as usize;
            let packets = alternating_stream(kind, &pcm, period, !celt_only);
            let (mono, stereo, changes) = channel_counts(&packets);
            assert!(mono >= 2 && stereo >= 2 && changes >= 3, "{}: {mono} mono, {stereo} stereo packets, {changes} changes", kind.name);
            for dec_ch in [1usize, 2] {
                for rate in DEC_RATES {
                    let (room, n) = (rate * 120 / 1000, rate * kind.frame_tenth_ms as usize / 10_000);
                    let exact = kind.mode == MODE_SILK;
                    // 16-bit output, then float output, each with its own
                    // pair of decoders.
                    for sixteen_bit in [true, false] {
                        let mut c_dec = CDecoder::new(rate, dec_ch);
                        let mut rs_dec = OpusDecoder::new(rate as i32, dec_ch).unwrap();
                        let (mut buf, mut buf_i16) = (vec![0f32; room * dec_ch], vec![0i16; room * dec_ch]);
                        // The largest difference, and the largest one as
                        // a multiple of the limit of its call.
                        let (mut worst, mut worst_ratio, mut bad_ranges) = (0f64, 0f64, 0);
                        for (i, p) in packets.iter().enumerate() {
                            // CELT-only concealment agrees less closely with
                            // libopus (tests/celtdec_plc.rs): no losses there.
                            let lost = !celt_only && i >= 3 && i % 7 == 3 && i + 1 < packets.len();
                            let recover = lost && (i / 7) % 2 == 1;
                            let (data, size): (&[u8], usize) = if recover {
                                (&packets[i + 1], n)
                            } else if lost {
                                (&[], n)
                            } else {
                                (p, room)
                            };
                            let c_data = if data.is_empty() { None } else { Some(data) };
                            let (difference, limit) = if sixteen_bit {
                                let expected = c_dec.decode_i16(c_data, size, recover).unwrap();
                                let got = if recover { rs_dec.decode_fec_i16(data, size, &mut buf_i16) } else { rs_dec.decode_i16(data, size, &mut buf_i16) };
                                assert_eq!(got.map(|n| n * dec_ch), Ok(expected.len()), "{signal}, {}, packet {i}", kind.name);
                                let at_the_limits = expected.iter().any(|&v| v == i16::MAX || v == i16::MIN);
                                (largest_difference(&expected, &buf_i16[..expected.len()]), if at_the_limits { 8.0 } else { 1.0 })
                            } else {
                                let expected = c_dec.decode(c_data, size, recover).unwrap();
                                let got = if recover { rs_dec.decode_fec(data, size, &mut buf) } else { rs_dec.decode(data, size, &mut buf) };
                                assert_eq!(got.map(|n| n * dec_ch), Ok(expected.len()), "{signal}, {}, packet {i}", kind.name);
                                let peak = expected.iter().fold(1f64, |m, &v| m.max(v.abs() as f64));
                                (largest_difference(&expected, &buf[..expected.len()]), 3e-5 * peak)
                            };
                            worst = worst.max(difference);
                            worst_ratio = worst_ratio.max(if exact { difference } else { difference / limit });
                            bad_ranges += (c_dec.final_range() != rs_dec.final_range()) as usize;
                            calls += 1;
                        }
                        if sixteen_bit {
                            largest_i16 = largest_i16.max(worst);
                        } else {
                            largest_float = largest_float.max(worst);
                        }
                        // For SILK-only any difference fails; otherwise one
                        // beyond the limit of its call.
                        let fails = if exact { worst_ratio > 0.0 } else { worst_ratio > 1.0 };
                        if !sixteen_bit {
                            largest_ratio = largest_ratio.max(worst_ratio);
                        }
                        if fails || bad_ranges != 0 {
                            failures.push(format!(
                                "{signal}, {}, {dec_ch}-channel decoder at {rate} Hz, {} output: largest difference {worst:e} ({worst_ratio:.2} of the limit), final range differs after {bad_ranges} calls",
                                kind.name,
                                if sixteen_bit { "16-bit" } else { "float" }
                            ));
                        }
                    }
                }
            }
        }
    }
    println!("{calls} calls: largest difference from libopus {largest_float:e} in float output, {largest_i16} in 16-bit output; the float call nearest its limit is at {largest_ratio:.2} of it");
    assert!(failures.is_empty(), "{} runs differ from libopus:\n{}", failures.len(), failures.join("\n"));
}
