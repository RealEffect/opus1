//! `OpusDecoder` entry points against C libopus: frames without data inside
//! a packet, the final range, and resetting the decoder.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{multitone, speech_fullband, to_stereo};
use opus1::{Channels, OpusDecoder, SampleRate, packet};

/// A stream from the C encoder with the mode and bandwidth forced.
fn c_stream(rate: usize, channels: usize, mode: i32, bandwidth: i32, bitrate: i32, frame: usize) -> (Vec<Vec<u8>>, Vec<u32>) {
    let mono = speech_fullband(rate, 2.0);
    let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };
    let mut enc = CEncoder::new(rate, channels, APP_VOIP);
    enc.force_mode(mode).bandwidth(bandwidth).bitrate(bitrate);
    enc.encode_all_with_ranges(&pcm, frame)
}

/// The decoder reports the final range the C encoder reports for the same
/// packet, in every mode, mono and stereo. Equal values mean that the
/// decoder read every symbol the encoder wrote.
#[test]
fn final_range_equals_the_c_encoders() {
    for (name, rate, channels, mode, bandwidth, bitrate, frame_ms) in [
        ("SILK-NB mono", 8000usize, 1usize, MODE_SILK, BW_NB, 12000, 20usize),
        ("SILK-WB mono 60 ms", 16000, 1, MODE_SILK, BW_WB, 24000, 60),
        ("SILK-WB stereo", 24000, 2, MODE_SILK, BW_WB, 40000, 20),
        ("Hybrid-SWB mono", 24000, 1, MODE_HYBRID, BW_SWB, 32000, 20),
        ("Hybrid-FB stereo 10 ms", 48000, 2, MODE_HYBRID, BW_FB, 64000, 10),
        ("CELT-FB mono", 48000, 1, MODE_CELT, BW_FB, 64000, 20),
        ("CELT-SWB stereo 5 ms", 24000, 2, MODE_CELT, BW_SWB, 96000, 5),
        ("CELT-WB mono 2.5 ms", 16000, 1, MODE_CELT, BW_WB, 64000, 2),
    ] {
        // 2 stands for 2.5 ms.
        let frame = if frame_ms == 2 { rate / 400 } else { rate * frame_ms / 1000 };
        let (packets, ranges) = c_stream(rate, channels, mode, bandwidth, bitrate, frame);
        let mut dec = OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
        )
        .unwrap();
        assert_eq!(dec.final_range(), 0, "{name}: before the first packet");
        let mut pcm = vec![0f32; frame * channels];
        let mut mismatches = 0;
        for (p, &range) in packets.iter().zip(&ranges) {
            dec.decode(p, frame, &mut pcm).unwrap();
            mismatches += (dec.final_range() != range) as usize;
        }
        assert!(packets.len() >= 30);
        assert_eq!(mismatches, 0, "{name}: final range differs on {mismatches} of {} packets", packets.len());
        // A lost packet has no final range.
        dec.decode(&[], frame, &mut pcm).unwrap();
        assert_eq!(dec.final_range(), 0, "{name}: after a lost packet");
    }
}

/// A stream whose mode changes: libopus adds a 5 ms CELT frame (the
/// "redundancy" frame) to the packets around a switch to or from CELT-only.
/// The final range of such a packet combines both frames, and the decoder
/// reports the same value as the encoder and as the C decoder.
#[test]
fn final_range_covers_redundancy_frames() {
    let rate = 48000;
    let frame = 960;
    let pcm = multitone(rate, rate * 4);
    let mut enc = CEncoder::new(rate, 1, APP_AUDIO);
    enc.bitrate(32000);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec =
        OpusDecoder::new(SampleRate::try_from(rate as i32).unwrap(), Channels::Mono).unwrap();
    let mut out = vec![0f32; frame];
    let (mut mismatches, mut modes) = (0, std::collections::BTreeSet::new());
    for (i, f) in pcm.chunks_exact(frame).enumerate() {
        // A new mode every 10 packets.
        enc.force_mode([MODE_SILK, MODE_CELT, MODE_HYBRID, MODE_CELT][(i / 10) % 4]);
        enc.bandwidth(if (i / 10) % 4 == 0 { BW_WB } else { BW_FB });
        let p = enc.encode(f);
        let range = enc.final_range();
        modes.insert(p[0] >> 3);
        c_dec.decode(Some(&p), frame, false).unwrap();
        assert_eq!(c_dec.final_range(), range, "packet {i}: libopus disagrees with itself");
        dec.decode(&p, frame, &mut out).unwrap();
        mismatches += (dec.final_range() != range) as usize;
    }
    assert!(modes.len() >= 3, "the stream changes mode: {modes:?}");
    assert_eq!(mismatches, 0, "final range differs on {mismatches} packets");
}

/// Decode `packets` with both decoders and return the lowest SNR of any
/// packet (opus1 against libopus) and the number of final range
/// mismatches.
fn compare_decoders(rate: usize, channels: usize, packets: &[Vec<u8>], max_frame: usize) -> (f64, usize) {
    let mut c_dec = CDecoder::new(rate, channels);
    let mut dec = OpusDecoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    let mut out = vec![0f32; max_frame * channels];
    let (mut worst, mut mismatches) = (200f64, 0);
    for (i, p) in packets.iter().enumerate() {
        let reference = c_dec.decode(Some(p), max_frame, false).unwrap_or_else(|e| panic!("packet {i}: libopus error {e}"));
        let n = dec.decode(p, max_frame, &mut out).unwrap_or_else(|e| panic!("packet {i}: {e}"));
        assert_eq!(n * channels, reference.len(), "packet {i}: duration");
        worst = worst.min(snr(&reference, &out[..n * channels]));
        mismatches += (c_dec.final_range() != dec.final_range()) as usize;
    }
    (worst, mismatches)
}

/// A frame of 0 or 1 bytes inside a packet stands for a lost frame
/// (libopus `opus_decode_frame`: `len <= 1`). Three frames of a SILK stream
/// go into one packet, with one of them replaced by such a frame.
#[test]
fn frames_without_data_inside_a_packet_are_concealed() {
    for (rate, channels, bandwidth, bitrate) in [(16000usize, 1usize, BW_WB, 24000), (48000, 1, BW_NB, 14000), (24000, 2, BW_WB, 40000)] {
        let frame = rate / 50;
        let (singles, _) = c_stream(rate, channels, MODE_SILK, bandwidth, bitrate, frame);
        for (gap, filler) in [(0usize, &[][..]), (1, &[][..]), (2, &[][..]), (1, &[0x5A][..]), (2, &[0xFF][..])] {
            let mut packets = Vec::new();
            for group in singles.as_chunks::<3>().0 {
                let toc = group[0][0];
                let mut frames: Vec<&[u8]> = group.iter().map(|p| &p[1..]).collect();
                frames[gap] = filler;
                let mut out = vec![0u8; 4000];
                let n = packet::write(toc, &frames, &mut out, false, false).unwrap();
                out.truncate(n);
                packets.push(out);
            }
            let (worst, mismatches) = compare_decoders(rate, channels, &packets, 3 * frame);
            println!("{rate} Hz, {channels} ch, frame {gap} replaced by {} byte(s): worst packet {worst:.1} dB against libopus", filler.len());
            assert!(worst > 90.0, "{rate} Hz, {channels} ch, gap {gap}: worst packet {worst:.1} dB against libopus");
            assert_eq!(mismatches, 0, "{rate} Hz, {channels} ch, gap {gap}: final range");
        }
    }
}

/// A packet of one byte is a table-of-contents byte without data. Frame
/// count codes 0 and 1 stand for one and two lost frames. Codes 2 and 3
/// need more bytes and are invalid.
#[test]
fn one_byte_packets_follow_libopus() {
    let rate = 16000;
    let frame = rate / 50;
    let (packets, _) = c_stream(rate, 1, MODE_SILK, BW_WB, 24000, frame);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec =
        OpusDecoder::new(SampleRate::try_from(rate as i32).unwrap(), Channels::Mono).unwrap();
    let mut out = vec![0f32; 6 * frame];
    for (i, p) in packets.iter().enumerate() {
        let toc = p[0] & 0xFC;
        let marker = [toc | (i % 4) as u8];
        let input: &[u8] = if i % 5 == 4 { &marker } else { p };
        let reference = c_dec.decode(Some(input), 6 * frame, false);
        let ours = dec.decode(input, 6 * frame, &mut out);
        match reference {
            Ok(reference) => {
                let n = ours.unwrap_or_else(|e| panic!("packet {i} ({input:02x?}): libopus accepts, opus1: {e}"));
                assert_eq!(n, reference.len(), "packet {i}");
                let s = snr(&reference, &out[..n]);
                assert!(s > 90.0, "packet {i}: {s:.1} dB against libopus");
                assert_eq!(dec.final_range(), c_dec.final_range(), "packet {i}");
            }
            Err(code) => {
                assert!(input.len() == 1 && input[0] & 3 >= 2, "packet {i}: libopus error {code}");
                assert!(ours.is_err(), "packet {i} ({input:02x?}): libopus rejects, opus1 accepts");
            }
        }
    }
}

/// After `reset_state` the decoder gives exactly what a new decoder gives.
#[test]
fn reset_state_gives_a_new_decoder() {
    for (name, rate, channels, mode, bandwidth, bitrate) in [
        ("SILK-WB", 16000usize, 1usize, MODE_SILK, BW_WB, 24000),
        ("SILK-WB stereo", 48000, 2, MODE_SILK, BW_WB, 40000),
        ("Hybrid-FB", 48000, 1, MODE_HYBRID, BW_FB, 40000),
        ("CELT-SWB stereo", 24000, 2, MODE_CELT, BW_SWB, 64000),
    ] {
        let frame = rate / 50;
        let (packets, ranges) = c_stream(rate, channels, mode, bandwidth, bitrate, frame);
        let (first, second) = packets.split_at(packets.len() / 2);
        let mut used = OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
        )
        .unwrap();
        let mut fresh = OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
        )
        .unwrap();
        let (mut a, mut b) = (vec![0f32; frame * channels], vec![0f32; frame * channels]);
        for p in first {
            used.decode(p, frame, &mut a).unwrap();
        }
        // End on a loss, so that the concealment state is not at rest either.
        used.decode(&[], frame, &mut a).unwrap();
        used.reset_state();
        assert_eq!(used.final_range(), 0, "{name}");
        assert_eq!((used.sampling_rate().hz(), used.channels().count()), (rate as u32, channels), "{name}");
        for (i, p) in second.iter().enumerate() {
            used.decode(p, frame, &mut a).unwrap();
            fresh.decode(p, frame, &mut b).unwrap();
            assert_eq!(a, b, "{name}: packet {i} after the reset");
            assert_eq!(used.final_range(), ranges[first.len() + i], "{name}: packet {i}");
        }
        // A loss right after a reset is silence, as for a new decoder.
        used.reset_state();
        used.decode(&[], frame, &mut a).unwrap();
        assert!(a.iter().all(|&v| v == 0.0), "{name}: loss after a reset");
    }
}

/// Wrong arguments are errors, not panics.
#[test]
fn bad_arguments_are_errors() {
    let (packets, _) = c_stream(16000, 1, MODE_SILK, BW_WB, 24000, 320);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut out = vec![0f32; 320];
    // Room for less than the packet holds.
    assert!(dec.decode(&packets[0], 160, &mut out).is_err());
    assert!(dec.decode(&packets[0], 320, &mut out[..319]).is_err());
    // A stereo packet for a mono decoder is no error: the decoder mixes it
    // down, as libopus does (tests/core_channels.rs).
    let (stereo, _) = c_stream(16000, 2, MODE_SILK, BW_WB, 32000, 320);
    assert_eq!(dec.decode(&stereo[0], 320, &mut out), Ok(320));
    // More than 120 ms in one packet.
    assert!(dec.decode(&[0x4B, 7, 1, 2, 3, 4, 5, 6, 7], 5760, &mut vec![0f32; 5760]).is_err());
    // None of this stops the decoder.
    assert_eq!(dec.decode(&packets[0], 320, &mut out), Ok(320));
}

/// A small deterministic random number generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// 2 to 63 random bytes, a third of them zero. With `channels` the
    /// stereo flag of the first byte is set for that count. Without, it is
    /// random like the rest.
    fn garbage(&mut self, channels: Option<usize>) -> Vec<u8> {
        let len = 2 + self.next() as usize % 62;
        let mut packet: Vec<u8> =
            (0..len).map(|_| if self.next().is_multiple_of(3) { 0 } else { self.next() as u8 }).collect();
        if let Some(channels) = channels {
            packet[0] = (packet[0] & !4) | if channels == 2 { 4 } else { 0 };
        }
        packet
    }
}

/// Random bytes as packets, each into a new decoder. In the first five
/// runs the packets say they code as many channels as the decoder has. In
/// the last four they say the other count, or either.
///
/// - opus1 and libopus accept and reject the same packets;
/// - no call panics, and what is decoded is finite;
/// - the float output is not clipped, in either decoder: as many packets
///   decode to peaks beyond ±1.0 in opus1 as in libopus;
/// - nearly all decoded packets match libopus's samples. The few that do
///   not are frames no encoder produces, where the two CELT decoders
///   differ. The limit keeps that number from growing.
#[test]
fn garbage_packets_decode_like_libopus() {
    let mut rng = Rng(99);
    for (rate, channels, packet_channels) in [
        (8000usize, 1usize, Some(1usize)),
        (16000, 1, Some(1)),
        (48000, 1, Some(1)),
        (48000, 2, Some(2)),
        (24000, 2, Some(2)),
        (48000, 2, Some(1)),
        (16000, 1, Some(2)),
        (12000, 2, None),
        (48000, 1, None),
    ] {
        let max = rate * 120 / 1000;
        let mut out = vec![0f32; max * channels];
        let (mut decoded, mut rejected, mut differ) = (0, 0, 0);
        let (mut over, mut over_c) = (0, 0);
        for round in 0..3000 {
            let packet = rng.garbage(packet_channels);
            let mut c_dec = CDecoder::new(rate, channels);
            let mut dec = OpusDecoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            let reference = c_dec.decode(Some(&packet), max, false);
            let ours = dec.decode(&packet, max, &mut out);
            match (reference, ours) {
                (Ok(reference), Ok(n)) => {
                    let ours = &out[..n * channels];
                    assert_eq!(ours.len(), reference.len(), "{rate} Hz, round {round}: duration");
                    assert!(ours.iter().all(|v| v.is_finite()), "{rate} Hz, round {round}: not finite: {packet:02x?}");
                    let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
                    over += (peak(ours) > 1.0) as usize;
                    over_c += (peak(&reference) > 1.0) as usize;
                    let diff = reference.iter().zip(ours).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    differ += (diff > 1e-3) as usize;
                    decoded += 1;
                }
                (Err(_), Err(_)) => rejected += 1,
                (reference, ours) => panic!(
                    "{rate} Hz, round {round}: libopus {:?}, opus1 {ours:?} for {packet:02x?}",
                    reference.map(|v| v.len())
                ),
            }
        }
        println!("{rate} Hz, {channels} ch, packets of {packet_channels:?} channels: {decoded} decoded, {rejected} rejected; peaks beyond 1.0 in {over} packets (libopus {over_c}); {differ} packets differ from libopus by more than 0.001");
        assert!(decoded > 1000 && rejected > 500);
        assert!(over > 10, "{rate} Hz: garbage was meant to overshoot");
        assert_eq!(over, over_c, "{rate} Hz: packets with peaks beyond 1.0");
        assert!(differ * 200 <= decoded, "{rate} Hz: {differ} of {decoded} decoded packets differ from libopus");
    }
}

/// Real packets, lost packets, table-of-contents bytes alone and random
/// bytes in any order, through one decoder: no call panics and the output
/// stays finite. Errors are fine.
#[test]
fn any_sequence_of_packets_is_handled() {
    let mut rng = Rng(7);
    for (rate, channels, mode, bandwidth, bitrate) in [
        (8000usize, 1usize, MODE_SILK, BW_NB, 12000),
        (16000, 2, MODE_SILK, BW_WB, 32000),
        (24000, 1, MODE_HYBRID, BW_SWB, 32000),
        (48000, 2, MODE_HYBRID, BW_FB, 64000),
        (48000, 1, MODE_CELT, BW_FB, 64000),
        (12000, 2, MODE_CELT, BW_WB, 64000),
    ] {
        let frame = rate / 50;
        let (packets, _) = c_stream(rate, channels, mode, bandwidth, bitrate, frame);
        let max = rate * 120 / 1000;
        let mut dec = OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
        )
        .unwrap();
        let mut out = vec![0f32; max * channels];
        let mut out_i16 = vec![0i16; max * channels];
        let (mut ok, mut errors) = (0, 0);
        for step in 0..4000 {
            let real = &packets[rng.next() as usize % packets.len()];
            // Random bytes that say either channel count.
            let garbage = rng.garbage(None);
            let marker = [rng.next() as u8];
            let lost_size = rate / 400 * (1 + rng.next() as usize % 48);
            let (data, size): (&[u8], usize) = match rng.next() % 8 {
                0 => (&[], lost_size),
                1 => (&marker, max),
                2 => (&garbage, max),
                _ => (real, max),
            };
            let result = match rng.next() % 5 {
                0 => dec
                    .decode_i16(data, size, &mut out_i16)
                    .inspect(|&n| out[..n * channels].fill(0.0)),
                1 => dec.decode_fec(data, lost_size, &mut out),
                _ => dec.decode(data, size, &mut out),
            };
            match result {
                Ok(n) => {
                    assert!(out[..n * channels].iter().all(|v| v.is_finite()), "{rate} Hz, step {step}: not finite");
                    ok += 1;
                }
                Err(_) => errors += 1,
            }
            if step % 500 == 499 {
                dec.reset_state();
            }
        }
        assert!(ok > 2000 && errors > 50, "{rate} Hz: {ok} calls decoded, {errors} errors");
    }
}
