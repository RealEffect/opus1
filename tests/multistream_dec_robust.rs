//! `MultistreamDecoder` on input that is not a well-formed stream, against
//! the C libopus multistream decoder.
//!
//! What is checked:
//!
//! - the constructor accepts and rejects the layouts and sampling rates that
//!   `opus_multistream_decoder_create` accepts and rejects;
//! - truncated and mutated packets, random bytes and packets for another
//!   layout never make the decoder panic, and are accepted or rejected as
//!   the C decoder does, with the same number of samples;
//! - a rejected packet leaves the decoder as it was;
//! - packets that the C encoder does not produce by itself: several frames
//!   per stream up to 120 ms, padding in every stream, streams without data.
//!
//! A packet whose stream is coded with another channel count than its
//! decoder has (a mono-coded packet in a stereo stream, or the reverse) is
//! valid. libopus converts it. `OpusDecoder` rejects it for now, so the
//! tests that feed such packets are ignored.

mod ms_support;
mod multistream_dec_common;

use audiopus_sys as sys;
use ms_support::*;
use multistream_dec_common::*;
use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};

/// Checks one set of constructor arguments against the C constructor.
/// Returns whether both accepted them.
fn same_verdict(fs: i32, channels: usize, streams: usize, coupled: usize, mapping: &[u8]) -> bool {
    let c = CMsDecoder::new(fs, channels, streams, coupled, mapping);
    let r = MultistreamDecoder::with_layout(fs, channels, streams, coupled, mapping);
    assert_eq!(
        c.is_ok(),
        r.is_ok(),
        "{fs} Hz, {channels} channels, {streams} streams, {coupled} coupled, mapping {:?}: C {:?}, opus-rs {:?}",
        &mapping[..channels.min(mapping.len()).min(12)],
        c.as_ref().err(),
        r.as_ref().err()
    );
    if let Ok(r) = &r {
        assert_eq!(r.layout().channels(), channels);
        assert_eq!(r.layout().streams(), streams);
        assert_eq!(r.layout().coupled_streams(), coupled);
        assert_eq!(r.layout().mapping(), &mapping[..channels]);
        assert_eq!(r.sampling_rate(), fs);
    }
    r.is_ok()
}

#[test]
fn layouts_are_accepted_and_rejected_like_libopus() {
    let counts = [0usize, 1, 2, 3, 6, 127, 128, 254, 255, 256, 300];
    let (mut accepted, mut rejected) = (0, 0);
    for channels in counts {
        for streams in counts {
            for coupled in counts {
                // The first value that is not a decoded channel.
                let decoded = (streams + coupled).min(255) as u8;
                // The mapping holds 300 entries, more than any valid layout
                // reads.
                let patterns: [Box<dyn Fn(usize) -> u8>; 6] = [
                    Box::new(|i| (i % decoded.max(1) as usize) as u8),
                    Box::new(|_| 0),
                    Box::new(|_| 255),
                    Box::new(move |i| if i == 0 { decoded.saturating_sub(1) } else { 0 }),
                    Box::new(move |i| if i + 1 == channels { decoded } else { 0 }),
                    Box::new(|i| if i == 0 { 254 } else { 255 }),
                ];
                for pattern in &patterns {
                    let mapping: Vec<u8> = (0..300).map(pattern).collect();
                    if same_verdict(48000, channels, streams, coupled, &mapping) {
                        accepted += 1;
                    } else {
                        rejected += 1;
                    }
                }
            }
        }
    }
    println!("{accepted} layouts accepted, {rejected} rejected");
    assert!(accepted >= 300 && rejected >= 3000, "{accepted} accepted, {rejected} rejected");

    // Random layouts.
    let mut rng = Rng(2024);
    let (mut accepted, mut rejected) = (0, 0);
    for _ in 0..20_000 {
        let channels = rng.below(10);
        let streams = rng.below(6);
        let coupled = rng.below(5);
        let mut mapping = [0u8; 16];
        for m in mapping.iter_mut() {
            *m = if rng.below(6) == 0 { 255 } else { rng.below(8) as u8 };
        }
        if same_verdict(16000, channels, streams, coupled, &mapping) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    println!("random: {accepted} layouts accepted, {rejected} rejected");
    assert!(accepted >= 2000 && rejected >= 2000, "{accepted} accepted, {rejected} rejected");

    // Sampling rates.
    for fs in [8000, 12000, 16000, 24000, 48000] {
        assert!(same_verdict(fs, 6, 4, 2, &[0, 4, 1, 2, 3, 5]), "{fs} Hz");
    }
    for fs in [0, -1, 1, 4000, 11025, 22050, 32000, 44100, 96000, i32::MAX, i32::MIN] {
        assert!(!same_verdict(fs, 6, 4, 2, &[0, 4, 1, 2, 3, 5]), "{fs} Hz");
    }
}

/// `MultistreamDecoder::new` checks the table of an Ogg Opus header like
/// `with_layout`. The table of the old decoder's panic is rejected.
#[test]
fn mapping_tables_are_checked() {
    let table = |channels: u8, stream_count: u8, coupled_count: u8, mapping: &[u8]| ChannelMappingTable {
        channels,
        stream_count,
        coupled_count,
        mapping: mapping.to_vec(),
    };
    let mut rng = Rng(7);
    let (mut accepted, mut rejected) = (0, 0);
    for _ in 0..5000 {
        let (channels, streams, coupled) = (rng.below(9), rng.below(6), rng.below(5));
        let mapping: Vec<u8> = (0..channels).map(|_| if rng.below(6) == 0 { 255 } else { rng.below(8) as u8 }).collect();
        let c = CMsDecoder::new(48000, channels, streams, coupled, &mapping);
        let r = MultistreamDecoder::new(48000, table(channels as u8, streams as u8, coupled as u8, &mapping));
        assert_eq!(c.is_ok(), r.is_ok(), "{channels} channels, {streams} streams, {coupled} coupled, {mapping:?}");
        if r.is_ok() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    assert!(accepted >= 500 && rejected >= 500, "{accepted} accepted, {rejected} rejected");
    // A table from a header: the tail of a 5.1 `OpusHead`.
    let parsed = ChannelMappingTable::parse(1, 6, &[4, 2, 0, 4, 1, 2, 3, 5]).unwrap();
    let dec = MultistreamDecoder::new(48000, parsed).unwrap();
    assert_eq!((dec.layout().streams(), dec.layout().coupled_streams()), (4, 2));
    // A mapping entry that names no decoded channel.
    assert!(MultistreamDecoder::new(48000, table(1, 1, 0, &[5])).is_err());
    // A mapping with the wrong number of entries.
    assert!(MultistreamDecoder::new(48000, table(2, 1, 1, &[0, 1, 0])).is_err());
    assert!(MultistreamDecoder::new(48000, table(2, 1, 1, &[0])).is_err());
}

/// Whether some stream of a valid packet is coded with another channel count
/// than its decoder has.
fn needs_channel_conversion(packet: &[u8], streams: usize, coupled: usize) -> bool {
    match split_streams(packet, streams) {
        Ok(parts) => parts.iter().enumerate().any(|(s, part)| toc(part.toc).stereo != (s < coupled)),
        // Both decoders reject the packet before they decode anything.
        Err(_) => false,
    }
}

/// Counts of what happened to the packets of a robustness run.
#[derive(Debug, Default)]
struct Verdicts {
    accepted: usize,
    rejected: usize,
    /// Left out: valid packets that need a channel conversion.
    skipped: usize,
}

/// A pair of decoders with the same layout, fed with the same packets.
struct Pair {
    c: CMsDecoder,
    r: MultistreamDecoder,
    channels: usize,
    streams: usize,
    coupled: usize,
    fs: i32,
    out: Vec<f32>,
    out16: Vec<i16>,
}

impl Pair {
    fn new(e: &Encoded, fs: i32) -> Self {
        let (c, r) = e.decoders(fs);
        let room = fs as usize / 25 * 3 * e.channels;
        Self { c, r, channels: e.channels, streams: e.streams, coupled: e.coupled, fs, out: vec![0.0; room], out16: vec![0; room] }
    }

    /// Gives `packet` to both decoders and checks that they agree on
    /// accepting it and on the number of samples. `kind` picks the call:
    /// float or 16 bits, plain or FEC.
    fn feed(&mut self, packet: &[u8], frame_size: usize, kind: usize, verdicts: &mut Verdicts, what: &str) {
        let fec = kind & 1 != 0;
        let sixteen = kind & 2 != 0;
        let c_packet = (!packet.is_empty()).then_some(packet);
        let (c, r) = if sixteen {
            let c = self.c.decode_i16(c_packet, frame_size, fec).map(|pcm| pcm.len() / self.channels);
            let r = if fec { self.r.decode_fec_i16(packet, frame_size, &mut self.out16) } else { self.r.decode_i16(packet, frame_size, &mut self.out16) };
            (c, r)
        } else {
            let c = self.c.decode_float(c_packet, frame_size, fec).map(|pcm| pcm.len() / self.channels);
            let r = if fec { self.r.decode_fec(packet, frame_size, &mut self.out) } else { self.r.decode(packet, frame_size, &mut self.out) };
            (c, r)
        };
        match (c, r) {
            (Ok(c), Ok(r)) => {
                assert_eq!(c, r, "{what}: samples per channel");
                verdicts.accepted += 1;
            }
            (Err(_), Err(_)) => verdicts.rejected += 1,
            (c, r) => panic!(
                "{what}: C {c:?}, opus-rs {r:?}; {} bytes, frame_size {frame_size}, fec {fec}, 16 bits {sixteen}: {}",
                packet.len(),
                describe_packet(packet, self.streams)
            ),
        }
    }
}

/// Changes a packet in one of eight ways.
fn mutate(packet: &[u8], other: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut p = packet.to_vec();
    match rng.below(8) {
        // Flip one bit.
        0 => {
            let i = rng.below(p.len());
            p[i] ^= 1 << rng.below(8);
        }
        // Replace one byte. Half of the time it is among the first bytes,
        // where the framing is.
        1 => {
            let i = if rng.below(2) == 0 { rng.below(p.len().min(8)) } else { rng.below(p.len()) };
            p[i] = rng.next() as u8;
        }
        // Cut the end off.
        2 => p.truncate(rng.below(p.len() + 1)),
        // Add bytes at the end.
        3 => {
            for _ in 0..1 + rng.below(40) {
                p.push(rng.next() as u8);
            }
        }
        // Take a byte out.
        4 => {
            let i = rng.below(p.len());
            p.remove(i);
        }
        // Repeat a byte.
        5 => {
            let i = rng.below(p.len());
            p.insert(i, p[i]);
        }
        // Random bytes.
        6 => {
            p = (0..1 + rng.below(60)).map(|_| rng.next() as u8).collect();
        }
        // A packet of another layout.
        _ => p = other.to_vec(),
    }
    p
}

/// Feeds mutated packets to both decoders. With `with_conversion`, packets
/// that need a channel conversion are fed as well.
fn run_mutations(with_conversion: bool) -> Verdicts {
    let sources = [
        encode(MONO, VOIP_FEC, 16, 960, Signal::Voiced, 400, 1.0),
        encode(STEREO, AUDIO, 64, 480, Signal::Tones, 400, 1.0),
        encode(SURROUND_5_1, AUDIO, 32, 960, Signal::Voiced, 400, 1.0),
        encode(SURROUND_5_1, AUDIO_CBR, 12, 2880, Signal::Tones, 600, 1.0),
        encode(SURROUND_7_1, VOIP_FEC, 16, 1920, Signal::Voiced, 400, 1.0),
        encode(DISCRETE_3, VOIP_FEC, 12, 2880, Signal::Voiced, 600, 1.0),
        encode(DISCRETE_5, AUDIO, 24, 120, Signal::Tones, 200, 1.0),
        encode(AMBISONICS_4, AUDIO_CBR, 32, 240, Signal::Voiced, 200, 1.0),
        encode(Layout::Generic, AUDIO, 64, 960, Signal::Tones, 400, 1.0),
    ];
    let mut rng = Rng(0xC0FFEE);
    let mut verdicts = Verdicts::default();
    for (k, e) in sources.iter().enumerate() {
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        let other = &sources[(k + 1) % sources.len()];
        for fs in [8000, 48000] {
            let mut pair = Pair::new(e, fs);
            let unit = fs as usize / 400;
            for round in 0..700 {
                let original = &e.packets[rng.below(e.packets.len())];
                // One packet in four is fed as it is, to keep a real stream
                // going between the damaged ones.
                let packet = if round % 4 == 0 { original.clone() } else { mutate(original, &other.packets[rng.below(other.packets.len())], &mut rng) };
                if !with_conversion && needs_channel_conversion(&packet, pair.streams, pair.coupled) {
                    verdicts.skipped += 1;
                    continue;
                }
                // Mostly the largest room, sometimes less than the packet
                // needs, sometimes not a multiple of 2.5 ms.
                let frame_size = match rng.below(8) {
                    0 => unit * (1 + rng.below(48)),
                    1 => 1 + rng.below(48 * unit),
                    _ => 48 * unit,
                };
                let kind = if rng.below(4) == 0 { rng.below(4) } else { 0 };
                let what = format!("{} at {} Hz, round {round}", e.label, pair.fs);
                pair.feed(&packet, frame_size, kind, &mut verdicts, &what);
            }
        }
    }
    println!("{verdicts:?}");
    verdicts
}

/// Measured: 7538 packets accepted, 4754 rejected, 308 left out because they
/// need a channel conversion.
#[test]
fn mutated_packets_are_accepted_or_rejected_like_libopus() {
    let v = run_mutations(false);
    assert!(v.accepted >= 4000 && v.rejected >= 3000, "{v:?}");
    assert!(v.accepted + v.rejected >= 10_000, "{v:?}");
}

#[test]
fn mutated_packets_with_another_channel_count_are_accepted_like_libopus() {
    let v = run_mutations(true);
    assert_eq!(v.skipped, 0);
    assert!(v.accepted + v.rejected >= 12_000, "{v:?}");
}

/// Every way of cutting a packet short.
#[test]
fn truncated_packets_are_accepted_or_rejected_like_libopus() {
    let mut verdicts = Verdicts::default();
    for layout in [MONO, QUAD, SURROUND_5_1, SURROUND_7_1, DISCRETE_5, AMBISONICS_4, Layout::Generic] {
        for (settings, kbps, frame) in [(AUDIO, 24, 960), (VOIP_FEC, 12, 1920), (AUDIO_CBR, 16, 480)] {
            let e = encode(layout, settings, kbps, frame, Signal::Voiced, 300, 1.0);
            if e.has_mono_coded_stereo_stream() {
                continue;
            }
            let mut pair = Pair::new(&e, 48000);
            for (i, packet) in e.packets.iter().enumerate().skip(2).take(4) {
                // The length 0 is a lost packet: both conceal.
                for len in 0..packet.len() {
                    let what = format!("{}, packet {i} cut to {len} bytes", e.label);
                    pair.feed(&packet[..len], 5760, 0, &mut verdicts, &what);
                }
                let what = format!("{}, packet {i}", e.label);
                pair.feed(packet, 5760, 0, &mut verdicts, &what);
            }
        }
    }
    println!("{verdicts:?}");
    assert!(verdicts.accepted >= 500 && verdicts.rejected >= 500, "{verdicts:?}");
}

/// Packets for a decoder with another number of streams.
#[test]
fn packets_for_another_stream_count_are_handled_like_libopus() {
    // 5 mono streams.
    let e = encode(DISCRETE_5, AUDIO, 32, 960, Signal::Voiced, 300, 1.0);
    assert_eq!((e.streams, e.coupled), (5, 0));
    let mut verdicts = Verdicts::default();
    // (streams, coupled, mapping) of decoders for the 5-stream packets.
    let layouts: [(usize, usize, &[u8]); 7] = [
        // Fewer streams: the rest of the packet looks like one long last
        // stream, which is valid when it is short enough.
        (1, 0, &[0]),
        (2, 0, &[0, 1]),
        (3, 0, &[0, 1, 2]),
        (4, 0, &[0, 1, 2, 3]),
        // The right number of streams.
        (5, 0, &[0, 1, 2, 3, 4]),
        // More streams than the packet has.
        (6, 0, &[0, 1, 2, 3, 4, 5]),
        (9, 0, &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ];
    for (streams, coupled, mapping) in layouts {
        let mut pair = Pair {
            c: CMsDecoder::new(48000, mapping.len(), streams, coupled, mapping).unwrap(),
            r: MultistreamDecoder::with_layout(48000, mapping.len(), streams, coupled, mapping).unwrap(),
            channels: mapping.len(),
            streams,
            coupled,
            fs: 48000,
            out: vec![0.0; 5760 * mapping.len()],
            out16: vec![0; 5760 * mapping.len()],
        };
        for (i, packet) in e.packets.iter().enumerate() {
            if needs_channel_conversion(packet, streams, coupled) {
                verdicts.skipped += 1;
                continue;
            }
            pair.feed(packet, 5760, 0, &mut verdicts, &format!("{streams} streams, packet {i}"));
        }
    }
    println!("{verdicts:?}");
    // More streams: every packet is rejected. The same or fewer: accepted.
    assert_eq!(verdicts.rejected, 2 * e.packets.len(), "{verdicts:?}");
    assert_eq!(verdicts.accepted, 5 * e.packets.len(), "{verdicts:?}");
    assert_eq!(verdicts.skipped, 0, "{verdicts:?}");
}

/// A packet that is rejected changes nothing: the packets after it decode as
/// if it had never arrived.
#[test]
fn a_rejected_packet_leaves_the_decoder_untouched() {
    for (layout, settings) in [(SURROUND_5_1, AUDIO), (DISCRETE_3, VOIP_FEC), (Layout::Generic, AUDIO)] {
        let e = encode(layout, settings, 32, 960, Signal::Voiced, 600, 1.0);
        let (mut c_dec, mut dec) = e.decoders(48000);
        let (_, mut clean) = e.decoders(48000);
        let ch = e.channels;
        let mut out = vec![0f32; 5760 * ch];
        let mut out_clean = vec![0f32; 5760 * ch];
        let mut rejected = 0;
        for (i, packet) in e.packets.iter().enumerate() {
            let parts = split_streams(packet, e.streams).unwrap();
            let last_start = packet.len() - parts.last().unwrap().used;
            // Damaged copies of this packet. In all of them the first
            // streams are intact and only the end is wrong.
            let mut bad: Vec<Vec<u8>> = Vec::new();
            // The last stream is missing.
            bad.push(packet[..last_start].to_vec());
            // The last stream has another duration: 10 ms in place of 20.
            let mut other_duration = packet.clone();
            other_duration[last_start] ^= 0x08;
            bad.push(other_duration);
            // The last stream claims more frames than 120 ms hold.
            let mut too_long = packet.clone();
            too_long[last_start] |= 0x03;
            too_long.insert(last_start + 1, 0x3F);
            bad.push(too_long);
            // One byte, for a decoder with several streams.
            bad.push(vec![packet[0]]);
            for (k, garbage) in bad.iter().enumerate() {
                let what = format!("{}, packet {i}, damaged copy {k}", e.label);
                assert!(c_dec.decode_float(Some(garbage), 5760, false).is_err(), "{what}: C accepted it");
                assert!(dec.decode(garbage, 5760, &mut out).is_err(), "{what}: accepted");
                assert!(dec.decode_i16(garbage, 5760, &mut vec![0i16; 5760 * ch]).is_err(), "{what}: accepted");
                assert!(dec.decode_fec(garbage, 960, &mut out).is_err(), "{what}: accepted by decode_fec");
                rejected += 1;
            }
            // Too little room for the packet.
            assert_eq!(c_dec.decode_float(Some(packet), 480, false), Err(OPUS_BUFFER_TOO_SMALL));
            assert!(dec.decode(packet, 480, &mut out).is_err(), "{}: decoded 20 ms into room for 10", e.label);
            assert!(dec.decode_fec(packet, 480, &mut out).is_err());
            assert!(dec.decode(packet, 0, &mut out).is_err());
            // Room in `frame_size`, but not in the output slice.
            assert!(dec.decode(packet, 960, &mut out[..960 * ch - 1]).is_err());

            let n = dec.decode(packet, 5760, &mut out).unwrap();
            let n_clean = clean.decode(packet, 5760, &mut out_clean).unwrap();
            assert_eq!(n, n_clean);
            assert!(
                out[..n * ch].iter().zip(&out_clean[..n * ch]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{}, packet {i}: a rejected packet changed the decoder",
                e.label
            );
            assert_eq!(dec.final_range(), e.ranges[i]);
            c_dec.decode_float(Some(packet), 5760, false).unwrap();
        }
        assert!(rejected >= 100);
    }
}

/// Joins `group`, consecutive packets of `e`, into one packet: every stream
/// gets the frames of all of them. With `padding`, every stream also gets
/// that many bytes more than it needs. Returns `None` when a stream changes
/// its configuration inside the group, which one packet cannot express.
fn join_packets(e: &Encoded, group: &[Vec<u8>], padding: usize) -> Option<Vec<u8>> {
    let split: Vec<Vec<StreamPacket>> = group.iter().map(|p| split_streams(p, e.streams).unwrap()).collect();
    let mut joined = Vec::new();
    for s in 0..e.streams {
        let toc_byte = split[0][s].toc & 0xFC;
        if split.iter().any(|parts| parts[s].toc & 0xFC != toc_byte) {
            return None;
        }
        let mut frames: Vec<&[u8]> = Vec::new();
        for parts in &split {
            let stream = &parts[s].packet;
            let parsed = opus_rs::packet::parse(stream, false).unwrap();
            frames.extend(parsed.frames(stream));
        }
        let self_delimited = s + 1 != e.streams;
        let mut buf = vec![0u8; 48 * 1277 + 8];
        let len = opus_rs::packet::write(toc_byte, &frames, &mut buf, self_delimited, false).unwrap();
        let len = if padding > 0 {
            opus_rs::packet::write(toc_byte, &frames, &mut buf[..len + padding], self_delimited, true).unwrap()
        } else {
            len
        };
        joined.extend_from_slice(&buf[..len]);
    }
    Some(joined)
}

/// Packets with several frames per stream, up to the 120 ms a packet may
/// hold, with and without padding in every stream. The C encoder does not
/// make such packets by itself, but any Ogg Opus file may hold them.
#[test]
fn long_and_padded_packets_match_libopus() {
    // (layout, settings, bitrate, frame, packets per joined packet, padding)
    let cases = [
        (SURROUND_5_1, AUDIO, 32, 960usize, 2usize, 0usize),
        (SURROUND_5_1, AUDIO, 32, 960, 3, 5),
        (SURROUND_5_1, AUDIO, 32, 960, 6, 0),
        (SURROUND_7_1, AUDIO, 24, 120, 48, 1),
        (SURROUND_7_1, AUDIO, 24, 240, 7, 300),
        (DISCRETE_3, VOIP_FEC, 12, 960, 3, 0),
        (DISCRETE_3, VOIP_FEC, 12, 960, 6, 2),
        (DISCRETE_5, VOIP_FEC, 24, 480, 12, 40),
        (AMBISONICS_4, AUDIO_CBR, 32, 480, 5, 0),
        (Layout::Generic, AUDIO, 64, 1920, 3, 17),
    ];
    for (layout, settings, kbps, frame, per_packet, padding) in cases {
        let e = encode(layout, settings, kbps, frame, Signal::Voiced, 1000.max(frame * per_packet / 8), 1.0);
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for fs in [16000, 48000] {
            let (mut c_dec, mut r_dec) = e.decoders(fs);
            let ch = e.channels;
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0f32; max_frame * ch];
            let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
            let mut joined_packets = 0;
            for (g, group) in e.packets.chunks(per_packet).enumerate() {
                let what = format!("{} at {fs} Hz, {per_packet} packets in one, padding {padding}, group {g}", e.label);
                // A group that cannot be joined is decoded packet by packet.
                let joined = if group.len() == per_packet { join_packets(&e, group, padding) } else { None };
                let packets: Vec<Vec<u8>> = match joined {
                    Some(p) => {
                        joined_packets += 1;
                        vec![p]
                    }
                    None => group.to_vec(),
                };
                for packet in &packets {
                    let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap_or_else(|err| panic!("{what}: C decoder: {err}"));
                    let n = r_dec.decode(packet, max_frame, &mut out).unwrap_or_else(|err| panic!("{what}: {err}"));
                    assert_eq!(c.len(), n * ch, "{what}: samples");
                    assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range");
                    c_all.extend_from_slice(&c);
                    r_all.extend_from_slice(&out[..n * ch]);
                }
                // The final range is that of the last frame.
                let last = (g * per_packet + group.len()) - 1;
                assert_eq!(r_dec.final_range(), e.ranges[last], "{what}: final range of the last frame");
            }
            assert!(joined_packets >= 2, "{}: only {joined_packets} joined packets", e.label);
            assert_eq!(c_all.len(), e.packets.len() * frame * fs as usize / FS as usize * ch, "{}", e.label);
            let snrs = channel_snrs(&c_all, &r_all, ch);
            assert!(snrs.iter().all(|&snr| snr >= 100.0), "{} at {fs} Hz, {per_packet} in one: {snrs:.0?} dB", e.label);
        }
    }
}

/// `opus_multistream_packet_pad` pads the last stream of a packet. The
/// padded packet decodes to the same samples.
#[test]
fn packets_padded_by_libopus_decode_to_the_same_samples() {
    for layout in [MONO, SURROUND_5_1, DISCRETE_3] {
        let e = encode(layout, AUDIO, 32, 960, Signal::Voiced, 400, 1.0);
        let (mut c_dec, mut r_dec) = e.decoders(48000);
        let (_, mut r_plain) = e.decoders(48000);
        let ch = e.channels;
        let mut out = vec![0f32; 960 * ch];
        let mut out_plain = vec![0f32; 960 * ch];
        for (i, packet) in e.packets.iter().enumerate() {
            let extra = [1usize, 2, 3, 100, 254, 255, 256, 700][i % 8];
            let mut padded = packet.clone();
            padded.resize(packet.len() + extra, 0);
            let ret = unsafe { sys::opus_multistream_packet_pad(padded.as_mut_ptr(), packet.len() as i32, padded.len() as i32, e.streams as i32) };
            assert_eq!(ret, 0, "opus_multistream_packet_pad");
            let c = c_dec.decode_float(Some(&padded), 960, false).unwrap();
            assert_eq!(r_dec.decode(&padded, 960, &mut out), Ok(960), "{}, packet {i}, {extra} bytes of padding", e.label);
            assert_eq!(r_plain.decode(packet, 960, &mut out_plain), Ok(960));
            assert!(out.iter().zip(&out_plain).all(|(a, b)| a.to_bits() == b.to_bits()), "{}, packet {i}: padding changed the output", e.label);
            assert_eq!(r_dec.final_range(), e.ranges[i]);
            let snr = min_channel_snr(&c, &out, ch);
            assert!(snr >= 100.0, "{}, packet {i}: {snr:.1} dB from the C decoder", e.label);
        }
    }
}

/// Packets written by hand: streams without data, and the shortest packets
/// there are.
#[test]
fn hand_made_packets_are_handled_like_libopus() {
    // A CELT fullband 20 ms TOC byte, mono and stereo.
    const MONO_TOC: u8 = 0xF8;
    const STEREO_TOC: u8 = 0xFC;
    let e = encode(SURROUND_5_1, AUDIO, 32, 960, Signal::Voiced, 200, 1.0);
    let mut verdicts = Verdicts::default();
    let mut pair = Pair::new(&e, 48000);
    let packets: Vec<Vec<u8>> = vec![
        // Every stream without data: 2 bytes each, 1 for the last. All four
        // streams are concealed.
        vec![STEREO_TOC, 0, STEREO_TOC, 0, MONO_TOC, 0, MONO_TOC],
        // One byte less than the shortest packet of 4 streams.
        vec![STEREO_TOC, 0, STEREO_TOC, 0, MONO_TOC, 0],
        // A frame of one byte in every stream, which is concealed too.
        vec![STEREO_TOC, 1, 0xAA, STEREO_TOC, 1, 0xAA, MONO_TOC, 1, 0xAA, MONO_TOC, 0xAA],
        // Two frames without data per stream (code 1): 40 ms.
        vec![STEREO_TOC | 1, 0, STEREO_TOC | 1, 0, MONO_TOC | 1, 0, MONO_TOC | 1],
        // Code 3 with 6 frames without data: 120 ms.
        vec![STEREO_TOC | 3, 6, 0, STEREO_TOC | 3, 6, 0, MONO_TOC | 3, 6, 0, MONO_TOC | 3, 6],
        // Code 3 with 7 frames: 140 ms, too long.
        vec![STEREO_TOC | 3, 7, 0, STEREO_TOC | 3, 7, 0, MONO_TOC | 3, 7, 0, MONO_TOC | 3, 7],
        // Code 3 with no frame.
        vec![STEREO_TOC | 3, 0, 0, STEREO_TOC | 3, 0, 0, MONO_TOC | 3, 0, 0, MONO_TOC | 3, 0],
        // The streams have different durations: 20, 20, 20 and 10 ms.
        vec![STEREO_TOC, 0, STEREO_TOC, 0, MONO_TOC, 0, MONO_TOC ^ 0x08],
        // A size field that points behind the end of the packet.
        vec![STEREO_TOC, 200, STEREO_TOC, 0, MONO_TOC, 0, MONO_TOC],
        // A 2-byte size field cut after its first byte.
        vec![STEREO_TOC, 1, 0xAA, STEREO_TOC, 1, 0xAA, MONO_TOC, 252],
        // Padding that is longer than the packet.
        vec![STEREO_TOC | 3, 0x41, 250, 0, STEREO_TOC, 0, MONO_TOC, 0, MONO_TOC],
        // Padding of 2 bytes in the first stream.
        vec![STEREO_TOC | 3, 0x41, 2, 0, 0, 0, STEREO_TOC, 0, MONO_TOC, 0, MONO_TOC],
    ];
    // Real packets before and between them, so that there is a signal to
    // conceal.
    for (i, packet) in packets.iter().enumerate() {
        for real in &e.packets[..3] {
            pair.feed(real, 5760, 0, &mut verdicts, "real packet");
        }
        for kind in 0..4 {
            let frame_size = if kind & 1 != 0 { 960 } else { 5760 };
            pair.feed(packet, frame_size, kind, &mut verdicts, &format!("hand-made packet {i}, call {kind}"));
        }
    }
    println!("{verdicts:?}");
    assert!(verdicts.accepted >= 36 + 16 && verdicts.rejected >= 16, "{verdicts:?}");
}
