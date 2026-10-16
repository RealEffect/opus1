//! CELT loss concealment against C libopus: cases beyond tests/celtdec_plc.rs.
//!
//! - Concealment calls whose length differs from the packet length.
//! - Streams whose frame size, bandwidth or channel count changes next to a
//!   loss.
//! - Packets with several frames.
//! - Signals with a strong pitch (the encoder then switches the pitch
//!   post-filter on), clicks, decaying tones, very loud and very quiet input.
//! - A CELT frame of one byte, which libopus treats as lost.
//!
//! Both decoders always get the same packets and the same calls. The limits
//! are those of tests/celtdec_plc.rs: see the top of that file for why pitch
//! repetition cannot match libopus to the last bit.
//!
//! Run with `--nocapture` to see the measured numbers.

mod celtdec_common;
mod k24_common;
use celtdec_common::*;
use k24_common::*;
use opus1::celt::{CeltDecoder, PcmLayout};
use opus1::modes::default_mode;
use opus1::range_coder::RangeCoder;

const ARM64: bool = cfg!(target_arch = "aarch64");
/// Lowest median SNR of pitch-concealed frames against libopus, in dB.
const PITCH_MEDIAN_DB: f64 = if ARM64 { 65.0 } else { 50.0 };
/// SNR below which a pitch-concealed frame counts as badly matched.
const POOR_DB: f64 = 40.0;
/// Lowest SNR of a frame that both decoders conceal with noise, in dB.
const NOISE_DB: f64 = 110.0;
/// Lowest SNR of a decoded frame that no loss has touched, in dB.
const CLEAN_DB: f64 = 100.0;

fn celt_encoder(rate: usize, channels: usize, bandwidth: i32, bitrate: i32) -> CEnc {
    let mut enc = CEnc::audio(rate, channels);
    enc.force_mode(MODE_CELT)
        .bandwidth(bandwidth)
        .bitrate(bitrate);
    enc
}

fn stereo_if(mono: Vec<f32>, channels: usize, rate: usize) -> Vec<f32> {
    if channels == 2 {
        to_stereo(&mono, rate)
    } else {
        mono
    }
}

/// Median of a list of SNR values, and how many are below `POOR_DB`.
fn median_and_poor(mut snrs: Vec<f64>) -> (f64, usize, f64) {
    assert!(!snrs.is_empty(), "nothing to compare");
    snrs.sort_by(|a, b| a.total_cmp(b));
    let poor = snrs.iter().filter(|s| **s < POOR_DB).count();
    (snrs[snrs.len() / 2], poor, snrs[0])
}

/// Both decoders, driven call by call.
struct Pair {
    c: CDecoder,
    rs: opus1::OpusDecoder,
    channels: usize,
    buf: Vec<f32>,
}

impl Pair {
    fn new(rate: usize, channels: usize) -> Self {
        Self {
            c: CDecoder::new(rate, channels),
            rs: opus1::OpusDecoder::new(rate as i32, channels).unwrap(),
            channels,
            buf: vec![0.0; 5760 * channels],
        }
    }

    /// Decode a packet, or conceal `frame` samples per channel (`None`).
    /// Returns the output of libopus and of opus1, limited to [-1, 1] (see
    /// `decode_with_loss` in celtdec_common).
    fn decode(&mut self, packet: Option<&[u8]>, frame: usize) -> (Vec<f32>, Vec<f32>) {
        let mut c = self.c.decode(packet, frame);
        let out = &mut self.buf[..frame * self.channels];
        let n = self.rs.decode(packet.unwrap_or(&[]), frame, out).unwrap();
        assert_eq!(n * self.channels, c.len(), "decoded length");
        let mut r = out[..n * self.channels].to_vec();
        for v in c.iter_mut().chain(r.iter_mut()) {
            *v = v.clamp(-1.0, 1.0);
        }
        (c, r)
    }
}

/// A lost packet may be concealed in several calls, or several lost packets
/// in one call. libopus accepts any multiple of 2.5 ms. It conceals in
/// pieces no longer than a frame of the last packet (20 ms in this stream)
/// and cuts a shorter rest into CELT frame sizes (`opus_decode_frame`: 10,
/// 5 and 2.5 ms). The CELT state must come out the same as in libopus for
/// every cut.
#[test]
fn concealment_calls_of_any_length_in_a_celt_stream() {
    let (rate, frame) = (48000usize, 960usize);
    // Each entry is the list of calls that conceals two lost 20 ms packets.
    let cuts: [(&str, Vec<usize>); 8] = [
        ("2 x 20 ms", vec![960, 960]),
        ("1 x 40 ms", vec![1920]),
        ("4 x 10 ms", vec![480; 4]),
        ("8 x 5 ms", vec![240; 8]),
        ("16 x 2.5 ms", vec![120; 16]),
        ("7.5 + 12.5 + 17.5 + 2.5 ms", vec![360, 600, 840, 120]),
        ("15 + 25 ms", vec![720, 1200]),
        ("2.5 + 37.5 ms", vec![120, 1800]),
    ];
    let mut failures = Vec::new();
    for channels in [1usize, 2] {
        let pcm = stereo_if(speech_fullband(rate, 3.0), channels, rate);
        let packets =
            celt_encoder(rate, channels, BW_FB, 64000 * channels as i32).encode_all(&pcm, frame);
        for (name, cut) in &cuts {
            assert_eq!(cut.iter().sum::<usize>(), 2 * frame);
            let mut pair = Pair::new(rate, channels);
            let (mut concealed, mut first_good, mut later) = (Vec::new(), Vec::new(), Vec::new());
            let mut i = 0;
            let mut good_since_loss = usize::MAX;
            while i < packets.len() {
                // Two lost packets out of every twelve.
                if i % 12 == 8 && i + 1 < packets.len() {
                    let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
                    for &n in cut {
                        let (c, r) = pair.decode(None, n);
                        c_all.extend(c);
                        r_all.extend(r);
                    }
                    concealed.push(snr(&c_all, &r_all));
                    good_since_loss = 0;
                    i += 2;
                    continue;
                }
                let (c, r) = pair.decode(Some(&packets[i]), frame);
                let db = snr(&c, &r);
                if good_since_loss == 0 {
                    first_good.push(db);
                } else if good_since_loss >= 3 {
                    later.push(db);
                }
                good_since_loss = good_since_loss.saturating_add(1);
                i += 1;
            }
            let (median, poor, worst) = median_and_poor(concealed);
            let (good_median, good_poor, good_worst) = median_and_poor(first_good);
            let (_, _, later_worst) = median_and_poor(later);
            println!(
                "CELT {channels} ch, {name}: concealed median {median:.1} dB, worst {worst:.1} dB; first good frame median {good_median:.1} dB, worst {good_worst:.1} dB; later frames worst {later_worst:.1} dB"
            );
            if median < PITCH_MEDIAN_DB
                || poor > 1
                || good_median < PITCH_MEDIAN_DB
                || good_poor > 1
                || later_worst < 80.0
            {
                failures.push(format!(
                    "CELT {channels} ch, {name}: concealed median {median:.1} dB ({poor} poor), first good median {good_median:.1} dB ({good_poor} poor), later worst {later_worst:.1} dB"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The same for a Hybrid stream, with the lengths that the SILK concealment
/// works in (10 and 20 ms). The CELT layer is noise from the decoder's seed,
/// so every frame must match libopus closely.
#[test]
fn concealment_calls_of_10_and_20_ms_in_a_hybrid_stream() {
    let (rate, frame) = (48000usize, 960usize);
    let cuts: [(&str, Vec<usize>); 4] = [
        ("2 x 20 ms", vec![960, 960]),
        ("1 x 40 ms", vec![1920]),
        ("4 x 10 ms", vec![480; 4]),
        ("10 + 20 + 10 ms", vec![480, 960, 480]),
    ];
    for channels in [1usize, 2] {
        let pcm = stereo_if(speech_fullband(rate, 3.0), channels, rate);
        let mut enc = CEnc::audio(rate, channels);
        enc.force_mode(MODE_HYBRID)
            .bandwidth(BW_FB)
            .bitrate(if channels == 1 { 32000 } else { 48000 });
        let packets = enc.encode_all(&pcm, frame);
        assert!(packets.iter().all(|p| toc(p[0]).mode == "Hybrid"));
        for (name, cut) in &cuts {
            let mut pair = Pair::new(rate, channels);
            let mut worst = 200f64;
            let mut i = 0;
            let mut lost_calls = 0;
            while i < packets.len() {
                if i % 12 == 8 && i + 1 < packets.len() {
                    for &n in cut {
                        let (c, r) = pair.decode(None, n);
                        worst = worst.min(snr(&c, &r));
                        lost_calls += 1;
                    }
                    i += 2;
                    continue;
                }
                let (c, r) = pair.decode(Some(&packets[i]), frame);
                worst = worst.min(snr(&c, &r));
                i += 1;
            }
            println!("Hybrid {channels} ch, {name}: worst frame {worst:.1} dB");
            assert!(lost_calls >= 10);
            assert!(
                worst > NOISE_DB,
                "Hybrid {channels} ch, {name}: worst frame {worst:.1} dB"
            );
        }
    }
}

/// A stream whose frame size changes, with the loss right at the change.
/// The first frame after a loss lowers its starting energies by an amount
/// that depends on its own size and on how long the loss was in units of
/// that size (libopus `celt_decode_with_ec`, "If recovering from packet
/// loss").
#[test]
fn frame_size_changes_next_to_a_loss() {
    let rate = 48000usize;
    let mut failures = Vec::new();
    for channels in [1usize, 2] {
        let pcm = stereo_if(speech_fullband(rate, 6.0), channels, rate);
        let mut enc = celt_encoder(rate, channels, BW_FB, 64000 * channels as i32);
        // 200 ms of each size, in an order that puts every size next to
        // every other one.
        let order = [
            960usize, 120, 480, 240, 960, 240, 120, 960, 480, 120, 240, 480,
        ];
        let mut packets: Vec<(Vec<u8>, usize)> = Vec::new();
        let mut at = 0;
        'outer: for round in 0.. {
            let size = order[round % order.len()];
            for _ in 0..9600 / size {
                let end = at + size * channels;
                if end > pcm.len() {
                    break 'outer;
                }
                packets.push((enc.encode(&pcm[at..end]), size));
                at = end;
            }
        }
        // Index of the first packet of every block but the first.
        let changes: Vec<usize> = (1..packets.len())
            .filter(|&i| packets[i].1 != packets[i - 1].1)
            .collect();
        // `lose_before`: the last packets of each block are lost, so the
        // first good frame has the new size. Otherwise the first packets of
        // each block are lost, and the receiver conceals them in pieces of
        // the old size, the only one it knows.
        for lose_before in [true, false] {
            for burst in [1usize, 3] {
                let mut lost = vec![false; packets.len()];
                for &b in &changes {
                    let range = if lose_before {
                        b - burst..b
                    } else {
                        b..b + burst
                    };
                    for k in range.filter(|&k| k < packets.len()) {
                        lost[k] = true;
                    }
                }
                let mut pair = Pair::new(rate, channels);
                let (mut concealed, mut first_good, mut later) =
                    (Vec::new(), Vec::new(), Vec::new());
                let mut good_since_loss = usize::MAX;
                let mut prev_size = packets[0].1;
                for (i, (p, size)) in packets.iter().enumerate() {
                    if lost[i] {
                        let mut todo = *size;
                        while todo > 0 {
                            let n = prev_size.min(todo);
                            let (c, r) = pair.decode(None, n);
                            concealed.push(snr(&c, &r));
                            todo -= n;
                        }
                        good_since_loss = 0;
                        continue;
                    }
                    let (c, r) = pair.decode(Some(p), *size);
                    let db = snr(&c, &r);
                    if power(&c) > 1e-10 {
                        if good_since_loss == 0 {
                            first_good.push(db);
                        } else if good_since_loss != usize::MAX && good_since_loss * size >= 2880 {
                            later.push(db);
                        }
                    }
                    good_since_loss = good_since_loss.saturating_add(1);
                    prev_size = *size;
                }
                let name = format!(
                    "{channels} ch, {burst} lost {} a change of frame size",
                    if lose_before { "before" } else { "after" }
                );
                let (median, poor, worst) = median_and_poor(concealed.clone());
                let (good_median, good_poor, good_worst) = median_and_poor(first_good.clone());
                let (_, _, later_worst) = median_and_poor(later);
                println!(
                    "{name}: {} concealed, median {median:.1} dB, worst {worst:.1} dB, {poor} poor; first good frame median {good_median:.1} dB, worst {good_worst:.1} dB, {good_poor} poor; later worst {later_worst:.1} dB",
                    concealed.len()
                );
                assert!(concealed.len() >= 10 && first_good.len() >= 10, "{name}");
                if median < PITCH_MEDIAN_DB
                    || poor > 1 + concealed.len() / 20
                    || good_median < PITCH_MEDIAN_DB
                    || good_poor > 1 + first_good.len() / 20
                    || later_worst < 80.0
                {
                    failures.push(name);
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Concealment calls shorter than 10 ms in a Hybrid stream. libopus lets
/// SILK conceal 10 ms and uses the head of it, and lets CELT conceal exactly
/// the length asked for (`opus_decode_frame`: `celt_frame_size`).
///
/// `OpusDecoder::conceal` (src/lib.rs) asks CELT for 10 ms instead and uses
/// the head. The CELT layer of the concealed frame then differs, and the
/// CELT state has moved on by the wrong amount: the concealed calls and the
/// two good frames after them match libopus to 20 to 35 dB only. The same
/// fault is in the 5 ms of concealment that bridge a switch from Hybrid to
/// CELT-only without a redundancy frame.
///
/// The change that makes this pass, in the Hybrid branch of `conceal`:
/// conceal `celt_n` samples of CELT, where `celt_n` is the largest of 20,
/// 10, 5 and 2.5 ms that fits in what is left, and use `celt_n` samples of
/// the SILK frame as well (not `n.min(remaining)`).
#[test]
fn concealment_calls_shorter_than_10_ms_in_a_hybrid_stream() {
    let (rate, frame) = (48000usize, 960usize);
    let cuts: [(&str, Vec<usize>); 4] = [
        ("4 x 5 ms", vec![240; 4]),
        ("8 x 2.5 ms", vec![120; 8]),
        ("15 + 5 ms", vec![720, 240]),
        ("7.5 + 12.5 ms", vec![360, 600]),
    ];
    let pcm = speech_fullband(rate, 3.0);
    let mut enc = CEnc::audio(rate, 1);
    enc.force_mode(MODE_HYBRID).bandwidth(BW_FB).bitrate(32000);
    let packets = enc.encode_all(&pcm, frame);
    let mut failures = Vec::new();
    for (name, cut) in &cuts {
        let mut pair = Pair::new(rate, 1);
        let (mut worst_lost, mut worst_good) = (200f64, 200f64);
        for (i, p) in packets.iter().enumerate() {
            if i % 12 == 8 {
                for &n in cut {
                    let (c, r) = pair.decode(None, n);
                    worst_lost = worst_lost.min(snr(&c, &r));
                }
            } else {
                let (c, r) = pair.decode(Some(p), frame);
                worst_good = worst_good.min(snr(&c, &r));
            }
        }
        println!(
            "Hybrid, {name}: worst concealed call {worst_lost:.1} dB, worst good frame {worst_good:.1} dB"
        );
        if worst_lost < NOISE_DB || worst_good < NOISE_DB {
            failures.push(format!(
                "{name}: concealed {worst_lost:.1} dB, good {worst_good:.1} dB"
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A concealment call that is longer than a frame of the stream. libopus
/// conceals in pieces of at most the frame size of the last packet
/// (`opus_decode_frame`: `frame_size = IMIN(frame_size, st->frame_size)`).
/// Every piece of a pitch-concealed run is 0.8 times as loud as the one
/// before, so the number of pieces is audible.
///
/// `OpusDecoder::conceal` (src/lib.rs) takes the largest frame that fits
/// instead: 20 ms of a stream with 2.5 ms frames is one piece, where
/// libopus makes eight. The same happens when a lost packet held several
/// short frames. Measured today: concealed calls -47 to 15 dB against
/// libopus, and the frames after them are wrong for about 60 ms.
///
/// The change that makes this pass: `OpusDecoder::decode` stores the frame
/// size from the packet's first byte (libopus `st->frame_size`), and
/// `conceal` limits every piece to it.
#[test]
fn concealment_calls_longer_than_a_frame_of_the_stream() {
    let rate = 48000usize;
    let mut failures = Vec::new();
    // (frame of the stream, frames per packet, length of a concealment call)
    for (frame, per_packet, call) in [
        (480usize, 1usize, 960usize),
        (240, 1, 960),
        (120, 1, 960),
        (240, 1, 480),
        (480, 3, 1440),
        (240, 4, 960),
        (120, 8, 960),
        (120, 5, 600),
    ] {
        for (mode, name) in [(MODE_CELT, "CELT"), (MODE_HYBRID, "Hybrid")] {
            if mode == MODE_HYBRID && frame != 480 {
                continue;
            }
            let pcm = speech_fullband(rate, 3.0);
            let mut enc = CEnc::audio(rate, 1);
            enc.force_mode(mode)
                .bandwidth(BW_FB)
                .bitrate(if mode == MODE_CELT { 64000 } else { 32000 });
            let frames = enc.encode_all(&pcm, frame);
            let packets: Vec<Vec<u8>> = frames
                .chunks_exact(per_packet)
                .map(|g| {
                    if per_packet == 1 {
                        g[0].clone()
                    } else {
                        let refs: Vec<&[u8]> = g.iter().map(|p| p.as_slice()).collect();
                        repacketize(&refs)
                    }
                })
                .collect();
            let packet_len = frame * per_packet;
            // Packets per concealment call.
            let lost_run = call / packet_len;
            assert_eq!(lost_run * packet_len, call);
            let period = (9600 / packet_len).max(lost_run + 4);
            let mut pair = Pair::new(rate, 1);
            let (mut concealed, mut later) = (Vec::new(), Vec::new());
            let mut i = 0;
            while i + lost_run <= packets.len() {
                if i % period == period - lost_run {
                    let (c, r) = pair.decode(None, call);
                    concealed.push(snr(&c, &r));
                    i += lost_run;
                } else {
                    let (c, r) = pair.decode(Some(&packets[i]), packet_len);
                    if power(&c) > 1e-10 {
                        later.push(snr(&c, &r));
                    }
                    i += 1;
                }
            }
            let (median, poor, worst) = median_and_poor(concealed);
            let (later_median, later_poor, later_worst) = median_and_poor(later);
            let case = format!(
                "{name}, {per_packet} x {} ms per packet, concealment calls of {} ms",
                frame as f64 / 48.0,
                call as f64 / 48.0
            );
            println!(
                "{case}: concealed median {median:.1} dB, worst {worst:.1} dB, {poor} poor; decoded frames median {later_median:.1} dB, worst {later_worst:.1} dB, {later_poor} poor"
            );
            let limit = if mode == MODE_CELT {
                PITCH_MEDIAN_DB
            } else {
                NOISE_DB
            };
            if median < limit || poor > 1 || later_poor > 1 {
                failures.push(case);
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Packets that hold two or three 20 ms frames (frame-count codes 1 to 3).
/// A lost packet is concealed with one call for its whole length, which
/// both decoders cut into 20 ms pieces.
#[test]
fn lost_packets_with_several_20_ms_frames() {
    let (rate, frame) = (48000usize, 960usize);
    let mut failures = Vec::new();
    for channels in [1usize, 2] {
        let pcm = stereo_if(speech_fullband(rate, 4.0), channels, rate);
        let frames =
            celt_encoder(rate, channels, BW_FB, 64000 * channels as i32).encode_all(&pcm, frame);
        for per_packet in [2usize, 3] {
            let packets: Vec<Vec<u8>> = frames
                .chunks_exact(per_packet)
                .map(|g| {
                    let refs: Vec<&[u8]> = g.iter().map(|p| p.as_slice()).collect();
                    repacketize(&refs)
                })
                .collect();
            assert!(packets.iter().all(|p| toc(p[0]).code >= 1));
            let packet_len = frame * per_packet;
            for burst in [1usize, 2] {
                let lost: Vec<bool> = (0..packets.len()).map(|i| i % (burst + 6) >= 6).collect();
                let (c, r) = decode_with_loss(&packets, &lost, rate, channels, packet_len);
                let s = loss_stats(&c, &r, &lost, rate, channels, packet_len);
                let case = format!(
                    "{channels} ch, {per_packet} x 20 ms per packet, {burst} lost in a row"
                );
                println!("{case}: {s}");
                if s.concealed.frames < 8
                    || s.concealed.median < PITCH_MEDIAN_DB
                    || s.concealed.poor > 1
                    || s.first_good.median < PITCH_MEDIAN_DB
                    || s.first_good.poor > 1
                    || s.recovered.worst < 100.0
                {
                    failures.push(format!("{case}: {s}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A stereo CELT stream whose bandwidth changes every 200 ms, without
/// losses and with a loss right before or right after each change.
///
/// Bands that a frame does not code lose their energy history in both
/// channels (libopus `celt_decode_with_ec`, "In case start or end were to
/// change"). When the bandwidth widens again, the new bands are predicted
/// from that cleared history.
#[test]
fn stereo_stream_that_changes_bandwidth() {
    let rate = 48000usize;
    let mut failures = Vec::new();
    for frame in [960usize, 240] {
        let pcm = to_stereo(&speech_fullband(rate, 4.0), rate);
        let mut enc = CEnc::audio(rate, 2);
        enc.force_mode(MODE_CELT).bitrate(96000);
        let bandwidths = [
            BW_FB, BW_NB, BW_SWB, BW_WB, BW_FB, BW_WB, BW_NB, BW_FB, BW_SWB,
        ];
        let per = 9600 / frame;
        let packets: Vec<Vec<u8>> = pcm
            .chunks_exact(frame * 2)
            .enumerate()
            .map(|(i, f)| {
                if i % per == 0 {
                    enc.bandwidth(bandwidths[(i / per) % bandwidths.len()]);
                }
                enc.encode(f)
            })
            .collect();
        for name in ["NB", "WB", "SWB", "FB"] {
            assert!(
                packets.iter().any(|p| toc(p[0]).bandwidth == name),
                "no {name} packet: {}",
                toc_summary(&packets)
            );
        }
        for (pattern, lost) in [
            ("no loss", vec![false; packets.len()]),
            (
                "loss before each change",
                (0..packets.len()).map(|i| i % per == per - 1).collect(),
            ),
            (
                "loss after each change",
                (0..packets.len()).map(|i| i % per == 0 && i > 0).collect(),
            ),
        ] {
            let (c, r) = decode_with_loss(&packets, &lost, rate, 2, frame);
            let len = frame * 2;
            let (mut clean, mut concealed, mut first_good) = (Vec::new(), Vec::new(), Vec::new());
            let mut since_loss = usize::MAX;
            for i in 0..packets.len() {
                let (cf, rf) = (&c[i * len..(i + 1) * len], &r[i * len..(i + 1) * len]);
                let db = snr(cf, rf);
                if lost[i] {
                    concealed.push(db);
                    since_loss = 0;
                } else if power(cf) > 1e-10 {
                    if since_loss == 0 {
                        first_good.push(db);
                    } else if since_loss == usize::MAX || since_loss * frame >= 5760 {
                        // No loss so far, or the last one ended 120 ms ago.
                        clean.push(db);
                    }
                    since_loss = since_loss.saturating_add(1);
                }
            }
            let case = format!("{} ms frames, {pattern}", frame as f64 / 48.0);
            let (_, _, clean_worst) = median_and_poor(clean);
            print!("{case}: frames away from a loss worst {clean_worst:.1} dB");
            let mut ok = clean_worst > CLEAN_DB;
            if !concealed.is_empty() {
                let (median, poor, worst) = median_and_poor(concealed);
                let (good_median, good_poor, good_worst) = median_and_poor(first_good);
                print!(
                    "; concealed median {median:.1} dB, worst {worst:.1} dB; first good frame median {good_median:.1} dB, worst {good_worst:.1} dB"
                );
                ok = ok
                    && median > PITCH_MEDIAN_DB
                    && poor <= 1
                    && good_median > PITCH_MEDIAN_DB
                    && good_poor <= 1;
            }
            println!();
            if !ok {
                failures.push(case);
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A stream that changes between mono and stereo packets, with the loss
/// right at the change, in a one-channel and in a two-channel `CeltDecoder`
/// (see tests/celtdec_channels.rs for the method).
///
/// The first frame after a loss lowers the energies it predicts from, in
/// both channels, whatever the frame's own channel count (libopus
/// `celt_decode_with_ec`: the loop runs `while (++c<2)`). A stereo frame
/// after a lost frame and mono frames needs the second channel's share.
#[test]
fn loss_at_a_change_between_mono_and_stereo() {
    let rate = 48000usize;
    let mut failures = Vec::new();
    for frame in [960usize, 240] {
        let pcm = to_stereo(&speech_fullband(rate, 4.0), rate);
        let mut enc = celt_encoder(rate, 2, BW_FB, 96000);
        let per = 4800 / frame;
        let packets: Vec<Vec<u8>> = pcm
            .chunks_exact(frame * 2)
            .enumerate()
            .map(|(i, f)| {
                if i % per == 0 {
                    enc.force_channels(if (i / per) % 2 == 0 { 2 } else { 1 });
                }
                enc.encode(f)
            })
            .collect();
        let stereo_packets = packets.iter().filter(|p| toc(p[0]).stereo).count();
        assert!(stereo_packets > packets.len() / 4 && stereo_packets < packets.len() * 3 / 4);
        for dec_channels in [1usize, 2] {
            for lose_before in [true, false] {
                let mut c_dec = CDecoder::new(rate, dec_channels);
                let mut dec = CeltDecoder::new(default_mode(), dec_channels, rate as i32);
                let mut planar = vec![0f32; frame * dec_channels];
                let (mut concealed, mut first_good, mut other) =
                    (Vec::new(), Vec::new(), Vec::new());
                let mut prev_lost = false;
                for (i, p) in packets.iter().enumerate() {
                    let lost = if lose_before {
                        i % per == per - 1
                    } else {
                        i % per == 0 && i > 0
                    };
                    let (c, n) = if lost {
                        (
                            c_dec.decode(None, frame),
                            dec.decode_lost(frame, &mut planar, PcmLayout::Planar).unwrap(),
                        )
                    } else {
                        dec.set_stream_channels(if toc(p[0]).stereo { 2 } else { 1 })
                            .unwrap();
                        let payload = &p[1..];
                        let mut rc = RangeCoder::new_decoder(payload);
                        let bits = (payload.len() * 8) as i32;
                        (
                            c_dec.decode(Some(p), frame),
                            dec.decode(&mut rc, bits, frame, &mut planar, PcmLayout::Planar),
                        )
                    };
                    assert_eq!(n, frame);
                    let r = interleave(&planar, dec_channels);
                    let db = snr(&c, &r);
                    if lost {
                        concealed.push(db);
                    } else if prev_lost {
                        first_good.push(db);
                    } else if power(&c) > 1e-10 {
                        other.push(db);
                    }
                    prev_lost = lost;
                }
                let case = format!(
                    "{} ms frames, {dec_channels}-channel decoder, loss {} each change",
                    frame as f64 / 48.0,
                    if lose_before { "before" } else { "after" }
                );
                let (median, poor, worst) = median_and_poor(concealed);
                let (good_median, good_poor, good_worst) = median_and_poor(first_good);
                let (other_median, other_poor, other_worst) = median_and_poor(other);
                println!(
                    "{case}: concealed median {median:.1} dB, worst {worst:.1} dB; first good frame median {good_median:.1} dB, worst {good_worst:.1} dB; other frames median {other_median:.1} dB, worst {other_worst:.1} dB"
                );
                if median < PITCH_MEDIAN_DB
                    || poor > 1
                    || good_median < PITCH_MEDIAN_DB
                    || good_poor > 1
                    || other_median < CLEAN_DB
                    || other_poor > 0
                {
                    failures.push(case);
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A voiced-like tone: 30 harmonics of `f0` with a slow vibrato. The encoder
/// switches its pitch post-filter on for such a signal.
fn harmonic(rate: usize, seconds: f64, f0: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let n = (rate as f64 * seconds) as usize;
    let mut phase = 0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let f = f0 * (1.0 + 0.01 * (2.0 * PI * 3.0 * t).sin());
            phase += 2.0 * PI * f / rate as f64;
            let mut v = 0.0;
            for h in 1..=30 {
                if f * (h as f64) < 0.45 * rate as f64 {
                    v += (phase * h as f64).sin() / h as f64;
                }
            }
            (0.2 * v) as f32
        })
        .collect()
}

/// Decaying tone bursts at irregular times over a low noise floor.
fn clicks(rate: usize, seconds: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let n = (rate as f64 * seconds) as usize;
    let mut seed = 4242u32;
    let mut out = vec![0f32; n];
    let (mut next, mut k) = (0usize, 0usize);
    while next < n {
        let f = [900.0, 2500.0, 5200.0, 400.0][k % 4];
        let tau = [0.004, 0.02, 0.008, 0.05][k % 4] * rate as f64;
        for (i, v) in out
            .iter_mut()
            .enumerate()
            .skip(next)
            .take((8.0 * tau) as usize)
        {
            let t = (i - next) as f64;
            *v += (0.6 * (-t / tau).exp() * (2.0 * PI * f * t / rate as f64).sin()) as f32;
        }
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        next += rate / 20 + (seed >> 20) as usize * rate / 40960;
        k += 1;
    }
    for v in out.iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *v += ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.001;
    }
    out
}

/// Whether a CELT-only packet with one frame switches the pitch post-filter
/// on (the first symbols of a CELT frame: silence flag, then the
/// post-filter flag).
fn postfilter_on(packet: &[u8]) -> bool {
    let payload = &packet[1..];
    let total = (payload.len() * 8) as i32;
    let mut rc = RangeCoder::new_decoder(payload);
    if rc.tell() == 1 && rc.decode_bit_logp(15) {
        return false;
    }
    rc.tell() + 16 <= total && rc.decode_bit_logp(1)
}

/// Losses in signals that the other tests do not use: a strong steady
/// pitch (post-filter on in nearly every frame), a higher pitch, decaying
/// clicks, a full-scale tone, input 5 dB above full scale, very quiet
/// speech and a chirp. Low and high bit rates, constant and variable, at
/// every frame size, mono and stereo, single losses and runs of three.
#[test]
fn losses_in_signals_with_pitch_clicks_and_extreme_levels() {
    let rate = 48000usize;
    let seconds = 2.0;
    let n = (rate as f64 * seconds) as usize;
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("pitch 140 Hz", harmonic(rate, seconds, 140.0)),
        ("pitch 310 Hz", harmonic(rate, seconds, 310.0)),
        ("clicks", clicks(rate, seconds)),
        ("full scale", full_scale(rate, n)),
        (
            "5 dB above full scale",
            full_scale(rate, n).iter().map(|v| v * 1.8).collect(),
        ),
        (
            "quiet speech",
            speech_fullband(rate, seconds)
                .iter()
                .map(|v| v * 0.002)
                .collect(),
        ),
        ("chirp", chirp(rate, n)),
    ];
    let mut failures = Vec::new();
    let (mut concealed_frames, mut poor_frames, mut with_postfilter) = (0usize, 0usize, 0usize);
    for (name, mono) in &signals {
        for channels in [1usize, 2] {
            let pcm = stereo_if(mono.clone(), channels, rate);
            for (frame, bitrate, vbr) in [
                (960usize, 24000, true),
                (480, 32000, false),
                (240, 48000, true),
                (120, 64000, false),
            ] {
                let mut enc = celt_encoder(rate, channels, BW_FB, bitrate * channels as i32);
                enc.vbr(vbr);
                let packets = enc.encode_all(&pcm, frame);
                let on = packets.iter().filter(|p| postfilter_on(p)).count();
                if name.starts_with("pitch") {
                    assert!(
                        on * 10 > packets.len() * 8,
                        "{name}: post-filter on in only {on} of {} frames",
                        packets.len()
                    );
                    with_postfilter += on;
                }
                for burst in [1usize, 3] {
                    let gap = (7200 / frame).max(8);
                    let lost: Vec<bool> = (0..packets.len())
                        .map(|i| i % (burst + gap) >= gap)
                        .collect();
                    let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                    let s = loss_stats(&c, &r, &lost, rate, channels, frame);
                    let case = format!(
                        "{name}, {channels} ch, {} ms at {bitrate} b/s, post-filter on in {on} of {} frames, {burst} lost in a row",
                        frame as f64 / 48.0,
                        packets.len()
                    );
                    println!("{case}: {s}");
                    concealed_frames += s.concealed.frames;
                    poor_frames += s.concealed.poor;
                    // The chirp and the clicks change faster than the other
                    // signals, so rounding weighs more: the limit on the
                    // median is 5 dB lower here. A run of lost frames keeps
                    // the pitch period and the predictor of its first
                    // frame, so one badly matched fit costs the whole run.
                    if s.concealed.frames < 5
                        || s.concealed.median < PITCH_MEDIAN_DB - 5.0
                        || s.concealed.poor > burst + 1
                        || s.first_good.median < PITCH_MEDIAN_DB - 5.0
                        || s.first_good.poor > 1
                        || (s.concealed.poor == 0 && s.level_diff_db > 0.2)
                    {
                        failures.push(format!("{case}: {s}"));
                    }
                }
            }
        }
    }
    println!(
        "{concealed_frames} concealed frames, {poor_frames} of them below {POOR_DB} dB; post-filter on in {with_postfilter} frames of the pitched signals"
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(concealed_frames > 2000);
    assert!(
        poor_frames * 100 <= concealed_frames,
        "{poor_frames} of {concealed_frames} concealed frames below {POOR_DB} dB"
    );
}

/// A stream from the C encoder that changes mode every 200 ms (CELT-only,
/// SILK-only and Hybrid, several bandwidths), with one packet lost at every
/// position from two packets before to two packets after each change.
///
/// libopus signals most changes with a redundancy frame in the last
/// SILK-only or Hybrid packet. When that packet is lost, the decoder
/// bridges the change with 5 ms of concealment in the old mode. When the
/// packet after it is lost, the concealment runs in CELT-only mode on the
/// 5 ms that the redundancy frame left in the CELT decoder.
#[test]
fn loss_next_to_a_mode_change() {
    let mut failures = Vec::new();
    let mut bridge_frames = 0;
    for (rate, channels) in [(48000usize, 1usize), (48000, 2), (24000, 1)] {
        let pcm = stereo_if(speech_fullband(rate, 6.0), channels, rate);
        for frame_ms in [20usize, 10] {
            let frame = rate * frame_ms / 1000;
            let mut enc = CEnc::audio(rate, channels);
            enc.bitrate(if channels == 1 { 32000 } else { 48000 });
            let fullband = if rate == 48000 { BW_FB } else { BW_SWB };
            let sequence = [
                (MODE_CELT, fullband),
                (MODE_SILK, BW_WB),
                (MODE_HYBRID, BW_SWB),
                (MODE_CELT, BW_SWB),
                (MODE_HYBRID, fullband),
                (MODE_SILK, BW_NB),
                (MODE_CELT, BW_WB),
                (MODE_HYBRID, BW_SWB),
                (MODE_SILK, BW_WB),
            ];
            let per = 200 / frame_ms;
            let packets: Vec<Vec<u8>> = pcm
                .chunks_exact(frame * channels)
                .enumerate()
                .map(|(i, f)| {
                    if i % per == 0 {
                        let (mode, bandwidth) = sequence[(i / per) % sequence.len()];
                        enc.force_mode(mode).bandwidth(bandwidth);
                    }
                    enc.encode(f)
                })
                .collect();
            for mode in ["CELT", "SILK", "Hybrid"] {
                assert!(packets.iter().any(|p| toc(p[0]).mode == mode));
            }
            for offset in [-2i32, -1, 0, 1, 2] {
                let lost: Vec<bool> = (0..packets.len())
                    .map(|i| i >= per && (i as i32 - offset).rem_euclid(per as i32) == 0)
                    .collect();
                let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                let len = frame * channels;
                let (mut worst, mut worst_away, mut poor) = (200f64, 200f64, 0);
                let mut since_loss = usize::MAX;
                for i in 0..packets.len() {
                    let (cf, rf) = (&c[i * len..(i + 1) * len], &r[i * len..(i + 1) * len]);
                    let db = snr(cf, rf);
                    if lost[i] {
                        since_loss = 0;
                    } else {
                        since_loss = since_loss.saturating_add(1);
                    }
                    if power(cf) < 1e-10 {
                        continue;
                    }
                    // The 5 ms bridge from Hybrid to CELT-only when the
                    // Hybrid packet with the redundancy frame was lost. It
                    // conceals 5 ms of a Hybrid stream, see
                    // `concealment_calls_shorter_than_10_ms_in_a_hybrid_stream`.
                    let bridge_from_hybrid = i >= 2
                        && lost[i - 1]
                        && toc(packets[i][0]).mode == "CELT"
                        && toc(packets[i - 2][0]).mode == "Hybrid";
                    if bridge_from_hybrid {
                        bridge_frames += 1;
                    }
                    worst = worst.min(db);
                    if db < POOR_DB {
                        poor += 1;
                    }
                    if since_loss.saturating_mul(frame_ms) >= 120 {
                        worst_away = worst_away.min(db);
                    }
                }
                let case = format!(
                    "{rate} Hz, {channels} ch, {frame_ms} ms frames, loss at {offset} from each change"
                );
                println!(
                    "{case}: worst frame {worst:.1} dB, {poor} below {POOR_DB} dB; 120 ms and more after a loss {worst_away:.1} dB"
                );
                if poor > 1 || worst_away < CLEAN_DB {
                    failures.push(case);
                }
            }
        }
    }
    println!("{bridge_frames} frames with a Hybrid to CELT-only bridge after a loss");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(bridge_frames > 0);
}

/// A CELT frame of one byte, or none, is a lost frame (libopus
/// `celt_decode_with_ec`: `data == NULL || len <= 1`): the decoder conceals
/// instead of reading the byte. The reference is `opus_decode_float` of a
/// two-byte or one-byte packet, which libopus also conceals.
#[test]
fn celt_frame_of_one_byte_is_concealed() {
    let (rate, frame) = (48000usize, 960usize);
    let pcm = speech_fullband(rate, 2.0);
    let packets = celt_encoder(rate, 1, BW_FB, 64000).encode_all(&pcm, frame);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec = CeltDecoder::new(default_mode(), 1, rate as i32);
    let mut out = vec![0f32; frame];
    // Decode all of `payload` into `out`.
    let decode = |dec: &mut CeltDecoder, payload: &[u8], out: &mut [f32]| {
        let mut rc = RangeCoder::new_decoder(payload);
        let bits = (payload.len() * 8) as i32;
        dec.decode(&mut rc, bits, frame, out, PcmLayout::Planar)
    };
    let (mut one_byte, mut no_byte) = (Vec::new(), Vec::new());
    for (i, p) in packets.iter().enumerate() {
        let short = [p[0], 0x5a];
        let c = match i % 7 {
            5 => {
                // One payload byte.
                assert_eq!(decode(&mut dec, &short[1..], &mut out), frame);
                c_dec.decode(Some(&short), frame)
            }
            2 => {
                // No payload byte.
                assert_eq!(decode(&mut dec, &[], &mut out), frame);
                c_dec.decode(Some(&short[..1]), frame)
            }
            _ => {
                assert_eq!(decode(&mut dec, &p[1..], &mut out), frame);
                c_dec.decode(Some(p), frame)
            }
        };
        let db = snr(&c, &out);
        match i % 7 {
            5 => {
                // A decoded byte 0x5a would be a frame of near silence.
                assert!(
                    (dbfs(&c) - dbfs(&out)).abs() < 1.0,
                    "packet {i}: level {:.1} dBFS, libopus {:.1} dBFS",
                    dbfs(&out),
                    dbfs(&c)
                );
                one_byte.push(db)
            }
            2 => no_byte.push(db),
            _ => {}
        }
    }
    let (median_one, poor_one, worst_one) = median_and_poor(one_byte);
    let (median_none, poor_none, worst_none) = median_and_poor(no_byte);
    println!(
        "one byte: median {median_one:.1} dB, worst {worst_one:.1} dB; no byte: median {median_none:.1} dB, worst {worst_none:.1} dB"
    );
    assert!(median_one > PITCH_MEDIAN_DB && poor_one <= 1);
    assert!(median_none > PITCH_MEDIAN_DB && poor_none <= 1);
}
