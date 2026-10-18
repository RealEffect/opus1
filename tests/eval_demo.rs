//! The `opus_demo -d` loop of `opus1::internals::eval::demo`, and the two
//! libopus functions it needs that opus1 now has too:
//! `OpusDecoder::last_packet_duration` (`OPUS_GET_LAST_PACKET_DURATION`)
//! and `packet::has_lbrr` (`opus_packet_has_lbrr`).
//!
//! The same loop drives opus1's decoder and libopus's
//! (`eval_common::libopus_demo`), on CELT, SILK, Hybrid and mode-switching
//! streams with lost packets, FEC and bursts of losses, decoded at several
//! rates. All of them give the same 16-bit output against the libopus the
//! tests link on 64-bit ARM macOS. (docs/10, "Decoder", names two other FEC
//! streams that agree with libopus to about 185 dB only; a stream like those
//! would show up here as samples one step apart.) Elsewhere, streams with
//! losses are only reported: a libopus built at -O2 conceals with other
//! last bits.

#![cfg(feature = "std")]

mod core_common;
mod eval_common;
mod k24_common;

use core_common::{APP_AUDIO, APP_VOIP, BW_FB, BW_SWB, BW_WB, CEncoder, MODE_CELT, MODE_HYBRID, MODE_SILK};
use eval_common::libopus_demo::{LibopusDecoder, libopus_has_lbrr};
use k24_common::{repacketize, speech_fullband, to_stereo};
use opus1::OpusDecoder;
use opus1::internals::eval::demo::{
    BitPacket, DemoDecoder, DemoOutput, decode_bitstream, opus_demo_s16, parse_bitstream, res2int24,
    write_record,
};
use opus1::packet;

/// A libopus stream: packets and the encoder's final range after each.
struct Stream {
    name: &'static str,
    rate: usize,
    channels: usize,
    packets: Vec<Vec<u8>>,
    ranges: Vec<u32>,
}

/// Speech up to the Nyquist frequency of `rate`, `seconds` long.
fn signal(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let mono = speech_fullband(rate, seconds);
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

fn encode(
    name: &'static str,
    rate: usize,
    channels: usize,
    frame_ms: f64,
    setup: impl Fn(&mut CEncoder),
) -> Stream {
    let mut enc = CEncoder::new(rate, channels, APP_VOIP);
    setup(&mut enc);
    let pcm = signal(rate, channels, 4.0);
    let frame = (rate as f64 * frame_ms / 1000.0) as usize;
    let (packets, ranges) = enc.encode_all_with_ranges(&pcm, frame);
    Stream { name, rate, channels, packets, ranges }
}

fn streams() -> Vec<Stream> {
    vec![
        encode("celt_fb_stereo_20ms", 48000, 2, 20.0, |e| {
            e.force_mode(MODE_CELT).bitrate(64000);
        }),
        encode("celt_fb_mono_5ms", 48000, 1, 5.0, |e| {
            e.force_mode(MODE_CELT).bitrate(48000);
        }),
        encode("silk_wb_mono_20ms", 16000, 1, 20.0, |e| {
            e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(20000);
        }),
        encode("silk_wb_mono_60ms", 16000, 1, 60.0, |e| {
            e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(20000);
        }),
        encode("silk_wb_mono_20ms_fec", 16000, 1, 20.0, |e| {
            e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(28000).fec(true, 20);
        }),
        encode("silk_wb_stereo_40ms_fec", 48000, 2, 40.0, |e| {
            e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(40000).fec(true, 20);
        }),
        encode("hybrid_swb_stereo_20ms_fec", 48000, 2, 20.0, |e| {
            e.force_mode(MODE_HYBRID).bandwidth(BW_SWB).bitrate(48000).fec(true, 15);
        }),
        encode("auto_fb_stereo_20ms", 48000, 2, 20.0, |e| {
            e.bandwidth(BW_FB).bitrate(32000).fec(true, 10);
        }),
    ]
}

/// Which packets the bitstream loses: single losses and bursts of three.
fn lost(i: usize) -> bool {
    i % 9 == 4 || (20..23).contains(&(i % 31))
}

/// The `opus_demo` bitstream of a stream, with the packets `lose` says
/// lost (written as `opus_demo -e` writes a lost packet: length 0, range
/// 0).
fn bitstream(stream: &Stream, lose: impl Fn(usize) -> bool) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, (p, &r)) in stream.packets.iter().zip(&stream.ranges).enumerate() {
        if lose(i) {
            write_record(&mut out, &[], 0);
        } else {
            write_record(&mut out, p, r);
        }
    }
    out
}

fn run<D: DemoDecoder>(decoder: &mut D, bytes: &[u8]) -> DemoOutput {
    let parsed = parse_bitstream(bytes);
    assert_eq!(parsed.error, None);
    decode_bitstream(decoder, &parsed.packets)
}

/// Samples that differ, and the largest difference.
fn differences(a: &[i16], b: &[i16]) -> (usize, i32) {
    assert_eq!(a.len(), b.len(), "output lengths");
    let mut count = 0;
    let mut max = 0;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x as i32 - y as i32).abs();
        count += (d != 0) as usize;
        max = max.max(d);
    }
    (count, max)
}

#[test]
fn loop_gives_the_same_output_for_opus1_and_libopus() {
    for stream in streams() {
        let mut configs = vec![(stream.rate as u32, stream.channels)];
        for c in [(16000, 1), (48000, 2), (8000, 2)] {
            if !configs.contains(&c) {
                configs.push(c);
            }
        }
        for (rate, channels) in configs {
            for lossy in [false, true] {
                let bytes = bitstream(&stream, |i| lossy && lost(i));
                let mut ours = OpusDecoder::new(rate as i32, channels).unwrap();
                let mut theirs = LibopusDecoder::new(rate, channels);
                let a = run(&mut ours, &bytes);
                let b = run(&mut theirs, &bytes);
                let label = format!("{} at {rate} Hz, {channels} ch, losses {lossy}", stream.name);
                assert_eq!(a.range_mismatch, None, "{label}: opus1 range");
                assert_eq!(b.range_mismatch, None, "{label}: libopus range");
                assert_eq!(
                    (a.packets, a.lost, a.fec_calls, a.decode_errors, a.range_checks),
                    (b.packets, b.lost, b.fec_calls, b.decode_errors, b.range_checks),
                    "{label}: counts"
                );
                assert_eq!(a.decode_errors, 0, "{label}");
                assert!(a.range_checks > 0, "{label}");
                let (count, max) = differences(&a.pcm, &b.pcm);
                println!(
                    "{label}: {} samples, {} lost, {} FEC calls, {} range checks, {count} differ (max {max})",
                    a.pcm.len(),
                    a.lost,
                    a.fec_calls,
                    a.range_checks
                );
                // Concealment is compared at the last bit only against the
                // libopus the tests link on 64-bit ARM macOS (Homebrew,
                // without vector loops). A libopus built at -O2, as on the
                // phone, conceals CELT frames with other last bits
                // (docs/09-testing.md, "Results on the phone").
                if !lossy || eval_common::exact_platform() {
                    assert_eq!(count, 0, "{label}: output differs (largest difference {max})");
                }
                if lossy && stream.name.contains("fec") {
                    assert!(a.fec_calls > 0, "{label}: no FEC call");
                }
            }
        }
    }
}

#[test]
fn a_corrupted_range_stops_the_loop_at_its_packet() {
    let stream = &streams()[0];
    let n = stream.packets.len();
    let packet_samples = 960 * 2;
    for (target, lose_before) in [(0usize, false), (5, false), (13, true), (n - 1, false)] {
        let mut bytes = Vec::new();
        for (i, (p, &r)) in stream.packets.iter().zip(&stream.ranges).enumerate() {
            if lose_before && i + 1 == target {
                write_record(&mut bytes, &[], 0);
            } else {
                write_record(&mut bytes, p, if i == target { r ^ 0x100 } else { r });
            }
        }
        for decoder in [&mut OpusDecoder::new(48000, 2).unwrap() as &mut dyn DemoDecoder, &mut LibopusDecoder::new(48000, 2)] {
            let out = run_dyn(decoder, &bytes);
            if target == 0 || lose_before {
                // The first packet and a packet after a loss are not checked.
                assert_eq!(out.range_mismatch, None, "packet {target}");
                assert_eq!(out.packets, n);
            } else {
                let m = out.range_mismatch.expect("a mismatch");
                assert_eq!(m.packet, target);
                assert_eq!(m.encoder, stream.ranges[target] ^ 0x100);
                assert_eq!(m.decoder, stream.ranges[target]);
                assert_eq!(out.packets, target + 1);
                // The packet itself was decoded and written before the check.
                assert_eq!(out.pcm.len(), (target + 1) * packet_samples);
            }
        }
    }
}

fn run_dyn(decoder: &mut dyn DemoDecoder, bytes: &[u8]) -> DemoOutput {
    let parsed = parse_bitstream(bytes);
    decode_bitstream(decoder, &parsed.packets)
}

/// One record of a bitstream: payload and encoder final range.
type Record = (Vec<u8>, u32);

#[test]
fn invalid_packets_and_early_losses() {
    let stream = &streams()[4];
    let p = &stream.packets;
    // A code 3 packet that announces 0 frames is invalid.
    let invalid = [p[10][0] | 0x3, 0x00];
    let cases: Vec<(&str, Vec<Record>)> = vec![
        // Losses before the first packet: the duration is still 0, so the
        // concealment calls fail and nothing is written for them.
        ("loss_first", vec![(vec![], 0), (vec![], 0), (p[2].clone(), 0), (p[3].clone(), 0)]),
        // A loss, then an invalid packet: opus_demo takes the error of
        // opus_packet_has_lbrr as "has LBRR" and tries FEC with it.
        ("invalid_after_loss", vec![(p[0].clone(), 0), (vec![], 0), (invalid.to_vec(), 0), (p[3].clone(), 0)]),
        ("invalid", vec![(p[0].clone(), 0), (invalid.to_vec(), 0), (p[2].clone(), 0)]),
        // A loss at the end is never decoded.
        ("loss_last", vec![(p[0].clone(), 0), (p[1].clone(), 0), (vec![], 0)]),
    ];
    for (name, records) in cases {
        let mut bytes = Vec::new();
        for (data, range) in &records {
            write_record(&mut bytes, data, *range);
        }
        let a = run(&mut OpusDecoder::new(16000, 1).unwrap(), &bytes);
        let b = run(&mut LibopusDecoder::new(16000, 1), &bytes);
        println!("{name}: errors {}, FEC calls {}, samples {}", a.decode_errors, a.fec_calls, a.pcm.len());
        assert_eq!((a.decode_errors, a.fec_calls, a.lost), (b.decode_errors, b.fec_calls, b.lost), "{name}");
        assert_eq!(a.pcm, b.pcm, "{name}");
        match name {
            "loss_first" => assert_eq!((a.decode_errors, a.pcm.len()), (2, 640)),
            "invalid_after_loss" => assert_eq!((a.decode_errors, a.fec_calls, a.pcm.len()), (2, 1, 640)),
            "invalid" => assert_eq!((a.decode_errors, a.pcm.len()), (1, 640)),
            _ => assert_eq!((a.decode_errors, a.pcm.len()), (0, 640)),
        }
    }
}

#[test]
fn a_cut_bitstream_decodes_its_prefix() {
    let stream = &streams()[2];
    let bytes = bitstream(stream, lost);
    let full = run(&mut OpusDecoder::new(16000, 1).unwrap(), &bytes);
    // Cut in the middle of the payload of a later record.
    let cut = &bytes[..bytes.len() / 2];
    let parsed = parse_bitstream(cut);
    assert!(parsed.error.is_some());
    let prefix = decode_bitstream(&mut OpusDecoder::new(16000, 1).unwrap(), &parsed.packets);
    assert!(!prefix.pcm.is_empty());
    assert_eq!(&full.pcm[..prefix.pcm.len()], &prefix.pcm[..]);
}

/// Settings applied to a new encoder.
type Setup = Box<dyn Fn(&mut CEncoder)>;

#[test]
fn has_lbrr_matches_libopus() {
    let mut checked = [0usize; 3];
    let mut check = |p: &[u8]| {
        let want = libopus_has_lbrr(p);
        let got = packet::has_lbrr(p);
        match got {
            Ok(true) => assert_eq!(want, 1, "{p:02x?}"),
            Ok(false) => assert_eq!(want, 0, "{p:02x?}"),
            Err(_) => assert!(want < 0, "{p:02x?}: libopus {want}"),
        }
        checked[match want {
            1 => 0,
            0 => 1,
            _ => 2,
        }] += 1;
    };
    // Real packets: SILK and Hybrid with and without FEC, mono and stereo,
    // 10 to 60 ms, CELT, and multi-frame packets.
    let mut encoders: Vec<(usize, usize, f64, Setup)> = Vec::new();
    for ch in [1, 2] {
        for ms in [10.0, 20.0, 40.0, 60.0] {
            encoders.push((16000, ch, ms, Box::new(|e: &mut CEncoder| {
                e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(30000).fec(true, 25);
            })));
            encoders.push((16000, ch, ms, Box::new(|e: &mut CEncoder| {
                e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(30000);
            })));
        }
        for ms in [10.0, 20.0] {
            encoders.push((48000, ch, ms, Box::new(|e: &mut CEncoder| {
                e.force_mode(MODE_HYBRID).bandwidth(BW_FB).bitrate(48000).fec(true, 25);
            })));
        }
        for ms in [2.5, 5.0, 20.0] {
            encoders.push((48000, ch, ms, Box::new(|e: &mut CEncoder| {
                e.force_mode(MODE_CELT).bitrate(64000);
            })));
        }
    }
    for (rate, ch, ms, setup) in &encoders {
        let mut enc = CEncoder::new(*rate, *ch, APP_AUDIO);
        setup(&mut enc);
        let pcm = signal(*rate, *ch, 1.0);
        let packets = enc.encode_all(&pcm, (*rate as f64 * ms / 1000.0) as usize);
        for p in &packets {
            check(p);
        }
        if *ms <= 20.0 {
            for [a, b, c] in packets.as_chunks::<3>().0 {
                check(&repacketize(&[a, b, c]));
                check(&repacketize(&[a, b]));
            }
        }
    }
    // Every first byte with short random payloads, valid or not.
    let mut rng = eval_common::cases::XorShift(5);
    for toc in 0..=255u8 {
        for len in 1..12 {
            for _ in 0..4 {
                let mut p = vec![toc];
                p.extend((1..len).map(|_| rng.next() as u8));
                check(&p);
            }
        }
    }
    println!("has_lbrr: {} with LBRR, {} without, {} invalid", checked[0], checked[1], checked[2]);
    assert!(checked.iter().all(|&c| c > 50), "{checked:?}");
    assert!(packet::has_lbrr(&[]).is_err());
}

/// One step of `last_packet_duration_matches_libopus`.
enum Op<'a> {
    Decode(&'a [u8], usize),
    Fec(&'a [u8], usize),
    Lose(usize),
    Decode16(&'a [u8], usize),
    Fec16(&'a [u8], usize),
    Lose16(usize),
    Reset,
}

#[test]
fn last_packet_duration_matches_libopus() {
    let celt20 = encode("celt20", 48000, 2, 20.0, |e| {
        e.force_mode(MODE_CELT).bitrate(64000);
    });
    let celt5 = encode("celt5", 48000, 2, 5.0, |e| {
        e.force_mode(MODE_CELT).bitrate(64000);
    });
    let silk20 = encode("silk20", 48000, 2, 20.0, |e| {
        e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(30000).fec(true, 25);
    });
    let silk60 = encode("silk60", 48000, 1, 60.0, |e| {
        e.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(30000).fec(true, 25);
    });
    let hybrid10 = encode("hybrid10", 48000, 2, 10.0, |e| {
        e.force_mode(MODE_HYBRID).bandwidth(BW_SWB).bitrate(40000).fec(true, 25);
    });
    let multi = repacketize(&[&silk20.packets[20], &silk20.packets[21], &silk20.packets[22]]);
    let invalid = [silk20.packets[5][0] | 0x3, 0x00];
    let s = &silk20.packets;
    let ops = [
        Op::Lose(480),
        Op::Decode(&celt20.packets[3], 5760),
        Op::Decode(&celt5.packets[4], 5760),
        Op::Lose(960),
        Op::Lose(120),
        Op::Decode(&silk60.packets[3], 5760),
        Op::Decode(&s[10], 5760),
        Op::Fec(&s[12], 960),
        Op::Fec(&s[13], 1920),
        Op::Fec(&s[14], 480),
        Op::Fec(&celt20.packets[5], 480),
        Op::Decode(&invalid, 5760),
        Op::Fec(&s[15], 1000),
        Op::Decode(&multi, 5760),
        Op::Decode(&multi, 960),
        Op::Decode(&hybrid10.packets[30], 5760),
        Op::Fec(&hybrid10.packets[32], 1440),
        Op::Reset,
        Op::Lose(240),
        Op::Decode(&s[30], 5760),
        Op::Lose16(9600),
        Op::Decode16(&s[31], 5760),
        Op::Fec16(&s[33], 7680),
        Op::Fec16(&s[34], 960),
        Op::Decode16(&invalid, 5760),
        Op::Reset,
        Op::Lose16(480),
    ];
    for (rate, channels) in [(48000u32, 2usize), (16000, 1), (8000, 2)] {
        let scale = |n: usize| n * rate as usize / 48000;
        let mut ours = OpusDecoder::new(rate as i32, channels).unwrap();
        let mut theirs = LibopusDecoder::new(rate, channels);
        assert_eq!(ours.last_packet_duration(), 0);
        let mut f32_buf = vec![0f32; 9600 * channels];
        let mut i16_buf = vec![0i16; 9600 * channels];
        for (step, op) in ops.iter().enumerate() {
            let (a, b) = match *op {
                Op::Decode(p, n) => (
                    ours.decode(p, scale(n), &mut f32_buf).is_ok(),
                    theirs.decode_float(p, scale(n), false).is_ok(),
                ),
                Op::Fec(p, n) => (
                    ours.decode_fec(p, scale(n), &mut f32_buf).is_ok(),
                    theirs.decode_float(p, scale(n), true).is_ok(),
                ),
                Op::Lose(n) => (
                    ours.decode(&[], scale(n), &mut f32_buf).is_ok(),
                    theirs.decode_float(&[], scale(n), false).is_ok(),
                ),
                Op::Decode16(p, n) => (
                    ours.decode_i16(p, scale(n), &mut i16_buf).is_ok(),
                    theirs.decode16(p, scale(n), false).is_ok(),
                ),
                Op::Fec16(p, n) => (
                    ours.decode_fec_i16(p, scale(n), &mut i16_buf).is_ok(),
                    theirs.decode16(p, scale(n), true).is_ok(),
                ),
                Op::Lose16(n) => (
                    ours.decode_i16(&[], scale(n), &mut i16_buf).is_ok(),
                    theirs.decode16(&[], scale(n), false).is_ok(),
                ),
                Op::Reset => {
                    ours.reset_state();
                    theirs.reset();
                    (true, true)
                }
            };
            assert_eq!(a, b, "{rate} Hz, step {step}: success");
            assert_eq!(
                DemoDecoder::last_packet_duration(&ours),
                theirs.last_packet_duration(),
                "{rate} Hz, step {step}"
            );
        }
    }
}

#[test]
fn sixteen_bit_output_matches_opus_decode24() {
    // libopus decodes the same packets to float and to 24 bits: the 24-bit
    // samples must be res2int24 of the float samples, and opus_demo_s16 the
    // 16-bit samples opus_demo writes from them. The input is speech at 2.5
    // times full scale, so that the decoded signal overshoots.
    let mut enc = CEncoder::new(48000, 2, APP_AUDIO);
    enc.force_mode(MODE_CELT).bitrate(96000);
    let pcm: Vec<f32> = signal(48000, 2, 2.0).iter().map(|&x| x * 2.5).collect();
    let packets = enc.encode_all(&pcm, 960);
    let mut float_dec = LibopusDecoder::new(48000, 2);
    let mut int_dec = LibopusDecoder::new(48000, 2);
    let (mut samples, mut beyond) = (0, 0);
    for p in &packets {
        let f = float_dec.decode_float(p, 5760, false).unwrap();
        let i = int_dec.decode24_vec(p, 5760, false).unwrap();
        assert_eq!(f.len(), i.len());
        for (&x, &y) in f.iter().zip(&i) {
            assert_eq!(res2int24(x), y, "sample {samples}");
            assert_eq!(opus_demo_s16(x), ((y.clamp(-0x7fff00, 0x7fff00) + 128) >> 8) as i16);
            beyond += (x.abs() > 1.0) as usize;
            samples += 1;
        }
    }
    println!("{samples} samples, {beyond} beyond full scale");
    assert!(beyond > 100);
    // The loop through opus1 never writes -32768.
    let records: Vec<BitPacket<'_>> = packets.iter().map(|p| BitPacket { data: p, enc_final_range: 0 }).collect();
    let out = decode_bitstream(&mut OpusDecoder::new(48000, 2).unwrap(), &records);
    assert!(out.pcm.contains(&32767) && out.pcm.contains(&-32767));
    assert!(!out.pcm.contains(&i16::MIN));
}
