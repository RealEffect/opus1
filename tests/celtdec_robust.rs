//! Robustness of the CELT decoder: extreme arguments, damaged frames and
//! lost frames. Nothing may panic, and where libopus has an equivalent call
//! the two must agree.
//!
//! A damaged CELT frame is still a frame that can be decoded: the range
//! decoder reads whatever bits there are. libopus decodes such a frame
//! without an error, and so does opus1. The two must then read the same
//! symbols, which shows as an equal final range, and produce the same
//! samples.
//!
//! Run with `--nocapture` to see the measured numbers.

mod celtdec_common;
mod k24_common;
use audiopus_sys as sys;
use celtdec_common::*;
use k24_common::*;
use opus1::internals::celt::{CeltDecoder, PcmLayout};
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;

/// Decode all of `payload` as one CELT frame of `frame` samples per channel.
fn decode_all(
    dec: &mut CeltDecoder,
    payload: &[u8],
    frame: usize,
    out: &mut [f32],
    layout: PcmLayout,
) -> usize {
    let mut rc = RangeCoder::new_decoder(payload);
    dec.decode(&mut rc, (payload.len() * 8) as i32, frame, out, layout)
}

/// A small deterministic random number generator.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

/// A band range without a band is a bad argument: libopus accepts a start
/// band of 0 to 20 and an end band of 1 to 21, and decodes only when the
/// first is below the second. opus1 used to divide by zero in the bit
/// allocation for such a range.
#[test]
fn empty_band_range_is_rejected() {
    let frame = [0x5au8; 60];
    for channels in [1usize, 2] {
        let mut dec = CeltDecoder::new(default_mode(), channels, 48000);
        let mut pcm = vec![0f32; 960 * channels];
        // Bands out of range: the setters refuse them.
        assert!(dec.set_start_band(21).is_err());
        assert!(dec.set_end_band(0).is_err());
        assert!(dec.set_end_band(22).is_err());
        // A range without a band: decoding refuses it.
        for (start, end) in [(17usize, 17usize), (17, 13), (5, 3), (20, 20)] {
            dec.set_start_band(start).unwrap();
            dec.set_end_band(end).unwrap();
            let n = decode_all(&mut dec, &frame, 960, &mut pcm, PcmLayout::Planar);
            assert_eq!(n, 0, "bands {start}..{end}");
        }
        // The decoder still works, and a range of one band is accepted.
        for (start, end) in [(0usize, 21usize), (17, 21), (20, 21), (0, 1), (17, 18)] {
            dec.set_start_band(start).unwrap();
            dec.set_end_band(end).unwrap();
            let n = decode_all(&mut dec, &frame, 960, &mut pcm, PcmLayout::Planar);
            assert_eq!(n, 960, "bands {start}..{end}");
            assert!(pcm.iter().all(|v| v.is_finite()));
        }
    }
}

/// A frame length that no frame can have is a bad argument (libopus
/// `celt_decode_with_ec`: `len < 0 || len > 1275`). opus1 accepts what
/// its range coder can hold, 2048 bytes.
#[test]
fn impossible_frame_length_is_rejected() {
    let frame = [0x5au8; 60];
    let mut dec = CeltDecoder::new(default_mode(), 1, 48000);
    let mut pcm = vec![0f32; 960];
    for bits in [
        -1i32,
        i32::MIN,
        2048 * 8 + 1,
        1 << 20,
        i32::MAX / 8,
        i32::MAX,
    ] {
        let mut rc = RangeCoder::new_decoder(&frame);
        assert_eq!(
            dec.decode(&mut rc, bits, 960, &mut pcm, PcmLayout::Planar),
            0,
            "{bits} bits"
        );
    }
    // More bits than the buffer holds: the range decoder reads zeros.
    for bits in [0i32, 8, 9, 480, 481, 1275 * 8, 2048 * 8] {
        let mut rc = RangeCoder::new_decoder(&frame);
        assert_eq!(
            dec.decode(&mut rc, bits, 960, &mut pcm, PcmLayout::Planar),
            960,
            "{bits} bits"
        );
        assert!(pcm.iter().all(|v| v.is_finite()));
    }
    assert_eq!(decode_all(&mut dec, &[0u8; 2049], 960, &mut pcm, PcmLayout::Planar), 0);
    assert_eq!(decode_all(&mut dec, &[0u8; 2048], 960, &mut pcm, PcmLayout::Planar), 960);
}

/// Every public decode function and band setter with arguments drawn at
/// random from valid, borderline and absurd values, in any order, on one
/// decoder, for every output layout. A call returns the frame size or
/// reports the bad argument, and never panics.
/// Run this test without `--release` as well: it then also catches
/// arithmetic overflow.
#[test]
fn extreme_arguments_do_not_panic() {
    let mut rng = Rng(0xBEEF);
    let frames = [
        0usize,
        1,
        60,
        119,
        120,
        121,
        160,
        240,
        480,
        960,
        961,
        1920,
        2048,
        2049,
        5760,
        usize::MAX / 2,
        usize::MAX,
    ];
    let starts = [0usize, 1, 16, 17, 18, 20, 21, 22, 100, usize::MAX];
    let ends = [0usize, 1, 13, 17, 18, 19, 20, 21, 22, 1000, usize::MAX];
    let lengths = [0usize, 1, 2, 3, 8, 40, 200, 1275, 1276, 3000];
    let out_lengths = [0usize, 1, 119, 120, 240, 960, 1919, 1920];
    let (mut calls, mut decoded) = (0u32, 0u32);
    for channels in [1usize, 2] {
        for rate in [48000i32, 24000, 16000, 12000, 8000] {
            let mut dec = CeltDecoder::new(default_mode(), channels, rate);
            let mut pcm = vec![0f32; 960 * 2];
            for round in 0..10000 {
                let frame = frames[rng.below(frames.len())];
                let start = starts[rng.below(starts.len())];
                let end = ends[rng.below(ends.len())];
                let len = lengths[rng.below(lengths.len())];
                let bytes: Vec<u8> = (0..len)
                    .map(|_| match round % 3 {
                        0 => 0xff,
                        1 => 0,
                        _ => rng.next() as u8,
                    })
                    .collect();
                let bit_counts = [
                    i32::MIN,
                    -1,
                    0,
                    1,
                    7,
                    8,
                    9,
                    15,
                    16,
                    (len * 8) as i32,
                    (len * 8) as i32 + 1000,
                    1275 * 8,
                    1276 * 8,
                    100000,
                    i32::MAX / 8,
                    i32::MAX,
                ];
                let bits = bit_counts[rng.below(bit_counts.len())];
                let out = &mut pcm[..out_lengths[rng.below(out_lengths.len())]];
                // Stream channel counts 0 and 3 are errors and change nothing.
                let _ = dec.set_stream_channels(rng.below(4));
                // A band out of range is an error and changes nothing.
                let n = match rng.below(6) {
                    0 => {
                        let _ = dec.set_start_band(start);
                        dec.decode_lost(frame, out, PcmLayout::Planar).unwrap_or(0)
                    }
                    1 => {
                        dec.set_start_band(0).unwrap();
                        dec.set_end_band(21).unwrap();
                        decode_all(&mut dec, &bytes, frame, out, PcmLayout::Planar)
                    }
                    2 => {
                        let _ = dec.set_start_band(start);
                        dec.set_end_band(21).unwrap();
                        decode_all(&mut dec, &bytes, frame, out, PcmLayout::Interleaved)
                    }
                    3 => {
                        let _ = dec.set_start_band(start);
                        dec.set_end_band(21).unwrap();
                        let mut rc = RangeCoder::new_decoder(&bytes);
                        dec.decode(&mut rc, bits, frame, out, PcmLayout::InterleavedAdd)
                    }
                    4 => {
                        dec.reset_state();
                        0
                    }
                    _ => {
                        let _ = dec.set_start_band(start);
                        let _ = dec.set_end_band(end);
                        let mut rc = RangeCoder::new_decoder(&bytes);
                        dec.decode(&mut rc, bits, frame, out, PcmLayout::Planar)
                    }
                };
                assert!(n == 0 || n == frame);
                if n != 0 {
                    decoded += 1;
                    assert!(out.len() >= frame * channels);
                    assert!(
                        out[..frame * channels].iter().all(|v| v.is_finite()),
                        "output is not finite"
                    );
                }
                calls += 1;
                let _ = dec.final_range();
            }
        }
    }
    println!("{calls} calls, {decoded} of them decoded or concealed a frame");
    assert!(decoded > 500, "too few valid calls: {decoded}");
}

/// End of the CELT band range for the bandwidth in a table-of-contents
/// byte (libopus `opus_decode_frame`).
fn end_band(toc_byte: u8) -> usize {
    match toc(toc_byte).bandwidth {
        "NB" => 13,
        "WB" => 17,
        "SWB" => 19,
        _ => 21,
    }
}

/// Damage a packet's payload. The first byte and at least two payload bytes
/// are kept: a shorter payload is a lost frame, which other tests cover.
fn damage(packet: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut p = packet.to_vec();
    match rng.below(5) {
        // Flip a few bits.
        0 => {
            for _ in 0..1 + rng.below(8) {
                let at = 1 + rng.below(p.len() - 1);
                p[at] ^= 1 << rng.below(8);
            }
        }
        // Cut the end off.
        1 => p.truncate(3 + rng.below(p.len() - 2)),
        // Replace everything with random bytes.
        2 => {
            for b in p[1..].iter_mut() {
                *b = rng.next() as u8;
            }
        }
        // All ones or all zeros.
        3 => {
            let fill = if rng.below(2) == 0 { 0xff } else { 0 };
            p[1..].fill(fill);
        }
        // Append random bytes.
        _ => {
            for _ in 0..1 + rng.below(40) {
                p.push(rng.next() as u8);
            }
        }
    }
    p
}

/// The stream settings of the damaged-frame tests: sampling rate, CELT
/// bandwidth, frame size in samples and bit rate per channel.
const DAMAGE_STREAMS: [(usize, i32, usize, i32); 6] = [
    (48000, BW_FB, 960, 48000),
    (48000, BW_FB, 960, 128000),
    (48000, BW_FB, 120, 48000),
    (48000, BW_SWB, 480, 48000),
    (16000, BW_WB, 320, 48000),
    (8000, BW_NB, 80, 48000),
];

/// What one stream of damaged frames gave.
struct DamageResult {
    /// Frames with damage that both decoders decoded.
    damaged: usize,
    /// SNR against libopus of every decoded (not concealed) frame, sorted.
    decoded_db: Vec<f64>,
    /// Frames that libopus decoded to more than 10 times full scale.
    loud: usize,
    /// Index of the first packet after which the final range differed.
    range_mismatch: Option<usize>,
}

/// Decode a CELT-only stream with `CeltDecoder` and with libopus. Every
/// `damage_every`-th packet is damaged and, if `lose` is set, one packet in
/// eleven is lost. The comparison stops at the first packet after which the
/// two decoders report different final ranges: their states differ from
/// there on.
fn damaged_stream(
    packets: &[Vec<u8>],
    rate: usize,
    stream_channels: usize,
    dec_channels: usize,
    frame: usize,
    damage_every: usize,
    lose: bool,
    rng: &mut Rng,
) -> DamageResult {
    let mut c_dec = CDecoder::new(rate, dec_channels);
    let mut dec = CeltDecoder::new(default_mode(), dec_channels, rate as i32);
    let mut planar = vec![0f32; frame * dec_channels];
    let mut result = DamageResult {
        damaged: 0,
        decoded_db: Vec::new(),
        loud: 0,
        range_mismatch: None,
    };
    for (i, packet) in packets.iter().enumerate() {
        let lost = lose && i % 11 == 7;
        let damaged = !lost && i % damage_every == 1;
        let p = if damaged {
            damage(packet, rng)
        } else {
            packet.clone()
        };
        let c = if lost {
            assert_eq!(dec.decode_lost(frame, &mut planar, PcmLayout::Planar), Ok(frame));
            c_dec.decode(None, frame)
        } else {
            dec.set_stream_channels(stream_channels).unwrap();
            dec.set_end_band(end_band(p[0])).unwrap();
            let n = decode_all(&mut dec, &p[1..], frame, &mut planar, PcmLayout::Planar);
            assert_eq!(n, frame, "packet {i}");
            c_dec.decode(Some(&p), frame)
        };
        assert_eq!(c.len(), frame * dec_channels, "packet {i}");
        let r = interleave(&planar, dec_channels);
        assert!(
            r.iter().all(|v| v.is_finite()),
            "packet {i}: output is not finite"
        );
        if lost {
            continue;
        }
        if dec.final_range() != c_dec.final_range() {
            result.range_mismatch = Some(i);
            break;
        }
        result.damaged += damaged as usize;
        if c.iter().any(|v| v.abs() > 10.0) {
            result.loud += 1;
        }
        if power(&c) > 1e-10 {
            result.decoded_db.push(snr(&c, &r));
        }
    }
    result.decoded_db.sort_by(|a, b| a.total_cmp(b));
    result
}

fn damage_test_packets(
    rate: usize,
    channels: usize,
    bandwidth: i32,
    frame: usize,
    bitrate: i32,
    seconds: f64,
) -> Vec<Vec<u8>> {
    let mono = speech_fullband(rate, seconds);
    let pcm = if channels == 2 {
        to_stereo(&mono, rate)
    } else {
        mono
    };
    let mut enc = CEnc::audio(rate, channels);
    enc.force_mode(MODE_CELT)
        .bandwidth(bandwidth)
        .bitrate(bitrate * channels as i32);
    enc.encode_all(&pcm, frame)
}

/// Damaged CELT frames through `CeltDecoder`, against `opus_decode_float`
/// of the same packet. Every third packet is damaged and one in eleven is
/// lost, in streams and decoders with one and two channels.
///
/// A damaged frame can decode to samples far above full scale. libopus's
/// float decoder passes them on without a limit, and so must `CeltDecoder`:
/// a limit would leave another history behind than libopus has, and the
/// frames after it would differ. The concealment's pitch search must also
/// cope with such a history the way libopus does.
///
/// The two decoders must report the same final range after every frame.
/// Until `RangeCoder::dec_uint` got the limit that libopus has (see
/// `damaged_frames_read_the_same_symbols_as_libopus`), about one damaged
/// frame in 500 ended with another final range, and about three of the 24
/// streams had to be left at that point.
#[test]
fn damaged_celt_frames_decode_like_libopus() {
    let mut rng = Rng(0x00C0_FFEE);
    let mut failures = Vec::new();
    let (mut damaged, mut loud, mut left_early) = (0, 0, 0);
    let (mut poor_frames, mut decoded_frames) = (0, 0);
    for (stream_channels, dec_channels) in [(1usize, 1usize), (2, 2), (1, 2), (2, 1)] {
        for (rate, bandwidth, frame, bitrate) in DAMAGE_STREAMS {
            let packets =
                damage_test_packets(rate, stream_channels, bandwidth, frame, bitrate, 2.0);
            let r = damaged_stream(
                &packets,
                rate,
                stream_channels,
                dec_channels,
                frame,
                3,
                true,
                &mut rng,
            );
            let name = format!(
                "{stream_channels}-channel stream, {dec_channels}-channel decoder, {rate} Hz, {frame} samples, {bitrate} b/s"
            );
            let n = r.decoded_db.len();
            let poor = r.decoded_db.iter().filter(|d| **d < 40.0).count();
            println!(
                "{name}: {} damaged frames, {} louder than 10 times full scale; decoded frames: median {:.1} dB, lowest tenth {:.1} dB, worst {:.1} dB, {poor} below 40 dB{}",
                r.damaged,
                r.loud,
                r.decoded_db[n / 2],
                r.decoded_db[n / 10],
                r.decoded_db[0],
                match r.range_mismatch {
                    Some(i) => format!("; final range differs after packet {i}"),
                    None => String::new(),
                }
            );
            damaged += r.damaged;
            loud += r.loud;
            left_early += r.range_mismatch.is_some() as usize;
            poor_frames += poor;
            decoded_frames += n;
            // Frames next to a concealed frame match to 60 to 100 dB, the
            // others to 135 dB. A concealed frame whose predictor order or
            // pitch period was decided by the last bit (see
            // tests/celtdec_plc.rs) leaves the frames of the next 20 ms
            // badly matched. One such event per stream is allowed.
            let poor_limit = (rate / 50 / frame).max(1);
            if r.decoded_db[n / 2] < 125.0 || r.decoded_db[n / 10] < 60.0 || poor > poor_limit {
                failures.push(name);
            }
        }
    }
    println!(
        "{damaged} damaged frames compared, {loud} frames louder than 10 times full scale, {left_early} streams left early"
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(
        poor_frames * 200 <= decoded_frames,
        "{poor_frames} of {decoded_frames} decoded frames below 40 dB"
    );
    assert!(damaged > 1000, "only {damaged} damaged frames compared");
    assert!(
        loud >= 10,
        "only {loud} loud frames: the test no longer covers them"
    );
    assert_eq!(left_early, 0, "streams with a final range mismatch");
}

/// Damaged CELT-only and Hybrid packets and lost packets through
/// `OpusDecoder`, against `opus_decode_float`. Both decoders must accept
/// and reject the same packets, and the output must stay finite and close
/// to libopus's.
///
/// Both outputs are limited to [-1, 1] before they are compared, as in
/// `decode_with_loss`.
#[test]
fn damaged_packets_and_losses_in_the_opus_decoder() {
    let mut rng = Rng(77);
    let mut failures = Vec::new();
    for (mode, bandwidth, name) in [
        (MODE_CELT, BW_FB, "CELT"),
        (MODE_HYBRID, BW_FB, "Hybrid FB"),
        (MODE_HYBRID, BW_SWB, "Hybrid SWB"),
    ] {
        for channels in [1usize, 2] {
            let mono = speech_fullband(48000, 2.0);
            let pcm = if channels == 2 {
                to_stereo(&mono, 48000)
            } else {
                mono
            };
            for (rate, frame_ms) in [(48000usize, 20usize), (48000, 10), (24000, 20), (8000, 10)] {
                let frame = rate * frame_ms / 1000;
                let mut enc = CEnc::audio(48000, channels);
                enc.force_mode(mode)
                    .bandwidth(bandwidth)
                    .bitrate(40000 * channels as i32);
                let packets = enc.encode_all(&pcm, 48 * frame_ms);
                let mut err = 0;
                let st =
                    unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
                assert_eq!(err, 0);
                let mut rs = opus1::OpusDecoder::new(rate as i32, channels).unwrap();
                let mut c_out = vec![0f32; frame * channels];
                let mut r_out = vec![0f32; frame * channels];
                let (mut accepted, mut rejected) = (0, 0);
                let mut decoded_db = Vec::new();
                let case = format!("{name}, {channels} ch, decoder at {rate} Hz, {frame_ms} ms");
                for (i, packet) in packets.iter().enumerate() {
                    let lost = i % 11 == 7;
                    let damaged = !lost && i % 3 == 1;
                    let p = if damaged {
                        damage(packet, &mut rng)
                    } else {
                        packet.clone()
                    };
                    let (ptr, len) = if lost {
                        (std::ptr::null(), 0)
                    } else {
                        (p.as_ptr(), p.len() as i32)
                    };
                    let c_ret = unsafe {
                        sys::opus_decode_float(st, ptr, len, c_out.as_mut_ptr(), frame as i32, 0)
                    };
                    let r_ret = rs.decode(if lost { &[] } else { &p }, frame, &mut r_out);
                    assert_eq!(
                        c_ret >= 0,
                        r_ret.is_ok(),
                        "{case}: packet {i} of {} bytes: libopus returns {c_ret}, opus1 {r_ret:?}",
                        p.len()
                    );
                    if c_ret < 0 {
                        rejected += 1;
                        continue;
                    }
                    accepted += 1;
                    assert_eq!(r_ret, Ok(c_ret as usize), "{case}: packet {i}");
                    assert!(
                        r_out.iter().all(|v| v.is_finite()),
                        "{case}: packet {i}: output is not finite"
                    );
                    let c: Vec<f32> = c_out.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
                    if !lost && power(&c) > 1e-10 {
                        decoded_db.push(snr(&c, &r_out));
                    }
                }
                unsafe { sys::opus_decoder_destroy(st) };
                decoded_db.sort_by(|a, b| a.total_cmp(b));
                let n = decoded_db.len();
                let poor = decoded_db.iter().filter(|d| **d < 40.0).count();
                println!(
                    "{case}: {accepted} accepted, {rejected} rejected; decoded frames: median {:.1} dB, lowest tenth {:.1} dB, worst {:.1} dB, {poor} below 40 dB",
                    decoded_db[n / 2],
                    decoded_db[n / 10],
                    decoded_db[0]
                );
                // A few frames may be far off: those next to a badly
                // matched concealed frame.
                if decoded_db[n / 2] < 125.0 || decoded_db[n / 10] < 60.0 || poor * 20 > n {
                    failures.push(case);
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A damaged CELT-only packet (20 ms, stereo, eight flipped bits) after
/// which libopus 1.6.1 reports the final range 0x159e9700.
const PACKET_WITH_OUT_OF_RANGE_INTEGER: &[&str] = &[
    "fcd913102fd6a7ae1f74c9fcb8fd941fcd9666463058b886562b62bb30542cd8",
    "6cd9813dc3c368bcc3947a5bc62420719df0fd6737ea95eb589d41ed54210bbd",
    "8c8143eb8cd1ff53aead3aa631caedc5e0f30cab7687849bd482e3c17220a5ba",
    "b8772d0bf27d1be5c0954a787340bdb406f37973710d64fdec2c68483a8e4ff9",
    "7a2df998ce391044cea3db2afa5a71491c83c6eb0ae5c78693fe48f79315972a",
    "3fc19a778a73848c97259065db0c8467f2a9648e404328ae4ea193ae22e4c3a7",
    "543834f7589f86138b",
];

fn hex_bytes(lines: &[&str]) -> Vec<u8> {
    let hex: String = lines.concat();
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// libopus `ec_dec_uint` (celt/entdec.c) decodes an integer below `ft` in
/// two parts when `ft` is above 256: the high bits with the range decoder,
/// the low bits as raw bits. A damaged frame can give a combination that is
/// `ft` or more. libopus then returns `ft - 1` and marks the frame as
/// faulty:
///
/// ```c
/// if(t<=_ft)return t;
/// _this->error=1;
/// return _ft;
/// ```
///
/// `RangeCoder::dec_uint` (src/range_coder.rs) used to return the
/// combination as it was. The stereo angle or the pulse index decoded from
/// it was then out of range, and the decoder read other symbols than
/// libopus for the rest of the frame. The frame and the five or so frames
/// after it came out different (down to 0 dB).
///
/// Measured over 152000 frames, half of them damaged: 163 frames with
/// another final range than libopus. With the three lines above in
/// `dec_uint`: none.
#[test]
fn damaged_frames_read_the_same_symbols_as_libopus() {
    // The smallest case: 257 values, the top value of the high part and a
    // raw bit of 1 give 257.
    let mut rc = RangeCoder::new_decoder(&[0xff; 4]);
    assert_eq!(rc.dec_uint(257), 256, "dec_uint(257) on a frame of ones");

    // A whole frame.
    let packet = hex_bytes(PACKET_WITH_OUT_OF_RANGE_INTEGER);
    assert_eq!(toc(packet[0]).to_string(), toc(0xfc).to_string());
    let mut c_dec = CDecoder::new(48000, 2);
    let mut dec = CeltDecoder::new(default_mode(), 2, 48000);
    let mut planar = vec![0f32; 2 * 960];
    let c = c_dec.decode(Some(&packet), 960);
    let n = decode_all(&mut dec, &packet[1..], 960, &mut planar, PcmLayout::Planar);
    assert_eq!(n, 960);
    assert_eq!(c_dec.final_range(), 0x159e_9700, "libopus");
    assert_eq!(dec.final_range(), 0x159e_9700, "opus1");
    let db = snr(&c, &interleave(&planar, 2));
    assert!(db > 100.0, "{db:.1} dB");

    // Many damaged frames: every final range must be equal.
    let mut total = 0;
    for seed in 1..=4u32 {
        let mut rng = Rng(seed * 7919);
        for (stream_channels, dec_channels) in [(1usize, 1usize), (2, 2), (1, 2), (2, 1)] {
            for (rate, bandwidth, frame, bitrate) in DAMAGE_STREAMS {
                let packets =
                    damage_test_packets(rate, stream_channels, bandwidth, frame, bitrate, 1.0);
                let r = damaged_stream(
                    &packets,
                    rate,
                    stream_channels,
                    dec_channels,
                    frame,
                    2,
                    false,
                    &mut rng,
                );
                assert_eq!(
                    r.range_mismatch, None,
                    "seed {seed}, {stream_channels}-channel stream, {dec_channels}-channel decoder, {rate} Hz, {frame} samples"
                );
                total += r.damaged;
            }
        }
    }
    println!("{total} damaged frames, all with libopus's final range");
    assert!(total > 5000);
}

/// Frames of random bytes drive the band energies up until the output is
/// thousands of times above full scale. libopus's float decoder has no
/// limit on its output (`SATURATE` does nothing in the float build), so
/// `CeltDecoder` must not have one either: the samples and the history
/// behind them have to stay equal.
#[test]
fn output_far_above_full_scale_is_not_limited() {
    // libopus limit of the fixed-point build, in units of full scale.
    const SIG_SAT: f32 = 536870911.0 / 32768.0;
    let frame = 120;
    for (channels, seed) in [(1usize, 5u32), (1, 25), (2, 14), (2, 24)] {
        let mut rng = Rng(seed);
        let mut c_dec = CDecoder::new(48000, channels);
        let mut dec = CeltDecoder::new(default_mode(), channels, 48000);
        let mut planar = vec![0f32; frame * channels];
        // CELT-only, fullband, 2.5 ms, one frame per packet.
        let toc_byte = 0xe0 | if channels == 2 { 0x04 } else { 0 };
        let (mut above, mut peak, mut worst, mut frames) = (0, 0f32, 200f64, 0);
        for i in 0..600 {
            let len = 20 + rng.below(60);
            let mut packet = vec![toc_byte];
            packet.extend((0..len).map(|_| rng.next() as u8));
            let c = c_dec.decode(Some(&packet), frame);
            let n = decode_all(&mut dec, &packet[1..], frame, &mut planar, PcmLayout::Planar);
            assert_eq!(n, frame);
            assert_eq!(
                dec.final_range(),
                c_dec.final_range(),
                "{channels} ch: final range after frame {i}"
            );
            let r = interleave(&planar, channels);
            assert!(r.iter().all(|v| v.is_finite()));
            let c_peak = c.iter().fold(0f32, |m, v| m.max(v.abs()));
            peak = peak.max(c_peak);
            above += (c_peak > SIG_SAT) as usize;
            frames += 1;
            if power(&c) > 1e-10 {
                worst = worst.min(snr(&c, &r));
            }
        }
        println!(
            "{channels} ch: {frames} frames, {above} of them above the fixed-point limit, loudest {peak:.0} times full scale, worst frame {worst:.1} dB"
        );
        assert!(
            above >= 10,
            "{channels} ch: only {above} frames above the limit, the test is void"
        );
        assert!(worst > 90.0, "{channels} ch: worst frame {worst:.1} dB");
    }
}
