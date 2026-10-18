//! The low-frequency-effects flag and the energy mask of `OpusEncoder`
//! against C libopus with `OPUS_SET_LFE` and `OPUS_SET_ENERGY_MASK`.
//!
//! A multistream encoder sets both on its stream encoders: the LFE flag on
//! the stream that carries the LFE channel of a surround layout, and for
//! every frame a mask that says how far the other channels cover each band
//! of a stream.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech_fullband, to_stereo, toc};
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusEncoder};

/// What an LFE channel carries: tones at 38 and 71 Hz that swell twice a
/// second, and a little noise.
fn rumble(rate: usize, seconds: f64) -> Vec<f32> {
    let n = (rate as f64 * seconds) as usize;
    let mut seed = 5u32;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let swell = 0.55 + 0.45 * (2.0 * std::f64::consts::PI * 2.0 * t).sin();
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((seed >> 8) as f64 / (1u32 << 23) as f64 - 1.0) * 1e-4;
            let low = 0.4 * (2.0 * std::f64::consts::PI * 38.0 * t).sin();
            let high = 0.15 * (2.0 * std::f64::consts::PI * 71.0 * t).sin();
            (swell * (low + high) + noise) as f32
        })
        .collect()
}

fn speech(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let mono = speech_fullband(rate, seconds);
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

fn encode_all(enc: &mut OpusEncoder, pcm: &[f32], frame: usize) -> Vec<Vec<u8>> {
    let mut buf = vec![0u8; 4000];
    pcm.chunks_exact(frame * enc.channels())
        .enumerate()
        .map(|(i, f)| {
            let n = enc.encode(f, frame, &mut buf).unwrap_or_else(|e| panic!("packet {i}: {e}"));
            buf[..n].to_vec()
        })
        .collect()
}

/// Decode with libopus. Every packet must decode to `frame` samples.
fn c_decode_all(packets: &[Vec<u8>], rate: usize, channels: usize, frame: usize) -> Vec<f32> {
    let mut dec = CDecoder::new(rate, channels);
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let frame_out = dec.decode(Some(p), frame, false).unwrap_or_else(|e| panic!("packet {i}: libopus error {e}"));
        assert_eq!(frame_out.len(), frame * channels, "packet {i}: duration");
        out.extend(frame_out);
    }
    out
}

/// SNR in dB of the libopus decode of `packets` against the encoder's
/// input, without the first 200 ms. The decoded signal is late by the
/// encoder's look-ahead: 6.5 ms for a mono `Audio` encoder in either
/// library (2.5 ms of CELT overlap and 4 ms of delay compensation).
fn c_decoded_snr(packets: &[Vec<u8>], pcm: &[f32], rate: usize, frame: usize) -> f64 {
    let out = c_decode_all(packets, rate, 1, frame);
    let delay = rate * 65 / 10_000;
    let skip = rate / 5;
    snr(&pcm[skip..out.len() - delay], &out[skip + delay..])
}

fn total_bytes(packets: &[Vec<u8>]) -> usize {
    packets.iter().map(|p| p.len()).sum()
}

/// Whether a packet codes nothing: every frame in it is empty.
fn is_placeholder(packet: &[u8]) -> bool {
    let parsed = opus1::packet::parse(packet, false).expect("a valid packet");
    parsed.sizes[..parsed.count].iter().all(|&size| size == 0)
}

// ---------------------------------------------------------------------------
// Low-frequency effects
// ---------------------------------------------------------------------------

/// An LFE stream at 2 to 8 kb/s, in every frame size, with a variable and a
/// constant bitrate, from 48 and 16 kHz input.
///
/// - Every packet has libopus's first byte. A packet that carries data is
///   CELT-only and narrowband, whatever the input rate (opus_encoder.c
///   1538-1539, 1683-1684).
/// - A packet too small to carry CELT data is a placeholder, byte for byte
///   libopus's (opus_encoder.c:1340-1406). In frames of 2.5 ms that is every
///   packet below 9.6 kb/s, in 5 ms below 4.8 kb/s, in 10 ms below 2.4 kb/s.
///   The first byte of such a packet names the mode and bandwidth of the
///   last packet that was coded: Hybrid and fullband in a new encoder, and
///   CELT-only for frames shorter than 10 ms.
/// - Constant bitrate: every packet has libopus's length.
/// - Variable bitrate, frames up to 20 ms: every packet is within 2 bytes
///   of libopus's, and the stream within 1 %.
/// - Decoded by libopus, the stream is within 3 dB of libopus's own.
///
/// One known difference is left out: packets of 40 and 60 ms with a
/// variable bitrate. libopus 1.6 limits each 20 ms part of such a packet to
/// its share of the bitrate. opus1 does not, on purpose (see
/// `OpusEncoder::encode_multiframe`), so its packets are larger there.
#[test]
fn lfe_streams_have_libopus_first_bytes_sizes_and_quality() {
    let mut failures = Vec::new();
    let (mut coded_streams, mut placeholder_streams) = (0, 0);
    for rate in [48000usize, 16000] {
        let pcm = rumble(rate, 3.0);
        for frame_tenth_ms in [25usize, 50, 100, 200, 400, 600] {
            let frame = rate * frame_tenth_ms / 10_000;
            let frame_rate = 10_000 / frame_tenth_ms as i32;
            for bitrate in [2000, 3000, 4000, 6000, 8000] {
                for cbr in [false, true] {
                    let what = format!("{rate} Hz, {} ms, {bitrate} b/s, {}", frame_tenth_ms as f64 / 10.0, if cbr { "CBR" } else { "VBR" });
                    let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
                    c_enc.lfe(true).bitrate(bitrate).vbr(!cbr);
                    let c_packets = c_enc.encode_all(&pcm, frame);
                    let mut enc = OpusEncoder::new(rate as i32, 1, Application::Audio).unwrap();
                    enc.set_lfe(true);
                    enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                    enc.set_vbr(!cbr);
                    let packets = encode_all(&mut enc, &pcm, frame);

                    let long_vbr = frame_tenth_ms > 200 && !cbr;
                    let (mut first_bytes, mut sizes, mut placeholders) = (0, 0, 0);
                    for (c, r) in c_packets.iter().zip(&packets) {
                        if is_placeholder(c) {
                            placeholders += 1;
                            if c != r {
                                failures.push(format!("{what}: placeholder {c:02x?} in libopus, {r:02x?} in opus1"));
                                break;
                            }
                            continue;
                        }
                        let t = toc(r[0]);
                        if (t.mode, t.bandwidth) != ("CELT", "NB") {
                            failures.push(format!("{what}: a packet with data is {t}"));
                            break;
                        }
                        // The two lowest bits say how the 20 ms parts of a
                        // longer packet are laid out, which follows from
                        // their sizes.
                        let mask = if long_vbr { 0xFC } else { 0xFF };
                        first_bytes += (c[0] & mask != r[0] & mask) as usize;
                        let diff = (c.len() as i64 - r.len() as i64).abs();
                        sizes += (diff > if cbr { 0 } else { 2 }) as usize;
                    }
                    if first_bytes != 0 {
                        failures.push(format!("{what}: {first_bytes} first bytes differ from libopus"));
                    }
                    let (c_total, rs_total) = (total_bytes(&c_packets) as f64, total_bytes(&packets) as f64);
                    if !long_vbr && (sizes != 0 || (rs_total - c_total).abs() > 0.01 * c_total) {
                        failures.push(format!("{what}: {sizes} packet sizes differ from libopus; {rs_total} bytes against {c_total}"));
                    }

                    // opus_encoder.c:1340: `st->bitrate_bps < 3*frame_rate*8`,
                    // with the bitrate of a constant-bitrate packet taken
                    // from its whole bytes.
                    let coded_bitrate = if cbr { ((bitrate / frame_rate + 4) / 8) * 8 * frame_rate } else { bitrate };
                    let expect_placeholders = coded_bitrate < 3 * frame_rate * 8 || (frame_rate < 50 && coded_bitrate < 2400);
                    if expect_placeholders {
                        placeholder_streams += 1;
                        if placeholders != packets.len() {
                            failures.push(format!("{what}: {placeholders} of {} packets are placeholders", packets.len()));
                        }
                        continue;
                    }
                    coded_streams += 1;
                    if placeholders != 0 {
                        failures.push(format!("{what}: {placeholders} placeholders in a stream that can carry data"));
                    }
                    let (c_snr, rs_snr) = (c_decoded_snr(&c_packets, &pcm, rate, frame), c_decoded_snr(&packets, &pcm, rate, frame));
                    if rs_snr < c_snr - 3.0 {
                        failures.push(format!("{what}: {rs_snr:.1} dB, libopus {c_snr:.1} dB"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
    // Both kinds of stream were there.
    assert!(coded_streams >= 70 && placeholder_streams >= 30, "{coded_streams} streams with data, {placeholder_streams} of placeholders");
}

/// At bitrates where an LFE channel gets enough bits, the low tones come
/// back, as well as from libopus.
#[test]
fn lfe_tones_come_back_like_from_libopus() {
    for rate in [48000usize, 24000] {
        let pcm = rumble(rate, 3.0);
        let frame = rate / 50;
        for bitrate in [6000, 8000] {
            let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
            c_enc.lfe(true).bitrate(bitrate);
            let c_snr = c_decoded_snr(&c_enc.encode_all(&pcm, frame), &pcm, rate, frame);
            let mut enc = OpusEncoder::new(rate as i32, 1, Application::Audio).unwrap();
            enc.set_lfe(true);
            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            let packets = encode_all(&mut enc, &pcm, frame);
            let rs_snr = c_decoded_snr(&packets, &pcm, rate, frame);
            // Without the flag the same bitrate gives far less: the bits go
            // to all bands.
            let mut plain = OpusEncoder::new(rate as i32, 1, Application::Audio).unwrap();
            plain.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            plain.set_force_mode(Some(ForcedMode::CeltOnly));
            plain.set_bandwidth(Bandwidth::Narrowband);
            let plain_snr = c_decoded_snr(&encode_all(&mut plain, &pcm, frame), &pcm, rate, frame);
            println!("{rate} Hz, {bitrate} b/s: opus1 {rs_snr:.1} dB, libopus {c_snr:.1} dB; opus1 without the LFE flag {plain_snr:.1} dB");
            assert!(rs_snr > 12.0 && rs_snr > c_snr - 2.0, "{rate} Hz, {bitrate} b/s: opus1 {rs_snr:.1} dB, libopus {c_snr:.1} dB");
            assert!(rs_snr > plain_snr + 3.0, "{rate} Hz, {bitrate} b/s: {rs_snr:.1} dB with the LFE flag, {plain_snr:.1} dB without");
        }
    }
}

// ---------------------------------------------------------------------------
// Energy mask
// ---------------------------------------------------------------------------

/// Masks of 21 values per channel, for two channels. The last two have
/// values beyond the range libopus limits a mask to for SILK's bitrate, -2
/// to 0.5 (opus_encoder.c:2091).
fn masks() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("-2", vec![-2.0; 42]),
        ("-1", vec![-1.0; 42]),
        ("-0.5", vec![-0.5; 42]),
        ("0", vec![0.0; 42]),
        ("+0.5", vec![0.5; 42]),
        ("falling to -2", (0..42).map(|i| -0.1 * (i % 21) as f32).collect()),
        ("-2 below 2 kHz", (0..42).map(|i| if i % 21 < 8 { -2.0 } else { 0.0 }).collect()),
        ("-10 and +0.5 in turn", (0..42).map(|i| if i % 2 == 0 { -10.0 } else { 0.5 }).collect()),
        ("+3", vec![3.0; 42]),
    ]
}

/// One forced stream for the mask tests.
#[derive(Clone, Copy)]
struct Stream {
    name: &'static str,
    rate: usize,
    channels: usize,
    mode: (ForcedMode, i32),
    bandwidth: (Bandwidth, i32),
    bitrate: i32,
    frame_ms: usize,
}

impl Stream {
    /// The packets of libopus and of opus1.
    fn packets(&self, pcm: &[f32], cbr: bool, mask: Option<&[f32]>) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let frame = self.rate * self.frame_ms / 1000;
        let mut c_enc = CEncoder::new(self.rate, self.channels, APP_AUDIO);
        c_enc.force_mode(self.mode.1).bandwidth(self.bandwidth.1).bitrate(self.bitrate).vbr(!cbr).energy_mask(mask);
        let mut enc = OpusEncoder::new(self.rate as i32, self.channels, Application::Audio).unwrap();
        enc.set_force_mode(Some(self.mode.0));
        enc.set_bandwidth(self.bandwidth.0);
        enc.set_bitrate(Bitrate::BitsPerSecond(self.bitrate)).unwrap();
        enc.set_vbr(!cbr);
        enc.set_energy_mask(mask);
        (c_enc.encode_all(pcm, frame), encode_all(&mut enc, pcm, frame))
    }

    /// Bytes of the whole stream from libopus and from opus1.
    fn bytes(&self, pcm: &[f32], cbr: bool, mask: Option<&[f32]>) -> (usize, usize) {
        let (c_packets, packets) = self.packets(pcm, cbr, mask);
        (total_bytes(&c_packets), total_bytes(&packets))
    }
}

const SILK: (ForcedMode, i32) = (ForcedMode::SilkOnly, MODE_SILK);
const HYBRID: (ForcedMode, i32) = (ForcedMode::Hybrid, MODE_HYBRID);
const CELT: (ForcedMode, i32) = (ForcedMode::CeltOnly, MODE_CELT);
const NB: (Bandwidth, i32) = (Bandwidth::Narrowband, BW_NB);
const MB: (Bandwidth, i32) = (Bandwidth::Mediumband, BW_MB);
const WB: (Bandwidth, i32) = (Bandwidth::Wideband, BW_WB);
const SWB: (Bandwidth, i32) = (Bandwidth::Superwideband, BW_SWB);
const FB: (Bandwidth, i32) = (Bandwidth::Fullband, BW_FB);

/// SILK-only streams with a variable bitrate: a mask moves the SILK bitrate
/// (opus_encoder.c:2068-2108). The size of the stream with a mask, as a
/// share of its size without, is libopus's to within 0.04. The shares
/// themselves go from a third (masks of -1 and below) to 1.5 (a stereo mask
/// of +0.5).
#[test]
fn energy_mask_moves_the_silk_bitrate_like_libopus() {
    let mut failures = Vec::new();
    for stream in [
        Stream { name: "SILK-WB mono 24 kb/s from 48 kHz", rate: 48000, channels: 1, mode: SILK, bandwidth: WB, bitrate: 24000, frame_ms: 20 },
        Stream { name: "SILK-NB mono 12 kb/s", rate: 16000, channels: 1, mode: SILK, bandwidth: NB, bitrate: 12000, frame_ms: 20 },
        Stream { name: "SILK-MB mono 16 kb/s", rate: 16000, channels: 1, mode: SILK, bandwidth: MB, bitrate: 16000, frame_ms: 20 },
        Stream { name: "SILK-WB stereo 40 kb/s", rate: 48000, channels: 2, mode: SILK, bandwidth: WB, bitrate: 40000, frame_ms: 20 },
        Stream { name: "SILK-WB mono 20 kb/s, 10 ms", rate: 16000, channels: 1, mode: SILK, bandwidth: WB, bitrate: 20000, frame_ms: 10 },
    ] {
        let pcm = speech(stream.rate, stream.channels, 2.0);
        let (c_plain, rs_plain) = stream.bytes(&pcm, false, None);
        let mut line = format!("{}:", stream.name);
        let (mut lowest, mut highest) = (1f64, 1f64);
        for (mask_name, mask) in masks() {
            let (c, rs) = stream.bytes(&pcm, false, Some(&mask));
            let (c_share, rs_share) = (c as f64 / c_plain as f64, rs as f64 / rs_plain as f64);
            line += &format!(" [{mask_name}] {rs_share:.3} (libopus {c_share:.3})");
            lowest = lowest.min(rs_share);
            highest = highest.max(rs_share);
            if (c_share - rs_share).abs() > 0.04 {
                failures.push(format!("{}, mask {mask_name}: {rs_share:.3} of the size without a mask, libopus {c_share:.3}", stream.name));
            }
        }
        println!("{line}");
        // The mask does change the bitrate, both ways.
        assert!(lowest < 0.45 && highest > 1.2, "{}: shares from {lowest:.3} to {highest:.3}", stream.name);
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// With a constant bitrate a mask changes no packet size, in any mode. A
/// SILK-only stream does not change at all: the mask only moves SILK's
/// bitrate, and only when that is variable (opus_encoder.c:2069,
/// `st->use_vbr`). libopus's stream stays the same too.
#[test]
fn energy_mask_leaves_a_constant_bitrate_alone() {
    for stream in [
        Stream { name: "SILK-WB mono", rate: 16000, channels: 1, mode: SILK, bandwidth: WB, bitrate: 24000, frame_ms: 20 },
        Stream { name: "Hybrid-FB stereo", rate: 48000, channels: 2, mode: HYBRID, bandwidth: FB, bitrate: 64000, frame_ms: 20 },
        Stream { name: "CELT-FB mono", rate: 48000, channels: 1, mode: CELT, bandwidth: FB, bitrate: 64000, frame_ms: 10 },
    ] {
        let pcm = speech(stream.rate, stream.channels, 1.0);
        let plain = stream.packets(&pcm, true, None);
        let sizes = |packets: &[Vec<u8>]| packets.iter().map(|p| p.len()).collect::<Vec<usize>>();
        assert_eq!(sizes(&plain.0), sizes(&plain.1), "{}: packet sizes without a mask", stream.name);
        for (mask_name, mask) in masks() {
            let masked = stream.packets(&pcm, true, Some(&mask));
            assert_eq!(sizes(&masked.0), sizes(&plain.0), "{}, mask {mask_name}: libopus", stream.name);
            assert_eq!(sizes(&masked.1), sizes(&plain.1), "{}, mask {mask_name}", stream.name);
            if stream.mode.1 == MODE_SILK {
                assert!(masked.0 == plain.0, "{}, mask {mask_name}: libopus's packets change", stream.name);
                assert!(masked.1 == plain.1, "{}, mask {mask_name}: the packets change", stream.name);
            }
        }
    }
}

/// CELT-only streams. The CELT encoder evens out a mask that stays the
/// same, so here the mask changes every 8 packets between a deep and a
/// shallow one, as the mask of a surround stream changes with the other
/// channels. The packets under the deep mask are smaller. Their mean size
/// and that of the others are within 2 % of libopus's, and the share of one
/// in the other is libopus's to within 0.03.
#[test]
fn a_changing_energy_mask_moves_celt_packet_sizes_like_libopus() {
    let mut failures = Vec::new();
    for stream in [
        Stream { name: "CELT-FB mono 64 kb/s", rate: 48000, channels: 1, mode: CELT, bandwidth: FB, bitrate: 64000, frame_ms: 20 },
        Stream { name: "CELT-FB stereo 96 kb/s", rate: 48000, channels: 2, mode: CELT, bandwidth: FB, bitrate: 96000, frame_ms: 20 },
        Stream { name: "CELT-FB mono 32 kb/s, 10 ms", rate: 48000, channels: 1, mode: CELT, bandwidth: FB, bitrate: 32000, frame_ms: 10 },
        Stream { name: "CELT-SWB mono 48 kb/s from 24 kHz", rate: 24000, channels: 1, mode: CELT, bandwidth: SWB, bitrate: 48000, frame_ms: 20 },
        Stream { name: "CELT-FB mono 48 kb/s, 5 ms", rate: 48000, channels: 1, mode: CELT, bandwidth: FB, bitrate: 48000, frame_ms: 5 },
    ] {
        let pcm = speech(stream.rate, stream.channels, 4.0);
        let frame = stream.rate * stream.frame_ms / 1000;
        for (deep, shallow) in [(-2.0f32, 0.0f32), (-1.0, 0.0), (-2.0, -0.5), (-0.5, 0.25)] {
            const PERIOD: usize = 8;
            let is_deep = |i: usize| (i / PERIOD).is_multiple_of(2);
            let mut c_enc = CEncoder::new(stream.rate, stream.channels, APP_AUDIO);
            c_enc.force_mode(stream.mode.1).bandwidth(stream.bandwidth.1).bitrate(stream.bitrate);
            let mut enc = OpusEncoder::new(stream.rate as i32, stream.channels, Application::Audio).unwrap();
            enc.set_force_mode(Some(stream.mode.0));
            enc.set_bandwidth(stream.bandwidth.0);
            enc.set_bitrate(Bitrate::BitsPerSecond(stream.bitrate)).unwrap();
            let mut buf = vec![0u8; 4000];
            let (mut c_sizes, mut rs_sizes) = (Vec::new(), Vec::new());
            for (i, f) in pcm.chunks_exact(frame * stream.channels).enumerate() {
                let mask = [if is_deep(i) { deep } else { shallow }; 42];
                c_enc.energy_mask(Some(&mask));
                enc.set_energy_mask(Some(&mask));
                c_sizes.push(c_enc.encode(f).len());
                rs_sizes.push(enc.encode(f, frame, &mut buf).unwrap());
            }
            // Mean packet size under one of the two masks, without the
            // first two rounds.
            let mean = |sizes: &[usize], deep: bool| -> f64 {
                let picked: Vec<usize> =
                    sizes.iter().enumerate().skip(2 * PERIOD).filter(|(i, _)| is_deep(*i) == deep).map(|(_, s)| *s).collect();
                picked.iter().sum::<usize>() as f64 / picked.len() as f64
            };
            let (c_deep, c_shallow) = (mean(&c_sizes, true), mean(&c_sizes, false));
            let (rs_deep, rs_shallow) = (mean(&rs_sizes, true), mean(&rs_sizes, false));
            let (c_share, rs_share) = (c_deep / c_shallow, rs_deep / rs_shallow);
            println!("{}, masks {deep} and {shallow}: {rs_deep:.1} and {rs_shallow:.1} bytes (libopus {c_deep:.1} and {c_shallow:.1})", stream.name);
            assert!(c_share < 0.85, "{}: libopus's packets under the deep mask are {c_share:.3} of the others", stream.name);
            if (rs_deep - c_deep).abs() > 0.02 * c_deep || (rs_shallow - c_shallow).abs() > 0.02 * c_shallow || (rs_share - c_share).abs() > 0.03 {
                failures.push(format!(
                    "{}, masks {deep} and {shallow}: {rs_deep:.1} and {rs_shallow:.1} bytes, libopus {c_deep:.1} and {c_shallow:.1}",
                    stream.name
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Hybrid streams with a variable bitrate. The mask moves bits between the
/// two layers: SILK's bitrate follows the mask, and CELT gets what SILK
/// leaves of the total. The size of the stream changes little, and as in
/// libopus: its share of the size without a mask is libopus's to within
/// 0.03.
///
/// Left out are the masks with which libopus's rate runs away, see
/// `a_mask_cannot_make_a_hybrid_stream_run_away`.
#[test]
fn energy_mask_in_hybrid_streams_follows_libopus() {
    let mut failures = Vec::new();
    for (stream, runs_away) in [
        (Stream { name: "Hybrid-SWB mono 24 kb/s", rate: 24000, channels: 1, mode: HYBRID, bandwidth: SWB, bitrate: 24000, frame_ms: 20 }, &[][..]),
        (Stream { name: "Hybrid-FB mono 16 kb/s", rate: 48000, channels: 1, mode: HYBRID, bandwidth: FB, bitrate: 16000, frame_ms: 20 }, &["+0.5", "+3"][..]),
        (Stream { name: "Hybrid-FB mono 10245 b/s", rate: 48000, channels: 1, mode: HYBRID, bandwidth: FB, bitrate: 10245, frame_ms: 20 }, &["0", "+0.5", "+3"][..]),
        (Stream { name: "Hybrid-FB mono 20 kb/s, 10 ms", rate: 48000, channels: 1, mode: HYBRID, bandwidth: FB, bitrate: 20000, frame_ms: 10 }, &["+0.5", "+3"][..]),
    ] {
        let pcm = speech(stream.rate, stream.channels, 2.0);
        let (c_plain, rs_plain) = stream.bytes(&pcm, false, None);
        let mut line = format!("{}:", stream.name);
        for (mask_name, mask) in masks() {
            if runs_away.contains(&mask_name) {
                continue;
            }
            let (c, rs) = stream.bytes(&pcm, false, Some(&mask));
            let (c_share, rs_share) = (c as f64 / c_plain as f64, rs as f64 / rs_plain as f64);
            line += &format!(" [{mask_name}] {rs_share:.3} (libopus {c_share:.3})");
            if (c_share - rs_share).abs() > 0.03 {
                failures.push(format!("{}, mask {mask_name}: {rs_share:.3} of the size without a mask, libopus {c_share:.3}", stream.name));
            }
        }
        println!("{line}");
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The one place where opus1 does not follow libopus with a mask, on
/// purpose.
///
/// In a Hybrid packet CELT's bitrate is the total less SILK's
/// (opus_encoder.c:2454). A mask that is not negative raises SILK's bitrate
/// (C 2105). At a low total it then comes within 500 b/s of the total, the
/// lowest bitrate CELT takes. libopus's request to CELT fails, CELT keeps
/// the maximum bitrate it was given just before (C 2286), and the packets
/// grow to the size of the buffer: a 10 kb/s stream comes out at about
/// 500 kb/s. opus1 gives CELT 501 b/s in that case, and the stream stays
/// near its bitrate.
#[test]
fn a_mask_cannot_make_a_hybrid_stream_run_away() {
    for (bitrate, mask_value) in [(10245, 0.0f32), (10245, 0.5), (16000, 0.5), (12000, 0.25)] {
        let stream = Stream { name: "Hybrid-FB mono", rate: 48000, channels: 1, mode: HYBRID, bandwidth: FB, bitrate, frame_ms: 20 };
        let pcm = speech(48000, 1, 1.0);
        let (c_plain, rs_plain) = stream.bytes(&pcm, false, None);
        let (c, rs) = stream.bytes(&pcm, false, Some(&[mask_value; 21]));
        let (c_share, rs_share) = (c as f64 / c_plain as f64, rs as f64 / rs_plain as f64);
        println!("Hybrid-FB mono {bitrate} b/s, mask {mask_value}: {rs_share:.2} times the size without a mask, libopus {c_share:.2} times");
        assert!(c_share > 10.0, "{bitrate} b/s, mask {mask_value}: libopus was expected to run away, it is at {c_share:.2}");
        // The SILK layer takes a few kb/s more; nothing else grows.
        assert!((1.0..1.6).contains(&rs_share), "{bitrate} b/s, mask {mask_value}: {rs_share:.2} times the size without a mask");
        // The stream is still one libopus decodes.
        let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
        enc.set_force_mode(Some(ForcedMode::Hybrid));
        enc.set_bandwidth(Bandwidth::Fullband);
        enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
        enc.set_energy_mask(Some(&[mask_value; 21]));
        let mut dec = CDecoder::new(48000, 1);
        let mut buf = vec![0u8; 4000];
        for (i, f) in pcm.as_chunks::<960>().0.iter().enumerate() {
            let n = enc.encode(f, 960, &mut buf).unwrap();
            assert_eq!(toc(buf[0]).mode, "Hybrid", "packet {i}");
            dec.decode(Some(&buf[..n]), 960, false).unwrap_or_else(|e| panic!("packet {i}: libopus error {e}"));
            assert_eq!(enc.final_range(), dec.final_range(), "packet {i}: final range");
        }
    }
}

/// An LFE encoder ignores the mask (opus_encoder.c:2069 and
/// celt_encoder.c:2114, `!st->lfe`): the packets are those without a mask.
#[test]
fn an_lfe_encoder_ignores_the_mask() {
    let pcm = rumble(48000, 1.0);
    let encode = |mask: Option<&[f32]>| {
        let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
        enc.set_lfe(true);
        enc.set_bitrate(Bitrate::BitsPerSecond(6000)).unwrap();
        enc.set_energy_mask(mask);
        encode_all(&mut enc, &pcm, 960)
    };
    let plain = encode(None);
    assert!(plain.iter().all(|p| p.len() > 2));
    assert_eq!(encode(Some(&[-2.0; 21])), plain);
    assert_eq!(encode(Some(&[0.5; 21])), plain);
}

/// Power of `x` above about 9 kHz at 48 kHz: three passes of a second
/// difference, which removes the band the SILK layer codes.
fn high_band_power(x: &[f32]) -> f64 {
    let mut y: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    for _ in 0..3 {
        let mut z = vec![0.0; y.len()];
        for i in 2..y.len() {
            z[i] = 0.25 * (y[i] - 2.0 * y[i - 1] + y[i - 2]);
        }
        y = z;
    }
    y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64
}

/// A Hybrid packet with few bits for CELT has its band above 8 kHz
/// attenuated before CELT codes it: by `1 - 2^(-celt_rate / 1024)`, where
/// `celt_rate` is what SILK leaves of the bitrate (opus_encoder.c:2057-2062,
/// `HB_gain`). Not so with an energy mask: a surround stream keeps the band
/// at full level.
///
/// The mask here is -0.2 in every band, which leaves the SILK bitrate where
/// it is (the offset is `16000 * (mask + 0.2)`), so the gain is all that
/// changes. Decoded by libopus:
///
/// - without a mask, the band above 8 kHz of the opus1 stream is as loud
///   as that of the libopus stream, to within 0.5 dB;
/// - with the mask it is louder by what libopus gains, to within 0.5 dB:
///   3 dB at 11 kb/s, where the gain is 0.70, and 1.7 dB at 16 kb/s.
#[test]
fn a_mask_keeps_the_hybrid_high_band_at_full_level() {
    let pcm = speech(48000, 1, 4.0);
    let stream = |bitrate| Stream { name: "Hybrid-FB mono", rate: 48000, channels: 1, mode: HYBRID, bandwidth: FB, bitrate, frame_ms: 20 };
    for (bitrate, least_gain_db) in [(11000, 2.0), (12000, 2.0), (14000, 1.2), (16000, 1.0)] {
        // Power above 8 kHz of both streams, decoded by libopus.
        let powers = |mask: Option<&[f32]>| -> (f64, f64) {
            let (c_packets, packets) = stream(bitrate).packets(&pcm, false, mask);
            (high_band_power(&c_decode_all(&c_packets, 48000, 1, 960)), high_band_power(&c_decode_all(&packets, 48000, 1, 960)))
        };
        let plain = powers(None);
        let masked = powers(Some(&[-0.2f32; 21]));
        let db = |ratio: f64| 10.0 * ratio.log10();
        let (c_gain, rs_gain) = (db(masked.0 / plain.0), db(masked.1 / plain.1));
        let (level_plain, level_masked) = (db(plain.1 / plain.0), db(masked.1 / masked.0));
        println!("Hybrid-FB mono {bitrate} b/s: the mask raises the band above 8 kHz by {rs_gain:.2} dB (libopus {c_gain:.2} dB); opus1 against libopus {level_plain:+.2} dB without and {level_masked:+.2} dB with the mask");
        assert!(c_gain > least_gain_db, "{bitrate} b/s: libopus gains {c_gain:.2} dB");
        assert!((rs_gain - c_gain).abs() < 0.5, "{bitrate} b/s: the mask raises the high band by {rs_gain:.2} dB, in libopus by {c_gain:.2} dB");
        assert!(level_plain.abs() < 0.5 && level_masked.abs() < 0.5, "{bitrate} b/s: high band against libopus {level_plain:+.2} dB without and {level_masked:+.2} dB with the mask");
    }
}

/// A mask may hold any values: libopus does not check them, and a
/// multistream encoder computes them from the signal. Values that are not
/// numbers, infinite ones and huge ones, in every mode, with a variable and
/// a constant bitrate, with and without the LFE flag:
///
/// - no call panics or fails;
/// - libopus decodes every packet to the frame duration, and the result is
///   finite;
/// - no packet is larger than three times what the bitrate gives the frame,
///   plus 20 bytes. A mask can raise the SILK bitrate only a little
///   (opus_encoder.c:2090-2095 limits each value to 0.5 and halves what is
///   above zero), and the guard on the CELT bitrate of a Hybrid packet
///   holds for these values too.
#[test]
fn a_mask_of_any_values_gives_packets_libopus_decodes() {
    let special = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        1e30,
        -1e30,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        100.0,
        -100.0,
    ];
    let mut seed = 77u32;
    let mut random = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as usize
    };
    let mut packets = 0usize;
    for rate in [8000usize, 16000, 48000] {
        for channels in [1usize, 2] {
            let pcm = speech(rate, channels, 0.5);
            for mode in [None, Some(ForcedMode::SilkOnly), Some(ForcedMode::Hybrid), Some(ForcedMode::CeltOnly)] {
                for (cbr, lfe) in [(false, false), (true, false), (false, true)] {
                    for bitrate in [6000, 24000, 96000] {
                        for frame_tenth_ms in [25usize, 100, 200, 600] {
                            let frame = rate * frame_tenth_ms / 10_000;
                            let what = format!("{rate} Hz, {channels} ch, {mode:?}, CBR {cbr}, LFE {lfe}, {bitrate} b/s, {} ms", frame_tenth_ms as f64 / 10.0);
                            let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
                            enc.set_force_mode(mode);
                            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                            enc.set_vbr(!cbr);
                            enc.set_lfe(lfe);
                            let mut dec = CDecoder::new(rate, channels);
                            let mut buf = vec![0u8; 1500];
                            let nominal = bitrate as usize * frame_tenth_ms / 80_000;
                            for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
                                // A third of the values are special ones at
                                // random, a third one special value for the
                                // whole frame, a third plain values from -4
                                // to 4.
                                let mask: Vec<f32> = (0..21 * channels)
                                    .map(|_| match random() % 3 {
                                        0 => special[random() % special.len()],
                                        1 => special[i % special.len()],
                                        _ => (random() % 8001) as f32 / 1000.0 - 4.0,
                                    })
                                    .collect();
                                enc.set_energy_mask(Some(&mask));
                                let n = enc.encode(f, frame, &mut buf).unwrap_or_else(|e| panic!("{what}, packet {i}: {e}"));
                                assert!(n <= 3 * nominal + 20, "{what}, packet {i}: {n} bytes");
                                let out = dec
                                    .decode(Some(&buf[..n]), frame, false)
                                    .unwrap_or_else(|e| panic!("{what}, packet {i}: libopus error {e}"));
                                assert_eq!(out.len(), frame * channels, "{what}, packet {i}: duration");
                                assert!(out.iter().all(|v| v.is_finite()), "{what}, packet {i}: not finite");
                                packets += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(packets > 50_000, "{packets} packets");
}

/// Low tones in bursts, with a little noise: 55 and 110 Hz.
fn bursts(rate: usize, seconds: f64) -> Vec<f32> {
    let n = (rate as f64 * seconds) as usize;
    let mut seed = 9u32;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let level = if (t * 3.0).fract() < 0.4 { 1.0 } else { 0.1 };
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((seed >> 8) as f64 / (1u32 << 23) as f64 - 1.0) * 2e-3;
            let low = 0.5 * (2.0 * std::f64::consts::PI * 55.0 * t).sin();
            let high = 0.2 * (2.0 * std::f64::consts::PI * 110.0 * t).sin();
            (level * (low + high) + noise) as f32
        })
        .collect()
}

/// The LFE flag with settings a surround encoder does not use, but a caller
/// can: every application, input at 8 to 48 kHz, two channels, bitrates up
/// to 64 kb/s, complexities from 0 to 10, in-band FEC switched on, and a
/// forced SILK-only or Hybrid mode with fullband asked for.
///
/// The flag wins over all of them, as in libopus (opus_encoder.c 1538-1539,
/// 1683-1684): every packet has libopus's first byte, which is CELT-only
/// and narrowband wherever the packet carries data. With a constant bitrate
/// every packet has libopus's size. With a variable bitrate at most one
/// packet in 20 is more than 2 bytes from libopus's, and the stream is
/// within 3 % of libopus's.
#[test]
fn lfe_wins_over_application_mode_bandwidth_and_fec_like_libopus() {
    let mut failures = Vec::new();
    let (mut streams, mut lowest, mut highest) = (0, 1f64, 1f64);
    // Constant bitrate, complexity, FEC, forced mode.
    let variants = [
        (false, 10, false, None),
        (true, 5, false, None),
        (false, 0, true, Some((ForcedMode::SilkOnly, MODE_SILK))),
        (false, 9, false, Some((ForcedMode::Hybrid, MODE_HYBRID))),
    ];
    for rate in [8000usize, 24000, 48000] {
        for channels in [1usize, 2] {
            let mono = bursts(rate, 1.5);
            let pcm: Vec<f32> = if channels == 2 { mono.iter().flat_map(|&v| [v, 0.7 * v]).collect() } else { mono };
            for (app_name, c_app, app) in [("voip", APP_VOIP, Application::Voip), ("audio", APP_AUDIO, Application::Audio), ("low delay", APP_LOWDELAY, Application::RestrictedLowDelay)] {
                for bitrate in [3000, 8000, 64000] {
                    for frame_ms in [5usize, 10, 20] {
                        for (cbr, complexity, fec, forced) in variants {
                            let frame = rate * frame_ms / 1000;
                            let what = format!(
                                "{rate} Hz, {channels} ch, {app_name}, {bitrate} b/s, {frame_ms} ms, CBR {cbr}, complexity {complexity}, FEC {fec}, forced {:?}",
                                forced.map(|f| f.0)
                            );
                            let mut c_enc = CEncoder::new(rate, channels, c_app);
                            c_enc.lfe(true).bitrate(bitrate).vbr(!cbr).complexity(complexity).fec(fec, 10).bandwidth(BW_FB);
                            let mut enc = OpusEncoder::new(rate as i32, channels, app).unwrap();
                            enc.set_lfe(true);
                            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                            enc.set_vbr(!cbr);
                            enc.set_complexity(complexity).unwrap();
                            enc.set_inband_fec(fec);
                            enc.set_packet_loss_perc(10).unwrap();
                            enc.set_bandwidth(Bandwidth::Fullband);
                            if let Some((mode, c_mode)) = forced {
                                c_enc.force_mode(c_mode);
                                enc.set_force_mode(Some(mode));
                            }
                            let c_packets = c_enc.encode_all(&pcm, frame);
                            let packets = encode_all(&mut enc, &pcm, frame);
                            streams += 1;

                            let first_bytes = c_packets.iter().zip(&packets).filter(|(c, r)| c[0] != r[0]).count();
                            let not_lfe = packets.iter().filter(|p| !is_placeholder(p) && (toc(p[0]).mode, toc(p[0]).bandwidth) != ("CELT", "NB")).count();
                            let allowed = if cbr { 0 } else { 2 };
                            let sizes = c_packets.iter().zip(&packets).filter(|(c, r)| (c.len() as i64 - r.len() as i64).abs() > allowed).count();
                            let ratio = total_bytes(&packets) as f64 / total_bytes(&c_packets) as f64;
                            (lowest, highest) = (lowest.min(ratio), highest.max(ratio));
                            let too_many_sizes = if cbr { sizes != 0 } else { sizes * 20 > packets.len() };
                            if first_bytes != 0 || not_lfe != 0 || too_many_sizes || !(0.97..=1.03).contains(&ratio) {
                                failures.push(format!(
                                    "{what}: {first_bytes} first bytes differ, {not_lfe} packets with data are not CELT-only narrowband, {sizes} of {} sizes differ, {ratio:.3} of libopus's bytes",
                                    packets.len()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{streams} streams: {lowest:.3} to {highest:.3} of libopus's bytes");
    assert!(failures.is_empty(), "{} of {streams} streams differ:\n{}", failures.len(), failures.join("\n"));
}
