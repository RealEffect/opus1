//! `OpusDecoder::decode_fec` against C `opus_decode_float` with
//! `decode_fec = 1`: a lost packet is rebuilt from the low-bitrate copy
//! (in-band forward error correction, FEC) that the next packet carries.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech_fullband, to_stereo};
use opus1::{Channels, OpusDecoder, SampleRate};

/// One stream to test: how the C encoder is set up and how it is decoded.
#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    rate: usize,
    channels: usize,
    mode: i32,
    bandwidth: i32,
    bitrate: i32,
    /// Frame duration in units of 2.5 ms (8 = 20 ms).
    frame_2_5ms: usize,
    /// In-band FEC switched on in the encoder.
    fec: bool,
}

impl Case {
    fn frame(&self) -> usize {
        self.rate / 400 * self.frame_2_5ms
    }

    fn packets(&self) -> Vec<Vec<u8>> {
        let mono = speech_fullband(self.rate, 3.0);
        let pcm = if self.channels == 2 { to_stereo(&mono, self.rate) } else { mono };
        let mut enc = CEncoder::new(self.rate, self.channels, APP_VOIP);
        enc.force_mode(self.mode).bandwidth(self.bandwidth).bitrate(self.bitrate);
        enc.fec(self.fec, if self.fec { 20 } else { 0 });
        enc.encode_all(&pcm, self.frame())
    }
}

/// What happens to each packet of a stream.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fate {
    /// Received and decoded.
    Good,
    /// Lost, and the next packet is there to recover it from.
    Recovered,
    /// Lost together with the packet after it: nothing to recover it from.
    /// It is produced together with the following `Recovered` packet, in one
    /// call that asks for two packet durations.
    LostBeforeRecovered,
}

/// Every tenth packet is lost. With `double`, every fortieth loss takes two
/// packets in a row.
fn fates(n: usize, double: bool) -> Vec<Fate> {
    let mut fates = vec![Fate::Good; n];
    for i in (7..n.saturating_sub(2)).step_by(10) {
        fates[i] = Fate::Recovered;
        if double && i % 20 == 7 {
            fates[i - 1] = Fate::LostBeforeRecovered;
        }
    }
    fates
}

struct Run {
    /// Output of libopus and of opus1, interleaved.
    c: Vec<f32>,
    rs: Vec<f32>,
    /// The loss-free decode by libopus.
    clean: Vec<f32>,
    /// Final range after every call, for libopus and opus1.
    ranges: Vec<(u32, u32)>,
    /// Sample ranges (per channel) that came out of an FEC call.
    recovered: Vec<std::ops::Range<usize>>,
    /// Indices into `ranges` of the FEC calls.
    fec_calls: Vec<usize>,
}

/// Decode a stream with losses in both decoders, calling FEC decode for
/// every lost packet that the next packet can stand in for.
fn run(case: &Case, double: bool) -> Run {
    let packets = case.packets();
    let frame = case.frame();
    let ch = case.channels;
    let fates = fates(packets.len(), double);
    let mut c_dec = CDecoder::new(case.rate, ch);
    let mut rs_dec = OpusDecoder::new(
        SampleRate::try_from(case.rate as i32).unwrap(),
        Channels::try_from(ch).unwrap(),
    )
    .unwrap();
    let mut clean_dec = CDecoder::new(case.rate, ch);
    let mut out = Run { c: vec![], rs: vec![], clean: vec![], ranges: vec![], recovered: vec![], fec_calls: vec![] };
    let mut buf = vec![0f32; 2 * frame * ch];
    for (i, p) in packets.iter().enumerate() {
        out.clean.extend(clean_dec.decode(Some(p), frame, false).unwrap());
        match fates[i] {
            Fate::LostBeforeRecovered => continue,
            Fate::Good => {
                out.c.extend(c_dec.decode(Some(p), frame, false).unwrap());
                let n = rs_dec.decode(p, frame, &mut buf[..frame * ch]).unwrap();
                assert_eq!(n, frame, "{}: packet {i}", case.name);
                out.rs.extend_from_slice(&buf[..frame * ch]);
            }
            Fate::Recovered => {
                // One or two packet durations, from the FEC data of packet i + 1.
                let lost = if i > 0 && fates[i - 1] == Fate::LostBeforeRecovered { 2 } else { 1 };
                let size = lost * frame;
                let start = out.c.len() / ch;
                out.c.extend(c_dec.decode(Some(&packets[i + 1]), size, true).unwrap());
                let n = rs_dec.decode_fec(&packets[i + 1], size, &mut buf[..size * ch]).unwrap();
                assert_eq!(n, size, "{}: FEC for packet {i}", case.name);
                out.rs.extend_from_slice(&buf[..size * ch]);
                out.recovered.push(start..start + size);
                out.fec_calls.push(out.ranges.len());
            }
        }
        out.ranges.push((c_dec.final_range(), rs_dec.final_range()));
    }
    assert_eq!(out.c.len(), out.rs.len());
    assert_eq!(out.c.len(), out.clean.len());
    assert!(out.recovered.len() >= 5, "{}: {} losses", case.name, out.recovered.len());
    out
}

impl Run {
    /// SNR of `x` against `reference` over the recovered samples only.
    fn recovered_snr(&self, reference: &[f32], x: &[f32], ch: usize) -> f64 {
        let (mut r, mut t) = (Vec::new(), Vec::new());
        for range in &self.recovered {
            r.extend_from_slice(&reference[range.start * ch..range.end * ch]);
            t.extend_from_slice(&x[range.start * ch..range.end * ch]);
        }
        snr(&r, &t)
    }
}

const SILK_MONO: [Case; 9] = [
    Case { name: "SILK-NB 8k 20ms", rate: 8000, channels: 1, mode: MODE_SILK, bandwidth: BW_NB, bitrate: 20000, frame_2_5ms: 8, fec: true },
    Case { name: "SILK-MB 12k 20ms", rate: 12000, channels: 1, mode: MODE_SILK, bandwidth: BW_MB, bitrate: 24000, frame_2_5ms: 8, fec: true },
    Case { name: "SILK-WB 16k 20ms", rate: 16000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 32000, frame_2_5ms: 8, fec: true },
    Case { name: "SILK-WB 24k 20ms", rate: 24000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 32000, frame_2_5ms: 8, fec: true },
    Case { name: "SILK-WB 48k 20ms", rate: 48000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 32000, frame_2_5ms: 8, fec: true },
    Case { name: "SILK-WB 16k 10ms", rate: 16000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 40000, frame_2_5ms: 4, fec: true },
    Case { name: "SILK-WB 24k 40ms", rate: 24000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 32000, frame_2_5ms: 16, fec: true },
    Case { name: "SILK-NB 48k 60ms", rate: 48000, channels: 1, mode: MODE_SILK, bandwidth: BW_NB, bitrate: 20000, frame_2_5ms: 24, fec: true },
    // No FEC data in the stream: the SILK decoder conceals, after it has
    // read from the packet that there is no copy.
    Case { name: "SILK-WB 16k 20ms, FEC off", rate: 16000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 24000, frame_2_5ms: 8, fec: false },
];

/// Mono SILK-only streams: FEC decoding gives the samples libopus gives,
/// in the recovered frames and in everything after them, and the same final
/// range after every call.
#[test]
fn mono_silk_fec_matches_libopus() {
    for case in &SILK_MONO {
        for double in [false, true] {
            let r = run(case, double);
            let whole = snr(&r.c, &r.rs);
            let recovered = r.recovered_snr(&r.c, &r.rs, 1);
            // How much the FEC copy is worth: recovered frames against the
            // loss-free decode.
            let gain = r.recovered_snr(&r.clean, &r.rs, 1);
            println!("{} (double loss: {double}): against libopus {whole:.1} dB, recovered frames {recovered:.1} dB; recovered frames against the loss-free decode {gain:.1} dB", case.name);
            assert!(recovered > 90.0 && whole > 90.0, "{}: {whole:.1} dB, recovered frames {recovered:.1} dB against libopus", case.name);
            let bad = r.ranges.iter().filter(|(c, rs)| c != rs).count();
            assert_eq!(bad, 0, "{}: final range differs after {bad} of {} calls", case.name, r.ranges.len());
        }
    }
}

/// Every third packet lost, for each of the three positions the losses can
/// take. Returns the lowest SNR of any call against libopus, the number of
/// calls after which the final range differs, and the number of losses.
fn dense_loss_run(case: &Case) -> (f64, usize, usize) {
    let packets = case.packets();
    let frame = case.frame();
    let (mut worst, mut range_mismatches, mut losses) = (200f64, 0, 0);
    for phase in 0..3 {
        let mut c_dec = CDecoder::new(case.rate, 1);
        let mut rs_dec = OpusDecoder::new(
            SampleRate::try_from(case.rate as i32).unwrap(),
            Channels::Mono,
        )
        .unwrap();
        let mut buf = vec![0f32; frame];
        for i in 0..packets.len() - 1 {
            let reference = if i > 3 && i % 3 == phase {
                rs_dec.decode_fec(&packets[i + 1], frame, &mut buf).unwrap();
                losses += 1;
                c_dec.decode(Some(&packets[i + 1]), frame, true).unwrap()
            } else {
                rs_dec.decode(&packets[i], frame, &mut buf).unwrap();
                c_dec.decode(Some(&packets[i]), frame, false).unwrap()
            };
            worst = worst.min(snr(&reference, &buf));
            range_mismatches += (c_dec.final_range() != rs_dec.final_range()) as usize;
        }
    }
    (worst, range_mismatches, losses)
}

/// A third of the packets lost: every call still gives libopus's samples
/// and final range. Frames that carry no FEC copy are concealed inside the
/// SILK decoder, so this covers every mix of copies and gaps.
#[test]
fn mono_silk_fec_with_dense_losses_matches_libopus() {
    for case in SILK_MONO.iter().filter(|c| c.frame_2_5ms <= 8) {
        let (worst, range_mismatches, losses) = dense_loss_run(case);
        println!("{}: {losses} losses, worst call {worst:.1} dB against libopus, {range_mismatches} final range mismatches", case.name);
        assert!(worst > 100.0 && range_mismatches == 0, "{}: worst call {worst:.1} dB, {range_mismatches} final range mismatches", case.name);
    }
}

/// The same for 40 and 60 ms SILK packets, which hold 2 or 3 SILK frames.
///
/// libopus decodes the FEC copy of a frame without reference to the frame
/// before it when that frame has no copy (dec_API.c:341-342,
/// `condCoding = LBRR_flags[FrameIndex - 1] ? CODE_CONDITIONALLY :
/// CODE_INDEPENDENTLY`). A decoder that always refers to the frame before
/// decodes such a packet to noise.
#[test]
fn mono_silk_fec_in_40_and_60_ms_packets_matches_libopus() {
    for case in [
        Case { name: "SILK-WB 16k 40ms", rate: 16000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 24000, frame_2_5ms: 16, fec: true },
        Case { name: "SILK-WB 16k 60ms", rate: 16000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 24000, frame_2_5ms: 24, fec: true },
        Case { name: "SILK-MB 12k 40ms", rate: 12000, channels: 1, mode: MODE_SILK, bandwidth: BW_MB, bitrate: 20000, frame_2_5ms: 16, fec: true },
        Case { name: "SILK-WB 48k 60ms", rate: 48000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 20000, frame_2_5ms: 24, fec: true },
    ] {
        let (worst, range_mismatches, losses) = dense_loss_run(&case);
        println!("{}: {losses} losses, worst call {worst:.1} dB against libopus, {range_mismatches} final range mismatches", case.name);
        assert!(worst > 100.0 && range_mismatches == 0, "{}: worst call {worst:.1} dB, {range_mismatches} final range mismatches", case.name);
    }
}

/// The measure of the gap this closes (see `k24_known_gaps.rs`): with FEC
/// the lost frames are closer to the loss-free decode than concealment gets
/// them, and as close as libopus gets them.
#[test]
fn fec_recovers_what_concealment_cannot() {
    let case = Case { name: "SILK-WB 24k 20ms", rate: 24000, channels: 1, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 32000, frame_2_5ms: 8, fec: true };
    let r = run(&case, false);
    // The same losses, concealed.
    let packets = case.packets();
    let fates = fates(packets.len(), false);
    let mut dec = OpusDecoder::new(SampleRate::Hz24000, Channels::Mono).unwrap();
    let mut concealed = Vec::new();
    let mut buf = vec![0f32; 480];
    for (i, p) in packets.iter().enumerate() {
        if fates[i] == Fate::Good {
            dec.decode(p, 480, &mut buf).unwrap();
        } else {
            dec.decode(&[], 480, &mut buf).unwrap();
        }
        concealed.extend_from_slice(&buf);
    }
    let with_fec = r.recovered_snr(&r.clean, &r.rs, 1);
    let c_with_fec = r.recovered_snr(&r.clean, &r.c, 1);
    let with_plc = r.recovered_snr(&r.clean, &concealed, 1);
    println!("lost frames against the loss-free decode: opus1 FEC {with_fec:.1} dB, libopus FEC {c_with_fec:.1} dB, opus1 concealment {with_plc:.1} dB");
    assert!(with_fec > c_with_fec - 0.1, "opus1 {with_fec:.1} dB, libopus {c_with_fec:.1} dB");
    assert!(with_fec > with_plc + 3.0, "FEC {with_fec:.1} dB, concealment {with_plc:.1} dB");
}

/// Hybrid streams, mono and stereo: the FEC copy covers the SILK layer
/// (below 8 kHz). The CELT layer above is concealed with noise, drawn from
/// a random state that both decoders keep alike.
///
/// - The recovered frames, and everything else, are libopus's samples.
/// - After an FEC call the final range is the random state of the CELT
///   concealment, as in libopus (opus_decoder.c:621), not the state of the
///   range decoder after the SILK copy.
#[test]
fn hybrid_fec_matches_libopus() {
    for case in [
        Case { name: "Hybrid-SWB 24k 20ms", rate: 24000, channels: 1, mode: MODE_HYBRID, bandwidth: BW_SWB, bitrate: 40000, frame_2_5ms: 8, fec: true },
        Case { name: "Hybrid-FB 48k 20ms", rate: 48000, channels: 1, mode: MODE_HYBRID, bandwidth: BW_FB, bitrate: 48000, frame_2_5ms: 8, fec: true },
        Case { name: "Hybrid-FB 48k 10ms", rate: 48000, channels: 1, mode: MODE_HYBRID, bandwidth: BW_FB, bitrate: 56000, frame_2_5ms: 4, fec: true },
        Case { name: "Hybrid-FB 48k stereo 20ms", rate: 48000, channels: 2, mode: MODE_HYBRID, bandwidth: BW_FB, bitrate: 72000, frame_2_5ms: 8, fec: true },
        Case { name: "Hybrid-SWB 24k stereo 10ms", rate: 24000, channels: 2, mode: MODE_HYBRID, bandwidth: BW_SWB, bitrate: 80000, frame_2_5ms: 4, fec: true },
    ] {
        for double in [false, true] {
            let r = run(&case, double);
            let ch = case.channels;
            let ours = r.recovered_snr(&r.clean, &r.rs, ch);
            let theirs = r.recovered_snr(&r.clean, &r.c, ch);
            let against_c = r.recovered_snr(&r.c, &r.rs, ch);
            let whole = snr(&r.c, &r.rs);
            let bad = r.ranges.iter().filter(|(c, rs)| c != rs).count();
            // The calls that recovered a frame leave a final range that is
            // not 0: the CELT decoder's.
            let after_fec = r.fec_calls.iter().filter(|&&call| r.ranges[call].1 != 0).count();
            println!("{} (double loss: {double}): recovered frames against the loss-free decode: opus1 {ours:.1} dB, libopus {theirs:.1} dB; opus1 against libopus: recovered frames {against_c:.1} dB, whole stream {whole:.1} dB; final range differs after {bad} of {} calls", case.name, r.ranges.len());
            assert!(against_c > 100.0 && whole > 100.0, "{}: recovered frames {against_c:.1} dB, whole stream {whole:.1} dB against libopus", case.name);
            assert_eq!(bad, 0, "{}: final range differs after {bad} of {} calls", case.name, r.ranges.len());
            assert_eq!(after_fec, r.fec_calls.len(), "{}: final range is 0 after an FEC call", case.name);
        }
    }
}

/// A CELT-only packet cannot carry FEC data: `decode_fec` does what a plain
/// loss does, and the final range is 0.
#[test]
fn celt_only_packets_are_concealed() {
    for case in [
        Case { name: "CELT-FB 48k", rate: 48000, channels: 1, mode: MODE_CELT, bandwidth: BW_FB, bitrate: 64000, frame_2_5ms: 8, fec: true },
        Case { name: "CELT-FB 48k stereo", rate: 48000, channels: 2, mode: MODE_CELT, bandwidth: BW_FB, bitrate: 96000, frame_2_5ms: 8, fec: true },
    ] {
        let packets = case.packets();
        let (frame, ch) = (case.frame(), case.channels);
        let fates = fates(packets.len(), false);
        let mut with_fec_call = OpusDecoder::new(
            SampleRate::try_from(case.rate as i32).unwrap(),
            Channels::try_from(ch).unwrap(),
        )
        .unwrap();
        let mut with_loss_call = OpusDecoder::new(
            SampleRate::try_from(case.rate as i32).unwrap(),
            Channels::try_from(ch).unwrap(),
        )
        .unwrap();
        let (mut a, mut b) = (vec![0f32; frame * ch], vec![0f32; frame * ch]);
        let mut recovered = 0;
        for (i, p) in packets.iter().enumerate() {
            if fates[i] == Fate::Good {
                with_fec_call.decode(p, frame, &mut a).unwrap();
                with_loss_call.decode(p, frame, &mut b).unwrap();
            } else {
                assert_eq!(with_fec_call.decode_fec(&packets[i + 1], frame, &mut a), Ok(frame));
                assert_eq!(with_fec_call.final_range(), 0, "{}", case.name);
                with_loss_call.decode(&[], frame, &mut b).unwrap();
                recovered += 1;
            }
            assert_eq!(a, b, "{}: packet {i}", case.name);
        }
        assert!(recovered >= 10);
    }
}

/// What `decode_fec` does with arguments libopus rejects or treats as a
/// loss.
#[test]
fn fec_argument_handling_follows_libopus() {
    let case = SILK_MONO[2];
    let packets = case.packets();
    let frame = case.frame();
    let mut c_dec = CDecoder::new(case.rate, 1);
    let mut rs_dec =
        OpusDecoder::new(SampleRate::try_from(case.rate as i32).unwrap(), Channels::Mono).unwrap();
    let mut buf = vec![0f32; 4 * frame];
    for p in &packets[..20] {
        c_dec.decode(Some(p), frame, false).unwrap();
        rs_dec.decode(p, frame, &mut buf).unwrap();
    }
    // Not a multiple of 2.5 ms: an error on both sides.
    assert!(c_dec.decode(Some(&packets[21]), frame + 1, true).is_err());
    assert!(rs_dec.decode_fec(&packets[21], frame + 1, &mut buf).is_err());
    // Shorter than a frame of the packet: concealed, like a loss of that
    // duration.
    let expected = c_dec.decode(Some(&packets[21]), frame / 2, true).unwrap();
    assert_eq!(rs_dec.decode_fec(&packets[21], frame / 2, &mut buf), Ok(frame / 2));
    assert!(snr(&expected, &buf[..frame / 2]) > 90.0);
    assert_eq!((c_dec.final_range(), rs_dec.final_range()), (0, 0));
    // An empty packet: a loss.
    let expected = c_dec.decode(None, frame, true).unwrap();
    assert_eq!(rs_dec.decode_fec(&[], frame, &mut buf), Ok(frame));
    assert!(snr(&expected, &buf[..frame]) > 90.0);
    // A broken packet: an error, and the decoder goes on afterwards.
    assert!(c_dec.decode(Some(&[0x4B]), frame, true).is_err());
    assert!(rs_dec.decode_fec(&[0x4B], frame, &mut buf).is_err());
    // An output buffer too small for the lost duration: an error.
    assert!(rs_dec.decode_fec(&packets[22], frame, &mut buf[..frame - 1]).is_err());
    let expected = c_dec.decode(Some(&packets[22]), frame, false).unwrap();
    rs_dec.decode(&packets[22], frame, &mut buf).unwrap();
    assert!(snr(&expected, &buf[..frame]) > 90.0);
}

/// Stereo SILK streams. The copy of a stereo frame has its own stereo
/// predictor, and a side channel only when the encoder kept one
/// (dec_API.c:286-303 and 327-328).
#[test]
fn stereo_silk_fec_matches_libopus() {
    for case in [
        Case { name: "SILK-WB 16k stereo 20ms", rate: 16000, channels: 2, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 48000, frame_2_5ms: 8, fec: true },
        Case { name: "SILK-WB 24k stereo 20ms", rate: 24000, channels: 2, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 48000, frame_2_5ms: 8, fec: true },
        Case { name: "SILK-NB 48k stereo 20ms", rate: 48000, channels: 2, mode: MODE_SILK, bandwidth: BW_NB, bitrate: 32000, frame_2_5ms: 8, fec: true },
        Case { name: "SILK-WB 48k stereo 60ms", rate: 48000, channels: 2, mode: MODE_SILK, bandwidth: BW_WB, bitrate: 48000, frame_2_5ms: 24, fec: true },
    ] {
        for double in [false, true] {
            let r = run(&case, double);
            let whole = snr(&r.c, &r.rs);
            let recovered = r.recovered_snr(&r.c, &r.rs, 2);
            println!("{} (double loss: {double}): against libopus {whole:.1} dB, recovered frames {recovered:.1} dB", case.name);
            assert!(recovered > 90.0 && whole > 90.0, "{}: {whole:.1} dB, recovered frames {recovered:.1} dB against libopus", case.name);
            let bad = r.ranges.iter().filter(|(c, rs)| c != rs).count();
            assert_eq!(bad, 0, "{}: final range differs after {bad} of {} calls", case.name, r.ranges.len());
        }
    }
}

/// FEC for a loss at which the stream changes its mode, its bandwidth or
/// its channel count.
///
/// The redundant copy of a Hybrid frame has no CELT layer, so libopus
/// conceals the band above 8 kHz. Before that it sets the CELT band range
/// from the packet the copy came in (`CELT_SET_END_BAND`,
/// opus_decoder.c:556-580). The concealment therefore covers the bands of
/// that packet, not the bands of the last CELT frame decoded. The two differ
/// when the packet before the loss was super-wideband and the one after it
/// is fullband, or the other way round, or when the packets before the loss
/// were SILK-only.
///
/// Every packet in turn is lost and recovered from the one after it, so a
/// loss falls on every position next to a change. Each call is compared
/// with libopus: the samples and the final range.
#[test]
fn fec_across_a_change_of_mode_or_bandwidth_matches_libopus() {
    const RATE: usize = 48000;
    let frame = RATE / 50;
    let pcm = to_stereo(&speech_fullband(RATE, 4.0), RATE);
    // What the C encoder is asked for, 140 ms each: mode, bandwidth,
    // channels, bitrate.
    let schedule = [
        (MODE_HYBRID, BW_FB, 2, 48000),
        (MODE_HYBRID, BW_SWB, 2, 40000),
        (MODE_SILK, BW_WB, 2, 32000),
        (MODE_HYBRID, BW_FB, 1, 32000),
        (MODE_HYBRID, BW_SWB, 1, 28000),
        (MODE_SILK, BW_WB, 1, 24000),
        (MODE_HYBRID, BW_FB, 2, 56000),
        (MODE_HYBRID, BW_SWB, 2, 44000),
    ];
    let mut enc = CEncoder::new(RATE, 2, APP_VOIP);
    enc.fec(true, 8);
    let mut packets = Vec::new();
    for (i, f) in pcm.chunks_exact(frame * 2).enumerate() {
        let (mode, bandwidth, channels, bitrate) = schedule[(i / 7) % schedule.len()];
        enc.force_mode(mode).bandwidth(bandwidth).force_channels(channels).bitrate(bitrate);
        packets.push(enc.encode(f));
    }
    // The first byte without the frame count code: mode, bandwidth, frame
    // size and channel count.
    let config = |p: &[u8]| p[0] >> 2;
    // CELT bands of a packet: 0 for SILK-only, 19 or 21 for Hybrid.
    let bands = |p: &[u8]| match p[0] >> 3 {
        0..=11 => 0,
        12 | 13 => 19,
        _ => 21,
    };

    let mut failures = Vec::new();
    // FEC calls at a change, by what changes.
    let (mut to_wider, mut to_narrower, mut from_silk, mut other_channels) = (0, 0, 0, 0);
    for dec_ch in [1usize, 2] {
        for rate in [16000usize, 48000] {
            let n = rate / 50;
            for phase in 0..7 {
                let mut c_dec = CDecoder::new(rate, dec_ch);
                let mut rs_dec = OpusDecoder::new(
                    SampleRate::try_from(rate as i32).unwrap(),
                    Channels::try_from(dec_ch).unwrap(),
                )
                .unwrap();
                let mut buf = vec![0f32; n * dec_ch];
                let (mut worst, mut worst_at, mut bad_ranges) = (200f64, 0, 0);
                for i in 0..packets.len() {
                    let lost = i >= 3 && i % 7 == phase && i + 1 < packets.len();
                    let (expected, got) = if lost {
                        let (before, next) = (&packets[i - 1], &packets[i + 1]);
                        if config(before) != config(next) {
                            to_wider += (bands(before) == 19 && bands(next) == 21) as usize;
                            to_narrower += (bands(before) == 21 && bands(next) == 19) as usize;
                            from_silk += (bands(before) == 0 && bands(next) != 0) as usize;
                            other_channels += ((before[0] ^ next[0]) & 4 != 0) as usize;
                        }
                        (c_dec.decode(Some(next), n, true), rs_dec.decode_fec(next, n, &mut buf))
                    } else {
                        (c_dec.decode(Some(&packets[i]), n, false), rs_dec.decode(&packets[i], n, &mut buf))
                    };
                    let expected = expected.unwrap();
                    assert_eq!(got, Ok(n), "packet {i}");
                    let call = snr(&expected, &buf);
                    if call < worst {
                        (worst, worst_at) = (call, i);
                    }
                    bad_ranges += (c_dec.final_range() != rs_dec.final_range()) as usize;
                }
                if worst < 100.0 || bad_ranges != 0 {
                    failures.push(format!(
                        "{dec_ch}-channel decoder at {rate} Hz, losses at {phase} of 7: worst call {worst:.1} dB (packet {worst_at}), final range differs after {bad_ranges} calls"
                    ));
                }
            }
        }
    }
    println!("FEC calls at a change: {to_wider} to fullband, {to_narrower} to super-wideband, {from_silk} from SILK-only to Hybrid, {other_channels} with another channel count");
    assert!(to_wider >= 8 && to_narrower >= 8 && from_silk >= 8 && other_channels >= 8, "the stream lacks changes");
    assert!(failures.is_empty(), "{} of 28 runs differ from libopus:\n{}", failures.len(), failures.join("\n"));
}
