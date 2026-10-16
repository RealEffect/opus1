//! Stereo input coded as one channel by the SILK encoder, and the change
//! between one and two coded channels in both directions, against libopus
//! with `OPUS_SET_FORCE_CHANNELS`.
//!
//! libopus's `silk_Encode` (silk/enc_API.c) handles
//! `nChannelsInternal < nChannelsAPI`: it mixes left and right down before
//! the resampler and codes a single channel. The packet before the change
//! is still a stereo packet, coded with `toMono` set: its stereo predictors
//! are zero and its width falls to zero. The first packet that codes two
//! channels again starts the side encoder and the stereo state afresh.
//!
//! The test drives `SilkEncoder` directly, the way `OpusEncoder` drives it
//! for SILK-only packets.
// `RangeCoder::finish` needs the `std` feature.
#![cfg(feature = "std")]

mod silkenc_common;
use silkenc_common::*;

use opus1::hp_cutoff::dc_reject_float;
use opus1::range_coder::RangeCoder;
use opus1::silk::decode_indices::{silk_stereo_decode_mid_only, silk_stereo_decode_pred};
use opus1::silk::enc_api::{SilkControl, SilkEncoder, SilkInput, silk_downmix_stereo};

/// C `FLOAT2INT16`.
fn float2int16(x: f32) -> i16 {
    (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16
}

/// A SILK-only wideband encoder for stereo input, used the way `OpusEncoder`
/// uses it (application "audio", variable bitrate, complexity 9), plus
/// libopus's rule for the number of coded channels
/// (opus_encoder.c:1560-1571).
struct StereoSilk {
    fs: usize,
    /// Samples per channel in a packet, at `fs`.
    frame: usize,
    bitrate: i32,
    /// Packet loss in percent; in-band FEC is on when it is not 0.
    loss: i32,
    silk: SilkEncoder,
    hp_mem: [f32; 4],
    prev_stream_channels: usize,
    to_mono: bool,
}

struct Packet {
    bytes: Vec<u8>,
    final_range: u32,
    /// Channels the packet codes.
    channels: usize,
    to_mono: bool,
    stereo_width_q14: i32,
}

impl StereoSilk {
    fn new(fs: usize, frame: usize, bitrate: i32, loss: i32) -> Self {
        Self {
            fs,
            frame,
            bitrate,
            loss,
            silk: SilkEncoder::new(fs as i32, 2),
            hp_mem: [0.0; 4],
            prev_stream_channels: 2,
            to_mono: false,
        }
    }

    /// Encode one packet of interleaved stereo input. `want` is the number
    /// of channels the caller asks for, as with `OPUS_SET_FORCE_CHANNELS`.
    fn encode(&mut self, pcm: &[f32], want: usize) -> Packet {
        let (fs, frame) = (self.fs, self.frame);
        let frame_ms = (frame * 1000 / fs) as i32;
        // opus_encoder.c:1560-1571: "Delay stereo->mono transition ... so
        // that SILK can do a smooth downmix".
        let mut stream = want;
        if stream == 1 && self.prev_stream_channels == 2 && !self.to_mono {
            self.to_mono = true;
            stream = 2;
        } else {
            self.to_mono = false;
        }

        // The input of SILK: `dc_reject`, then 16 bits.
        let mut filtered = vec![0f32; 2 * frame];
        dc_reject_float(pcm, 3, &mut filtered, &mut self.hp_mem, frame, 2, fs as i32);
        let pcm16: Vec<i16> = filtered.iter().map(|&x| float2int16(x)).collect();

        // opus_encoder.c: the packet's bits less the first byte.
        let total_rate = (self.bitrate * frame_ms / 1000 - 8) * 1000 / frame_ms;
        let mut ctl = SilkControl {
            channels_internal: stream,
            payload_size_ms: frame_ms,
            min_khz: 16,
            max_khz: 16,
            desired_khz: 16,
            bit_rate: total_rate,
            max_bits: 1274 * 8,
            complexity: 9,
            packet_loss_perc: self.loss,
            lbrr_coded: self.loss > 0,
            activity: 1,
            to_mono: self.to_mono,
            ..SilkControl::default()
        };
        let mut rc = RangeCoder::new_encoder(1275);
        self.silk.encode(&mut ctl, SilkInput::Int16(&pcm16), None, &mut rc).expect("SILK error");
        assert_eq!(rc.error, 0, "range coder overflow");
        let final_range = rc.rng;
        // SILK wideband: configurations 8 to 11 are 10, 20, 40 and 60 ms.
        let config = match frame_ms {
            10 => 8u8,
            20 => 9,
            40 => 10,
            _ => 11,
        };
        let mut bytes = vec![(config << 3) | (((stream == 2) as u8) << 2)];
        bytes.extend(rc.finish());
        self.prev_stream_channels = stream;
        Packet {
            bytes,
            final_range,
            channels: stream,
            to_mono: self.to_mono,
            stereo_width_q14: ctl.stereo_width_q14,
        }
    }
}

/// The channels asked for at packet `i`: stereo, mono from packet 20, stereo
/// from 45, mono from 70, stereo from 85.
fn wanted(i: usize) -> usize {
    match i {
        0..20 => 2,
        20..45 => 1,
        45..70 => 2,
        70..85 => 1,
        _ => 2,
    }
}
const PACKETS: usize = 110;
/// The first packets that code one channel, and that code two again.
const TO_MONO: [usize; 2] = [21, 71];
const TO_STEREO: [usize; 2] = [45, 85];

struct Streams {
    frame: usize,
    pcm: Vec<f32>,
    rs: Vec<Packet>,
    c: Vec<Vec<u8>>,
}

fn streams(fs: usize, frame_ms: usize, bitrate: i32, loss: i32) -> Streams {
    let frame = fs * frame_ms / 1000;
    streams_of(speech(fs, 2, PACKETS * frame, 0.5), fs, frame_ms, bitrate, loss)
}

/// Both encoders on the interleaved stereo signal `pcm`.
fn streams_of(pcm: Vec<f32>, fs: usize, frame_ms: usize, bitrate: i32, loss: i32) -> Streams {
    let frame = fs * frame_ms / 1000;
    let mut rs_enc = StereoSilk::new(fs, frame, bitrate, loss);
    let mut c_enc = CEncoder::new(fs, 2, APP_AUDIO);
    c_enc.silk_only().bandwidth(BW_WB).bitrate(bitrate).complexity(9).fec(loss > 0, loss);
    let (mut rs, mut c) = (Vec::new(), Vec::new());
    for (i, chunk) in pcm.chunks_exact(2 * frame).enumerate() {
        c_enc.force_channels(wanted(i) as i32);
        c.push(c_enc.encode(chunk));
        rs.push(rs_enc.encode(chunk, wanted(i)));
    }
    Streams { frame, pcm, rs, c }
}

/// The LBRR flag of each coded channel in the header of a SILK packet.
fn lbrr_flags(packet: &[u8], frame_ms: usize) -> Vec<bool> {
    let n = (frame_ms / 20).max(1);
    let channels = 1 + ((packet[0] >> 2) & 1) as usize;
    let header = u16::from_be_bytes([packet[1], *packet.get(2).unwrap_or(&0)]);
    (0..channels).map(|c| (header >> (15 - (c * (n + 1) + n))) & 1 == 1).collect()
}

/// The packets code one or two channels where libopus's do, the libopus
/// decoder reads every packet to its end (its final range is the
/// encoder's), and FEC data is carried in the same packets. In particular
/// the first packet after a change carries none, in both directions.
#[test]
fn packets_have_libopus_channel_count_and_fec_flags() {
    for (fs, frame_ms) in [(16_000, 20), (48_000, 20), (16_000, 60), (48_000, 10)] {
        let s = streams(fs, frame_ms, 44_000, 10);
        let what = format!("{fs} Hz, {frame_ms} ms");
        let mut dec = CDecoder::new(fs, 2);
        let mut fec_packets = 0;
        for (i, (rs, c)) in s.rs.iter().zip(&s.c).enumerate() {
            assert_eq!(rs.bytes[0], c[0], "{what}: first byte of packet {i}");
            assert_eq!(rs.channels == 1, TO_MONO[0] <= i && i < TO_STEREO[0] || TO_MONO[1] <= i && i < TO_STEREO[1]);
            dec.decode(Some(&rs.bytes), s.frame, false).unwrap();
            assert_eq!(dec.final_range(), rs.final_range, "{what}: final range of packet {i}");
            let (rs_flags, c_flags) = (lbrr_flags(&rs.bytes, frame_ms), lbrr_flags(c, frame_ms));
            // The mid, or the single channel, must agree. The side's voice
            // activity depends on the bits spent so far, see silkenc_fec.rs.
            assert_eq!(rs_flags[0], c_flags[0], "{what}: FEC flag of packet {i}");
            fec_packets += c_flags[0] as usize;
            if TO_MONO.contains(&i) || TO_STEREO.contains(&i) {
                assert!(
                    rs_flags.iter().all(|&f| !f),
                    "{what}: packet {i} follows a change of the channel count and carries FEC data"
                );
            }
        }
        assert!(fec_packets > PACKETS / 3, "{what}: {fec_packets} packets with FEC data");
    }
}

/// The packet before the first one-channel packet is coded with `toMono`:
/// each of its frames has zero stereo predictors, as in libopus, and the
/// width reported for it is 0. Other stereo packets have predictors.
#[test]
fn the_last_stereo_packet_has_zero_predictors() {
    for (fs, frame_ms) in [(16_000, 20), (48_000, 20), (16_000, 10)] {
        // No FEC, so that the predictors of the first frame come right
        // after the four flags of the header.
        let s = streams(fs, frame_ms, 40_000, 0);
        let first_pred = |packet: &[u8]| {
            let mut rc = RangeCoder::new_decoder(&packet[1..]);
            for _ in 0..4 {
                rc.decode_bit_logp(1);
            }
            silk_stereo_decode_pred(&mut rc)
        };
        let mut other_zero = 0;
        for (i, (rs, c)) in s.rs.iter().zip(&s.c).enumerate() {
            if rs.channels == 1 {
                continue;
            }
            let last_stereo = TO_MONO.contains(&(i + 1));
            assert_eq!(rs.to_mono, last_stereo, "packet {i}");
            if last_stereo {
                assert_eq!(first_pred(c), [0, 0], "{fs} Hz {frame_ms} ms: libopus packet {i}");
                assert_eq!(first_pred(&rs.bytes), [0, 0], "{fs} Hz {frame_ms} ms: packet {i}");
                assert_eq!(rs.stereo_width_q14, 0);
            } else if first_pred(&rs.bytes) == [0, 0] {
                other_zero += 1;
            }
        }
        assert!(other_zero < 10, "{other_zero} other stereo packets with zero predictors");
    }
}

/// Whether the first frame of a stereo packet without FEC data codes the
/// mid only. `None` when the side has voice activity: the flag is then not
/// written, and the side is coded.
fn first_frame_mid_only(packet: &[u8], frame_ms: usize) -> Option<bool> {
    assert!(packet[0] & 4 != 0, "not a stereo packet");
    let n = (frame_ms / 20).max(1);
    let mut rc = RangeCoder::new_decoder(&packet[1..]);
    // The mid's voice activity and FEC flags, then the side's.
    let flags: Vec<bool> = (0..2 * (n + 1)).map(|_| rc.decode_bit_logp(1)).collect();
    assert!(!flags[n] && !flags[2 * n + 1], "the packet carries FEC data");
    silk_stereo_decode_pred(&mut rc);
    (!flags[n + 1]).then(|| silk_stereo_decode_mid_only(&mut rc))
}

/// The first packet that codes two channels again starts from the stereo
/// state that libopus sets up for it (silk/enc_API.c:194-211): no memory of
/// the predictors, of the width and of the levels of mid and side from the
/// stereo packets before the one-channel packets.
///
/// The input has two unrelated channels until the middle of the one-channel
/// packets and two equal channels from there on. With the levels reset, the
/// first two-channel packet finds no side at all and codes the mid only, as
/// libopus's packet does. With the levels of the unrelated channels kept, it
/// would code a side at full width.
#[test]
fn the_first_stereo_packet_after_mono_starts_from_a_new_stereo_state() {
    for (fs, frame_ms) in [(16_000usize, 20usize), (48_000, 20), (16_000, 40), (16_000, 60)] {
        let frame = fs * frame_ms / 1000;
        let n = PACKETS * frame;
        // The left channel is voiced throughout, so that the smoothers of the
        // stereo state, which follow the speech activity, move in every frame.
        let (a, b) = (voiced_like(fs, 1, n, 5), speech(fs, 1, n, 7.0));
        // Equal channels from packet 30, which lies in the first stretch of
        // one-channel packets.
        let equal_from = 30 * frame;
        let pcm: Vec<f32> = (0..n).flat_map(|i| [a[i], if i < equal_from { b[i] } else { a[i] }]).collect();
        let s = streams_of(pcm, fs, frame_ms, 44_000, 0);
        let what = format!("{fs} Hz, {frame_ms} ms");

        // The stereo packets before the change code a side: the levels that
        // must not survive are there.
        let side_coded = (5..19).filter(|&i| first_frame_mid_only(&s.c[i], frame_ms) != Some(true)).count();
        assert!(side_coded >= 12, "{what}: libopus codes a side in {side_coded} of 14 packets");

        // The levels follow the input at a speed set by the speech activity
        // of the frame before. After seconds of a steady voiced sound SILK
        // takes it for background noise, the levels then stay where the
        // reset put them, and the packet codes a side. Such a packet tells
        // nothing about the reset and is left out.
        let mut checked = 0;
        for &i in &TO_STEREO {
            assert_eq!(s.rs[i].channels, 2);
            if first_frame_mid_only(&s.c[i], frame_ms) != Some(true) {
                continue;
            }
            checked += 1;
            assert_eq!(first_frame_mid_only(&s.rs[i].bytes, frame_ms), Some(true), "{what}: packet {i}");
        }
        assert!(checked >= 1, "{what}: no libopus packet after a change codes the mid only");
    }
}

/// (left + right) / 2 of an interleaved stereo signal.
fn mid(stereo: &[f32]) -> Vec<f32> {
    stereo.as_chunks::<2>().0.iter().map(|lr| 0.5 * (lr[0] + lr[1])).collect()
}

/// (left - right) / 2 of an interleaved stereo signal.
fn side(stereo: &[f32]) -> Vec<f32> {
    stereo.as_chunks::<2>().0.iter().map(|lr| 0.5 * (lr[0] - lr[1])).collect()
}

/// The decoded signal is as close to the input as that of libopus's
/// packets over the whole stream, and the 100 ms around each change of the
/// channel count, where a fault in the hand-over would show, cost no more
/// than they cost libopus.
#[test]
fn decoded_quality_matches_libopus_also_around_the_changes() {
    for (fs, frame_ms) in [(16_000usize, 20usize), (48_000, 20), (16_000, 60)] {
        let s = streams(fs, frame_ms, 44_000, 0);
        let what = format!("{fs} Hz, {frame_ms} ms");
        let decode = |packets: &mut dyn Iterator<Item = &[u8]>| {
            let mut dec = CDecoder::new(fs, 2);
            let mut out = Vec::new();
            for p in packets {
                out.extend(dec.decode(Some(p), s.frame, false).unwrap());
            }
            out
        };
        let rs_out = decode(&mut s.rs.iter().map(|p| &p.bytes[..]));
        let c_out = decode(&mut s.c.iter().map(|p| &p[..]));

        // One-channel packets decode to two equal channels.
        let one_channel = TO_MONO[0] + 2..TO_STEREO[0];
        let range = one_channel.start * s.frame..one_channel.end * s.frame;
        assert!(level_db(&side(&rs_out)[range.clone()]) < -90.0, "{what}: side of mono packets");
        // Stereo packets do not: the input has a side.
        let stereo = 5 * s.frame..18 * s.frame;
        let (rs_side, c_side) = (level_db(&side(&rs_out)[stereo.clone()]), level_db(&side(&c_out)[stereo]));
        assert!((rs_side - c_side).abs() < 3.0, "{what}: side level {rs_side:.1} dB, libopus {c_side:.1} dB");

        // The mid is coded throughout.
        let (reference, rs_mid, c_mid) = (mid(&s.pcm), mid(&rs_out), mid(&c_out));
        let (rs_snr, rs_delay) = best_snr(&reference, &rs_mid, 1, fs / 40);
        let (c_snr, c_delay) = best_snr(&reference, &c_mid, 1, fs / 40);
        // The best alignment may fall on either side of a half sample.
        assert!(rs_delay.abs_diff(c_delay) <= 1, "{what}: delay {rs_delay}, libopus {c_delay}");
        // On this recording the SILK encoder of opus1 is 1 to 2.5 dB under
        // libopus at these rates, with or without changes of the channel
        // count (`OpusEncoder` shows the same for plain mono and stereo).
        assert!(rs_snr > c_snr - 3.0, "{what}: opus1 {rs_snr:.2} dB, libopus {c_snr:.2} dB");

        // Around the changes: from the packet before to the fourth after.
        let (mut rs_e, mut c_e, mut sig) = (0f64, 0f64, 0f64);
        for &change in TO_MONO.iter().chain(&TO_STEREO) {
            let frames_100ms = (100 / frame_ms).max(2);
            let (start, end) = ((change - 1) * s.frame, (change - 1 + frames_100ms) * s.frame);
            for i in start..end {
                let r = reference[i] as f64;
                sig += r * r;
                rs_e += (r - rs_mid[i + rs_delay] as f64).powi(2);
                c_e += (r - c_mid[i + c_delay] as f64).powi(2);
            }
        }
        let (rs_change, c_change) = (10.0 * (sig / rs_e).log10(), 10.0 * (sig / c_e).log10());
        // What the changes cost: the figure around them against the figure
        // of the whole stream, for each encoder.
        assert!(
            rs_change - rs_snr > c_change - c_snr - 2.0,
            "{what}: around the changes opus1 {rs_change:.2} dB (stream {rs_snr:.2} dB), \
             libopus {c_change:.2} dB (stream {c_snr:.2} dB)"
        );
        // Shown with `--nocapture`.
        println!(
            "{what}: stream opus1 {rs_snr:.2} dB, libopus {c_snr:.2} dB; \
             around the changes {rs_change:.2} dB, {c_change:.2} dB"
        );
    }
}

/// `silk_downmix_stereo` is libopus's rounding: half the sum, ties away
/// from minus infinity, and no clipping of the sum.
#[test]
fn downmix_rounds_like_libopus() {
    let stereo = [1i16, 2, -1, -2, 32767, 32767, -32768, -32768, 3, -3, 32767, -32768, 0, 1];
    let mut mono = [0i16; 7];
    silk_downmix_stereo(&stereo, &mut mono);
    // silk_RSHIFT_ROUND( sum, 1 ) = ( sum + 1 ) >> 1.
    assert_eq!(mono, [2, -1, 32767, -32768, 0, 0, 1]);
}
