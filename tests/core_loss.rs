//! Lost packets against C libopus, in the cases where the lost duration is
//! not the frame duration of the stream, where FEC decode is the first call
//! into a decoder, and around a decoder reset.
//!
//! Most streams are SILK-only: SILK conceals with integer arithmetic, so
//! opus-rs has to give libopus's samples exactly. The CELT layer of a Hybrid
//! stream is concealed with noise from a random state that both decoders
//! keep alike, and agrees to rounding. A CELT-only stream is concealed by
//! repeating the last pitch period, which agrees less closely
//! (tests/celtdec_plc.rs), so no samples of that are compared here.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{multitone, speech, speech_fullband, to_stereo};
use opus_rs::OpusDecoder;

/// Packets from the C encoder: speech (the 16 kHz recording, resampled),
/// with the mode and bandwidth forced.
fn c_packets(rate: usize, channels: usize, mode: i32, bandwidth: i32, bitrate: i32, frame: usize, fec: bool) -> Vec<Vec<u8>> {
    let mono = speech(rate, 2.0);
    let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };
    let mut enc = CEncoder::new(rate, channels, APP_VOIP);
    enc.force_mode(mode).bandwidth(bandwidth).bitrate(bitrate);
    enc.fec(fec, if fec { 25 } else { 0 });
    enc.encode_all(&pcm, frame)
}

/// SILK-only streams of every frame duration: (name, rate, channels,
/// bandwidth, bitrate, frame duration in units of 2.5 ms).
const SILK_STREAMS: [(&str, usize, usize, i32, i32, usize); 8] = [
    ("SILK-NB 8k 10ms", 8000, 1, BW_NB, 20000, 4),
    ("SILK-MB 12k 10ms", 12000, 1, BW_MB, 24000, 4),
    ("SILK-WB 16k 10ms", 16000, 1, BW_WB, 32000, 4),
    ("SILK-WB 48k 10ms", 48000, 1, BW_WB, 32000, 4),
    ("SILK-WB 16k stereo 10ms", 16000, 2, BW_WB, 48000, 4),
    ("SILK-WB 24k 20ms", 24000, 1, BW_WB, 24000, 8),
    ("SILK-NB 48k stereo 40ms", 48000, 2, BW_NB, 28000, 16),
    ("SILK-WB 16k 60ms", 16000, 1, BW_WB, 24000, 24),
];

/// A loss of any duration from 2.5 to 120 ms, in streams of 10, 20, 40 and
/// 60 ms frames: the concealed samples and everything decoded after them
/// are libopus's.
///
/// libopus conceals a loss in pieces: never more than one frame of the last
/// packet (`opus_decode_frame`: `frame_size = IMIN(frame_size,
/// st->frame_size)`), and never more than 20 ms. SILK attenuates more with
/// every piece, so 20 ms lost after 10 ms packets must be two pieces of
/// 10 ms. One piece of 20 ms gives other samples.
#[test]
fn a_loss_of_any_duration_is_concealed_like_libopus() {
    for (name, rate, channels, bandwidth, bitrate, frame_2_5ms) in SILK_STREAMS {
        let frame = rate / 400 * frame_2_5ms;
        let packets = c_packets(rate, channels, MODE_SILK, bandwidth, bitrate, frame, false);
        assert!(packets.len() >= 30, "{name}");
        for lost_2_5ms in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 48] {
            let lost = rate / 400 * lost_2_5ms;
            let mut c_dec = CDecoder::new(rate, channels);
            let mut dec = OpusDecoder::new(rate as i32, channels).unwrap();
            let mut buf = vec![0f32; frame.max(lost) * channels];
            let mut worst = 200f64;
            let mut losses = 0;
            for (i, p) in packets.iter().enumerate().take(30) {
                if i % 6 == 4 {
                    let reference = c_dec.decode(None, lost, false).unwrap();
                    assert_eq!(dec.decode(&[], lost, &mut buf), Ok(lost), "{name}");
                    worst = worst.min(snr(&reference, &buf[..lost * channels]));
                    assert_eq!((c_dec.final_range(), dec.final_range()), (0, 0), "{name}");
                    losses += 1;
                }
                let reference = c_dec.decode(Some(p), frame, false).unwrap();
                assert_eq!(dec.decode(p, frame, &mut buf), Ok(frame), "{name}");
                worst = worst.min(snr(&reference, &buf[..frame * channels]));
                assert_eq!(c_dec.final_range(), dec.final_range(), "{name}: packet {i}");
            }
            assert!(losses >= 4);
            assert!(worst > 100.0, "{name}, {} ms lost: worst call {worst:.1} dB against libopus", lost_2_5ms as f32 * 2.5);
        }
    }
}

/// Number of samples in which two 16-bit signals differ. The lengths must
/// be equal.
fn differing(reference: &[i16], ours: &[i16]) -> usize {
    assert_eq!(reference.len(), ours.len());
    reference.iter().zip(ours).filter(|(a, b)| a != b).count()
}

/// The same with 16-bit output, for losses longer than the 120 ms that
/// `decode_i16` conceals in one step: every sample is libopus's.
#[test]
fn a_long_loss_with_16_bit_output_is_concealed_like_libopus() {
    for (name, rate, channels, bandwidth, bitrate, frame_2_5ms) in SILK_STREAMS {
        let frame = rate / 400 * frame_2_5ms;
        let packets = c_packets(rate, channels, MODE_SILK, bandwidth, bitrate, frame, true);
        let mut c_dec = CDecoder::new(rate, channels);
        let mut dec = OpusDecoder::new(rate as i32, channels).unwrap();
        let long = rate / 400 * 136; // 340 ms
        let mut buf = vec![0i16; long.max(frame) * channels];
        for (i, p) in packets.iter().enumerate().take(24) {
            if i == 9 {
                // Lost, concealed.
                let reference = c_dec.decode_i16(None, long, false).unwrap();
                assert_eq!(dec.decode_i16(&[], long, &mut buf), Ok(long), "{name}");
                assert_eq!(differing(&reference, &buf[..long * channels]), 0, "{name}: long loss");
            }
            if i == 17 && channels == 1 && frame_2_5ms <= 8 {
                // Lost, the end recovered from this packet's FEC data. Mono
                // 10 and 20 ms only: see `tests/core_fec.rs` for the rest.
                let reference = c_dec.decode_i16(Some(p), long, true).unwrap();
                assert_eq!(dec.decode_fec_i16(p, long, &mut buf), Ok(long), "{name}");
                assert_eq!(differing(&reference, &buf[..long * channels]), 0, "{name}: long loss with FEC");
            }
            let reference = c_dec.decode_i16(Some(p), frame, false).unwrap();
            assert_eq!(dec.decode_i16(p, frame, &mut buf), Ok(frame), "{name}");
            assert_eq!(differing(&reference, &buf[..frame * channels]), 0, "{name}: packet {i}");
        }
    }
}

/// `decode_fec_i16` with a packet that cannot be read is an error that
/// leaves the decoder as it was, also for a loss longer than 120 ms: libopus
/// reads the packet before it conceals anything.
#[test]
fn a_long_loss_with_a_broken_fec_packet_changes_nothing() {
    let (rate, frame) = (16000usize, 320usize);
    let packets = c_packets(rate, 1, MODE_SILK, BW_WB, 28000, frame, true);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec = OpusDecoder::new(rate as i32, 1).unwrap();
    let long = 10 * frame;
    let mut buf = vec![0i16; long];
    for (i, p) in packets.iter().enumerate().take(30) {
        if i % 5 == 3 {
            // Frame count code 3 without its frame count byte; a code 2
            // packet whose first size runs past the end.
            for broken in [&[0x4B][..], &[0x4A, 200, 1, 2]] {
                assert!(c_dec.decode_i16(Some(broken), long, true).is_err());
                assert!(dec.decode_fec_i16(broken, long, &mut buf).is_err());
            }
        }
        let reference = c_dec.decode_i16(Some(p), frame, false).unwrap();
        assert_eq!(dec.decode_i16(p, frame, &mut buf), Ok(frame));
        assert_eq!(differing(&reference, &buf[..frame]), 0, "packet {i}");
        assert_eq!(c_dec.final_range(), dec.final_range(), "packet {i}");
    }
}

/// FEC decode as the first call into a new decoder, and as the first call
/// after a reset: samples and final range are libopus's, for one, two and
/// three lost frames, and for the packets decoded afterwards.
#[test]
fn fec_decode_as_the_first_call_matches_libopus() {
    for (name, rate, bandwidth, bitrate, frame_2_5ms) in [
        ("SILK-NB 8k 10ms", 8000usize, BW_NB, 20000, 4usize),
        ("SILK-WB 16k 20ms", 16000, BW_WB, 28000, 8),
        ("SILK-MB 48k 20ms", 48000, BW_MB, 24000, 8),
        ("SILK-WB 24k 10ms", 24000, BW_WB, 36000, 4),
    ] {
        let frame = rate / 400 * frame_2_5ms;
        let packets = c_packets(rate, 1, MODE_SILK, bandwidth, bitrate, frame, true);
        for lost_frames in [1usize, 2, 3] {
            let size = frame * lost_frames;
            let mut c_dec = CDecoder::new(rate, 1);
            let mut dec = OpusDecoder::new(rate as i32, 1).unwrap();
            let mut buf = vec![0f32; size];
            for round in 0..3 {
                let first = 10 + 7 * round;
                let what = format!("{name}, {lost_frames} frames lost, round {round}");
                let reference = c_dec.decode(Some(&packets[first]), size, true).unwrap();
                assert_eq!(dec.decode_fec(&packets[first], size, &mut buf), Ok(size), "{what}");
                let s = snr(&reference, &buf);
                assert!(s > 100.0, "{what}: FEC call {s:.1} dB against libopus");
                assert_eq!(c_dec.final_range(), dec.final_range(), "{what}: FEC call");
                // The frame before the copy is silence: there was nothing
                // to conceal from.
                assert!(buf[..size - frame].iter().all(|&v| v == 0.0), "{what}");
                for (i, p) in packets[first..first + 6].iter().enumerate() {
                    let reference = c_dec.decode(Some(p), frame, false).unwrap();
                    assert_eq!(dec.decode(p, frame, &mut buf), Ok(frame), "{what}");
                    let s = snr(&reference, &buf[..frame]);
                    assert!(s > 100.0, "{what}: packet {i} after it {s:.1} dB against libopus");
                    assert_eq!(c_dec.final_range(), dec.final_range(), "{what}: packet {i}");
                }
                c_dec.reset();
                dec.reset_state();
            }
        }
    }
}

/// One decoder through streams of every mode, with a reset between them:
/// after each reset the output is what libopus gives after
/// `OPUS_RESET_STATE`. Each stream ends on a loss, so that the reset has
/// concealment state to clear as well.
#[test]
fn reset_state_between_modes_matches_libopus() {
    let rate = 48000;
    let frame = 960;
    let pcm = multitone(rate, rate * 3 / 2);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec = OpusDecoder::new(rate as i32, 1).unwrap();
    let mut buf = vec![0f32; frame];
    let mut buf16 = vec![0i16; frame];
    for (name, mode, bandwidth, bitrate) in [
        ("CELT-FB", MODE_CELT, BW_FB, 64000),
        ("SILK-WB", MODE_SILK, BW_WB, 24000),
        ("Hybrid-FB", MODE_HYBRID, BW_FB, 40000),
        ("SILK-NB", MODE_SILK, BW_NB, 16000),
        ("CELT-NB", MODE_CELT, BW_NB, 32000),
        ("Hybrid-SWB", MODE_HYBRID, BW_SWB, 32000),
    ] {
        let mut enc = CEncoder::new(rate, 1, APP_AUDIO);
        enc.force_mode(mode).bandwidth(bandwidth).bitrate(bitrate);
        let packets = enc.encode_all(&pcm, frame);
        for (i, p) in packets.iter().enumerate().take(20) {
            let reference = c_dec.decode(Some(p), frame, false).unwrap();
            assert_eq!(dec.decode(p, frame, &mut buf), Ok(frame), "{name}");
            let s = snr(&reference, &buf);
            assert!(s > 100.0, "{name}: packet {i} after the reset: {s:.1} dB against libopus");
            assert_eq!(c_dec.final_range(), dec.final_range(), "{name}: packet {i}");
        }
        // A packet through the 16-bit path leaves soft clipper memory, and
        // a loss leaves concealment state.
        let reference = c_dec.decode_i16(Some(&packets[20]), frame, false).unwrap();
        assert_eq!(dec.decode_i16(&packets[20], frame, &mut buf16), Ok(frame), "{name}");
        let worst = reference.iter().zip(&buf16).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        assert!(worst <= 1, "{name}: 16-bit output differs by {worst}");
        c_dec.decode(None, frame, false).unwrap();
        dec.decode(&[], frame, &mut buf).unwrap();
        c_dec.reset();
        dec.reset_state();
        assert_eq!((c_dec.final_range(), dec.final_range()), (0, 0), "{name}: after the reset");
    }
}

/// FEC decode right after CELT-only packets conceals, whatever the next
/// packet carries (`opus_decode_native`: `st->mode == MODE_CELT_ONLY`). The
/// stream switches from CELT-only to SILK with FEC data, and the last
/// CELT-only packet's successor is lost.
///
/// The concealment of a CELT-only frame does not match libopus sample for
/// sample (tests/celtdec_plc.rs), so the samples are compared with a second
/// opus-rs decoder that is told of a plain loss. libopus supplies what it
/// can: the call succeeds there too and leaves a final range of 0.
#[test]
fn fec_decode_after_celt_only_packets_conceals() {
    for (rate, bandwidth) in [(16000usize, BW_WB), (48000, BW_WB), (48000, BW_FB)] {
        let frame = rate / 50;
        let pcm = speech(rate, 2.0);
        let mut enc = CEncoder::new(rate, 1, APP_VOIP);
        enc.bitrate(28000).fec(true, 25).bandwidth(bandwidth);
        let mut packets = Vec::new();
        for (i, input) in pcm.chunks_exact(frame).enumerate() {
            // 12 packets CELT-only, 12 packets SILK or Hybrid, and so on.
            enc.force_mode(if (i / 12) % 2 == 0 { MODE_CELT } else { MODE_SILK });
            packets.push(enc.encode(input));
        }
        let mut c_dec = CDecoder::new(rate, 1);
        let mut with_fec_call = OpusDecoder::new(rate as i32, 1).unwrap();
        let mut with_loss_call = OpusDecoder::new(rate as i32, 1).unwrap();
        let (mut a, mut b) = (vec![0f32; frame], vec![0f32; frame]);
        let mut checked = 0;
        let mut i = 0;
        while i + 2 < packets.len() {
            let celt_only = |p: &[u8]| p[0] & 0x80 != 0;
            // Lose packet i + 1 when packet i is CELT-only and packet i + 2
            // is not: packet i + 2 can carry a copy of the lost one.
            let lose_next = celt_only(&packets[i]) && !celt_only(&packets[i + 2]);
            c_dec.decode(Some(&packets[i]), frame, false).unwrap();
            with_fec_call.decode(&packets[i], frame, &mut a).unwrap();
            with_loss_call.decode(&packets[i], frame, &mut b).unwrap();
            assert_eq!(a, b, "{rate} Hz: packet {i}");
            if lose_next {
                let what = format!("{rate} Hz: FEC call with packet {} after CELT-only packet {i}", i + 2);
                c_dec.decode(Some(&packets[i + 2]), frame, true).unwrap();
                assert_eq!(with_fec_call.decode_fec(&packets[i + 2], frame, &mut a), Ok(frame), "{what}");
                with_loss_call.decode(&[], frame, &mut b).unwrap();
                assert_eq!(a, b, "{what}: not what a plain loss gives");
                assert_eq!((c_dec.final_range(), with_fec_call.final_range()), (0, 0), "{what}");
                checked += 1;
                i += 1;
            }
            i += 1;
        }
        assert!(checked >= 3, "{rate} Hz: {checked} switches from CELT-only checked");
    }
}

/// A low-pass filter: windowed sinc, `taps` coefficients, cutoff as a
/// fraction of the sampling rate.
fn low_pass(x: &[f32], cutoff: f64, taps: usize) -> Vec<f32> {
    let mid = (taps / 2) as isize;
    let h: Vec<f64> = (0..taps as isize)
        .map(|n| {
            let k = (n - mid) as f64;
            let sinc = if k == 0.0 { 2.0 * cutoff } else { (2.0 * std::f64::consts::PI * cutoff * k).sin() / (std::f64::consts::PI * k) };
            // Blackman window.
            let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (taps - 1) as f64).cos()
                + 0.08 * (4.0 * std::f64::consts::PI * n as f64 / (taps - 1) as f64).cos();
            sinc * w
        })
        .collect();
    (0..x.len() as isize)
        .map(|i| {
            let mut acc = 0f64;
            for (n, c) in h.iter().enumerate() {
                let j = i + mid - n as isize;
                if j >= 0 && (j as usize) < x.len() {
                    acc += c * x[j as usize] as f64;
                }
            }
            acc as f32
        })
        .collect()
}

/// Hybrid streams: the FEC copy is the SILK layer, below 8 kHz. Above it
/// the frame is concealed by CELT with noise. The recovered frames are
/// libopus's below 6 kHz, where only the copy counts, and over the whole
/// band.
#[test]
fn hybrid_fec_matches_libopus_below_6_khz() {
    for (name, rate, bandwidth, bitrate, frame_2_5ms) in [
        ("Hybrid-SWB 24k 20ms", 24000usize, BW_SWB, 40000, 8usize),
        ("Hybrid-FB 48k 20ms", 48000, BW_FB, 48000, 8),
        ("Hybrid-FB 48k 10ms", 48000, BW_FB, 56000, 4),
    ] {
        let frame = rate / 400 * frame_2_5ms;
        // Speech with content up to the Nyquist frequency, so that the
        // CELT layer has something to code.
        let mut enc = CEncoder::new(rate, 1, APP_VOIP);
        enc.force_mode(MODE_HYBRID).bandwidth(bandwidth).bitrate(bitrate).fec(true, 25);
        let packets = enc.encode_all(&speech_fullband(rate, 2.0), frame);
        assert!(packets.iter().all(|p| p[0] & 0xE0 == 0x60), "{name}: Hybrid packets");
        let mut c_dec = CDecoder::new(rate, 1);
        let mut dec = OpusDecoder::new(rate as i32, 1).unwrap();
        let mut buf = vec![0f32; frame];
        let (mut c_out, mut rs_out) = (Vec::new(), Vec::new());
        let mut recovered = Vec::new();
        for i in 0..packets.len() - 1 {
            if i % 8 == 5 {
                // Packet i is lost and rebuilt from packet i + 1.
                recovered.push(c_out.len()..c_out.len() + frame);
                c_out.extend(c_dec.decode(Some(&packets[i + 1]), frame, true).unwrap());
                assert_eq!(dec.decode_fec(&packets[i + 1], frame, &mut buf), Ok(frame), "{name}");
            } else {
                c_out.extend(c_dec.decode(Some(&packets[i]), frame, false).unwrap());
                assert_eq!(dec.decode(&packets[i], frame, &mut buf), Ok(frame), "{name}");
            }
            rs_out.extend_from_slice(&buf);
        }
        assert!(recovered.len() >= 8, "{name}");
        let cutoff = 6000.0 / rate as f64;
        let (c_low, rs_low) = (low_pass(&c_out, cutoff, 255), low_pass(&rs_out, cutoff, 255));
        let (mut r, mut t) = (Vec::new(), Vec::new());
        for range in &recovered {
            r.extend_from_slice(&c_low[range.clone()]);
            t.extend_from_slice(&rs_low[range.clone()]);
        }
        let below = snr(&r, &t);
        let (mut r, mut t) = (Vec::new(), Vec::new());
        for range in &recovered {
            r.extend_from_slice(&c_out[range.clone()]);
            t.extend_from_slice(&rs_out[range.clone()]);
        }
        let whole = snr(&r, &t);
        println!("{name}: recovered frames against libopus: {below:.1} dB below 6 kHz, {whole:.1} dB over the whole band");
        assert!(below > 100.0, "{name}: recovered frames below 6 kHz: {below:.1} dB against libopus");
        assert!(whole > 100.0, "{name}: recovered frames over the whole band: {whole:.1} dB against libopus");
    }
}
