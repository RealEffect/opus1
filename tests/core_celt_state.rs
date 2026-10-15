//! The state and the settings of the CELT encoder inside `OpusEncoder`
//! (opus_encoder.c): what it is told before every frame, and how it starts
//! again when the mode changes.
//!
//! The CELT layer of opus-rs does not code the same bytes as libopus (its
//! band shapes differ), but the first flags of a CELT frame are decided
//! the same way. These tests read them from the packets of both encoders:
//! the silence flag, the pitch post-filter flag, the transient flag and the
//! intra flag of the coarse energy (celt_decoder.c, "Decode the global
//! flags").

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech, to_stereo};
use opus_rs::range_coder::RangeCoder;
use opus_rs::{Application, ForcedMode, OpusEncoder};

/// The first flags of a CELT-only frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Flags {
    silence: bool,
    postfilter: bool,
    transient: bool,
    intra: bool,
}

/// Read the flags of a CELT-only packet with one frame, as
/// `celt_decode_with_ec` does (celt_decoder.c).
fn flags(packet: &[u8], frame_ms_x10: usize) -> Flags {
    assert!(packet[0] & 0x80 != 0, "not a CELT-only packet");
    assert_eq!(packet[0] & 3, 0, "not a packet with one frame");
    let frame = &packet[1..];
    let lm = match frame_ms_x10 {
        25 => 0,
        50 => 1,
        100 => 2,
        _ => 3,
    };
    let total = 8 * frame.len() as i32;
    let mut rc = RangeCoder::new_decoder(frame);
    let silence = if total <= 1 {
        true
    } else if rc.tell() == 1 {
        rc.decode_bit_logp(15)
    } else {
        false
    };
    let mut out = Flags { silence, postfilter: false, transient: false, intra: false };
    if silence {
        return out;
    }
    if rc.tell() + 16 <= total {
        out.postfilter = rc.decode_bit_logp(1);
        if out.postfilter {
            let octave = rc.dec_uint(6);
            rc.dec_bits(4 + octave);
            rc.dec_bits(3);
            if rc.tell() + 2 <= total {
                rc.decode_icdf(&[2, 1, 0], 2);
            }
        }
    }
    out.transient = lm > 0 && rc.tell() + 3 <= total && rc.decode_bit_logp(3);
    out.intra = rc.tell() + 3 <= total && rc.decode_bit_logp(3);
    out
}

/// Speech, mono or stereo.
fn signal(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let mono = speech(rate, seconds);
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

/// A CELT-only stream from both encoders at complexity 10, with `setup`
/// applied to each. Returns the flags of every packet, opus-rs and libopus.
fn celt_flags(
    rate: usize,
    channels: usize,
    frame_ms_x10: usize,
    setup: &dyn Fn(&mut OpusEncoder, &mut CEncoder),
) -> Vec<(Flags, Flags)> {
    let frame = rate * frame_ms_x10 / 10000;
    let pcm = signal(rate, channels, 2.0);
    let mut c_enc = CEncoder::new(rate, channels, APP_AUDIO);
    c_enc.complexity(10).bitrate(48000 * channels as i32).force_mode(MODE_CELT);
    let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
    enc.complexity = 10;
    enc.bitrate_bps = 48000 * channels as i32;
    enc.set_force_mode(Some(ForcedMode::CeltOnly));
    setup(&mut enc, &mut c_enc);
    let mut buf = vec![0u8; 1500];
    pcm.chunks_exact(frame * channels)
        .map(|input| {
            let c_packet = c_enc.encode(input);
            let n = enc.encode(input, frame, &mut buf).unwrap();
            (flags(&buf[..n], frame_ms_x10), flags(&c_packet, frame_ms_x10))
        })
        .collect()
}

/// `OPUS_SET_PREDICTION_DISABLED`: libopus sets `CELT_SET_PREDICTION(0)`
/// before every CELT frame (opus_encoder.c:2288-2294). Every frame is then
/// coded without the pitch post-filter and with intra coarse energy, so
/// that a decoder can start with any packet. Before opus-rs passed the
/// setting on, its frames used the post-filter and inter energy like
/// those of a stream with prediction.
#[test]
fn prediction_disabled_codes_every_celt_frame_alone_like_libopus() {
    for (rate, channels, frame_ms_x10) in [(48000usize, 1usize, 200usize), (48000, 2, 100), (16000, 1, 50)] {
        let disable = |enc: &mut OpusEncoder, c_enc: &mut CEncoder| {
            enc.set_prediction_disabled(true);
            c_enc.set(audiopus_sys::OPUS_SET_PREDICTION_DISABLED_REQUEST, 1);
        };
        let what = format!("{rate} Hz, {channels} channels, {} ms", frame_ms_x10 as f64 / 10.0);
        for (i, (got, want)) in celt_flags(rate, channels, frame_ms_x10, &disable).into_iter().enumerate() {
            assert!(!want.postfilter && (want.intra || want.silence), "{what}: libopus, packet {i}: {want:?}");
            assert_eq!(got, want, "{what}: packet {i}");
        }
    }
}

/// `OPUS_SET_PACKET_LOSS_PERC` goes to the CELT encoder too
/// (opus_encoder.c:2979-2989). Above 8 % loss the CELT encoder turns the
/// pitch pre-filter off (celt_encoder.c `run_prefilter`). Without the
/// setting, opus-rs used the post-filter in these frames as it does
/// without loss.
#[test]
fn packet_loss_turns_the_celt_pitch_filter_off_like_libopus() {
    for (rate, channels) in [(48000usize, 1usize), (48000, 2), (24000, 1)] {
        let what = format!("{rate} Hz, {channels} channels");
        // Without loss the speech is pitched enough for the filter.
        let lossless = celt_flags(rate, channels, 200, &|_, _| {});
        let with_filter = lossless.iter().filter(|(got, _)| got.postfilter).count();
        assert!(with_filter * 5 >= lossless.len(), "{what}: post-filter in {with_filter} frames");
        let lossy = |enc: &mut OpusEncoder, c_enc: &mut CEncoder| {
            enc.packet_loss_perc = 10;
            c_enc.set(audiopus_sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, 10);
        };
        for (i, (got, want)) in celt_flags(rate, channels, 200, &lossy).into_iter().enumerate() {
            assert!(!want.postfilter, "{what}: libopus, packet {i}");
            assert_eq!(got.postfilter, want.postfilter, "{what}: packet {i}");
        }
    }
}

/// SILK-only 20 ms packets, then 5 ms packets, which only CELT codes. The
/// switch has no redundant frame: a frame under 10 ms cannot stay in the
/// old mode (opus_encoder.c:1541-1558). The CELT encoder starts again,
/// codes the 2.5 ms before the frame into two bytes it throws away, and
/// codes the first frame without prediction (C 2478-2486): no post-filter,
/// intra energy. The frames after it predict again.
#[test]
fn celt_after_silk_starts_afresh_without_prediction_like_libopus() {
    for (rate, channels) in [(48000usize, 1usize), (16000, 1), (48000, 2)] {
        let what = format!("{rate} Hz, {channels} channels");
        let pcm = signal(rate, channels, 2.0);
        let mut c_enc = CEncoder::new(rate, channels, APP_AUDIO);
        c_enc.complexity(10).bitrate(64000);
        let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
        enc.complexity = 10;
        enc.bitrate_bps = 64000;
        let mut buf = vec![0u8; 1500];
        let (long, short) = (rate / 50, rate / 200);
        let mut at = 0;
        let mut celt = Vec::new();
        for step in 0..4 {
            // 300 ms of SILK-only, then 100 ms of 5 ms CELT-only frames.
            c_enc.force_mode(MODE_SILK).bandwidth(BW_WB);
            enc.set_force_mode(Some(ForcedMode::SilkOnly));
            enc.set_bandwidth(opus_rs::Bandwidth::Wideband);
            for _ in 0..15 {
                let input = &pcm[at * channels..(at + long) * channels];
                at += long;
                let c_packet = c_enc.encode(input);
                enc.encode(input, long, &mut buf).unwrap();
                assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "{what}: step {step}: first byte");
            }
            c_enc.force_mode(MODE_CELT);
            enc.set_force_mode(Some(ForcedMode::CeltOnly));
            for i in 0..20 {
                let input = &pcm[at * channels..(at + short) * channels];
                at += short;
                let c_packet = c_enc.encode(input);
                let n = enc.encode(input, short, &mut buf).unwrap();
                celt.push((step, i, flags(&buf[..n], 50), flags(&c_packet, 50)));
            }
        }
        for (step, i, got, want) in celt {
            if i == 0 {
                assert!(!want.postfilter && want.intra, "{what}: libopus, step {step}: {want:?}");
                assert_eq!(got, want, "{what}: step {step}, first CELT frame");
            } else {
                assert_eq!(got.intra, want.intra, "{what}: step {step}, CELT frame {i}: intra");
            }
        }
    }
}
