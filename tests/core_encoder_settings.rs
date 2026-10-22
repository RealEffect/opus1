//! The settings of `OpusEncoder` against libopus's `opus_encoder_ctl`:
//! which values each setter refuses, what the getters return, and the
//! packets of the settings that this interface added to the single-stream
//! encoder: the automatic bitrate, which is the default, the maximum
//! bitrate, and unconstrained variable bitrate (VBR).
//!
//! The encoders run at complexity 5 where packets are compared. Both then
//! make their decisions without the speech/music analysis, which they run
//! from complexity 7 on (`core_analysis_libopus.rs` compares that).

mod core_common;
mod k24_common;

use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech, to_stereo};
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

/// libopus's `OPUS_AUTO` and `OPUS_BITRATE_MAX`.
const OPUS_AUTO: i32 = -1000;
const OPUS_BITRATE_MAX: i32 = -1;

const RATES: [usize; 5] = [8000, 12000, 16000, 24000, 48000];

fn app(c_app: i32) -> Application {
    match c_app {
        APP_VOIP => Application::Voip,
        APP_AUDIO => Application::Audio,
        _ => Application::RestrictedLowDelay,
    }
}

fn signal(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let voice = speech(rate, seconds);
    let tones = multitone(rate, voice.len());
    let mono: Vec<f32> = voice.iter().zip(&tones).map(|(v, t)| 0.6 * v + 0.3 * t).collect();
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

/// Every getter of opus1 against the matching `OPUS_GET_*` request.
fn assert_same_settings(rust: &OpusEncoder, c: &mut CEncoder, label: &str) {
    assert_eq!(rust.bitrate(), c.get(sys::OPUS_GET_BITRATE_REQUEST), "{label}: bitrate");
    assert_eq!(rust.complexity(), c.get(sys::OPUS_GET_COMPLEXITY_REQUEST), "{label}: complexity");
    assert_eq!(rust.vbr() as i32, c.get(sys::OPUS_GET_VBR_REQUEST), "{label}: VBR");
    assert_eq!(
        rust.vbr_constraint() as i32,
        c.get(sys::OPUS_GET_VBR_CONSTRAINT_REQUEST),
        "{label}: VBR constraint"
    );
    assert_eq!(rust.inband_fec() as i32, c.get(sys::OPUS_GET_INBAND_FEC_REQUEST), "{label}: FEC");
    assert_eq!(
        rust.packet_loss_perc(),
        c.get(sys::OPUS_GET_PACKET_LOSS_PERC_REQUEST),
        "{label}: packet loss"
    );
}

/// A value libopus refuses is refused, and the setting keeps its value; a
/// value libopus takes is taken, and both read back the same.
#[test]
fn setters_refuse_what_libopus_refuses() {
    let bitrates = [
        i32::MIN, -1001, OPUS_AUTO, -999, -2, OPUS_BITRATE_MAX, 0, 1, 499, 500, 501, 64_000,
        749_999, 750_000, 750_001, 1_499_999, 1_500_000, 1_500_001, 3_000_000, i32::MAX,
    ];
    let complexities = [i32::MIN, -1, 0, 1, 5, 9, 10, 11, i32::MAX];
    let losses = [i32::MIN, -1, 0, 1, 50, 100, 101, i32::MAX];
    for rate in RATES {
        for channels in [1, 2] {
            let label = format!("{rate} Hz, {channels} channels");
            let mut rust = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
                Application::Audio,
            )
            .unwrap();
            let mut c = CEncoder::new(rate, channels, APP_AUDIO);
            assert_same_settings(&rust, &mut c, &format!("{label}, new"));
            for v in bitrates {
                let c_ok = c.try_set(sys::OPUS_SET_BITRATE_REQUEST, v) == 0;
                let setting = match v {
                    OPUS_AUTO => Bitrate::Auto,
                    OPUS_BITRATE_MAX => Bitrate::Max,
                    _ => Bitrate::BitsPerSecond(v),
                };
                assert_eq!(rust.set_bitrate(setting).is_ok(), c_ok, "{label}: bitrate {v}");
                assert_same_settings(&rust, &mut c, &format!("{label}, bitrate {v}"));
            }
            for v in complexities {
                let c_ok = c.try_set(sys::OPUS_SET_COMPLEXITY_REQUEST, v) == 0;
                assert_eq!(rust.set_complexity(v).is_ok(), c_ok, "{label}: complexity {v}");
                assert_same_settings(&rust, &mut c, &format!("{label}, complexity {v}"));
            }
            for v in losses {
                let c_ok = c.try_set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, v) == 0;
                assert_eq!(rust.set_packet_loss_perc(v).is_ok(), c_ok, "{label}: loss {v}");
                assert_same_settings(&rust, &mut c, &format!("{label}, loss {v}"));
            }
            for on in [false, true] {
                rust.set_vbr(on);
                c.vbr(on);
                rust.set_vbr_constraint(!on);
                c.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, !on as i32);
                rust.set_inband_fec(on);
                c.set(sys::OPUS_SET_INBAND_FEC_REQUEST, on as i32);
                assert_same_settings(&rust, &mut c, &format!("{label}, switches {on}"));
            }
            // A reset keeps the settings, in both.
            rust.reset_state();
            c.reset();
            assert_same_settings(&rust, &mut c, &format!("{label}, reset"));
        }
    }
}

/// `bitrate()` is libopus `OPUS_GET_BITRATE`: the automatic and the
/// maximum bitrate depend on the duration of the last frame that was coded,
/// and every setting counts with at most what 1276 bytes per frame carry.
#[test]
fn bitrate_readback_follows_the_last_frame() {
    for (c_app, rate, channels) in [(APP_AUDIO, 48000, 2), (APP_VOIP, 16000, 1), (APP_LOWDELAY, 24000, 1)] {
        let pcm = signal(rate, channels, 0.5);
        for setting in [Bitrate::Auto, Bitrate::Max, Bitrate::BitsPerSecond(600_000)] {
            let c_value = match setting {
                Bitrate::Auto => OPUS_AUTO,
                Bitrate::Max => OPUS_BITRATE_MAX,
                Bitrate::BitsPerSecond(bps) => bps,
            };
            let mut rust = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
                app(c_app),
            )
            .unwrap();
            rust.set_complexity(5).unwrap();
            rust.set_bitrate(setting).unwrap();
            let mut c = CEncoder::new(rate, channels, c_app);
            c.complexity(5).bitrate(c_value);
            let mut out = vec![0u8; 4000];
            // 2.5, 5, 10, 20, 40 and 60 ms.
            for tenths in [25, 50, 100, 200, 400, 600] {
                let frame = rate * tenths / 10_000;
                let label = format!("{rate} Hz, {channels} channels, {setting:?}, {tenths} tenths of ms");
                let input = &pcm[..frame * channels];
                rust.encode(input, frame, &mut out).unwrap();
                c.encode_max(input, 4000);
                assert_eq!(rust.bitrate(), c.get(sys::OPUS_GET_BITRATE_REQUEST), "{label}");
            }
            rust.reset_state();
            c.reset();
            assert_eq!(rust.bitrate(), c.get(sys::OPUS_GET_BITRATE_REQUEST), "{rate} Hz, {setting:?}, reset");
        }
    }
}

/// Encodes `pcm` with both encoders, frame by frame, into buffers of
/// `max_bytes`, and returns the packets, opus1's first.
fn encode_both(
    rust: &mut OpusEncoder,
    c: &mut CEncoder,
    pcm: &[f32],
    frame: usize,
    max_bytes: usize,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let channels = rust.channels().count();
    let mut out = vec![0u8; max_bytes];
    let mut rust_packets = Vec::new();
    let mut c_packets = Vec::new();
    for chunk in pcm.chunks_exact(frame * channels) {
        let len = rust.encode(chunk, frame, &mut out).unwrap();
        rust_packets.push(out[..len].to_vec());
        c_packets.push(c.encode_max(chunk, max_bytes));
    }
    (rust_packets, c_packets)
}

/// The first byte and the length of every packet, for comparison.
fn tocs_and_sizes(packets: &[Vec<u8>]) -> Vec<(u8, usize)> {
    packets.iter().map(|p| (p[0], p.len())).collect()
}

/// `Bitrate::Max` writes libopus's packets: with a constant bitrate every
/// packet has libopus's size. That is 1.5 Mb/s or the buffer, whichever is
/// less; a 40 or 60 ms packet of 20 ms CELT frames is padded to the whole
/// buffer.
///
/// Only a buffer of more than 7,500 bytes shows the padding of the whole
/// buffer (opus_encoder.c:1749). With a smaller one the bitrate of the
/// buffer is below 1.5 Mb/s, so the constant size is the buffer anyway. At
/// 40 ms, 1.5 Mb/s is 7,500 bytes; at 60 ms the encoder counts with at most
/// 7,656 bytes (6 × 1276).
#[test]
fn maximum_bitrate_gives_the_packets_of_libopus() {
    for (rate, channels) in [(48000, 1), (48000, 2), (24000, 1)] {
        let pcm = signal(rate, channels, 0.6);
        for tenths in [100, 200, 400, 600] {
            let frame = rate * tenths / 10_000;
            for max_bytes in [60, 200, 1276, 3000, 10_000] {
                for vbr in [false, true] {
                    let label = format!("{rate} Hz, {channels} ch, {tenths} tenths of ms, {max_bytes} bytes, VBR {vbr}");
                    let mut rust = OpusEncoder::new(
                        SampleRate::try_from(rate as i32).unwrap(),
                        Channels::try_from(channels).unwrap(),
                        Application::Audio,
                    )
                    .unwrap();
                    rust.set_complexity(5).unwrap();
                    rust.set_bitrate(Bitrate::Max).unwrap();
                    rust.set_vbr(vbr);
                    rust.set_force_mode(Some(ForcedMode::CeltOnly));
                    let mut c = CEncoder::new(rate, channels, APP_AUDIO);
                    c.complexity(5).bitrate(OPUS_BITRATE_MAX).vbr(vbr).force_mode(MODE_CELT);
                    let (r, c) = encode_both(&mut rust, &mut c, &pcm, frame, max_bytes);
                    if vbr {
                        // VBR sizes follow the signal, which both CELT
                        // encoders analyse alike but not always to the bit.
                        let toc = |p: &[Vec<u8>]| p.iter().map(|p| p[0]).collect::<Vec<_>>();
                        assert_eq!(toc(&r), toc(&c), "{label}: TOC bytes");
                    } else {
                        assert_eq!(tocs_and_sizes(&r), tocs_and_sizes(&c), "{label}");
                    }
                }
            }
        }
    }
}

/// Unconstrained VBR for CELT-only packets: opus1's packets have libopus's
/// first byte, and their sizes follow libopus's. Turning the constraint off
/// changes the sizes in both.
#[test]
fn unconstrained_vbr_follows_libopus() {
    for (rate, channels, bitrate) in [(48000, 1, 48_000), (48000, 2, 96_000), (24000, 1, 32_000)] {
        let pcm = signal(rate, channels, 2.0);
        let frame = rate / 50;
        let mut totals = Vec::new();
        for constrained in [true, false] {
            let label = format!("{rate} Hz, {channels} ch, {bitrate} b/s, constrained {constrained}");
            let mut rust = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
                Application::RestrictedLowDelay,
            )
            .unwrap();
            rust.set_complexity(5).unwrap();
            rust.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            rust.set_vbr_constraint(constrained);
            let mut c = CEncoder::new(rate, channels, APP_LOWDELAY);
            c.complexity(5).bitrate(bitrate);
            c.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, constrained as i32);
            let (r, c) = encode_both(&mut rust, &mut c, &pcm, frame, 1276);
            let toc = |p: &[Vec<u8>]| p.iter().map(|p| p[0]).collect::<Vec<_>>();
            assert_eq!(toc(&r), toc(&c), "{label}: TOC bytes");
            let same = r.iter().zip(&c).filter(|(a, b)| a.len() == b.len()).count();
            let (r_total, c_total): (usize, usize) =
                (r.iter().map(Vec::len).sum(), c.iter().map(Vec::len).sum());
            println!("{label}: {same} of {} sizes equal, {r_total} bytes against {c_total}", r.len());
            assert!(same * 10 >= r.len() * 9, "{label}: {same} of {} sizes equal", r.len());
            assert!(r_total.abs_diff(c_total) * 100 <= c_total, "{label}: {r_total} bytes against {c_total}");
            totals.push((r_total, r.iter().map(Vec::len).max().unwrap()));
        }
        // Without the constraint the largest packet is larger.
        assert!(totals[1].1 > totals[0].1, "{rate} Hz, {channels} ch: {totals:?}");
    }
}

/// An encoder that is not given a bitrate uses libopus's automatic one
/// (`OPUS_AUTO`, 60 × rate / frame size + rate × channels b/s): its packets
/// have the configuration (mode, bandwidth, frame duration) and the stereo
/// flag of libopus's default encoder, and with a constant bitrate also its
/// sizes. The frame count code of the first byte is not compared: a SILK
/// frame that needs padding to reach the constant size is written with
/// code 3, and opus1's fixed-point SILK encoder writes frames of other
/// sizes than the float one of libopus.
#[test]
fn default_bitrate_is_that_of_libopus() {
    // Input rate, channels, application, frame duration in tenths of ms,
    // constant bitrate.
    let cases = [
        (8000, 1, APP_VOIP, 200, true),
        (16000, 1, APP_VOIP, 200, true),
        (16000, 2, APP_VOIP, 200, false),
        (24000, 1, APP_VOIP, 100, false),
        (24000, 1, APP_AUDIO, 200, true),
        (48000, 1, APP_AUDIO, 200, false),
        (48000, 2, APP_AUDIO, 200, true),
        (48000, 2, APP_AUDIO, 600, true),
        (48000, 1, APP_LOWDELAY, 25, true),
        (12000, 2, APP_AUDIO, 400, false),
    ];
    for (rate, channels, c_app, tenths, cbr) in cases {
        let frame = rate * tenths / 10_000;
        let label = format!("{rate} Hz, {channels} ch, application {c_app}, {tenths} tenths of ms, CBR {cbr}");
        let mut rust = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
            app(c_app),
        )
        .unwrap();
        assert_eq!(rust.bitrate_setting(), Bitrate::Auto, "{label}");
        rust.set_complexity(5).unwrap();
        rust.set_vbr(!cbr);
        let mut c = CEncoder::new(rate, channels, c_app);
        c.complexity(5).vbr(!cbr);
        let pcm = signal(rate, channels, 1.2);
        let (r, c_packets) = encode_both(&mut rust, &mut c, &pcm, frame, 1276 * 3);
        assert_eq!(rust.bitrate(), c.get(sys::OPUS_GET_BITRATE_REQUEST), "{label}: bitrate");
        // A 40 or 60 ms CELT packet is made of 20 ms frames, and the last
        // frame counts. Shorter packets are one frame.
        if tenths <= 200 {
            assert_eq!(
                rust.bitrate(),
                ((60 * rate / frame + rate * channels) as i32).min(1276 * 8 * (rate / frame) as i32),
                "{label}: the automatic bitrate"
            );
        }
        let config = |p: &[Vec<u8>]| p.iter().map(|p| p[0] & 0xFC).collect::<Vec<_>>();
        assert_eq!(config(&r), config(&c_packets), "{label}: configurations");
        if cbr {
            let sizes = |p: &[Vec<u8>]| p.iter().map(Vec::len).collect::<Vec<_>>();
            assert_eq!(sizes(&r), sizes(&c_packets), "{label}: sizes");
        }
    }
}
