//! Issue #54: CELT from 24 kHz input topped out at ~20 dB SNR at any bitrate
//! (20.1 / 20.9 / 21.1 dB at 32 / 64 / 128 kbps on the issue's chirp), while
//! 48 kHz input and libopus's own encoder at 24 kHz kept improving with bits.
//!
//! Root cause: `run_prefilter`'s cancel path. libopus 1.6 reverts the pitch
//! pre-filter for a frame when filtering makes it louder (`cancel_pitch`),
//! but it still fades the *previous* frame's comb filter out over the first
//! `overlap` samples (celt_encoder.c:1572-1584), because the decoder's
//! post-filter always cross-fades from the old gain to the new one. opus1
//! restored the unfiltered input over the whole frame instead, so after every
//! "on" frame that was cancelled, the decoder post-filtered 120 samples the
//! encoder never pre-filtered: an error burst no bitrate can remove.
//!
//! 48 kHz input rarely cancels: the comb filter attenuates the tone itself.
//! From 24 kHz, the zero-stuffed, pre-emphasized frame is dominated by its
//! image above 12 kHz, which the comb filter barely touches, so whether the
//! filter "helps" flips with the parity of the pitch period as the chirp
//! sweeps, and the cancel fired on about every other frame.
//!
//! The harness is the issue's: encode in 20 ms `Audio` frames, decode with
//! libopus (the `opus` dev-dependency) at 48 kHz, drop the 312-sample
//! pre-skip and take the SNR against the analytic chirp. On the unfixed tree
//! opus1 scored ~21 dB at 64 and 128 kbps against libopus's 37-40. libopus
//! 1.5.2 has no `cancel_pitch` at all, so it is a reference for quality, not
//! for bits: the assertions are relative to libopus's own encoder and to
//! opus1's lower bitrates, never absolute dB.
//!
//! Both encoders run at the default complexity 9, with libopus's speech and
//! music analysis. At 64 and 128 kbps both code the chirp as CELT
//! super-wideband, and opus1's packets are libopus's byte for byte. At
//! 32 kbps both code it as Hybrid, so the test of the rising SNR forces
//! opus1 to CELT-only.
#![cfg(feature = "analysis")]

use opus::{Application as CApp, Bitrate, Channels as CCh, Decoder as CDec, Encoder as CEnc};
use opus1::{Application, Channels, ForcedMode, OpusEncoder, SampleRate};

const PRE_SKIP: usize = 312;

/// The issue's chirp: 200 Hz rising at 1200 Hz/s, amplitude 0.4 × `level`.
fn chirp(rate: usize, n: usize, level: f64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            ((std::f64::consts::TAU * (200.0 * t + 600.0 * t * t)).sin() * 0.4 * level) as f32
        })
        .collect()
}

/// 1.3 s of chirp, zero-padded to whole 20 ms frames covering the pre-skip.
fn input(rate: usize, level: f64) -> Vec<f32> {
    let (frame, len) = (rate / 50, rate * 13 / 10);
    let mut x = chirp(rate, len, level);
    x.resize(
        (len + PRE_SKIP * rate / 48_000).div_ceil(frame) * frame,
        0.0,
    );
    x
}

/// opus1's packets, CELT-only when `force_celt`.
fn opus1_packets(rate: usize, bitrate: i32, level: f64, force_celt: bool) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::Mono,
        Application::Audio,
    )
    .unwrap();
    enc.set_bitrate(opus1::Bitrate::BitsPerSecond(bitrate)).unwrap();
    if force_celt {
        enc.set_force_mode(Some(ForcedMode::CeltOnly));
    }
    let frame = rate / 50;
    let mut pkt = [0u8; 4000];
    input(rate, level)
        .chunks(frame)
        .map(|f| {
            let n = enc.encode(f, frame, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

fn libopus_packets(rate: usize, bitrate: i32, level: f64) -> Vec<Vec<u8>> {
    let mut enc = CEnc::new(rate as u32, CCh::Mono, CApp::Audio).unwrap();
    enc.set_bitrate(Bitrate::Bits(bitrate)).unwrap();
    // Same look-ahead as opus1, so the same pre-skip applies to both.
    assert_eq!(
        enc.get_lookahead().unwrap() as usize * 48_000 / rate,
        PRE_SKIP
    );
    let mut pkt = [0u8; 4000];
    input(rate, level)
        .chunks(rate / 50)
        .map(|f| {
            let n = enc.encode_float(f, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

/// libopus's packets forced to CELT-only (`OPUS_SET_FORCE_MODE`, which the
/// `opus` crate does not offer).
fn libopus_celt_packets(rate: usize, bitrate: i32, level: f64) -> Vec<Vec<u8>> {
    use audiopus_sys as sys;
    let mut err = 0;
    let st = unsafe { sys::opus_encoder_create(rate as i32, 1, sys::OPUS_APPLICATION_AUDIO, &mut err) };
    assert_eq!(err, 0);
    unsafe {
        sys::opus_encoder_ctl(st, sys::OPUS_SET_BITRATE_REQUEST, bitrate);
        // OPUS_SET_FORCE_MODE (opus_private.h) to MODE_CELT_ONLY.
        sys::opus_encoder_ctl(st, 11002, 1002);
    }
    let packets = input(rate, level)
        .chunks(rate / 50)
        .map(|f| {
            let mut out = vec![0u8; 4000];
            let n = unsafe {
                sys::opus_encode_float(st, f.as_ptr(), (rate / 50) as i32, out.as_mut_ptr(), 4000)
            };
            assert!(n > 0);
            out.truncate(n as usize);
            out
        })
        .collect();
    unsafe { sys::opus_encoder_destroy(st) };
    packets
}

/// SNR (dB) of the libopus 48 kHz decode against the analytic chirp.
fn snr(rate: usize, packets: &[Vec<u8>], level: f64) -> f64 {
    let mut dec = CDec::new(48_000, CCh::Mono).unwrap();
    let (mut out, mut pcm) = (Vec::new(), vec![0f32; 5760]);
    for p in packets {
        let n = dec.decode_float(p, &mut pcm, false).unwrap();
        out.extend_from_slice(&pcm[..n]);
    }
    let len = rate * 13 / 10 * (48_000 / rate);
    let got = &out[PRE_SKIP..PRE_SKIP + len];
    let want = chirp(48_000, len, level);
    let (mut sig, mut err) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(&want) {
        sig += f64::from(w) * f64::from(w);
        err += f64::from(g - w) * f64::from(g - w);
    }
    10.0 * (sig / err).log10()
}

/// TOC configs that are not CELT super-wideband 20 ms (config 27).
fn non_celt_swb(packets: &[Vec<u8>]) -> usize {
    packets.iter().filter(|p| p[0] >> 3 != 27).count()
}

#[test]
fn celt_24k_tracks_libopus() {
    for level in [0.9, 1.0, 1.1] {
        for bitrate in [64_000, 128_000] {
            let ours = opus1_packets(24_000, bitrate, level, false);
            let theirs = libopus_packets(24_000, bitrate, level);
            if cfg!(target_arch = "aarch64") {
                assert!(ours == theirs, "{bitrate} b/s, level {level}: not libopus's packets");
            }
            assert_eq!(non_celt_swb(&ours), 0, "opus1 left CELT SWB at {bitrate}");
            assert_eq!(
                non_celt_swb(&theirs),
                0,
                "libopus left CELT SWB at {bitrate}"
            );
            let (rs, c) = (snr(24_000, &ours, level), snr(24_000, &theirs, level));
            assert!(
                rs >= c - 3.0,
                "24 kHz CELT at {} kbps (level {level}): opus1 {rs:.1} dB vs libopus {c:.1} dB",
                bitrate / 1000
            );
        }
    }
}

/// The SNR of CELT from 24 kHz input rises with the bitrate: by at least
/// 2 dB per doubling (the bug left it within 1 dB from 32 to 128 kbps).
/// Forced to CELT-only, opus1's packets are libopus's, so libopus's CELT
/// encoder rises by exactly as much. (Before the speech and music analysis
/// was ported, opus1 rose by 3 dB or more per doubling here. With the
/// analysis, opus1 and libopus rise by 2.7 dB from 64 to 128 kbps at level
/// 0.9: 39.7 to 42.4 dB.)
#[test]
fn celt_24k_snr_rises_with_bitrate() {
    for level in [0.9, 1.0, 1.1] {
        let mut prev = f64::NEG_INFINITY;
        for bitrate in [32_000, 64_000, 128_000] {
            let packets = opus1_packets(24_000, bitrate, level, true);
            assert_eq!(
                non_celt_swb(&packets),
                0,
                "opus1 left CELT SWB at {bitrate}"
            );
            if cfg!(target_arch = "aarch64") {
                assert!(
                    packets == libopus_celt_packets(24_000, bitrate, level),
                    "{bitrate} b/s, level {level}: not libopus's CELT packets"
                );
            }
            let db = snr(24_000, &packets, level);
            assert!(
                db >= prev + 2.0,
                "24 kHz SNR stopped rising at {} kbps (level {level}): {db:.1} dB after {prev:.1} dB",
                bitrate / 1000
            );
            prev = db;
        }
    }
}
