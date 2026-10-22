//! Issue #38: SILK/hybrid output at 48 kHz used to lag libopus's pre-skip by
//! up to 17 samples, and low-bitrate Audio mangled the first ~200 ms.
//!
//! Root causes:
//! 1. The encoder resampled the API-rate input with the stateless
//!    `silk_resampler_down2` + `down2_3` / `down_1_3` decimators and then
//!    compensated the delay inside `silk_encode` with a hard-coded
//!    internal-rate delay buffer — the wrong rate and the wrong values
//!    (10 samples at 16 kHz ≈ 30 at 48 kHz, versus libopus's
//!    `delay_matrix_enc[48][16] = 12` plus the down-FIR group delay).
//! 2. `silk_encode_prefill` was never invoked, so SILK restarts after CELT
//!    started from a cold encoder state.
//! 3. The low-bitrate Audio damage was in CELT, not SILK: opus1 codes
//!    Audio at 20-24 kbps CELT-only, and its CELT encoder ran `tone_detect`
//!    and `transient_analysis` on a buffer whose overlap head held the
//!    previous frame's *prefiltered* samples while the body was unfiltered.
//!    libopus 1.6.1 first overwrites the head with the unfiltered
//!    `prefilter_mem` tail (celt_encoder.c:2017). With a strong comb-filter
//!    gain on a low tone, the step from head to body read as a transient on
//!    nearly every frame, and short blocks starved the 200-400 Hz bands.
//!
//! The encoder now runs the libopus `silk_resampler` (encoder direction:
//! `delay_matrix_enc` + private down-FIR) on the API-rate input, prefills
//! the SILK encoder on CELT→SILK restarts (opus_encoder.c:1448-1455,
//! 1806-1826), and analyses CELT transients on unfiltered history. The
//! alignment below is measured through libopus's decoder (the `opus`
//! dev-dependency) after the 312-sample (6.5 ms) pre-skip, per 100 ms window,
//! exactly like the issue's harness. Reference values for libopus's own
//! encoder on the same signal: Voip 32 kbps lags -5..-2 samples at
//! corr ≥ 0.99 after a weaker first window, then 29 samples in the window
//! where its speech and music analysis moves the stream from Hybrid to
//! CELT-only (1.0 to 1.1 s), and 0 after it. opus1 makes the same switch at
//! the same packet, and the test compares the two profiles window by
//! window. Audio 20-24 kbps forced CELT-only is lag 0 (-1 in the first
//! window at 20 kbps), corr ≥ 0.99 from the first window, and opus1's
//! packets are libopus's byte for byte. (Without forcing, libopus and opus1 code the chirp as
//! Hybrid at Audio 20-32 kbps until the same switch, so the CELT tests
//! below force CELT-only.)
// Some helpers serve only the tests that need the speech and music analysis.
#![cfg_attr(not(feature = "analysis"), allow(dead_code))]

use opus::{
    Application as CApp, Bitrate as CBitrate, Channels as CCh, Decoder as CDec, Encoder as CEnc,
};
use opus1::internals::range_coder::RangeCoder;
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

const SR: usize = 48000;
const FRAME: usize = 960;

/// 1.3 s logarithmic-ish chirp, RMS ≈ 0.28 like the issue's repro.
fn chirp(sr: usize, seconds: usize) -> Vec<f32> {
    let n = sr * seconds / 10;
    let mut input = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr as f64;
        let phase = 2.0 * std::f64::consts::PI * (100.0 * t + 1950.0 / 1.3 * t * t);
        input.push((0.28 * 2.0f64.sqrt() * phase.sin()) as f32);
    }
    input
}

/// Encode the chirp with opus1 in 20 ms frames; returns input and packets.
/// `Audio` is forced to CELT-only, the encoder the Audio tests look at.
fn encode_chirp(app: Application, bitrate: i32) -> (Vec<f32>, Vec<Vec<u8>>) {
    let input = chirp(SR, 13);
    let mut enc =
        OpusEncoder::new(SampleRate::try_from(SR as i32).unwrap(), Channels::Mono, app).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
    if app == Application::Audio {
        enc.set_force_mode(Some(ForcedMode::CeltOnly));
    }
    let mut pkt = vec![0u8; 1500];
    let packets = input
        .as_chunks::<FRAME>().0.iter()
        .map(|frame| {
            let len = enc.encode(frame, FRAME, &mut pkt).unwrap();
            pkt[..len].to_vec()
        })
        .collect();
    (input, packets)
}

/// Encode the whole stream with opus1, decode with libopus, and return the
/// per-window (best lag, correlation) after the 312-sample pre-skip.
fn lag_profile(app: Application, bitrate: i32) -> Vec<(i64, f64)> {
    let (input, packets) = encode_chirp(app, bitrate);
    profile_of(&input, &packets)
}

/// The same profile for libopus's own encoder, Voip at `bitrate`.
fn libopus_voip_profile(bitrate: i32) -> Vec<(i64, f64)> {
    let input = chirp(SR, 13);
    let mut enc = CEnc::new(SR as u32, CCh::Mono, CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
    let packets: Vec<Vec<u8>> = input
        .as_chunks::<FRAME>().0.iter()
        .map(|frame| enc.encode_vec_float(frame, 1500).unwrap())
        .collect();
    profile_of(&input, &packets)
}

/// Decode `packets` with libopus and return the per-window (best lag,
/// correlation) against `input` after the 312-sample pre-skip.
fn profile_of(input: &[f32], packets: &[Vec<u8>]) -> Vec<(i64, f64)> {
    let mut dec = CDec::new(SR as u32, CCh::Mono).unwrap();
    let mut pcm = Vec::new();
    for pkt in packets {
        let mut buf = vec![0f32; FRAME * 2];
        let got = dec.decode_float(pkt, &mut buf, false).unwrap_or(0);
        pcm.extend_from_slice(&buf[..got]);
    }

    let pre_skip = 312usize;
    let win = SR / 10;
    let mut profile = Vec::new();
    let mut start = 0usize;
    while start + win < input.len() && start + pre_skip + win < pcm.len() {
        let mut best = (0i64, f64::MIN);
        for lag in -40i64..=40 {
            let mut xy = 0f64;
            let mut xx = 0f64;
            let mut yy = 0f64;
            for (k, &x) in input[start..start + win].iter().enumerate() {
                let x = x as f64;
                let j = (start + pre_skip + k) as i64 + lag;
                if j < 0 || j as usize >= pcm.len() {
                    continue;
                }
                let y = pcm[j as usize] as f64;
                xy += x * y;
                xx += x * x;
                yy += y * y;
            }
            let corr = xy / (xx.sqrt() * yy.sqrt() + 1e-12);
            if corr > best.1 {
                best = (lag, corr);
            }
        }
        profile.push(best);
        start += win;
    }
    profile
}

/// The transient flag of a single-frame CELT-only packet, read the way
/// celt_decoder.c does: silence, then the post-filter parameters, then the
/// transient bit.
fn celt_transient_flag(packet: &[u8]) -> bool {
    assert!(
        packet[0] & 0x80 != 0,
        "not a CELT-only packet: TOC {:#04x}",
        packet[0]
    );
    assert_eq!(packet[0] & 0x03, 0, "not a single-frame packet");
    let payload = &packet[1..];
    let total_bits = payload.len() as i32 * 8;
    let mut rc = RangeCoder::new_decoder(payload);
    if rc.decode_bit_logp(15) {
        return false; // silence
    }
    if rc.tell() + 16 <= total_bits && rc.decode_bit_logp(1) {
        let octave = rc.dec_uint(6);
        rc.dec_bits(4 + octave); // pitch period
        rc.dec_bits(3); // gain
        if rc.tell() + 2 <= total_bits {
            rc.decode_icdf(&[2, 1, 0], 2); // tapset
        }
    }
    rc.tell() + 3 <= total_bits && rc.decode_bit_logp(3)
}

#[cfg(feature = "analysis")]
#[test]
fn voip_48k_tracks_libopus_preskip() {
    let profile = lag_profile(Application::Voip, 32000);
    let reference = libopus_voip_profile(32000);
    assert_eq!(profile.len(), reference.len());
    // Every window has libopus's lag within a sample and its correlation
    // within 0.01. In the Hybrid part (from the second window to the
    // switch to CELT-only) that is within a couple of samples of the
    // pre-skip at waveform fidelity; libopus's own encoder measures -5..-2.
    for (w, (&(lag, corr), &(c_lag, c_corr))) in profile.iter().zip(&reference).enumerate() {
        assert!(
            (lag - c_lag).abs() <= 1 && corr >= c_corr - 0.01,
            "Voip window {w}: lag {lag}, corr {corr:.3} vs libopus's {c_lag}, {c_corr:.3}; \
             profile {profile:?}, libopus {reference:?}"
        );
        if (1..10).contains(&w) {
            assert!(lag.abs() <= 5 && corr >= 0.98, "Voip window {w}: lag {lag}, corr {corr:.3}");
        }
    }
}

/// libopus's packets for the chirp, Audio, forced CELT-only.
fn libopus_celt_packets(bitrate: i32) -> Vec<Vec<u8>> {
    use audiopus_sys as sys;
    let input = chirp(SR, 13);
    let mut err = 0;
    let st = unsafe { sys::opus_encoder_create(SR as i32, 1, sys::OPUS_APPLICATION_AUDIO, &mut err) };
    assert_eq!(err, 0);
    unsafe {
        sys::opus_encoder_ctl(st, sys::OPUS_SET_BITRATE_REQUEST, bitrate);
        // OPUS_SET_FORCE_MODE (opus_private.h) to MODE_CELT_ONLY.
        sys::opus_encoder_ctl(st, 11002, 1002);
    }
    let packets = input
        .as_chunks::<FRAME>().0.iter()
        .map(|frame| {
            let mut out = vec![0u8; 1500];
            let n = unsafe {
                sys::opus_encode_float(st, frame.as_ptr(), FRAME as i32, out.as_mut_ptr(), 1500)
            };
            assert!(n > 0);
            out.truncate(n as usize);
            out
        })
        .collect();
    unsafe { sys::opus_encoder_destroy(st) };
    packets
}

#[cfg(feature = "analysis")]
#[test]
fn audio_low_rate_celt_startup_is_clean() {
    // Previously corr 0.56-0.66 for the first ~200 ms. libopus forced to
    // CELT-only is lag 0 (-1 in the first window at 20 kb/s, where its
    // speech and music analysis has not settled), corr ≥ 0.99 from the
    // first window. On 64-bit ARM opus1's packets are libopus's.
    for bitrate in [20000, 24000] {
        let (_, packets) = encode_chirp(Application::Audio, bitrate);
        if cfg!(target_arch = "aarch64") {
            assert!(packets == libopus_celt_packets(bitrate), "Audio {bitrate}: not libopus's packets");
        }
        let profile = lag_profile(Application::Audio, bitrate);
        for (w, &(lag, corr)) in profile.iter().enumerate() {
            assert!(
                (lag == 0 || (w == 0 && lag == -1)) && corr >= 0.99,
                "Audio {bitrate} window {w}: lag {lag}, correlation {corr:.3}; \
                 profile {profile:?}"
            );
        }
    }
}

#[test]
fn celt_steady_tone_flags_transient_only_on_first_frame() {
    // The chirp's onset is the only transient; libopus flags frame 0 alone.
    for bitrate in [20000, 24000, 32000, 64000] {
        let (_, packets) = encode_chirp(Application::Audio, bitrate);
        let flagged: Vec<usize> = packets
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, p)| celt_transient_flag(p))
            .map(|(i, _)| i)
            .collect();
        assert!(
            flagged.is_empty(),
            "Audio {bitrate}: transient flagged on steady-tone frames {flagged:?}"
        );
    }
}
