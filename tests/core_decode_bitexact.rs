//! The float output of `OpusDecoder` against libopus `opus_decode_float`,
//! sample by sample, in places that once differed.
//!
//! On 64-bit ARM, where opus1 rounds as clang compiles libopus, the bits
//! must be equal. On other targets the tests require an agreement of
//! 100 dB.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::load_speech_16k;
use opus1::OpusDecoder;

/// Decoded float output of one call: libopus's and opus1's.
struct Call {
    c: Vec<f32>,
    ours: Vec<f32>,
}

/// Compare two decoded calls. On 64-bit ARM the bits must be equal.
fn assert_same(label: &str, call: &Call) {
    assert_eq!(call.c.len(), call.ours.len(), "{label}: lengths");
    if cfg!(target_arch = "aarch64") {
        let diff = (0..call.c.len()).filter(|&i| call.c[i].to_bits() != call.ours[i].to_bits()).count();
        if let Some(i) = (0..call.c.len()).find(|&i| call.c[i].to_bits() != call.ours[i].to_bits()) {
            panic!("{label}: {diff} of {} samples differ, first at {i}: libopus {}, opus1 {}", call.c.len(), call.c[i], call.ours[i]);
        }
    } else {
        let db = snr(&call.c, &call.ours);
        assert!(db >= 100.0, "{label}: {db:.1} dB");
    }
}

/// Whether a packet is SILK-only (configurations 0 to 11).
fn is_silk(p: &[u8]) -> bool {
    p[0] >> 3 < 12
}

/// Whether a packet is CELT-only (configurations 16 to 31).
fn is_celt(p: &[u8]) -> bool {
    p[0] >> 3 >= 16
}

/// libopus's encoder, 16 kHz mono speech at 24 kb/s for VoIP, switches
/// from SILK to CELT after 28 packets. The last SILK packet carries a
/// redundant 5 ms CELT frame. Its decoder fades the end of the SILK frame
/// into that CELT frame (`smooth_fade`), with a weight `w`:
/// `w*celt + (1-w)*silk`. clang computes `(1-w)*silk` first and adds
/// `w*celt` in one fused multiply-add. opus1 rounded both products, and 1
/// to 14 samples of that packet (depending on the output rate) differed
/// from libopus in the last bit.
#[test]
fn silk_to_celt_switch_fades_like_libopus() {
    let rate = 16000;
    let pcm = load_speech_16k();
    let mut enc = CEncoder::new(rate, 1, APP_VOIP);
    enc.bitrate(24000);
    let packets = enc.encode_all(&pcm[..rate * 4], rate / 50);
    let switch = packets.windows(2).position(|w| is_silk(&w[0]) && is_celt(&w[1]));
    assert_eq!(switch, Some(27), "the stream was meant to switch from SILK to CELT after packet 27");
    for out_rate in [8000usize, 12000, 16000, 24000, 48000] {
        let mut c = CDecoder::new(out_rate, 1);
        let mut ours = OpusDecoder::new(out_rate as i32, 1).unwrap();
        let n = out_rate / 50;
        let mut buf = vec![0f32; n];
        for (k, p) in packets.iter().enumerate() {
            assert_eq!(ours.decode(p, n, &mut buf), Ok(n));
            let call = Call { c: c.decode(Some(p), n, false).unwrap(), ours: buf.clone() };
            assert_same(&format!("{out_rate} Hz, packet {k}"), &call);
        }
    }
}

/// Packets of libopus's encoder whose bitrate changes every 200 ms, so
/// that the coding mode changes back and forth. Every change between
/// CELT-only and SILK-only or Hybrid is faded with `smooth_fade`: through
/// a redundant CELT frame, or through 5 ms of CELT concealment
/// (`pcm_transition`) where the packet has no redundant frame.
fn switching_packets(rate: usize, channels: usize, application: i32) -> Vec<Vec<u8>> {
    let pcm = load_speech_16k();
    let mono = k24_common::resample(&pcm[..16000 * 6], 16000, rate);
    let pcm = if channels == 2 { k24_common::to_stereo(&mono, rate) } else { mono };
    let mut enc = CEncoder::new(rate, channels, application);
    let frame = rate / 50;
    let bitrates = [10000, 64000, 16000, 40000, 8000, 96000, 24000, 12000];
    pcm.chunks_exact(frame * channels)
        .enumerate()
        .map(|(k, f)| {
            enc.bitrate(bitrates[k / 10 % bitrates.len()] * channels as i32);
            enc.encode(f)
        })
        .collect()
}

/// Mode switches in both directions, at every output rate, mono and
/// stereo, with every 7th packet lost. Before the fade was fixed, the 8
/// switching streams of `perf_harness modes` agreed with libopus to 172
/// to 192 dB.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn mode_switches_decode_like_libopus() {
    let mut to_celt = 0;
    let mut from_celt = 0;
    for (rate, channels, application) in [(48000, 1, APP_VOIP), (48000, 2, APP_AUDIO), (24000, 1, APP_AUDIO), (16000, 2, APP_VOIP)] {
        let packets = switching_packets(rate, channels, application);
        to_celt += packets.windows(2).filter(|w| !is_celt(&w[0]) && is_celt(&w[1])).count();
        from_celt += packets.windows(2).filter(|w| is_celt(&w[0]) && !is_celt(&w[1])).count();
        for out_rate in [8000usize, 12000, 16000, 24000, 48000] {
            let mut c = CDecoder::new(out_rate, channels);
            let mut ours = OpusDecoder::new(out_rate as i32, channels).unwrap();
            let n = out_rate / 50;
            let mut buf = vec![0f32; n * channels];
            for (k, p) in packets.iter().enumerate() {
                let lost = k % 7 == 6;
                let (reference, got) = if lost {
                    (c.decode(None, n, false), ours.decode(&[], n, &mut buf))
                } else {
                    (c.decode(Some(p), n, false), ours.decode(p, n, &mut buf))
                };
                assert_eq!(got, Ok(n));
                let call = Call { c: reference.unwrap(), ours: buf.clone() };
                assert_same(&format!("{rate} Hz {channels} ch stream, {out_rate} Hz, packet {k}, lost {lost}"), &call);
            }
        }
    }
    println!("{to_celt} switches to CELT, {from_celt} from CELT");
    assert!(to_celt >= 8 && from_celt >= 6, "{to_celt} switches to CELT, {from_celt} from CELT");
}
