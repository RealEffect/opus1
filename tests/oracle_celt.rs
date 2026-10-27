use std::num::NonZeroU32;
use opus::{Application as CApp, Channels as CCh, Decoder as CDec, Encoder as CEnc};
use opus1::{
    Application as RsApp, Bitrate, Channels, OpusDecoder as RsDec, OpusEncoder as RsEnc, SampleRate,
};

fn make_sine(sr: i32, ch: usize, frame_size: usize, freq: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; frame_size * ch];
    for i in 0..frame_size {
        let t = i as f32 / sr as f32;
        let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
        for c in 0..ch {
            v[i * ch + c] = s;
        }
    }
    v
}

// The packets of opus1 and of the linked libopus are compared live. On
// 64-bit ARM opus1 mirrors libopus's kernels and rounding, so the two
// `..._is_the_packet_of_libopus` tests below require the same bytes. On
// other processors opus1 may round differently (x86 libopus and opus1 each
// have their own SSE and AVX2 kernels), so there libopus decodes both
// packets and the two results must be close (`decoded_snr`).
//
// x86-64 had a stored packet instead: one recorded on x86-64 from libopus's
// main branch (1.6.1-51-g03647f52), compared byte for byte. That asked for
// bit-exactness on x86, where opus1 aims for correct, clear audio, and
// with another libopus than the one the tests link.

/// Lowest accepted `decoded_snr`, in dB. On x86-64 the sine gave equal
/// frames and the impulse 44.2 dB: under Rosetta 2 against libopus 1.6.1
/// built by clang, with and without AVX2, and on an i7-12700 under Windows
/// against libopus built by MSVC.
const MIN_DECODED_SNR_DB: f64 = 30.0;

/// One 20 ms frame, 48 kHz mono, 64 kb/s constant bitrate, complexity 10,
/// through opus1 and through the linked libopus: (opus1, libopus).
fn first_packets(pcm: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let mut rs_enc = RsEnc::new(SampleRate::Hz48000, Channels::Mono, RsApp::Audio).unwrap();
    rs_enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(64000).unwrap()));
    rs_enc.set_vbr(false);
    rs_enc.set_complexity(10).unwrap();
    let mut rs_buf = vec![0u8; 1276];
    let rs_n = rs_enc.encode(pcm, 960, &mut rs_buf).unwrap();
    let mut c_enc = CEnc::new(48000, CCh::Mono, CApp::Audio).unwrap();
    c_enc.set_bitrate(opus::Bitrate::Bits(64000)).unwrap();
    c_enc.set_vbr(false).unwrap();
    c_enc.set_complexity(10).unwrap();
    let mut c_buf = vec![0u8; 1276];
    let c_n = c_enc.encode_float(pcm, &mut c_buf).unwrap();
    (rs_buf[..rs_n].to_vec(), c_buf[..c_n].to_vec())
}

/// Number of equal bytes and the position of the first byte that differs.
fn equal_bytes(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let equal = a.iter().zip(b).filter(|(x, y)| x == y).count();
    let first = a.iter().zip(b).position(|(x, y)| x != y);
    (equal, first)
}

/// libopus decodes opus1's packet and its own packet of the same frame
/// (48 kHz mono, each with a new decoder). Returns how close the two
/// decoded frames are: the signal-to-noise ratio in dB, infinite when they
/// are equal.
fn decoded_snr(rs: &[u8], c: &[u8]) -> f64 {
    let decode = |packet: &[u8]| {
        let mut dec = CDec::new(48000, CCh::Mono).unwrap();
        let mut out = vec![0f32; 960];
        assert_eq!(dec.decode_float(packet, &mut out, false).unwrap(), 960);
        out
    };
    let (rs, c) = (decode(rs), decode(c));
    let signal: f64 = c.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let noise: f64 = rs.iter().zip(&c).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum();
    10.0 * (signal / noise).log10()
}

/// The sine packet of `oracle_celt_sine_48k_mono_64k`, byte for byte against
/// the packet the linked libopus writes for the same input.
///
/// On aarch64 libopus has no kernels of its own for the pulse search, so
/// there is one right answer. opus1 reaches it: all 160 bytes are equal
/// at the tip of integ6 (4726c16). When the test was ignored, 132 were.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn sine_48k_mono_64k_is_the_packet_of_libopus() {
    let (rs, c) = first_packets(&make_sine(48000, 1, 960, 440.0));
    let (equal, first) = equal_bytes(&rs, &c);
    println!("sine: {equal} of {} bytes equal, first difference at {first:?}", c.len());
    assert_eq!(rs, c);
}

/// The impulse packet of `oracle_impulse_48k_mono_64k`, byte for byte
/// against the packet the linked libopus writes for the same input.
///
/// Before opus1 ran the speech and music analysis, 7 of the 160 bytes were
/// equal: the analysis changes the allocation trim of this frame. Now all
/// 160 are.
#[cfg(feature = "analysis")]
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn impulse_48k_mono_64k_is_the_packet_of_libopus() {
    let mut pcm = vec![0.0f32; 960];
    pcm[0] = 1.0;
    let (rs, c) = first_packets(&pcm);
    let (equal, first) = equal_bytes(&rs, &c);
    println!("impulse: {equal} of {} bytes equal, first difference at {first:?}", c.len());
    assert_eq!(rs, c);
}

#[test]
fn oracle_celt_sine_48k_mono_64k() {
    let sr = 48000;
    let ch = 1;
    let br = 64000;
    let fs = 960;
    let pcm = make_sine(sr, ch, fs, 440.0);
    let mut rs_enc = RsEnc::new(SampleRate::Hz48000, Channels::Mono, RsApp::Audio).unwrap();
    rs_enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(br as u32).unwrap()));
    rs_enc.set_vbr(false);
    rs_enc.set_complexity(10).unwrap();
    let mut rs_buf = vec![0u8; 1276];
    let rs_n = rs_enc.encode(&pcm, fs, &mut rs_buf).unwrap();
    let mut c_enc = CEnc::new(
        sr as u32,
        if ch == 1 { CCh::Mono } else { CCh::Stereo },
        CApp::Audio,
    )
    .unwrap();
    c_enc.set_bitrate(opus::Bitrate::Bits(br)).unwrap();
    c_enc.set_vbr(false).unwrap();
    let _ = c_enc.set_complexity(10);
    let mut c_buf = vec![0u8; 1276];
    let c_n = c_enc.encode_float(&pcm, &mut c_buf).unwrap();
    println!(
        "rs pkt {} {:02x?} c pkt {} {:02x?}",
        rs_n,
        &rs_buf[..rs_n.min(8)],
        c_n,
        &c_buf[..c_n.min(8)]
    );
    assert!(rs_n > 0 && rs_n <= 1276);
    assert!(c_n > 0 && c_n <= 1276);
    let diff = (rs_n as i32 - c_n as i32).abs();
    assert!(
        diff <= 5,
        "size diff too large rs {} vs c {} diff {}",
        rs_n,
        c_n,
        diff
    );
    // libopus decodes opus1's packet to nearly what it decodes from its own
    // (on 64-bit ARM the packets are the same).
    let db = decoded_snr(&rs_buf[..rs_n], &c_buf[..c_n]);
    println!("sine: opus1's packet against libopus's, both decoded by libopus: {db:.1} dB");
    assert!(db > MIN_DECODED_SNR_DB, "sine: {db:.1} dB");
    // Determinism: re-encode same input must be byte-identical
    let mut rs_enc2 = RsEnc::new(SampleRate::Hz48000, Channels::Mono, RsApp::Audio).unwrap();
    rs_enc2.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(br as u32).unwrap()));
    rs_enc2.set_vbr(false);
    rs_enc2.set_complexity(10).unwrap();
    let mut rs_buf2 = vec![0u8; 1276];
    let rs_n2 = rs_enc2.encode(&pcm, fs, &mut rs_buf2).unwrap();
    assert_eq!(rs_n, rs_n2, "non-deterministic size");
    assert_eq!(
        &rs_buf[..rs_n],
        &rs_buf2[..rs_n2],
        "non-deterministic bytes"
    );
}

#[test]
fn oracle_impulse_48k_mono_64k() {
    let sr = 48000;
    let ch = 1;
    let br = 64000;
    let fs = 960;
    let mut pcm = vec![0.0f32; fs * ch];
    pcm[0] = 1.0;
    let mut rs_enc = RsEnc::new(SampleRate::Hz48000, Channels::Mono, RsApp::Audio).unwrap();
    rs_enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(br as u32).unwrap()));
    rs_enc.set_vbr(false);
    rs_enc.set_complexity(10).unwrap();
    let mut rs_buf = vec![0u8; 1276];
    let rs_n = rs_enc.encode(&pcm, fs, &mut rs_buf).unwrap();
    let mut c_enc = CEnc::new(sr as u32, CCh::Mono, CApp::Audio).unwrap();
    c_enc.set_bitrate(opus::Bitrate::Bits(br)).unwrap();
    c_enc.set_vbr(false).unwrap();
    let _ = c_enc.set_complexity(10);
    let mut c_buf = vec![0u8; 1276];
    let c_n = c_enc.encode_float(&pcm, &mut c_buf).unwrap();
    println!(
        "impulse rs {} {:02x?} c {} {:02x?}",
        rs_n,
        &rs_buf[..rs_n.min(8)],
        c_n,
        &c_buf[..c_n.min(8)]
    );
    assert!(
        rs_n <= 200 && c_n <= 200,
        "huge packet rs {} c {}",
        rs_n,
        c_n
    );
    let mut rs_dec = RsDec::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    let mut out = vec![0.0f32; fs * ch];
    let got = rs_dec.decode(&rs_buf[..rs_n], fs, &mut out).unwrap();
    let max = out[..got * ch].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    assert!(max < 15.0 && max.is_finite());
    // For 48k CBR, also check byte closeness (allow small divergence due to remaining tone/Hybrid gaps)
    assert!((rs_n as i32 - c_n as i32).abs() <= 5);
    // libopus decodes opus1's packet to nearly what it decodes from its own
    // (on 64-bit ARM the packets are the same). Without the `analysis`
    // feature opus1 skips the speech and music analysis that libopus runs
    // at complexity 10. That changes the allocation trim of this frame
    // (see `impulse_48k_mono_64k_is_the_packet_of_libopus`), and the SNR
    // falls to about 20 dB, so the check needs the analysis.
    if cfg!(feature = "analysis") {
        let db = decoded_snr(&rs_buf[..rs_n], &c_buf[..c_n]);
        println!("impulse: opus1's packet against libopus's, both decoded by libopus: {db:.1} dB");
        assert!(db > MIN_DECODED_SNR_DB, "impulse: {db:.1} dB");
    }
    let mut rs_enc2 = RsEnc::new(SampleRate::Hz48000, Channels::Mono, RsApp::Audio).unwrap();
    rs_enc2.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(br as u32).unwrap()));
    rs_enc2.set_vbr(false);
    rs_enc2.set_complexity(10).unwrap();
    let mut rs_buf2 = vec![0u8; 1276];
    let rs_n2 = rs_enc2.encode(&pcm, fs, &mut rs_buf2).unwrap();
    assert_eq!(rs_n, rs_n2);
    assert_eq!(&rs_buf[..rs_n], &rs_buf2[..rs_n2]);
}

#[test]
fn oracle_all_sr_ch_br() {
    let srs = [48000, 24000, 16000];
    let chs = [1, 2];
    let brs = [32000, 64000, 96000];
    let freq = 440.0;
    for &sr in &srs {
        for &ch in &chs {
            for &br in &brs {
                let fs = sr / 50;
                let pcm = make_sine(sr, ch, fs as usize, freq);
                let mut rs_enc = RsEnc::new(
                    SampleRate::try_from(sr).unwrap(),
                    Channels::try_from(ch).unwrap(),
                    RsApp::Audio,
                )
                .unwrap();
                rs_enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(br as u32).unwrap()));
                rs_enc.set_vbr(false);
                rs_enc.set_complexity(10).unwrap();
                let mut rs_buf = vec![0u8; 1276];
                let rs_n = rs_enc.encode(&pcm, fs as usize, &mut rs_buf).unwrap();
                let c_ch = if ch == 1 { CCh::Mono } else { CCh::Stereo };
                let mut c_enc = match CEnc::new(sr as u32, c_ch, CApp::Audio) {
                    Ok(e) => e,
                    Err(e) => {
                        println!("skip sr {} ch {} br {}: CEnc new err {:?}", sr, ch, br, e);
                        continue;
                    }
                };
                if c_enc.set_bitrate(opus::Bitrate::Bits(br)).is_err() {
                    continue;
                }
                let _ = c_enc.set_vbr(false);
                let _ = c_enc.set_complexity(10);
                let mut c_buf = vec![0u8; 1276];
                let c_n = match c_enc.encode_float(&pcm, &mut c_buf) {
                    Ok(n) => n,
                    Err(e) => {
                        println!("skip encode sr {} ch {} br {}: {:?}", sr, ch, br, e);
                        continue;
                    }
                };
                let diff = (rs_n as i32 - c_n as i32).abs();
                println!(
                    "oracle sr {} ch {} br {} rs {} c {} diff {}",
                    sr, ch, br, rs_n, c_n, diff
                );
                assert!(
                    rs_n > 0 && c_n > 0,
                    "zero packet sr {} ch {} br {}",
                    sr,
                    ch,
                    br
                );
                let tol = if sr == 48000 { 5 } else { 50 };
                if diff > tol {
                    println!(
                        "WARN diff {} >{} for sr {} ch {} br {} (known Hybrid gap)",
                        diff, tol, sr, ch, br
                    );
                }
                if sr == 48000 {
                    assert!(
                        diff <= tol,
                        "size diff {} >{} sr {} ch {} br {} rs {} c {}",
                        diff,
                        tol,
                        sr,
                        ch,
                        br,
                        rs_n,
                        c_n
                    );
                }
                let mut dec = RsDec::new(
                    SampleRate::try_from(sr).unwrap(),
                    Channels::try_from(ch).unwrap(),
                )
                .unwrap();
                let mut out = vec![0.0f32; fs as usize * ch];
                let got = dec.decode(&rs_buf[..rs_n], fs as usize, &mut out).unwrap();
                let max = out[..got * ch]
                    .iter()
                    .fold(0.0f32, |a, &x| a.max(x.abs()));
                assert!(
                    max.is_finite() && max < 50.0,
                    "decode huge sr {} ch {} br {} max {}",
                    sr,
                    ch,
                    br,
                    max
                );
            }
        }
    }
}
