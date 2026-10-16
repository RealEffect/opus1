//! How close the opus1 decoder is to the libopus decoder on streams with a
//! CELT part.
//!
//! libopus encodes a music-like test signal with the settings of the CELT
//! and Hybrid bench scenarios. Both decoders then decode the same packets.
//! The test prints, per stream, the signal-to-noise ratio of the opus1
//! output against the libopus output (libopus power divided by the power of
//! the difference, in decibels), the largest difference, and the share of
//! samples that are equal bit for bit. Run with `--nocapture` to see them.
//!
//! The FFT and the MDCT are one part of the decoder. The table shows what a
//! change of their rounding does to the decoded signal.

use audiopus_sys as ffi;
use opus1::OpusDecoder;
use std::os::raw::c_int;

const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
const MODE_HYBRID: c_int = 1001;
const MODE_CELT_ONLY: c_int = 1002;
const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;

const RATE: usize = 48_000;
const SECONDS: usize = 4;

/// One stream to decode.
struct Stream {
    name: &'static str,
    channels: usize,
    bitrate: i32,
    /// Frame length in samples per channel at 48 kHz.
    frame: usize,
    mode: c_int,
}

fn streams() -> Vec<Stream> {
    let s = |name, channels, bitrate, frame, mode| Stream {
        name,
        channels,
        bitrate,
        frame,
        mode,
    };
    vec![
        s("celt_fb_48k_mono_64k_20ms", 1, 64_000, 960, MODE_CELT_ONLY),
        s("celt_fb_48k_mono_24k_20ms", 1, 24_000, 960, MODE_CELT_ONLY),
        s("celt_fb_48k_stereo_128k_20ms", 2, 128_000, 960, MODE_CELT_ONLY),
        s("celt_fb_48k_stereo_128k_10ms", 2, 128_000, 480, MODE_CELT_ONLY),
        s("celt_fb_48k_stereo_128k_5ms", 2, 128_000, 240, MODE_CELT_ONLY),
        s("celt_fb_48k_stereo_128k_2p5ms", 2, 128_000, 120, MODE_CELT_ONLY),
        s("celt_fb_48k_stereo_256k_20ms", 2, 256_000, 960, MODE_CELT_ONLY),
        s("hybrid_fb_48k_mono_32k_20ms", 1, 32_000, 960, MODE_HYBRID),
        s("hybrid_fb_48k_stereo_40k_20ms", 2, 40_000, 960, MODE_HYBRID),
    ]
}

/// Deterministic pseudo-random numbers, so every run tests the same data.
struct Lcg(u64);

impl Lcg {
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

/// A music-like signal: chords of decaying notes with harmonics, a noise
/// burst twice a second (a transient, so the coder uses short blocks), and
/// quiet background noise. Interleaved, peak about 0.5.
fn test_signal(channels: usize) -> Vec<f32> {
    use std::f64::consts::TAU;
    const NOTES: [[f64; 3]; 4] = [
        [220.0, 277.18, 329.63],
        [174.61, 220.0, 261.63],
        [261.63, 329.63, 392.0],
        [196.0, 246.94, 293.66],
    ];
    let n = RATE * SECONDS;
    let mut rng = Lcg(0x51_6e41);
    let mut out = vec![0f32; n * channels];
    for i in 0..n {
        let t = i as f64 / RATE as f64;
        let step = (t * 2.0) as usize;
        let since = t - step as f64 / 2.0;
        let chord = NOTES[step % NOTES.len()];
        let decay = (-3.0 * since).exp();
        let mut left = 0.0;
        let mut right = 0.0;
        for (k, &f) in chord.iter().enumerate() {
            let mut tone = 0.0;
            for h in 1..=8 {
                tone += (TAU * f * h as f64 * t).sin() / (h * h) as f64;
            }
            // Each note sits at another place between the two channels.
            let pan = k as f64 / 2.0;
            left += tone * (1.0 - pan) * decay;
            right += tone * pan * decay;
        }
        let noise = rng.unit() as f64;
        let burst = if since < 0.01 { 0.4 * noise } else { 0.0 };
        left = 0.12 * left + burst + 0.002 * noise;
        right = 0.12 * right + 0.7 * burst + 0.002 * rng.unit() as f64;
        if channels == 1 {
            out[i] = (0.5 * (left + right)) as f32;
        } else {
            out[2 * i] = left as f32;
            out[2 * i + 1] = right as f32;
        }
    }
    out
}

/// The packets libopus makes of `pcm` for the stream.
fn encode(stream: &Stream, pcm: &[f32]) -> Vec<Vec<u8>> {
    let mut error: c_int = 0;
    // SAFETY: plain C calls on a state that this function creates and frees.
    // `pcm` is read in whole frames, the packet buffer has room for the
    // largest packet.
    unsafe {
        let enc = ffi::opus_encoder_create(
            RATE as i32,
            stream.channels as c_int,
            ffi::OPUS_APPLICATION_AUDIO,
            &mut error,
        );
        assert!(!enc.is_null() && error == ffi::OPUS_OK);
        for (request, value) in [
            (ffi::OPUS_SET_BITRATE_REQUEST, stream.bitrate),
            (ffi::OPUS_SET_COMPLEXITY_REQUEST, 10),
            (OPUS_SET_FORCE_MODE_REQUEST, stream.mode),
            (ffi::OPUS_SET_BANDWIDTH_REQUEST, OPUS_BANDWIDTH_FULLBAND),
            (ffi::OPUS_SET_FORCE_CHANNELS_REQUEST, stream.channels as c_int),
        ] {
            assert_eq!(ffi::opus_encoder_ctl(enc, request, value), ffi::OPUS_OK);
        }
        let mut packets = Vec::new();
        let mut buf = vec![0u8; 4000];
        for frame in pcm.chunks_exact(stream.frame * stream.channels) {
            let len = ffi::opus_encode_float(
                enc,
                frame.as_ptr(),
                stream.frame as c_int,
                buf.as_mut_ptr(),
                buf.len() as i32,
            );
            assert!(len > 0, "{}: opus_encode_float returned {len}", stream.name);
            packets.push(buf[..len as usize].to_vec());
        }
        ffi::opus_encoder_destroy(enc);
        packets
    }
}

/// The samples the libopus decoder makes of the packets.
fn decode_c(stream: &Stream, packets: &[Vec<u8>]) -> Vec<f32> {
    let mut error: c_int = 0;
    let mut out = Vec::new();
    let mut frame = vec![0f32; stream.frame * stream.channels];
    // SAFETY: plain C calls on a state that this function creates and frees.
    // `frame` has room for one frame of all channels.
    unsafe {
        let dec = ffi::opus_decoder_create(RATE as i32, stream.channels as c_int, &mut error);
        assert!(!dec.is_null() && error == ffi::OPUS_OK);
        for p in packets {
            let n = ffi::opus_decode_float(
                dec,
                p.as_ptr(),
                p.len() as i32,
                frame.as_mut_ptr(),
                stream.frame as c_int,
                0,
            );
            assert_eq!(n as usize, stream.frame, "{}", stream.name);
            out.extend_from_slice(&frame);
        }
        ffi::opus_decoder_destroy(dec);
    }
    out
}

/// The samples the opus1 decoder makes of the packets.
fn decode_rust(stream: &Stream, packets: &[Vec<u8>]) -> Vec<f32> {
    let mut dec = OpusDecoder::new(RATE as i32, stream.channels).unwrap();
    let mut out = Vec::new();
    let mut frame = vec![0f32; stream.frame * stream.channels];
    for p in packets {
        let n = dec.decode(p, stream.frame, &mut frame).unwrap();
        assert_eq!(n, stream.frame, "{}", stream.name);
        out.extend_from_slice(&frame);
    }
    out
}

/// The opus1 decoder may differ from the libopus decoder in the last bits
/// only: at least 100 dB below the signal on every stream.
const MIN_SNR_DB: f64 = 100.0;

#[test]
fn decoded_signal_against_libopus() {
    println!(
        "{:<32} {:>10} {:>12} {:>12}",
        "stream", "SNR dB", "largest diff", "equal bits %"
    );
    for stream in streams() {
        let pcm = test_signal(stream.channels);
        let packets = encode(&stream, &pcm);
        let c = decode_c(&stream, &packets);
        let r = decode_rust(&stream, &packets);
        assert_eq!(c.len(), r.len());

        let signal: f64 = c.iter().map(|&v| v as f64 * v as f64).sum();
        let noise: f64 = c
            .iter()
            .zip(&r)
            .map(|(&a, &b)| (a as f64 - b as f64) * (a as f64 - b as f64))
            .sum();
        let largest = c.iter().zip(&r).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let equal = c.iter().zip(&r).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let snr = 10.0 * (signal / noise).log10();
        println!(
            "{:<32} {:>10.1} {:>12.2e} {:>12.2}",
            stream.name,
            snr,
            largest,
            100.0 * equal as f64 / c.len() as f64
        );
        assert!(signal > 0.0, "{}: silent", stream.name);
        assert!(snr >= MIN_SNR_DB, "{}: {snr:.1} dB", stream.name);
    }
}
