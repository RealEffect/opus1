//! The front end of the CELT encoder, through the public API: the silence
//! scan, the pre-emphasis for every input rate, and the input history that
//! one frame hands to the next.
//!
//! The kernels behind these have unit tests of their own in `src/celt.rs`
//! (`enc_kernel_tests`), which compare them bit for bit with the plain
//! loops they replace. The tests here check what a caller can observe.

use opus1::internals::celt::{CeltDecoder, CeltEncoder, PcmLayout};
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;

/// Input rates of the CELT encoder and how many 48 kHz samples one input
/// sample stands for.
const RATES: [(i32, usize); 5] = [(48000, 1), (24000, 2), (16000, 3), (12000, 4), (8000, 6)];

/// MDCT overlap of the 48 kHz mode, in 48 kHz samples.
const OVERLAP: usize = 120;

/// Encodes one frame and returns the packet bytes. `pcm` holds `frame`
/// samples per channel, the channels interleaved.
fn encode(enc: &mut CeltEncoder, pcm: &[f32], frame: usize, bytes: u32) -> Vec<u8> {
    let mut rc = RangeCoder::new_encoder(bytes);
    enc.encode(pcm, frame, &mut rc);
    rc.done();
    rc.buf[..bytes as usize].to_vec()
}

/// The first symbol of a CELT frame: whether the frame is coded as silence.
fn silence_flag(packet: &[u8]) -> bool {
    RangeCoder::new_decoder(packet).decode_bit_logp(15)
}

/// The positions worth testing in a frame of `n` samples whose last
/// `tail` samples are the overlap part: both ends of the frame and both
/// sides of the place where the overlap part starts.
fn positions(n: usize, tail: usize) -> Vec<usize> {
    let split = n - tail;
    let mut p: Vec<usize> = (0..n)
        .filter(|&i| i < 20 || i + 20 >= n || (i + 20 >= split && i < split + 20))
        .collect();
    p.dedup();
    p
}

/// One sample above the silence threshold, at any position of any channel,
/// makes its frame a normal frame. The scan looks at the samples in groups
/// of 16 and 4 and then one by one, so every position has to be found.
///
/// A sample in the last `overlap` samples of a frame also reaches into the
/// next frame through the window, so that frame is not silent either. A
/// sample before that leaves the next frame silent.
#[test]
fn one_loud_sample_anywhere_ends_the_silence() {
    let mode = default_mode();
    // Default depth is 24 bits: silence is everything up to 2^-24.
    let loud = 1.0e-3f32;
    for (rate, upsample) in RATES {
        for lm in 0..4 {
            let n = (120usize << lm) / upsample;
            let tail = OVERLAP / upsample;
            for channels in 1..=2usize {
                for ch in 0..channels {
                    for pos in positions(n, tail) {
                        let mut enc = CeltEncoder::with_sampling_rate(mode, channels, rate);
                        let mut pcm = vec![0.0f32; n * channels];
                        let what = format!("{rate} Hz, {n} samples, ch {ch}/{channels}, pos {pos}");

                        // A silent frame first.
                        assert!(silence_flag(&encode(&mut enc, &pcm, n, 200)), "{what}: before");

                        pcm[pos * channels + ch] = if pos % 2 == 0 { loud } else { -loud };
                        assert!(!silence_flag(&encode(&mut enc, &pcm, n, 200)), "{what}: frame");

                        pcm[pos * channels + ch] = 0.0;
                        let next = silence_flag(&encode(&mut enc, &pcm, n, 200));
                        assert_eq!(next, pos < n - tail, "{what}: next frame");

                        assert!(silence_flag(&encode(&mut enc, &pcm, n, 200)), "{what}: after");
                    }
                }
            }
        }
    }
}

/// The threshold itself: a frame whose largest sample is exactly 2^-24 is
/// silent, and the next larger value is not. Samples below the threshold
/// around it do not change that.
#[test]
fn silence_threshold_is_exact() {
    let mode = default_mode();
    let threshold = 1.0f32 / (1u32 << 24) as f32;
    let above = f32::from_bits(threshold.to_bits() + 1);
    for n in [120usize, 240, 480, 960] {
        // The last sample before the overlap part, the first sample of it
        // and the last sample of the frame. A frame of 120 samples is all
        // overlap part, so it has no sample before it.
        let before_overlap = n.checked_sub(OVERLAP + 1);
        let fixed = [0usize, 5, 17, n - OVERLAP, n - 1];
        for pos in fixed.into_iter().chain(before_overlap) {
            for (value, silent) in [(threshold, true), (-threshold, true), (above, false), (-above, false)] {
                let mut enc = CeltEncoder::new(mode, 2);
                let mut pcm = vec![0.5 * threshold; 2 * n];
                // The right channel.
                pcm[2 * pos + 1] = value;
                let got = silence_flag(&encode(&mut enc, &pcm, n, 200));
                assert_eq!(got, silent, "n={n} pos={pos} value={value:e}");
            }
        }
    }
}

/// Small deterministic generator.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / (1u32 << 30) as f32 - 1.0
    }
}

/// A few sine tones below 3 kHz plus a little noise, in the range -0.4 to
/// 0.4. `phase` shifts the tones, so two channels can differ.
fn signal(rng: &mut Rng, rate: i32, start: usize, len: usize, phase: f32) -> Vec<f32> {
    (start..start + len)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.2 * (2.0 * std::f32::consts::PI * 440.0 * t + phase).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1250.0 * t + 2.0 * phase).sin()
                + 0.05 * (2.0 * std::f32::consts::PI * 2900.0 * t).sin()
                + 0.01 * rng.unit()
        })
        .collect()
}

/// Signal-to-noise ratio of `out` against `input` in dB, at the delay
/// between 0 and `max_delay` samples that fits best. The first `skip`
/// samples are left out.
fn best_snr(input: &[f32], out: &[f32], skip: usize, max_delay: usize) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for delay in 0..=max_delay {
        let mut sig = 0.0f64;
        let mut err = 0.0f64;
        for i in skip..input.len() - max_delay {
            let d = (input[i] - out[i + delay]) as f64;
            sig += (input[i] as f64) * (input[i] as f64);
            err += d * d;
        }
        best = best.max((10.0 * (sig / (err + 1e-30)).log10()) as f32);
    }
    best
}

/// Encodes with frame sizes that change from frame to frame and decodes
/// again. Each frame's transform starts with the last 120 samples of the
/// frame before it, whatever the two frame sizes are. If the encoder
/// handed over the wrong samples, the decoded signal would break at the
/// frame borders and this signal-to-noise ratio would collapse.
#[test]
fn history_survives_changing_frame_sizes() {
    let mode = default_mode();
    // Frame sizes in units of 2.5 ms, in an order that puts every size
    // after every other size.
    let pattern = [3usize, 0, 2, 1, 3, 3, 1, 0, 0, 2, 2, 3, 1, 1, 2, 0, 3, 2, 0, 1, 3];
    for (rate, upsample) in RATES {
        for channels in 1..=2usize {
            let mut enc = CeltEncoder::with_sampling_rate(mode, channels, rate);
            let mut dec = CeltDecoder::new(mode, channels, rate);
            let mut rng = Rng(rate as u64 + channels as u64);
            // Per channel: everything that went in and came out.
            let mut all_in = vec![Vec::new(); channels];
            let mut all_out = vec![Vec::new(); channels];
            let mut start = 0usize;
            for round in 0..3 {
                for &lm in &pattern {
                    let n = (120usize << lm) / upsample;
                    let mut pcm = vec![0.0f32; n * channels];
                    for (ch, kept) in all_in.iter_mut().enumerate() {
                        let x = signal(&mut rng, rate, start, n, ch as f32 * 0.7);
                        for (i, &v) in x.iter().enumerate() {
                            pcm[i * channels + ch] = v;
                        }
                        kept.extend_from_slice(&x);
                    }
                    start += n;
                    // 160 bytes per 2.5 ms: far more than the signal needs.
                    let packet = encode(&mut enc, &pcm, n, 160 << lm);
                    assert!(!silence_flag(&packet), "{rate} Hz round {round}");
                    // The decoder returns one channel after the other.
                    let mut out = vec![0.0f32; n * channels];
                    let mut rc = RangeCoder::new_decoder(&packet);
                    let bits = (packet.len() * 8) as i32;
                    assert_eq!(dec.decode(&mut rc, bits, n, &mut out, PcmLayout::Planar), n);
                    for (ch, kept) in all_out.iter_mut().enumerate() {
                        kept.extend_from_slice(&out[ch * n..(ch + 1) * n]);
                    }
                }
            }
            for ch in 0..channels {
                let snr = best_snr(&all_in[ch], &all_out[ch], 2000 / upsample, 200);
                println!("{rate} Hz, channel {ch} of {channels}: {snr:.1} dB");
                assert!(
                    snr > 30.0,
                    "{rate} Hz, channel {ch} of {channels}: only {snr:.1} dB after decoding"
                );
            }
        }
    }
}
