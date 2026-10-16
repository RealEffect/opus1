//! The CELT encoder with its settings changed between frames.
//!
//! Every setting the Opus encoder can change from one CELT frame to the
//! next is changed at random here: frame duration, bitrate, VBR and its
//! constraint, packet size, complexity, coded bandwidth, coded channels,
//! prediction mode, packet loss, LSB depth, and `reset_state`. The input
//! goes from silence to full scale. Nothing may panic, no packet may be
//! longer than its budget, and C libopus must decode every packet to the
//! same final range as the encoder's.

mod celtenc_common;

use celtenc_common::Noise;
use opus1::internals::celt::CeltEncoder;
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;

/// One frame of `n` samples per channel, the channels interleaved.
fn signal(kind: u32, n: usize, channels: usize, noise: &mut Noise, phase: &mut f32) -> Vec<f32> {
    let mut x = vec![0.0f32; n * channels];
    for i in 0..n {
        *phase += 0.031 + 0.0007 * (kind as f32);
        let s = match kind {
            // Silence.
            0 => 0.0,
            // Below the 24-bit LSB.
            1 => 3e-8 * noise.next(),
            // Very quiet.
            2 => 1e-5 * noise.next(),
            // Full-scale square wave.
            3 => {
                if (i / 37) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            // Loud noise.
            4 => 0.9 * noise.next().clamp(-1.0, 1.0),
            // A tone.
            5 => 0.7 * phase.sin(),
            // A DC offset near full scale.
            6 => 0.99,
            // Clicks.
            _ => {
                if i % 97 == 0 {
                    1.0
                } else {
                    0.0
                }
            }
        };
        x[i * channels] = s;
        if channels == 2 {
            // The second channel is the first turned upside down for the
            // even kinds: all side, no mid.
            x[2 * i + 1] = if kind % 2 == 0 { -s } else { 0.5 * s };
        }
    }
    x
}

/// Encode one frame. Returns the payload and the final range.
fn encode(enc: &mut CeltEncoder, pcm: &[f32], n: usize, bytes: usize) -> (Vec<u8>, u32) {
    let mut rc = RangeCoder::new_encoder(bytes as u32);
    enc.encode(pcm, n, &mut rc);
    rc.done();
    assert_eq!(rc.error, 0, "range coder error");
    let len = rc.storage as usize;
    assert!(len <= bytes, "{len} bytes for a budget of {bytes}");
    (rc.buf[..len].to_vec(), rc.rng)
}

#[test]
fn settings_changed_between_frames() {
    let mut rng = Noise(12345);
    // A uniform integer below `n`.
    let mut pick = |n: usize| -> usize { ((rng.next().abs() * 1e6) as usize) % n };
    let mut frames = 0usize;
    for (fs, channels) in [(48_000usize, 2usize), (48_000, 1), (16_000, 2), (8_000, 1), (24_000, 2), (12_000, 1)] {
        let mut enc = CeltEncoder::with_sampling_rate(default_mode(), channels, fs as i32);
        let ch = if channels == 2 {
            opus::Channels::Stereo
        } else {
            opus::Channels::Mono
        };
        let mut dec = opus::Decoder::new(fs as u32, ch).expect("opus decoder");
        let mut noise = Noise(fs as u32 + channels as u32);
        let mut phase = 0.0f32;
        let mut out = vec![0.0f32; fs / 50 * channels];
        for k in 0..400 {
            let lm = pick(4);
            let n = fs / 400 << lm;
            let vbr = pick(2) == 0;
            let bitrate = [6_000, 12_000, 24_000, 48_000, 96_000, 160_000, 320_000, 510_000][pick(8)];
            let bytes = if vbr {
                [2, 3, 5, 8, 40, 300, 1275][pick(7)]
            } else {
                // The packet size of the bitrate, or a tiny one.
                if pick(4) == 0 {
                    2 + pick(8)
                } else {
                    (bitrate * n / fs / 8).clamp(2, 1275)
                }
            };
            enc.set_vbr(vbr);
            let celt_bitrate = if vbr { bitrate as i32 } else { -1 };
            enc.set_bitrate(celt_bitrate).unwrap();
            enc.set_constrained_vbr(pick(2) == 0);
            enc.complexity = pick(11) as i32;
            let end = [13usize, 17, 19, 21][pick(4)];
            enc.set_end_band(end);
            let stream_channels = if channels == 2 { 1 + pick(2) } else { 1 };
            enc.set_stream_channels(stream_channels).expect("channels");
            enc.set_prediction(pick(3) as i32).expect("prediction");
            enc.set_packet_loss_perc([0, 1, 3, 5, 10, 25, 100][pick(7)]).expect("loss");
            enc.set_lsb_depth([8, 16, 24][pick(3)]);
            if pick(25) == 0 {
                enc.reset_state();
            }
            let kind = pick(8) as u32;
            let pcm = signal(kind, n, channels, &mut noise, &mut phase);
            let (payload, range) = encode(&mut enc, &pcm, n, bytes);

            let toc = celtenc_common::celt_toc_lm(end, stream_channels, lm);
            let mut packet = vec![toc];
            packet.extend_from_slice(&payload);
            let got = dec
                .decode_float(&packet, &mut out, false)
                .unwrap_or_else(|e| panic!("{fs} Hz {channels} ch frame {k}: libopus rejected it: {e}"));
            assert_eq!(got, n, "{fs} Hz {channels} ch frame {k}");
            assert_eq!(
                dec.get_final_range().expect("final range"),
                range,
                "{fs} Hz {channels} ch frame {k}: lm {lm}, {bytes} bytes, vbr {vbr}, end {end}, \
                 {stream_channels} coded channels, signal {kind}"
            );
            frames += 1;
        }
    }
    println!("{frames} frames");
}
