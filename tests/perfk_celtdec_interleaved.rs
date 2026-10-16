//! The CELT decoder's interleaved output layouts
//! (`PcmLayout::Interleaved`, `PcmLayout::InterleavedAdd`) must give the
//! same samples as the planar layout, bit for bit, in the other order. With
//! `InterleavedAdd` they must give the sum of those samples and what the
//! buffer held.
//!
//! Two decoders are fed the same packets. One writes planar, the other
//! interleaved. Their states stay equal, so every frame can be compared.

use opus1::internals::celt::{CeltDecoder, CeltEncoder, PcmLayout};
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;

/// A tone with noise bursts and pauses, different in the two channels.
fn test_signal(channels: usize, samples: usize) -> Vec<f32> {
    let mut seed = 5u32;
    let mut pcm = vec![0.0f32; samples * channels];
    for i in 0..samples {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 8) as f32 / 8_388_608.0 - 1.0;
        let t = i as f32 / 48000.0;
        let tone = (2.0 * std::f32::consts::PI * 330.0 * t).sin();
        let burst = (i / 1700) % 6 == 4;
        let pause = (i / 9000) % 5 == 3;
        for ch in 0..channels {
            let side = if ch == 0 { 1.0 } else { -0.7 };
            pcm[i * channels + ch] = if pause {
                0.0
            } else if burst {
                0.5 * noise * side
            } else {
                0.3 * tone * side + 0.02 * noise
            };
        }
    }
    pcm
}

/// Encode `frames` frames of `n` samples (at 48 kHz) with `bytes` bytes each.
fn encode(
    channels: usize,
    n: usize,
    frames: usize,
    bytes: usize,
    start_band: usize,
) -> Vec<Vec<u8>> {
    let mode = default_mode();
    let pcm = test_signal(channels, frames * n);
    let mut enc = CeltEncoder::new(mode, channels);
    enc.set_start_band(start_band).unwrap();
    (0..frames)
        .map(|k| {
            let frame = &pcm[k * n * channels..(k + 1) * n * channels];
            let mut rc = RangeCoder::new_encoder(bytes as u32);
            enc.encode(frame, n, &mut rc);
            rc.done();
            rc.buf[..bytes].to_vec()
        })
        .collect()
}

/// A value that depends on the position, to fill a buffer before an `add`.
fn base_value(i: usize) -> f32 {
    ((i * 37 % 201) as f32 - 100.0) / 256.0
}

/// Decode `packets` with a planar and with an interleaved decoder and
/// compare every frame. `lost(k)` says which frames are concealed.
#[allow(clippy::too_many_arguments)]
fn compare(
    packets: &[Vec<u8>],
    stream_channels: usize,
    channels: usize,
    rate: i32,
    n: usize,
    start_band: usize,
    add: bool,
    lost: impl Fn(usize) -> bool,
) {
    let mode = default_mode();
    let frame_size = n * rate as usize / 48000;
    let mut planar_dec = CeltDecoder::new(mode, channels, rate);
    let mut inter_dec = CeltDecoder::new(mode, channels, rate);
    for dec in [&mut planar_dec, &mut inter_dec] {
        dec.set_stream_channels(stream_channels).unwrap();
        dec.set_start_band(start_band).unwrap();
        dec.set_end_band(mode.nb_ebands).unwrap();
    }
    let layout = if add {
        PcmLayout::InterleavedAdd
    } else {
        PcmLayout::Interleaved
    };
    let mut planar = vec![0.0f32; frame_size * channels];
    // Two extra samples show a write past the frame.
    let mut inter = vec![0.0f32; frame_size * channels + 2];
    let what = format!(
        "{stream_channels} -> {channels} channels, {rate} Hz, n {n}, start {start_band}, add {add}"
    );

    for (k, packet) in packets.iter().enumerate() {
        for (i, v) in inter.iter_mut().enumerate() {
            *v = base_value(i + k);
        }
        let bits = (packet.len() * 8) as i32;
        if lost(k) {
            planar_dec
                .decode_lost(frame_size, &mut planar, PcmLayout::Planar)
                .unwrap();
            let got = inter_dec.decode_lost(frame_size, &mut inter, layout).unwrap();
            assert_eq!(got, frame_size, "{what}: frame {k}");
        } else {
            let mut rc = RangeCoder::new_decoder(packet);
            let got = planar_dec.decode(&mut rc, bits, frame_size, &mut planar, PcmLayout::Planar);
            assert_eq!(got, frame_size, "{what}: frame {k}");
            let mut rc = RangeCoder::new_decoder(packet);
            let got = inter_dec.decode(&mut rc, bits, frame_size, &mut inter, layout);
            assert_eq!(got, frame_size, "{what}: frame {k}");
        }
        assert_eq!(
            planar_dec.final_range(),
            inter_dec.final_range(),
            "{what}: frame {k}"
        );

        for i in 0..frame_size {
            for ch in 0..channels {
                let at = i * channels + ch;
                let sample = planar[ch * frame_size + i];
                let want = if add {
                    base_value(at + k) + sample
                } else {
                    sample
                };
                assert_eq!(
                    inter[at].to_bits(),
                    want.to_bits(),
                    "{what}: frame {k}, sample {i}, channel {ch}: {} instead of {want}",
                    inter[at]
                );
            }
        }
        for (j, v) in inter[frame_size * channels..].iter().enumerate() {
            let at = frame_size * channels + j;
            assert_eq!(
                v.to_bits(),
                base_value(at + k).to_bits(),
                "{what}: wrote past the frame"
            );
        }
    }
}

#[test]
fn interleaved_output_matches_planar_output() {
    for channels in [1usize, 2] {
        for n in [120usize, 240, 480, 960] {
            for (start_band, bytes) in [(0usize, 60usize), (17, 20)] {
                let packets = encode(channels, n, 24, bytes, start_band);
                for rate in [48000, 24000, 16000, 12000, 8000] {
                    for add in [false, true] {
                        compare(
                            &packets,
                            channels,
                            channels,
                            rate,
                            n,
                            start_band,
                            add,
                            |_| false,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn interleaved_output_matches_planar_output_with_lost_frames() {
    for channels in [1usize, 2] {
        for n in [120usize, 480, 960] {
            for (start_band, bytes) in [(0usize, 60usize), (17, 20)] {
                let packets = encode(channels, n, 40, bytes, start_band);
                // Single losses, and a loss long enough to go from the
                // pitch concealment to the noise concealment.
                let lost = |k: usize| k % 9 == 7 || (20..34).contains(&k);
                for rate in [48000, 16000] {
                    for add in [false, true] {
                        compare(&packets, channels, channels, rate, n, start_band, add, lost);
                    }
                }
            }
        }
    }
}

#[test]
fn interleaved_output_matches_planar_output_when_channel_counts_differ() {
    for (stream_channels, channels) in [(1usize, 2usize), (2, 1)] {
        for n in [240usize, 960] {
            let packets = encode(stream_channels, n, 24, 60, 0);
            for rate in [48000, 24000] {
                for add in [false, true] {
                    compare(&packets, stream_channels, channels, rate, n, 0, add, |k| {
                        k == 11
                    });
                }
            }
        }
    }
}

#[test]
fn interleaved_layouts_reject_a_short_buffer() {
    let mode = default_mode();
    let packets = encode(2, 960, 1, 60, 0);
    let mut dec = CeltDecoder::new(mode, 2, 48000);
    let mut short = vec![0.0f32; 2 * 960 - 1];
    let mut rc = RangeCoder::new_decoder(&packets[0]);
    let got = dec.decode(&mut rc, 480, 960, &mut short, PcmLayout::Interleaved);
    assert_eq!(got, 0);
    assert!(
        dec.decode_lost(960, &mut short, PcmLayout::InterleavedAdd)
            .is_err()
    );
    assert!(short.iter().all(|&v| v == 0.0));
}
