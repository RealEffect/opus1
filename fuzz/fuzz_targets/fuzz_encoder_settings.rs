//! The single-stream encoder with settings that change between frames.
//!
//! The input chooses a sampling rate, a channel count and an application.
//! Then, before each frame, it may change the bitrate, constant or variable
//! bitrate (CBR or VBR), the complexity, the forced mode, the bandwidth and
//! the maximum bandwidth, the forced channel count, in-band forward error
//! correction (FEC) and the expected packet loss, the low-frequency-effects
//! (LFE) flag, the energy mask, prediction, the frame size, the size of the
//! output buffer, the input signal and its format (`f32` or `i16`), or
//! reset the encoder.
//!
//! What must hold for every frame:
//!
//! - A frame of 2.5, 5, 10, 20, 40 or 60 ms into an output buffer of at
//!   least 1 byte is encoded. Any other frame size is refused.
//! - The packet fits the buffer, and libopus's decoder decodes it to
//!   exactly one frame. Its final range (the state of the range decoder
//!   after the packet) equals the encoder's final range.
//! - opus1's decoder does the same. Its output is within 50 dB of
//!   libopus's until a packet has a frame of 0 or 1 bytes, which the
//!   decoders conceal, or switches from CELT-only to SILK or Hybrid.
//! - A second encoder with the same settings, always given `f32` samples
//!   (for 16-bit input the samples `s as f32 / 32768.0`), writes the same
//!   packets: `encode_i16` promises this, and an encoder must not depend
//!   on anything but its settings and input.

#![no_main]

mod common;

use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::{Application, Bandwidth, ForcedMode, OpusDecoder, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});

const BANDWIDTHS: [Bandwidth; 6] = [
    Bandwidth::Auto,
    Bandwidth::Narrowband,
    Bandwidth::Mediumband,
    Bandwidth::Wideband,
    Bandwidth::Superwideband,
    Bandwidth::Fullband,
];

/// One change of a setting, made on both encoders.
fn change(u: &mut Unstructured, encs: [&mut OpusEncoder; 2], channels: usize) -> Result<()> {
    match u.int_in_range(0..=13)? {
        0 | 1 => {
            let v = match u.int_in_range(0..=7)? {
                0 => u.arbitrary()?,
                1 => *u.choose(&[i32::MIN, -1, 0, 1, 499, 500, 3_000_000, i32::MAX])?,
                2 => u.int_in_range(500..=8_000)?,
                3 => u.int_in_range(100_000..=600_000)?,
                _ => u.int_in_range(6_000..=100_000)?,
            };
            for e in encs {
                e.bitrate_bps = v;
            }
        }
        2 => {
            let v = u.arbitrary()?;
            for e in encs {
                e.use_cbr = v;
            }
        }
        3 => {
            let v = if u.ratio(1, 8)? { u.arbitrary()? } else { u.int_in_range(0..=10)? };
            for e in encs {
                e.complexity = v;
            }
        }
        4 => {
            let mode = *u.choose(&[None, Some(ForcedMode::SilkOnly), Some(ForcedMode::Hybrid), Some(ForcedMode::CeltOnly)])?;
            for e in encs {
                e.set_force_mode(mode);
            }
        }
        5 => {
            let v = *u.choose(&BANDWIDTHS)?;
            for e in encs {
                e.set_bandwidth(v);
            }
        }
        6 => {
            // `Auto` is refused, which the encoder must survive.
            let v = *u.choose(&BANDWIDTHS)?;
            for e in encs {
                let _ = e.set_max_bandwidth(v);
            }
        }
        7 => {
            let n = *u.choose(&[None, Some(0), Some(1), Some(2), Some(3)])?;
            let ok = n.is_none_or(|n| (1..=channels).contains(&n));
            for e in encs {
                assert_eq!(e.set_force_channels(n).is_ok(), ok, "set_force_channels({n:?})");
            }
        }
        8 => {
            let fec = u.arbitrary()?;
            let loss = if u.ratio(1, 8)? { u.arbitrary()? } else { u.int_in_range(0..=100)? };
            for e in encs {
                e.use_inband_fec = fec;
                e.packet_loss_perc = loss;
            }
        }
        9 => {
            let v = u.ratio(1, 3)?;
            for e in encs {
                e.set_lfe(v);
            }
        }
        10 => {
            let mask: Option<Vec<f32>> = match u.int_in_range(0..=3)? {
                0 => None,
                // Values from the surround analysis lie about here.
                1 | 2 => {
                    let mut m = Vec::with_capacity(21 * channels);
                    for _ in 0..21 * channels {
                        m.push(u.int_in_range(-2000..=600)? as f32 / 100.0);
                    }
                    Some(m)
                }
                // Any bit pattern, and sometimes too short.
                _ => {
                    let n = if u.ratio(1, 4)? { u.int_in_range(0..=21 * channels)? } else { 21 * channels };
                    let mut m = Vec::with_capacity(n);
                    for _ in 0..n {
                        m.push(f32::from_bits(u.arbitrary()?));
                    }
                    Some(m)
                }
            };
            for e in encs {
                e.set_energy_mask(mask.as_deref());
            }
        }
        11 => {
            let v = u.arbitrary()?;
            for e in encs {
                e.set_prediction_disabled(v);
            }
        }
        12 => {
            for e in encs {
                let _ = e.enable_hybrid_mode();
            }
        }
        _ => {
            for e in encs {
                e.reset_state();
            }
        }
    }
    Ok(())
}

fn run(data: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(data);
    let fs = *u.choose(&RATES)?;
    let channels = u.int_in_range(1..=2)?;
    let app = *u.choose(&[Application::Voip, Application::Audio, Application::RestrictedLowDelay])?;
    let mut enc = OpusEncoder::new(fs, channels, app).unwrap();
    // The same settings, always given `f32` samples: for 16-bit input the
    // samples `s as f32 / 32768.0`. `encode_i16` promises the packets of
    // `encode` for those samples, and both encoders must be deterministic.
    let mut twin = OpusEncoder::new(fs, channels, app).unwrap();
    let mut c_dec = CDecoder::new(fs, channels);
    let mut dec = OpusDecoder::new(fs, channels).unwrap();
    let mut signal = Signal::new(&mut u)?;
    let sizes = encoder_frame_sizes(fs);
    let mut frame_size = *u.choose(&sizes)?;
    let mut out_len = 1500;
    let mut i16_in = false;

    let mut pcm = Vec::new();
    let mut pcm16 = Vec::new();
    let mut packet = vec![0u8; 4000];
    let mut twin_packet = vec![0u8; 4000];
    let mut c_out = vec![0f32; 2880 * 2];
    let mut r_out = vec![0f32; 2880 * 2];

    // No packet so far had a frame that the decoders conceal: the two
    // decoders' output must be close (see `assert_close`).
    let mut clean = true;
    let mut modes = Modes::default();
    let frames = u.int_in_range(1..=20)?;
    for n in 0..frames {
        if u.is_empty() {
            break;
        }
        for _ in 0..u.int_in_range(0..=3)? {
            match u.int_in_range(0..=9)? {
                0..=6 => change(&mut u, [&mut enc, &mut twin], channels)?,
                7 => frame_size = *u.choose(&sizes)?,
                8 => {
                    out_len = match u.int_in_range(0..=3)? {
                        0 => u.int_in_range(1..=12)?,
                        1 => u.int_in_range(13..=200)?,
                        _ => u.int_in_range(200..=4000)?,
                    }
                }
                _ => {
                    signal.change(&mut u)?;
                    i16_in = u.arbitrary()?;
                }
            }
        }
        // Sometimes a frame size the encoder must refuse.
        let bad_size = if u.ratio(1, 32)? {
            Some(*u.choose(&[0, 1, fs as usize / 1000, fs as usize * 3 / 100, fs as usize / 25 * 2, fs as usize / 25 * 3])?)
        } else {
            None
        };
        let size = bad_size.unwrap_or(frame_size);
        let extra = if signal_wants_bytes(&mut u)? {
            let k = u.int_in_range(0..=128)?;
            u.bytes(k)?
        } else {
            &[][..]
        };
        signal.frame(fs, channels, size, extra, &mut pcm);
        if u.ratio(1, 32)? && !pcm.is_empty() {
            // Input shorter than the frame: refused, and the encoder keeps
            // its state, which the twin encoder checks on the next frames.
            let r = enc.encode(&pcm[..pcm.len() - 1], size, &mut packet[..out_len]);
            assert!(r.is_err(), "frame {n}: short input was accepted");
        }
        let out = &mut packet[..out_len];
        let r = if i16_in {
            pcm16.clear();
            pcm16.extend(pcm.iter().map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0) as i16));
            pcm.clear();
            pcm.extend(pcm16.iter().map(|&x| x as f32 / 32768.0));
            enc.encode_i16(&pcm16, size, out)
        } else {
            enc.encode(&pcm, size, out)
        };
        let twin_out = &mut twin_packet[..out_len];
        let twin_r = twin.encode(&pcm, size, twin_out);
        match (&r, &twin_r) {
            (Ok(a), Ok(b)) => {
                assert_eq!(packet[..*a], twin_packet[..*b], "frame {n}: i16={i16_in}, the twin encoder's packet differs");
                assert_eq!(enc.final_range(), twin.final_range(), "frame {n}: the twin encoder's final range differs");
            }
            (Err(_), Err(_)) => {}
            _ => panic!("frame {n}: i16={i16_in}, encoder {r:?}, twin encoder {twin_r:?}"),
        }
        if bad_size.is_some() {
            assert!(r.is_err(), "frame {n}: frame size {size} at {fs} Hz was accepted");
            continue;
        }
        let len = match r {
            Ok(len) => len,
            Err(e) => panic!("frame {n}: encode of {size} samples into {out_len} bytes failed: {e}"),
        };
        assert!((1..=out_len).contains(&len), "frame {n}: packet of {len} bytes for a buffer of {out_len}");
        let p = &packet[..len];
        let range = enc.final_range();

        let got = c_dec.decode_float(p, &mut c_out, 2880, false);
        assert_eq!(got, size as i32, "frame {n}: libopus decodes {got} samples of a {size}-sample packet {p:02x?}");
        assert_eq!(c_dec.final_range(), range, "frame {n}: libopus final range, packet {p:02x?}");

        let got = dec.decode(p, 2880, &mut r_out[..2880 * channels]);
        assert_eq!(got, Ok(size), "frame {n}: opus1 decoder, packet {p:02x?}");
        assert_eq!(dec.final_range(), range, "frame {n}: opus1 decoder final range, packet {p:02x?}");
        clean &= whole_frames(p, 1) && !modes.leaves_celt(p, 1);
        modes.decoded(p, 1);
        if clean {
            let k = size * channels;
            assert_close(&format!("frame {n}, packet {p:02x?}"), &r_out[..k], &c_out[..k], 1e-10);
        }
    }
    Ok(())
}

/// Whether the next frame plays bytes of the input (the `Signal` kind that
/// uses them); asked here so that the input decides it.
fn signal_wants_bytes(u: &mut Unstructured) -> Result<bool> {
    u.ratio(1, 4)
}
