//! The multistream encoder, checked with libopus's multistream decoder.
//!
//! The input chooses a sampling rate, an application and a layout: either
//! a mapping family of the Ogg Opus header (`new_surround`) or channels,
//! streams, coupled streams and a mapping (`new`). libopus's
//! `opus_multistream_surround_encoder_create` and
//! `opus_multistream_encoder_create` must accept and refuse the same
//! layouts, and for a mapping family choose the same streams and mapping.
//!
//! Then the input gives frames, and between them changes the bitrate
//! (automatic, maximum or a value), VBR or CBR, the complexity, in-band
//! forward error correction (FEC) and the expected loss, the bandwidth and
//! the maximum bandwidth, the forced mode, discontinuous transmission
//! (DTX), settings of single streams, the frame size, the output buffer,
//! the signal and its format, or resets the encoder. What must hold for every frame:
//!
//! - With an output buffer of 4000 bytes per stream the frame is encoded.
//! - Every packet is accepted by libopus's multistream decoder, which
//!   decodes exactly one frame and ends in the encoder's final range.
//! - With a constant bitrate and an explicit or the maximum bitrate, the
//!   packet has the length libopus's rule gives.
//! - opus1's multistream decoder does the same. Its output is within 50 dB
//!   of libopus's until a packet has a frame of 0 or 1 bytes, which the
//!   decoders conceal, or switches a stream from CELT-only to SILK or
//!   Hybrid.
//! - A second encoder with the same layout and settings, always given
//!   `f32` samples (for 16-bit input the samples `s as f32 / 32768.0`,
//!   coded with an LSB depth of 16), writes the same packets.

#![no_main]

mod common;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, InbandFec, SampleRate};

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});

/// The bandwidth settings: `None` is the automatic choice.
const BANDWIDTHS: [Option<Bandwidth>; 6] = [
    None,
    Some(Bandwidth::Narrowband),
    Some(Bandwidth::Mediumband),
    Some(Bandwidth::Wideband),
    Some(Bandwidth::Superwideband),
    Some(Bandwidth::Fullband),
];

/// The libopus application constant.
fn c_app(app: Application) -> i32 {
    app.to_raw()
}

/// Builds the encoder, and checks that libopus accepts the same layouts.
fn build(u: &mut Unstructured, fs: i32, app: Application) -> Result<Option<MultistreamEncoder>> {
    if u.arbitrary()? {
        let family = *u.choose(&[0u8, 1, 1, 1, 2, 2, 3, 255, 255])?;
        let channels = if u.ratio(1, 16)? { u.int_in_range(0..=18)? } else { u.int_in_range(1..=8)? };
        let rust = MultistreamEncoder::new_surround(
            SampleRate::try_from(fs).unwrap(),
            channels,
            family,
            app,
        );
        let (mut streams, mut coupled) = (0, 0);
        let mut mapping = [0u8; 256];
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_surround_encoder_create(
                fs,
                channels as i32,
                family as i32,
                &mut streams,
                &mut coupled,
                mapping.as_mut_ptr(),
                c_app(app),
                &mut err,
            )
        };
        let c_ok = err == 0 && !st.is_null();
        if !st.is_null() {
            unsafe { sys::opus_multistream_encoder_destroy(st) };
        }
        match rust {
            Ok(enc) => {
                assert!(c_ok, "new_surround({channels}, family {family}) accepted, libopus refuses ({err})");
                let l = enc.layout();
                assert_eq!(
                    (l.channels(), l.streams(), l.coupled_streams(), l.mapping()),
                    (channels, streams as usize, coupled as usize, &mapping[..channels]),
                    "new_surround({channels}, family {family}) layout"
                );
                Ok(Some(enc))
            }
            Err(e) => {
                assert!(!c_ok, "new_surround({channels}, family {family}) refused ({e}), libopus accepts");
                Ok(None)
            }
        }
    } else {
        let channels = u.int_in_range(0..=10)?;
        let streams = u.int_in_range(0..=channels + 1)?;
        let coupled = u.int_in_range(0..=streams + 1)?;
        let mut mapping = vec![0u8; channels.max(1)];
        for m in mapping.iter_mut() {
            *m = match u.int_in_range(0..=9)? {
                0 => 255,
                1 => u.arbitrary()?,
                _ => u.int_in_range(0..=(streams + coupled).max(1) as u8 - 1)?,
            };
        }
        let rust = MultistreamEncoder::new(
            SampleRate::try_from(fs).unwrap(),
            channels,
            streams,
            coupled,
            &mapping,
            app,
        );
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_encoder_create(
                fs,
                channels as i32,
                streams as i32,
                coupled as i32,
                mapping.as_ptr(),
                c_app(app),
                &mut err,
            )
        };
        let c_ok = err == 0 && !st.is_null();
        if !st.is_null() {
            unsafe { sys::opus_multistream_encoder_destroy(st) };
        }
        assert_eq!(
            rust.is_ok(),
            c_ok,
            "new({channels}, {streams}, {coupled}, {mapping:?}): opus1 {:?}, libopus {err}",
            rust.as_ref().err()
        );
        Ok(rust.ok())
    }
}

/// One change of a setting, made on both encoders.
fn change(u: &mut Unstructured, encs: [&mut MultistreamEncoder; 2]) -> Result<()> {
    match u.int_in_range(0..=11)? {
        0 | 1 => {
            let b = match u.int_in_range(0..=5)? {
                0 => Bitrate::Auto,
                1 => Bitrate::Max,
                2 => Bitrate::BitsPerSecond(u.arbitrary()?),
                3 => Bitrate::BitsPerSecond(NonZeroU32::new(u.int_in_range(500..=20_000)?).unwrap()),
                _ => Bitrate::BitsPerSecond(NonZeroU32::new(u.int_in_range(20_000..=1_000_000)?).unwrap()),
            };
            for e in encs {
                e.set_bitrate(b);
            }
        }
        2 => {
            let v = u.arbitrary()?;
            for e in encs {
                e.set_vbr(v);
            }
        }
        3 => {
            let v: u8 = if u.ratio(1, 8)? { u.arbitrary()? } else { u.int_in_range(0..=10)? };
            for e in encs {
                let _ = e.set_complexity(v);
            }
        }
        4 => {
            let fec = *u.choose(&[InbandFec::Off, InbandFec::On, InbandFec::OnUnlessMusic])?;
            let loss = u.int_in_range(0..=101)?;
            for e in encs {
                e.set_inband_fec(fec);
                let _ = e.set_packet_loss_perc(loss);
            }
        }
        5 => {
            let v = *u.choose(&BANDWIDTHS)?;
            for e in encs {
                e.set_bandwidth(v);
            }
        }
        6 => {
            // `None` leaves the maximum as it is.
            if let Some(v) = *u.choose(&BANDWIDTHS)? {
                for e in encs {
                    e.set_max_bandwidth(v);
                }
            }
        }
        7 => {
            let mode = *u.choose(&[None, Some(ForcedMode::SilkOnly), Some(ForcedMode::Hybrid), Some(ForcedMode::CeltOnly)])?;
            for e in encs {
                e.set_force_mode(mode);
            }
        }
        8 => {
            // Settings of one stream that the multistream encoder does not
            // forward.
            let s = u.int_in_range(0..=encs[0].layout().streams() - 1)?;
            let channels = *u.choose(&[None, Some(Channels::Mono), Some(Channels::Stereo)])?;
            let disabled = u.arbitrary()?;
            for e in encs {
                if let Some(stream) = e.stream_encoder_mut(s) {
                    stream.set_force_channels(channels);
                    stream.set_prediction_disabled(disabled);
                }
            }
        }
        9 => {
            let v = u.arbitrary()?;
            for e in encs {
                e.set_dtx(v);
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
    let app = *u.choose(&[Application::Voip, Application::Audio, Application::RestrictedLowDelay])?;
    // The twin is built from the same bytes as the encoder, so it gets the
    // same layout.
    let mut twin_u = Unstructured::new(u.peek_bytes(u.len()).unwrap_or(&[]));
    let Some(mut enc) = build(&mut u, fs, app)? else {
        return Ok(());
    };
    let mut twin = build(&mut twin_u, fs, app)?.expect("the twin encoder was not built");
    let layout = enc.layout().clone();
    let (channels, streams) = (layout.channels(), layout.streams());
    let mut c_dec = CMsDecoder::new(fs, channels, streams, layout.coupled_streams(), layout.mapping())
        .unwrap_or_else(|e| panic!("libopus refuses the decoder for the encoder's layout {layout:?}: {e}"));
    let mut dec = MultistreamDecoder::with_layout(
        SampleRate::try_from(fs).unwrap(),
        channels,
        streams,
        layout.coupled_streams(),
        layout.mapping(),
    )
    .expect("opus1 refuses the decoder for the encoder's layout");

    let mut signal = Signal::new(&mut u)?;
    let sizes = encoder_frame_sizes(fs);
    let mut frame_size = *u.choose(&sizes)?;
    let big = 4000 * streams;
    let mut out_len = big;
    let mut i16_in = false;
    let mut pcm = Vec::new();
    let mut pcm16 = Vec::new();
    let mut packet = vec![0u8; big.max(8000)];
    let mut twin_packet = vec![0u8; big.max(8000)];
    let mut c_out = vec![0f32; 5760 * channels];
    let mut r_out = vec![0f32; 5760 * channels];

    // No packet so far had a frame that the decoders conceal: the two
    // decoders' output must be close (see `assert_close`).
    let mut clean = true;
    let mut modes = Modes::default();
    let frames = u.int_in_range(1..=10)?;
    for n in 0..frames {
        if u.is_empty() {
            break;
        }
        for _ in 0..u.int_in_range(0..=3)? {
            match u.int_in_range(0..=7)? {
                0..=4 => change(&mut u, [&mut enc, &mut twin])?,
                5 => frame_size = *u.choose(&sizes)?,
                6 => {
                    out_len = match u.int_in_range(0..=3)? {
                        0 => u.int_in_range(1..=4 * streams + 4)?,
                        1 => u.int_in_range(1..=8000)?,
                        _ => big,
                    }
                }
                _ => {
                    signal.change(&mut u)?;
                    i16_in = u.arbitrary()?;
                }
            }
        }
        signal.frame(fs, channels, frame_size, &[], &mut pcm);
        if u.ratio(1, 32)? && !pcm.is_empty() {
            // Input shorter than the frame: refused, and the encoder keeps
            // its state, which the twin encoder checks on the next frames.
            let r = enc.encode(&pcm[..pcm.len() - 1], frame_size, &mut packet[..out_len]);
            assert!(r.is_err(), "frame {n}: short input was accepted");
        }
        let out = &mut packet[..out_len];
        let r = if i16_in {
            pcm16.clear();
            pcm16.extend(pcm.iter().map(|&x| (x * 32768.0).clamp(-32768.0, 32767.0) as i16));
            pcm.clear();
            pcm.extend(pcm16.iter().map(|&x| x as f32 / 32768.0));
            enc.encode_i16(&pcm16, frame_size, out)
        } else {
            enc.encode(&pcm, frame_size, out)
        };
        // `encode_i16` codes with an LSB depth of at most 16, as libopus's
        // `opus_multistream_encode` does.
        twin.set_lsb_depth(if i16_in { 16 } else { 24 }).unwrap();
        let twin_r = twin.encode(&pcm, frame_size, &mut twin_packet[..out_len]);
        match (&r, &twin_r) {
            (Ok(a), Ok(b)) => {
                assert_eq!(packet[..*a], twin_packet[..*b], "frame {n}: i16={i16_in}, the twin encoder's packet differs");
                assert_eq!(enc.final_range(), twin.final_range(), "frame {n}: the twin encoder's final range differs");
            }
            (Err(_), Err(_)) => {}
            _ => panic!("frame {n}: i16={i16_in}, encoder {r:?}, twin encoder {twin_r:?}"),
        }
        let len = match r {
            Ok(len) => len,
            Err(e) => {
                assert!(out_len < big, "frame {n}: encode of {frame_size} samples into {out_len} bytes failed: {e}");
                continue;
            }
        };
        assert!(len <= out_len);
        let p = &packet[..len];
        // libopus's rule for a constant bitrate (opus_multistream_encoder.c
        // 918-928): the packet has the bytes of the bitrate, rounded to the
        // nearest byte, at least the smallest packet (one byte for the last
        // stream, two for each other), at most the buffer; at the maximum
        // bitrate it fills the buffer. The last stream is padded to that
        // length. The automatic bitrate goes by the allocation, which is not
        // known here.
        if !enc.vbr() {
            let smallest = 2 * streams - 1;
            let expected = match enc.bitrate_setting() {
                Bitrate::Auto => None,
                Bitrate::Max => Some(out_len),
                Bitrate::BitsPerSecond(bps) => {
                    let bytes = ((bitrate_to_bits(bps.get() as i32, fs, frame_size) + 4) / 8).max(0) as usize;
                    Some(out_len.min(smallest.max(bytes)))
                }
            };
            if let Some(expected) = expected {
                assert_eq!(
                    len,
                    expected,
                    "frame {n}: constant bitrate {:?}, {frame_size} samples, {out_len}-byte buffer",
                    enc.bitrate_setting()
                );
            }
        }
        let range = enc.final_range();

        let got = c_dec.decode_float(p, &mut c_out, 5760, false);
        assert_eq!(got, frame_size as i32, "frame {n}: libopus decodes {got} samples, packet {p:02x?}");
        assert_eq!(c_dec.final_range(), range, "frame {n}: libopus final range, packet {p:02x?}");

        let got = dec.decode(p, 5760, &mut r_out);
        assert_eq!(got, Ok(frame_size), "frame {n}: opus1 decoder, packet {p:02x?}");
        assert_eq!(dec.final_range(), range, "frame {n}: opus1 decoder final range");
        clean &= whole_frames(p, streams) && !modes.leaves_celt(p, streams);
        modes.decoded(p, streams);
        if clean {
            let k = frame_size * channels;
            assert_close(&format!("frame {n}, packet {p:02x?}"), &r_out[..k], &c_out[..k], 1e-10);
        }
    }
    Ok(())
}
