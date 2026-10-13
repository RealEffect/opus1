//! `CeltDecoder` with a channel count that differs from the packet's,
//! against C libopus.
//!
//! libopus decodes any stream with any decoder: a mono packet is written to
//! both channels of a stereo decoder, and a stereo packet is mixed down by a
//! mono decoder (`CELT_SET_CHANNELS`, and `celt_synthesis` in
//! celt/celt_decoder.c). `CeltDecoder::set_stream_channels` does the same.
//!
//! `OpusDecoder::decode` still rejects such packets, so these tests work on
//! the CELT decoder itself. The packets are CELT-only packets from the C
//! encoder with one frame each. Without the first byte (the table of
//! contents) such a packet is exactly one CELT frame. The reference is C
//! `opus_decode_float` of the whole packet in a decoder with the other
//! channel count.
//!
//! Run with `--nocapture` to see the measured numbers.

mod celtdec_common;
mod k24_common;
use celtdec_common::*;
use k24_common::*;
use opus_rs::celt::CeltDecoder;
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

/// Lowest SNR against libopus of any decoded (not concealed) frame, in dB.
/// A stream as a whole reaches about 135 dB, the same as with matching
/// channel counts. Single quiet frames are a little lower.
const DECODED_DB: f64 = 100.0;
/// Lowest SNR against libopus of a whole decoded stream, in dB.
const STREAM_DB: f64 = 125.0;

/// End of the CELT band range for the bandwidth in a table-of-contents
/// byte (libopus `opus_decode_frame`).
fn end_band(toc_byte: u8) -> usize {
    match toc(toc_byte).bandwidth {
        "NB" => 13,
        "WB" => 17,
        "SWB" => 19,
        _ => 21,
    }
}

/// Opus-rs CELT decoder with `channels` output channels, fed with whole
/// CELT-only packets.
struct RsCelt {
    dec: CeltDecoder,
    channels: usize,
    planar: Vec<f32>,
}

impl RsCelt {
    fn new(rate: usize, channels: usize) -> Self {
        Self {
            dec: CeltDecoder::new(default_mode(), channels, rate as i32),
            channels,
            planar: vec![0.0; 960 * channels],
        }
    }

    /// Decode one packet, or conceal a lost one (`None`), into `frame`
    /// samples per channel. Returns the interleaved samples.
    fn decode(&mut self, packet: Option<&[u8]>, frame: usize) -> Vec<f32> {
        let out = &mut self.planar[..frame * self.channels];
        let n = match packet {
            Some(p) => {
                let t = toc(p[0]);
                assert_eq!(
                    (t.mode, t.code),
                    ("CELT", 0),
                    "the test needs one CELT frame per packet"
                );
                self.dec
                    .set_stream_channels(if t.stereo { 2 } else { 1 })
                    .unwrap();
                let payload = &p[1..];
                let mut rc = RangeCoder::new_decoder(payload);
                self.dec.decode_from_range_coder_with_band_range(
                    &mut rc,
                    (payload.len() * 8) as i32,
                    frame,
                    out,
                    0,
                    end_band(p[0]),
                )
            }
            None => self.dec.decode_lost(frame, out, 0).unwrap(),
        };
        assert_eq!(n, frame);
        interleave(out, self.channels)
    }
}

fn signal(rate: usize, channels: usize, seconds: f64, music: bool) -> Vec<f32> {
    let mono = if music {
        multitone(rate, (rate as f64 * seconds) as usize)
    } else {
        speech_fullband(rate, seconds)
    };
    if channels == 2 {
        to_stereo(&mono, rate)
    } else {
        mono
    }
}

/// Stereo with the two channels mostly out of phase. At a low bitrate the
/// encoder then codes bands with inverted phase, which a mono decoder must
/// not apply (libopus `disable_inv`).
fn out_of_phase(rate: usize, seconds: f64) -> Vec<f32> {
    let mono = speech_fullband(rate, seconds);
    let mut seed = 99u32;
    let mut out = Vec::with_capacity(mono.len() * 2);
    for &s in &mono {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.01;
        out.push(s);
        out.push(-0.9 * s + noise);
    }
    out
}

fn celt_encoder(rate: usize, channels: usize, bandwidth: i32, bitrate: i32) -> CEnc {
    let mut enc = CEnc::audio(rate, channels);
    enc.force_mode(MODE_CELT)
        .bandwidth(bandwidth)
        .bitrate(bitrate);
    enc
}

/// Decode `packets` with a C decoder and an opus-rs CELT decoder, both with
/// `dec_channels` channels at `rate`. Checks the final range of every
/// packet and returns the lowest frame SNR and the SNR of the whole stream.
fn compare_stream(
    packets: &[Vec<u8>],
    rate: usize,
    dec_channels: usize,
    frame: usize,
    name: &str,
) -> (f64, f64) {
    let mut c_dec = CDecoder::new(rate, dec_channels);
    let mut rs_dec = RsCelt::new(rate, dec_channels);
    let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
    let mut worst = 200f64;
    for (i, p) in packets.iter().enumerate() {
        let c = c_dec.decode(Some(p), frame);
        let r = rs_dec.decode(Some(p), frame);
        assert_eq!(c.len(), r.len(), "{name}: packet {i}");
        assert_eq!(
            rs_dec.dec.final_range(),
            c_dec.final_range(),
            "{name}: final range after packet {i}"
        );
        // Frames that are silent in both decoders say nothing.
        if power(&c) > 1e-10 {
            worst = worst.min(snr(&c, &r));
        }
        c_all.extend(c);
        r_all.extend(r);
    }
    let whole = snr(&c_all, &r_all);
    println!("{name}: whole stream {whole:.1} dB, worst frame {worst:.1} dB");
    (worst, whole)
}

/// A mono stream in a two-channel decoder: both channels carry the mono
/// signal, as in libopus.
#[test]
fn mono_stream_in_a_stereo_decoder() {
    let mut failures = Vec::new();
    for music in [false, true] {
        let pcm = signal(48000, 1, 1.5, music);
        for enc_frame in [120usize, 240, 480, 960] {
            let packets = celt_encoder(48000, 1, BW_FB, 64000).encode_all(&pcm, enc_frame);
            assert!(packets.iter().all(|p| !toc(p[0]).stereo));
            for rate in [48000usize, 24000, 16000, 8000] {
                let frame = enc_frame * rate / 48000;
                let name = format!(
                    "mono {} at {} ms, stereo decoder at {rate} Hz",
                    if music { "music" } else { "speech" },
                    enc_frame as f64 / 48.0
                );
                let (worst, whole) = compare_stream(&packets, rate, 2, frame, &name);
                if worst < DECODED_DB || whole < STREAM_DB {
                    failures.push(format!(
                        "{name}: whole stream {whole:.1} dB, worst frame {worst:.1} dB"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The two output channels of a mono stream are identical.
#[test]
fn mono_stream_gives_two_equal_channels() {
    let pcm = signal(48000, 1, 0.5, false);
    let packets = celt_encoder(48000, 1, BW_FB, 64000).encode_all(&pcm, 960);
    let mut dec = RsCelt::new(48000, 2);
    for p in &packets {
        let out = dec.decode(Some(p), 960);
        assert!(out.chunks_exact(2).all(|s| s[0] == s[1]));
    }
}

/// A stereo stream in a one-channel decoder: the two channels are mixed
/// down, as in libopus.
#[test]
fn stereo_stream_in_a_mono_decoder() {
    let mut failures = Vec::new();
    for music in [false, true] {
        let pcm = signal(48000, 2, 1.5, music);
        for enc_frame in [120usize, 240, 480, 960] {
            let packets = celt_encoder(48000, 2, BW_FB, 96000).encode_all(&pcm, enc_frame);
            assert!(packets.iter().all(|p| toc(p[0]).stereo));
            for rate in [48000usize, 24000, 16000, 8000] {
                let frame = enc_frame * rate / 48000;
                let name = format!(
                    "stereo {} at {} ms, mono decoder at {rate} Hz",
                    if music { "music" } else { "speech" },
                    enc_frame as f64 / 48.0
                );
                let (worst, whole) = compare_stream(&packets, rate, 1, frame, &name);
                if worst < DECODED_DB || whole < STREAM_DB {
                    failures.push(format!(
                        "{name}: whole stream {whole:.1} dB, worst frame {worst:.1} dB"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Stereo with out-of-phase channels at a low bitrate. A mono decoder
/// decodes such a stream without the phase inversion, so that the two
/// channels do not cancel in the mix (libopus `disable_inv`). A stereo
/// decoder applies it. Both must match libopus.
#[test]
fn out_of_phase_stereo_in_both_decoders() {
    let pcm = out_of_phase(48000, 2.0);
    for (enc_frame, bitrate) in [(960usize, 24000), (480, 32000)] {
        let packets = celt_encoder(48000, 2, BW_FB, bitrate).encode_all(&pcm, enc_frame);
        for dec_channels in [1usize, 2] {
            let name = format!(
                "out-of-phase stereo, {} ms at {bitrate} b/s, {dec_channels}-channel decoder",
                enc_frame as f64 / 48.0
            );
            // The mix of two out-of-phase channels is quiet, so rounding
            // weighs more: the whole stream gets a lower limit here.
            let (worst, whole) = compare_stream(&packets, 48000, dec_channels, enc_frame, &name);
            assert!(
                worst > DECODED_DB && whole > STREAM_DB - 10.0,
                "{name}: whole stream {whole:.1} dB, worst frame {worst:.1} dB"
            );
        }
    }
}

/// The out-of-phase stream really uses phase inversion: the mono decoder's
/// output is not the plain mix of the stereo decoder's two channels.
#[test]
fn mono_decoder_does_not_apply_phase_inversion() {
    let pcm = out_of_phase(48000, 2.0);
    let packets = celt_encoder(48000, 2, BW_FB, 24000).encode_all(&pcm, 960);
    let (mut mono_dec, mut stereo_dec) = (RsCelt::new(48000, 1), RsCelt::new(48000, 2));
    let (mut mono, mut mixed) = (Vec::new(), Vec::new());
    for p in &packets {
        mono.extend(mono_dec.decode(Some(p), 960));
        mixed.extend(
            stereo_dec
                .decode(Some(p), 960)
                .chunks_exact(2)
                .map(|s| 0.5 * (s[0] + s[1])),
        );
    }
    let db = snr(&mono, &mixed);
    println!("mono decoder against the mix of the stereo decoder: {db:.1} dB");
    assert!(
        db < 30.0,
        "the stream has no phase inversion, the test is void ({db:.1} dB)"
    );
}

/// One stream whose packets change between mono and stereo, decoded by a
/// one-channel and by a two-channel decoder. The encoder gets stereo input
/// and is told every 100 ms to code one or two channels.
#[test]
fn stream_that_changes_between_mono_and_stereo() {
    for music in [false, true] {
        let pcm = signal(48000, 2, 3.0, music);
        for enc_frame in [240usize, 960] {
            let mut enc = celt_encoder(48000, 2, BW_FB, 96000);
            let per_switch = 4800 / enc_frame;
            let packets: Vec<Vec<u8>> = pcm
                .chunks_exact(enc_frame * 2)
                .enumerate()
                .map(|(i, f)| {
                    if i % per_switch == 0 {
                        enc.force_channels(if (i / per_switch) % 2 == 0 { 2 } else { 1 });
                    }
                    enc.encode(f)
                })
                .collect();
            let stereo_packets = packets.iter().filter(|p| toc(p[0]).stereo).count();
            assert!(
                stereo_packets > packets.len() / 4 && stereo_packets < packets.len() * 3 / 4,
                "expected a mix of mono and stereo packets, got {stereo_packets} of {}",
                packets.len()
            );
            for dec_channels in [1usize, 2] {
                for rate in [48000usize, 24000] {
                    let frame = enc_frame * rate / 48000;
                    let name = format!(
                        "changing channels, {} at {} ms, {dec_channels}-channel decoder at {rate} Hz",
                        if music { "music" } else { "speech" },
                        enc_frame as f64 / 48.0
                    );
                    let (worst, whole) = compare_stream(&packets, rate, dec_channels, frame, &name);
                    assert!(
                        worst > DECODED_DB && whole > STREAM_DB,
                        "{name}: whole stream {whole:.1} dB, worst frame {worst:.1} dB"
                    );
                }
            }
        }
    }
}

/// Narrower CELT bandwidths with the other channel count. The band range
/// ends at band 13 (narrowband), 17 (wideband) or 19 (super-wideband).
#[test]
fn narrower_bandwidths_with_the_other_channel_count() {
    for (rate, bw, bw_name) in [
        (8000usize, BW_NB, "NB"),
        (16000, BW_WB, "WB"),
        (24000, BW_SWB, "SWB"),
    ] {
        for stream_channels in [1usize, 2] {
            let pcm = signal(rate, stream_channels, 1.5, false);
            let frame = rate / 50;
            let packets = celt_encoder(rate, stream_channels, bw, 40000 * stream_channels as i32)
                .encode_all(&pcm, frame);
            assert!(packets.iter().all(|p| toc(p[0]).bandwidth == bw_name));
            let dec_channels = 3 - stream_channels;
            let name = format!(
                "CELT {bw_name}, {stream_channels}-channel stream, {dec_channels}-channel decoder"
            );
            let (worst, whole) = compare_stream(&packets, rate, dec_channels, frame, &name);
            assert!(
                worst > DECODED_DB && whole > STREAM_DB,
                "{name}: whole stream {whole:.1} dB, worst frame {worst:.1} dB"
            );
        }
    }
}

/// Lost packets in streams with the other channel count. The concealment
/// works on the decoder's own channels, whatever the stream had. The limits
/// are those of tests/celtdec_plc.rs for pitch repetition.
#[test]
fn losses_with_the_other_channel_count() {
    for (stream_channels, dec_channels) in [(1usize, 2usize), (2, 1)] {
        let pcm = signal(48000, stream_channels, 3.0, false);
        for frame in [240usize, 960] {
            let packets = celt_encoder(
                48000,
                stream_channels,
                BW_FB,
                64000 * stream_channels as i32,
            )
            .encode_all(&pcm, frame);
            let mut c_dec = CDecoder::new(48000, dec_channels);
            let mut rs_dec = RsCelt::new(48000, dec_channels);
            let (mut concealed, mut later) = (Vec::new(), Vec::new());
            let mut good_since_loss = 0usize;
            for (i, p) in packets.iter().enumerate() {
                // One loss every 100 ms or more, and a burst of seven
                // every second, which ends in noise for 20 ms frames.
                let period = (4800 / frame).max(10);
                let lost = i % period == period - 3
                    || (i % (48000 / frame) >= 40 && i % (48000 / frame) < 47);
                let packet = if lost { None } else { Some(p.as_slice()) };
                let c = c_dec.decode(packet, frame);
                let r = rs_dec.decode(packet, frame);
                let db = snr(&c, &r);
                if lost {
                    concealed.push(db);
                    good_since_loss = 0;
                } else {
                    good_since_loss += 1;
                    // Good frames that start 60 ms or more after a loss.
                    if (good_since_loss - 1) * frame >= 2880 {
                        later.push(db);
                    }
                }
            }
            concealed.sort_by(|a, b| a.total_cmp(b));
            later.sort_by(|a, b| a.total_cmp(b));
            let median = concealed[concealed.len() / 2];
            let poor = concealed.iter().filter(|d| **d < 40.0).count();
            let name = format!(
                "{stream_channels}-channel stream, {dec_channels}-channel decoder, {} ms",
                frame as f64 / 48.0
            );
            println!(
                "{name}: {} concealed frames, median {median:.1} dB, worst {:.1} dB, {poor} below 40 dB; good frames 60 ms after a loss: worst {:.1} dB",
                concealed.len(),
                concealed[0],
                later[0]
            );
            // Measured on 64-bit ARM. Other processors round the sums of
            // the concealment differently from libopus and get wider
            // limits, as in tests/celtdec_plc.rs.
            let arm64 = cfg!(target_arch = "aarch64");
            let (min_median, max_poor, min_later) = if arm64 {
                (65.0, 1 + concealed.len() / 50, 85.0)
            } else {
                (55.0, 1 + concealed.len() / 10, 60.0)
            };
            assert!(
                median > min_median,
                "{name}: median of the concealed frames {median:.1} dB"
            );
            assert!(
                poor <= max_poor,
                "{name}: {poor} concealed frames below 40 dB"
            );
            assert!(
                later[0] > min_later,
                "{name}: good frames after a loss {:.1} dB",
                later[0]
            );
        }
    }
}

/// Arguments that cannot be honoured give an error and leave the decoder
/// usable.
#[test]
fn bad_arguments_are_errors() {
    let mut dec = CeltDecoder::new(default_mode(), 2, 48000);
    assert_eq!(dec.final_range(), 0);
    assert_eq!(dec.stream_channels(), 2);
    assert!(dec.set_stream_channels(0).is_err());
    assert!(dec.set_stream_channels(3).is_err());
    assert_eq!(dec.stream_channels(), 2);
    assert!(dec.set_stream_channels(1).is_ok());
    assert_eq!(dec.stream_channels(), 1);

    let mut pcm = vec![0f32; 2 * 960];
    // Not a CELT frame size.
    assert!(dec.decode_lost(100, &mut pcm, 0).is_err());
    assert!(dec.decode_lost(0, &mut pcm, 0).is_err());
    assert!(dec.decode_lost(1920, &mut pcm, 0).is_err());
    // Output too small for two channels.
    assert!(dec.decode_lost(960, &mut pcm[..960], 0).is_err());
    // Not a band.
    assert!(dec.decode_lost(960, &mut pcm, 21).is_err());
    // Valid calls still work, for every frame size and both start bands.
    for frame in [120usize, 240, 480, 960] {
        assert_eq!(dec.decode_lost(frame, &mut pcm, 0), Ok(frame));
        assert_eq!(dec.decode_lost(frame, &mut pcm, 17), Ok(frame));
    }
    assert!(pcm.iter().all(|v| v.is_finite()));
}

/// Arbitrary bytes as a CELT frame, with every pairing of stream and
/// decoder channels, mixed with lost frames: no panic, finite output.
#[test]
fn garbage_frames_do_not_panic() {
    let mut seed = 0xC0FFEEu32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed >> 8
    };
    for dec_channels in [1usize, 2] {
        for rate in [48000i32, 16000, 8000] {
            let mut dec = CeltDecoder::new(default_mode(), dec_channels, rate);
            let mut pcm = vec![0f32; 960 * dec_channels];
            for _ in 0..400 {
                let frame48 = [120usize, 240, 480, 960][next() as usize % 4];
                let frame = frame48 * rate as usize / 48000;
                dec.set_stream_channels(1 + next() as usize % 2).unwrap();
                let start = if next() % 4 == 0 { 17 } else { 0 };
                let out = &mut pcm[..frame * dec_channels];
                if next() % 5 == 0 {
                    assert_eq!(dec.decode_lost(frame, out, start), Ok(frame));
                } else {
                    let len = next() as usize % 200;
                    let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
                    let end = [13usize, 17, 19, 21][next() as usize % 4].max(start + 1);
                    let mut rc = RangeCoder::new_decoder(&bytes);
                    let n = dec.decode_from_range_coder_with_band_range(
                        &mut rc,
                        (len * 8) as i32,
                        frame,
                        out,
                        start,
                        end,
                    );
                    assert_eq!(n, frame);
                }
                assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
            }
        }
    }
}
