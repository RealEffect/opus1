//! The controls of the opus-rs multistream encoder, against C libopus.
//!
//! - Settings changed in a running stream: every packet still decodes in C
//!   with the encoder's final range, and the change shows in the packets.
//! - Every setting reaches every stream encoder, and the getters report it.
//! - Arguments and layouts libopus rejects are rejected, and those it
//!   accepts are accepted.
//! - Output buffers down to the smallest possible one.
//! - `reset_state`, `final_range`, `stream_encoder`.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use audiopus_sys as sys;
use ms_support::Bandwidth as TocBandwidth;
use ms_support::*;
use multistream_enc_common::*;
use opus_rs::multistream::encoder::{Bitrate, MultistreamEncoder};
use opus_rs::{Application, ForcedMode};

/// Whether a stream of the packet is stereo and in Hybrid mode. The final
/// range of such a packet is not compared: see the note on the fault in
/// src/celt.rs at the top of tests/multistream_enc_validity.rs.
fn has_stereo_hybrid_stream(packet: &[u8], streams: usize) -> bool {
    split_streams(packet, streams).is_ok_and(|parts| {
        parts.iter().any(|p| {
            let t = toc(p.toc);
            t.mode == Mode::Hybrid && t.stereo
        })
    })
}

/// Encodes `frames` frames of `pcm`, starting at frame `*next`, and checks
/// every packet with the C decoder. Returns the packets.
fn encode_checked(
    enc: &mut MultistreamEncoder,
    dec: &mut CMsDecoder,
    pcm: &[f32],
    next: &mut usize,
    frames: usize,
    frame_size: usize,
    what: &str,
) -> Vec<Vec<u8>> {
    let channels = enc.layout().channels();
    let streams = enc.layout().streams();
    let mut out = vec![0u8; max_packet_bytes(streams)];
    let mut packets = Vec::new();
    for _ in 0..frames {
        let start = *next;
        *next += frame_size * channels;
        let n = enc
            .encode(&pcm[start..*next], frame_size, &mut out)
            .unwrap_or_else(|e| panic!("{what}: encode failed: {e}"));
        let packet = &out[..n];
        let decoded = dec
            .decode_float(Some(packet), frame_size, false)
            .unwrap_or_else(|e| {
                panic!(
                    "{what}: C decode failed: {e}: {}",
                    describe_packet(packet, streams)
                )
            });
        assert_eq!(
            decoded.len(),
            frame_size * channels,
            "{what}: decoded length"
        );
        if !has_stereo_hybrid_stream(packet, streams) {
            assert_eq!(
                enc.final_range(),
                dec.final_range(),
                "{what}: final range: {}",
                describe_packet(packet, streams)
            );
        }
        packets.push(packet.to_vec());
    }
    packets
}

/// Average packet size in bytes.
fn mean_len(packets: &[Vec<u8>]) -> f64 {
    packets.iter().map(|p| p.len()).sum::<usize>() as f64 / packets.len() as f64
}

#[test]
fn settings_change_in_a_running_surround_stream() {
    let fs = 48_000;
    for case in SURROUND_CASES {
        let channels = case.channels as i32;
        let mut enc = rust_encoder(case, fs, Application::Audio);
        let mut dec = c_decoder(&enc);
        let pcm = voiced(fs, case.channels, 4 * fs as usize);
        let mut next = 0;
        let name = case.name;

        // Bitrate up and down, in 20 ms frames.
        enc.set_bitrate(Bitrate::BitsPerSecond(64_000 * channels))
            .unwrap();
        let high = encode_checked(&mut enc, &mut dec, &pcm, &mut next, 20, 960, name);
        enc.set_bitrate(Bitrate::BitsPerSecond(16_000 * channels))
            .unwrap();
        let low = encode_checked(&mut enc, &mut dec, &pcm, &mut next, 20, 960, name);
        assert!(
            mean_len(&low) < 0.4 * mean_len(&high),
            "{name}: a quarter of the bitrate gives {:.0} bytes after {:.0}",
            mean_len(&low),
            mean_len(&high)
        );

        // Constant bitrate, then another frame size, then another bitrate.
        enc.set_vbr(false);
        assert!(!enc.vbr());
        for p in encode_checked(&mut enc, &mut dec, &pcm, &mut next, 10, 960, name) {
            assert_eq!(p.len() as i32, 16_000 * channels / 400, "{name}: CBR 20 ms");
        }
        for p in encode_checked(&mut enc, &mut dec, &pcm, &mut next, 10, 480, name) {
            assert_eq!(p.len() as i32, 16_000 * channels / 800, "{name}: CBR 10 ms");
        }
        enc.set_bitrate(Bitrate::BitsPerSecond(40_000 * channels))
            .unwrap();
        for p in encode_checked(&mut enc, &mut dec, &pcm, &mut next, 4, 2880, name) {
            assert_eq!(
                p.len() as i32,
                40_000 * channels * 3 / 400,
                "{name}: CBR 60 ms"
            );
        }
        for p in encode_checked(&mut enc, &mut dec, &pcm, &mut next, 20, 120, name) {
            assert_eq!(
                p.len() as i32,
                40_000 * channels / 3200,
                "{name}: CBR 2.5 ms"
            );
        }

        // Back to variable bitrate, with the automatic and the maximum
        // bitrate.
        enc.set_vbr(true);
        enc.set_bitrate(Bitrate::Auto).unwrap();
        encode_checked(&mut enc, &mut dec, &pcm, &mut next, 10, 960, name);
        enc.set_bitrate(Bitrate::Max).unwrap();
        let max = encode_checked(&mut enc, &mut dec, &pcm, &mut next, 5, 960, name);
        assert!(mean_len(&max) > mean_len(&high), "{name}: maximum bitrate");

        // Complexity, forward error correction and expected loss.
        enc.set_bitrate(Bitrate::BitsPerSecond(24_000 * channels))
            .unwrap();
        for complexity in [0, 10, 5] {
            enc.set_complexity(complexity).unwrap();
            assert_eq!(enc.complexity(), complexity);
            encode_checked(&mut enc, &mut dec, &pcm, &mut next, 8, 960, name);
        }
        enc.set_inband_fec(true);
        enc.set_packet_loss_perc(20).unwrap();
        assert!(enc.inband_fec());
        assert_eq!(enc.packet_loss_perc(), 20);
        encode_checked(&mut enc, &mut dec, &pcm, &mut next, 10, 960, name);
        enc.set_inband_fec(false);
        enc.set_packet_loss_perc(0).unwrap();

        // In surround mode the encoder sets the bandwidth of every stream
        // on every frame. A bandwidth the caller asks for does not hold.
        enc.set_bandwidth(opus_rs::Bandwidth::Narrowband);
        assert_eq!(enc.bandwidth(), opus_rs::Bandwidth::Narrowband);
        let streams = enc.layout().streams();
        for p in encode_checked(&mut enc, &mut dec, &pcm, &mut next, 5, 960, name) {
            let parts = split_streams(&p, streams).unwrap();
            for (s, part) in parts.iter().enumerate() {
                let want = if Some(s) == enc.lfe_stream() {
                    TocBandwidth::Nb
                } else {
                    TocBandwidth::Fb
                };
                assert_eq!(toc(part.toc).bandwidth, want, "{name}: stream {s}");
            }
        }
        enc.set_bandwidth(opus_rs::Bandwidth::Auto);

        // A reset in the middle of the stream.
        enc.reset_state();
        encode_checked(&mut enc, &mut dec, &pcm, &mut next, 10, 960, name);
        assert_eq!(
            enc.bitrate_setting(),
            Bitrate::BitsPerSecond(24_000 * channels)
        );
    }
}

#[test]
fn bandwidth_settings_reach_the_streams_of_a_plain_layout() {
    let fs = 48_000;
    let frame_size = 960;
    // Three mono streams at a bitrate where each one is CELT fullband.
    let case = Case::family("family 255, 3 channels", 3, 255);
    let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
    configure(&mut rust, &mut c, 3 * 64_000, true);
    let mut dec = c_decoder(&rust);
    let pcm = voiced(fs, 3, 2 * fs as usize);
    let mut next = 0;

    // Each step: the opus-rs setting, the same setting on the C encoder,
    // and the bandwidth every stream must signal afterwards.
    use opus_rs::Bandwidth as Bw;
    let steps: [(&str, Option<Bw>, Option<Bw>, TocBandwidth); 6] = [
        ("default", None, None, TocBandwidth::Fb),
        ("wideband", Some(Bw::Wideband), None, TocBandwidth::Wb),
        ("narrowband", Some(Bw::Narrowband), None, TocBandwidth::Nb),
        ("automatic again", Some(Bw::Auto), None, TocBandwidth::Fb),
        (
            "at most super-wideband",
            None,
            Some(Bw::Superwideband),
            TocBandwidth::Swb,
        ),
        (
            "at most fullband",
            None,
            Some(Bw::Fullband),
            TocBandwidth::Fb,
        ),
    ];
    for (what, bandwidth, max_bandwidth, want) in steps {
        if let Some(bw) = bandwidth {
            rust.set_bandwidth(bw);
            assert_eq!(rust.bandwidth(), bw);
            c.ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bw as i32);
        }
        if let Some(bw) = max_bandwidth {
            rust.set_max_bandwidth(bw).unwrap();
            assert_eq!(rust.max_bandwidth(), bw);
            c.ctl(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw as i32);
        }
        let start = next;
        let packets = encode_checked(&mut rust, &mut dec, &pcm, &mut next, 10, frame_size, what);
        let c_packets = c.encode_all(&pcm[start..next], frame_size);
        for (f, (p, cp)) in packets.iter().zip(&c_packets).enumerate() {
            let parts = split_streams(p, 3).unwrap();
            let c_parts = split_streams(cp, 3).unwrap();
            for s in 0..3 {
                let (t, ct) = (toc(parts[s].toc), toc(c_parts[s].toc));
                assert_eq!(t.bandwidth, want, "{what}: frame {f}: stream {s} is {t}");
                assert_eq!(
                    ct.bandwidth, want,
                    "{what}: frame {f}: C stream {s} is {ct}"
                );
            }
        }
    }
    assert!(rust.set_max_bandwidth(Bw::Auto).is_err());
    assert_eq!(rust.max_bandwidth(), Bw::Fullband);
    assert_ne!(c.try_ctl(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, OPUS_AUTO), 0);
}

#[test]
fn forced_mode_reaches_the_streams_of_a_plain_layout() {
    let fs = 48_000;
    let frame_size = 960;
    let case = Case::generic("generic 4 channels", 4, 3, 1, &[2, 0, 1, 3]);
    let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
    configure(&mut rust, &mut c, 4 * 24_000, true);
    let mut dec = c_decoder(&rust);
    let pcm = voiced(fs, 4, 2 * fs as usize);
    let mut next = 0;
    for (mode, c_mode, want) in [
        (Some(ForcedMode::CeltOnly), MODE_CELT_ONLY, Some(Mode::Celt)),
        (Some(ForcedMode::Hybrid), MODE_HYBRID, Some(Mode::Hybrid)),
        (
            Some(ForcedMode::SilkOnly),
            MODE_SILK_ONLY,
            Some(Mode::Hybrid),
        ),
        (None, OPUS_AUTO, None),
    ] {
        rust.set_force_mode(mode);
        c.ctl(OPUS_SET_FORCE_MODE_REQUEST, c_mode);
        let start = next;
        let what = format!("{mode:?}");
        let packets = encode_checked(&mut rust, &mut dec, &pcm, &mut next, 10, frame_size, &what);
        let c_packets = c.encode_all(&pcm[start..next], frame_size);
        let Some(want) = want else { continue };
        // SILK-only at fullband is Hybrid in both encoders. The first
        // packet after a change is left out: libopus codes one more packet
        // in the old mode when it leaves or enters CELT-only.
        for (f, (p, cp)) in packets.iter().zip(&c_packets).enumerate().skip(1) {
            for (s, (part, c_part)) in split_streams(p, 3)
                .unwrap()
                .iter()
                .zip(&split_streams(cp, 3).unwrap())
                .enumerate()
            {
                assert_eq!(toc(part.toc).mode, want, "{what}: frame {f}: stream {s}");
                assert_eq!(
                    toc(c_part.toc).mode,
                    want,
                    "{what}: frame {f}: C stream {s}"
                );
            }
        }
    }
}

#[test]
fn settings_reach_every_stream_and_the_getters_report_them() {
    for case in CASES {
        let (mut rust, mut c) = encoders(case, 48_000, Application::Audio);
        let streams = c.streams;
        // The values of a new encoder are those of libopus.
        assert_eq!(rust.complexity(), c.get(sys::OPUS_GET_COMPLEXITY_REQUEST));
        assert_eq!(rust.vbr() as i32, c.get(sys::OPUS_GET_VBR_REQUEST));
        assert_eq!(
            rust.inband_fec() as i32,
            c.get(sys::OPUS_GET_INBAND_FEC_REQUEST)
        );
        assert_eq!(
            rust.packet_loss_perc(),
            c.get(sys::OPUS_GET_PACKET_LOSS_PERC_REQUEST)
        );
        assert_eq!(
            rust.sampling_rate(),
            c.get(sys::OPUS_GET_SAMPLE_RATE_REQUEST)
        );
        assert_eq!(rust.lookahead() as i32, c.lookahead());
        assert_eq!(rust.bitrate_setting(), Bitrate::Auto);
        assert_eq!(rust.bandwidth(), opus_rs::Bandwidth::Auto);
        assert_eq!(rust.max_bandwidth(), opus_rs::Bandwidth::Fullband);
        assert_eq!(rust.application(), Application::Audio);

        rust.set_complexity(3).unwrap();
        rust.set_vbr(false);
        rust.set_inband_fec(true);
        rust.set_packet_loss_perc(17).unwrap();
        c.set_complexity(3)
            .set_vbr(false)
            .set_inband_fec(true)
            .set_packet_loss_perc(17);
        assert_eq!(rust.complexity(), 3);
        assert!(!rust.vbr());
        assert!(rust.inband_fec());
        assert_eq!(rust.packet_loss_perc(), 17);
        for s in 0..streams {
            let enc = rust.stream_encoder(s).unwrap();
            assert_eq!(
                enc.complexity,
                c.stream_get(s, sys::OPUS_GET_COMPLEXITY_REQUEST)
            );
            assert_eq!(
                !enc.use_cbr as i32,
                c.stream_get(s, sys::OPUS_GET_VBR_REQUEST)
            );
            assert_eq!(
                enc.use_inband_fec as i32,
                c.stream_get(s, sys::OPUS_GET_INBAND_FEC_REQUEST)
            );
            assert_eq!(
                enc.packet_loss_perc,
                c.stream_get(s, sys::OPUS_GET_PACKET_LOSS_PERC_REQUEST)
            );
            assert_eq!(enc.channels(), if s < c.coupled { 2 } else { 1 });
            assert_eq!(enc.sampling_rate(), 48_000);
            // Only the stream of the LFE channel has the LFE flag.
            assert_eq!(
                enc.lfe(),
                Some(s) == rust.lfe_stream(),
                "{}: stream {s}",
                case.name
            );
        }
        // No stream beyond the last. libopus: OPUS_BAD_ARG.
        assert!(rust.stream_encoder(streams).is_none());
        assert!(rust.stream_encoder_mut(streams).is_none());
        assert!(rust.stream_encoder_mut(streams - 1).is_some());
    }
}

#[test]
fn out_of_range_settings_are_rejected_like_libopus() {
    let case = Case::family("5.1", 6, 1);
    let (mut rust, mut c) = encoders(&case, 48_000, Application::Audio);
    for complexity in [-1000, -1, 0, 1, 9, 10, 11, 1000] {
        let ok = c.try_ctl(sys::OPUS_SET_COMPLEXITY_REQUEST, complexity) == 0;
        assert_eq!(
            rust.set_complexity(complexity).is_ok(),
            ok,
            "complexity {complexity}"
        );
        assert_eq!(rust.complexity(), c.get(sys::OPUS_GET_COMPLEXITY_REQUEST));
    }
    for percent in [-1000, -1, 0, 1, 50, 100, 101, 1000] {
        let ok = c.try_ctl(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, percent) == 0;
        assert_eq!(
            rust.set_packet_loss_perc(percent).is_ok(),
            ok,
            "loss {percent}"
        );
        assert_eq!(
            rust.packet_loss_perc(),
            c.get(sys::OPUS_GET_PACKET_LOSS_PERC_REQUEST)
        );
    }
    // A bitrate must be positive. The automatic and the maximum bitrate
    // have their own values in opus-rs, so the numbers libopus uses for
    // them, -1000 and -1, are not bitrates here.
    for bitrate in [
        i32::MIN,
        -5000,
        -1001,
        -2,
        0,
        1,
        499,
        2999,
        3000,
        4_500_000,
        4_500_001,
        i32::MAX,
    ] {
        let ok = c.try_ctl(sys::OPUS_SET_BITRATE_REQUEST, bitrate) == 0;
        assert_eq!(
            rust.set_bitrate(Bitrate::BitsPerSecond(bitrate)).is_ok(),
            ok,
            "bitrate {bitrate}"
        );
        if ok {
            // 500 to 750 000 bits per second per channel.
            let limited = bitrate.clamp(3000, 4_500_000);
            assert_eq!(rust.bitrate_setting(), Bitrate::BitsPerSecond(limited));
        }
    }
    for bitrate in [-1000, -1] {
        assert!(rust.set_bitrate(Bitrate::BitsPerSecond(bitrate)).is_err());
    }
    // The limits are those of libopus: both encoders give their streams
    // the same bitrates at either end.
    let pcm = voiced(48_000, 6, 960);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    for bitrate in [1, i32::MAX] {
        configure(&mut rust, &mut c, bitrate, true);
        rust.encode(&pcm, 960, &mut out).unwrap();
        c.encode_float(&pcm, 960, out.len()).unwrap();
        for s in 0..c.streams {
            assert_eq!(
                rust.stream_encoder(s).unwrap().bitrate_bps.min(510_400),
                c.stream_bitrate(s),
                "bitrate {bitrate}: stream {s}"
            );
        }
    }
}

/// The sampling rates, channel counts and mapping families the surround
/// constructor takes.
#[test]
fn surround_constructor_accepts_what_libopus_accepts() {
    let mut accepted = 0;
    for fs in [
        8_000, 12_000, 16_000, 24_000, 48_000, 0, -48_000, 44_100, 96_000, 1,
    ] {
        for family in [0u8, 1, 2, 3, 4, 254, 255] {
            for channels in (0..=12).chain([16, 18, 25, 27]) {
                let c = CMsEncoder::surround(fs, channels, family as i32, APP_AUDIO);
                let rust =
                    MultistreamEncoder::new_surround(fs, channels, family, Application::Audio);
                assert_eq!(
                    rust.is_ok(),
                    c.is_ok(),
                    "{fs} Hz, {channels} channels, family {family}: opus-rs {:?}, C {:?}",
                    rust.as_ref().err(),
                    c.as_ref().err()
                );
                if let (Ok(rust), Ok(c)) = (rust, c) {
                    let layout = rust.layout();
                    assert_eq!(
                        (layout.streams(), layout.coupled_streams(), layout.mapping()),
                        (c.streams, c.coupled, &c.mapping[..])
                    );
                    accepted += 1;
                }
            }
        }
    }
    // The largest layouts, once.
    for (channels, family) in [(255usize, 255u8), (256, 255), (227, 2), (228, 2)] {
        let c = CMsEncoder::surround(48_000, channels, family as i32, APP_AUDIO);
        let rust = MultistreamEncoder::new_surround(48_000, channels, family, Application::Audio);
        assert_eq!(
            rust.is_ok(),
            c.is_ok(),
            "{channels} channels, family {family}"
        );
    }
    println!("{accepted} layouts accepted by both");
    assert!(accepted > 100);
}

/// A small deterministic random number generator.
struct Lcg(u32);

impl Lcg {
    fn below(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) % n
    }
}

/// Layouts for the generic constructor: the ones libopus rejects must be
/// rejected, and the ones it accepts must be accepted.
#[test]
fn generic_constructor_accepts_what_libopus_accepts() {
    let check = |fs: i32, channels: usize, streams: usize, coupled: usize, mapping: &[u8]| {
        let c = CMsEncoder::generic(fs, channels, streams, coupled, mapping, APP_AUDIO);
        let rust =
            MultistreamEncoder::new(fs, channels, streams, coupled, mapping, Application::Audio);
        assert_eq!(
            rust.is_ok(),
            c.is_ok(),
            "{fs} Hz, {channels} channels, {streams} streams, {coupled} coupled, {mapping:?}: \
             opus-rs {:?}, C {:?}",
            rust.as_ref().err(),
            c.as_ref().err()
        );
        rust.is_ok()
    };
    // Each rule of libopus, one layout that breaks it.
    let full = [0u8; 256];
    assert!(check(48_000, 3, 2, 1, &[0, 2, 1]));
    assert!(!check(44_100, 3, 2, 1, &[0, 2, 1])); // sampling rate
    assert!(!check(48_000, 0, 1, 0, &full)); // no channel
    assert!(!check(48_000, 256, 1, 0, &full)); // too many channels
    assert!(!check(48_000, 2, 0, 0, &[0, 0])); // no stream
    assert!(!check(48_000, 4, 1, 2, &[0, 1, 2, 3])); // more coupled streams than streams
    assert!(!check(48_000, 2, 2, 1, &[0, 1])); // more coded channels than channels
    assert!(!check(48_000, 3, 2, 1, &[0, 1, 3])); // a coded channel that does not exist
    assert!(!check(48_000, 3, 2, 1, &[0, 1, 0])); // the mono stream has no input
    assert!(!check(48_000, 3, 2, 1, &[0, 2, 2])); // the stereo stream has no right input
    assert!(!check(48_000, 3, 2, 1, &[1, 2, 2])); // the stereo stream has no left input
    assert!(check(48_000, 3, 1, 1, &[0, 1, 255])); // a channel that is not coded
    assert!(check(48_000, 4, 2, 0, &[0, 1, 1, 0])); // two channels for one stream
    assert!(check(8_000, 255, 255, 0, &(0..=254).collect::<Vec<u8>>()));
    // A mapping with fewer entries than channels cannot be given to C.
    assert!(MultistreamEncoder::new(48_000, 3, 2, 1, &[0, 2], Application::Audio).is_err());

    // Random layouts, valid and invalid.
    let mut rng = Lcg(2024);
    let (mut accepted, mut rejected) = (0, 0);
    for _ in 0..3000 {
        let channels = 1 + rng.below(8) as usize;
        let streams = rng.below(6) as usize;
        let coupled = rng.below(4) as usize;
        let mapping: Vec<u8> = (0..channels)
            .map(|_| match rng.below(12) {
                0 => 255,
                1 => 9,
                _ => rng.below((streams + coupled).max(1) as u32 + 1) as u8,
            })
            .collect();
        if check(48_000, channels, streams, coupled, &mapping) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    println!("{accepted} random layouts accepted, {rejected} rejected");
    assert!(accepted > 50 && rejected > 50);
}

#[test]
fn frame_sizes_and_buffers_are_checked_like_libopus() {
    for fs in [48_000, 24_000, 16_000, 12_000, 8_000] {
        let case = Case::family("5.1", 6, 1);
        let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
        configure(&mut rust, &mut c, 96_000, true);
        let streams = c.streams;
        let unit = fs as usize / 400;
        let pcm = vec![0.01f32; 6 * 50 * unit];
        let mut out = vec![0u8; max_packet_bytes(streams)];
        // Frame sizes in units of 2.5 ms, and some that are no multiple.
        let sizes = (0..=50usize)
            .map(|k| k * unit)
            .chain([1, unit - 1, unit + 1, 8 * unit + 1]);
        for frame_size in sizes {
            let c_ok = c.encode_float(&pcm, frame_size, out.len()).is_ok();
            let rust_ok = rust.encode(&pcm, frame_size, &mut out).is_ok();
            // 80, 100 and 120 ms: libopus takes them, opus-rs does not
            // (difference D3 of the design).
            let long = [32, 40, 48].map(|k| k * unit).contains(&frame_size);
            assert_eq!(rust_ok, c_ok && !long, "{fs} Hz: frame size {frame_size}");
            assert_eq!(
                c_ok,
                [1, 2, 4, 8, 16, 24, 32, 40, 48]
                    .map(|k| k * unit)
                    .contains(&frame_size)
            );
        }
        // Sizes that are no frame size at all, however large.
        for frame_size in [
            usize::MAX,
            usize::MAX / 2,
            1 << 32,
            fs as usize,
            6 * fs as usize,
        ] {
            assert!(rust.encode(&pcm, frame_size, &mut out).is_err());
        }
        // Input that is one sample short of a frame.
        let frame_size = 8 * unit;
        assert!(
            rust.encode(&pcm[..6 * frame_size - 1], frame_size, &mut out)
                .is_err()
        );
        assert!(
            rust.encode_i16(&vec![0i16; 6 * frame_size - 1], frame_size, &mut out)
                .is_err()
        );
        assert!(
            rust.encode(&pcm[..6 * frame_size], frame_size, &mut out)
                .is_ok()
        );
        // The smallest packet: one byte per stream and one more for every
        // stream but the last.
        for len in 0..2 * streams + 2 {
            let c_result = c.encode_float(&pcm, frame_size, len);
            let rust_result = rust.encode(&pcm, frame_size, &mut out[..len]);
            assert_eq!(
                rust_result.is_ok(),
                c_result.is_ok(),
                "{fs} Hz: buffer of {len} bytes"
            );
            assert_eq!(rust_result.is_ok(), len >= 2 * streams - 1);
            if let Err(e) = c_result {
                assert_eq!(e, OPUS_BUFFER_TOO_SMALL);
            }
        }
    }
}

/// One run of `tight_buffers`: what happened to the frames.
#[derive(Default)]
struct Tight {
    /// Frames both encoders coded.
    both: usize,
    /// Frames libopus coded and opus-rs refused.
    only_c: Vec<String>,
}

/// Encodes with output buffers from two bytes below the smallest packet
/// upwards, with both encoders. Whatever opus-rs writes, C must decode.
fn tight_buffers(frame_sizes: &[usize]) -> Tight {
    let fs = 48_000;
    let mut tight = Tight::default();
    for case in CASES {
        for &frame_size in frame_sizes {
            for vbr in [true, false] {
                for per_channel in [16_000, 64_000] {
                    let (mut rust, mut c) = encoders(case, fs, Application::Audio);
                    configure(&mut rust, &mut c, per_channel * case.channels as i32, vbr);
                    let mut dec = c_decoder(&rust);
                    let streams = c.streams;
                    let smallest = 2 * streams - 1;
                    let lens: Vec<usize> = (smallest.saturating_sub(2)..smallest + 30)
                        .chain([60, 100, 200])
                        .collect();
                    let pcm = voiced(fs, case.channels, frame_size * lens.len());
                    let frames = pcm.chunks_exact(frame_size * case.channels);
                    for (frame, &len) in frames.zip(&lens) {
                        let label = format!(
                            "{} {} {} {per_channel} b/s per channel, buffer {len} (smallest {smallest})",
                            case.name,
                            ms(frame_size, fs),
                            if vbr { "VBR" } else { "CBR" },
                        );
                        let mut out = vec![0u8; len];
                        let rust_result = rust.encode(frame, frame_size, &mut out);
                        let c_result = c.encode_float(frame, frame_size, len);
                        if len < smallest {
                            assert!(rust_result.is_err() && c_result.is_err(), "{label}");
                            continue;
                        }
                        let c_packet = c_result.unwrap_or_else(|e| panic!("{label}: C error {e}"));
                        let n = match rust_result {
                            Ok(n) => n,
                            Err(e) => {
                                assert_eq!(e, "Output buffer too small", "{label}");
                                tight.only_c.push(label);
                                // The streams that were coded before the
                                // error moved on. Start both sides afresh.
                                rust.reset_state();
                                dec = c_decoder(&rust);
                                continue;
                            }
                        };
                        tight.both += 1;
                        let packet = &out[..n];
                        let decoded = dec
                            .decode_float(Some(packet), frame_size, false)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "{label}: C decode error {e}: {}",
                                    describe_packet(packet, streams)
                                )
                            });
                        assert_eq!(decoded.len(), frame_size * case.channels, "{label}");
                        if !has_stereo_hybrid_stream(packet, streams) {
                            assert_eq!(rust.final_range(), dec.final_range(), "{label}");
                        }
                        if !vbr {
                            assert_eq!(n, c_packet.len(), "{label}: packet size");
                        }
                    }
                }
            }
        }
    }
    tight
}

/// For frames up to 20 ms every buffer libopus can use, opus-rs can use.
#[test]
fn tight_output_buffers_up_to_20_ms() {
    let tight = tight_buffers(&[120, 240, 480, 960]);
    println!("{} frames coded by both encoders", tight.both);
    assert!(
        tight.only_c.is_empty(),
        "opus-rs refused: {:?}",
        tight.only_c
    );
    assert!(tight.both > 5000);
}

/// 40 and 60 ms frames: opus-rs codes the frames it accepts correctly, but
/// refuses buffers libopus can use. The cause is in the stream encoder
/// (src/lib.rs): a 40 or 60 ms frame that is not SILK-only is split into
/// 20 ms parts before the rule "too few bytes: write a packet without
/// audio" is applied, and each part needs 2 bytes. libopus applies the rule
/// to the whole frame first. A stream that is left 1 to 4 bytes then fails.
#[test]
fn tight_output_buffers_with_40_and_60_ms_frames() {
    let tight = tight_buffers(&[1920, 2880]);
    println!(
        "{} frames coded by both encoders, {} refused by opus-rs",
        tight.both,
        tight.only_c.len()
    );
    // Whatever was refused, most frames must be coded. Measured: 2171 coded
    // and 733 refused.
    assert!(tight.both > 2000, "only {} frames coded", tight.both);
    // With variable bitrate only the smallest buffers fail.
    for label in tight.only_c.iter().filter(|l| l.contains("VBR")) {
        assert!(label.contains("60 ms"), "{label}");
    }
}

#[test]
fn tight_output_buffers_with_40_and_60_ms_frames_like_libopus() {
    let tight = tight_buffers(&[1920, 2880]);
    assert!(
        tight.only_c.is_empty(),
        "opus-rs refused {} buffers libopus used, for example {:?}",
        tight.only_c.len(),
        &tight.only_c[..tight.only_c.len().min(5)]
    );
}

#[test]
fn reset_state_starts_the_stream_anew() {
    let fs = 48_000;
    let frame_size = 960;
    for case in CASES {
        let channels = case.channels as i32;
        let pcm = voiced(fs, case.channels, 12 * frame_size);
        let settings = |enc: &mut MultistreamEncoder| {
            enc.set_bitrate(Bitrate::BitsPerSecond(20_000 * channels))
                .unwrap();
            enc.set_complexity(7).unwrap();
            enc.set_vbr(false);
            enc.set_packet_loss_perc(5).unwrap();
        };
        let mut fresh = rust_encoder(case, fs, Application::Audio);
        settings(&mut fresh);
        let want = rust_encode_all(&mut fresh, &pcm, frame_size);

        // An encoder that has coded another signal, then is reset.
        let mut enc = rust_encoder(case, fs, Application::Audio);
        settings(&mut enc);
        rust_encode_all(
            &mut enc,
            &noise(case.channels, 12 * frame_size, 0.5, 3),
            frame_size,
        );
        enc.reset_state();
        assert_eq!(
            enc.bitrate_setting(),
            Bitrate::BitsPerSecond(20_000 * channels)
        );
        assert_eq!(enc.complexity(), 7);
        assert!(!enc.vbr());
        assert_eq!(enc.packet_loss_perc(), 5);
        let got = rust_encode_all(&mut enc, &pcm, frame_size);
        assert_eq!(got, want, "{}: packets after a reset", case.name);

        // Without the reset the history shows in the first packets.
        let mut enc = rust_encoder(case, fs, Application::Audio);
        settings(&mut enc);
        rust_encode_all(
            &mut enc,
            &noise(case.channels, 12 * frame_size, 0.5, 3),
            frame_size,
        );
        let got = rust_encode_all(&mut enc, &pcm, frame_size);
        assert_ne!(
            got[0], want[0],
            "{}: the test signal does not show history",
            case.name
        );
    }
}

#[test]
fn final_range_is_the_xor_of_the_stream_encoders() {
    let fs = 48_000;
    let frame_size = 960;
    for case in CASES {
        let mut enc = rust_encoder(case, fs, Application::Audio);
        enc.set_bitrate(Bitrate::BitsPerSecond(32_000 * case.channels as i32))
            .unwrap();
        assert_eq!(
            enc.final_range(),
            0,
            "{}: before the first frame",
            case.name
        );
        let streams = enc.layout().streams();
        let pcm = voiced(fs, case.channels, 10 * frame_size);
        let mut out = vec![0u8; max_packet_bytes(streams)];
        let mut nonzero = 0;
        for frame in pcm.chunks_exact(frame_size * case.channels) {
            let n = enc.encode(frame, frame_size, &mut out).unwrap();
            let xor = (0..streams).fold(0, |acc, s| {
                acc ^ enc.stream_encoder(s).unwrap().final_range()
            });
            assert_eq!(enc.final_range(), xor, "{}", case.name);
            // Every stream of the packet, decoded on its own by C, ends in
            // the state its encoder reports.
            let parts = split_streams(&out[..n], streams).unwrap();
            for (s, part) in parts.iter().enumerate() {
                let t = toc(part.toc);
                if t.mode == Mode::Hybrid && t.stereo {
                    continue;
                }
                let mut dec = CDecoder::new(fs, if t.stereo { 2 } else { 1 });
                // A new decoder reads the same bits whatever came before.
                dec.decode_float(Some(&part.packet), frame_size).unwrap();
                assert_eq!(
                    enc.stream_encoder(s).unwrap().final_range(),
                    dec.final_range(),
                    "{}: stream {s}",
                    case.name
                );
            }
            nonzero += (enc.final_range() != 0) as usize;
        }
        assert!(nonzero >= 9, "{}: final range is 0 too often", case.name);
    }
}

/// In surround mode the encoder gives every stream its energy mask on
/// every frame, as libopus does. A mask set by hand does not survive a
/// frame there. In any other layout the encoder leaves the mask alone.
///
/// After a frame every stream of a surround layout has the 21 values per
/// channel the surround analysis gave it. tests/multistream_enc_streams.rs
/// checks the values.
#[test]
fn surround_mode_sets_the_energy_mask_of_every_stream() {
    let fs = 48_000;
    let frame_size = 960;
    let by_hand = [123.0f32; 42];
    for case in CASES {
        let mut enc = rust_encoder(case, fs, Application::Audio);
        enc.set_bitrate(Bitrate::BitsPerSecond(32_000 * case.channels as i32))
            .unwrap();
        let streams = enc.layout().streams();
        let surround = enc.mapping_type() == opus_rs::multistream::layout::MappingType::Surround;
        let pcm = voiced(fs, case.channels, 3 * frame_size);
        let mut out = vec![0u8; max_packet_bytes(streams)];
        for frame in pcm.chunks_exact(frame_size * case.channels) {
            for s in 0..streams {
                let stream = enc.stream_encoder_mut(s).unwrap();
                stream.set_energy_mask(Some(&by_hand));
                assert!(stream.energy_mask().is_some());
            }
            enc.encode(frame, frame_size, &mut out).unwrap();
            for s in 0..streams {
                let stream = enc.stream_encoder(s).unwrap();
                let values = 21 * stream.channels();
                match stream.energy_mask() {
                    Some(mask) if surround => {
                        assert_eq!(mask.len(), values, "{}: stream {s}", case.name);
                        assert_ne!(mask, &by_hand[..values], "{}: stream {s}", case.name);
                    }
                    Some(mask) => {
                        assert_eq!(mask, &by_hand[..values], "{}: stream {s}", case.name)
                    }
                    None if surround => panic!("{}: stream {s} has no mask", case.name),
                    None => panic!("{}: stream {s} lost its mask", case.name),
                }
            }
        }
    }
}
