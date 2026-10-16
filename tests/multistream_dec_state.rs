//! Third set of checks of `MultistreamDecoder`: what a decoder carries from
//! one call to the next.
//!
//! The other `multistream_dec_*` tests feed their packets in fixed patterns.
//! Here the order of the calls is random: decoding, lost packets concealed
//! for any duration, packets rebuilt from forward error correction (FEC)
//! data, resets, float and 16-bit output in turn, and calls that fail.
//!
//! What is checked:
//!
//! - streams that are SILK-only, which both decoders conceal and rebuild
//!   alike: after every call the samples, the number of samples and the
//!   final range are those of the C libopus multistream decoder;
//! - layouts with CELT and Hybrid streams: after every call the output is,
//!   bit for bit, the output of one `OpusDecoder` per stream that got the
//!   same calls, and the verdict and the number of samples are libopus's;
//! - FEC on packets that hold several frames per stream;
//! - a reset of one stream decoder through `stream_decoder_mut`, against
//!   `OPUS_MULTISTREAM_GET_DECODER_STATE` and `OPUS_RESET_STATE`;
//! - damaged packets on the widest layouts: 255 mono streams, and 127 stereo
//!   streams with one mono stream;
//! - packets that no encoder wrote, built stream by stream from random
//!   framing and random frame data.
// The multistream API needs the `std` feature.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_dec_common;

use audiopus_sys as sys;
use ms_support::*;
use multistream_dec_common::*;
use opus1::OpusDecoder;
use std::f64::consts::PI;

const OPUS_SET_FORCE_CHANNELS_REQUEST: i32 = 4022;

/// A C multistream decoder with the controls that `ms_support::CMsDecoder`
/// does not offer.
struct RawCDecoder {
    st: *mut sys::OpusMSDecoder,
    channels: usize,
}

impl RawCDecoder {
    fn for_packets(e: &Encoded, fs: i32) -> Self {
        assert!(e.mapping.len() >= e.channels && e.channels <= 255);
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(fs, e.channels as i32, e.streams as i32, e.coupled as i32, e.mapping.as_ptr(), &mut err)
        };
        assert!(!st.is_null() && err == 0, "opus_multistream_decoder_create: {err}");
        Self { st, channels: e.channels }
    }

    /// `opus_multistream_decode_float`. An empty `packet` is a lost packet.
    fn decode_float(&mut self, packet: &[u8], frame_size: usize, fec: bool) -> Result<Vec<f32>, i32> {
        let mut out = vec![0f32; frame_size * self.channels];
        let ptr = if packet.is_empty() { std::ptr::null() } else { packet.as_ptr() };
        let n = unsafe {
            sys::opus_multistream_decode_float(self.st, ptr, packet.len() as i32, out.as_mut_ptr(), frame_size as i32, fec as i32)
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize * self.channels);
        Ok(out)
    }

    /// `opus_multistream_decode`: 16-bit output.
    fn decode_i16(&mut self, packet: &[u8], frame_size: usize, fec: bool) -> Result<Vec<i16>, i32> {
        let mut out = vec![0i16; frame_size * self.channels];
        let ptr = if packet.is_empty() { std::ptr::null() } else { packet.as_ptr() };
        let n = unsafe {
            sys::opus_multistream_decode(self.st, ptr, packet.len() as i32, out.as_mut_ptr(), frame_size as i32, fec as i32)
        };
        if n < 0 {
            return Err(n);
        }
        out.truncate(n as usize * self.channels);
        Ok(out)
    }

    /// `OPUS_RESET_STATE` on the multistream decoder.
    fn reset_state(&mut self) {
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
    }

    /// `OPUS_RESET_STATE` on the decoder of stream `s` alone.
    fn reset_stream(&mut self, s: usize) {
        let mut sub: *mut sys::OpusDecoder = std::ptr::null_mut();
        let ret = unsafe {
            sys::opus_multistream_decoder_ctl(self.st, OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST, s as i32, &mut sub as *mut *mut sys::OpusDecoder)
        };
        assert!(ret == 0 && !sub.is_null(), "OPUS_MULTISTREAM_GET_DECODER_STATE({s}) = {ret}");
        let ret = unsafe { sys::opus_decoder_ctl(sub, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE on stream {s}");
    }

    fn final_range(&mut self) -> u32 {
        let mut v = 0u32;
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut v as *mut u32) };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE");
        v
    }
}

impl Drop for RawCDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_decoder_destroy(self.st) }
    }
}

/// Bursts of a chord of three tones with pauses between them, on a small
/// constant offset. Every channel has its own root note, burst length and
/// start. The largest value is 0.26 times `gain`.
fn chord_bursts(fs: i32, channels: usize, samples: usize, gain: f32) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let root = 110.0 * (1.0 + 0.31 * (c % 7) as f64);
        let on = fs as usize * (180 + 25 * (c % 5)) / 1000;
        let off = fs as usize * (60 + 10 * (c % 3)) / 1000;
        // A burst rises and falls within 5 ms.
        let ramp = fs as usize / 200;
        for i in 0..samples {
            let t = i as f64 / fs as f64;
            let k = (i + 977 * c) % (on + off);
            let gate = if k >= on { 0.0 } else { (k.min(on - k) as f64 / ramp as f64).min(1.0) };
            let chord = (2.0 * PI * root * t).sin() + 0.6 * (2.0 * PI * 2.5 * root * t).sin() + 0.4 * (2.0 * PI * 4.01 * root * t).sin();
            pcm[i * channels + c] = gain * (0.02 + 0.12 * gate * chord) as f32;
        }
    }
    pcm
}

/// Encodes `milliseconds` of `chord_bursts` with `enc` in frames of `frame`
/// samples. The decoders get the mapping of the encoder.
fn encode_bursts(mut enc: CMsEncoder, kbps_per_channel: i32, frame: usize, milliseconds: usize, gain: f32, label: &str) -> Encoded {
    let channels = enc.channels;
    enc.set_bitrate(kbps_per_channel * 1000 * channels as i32);
    let samples = FS as usize * milliseconds / 1000 / frame * frame;
    let pcm = chord_bursts(FS, channels, samples, gain);
    let (mut packets, mut ranges) = (Vec::new(), Vec::new());
    for chunk in pcm.chunks_exact(frame * channels) {
        let packet = enc.encode_float(chunk, frame, max_packet_bytes(enc.streams)).unwrap_or_else(|e| panic!("{label}: C encoder: {e}"));
        ranges.push(enc.final_range());
        packets.push(packet);
    }
    Encoded {
        channels,
        streams: enc.streams,
        coupled: enc.coupled,
        mapping: enc.mapping.clone(),
        family: 255,
        input_channels: channels,
        packets,
        ranges,
        label: format!("{label}, {kbps_per_channel} kb/s per channel, {} ms frames, gain {gain}", frame as f64 / 48.0),
    }
}

/// A C encoder whose streams are all SILK-only and carry FEC data:
/// `coupled` stereo streams, forced to code two channels, then mono streams.
fn silk_encoder(streams: usize, coupled: usize) -> CMsEncoder {
    let channels = streams + coupled;
    let mapping: Vec<u8> = (0..channels as u8).collect();
    let mut enc = CMsEncoder::generic(FS, channels, streams, coupled, &mapping, APP_VOIP).unwrap();
    enc.set_inband_fec(true).set_packet_loss_perc(20);
    for s in 0..streams {
        assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK_ONLY), 0);
        // SILK alone codes up to wideband.
        assert_eq!(enc.stream_ctl(s, sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, sys::OPUS_BANDWIDTH_WIDEBAND), 0);
        if s < coupled {
            assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_CHANNELS_REQUEST, 2), 0);
        }
    }
    enc
}

/// The mode of each stream of a packet.
fn stream_modes(packet: &[u8], streams: usize) -> Vec<Mode> {
    split_streams(packet, streams).unwrap().iter().map(|s| toc(s.toc).mode).collect()
}

/// Whether a mono SILK stream carries FEC data for the packet before it.
///
/// A SILK packet starts with one voice activity bit per SILK frame and then
/// one bit that says whether FEC data follows. They are the first bits of
/// the first byte. A SILK frame has 20 ms, or 10 ms in a 10 ms packet.
fn mono_silk_stream_has_fec(stream: &StreamPacket) -> bool {
    let t = toc(stream.toc);
    assert!(t.mode == Mode::Silk && !t.stereo);
    let silk_frames = (t.frame_48k / 960).max(1);
    let parsed = opus1::packet::parse(&stream.packet, false).unwrap();
    let first = parsed.frame(&stream.packet, 0);
    !first.is_empty() && first[0] & (0x80 >> silk_frames) != 0
}

/// One call on a multistream decoder.
#[derive(Clone, Copy, Debug)]
struct CallSpec {
    /// `decode_fec` in place of `decode`.
    fec: bool,
    /// Index of the packet, or `None` for an empty packet.
    packet: Option<usize>,
    /// The packet is cut to one byte less than its streams need.
    cut: bool,
    /// `frame_size` is this many units of 2.5 ms...
    units: usize,
    /// ...plus this many samples.
    extra: usize,
    /// 16-bit output.
    sixteen: bool,
    /// Whether the call must succeed.
    ok: bool,
}

#[derive(Clone, Copy, Debug)]
enum Step {
    /// `reset_state`.
    Reset,
    Call(CallSpec),
}

/// A random order of calls for a stream of `packets` packets of
/// `frame_units` units of 2.5 ms each.
///
/// About every 5th packet is lost. The gap is then left alone, concealed
/// for its duration or for any other, or rebuilt from the next packet for
/// its duration or a longer one. Between the packets there are resets and
/// calls that must fail. With `start_lost`, the first packet is lost, so the
/// first call on the new decoder conceals or rebuilds.
fn random_steps(packets: usize, frame_units: usize, rng: &mut Rng, start_lost: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut lost_run = 0usize;
    let good = |fec, packet, units, sixteen| Step::Call(CallSpec { fec, packet, cut: false, units, extra: 0, sixteen, ok: true });
    for i in 0..packets {
        if rng.below(13) == 0 {
            steps.push(Step::Reset);
        }
        if i + 1 < packets && (rng.below(5) == 0 || (i == 0 && start_lost)) {
            lost_run += 1;
            continue;
        }
        let sixteen = rng.below(2) == 0;
        if lost_run > 0 {
            match rng.below(7) {
                // The application does not report the loss.
                0 => {}
                // Every lost packet is concealed for its own duration.
                1 => {
                    for _ in 0..lost_run {
                        steps.push(good(false, None, frame_units, sixteen));
                    }
                }
                // Concealed for any other duration, up to 120 ms.
                2 => steps.push(good(false, None, 1 + rng.below(48), sixteen)),
                // The last lost packet is rebuilt, the ones before it are
                // concealed.
                3 | 4 => {
                    for _ in 1..lost_run {
                        steps.push(good(false, None, frame_units, sixteen));
                    }
                    steps.push(good(true, Some(i), frame_units, sixteen));
                }
                // Rebuilt with a longer duration: the start is concealed.
                5 => steps.push(good(true, Some(i), (frame_units + 1 + rng.below(20)).min(48), sixteen)),
                // `decode_fec` without a packet conceals.
                _ => steps.push(good(true, None, frame_units, sixteen)),
            }
            lost_run = 0;
        }
        if rng.below(5) == 0 {
            let bad = |fec, packet, cut, units, extra| Step::Call(CallSpec { fec, packet, cut, units, extra, sixteen, ok: false });
            steps.push(match rng.below(6) {
                // No room at all.
                0 => bad(false, Some(i), false, 0, 0),
                // A lost duration that is not a multiple of 2.5 ms.
                1 => bad(false, None, false, 1 + rng.below(47), 1),
                2 => bad(true, Some(i), false, frame_units, 1),
                // Less room than the packet needs.
                3 if frame_units > 1 => bad(false, Some(i), false, frame_units - 1, 0),
                4 if frame_units > 1 => bad(true, Some(i), false, frame_units - 1, 0),
                // A packet that ends inside its streams.
                _ => bad(false, Some(i), true, 48, 0),
            });
        }
        // Room for 120 ms, or exactly the room the packet needs.
        let units = if rng.below(3) == 0 { frame_units } else { 48 };
        steps.push(good(false, Some(i), units, sixteen));
    }
    steps
}

/// The bytes of the packet of `spec`.
fn packet_of<'a>(e: &'a Encoded, spec: &CallSpec) -> &'a [u8] {
    match spec.packet {
        None => &[],
        Some(i) if spec.cut => &e.packets[i][..2 * e.streams - 2],
        Some(i) => &e.packets[i],
    }
}

/// What a run of `random_steps` against the C decoder counted.
#[derive(Debug, Default)]
struct Tally {
    decoded: usize,
    concealed: usize,
    rebuilt: usize,
    /// Rebuilt from a packet in which some mono stream carries FEC data.
    rebuilt_from_fec_data: usize,
    rejected: usize,
    resets: usize,
    samples_16: usize,
    /// 16-bit samples one step away from the C decoder's.
    differing_16: usize,
    /// 16-bit samples at full scale.
    full_scale_16: usize,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.decoded += other.decoded;
        self.concealed += other.concealed;
        self.rebuilt += other.rebuilt;
        self.rebuilt_from_fec_data += other.rebuilt_from_fec_data;
        self.rejected += other.rejected;
        self.resets += other.resets;
        self.samples_16 += other.samples_16;
        self.differing_16 += other.differing_16;
        self.full_scale_16 += other.full_scale_16;
    }
}

/// Runs `steps` on both decoders. Every stream of `e` must be SILK-only.
/// After every call the two decoders must agree on the verdict, the number
/// of samples, the final range and the samples: float output within 100 dB
/// per channel, 16-bit output within one step.
fn run_steps_against_c(e: &Encoded, fs: i32, steps: &[Step]) -> Tally {
    let mut c_dec = RawCDecoder::for_packets(e, fs);
    let (_, mut r_dec) = e.decoders(fs);
    let ch = e.channels;
    let unit = fs as usize / 400;
    let max_frame = 48 * unit;
    let mut out = vec![0f32; max_frame * ch];
    let mut out16 = vec![0i16; max_frame * ch];
    let mut tally = Tally::default();
    for (k, step) in steps.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, step {k}: {step:?}", e.label);
        let spec = match step {
            Step::Reset => {
                c_dec.reset_state();
                r_dec.reset_state();
                assert_eq!((c_dec.final_range(), r_dec.final_range()), (0, 0), "{}: final range", what());
                tally.resets += 1;
                continue;
            }
            Step::Call(spec) => spec,
        };
        let packet = packet_of(e, spec);
        let frame_size = spec.units * unit + spec.extra;
        if spec.sixteen {
            let c = c_dec.decode_i16(packet, frame_size, spec.fec);
            let r = if spec.fec { r_dec.decode_fec_i16(packet, frame_size, &mut out16) } else { r_dec.decode_i16(packet, frame_size, &mut out16) };
            assert_eq!((c.is_ok(), r.is_ok()), (spec.ok, spec.ok), "{}: C {:?}, opus1 {r:?}", what(), c.as_ref().map(|pcm| pcm.len() / ch));
            if let (Ok(c), Ok(n)) = (c, r) {
                assert_eq!(c.len(), n * ch, "{}: samples", what());
                for (&c, &r) in c.iter().zip(&out16[..n * ch]) {
                    let d = (c as i32 - r as i32).abs();
                    assert!(d <= 1, "{}: C {c}, opus1 {r}", what());
                    tally.differing_16 += (d != 0) as usize;
                    tally.full_scale_16 += (c == i16::MAX || c == i16::MIN) as usize;
                }
                tally.samples_16 += c.len();
            }
        } else {
            let c = c_dec.decode_float(packet, frame_size, spec.fec);
            let r = if spec.fec { r_dec.decode_fec(packet, frame_size, &mut out) } else { r_dec.decode(packet, frame_size, &mut out) };
            assert_eq!((c.is_ok(), r.is_ok()), (spec.ok, spec.ok), "{}: C {:?}, opus1 {r:?}", what(), c.as_ref().map(|pcm| pcm.len() / ch));
            if let (Ok(c), Ok(n)) = (c, r) {
                assert_eq!(c.len(), n * ch, "{}: samples", what());
                for channel in 0..ch {
                    let (mut signal, mut noise) = (0f64, 0f64);
                    for (c_frame, r_frame) in c.chunks_exact(ch).zip(out.chunks_exact(ch)) {
                        signal += (c_frame[channel] as f64).powi(2);
                        noise += ((c_frame[channel] - r_frame[channel]) as f64).powi(2);
                    }
                    // 100 dB below the signal, or 140 dB below full scale.
                    assert!(
                        noise <= signal * 1e-10 + n as f64 * 1e-14,
                        "{}: channel {channel} is {:.1} dB from the C decoder",
                        what(),
                        snr_db(signal, noise)
                    );
                }
            }
        }
        if !spec.ok {
            tally.rejected += 1;
            continue;
        }
        assert_eq!(r_dec.final_range(), c_dec.final_range(), "{}: final range", what());
        match (spec.fec, spec.packet) {
            (false, Some(i)) => {
                assert_eq!(r_dec.final_range(), e.ranges[i], "{}: final range of the encoder", what());
                tally.decoded += 1;
            }
            (true, Some(i)) => {
                tally.rebuilt += 1;
                let parts = split_streams(&e.packets[i], e.streams).unwrap();
                tally.rebuilt_from_fec_data += parts[e.coupled..].iter().any(mono_silk_stream_has_fec) as usize;
            }
            (_, None) => {
                assert_eq!(r_dec.final_range(), 0, "{}: final range after a concealed packet", what());
                tally.concealed += 1;
            }
        }
    }
    tally
}

/// SILK-only streams, mono and stereo, with FEC data: any order of calls
/// gives what the C decoder gives.
#[test]
fn random_call_orders_on_silk_streams_match_libopus() {
    let mut total = Tally::default();
    let mut seed = 1u32;
    // (streams, stereo streams, gain)
    for (streams, coupled, gain) in [(3usize, 0usize, 1.0f32), (3, 0, 5.0), (3, 2, 1.0), (2, 1, 5.0)] {
        for frame in [480usize, 960, 1920, 2880] {
            // At least 50 packets.
            let milliseconds = (50 * frame / 48).max(1200);
            let e = encode_bursts(silk_encoder(streams, coupled), 20, frame, milliseconds, gain, &format!("SILK, {streams} streams, {coupled} stereo"));
            for (i, p) in e.packets.iter().enumerate() {
                assert!(stream_modes(p, streams).iter().all(|&m| m == Mode::Silk), "{}: packet {i} is not SILK-only", e.label);
            }
            assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
            for fs in [8000, 12000, 24000] {
                seed += 1;
                let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9));
                let steps = random_steps(e.packets.len(), frame / 120, &mut rng, seed % 3 == 0);
                total.add(&run_steps_against_c(&e, fs, &steps));
            }
        }
    }
    println!("{total:?}");
    assert!(total.decoded >= 2000 && total.concealed >= 150 && total.rebuilt >= 150, "{total:?}");
    assert!(total.rejected >= 300 && total.resets >= 100, "{total:?}");
    // The rebuilt frames come from FEC data, not from concealment alone.
    assert!(total.rebuilt_from_fec_data * 2 >= total.rebuilt, "{total:?}");
    // The loud signals reach full scale, where the soft clip works.
    assert!(total.full_scale_16 > 0, "{total:?}");
    assert!(total.differing_16 * 100 <= total.samples_16, "{total:?}");
}

/// Copies the outputs of the stream decoders to the output channels that
/// the mapping names. `decoded[s]` holds `samples` samples of stream `s`,
/// interleaved.
fn route<T: Copy + Default>(e: &Encoded, decoded: &[Vec<T>], samples: usize) -> Vec<T> {
    let mut out = vec![T::default(); samples * e.channels];
    for (c, &m) in e.mapping.iter().enumerate() {
        let m = m as usize;
        if m == 255 {
            continue;
        }
        // Decoded channels: 2 per stereo stream first, then 1 per mono stream.
        let (stream, stream_channels, index) = if m < 2 * e.coupled { (m / 2, 2, m % 2) } else { (m - e.coupled, 1, 0) };
        for i in 0..samples {
            out[i * e.channels + c] = decoded[stream][i * stream_channels + index];
        }
    }
    out
}

/// Runs `steps` on a multistream decoder, on one `OpusDecoder` per stream
/// and on the C multistream decoder.
///
/// The multistream output must be the output of the stream decoders, bit
/// for bit, and its final range the XOR of theirs. A call that must fail is
/// not given to the stream decoders at all, so it must leave the multistream
/// decoder as it was. The C decoder must give the same verdict and number of
/// samples. Returns the number of calls that succeeded.
fn run_steps_against_stream_decoders(e: &Encoded, fs: i32, steps: &[Step]) -> usize {
    let mut c_dec = RawCDecoder::for_packets(e, fs);
    let (_, mut ms) = e.decoders(fs);
    let mut singles: Vec<OpusDecoder> = (0..e.streams).map(|s| OpusDecoder::new(fs, if s < e.coupled { 2 } else { 1 }).unwrap()).collect();
    let ch = e.channels;
    let unit = fs as usize / 400;
    let max_frame = 48 * unit;
    let mut out = vec![0f32; max_frame * ch];
    let mut out16 = vec![0i16; max_frame * ch];
    let mut calls = 0;
    for (k, step) in steps.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, step {k}: {step:?}", e.label);
        let spec = match step {
            Step::Reset => {
                c_dec.reset_state();
                ms.reset_state();
                singles.iter_mut().for_each(|dec| dec.reset_state());
                continue;
            }
            Step::Call(spec) => spec,
        };
        let packet = packet_of(e, spec);
        let frame_size = spec.units * unit + spec.extra;
        // The outputs start with values that are not silence, so that a
        // silent channel has to be written.
        out.fill(0.25);
        out16.fill(77);
        let (c, r) = if spec.sixteen {
            let c = c_dec.decode_i16(packet, frame_size, spec.fec).map(|pcm| pcm.len() / ch);
            let r = if spec.fec { ms.decode_fec_i16(packet, frame_size, &mut out16) } else { ms.decode_i16(packet, frame_size, &mut out16) };
            (c, r)
        } else {
            let c = c_dec.decode_float(packet, frame_size, spec.fec).map(|pcm| pcm.len() / ch);
            let r = if spec.fec { ms.decode_fec(packet, frame_size, &mut out) } else { ms.decode(packet, frame_size, &mut out) };
            (c, r)
        };
        assert_eq!((c.is_ok(), r.is_ok()), (spec.ok, spec.ok), "{}: C {c:?}, opus1 {r:?}", what());
        let Ok(n) = r else {
            // A call that fails writes nothing.
            assert!(out.iter().all(|&v| v == 0.25) && out16.iter().all(|&v| v == 77), "{}: a rejected call wrote samples", what());
            continue;
        };
        assert_eq!(c, Ok(n), "{}: samples", what());
        // Nothing is written behind the decoded samples.
        assert!(out[n * ch..].iter().all(|&v| v == 0.25) && out16[n * ch..].iter().all(|&v| v == 77), "{}: written behind the samples", what());
        calls += 1;

        // The same call on every stream decoder, with the stream's own
        // packet.
        let parts = spec.packet.map(|i| split_streams(&e.packets[i], e.streams).unwrap());
        let mut range = 0;
        let (mut want, mut want16) = (Vec::new(), Vec::new());
        for (s, dec) in singles.iter_mut().enumerate() {
            let stream_channels = if s < e.coupled { 2 } else { 1 };
            let stream_packet: &[u8] = parts.as_ref().map_or(&[], |p| &p[s].packet);
            if spec.sixteen {
                let mut buf = vec![0i16; frame_size * stream_channels];
                let got = if spec.fec { dec.decode_fec_i16(stream_packet, frame_size, &mut buf) } else { dec.decode_i16(stream_packet, frame_size, &mut buf) };
                assert_eq!(got, Ok(n), "{}: stream {s}", what());
                want16.push(buf);
            } else {
                let mut buf = vec![0f32; frame_size * stream_channels];
                let got = if spec.fec { dec.decode_fec(stream_packet, frame_size, &mut buf) } else { dec.decode(stream_packet, frame_size, &mut buf) };
                assert_eq!(got, Ok(n), "{}: stream {s}", what());
                want.push(buf);
            }
            range ^= dec.final_range();
        }
        if spec.sixteen {
            assert_eq!(&out16[..n * ch], &route(e, &want16, n)[..], "{}: 16-bit output differs from the stream decoders'", what());
        } else {
            let want = route(e, &want, n);
            assert!(
                out[..n * ch].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{}: float output differs from the stream decoders'",
                what()
            );
        }
        assert_eq!(ms.final_range(), range, "{}: final range", what());
        if let (false, Some(i)) = (spec.fec, spec.packet) {
            assert_eq!(range, e.ranges[i], "{}: final range of the encoder", what());
            assert_eq!(c_dec.final_range(), e.ranges[i], "{}: final range of the C decoder", what());
        }
    }
    calls
}

/// Layouts with CELT and Hybrid streams. The loss concealment of CELT is not
/// sample-exact with libopus, so the samples are compared with one
/// `OpusDecoder` per stream, which must give the same bits in any order of
/// calls.
#[test]
fn random_call_orders_give_the_output_of_one_decoder_per_stream() {
    // (layout, settings, kb/s per channel, frame, gain)
    let cases = [
        (SURROUND_5_1, AUDIO, 24, 960usize, 5.0f32),
        (SURROUND_7_1, AUDIO, 32, 240, 1.0),
        (QUAD, VOIP_FEC, 24, 480, 1.0),
        (DISCRETE_5, VOIP_FEC, 20, 1920, 5.0),
        (AMBISONICS_4, AUDIO_CBR, 32, 2880, 1.0),
        (Layout::Generic, VOIP_FEC, 32, 960, 5.0),
    ];
    let mut calls = 0;
    let mut modes = std::collections::BTreeSet::new();
    for (k, (layout, settings, kbps, frame, gain)) in cases.into_iter().enumerate() {
        let (enc, _) = c_encoder(layout, settings);
        let milliseconds = (40 * frame / 48).max(1000);
        let mut e = encode_bursts(enc, kbps, frame, milliseconds, gain, &format!("{layout:?} {settings:?}"));
        if layout == Layout::Generic {
            e.channels = GENERIC_DECODER_MAPPING.len();
            e.mapping = GENERIC_DECODER_MAPPING.to_vec();
        }
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for p in &e.packets {
            modes.extend(stream_modes(p, e.streams).iter().map(|m| format!("{m:?}")));
        }
        for (j, fs) in [8000, 12000, 24000, 48000].into_iter().enumerate() {
            let mut rng = Rng(77 + 16 * k as u32 + j as u32);
            let steps = random_steps(e.packets.len(), frame / 120, &mut rng, (k + j) % 2 == 0);
            calls += run_steps_against_stream_decoders(&e, fs, &steps);
        }
    }
    println!("{calls} calls, stream modes {modes:?}");
    assert!(calls >= 1200, "only {calls} calls");
    assert_eq!(modes.len(), 3, "{modes:?}");
}

/// Joins `group`, consecutive packets of `e`, into one packet in which
/// every stream holds the frames of all of them. Returns `None` when a
/// stream changes its configuration inside the group.
fn join_packets(e: &Encoded, group: &[Vec<u8>]) -> Option<Vec<u8>> {
    let split: Vec<Vec<StreamPacket>> = group.iter().map(|p| split_streams(p, e.streams).unwrap()).collect();
    let mut joined = Vec::new();
    for s in 0..e.streams {
        let toc_byte = split[0][s].toc & 0xFC;
        if split.iter().any(|parts| parts[s].toc & 0xFC != toc_byte) {
            return None;
        }
        let mut frames: Vec<&[u8]> = Vec::new();
        for parts in &split {
            let stream = &parts[s].packet;
            let parsed = opus1::packet::parse(stream, false).unwrap();
            frames.extend(parsed.frames(stream));
        }
        let mut buf = vec![0u8; 6 * 1277 + 8];
        let len = opus1::packet::write(toc_byte, &frames, &mut buf, s + 1 != e.streams, false).unwrap();
        joined.extend_from_slice(&buf[..len]);
    }
    Some(joined)
}

/// FEC data stands for one frame: the frame before the first frame of the
/// packet that carries it. When that packet holds several frames, a
/// multistream decoder still wants room for the whole packet, and it
/// conceals everything before the last frame of the lost duration.
#[test]
fn fec_on_packets_with_several_frames_matches_libopus() {
    for (streams, coupled, per_packet) in [(3usize, 0usize, 2usize), (3, 0, 3), (2, 1, 2), (4, 0, 5)] {
        let e = encode_bursts(silk_encoder(streams, coupled), 20, 960, 4000, 1.0, &format!("SILK, {streams} streams, {coupled} stereo"));
        // A group in which a stream changes its bandwidth cannot be one
        // packet. Its packets are then fed one by one.
        let groups: Vec<Vec<Vec<u8>>> = e
            .packets
            .chunks_exact(per_packet)
            .map(|group| join_packets(&e, group).map_or_else(|| group.to_vec(), |joined| vec![joined]))
            .collect();
        for fs in [12000, 48000] {
            let mut c_dec = RawCDecoder::for_packets(&e, fs);
            let (_, mut r_dec) = e.decoders(fs);
            let ch = e.channels;
            let unit = fs as usize / 400;
            // One frame has 20 ms.
            let frame = 8 * unit;
            let whole = per_packet * frame;
            let mut out = vec![0f32; 48 * unit * ch];
            let (mut rebuilt, mut rejected) = (0, 0);
            for (g, group) in groups.iter().enumerate() {
                let what = format!("{} at {fs} Hz, {per_packet} frames per packet, group {g}", e.label);
                // Every third group is lost.
                if g % 3 == 2 {
                    continue;
                }
                if g > 0 && g % 3 == 0 && group.len() == 1 {
                    let packet = &group[0];
                    // Room for less than the packet: one frame, and one unit
                    // less than the packet. `opus_decode` would conceal;
                    // the multistream decoder rejects the call.
                    for small in [frame, whole - unit] {
                        assert_eq!(c_dec.decode_float(packet, small, true), Err(OPUS_BUFFER_TOO_SMALL), "{what}: {small} samples");
                        assert!(r_dec.decode_fec(packet, small, &mut out).is_err(), "{what}: {small} samples");
                        rejected += 1;
                    }
                    // The lost duration: the packet's, one frame more, or
                    // 120 ms.
                    let lost = [whole, (whole + frame).min(48 * unit), 48 * unit][(g / 3) % 3];
                    let c = c_dec.decode_float(packet, lost, true).unwrap_or_else(|err| panic!("{what}: C FEC: {err}"));
                    let n = r_dec.decode_fec(packet, lost, &mut out).unwrap_or_else(|err| panic!("{what}: FEC: {err}"));
                    assert_eq!((n, c.len()), (lost, lost * ch), "{what}: samples from FEC");
                    let snr = min_channel_snr(&c, &out[..n * ch], ch);
                    assert!(snr >= 100.0, "{what}: {lost} samples rebuilt, {snr:.1} dB from the C decoder");
                    assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range after FEC");
                    rebuilt += 1;
                }
                let mut samples = 0;
                for packet in group {
                    assert!(stream_modes(packet, streams).iter().all(|&m| m == Mode::Silk), "{what}: not SILK-only");
                    let c = c_dec.decode_float(packet, 48 * unit, false).unwrap_or_else(|err| panic!("{what}: C decoder: {err}"));
                    let n = r_dec.decode(packet, 48 * unit, &mut out).unwrap_or_else(|err| panic!("{what}: {err}"));
                    assert_eq!(c.len(), n * ch, "{what}: samples");
                    let snr = min_channel_snr(&c, &out[..n * ch], ch);
                    assert!(snr >= 100.0, "{what}: {snr:.1} dB from the C decoder");
                    assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range");
                    samples += n;
                }
                assert_eq!(samples, whole, "{what}: samples of the group");
                // The final range is that of the last frame.
                assert_eq!(r_dec.final_range(), e.ranges[(g + 1) * per_packet - 1], "{what}: final range of the encoder");
            }
            assert!(rebuilt >= 8 && rejected >= 16, "{}: {rebuilt} rebuilt, {rejected} rejected", e.label);
        }
    }
}

/// `stream_decoder_mut` reaches the decoder of one stream. A reset of that
/// decoder alone gives what the C decoder gives when the stream decoder from
/// `OPUS_MULTISTREAM_GET_DECODER_STATE` is reset.
#[test]
fn a_reset_of_one_stream_decoder_matches_libopus() {
    let (celt, _) = c_encoder(SURROUND_5_1, AUDIO);
    let (mixed, _) = c_encoder(DISCRETE_5, VOIP_FEC);
    let cases = [
        encode_bursts(celt, 32, 960, 1200, 1.0, "5.1"),
        encode_bursts(mixed, 24, 480, 1200, 1.0, "5 mono streams"),
        encode_bursts(silk_encoder(3, 1), 20, 1920, 2400, 1.0, "SILK, 3 streams, 1 stereo"),
    ];
    for e in &cases {
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for fs in [16000, 48000] {
            let mut c_dec = RawCDecoder::for_packets(e, fs);
            let (_, mut r_dec) = e.decoders(fs);
            // A decoder that is never reset, to see that a reset is heard.
            let (_, mut plain) = e.decoders(fs);
            let ch = e.channels;
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0f32; max_frame * ch];
            let mut out_plain = vec![0f32; max_frame * ch];
            let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
            let mut changed = false;
            for (i, packet) in e.packets.iter().enumerate() {
                let what = format!("{} at {fs} Hz, packet {i}", e.label);
                // Before packets 7, 14, 21, ...: reset one stream, another
                // one each time.
                if i > 0 && i % 7 == 0 {
                    let s = (i / 7) % e.streams;
                    let before = r_dec.final_range();
                    let stream_range = r_dec.stream_decoder(s).unwrap().final_range();
                    c_dec.reset_stream(s);
                    r_dec.stream_decoder_mut(s).unwrap().reset_state();
                    assert_eq!(r_dec.final_range(), before ^ stream_range, "{what}: final range after a reset of stream {s}");
                    assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range of the C decoder");
                }
                let c = c_dec.decode_float(packet, max_frame, false).unwrap();
                let n = r_dec.decode(packet, max_frame, &mut out).unwrap();
                assert_eq!(c.len(), n * ch, "{what}: samples");
                assert_eq!(r_dec.final_range(), e.ranges[i], "{what}: final range");
                assert_eq!(plain.decode(packet, max_frame, &mut out_plain), Ok(n));
                changed |= out[..n * ch].iter().zip(&out_plain[..n * ch]).any(|(a, b)| a.to_bits() != b.to_bits());
                c_all.extend_from_slice(&c);
                r_all.extend_from_slice(&out[..n * ch]);
            }
            assert!(changed, "{} at {fs} Hz: the resets changed nothing", e.label);
            let snrs = channel_snrs(&c_all, &r_all, ch);
            assert!(snrs.iter().all(|&snr| snr >= 100.0), "{} at {fs} Hz: {snrs:.0?} dB from the C decoder", e.label);
        }
    }
}

/// Changes a packet of many streams in one of six ways.
fn damage(packet: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut p = packet.to_vec();
    match rng.below(6) {
        // Flip one bit.
        0 => {
            let i = rng.below(p.len());
            p[i] ^= 1 << rng.below(8);
        }
        // Replace one byte.
        1 => {
            let i = rng.below(p.len());
            p[i] = rng.next() as u8;
        }
        // Cut the end off. Half of the time only a few bytes.
        2 => {
            let cut = if rng.below(2) == 0 { 1 + rng.below(4) } else { rng.below(p.len()) };
            p.truncate(p.len().saturating_sub(cut));
        }
        // Take a byte out.
        3 => {
            let i = rng.below(p.len());
            p.remove(i);
        }
        // Repeat a byte.
        4 => {
            let i = rng.below(p.len());
            p.insert(i, p[i]);
        }
        // Add bytes at the end.
        _ => {
            for _ in 0..1 + rng.below(20) {
                p.push(rng.next() as u8);
            }
        }
    }
    p
}

/// Damaged packets on the widest layouts libopus accepts. Both decoders
/// must agree on accepting a packet and on the number of samples.
///
/// A packet whose damage changes the channel count of a stream is left out:
/// libopus converts it, and `OpusDecoder` rejects it for now.
#[test]
fn damaged_packets_on_the_widest_layouts_get_the_verdict_of_libopus() {
    // 255 mono streams.
    let mono = encode(Layout::Family { channels: 255, family: 255 }, AUDIO, 8, 960, Signal::Voiced, 100, 1.0);
    assert_eq!((mono.streams, mono.coupled), (255, 0));
    // 127 stereo streams, forced to code two channels, and one mono stream.
    let mapping: Vec<u8> = (0..255).map(|c| c as u8).collect();
    let mut enc = CMsEncoder::generic(FS, 255, 128, 127, &mapping, APP_AUDIO).unwrap();
    for s in 0..127 {
        assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_CHANNELS_REQUEST, 2), 0);
    }
    let stereo = encode_bursts(enc, 12, 480, 60, 1.0, "127 stereo streams and 1 mono stream");
    assert_eq!((stereo.streams, stereo.coupled), (128, 127));

    let mut rng = Rng(0xBADC_0DE);
    let (mut accepted, mut rejected, mut skipped) = (0, 0, 0);
    for (e, fs) in [(&mono, 8000), (&stereo, 16000)] {
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        let mut c_dec = RawCDecoder::for_packets(e, fs);
        let (_, mut r_dec) = e.decoders(fs);
        let max_frame = fs as usize / 25 * 3;
        let mut out = vec![0f32; max_frame * e.channels];
        let mut out16 = vec![0i16; max_frame * e.channels];
        for round in 0..160 {
            let original = &e.packets[rng.below(e.packets.len())];
            // One packet in four is fed as it is.
            let packet = if round % 4 == 0 { original.clone() } else { damage(original, &mut rng) };
            if let Ok(parts) = split_streams(&packet, e.streams) {
                if parts.iter().enumerate().any(|(s, part)| toc(part.toc).stereo != (s < e.coupled)) {
                    skipped += 1;
                    continue;
                }
            }
            let what = format!("{} at {fs} Hz, round {round}, {} bytes", e.label, packet.len());
            let fec = rng.below(8) == 0;
            let frame_size = if fec { fs as usize / 50 } else { max_frame };
            let (c, r) = if rng.below(4) == 0 {
                let c = c_dec.decode_i16(&packet, frame_size, fec).map(|pcm| pcm.len() / e.channels);
                let r = if fec { r_dec.decode_fec_i16(&packet, frame_size, &mut out16) } else { r_dec.decode_i16(&packet, frame_size, &mut out16) };
                (c, r)
            } else {
                let c = c_dec.decode_float(&packet, frame_size, fec).map(|pcm| pcm.len() / e.channels);
                let r = if fec { r_dec.decode_fec(&packet, frame_size, &mut out) } else { r_dec.decode(&packet, frame_size, &mut out) };
                (c, r)
            };
            match (c, r) {
                (Ok(c), Ok(r)) => {
                    assert_eq!(c, r, "{what}: samples per channel");
                    accepted += 1;
                }
                (Err(_), Err(_)) => rejected += 1,
                (c, r) => panic!("{what}: C {c:?}, opus1 {r:?}"),
            }
        }
    }
    println!("{accepted} accepted, {rejected} rejected, {skipped} left out");
    assert!(accepted >= 100 && rejected >= 60, "{accepted} accepted, {rejected} rejected, {skipped} left out");
}

/// Packets that no encoder wrote. Every stream gets a table-of-contents
/// (TOC) byte, the size fields of its frame count code and random bytes as
/// frames. Most packets give all streams the same duration. Some do not,
/// some are cut short and some are random bytes altogether. No call may
/// panic, and both decoders must agree on accepting a packet and on the
/// number of samples.
///
/// Every stream has the channel count of its decoder (see
/// `damaged_packets_on_the_widest_layouts_get_the_verdict_of_libopus`).
#[test]
fn made_up_packets_get_the_verdict_of_libopus() {
    // (channels, streams, stereo streams, mapping)
    let layouts: [(usize, usize, usize, Vec<u8>); 6] = [
        (1, 1, 0, vec![0]),
        (2, 1, 1, vec![0, 1]),
        (6, 4, 2, vec![0, 4, 1, 2, 3, 5]),
        (3, 3, 0, vec![0, 1, 2]),
        (5, 2, 1, vec![2, 0, 255, 1, 2]),
        (255, 2, 2, (0..255).map(|c| [0u8, 1, 2, 3, 255][c % 5]).collect()),
    ];
    let mut rng = Rng(0x5EED_0001);
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (k, (channels, streams, coupled, mapping)) in layouts.iter().enumerate() {
        let (channels, streams, coupled) = (*channels, *streams, *coupled);
        for fs in [[8000, 24000], [12000, 48000], [16000, 48000]][k % 3] {
            let e = Encoded {
                channels,
                streams,
                coupled,
                mapping: mapping.clone(),
                family: 255,
                input_channels: streams + coupled,
                packets: Vec::new(),
                ranges: Vec::new(),
                label: format!("{channels} channels, {streams} streams, {coupled} stereo"),
            };
            let mut c_dec = RawCDecoder::for_packets(&e, fs);
            let (_, mut r_dec) = e.decoders(fs);
            let unit = fs as usize / 400;
            let mut out = vec![0f32; 48 * unit * channels];
            let mut out16 = vec![0i16; 48 * unit * channels];
            for round in 0..1500 {
                let mut packet = Vec::new();
                if rng.below(10) == 0 {
                    packet = (0..rng.below(60)).map(|_| rng.next() as u8).collect();
                } else {
                    // The configuration says the mode, the bandwidth and
                    // the frame duration.
                    let config = rng.below(32) as u8;
                    for s in 0..streams {
                        let config = if rng.below(12) == 0 { rng.below(32) as u8 } else { config };
                        let code = if rng.below(4) == 0 { rng.below(4) as u8 } else { 0 };
                        packet.push((config << 3) | (((s < coupled) as u8) << 2) | code);
                        if code == 3 {
                            // The frame count byte, mostly with few frames.
                            packet.push(rng.next() as u8 & if rng.below(3) == 0 { 0xFF } else { 0x07 });
                        }
                        let bytes = rng.below(40);
                        // The size field that every stream but the last has.
                        if s + 1 != streams || rng.below(6) == 0 {
                            packet.push(if rng.below(10) == 0 { rng.next() as u8 } else { bytes as u8 });
                        }
                        packet.extend((0..bytes).map(|_| rng.next() as u8));
                    }
                    if rng.below(10) == 0 {
                        packet.truncate(rng.below(packet.len() + 1));
                    }
                }
                if let Ok(parts) = split_streams(&packet, streams) {
                    if parts.iter().enumerate().any(|(s, part)| toc(part.toc).stereo != (s < coupled)) {
                        continue;
                    }
                }
                // Mostly room for 120 ms, sometimes any number of samples.
                let frame_size = match rng.below(6) {
                    0 => 1 + rng.below(48 * unit + 2),
                    1 => unit * (1 + rng.below(50)),
                    _ => 48 * unit,
                };
                let kind = rng.below(6);
                let (fec, sixteen) = (kind == 1 || kind == 2, kind == 0 || kind == 2);
                let what = format!("{} at {fs} Hz, round {round}, frame_size {frame_size}, fec {fec}, 16 bits {sixteen}, packet {packet:?}", e.label);
                let (c, r) = if sixteen {
                    let c = c_dec.decode_i16(&packet, frame_size, fec).map(|pcm| pcm.len() / channels);
                    let r = if fec { r_dec.decode_fec_i16(&packet, frame_size, &mut out16) } else { r_dec.decode_i16(&packet, frame_size, &mut out16) };
                    (c, r)
                } else {
                    let c = c_dec.decode_float(&packet, frame_size, fec).map(|pcm| pcm.len() / channels);
                    let r = if fec { r_dec.decode_fec(&packet, frame_size, &mut out) } else { r_dec.decode(&packet, frame_size, &mut out) };
                    (c, r)
                };
                match (c, r) {
                    (Ok(c), Ok(r)) => {
                        assert_eq!(c, r, "{what}: samples per channel");
                        accepted += 1;
                    }
                    (Err(_), Err(_)) => rejected += 1,
                    (c, r) => panic!("{what}: C {c:?}, opus1 {r:?}"),
                }
            }
        }
    }
    println!("{accepted} accepted, {rejected} rejected");
    assert!(accepted >= 5000 && rejected >= 5000, "{accepted} accepted, {rejected} rejected");
}
