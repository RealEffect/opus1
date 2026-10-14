//! Second set of checks of `MultistreamDecoder` against the C libopus
//! multistream decoder, with layouts, signals and encoder settings that the
//! other `multistream_dec_*` tests do not use.
//!
//! What is checked:
//!
//! - layouts: mapping family 1 with 3, 5 and 7 channels, ambisonics with 9,
//!   11 and 16 channels, family 255 with 2 and 8 channels, two stereo streams
//!   with and without mono streams, and 127 stereo streams plus one mono
//!   stream;
//! - signals: white noise, and a sweep with clicks;
//! - encoder settings: low-delay application, low complexity, an encoder at
//!   16 and 24 kHz, a frame size and a bitrate that change from packet to
//!   packet, and a coding mode forced per stream so that stereo streams are
//!   coded with SILK and Hybrid;
//! - discontinuous transmission (DTX): streams without data in a packet;
//! - `reset_state` in the middle of a stream against `OPUS_RESET_STATE`;
//! - 16-bit output with lost packets and forward error correction (FEC),
//!   silent channels in 16-bit output, and float and 16-bit calls in turn;
//! - packets with damaged frame data that both decoders accept, and packets
//!   with random frame data;
//! - every packet of 1 byte and many of 2 and 3 bytes;
//! - arguments at the limits of their types.
//!
//! One finding is written into `changing_frame_sizes_and_bitrates_match_libopus`:
//! a stream without lost packets can still run the CELT loss concealment,
//! which is not sample-exact with libopus.

mod ms_support;
mod multistream_dec_common;

use audiopus_sys as sys;
use ms_support::*;
use multistream_dec_common::*;
use opus_rs::multistream::MultistreamDecoder;
use std::f64::consts::PI;

/// The limit for "the same samples as the C decoder", as in
/// `multistream_dec_fidelity.rs`.
const MIN_SNR_DB: f64 = 100.0;

const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;
const OPUS_SET_FORCE_CHANNELS_REQUEST: i32 = 4022;
const OPUS_SET_DTX_REQUEST: i32 = 4016;

/// A C multistream decoder with the controls that `ms_support::CMsDecoder`
/// does not offer.
struct RawCDecoder {
    st: *mut sys::OpusMSDecoder,
    channels: usize,
}

impl RawCDecoder {
    fn new(fs: i32, channels: usize, streams: usize, coupled: usize, mapping: &[u8]) -> Self {
        assert!(mapping.len() >= channels && channels <= 255);
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(fs, channels as i32, streams as i32, coupled as i32, mapping.as_ptr(), &mut err)
        };
        assert!(!st.is_null() && err == 0, "opus_multistream_decoder_create: {err}");
        Self { st, channels }
    }

    fn for_packets(e: &Encoded, fs: i32) -> Self {
        Self::new(fs, e.channels, e.streams, e.coupled, &e.mapping)
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

    /// `OPUS_RESET_STATE` on the multistream decoder.
    fn reset_state(&mut self) {
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
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

/// A sweep from 150 Hz up to a top frequency that differs per channel, with
/// a click every 170 ms. The clicks make CELT switch to short blocks.
fn sweep_with_clicks(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let top = 3000.0 + 1500.0 * (c % 6) as f64;
        let seconds = samples as f64 / fs as f64;
        let mut phase = 0.0f64;
        let click_period = fs as usize * 170 / 1000 + 37 * c;
        for i in 0..samples {
            let t = i as f64 / fs as f64;
            let hz = 150.0 + (top - 150.0) * (t / seconds);
            phase += 2.0 * PI * hz / fs as f64;
            let mut v = 0.2 * phase.sin();
            let since_click = i % click_period;
            if since_click < 48 {
                v += 0.6 * (1.0 - since_click as f64 / 48.0) * if since_click % 2 == 0 { 1.0 } else { -1.0 };
            }
            pcm[i * channels + c] = v as f32;
        }
    }
    pcm
}

/// The signals of this file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Input {
    /// White noise at -12 dB relative to full scale.
    Noise,
    Sweep,
    /// `ms_support::voiced`.
    Speech,
    /// Speech, then silence, then speech, a third of the time each.
    SpeechWithPause,
}

fn input(kind: Input, fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    match kind {
        Input::Noise => noise(channels, samples, 0.25, 77),
        Input::Sweep => sweep_with_clicks(fs, channels, samples),
        Input::Speech => voiced(fs, channels, samples),
        Input::SpeechWithPause => {
            let mut pcm = voiced(fs, channels, samples);
            let third = samples / 3 * channels;
            pcm[third..2 * third].fill(0.0);
            pcm
        }
    }
}

/// Encodes `milliseconds` of `kind` with `enc`. The frame size of packet `i`
/// is `frames[i % frames.len()]` samples at the encoder's rate, and its
/// bitrate `kbps_per_channel[i % kbps_per_channel.len()]`.
fn encode_with(
    mut enc: CMsEncoder,
    decoder_layout: Option<&[u8]>,
    kind: Input,
    frames: &[usize],
    kbps_per_channel: &[i32],
    milliseconds: usize,
    label: &str,
) -> Encoded {
    let channels = enc.channels;
    let samples = enc.fs as usize * milliseconds / 1000;
    let pcm = input(kind, enc.fs, channels, samples);
    let (mut packets, mut ranges) = (Vec::new(), Vec::new());
    let mut pos = 0;
    for i in 0.. {
        let frame = frames[i % frames.len()];
        if (pos + frame) * channels > pcm.len() {
            break;
        }
        enc.set_bitrate(kbps_per_channel[i % kbps_per_channel.len()] * 1000 * channels as i32);
        let packet = enc
            .encode_float(&pcm[pos * channels..(pos + frame) * channels], frame, max_packet_bytes(enc.streams))
            .unwrap_or_else(|e| panic!("{label}: C encoder: {e}"));
        ranges.push(enc.final_range());
        packets.push(packet);
        pos += frame;
    }
    let mapping = decoder_layout.map_or_else(|| enc.mapping.clone(), |m| m.to_vec());
    Encoded {
        channels: mapping.len(),
        streams: enc.streams,
        coupled: enc.coupled,
        mapping,
        family: 255,
        input_channels: channels,
        packets,
        ranges,
        label: format!("{label}, {kind:?}, frames {frames:?}, {kbps_per_channel:?} kb/s per channel"),
    }
}

/// For every packet of `e`: whether decoding it runs the CELT loss
/// concealment although nothing is lost. That happens in two cases:
///
/// - a CELT stream of the packet holds no data, because the bitrate left it
///   no byte: the decoder conceals that frame;
/// - a stream changes from CELT to SILK or Hybrid: unless the packet carries
///   a redundant CELT frame, the decoder conceals 5 ms of CELT and fades
///   from them into the new mode (libopus `opus_decode_frame`, `transition`).
///
/// The CELT concealment of opus-rs is close to libopus but not sample-exact,
/// so such a packet, and the one after it, cannot be held to the limit of
/// the others.
fn celt_concealment_packets(e: &Encoded) -> Vec<bool> {
    let mut previous: Vec<Option<Mode>> = vec![None; e.streams];
    e.packets
        .iter()
        .map(|p| {
            let mut conceals = false;
            for (s, part) in split_streams(p, e.streams).unwrap().iter().enumerate() {
                let mode = toc(part.toc).mode;
                conceals |= mode == Mode::Celt && part.payload_len <= 1;
                conceals |= mode != Mode::Celt && previous[s] == Some(Mode::Celt);
                previous[s] = Some(mode);
            }
            conceals
        })
        .collect()
}

/// Whether some packet of `e` runs the CELT concealment.
fn runs_celt_concealment(e: &Encoded) -> bool {
    celt_concealment_packets(e).contains(&true)
}

/// Decodes every packet of `e` at `fs` with both decoders, checks the number
/// of samples and the final range of every packet, and returns the SNR
/// against the C decoder for every packet: the smallest one over the
/// channels, in dB. A difference below -140 dB relative to full scale counts
/// as none.
fn packet_snrs(e: &Encoded, fs: i32) -> Vec<f64> {
    let (mut c_dec, mut r_dec) = e.decoders(fs);
    let max_frame = fs as usize / 25 * 3;
    let mut out = vec![0f32; max_frame * e.channels];
    let mut snrs = Vec::new();
    for (i, packet) in e.packets.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, packet {i} ({})", e.label, describe_packet(packet, e.streams));
        let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap_or_else(|err| panic!("{}: C decoder: {err}", what()));
        let n = r_dec.decode(packet, max_frame, &mut out).unwrap_or_else(|err| panic!("{}: opus-rs decoder: {err}", what()));
        assert_eq!(n * e.channels, c.len(), "{}: samples per channel", what());
        assert_eq!(r_dec.final_range(), e.ranges[i], "{}: final range", what());
        let mut worst = f64::INFINITY;
        for ch in 0..e.channels {
            let (mut signal, mut noise) = (0f64, 0f64);
            for (c_frame, r_frame) in c.chunks_exact(e.channels).zip(out.chunks_exact(e.channels)) {
                signal += (c_frame[ch] as f64).powi(2);
                noise += ((c_frame[ch] - r_frame[ch]) as f64).powi(2);
            }
            let silent = noise <= n as f64 * 1e-14;
            worst = worst.min(if silent { 200.0 } else { snr_db(signal, noise) });
        }
        snrs.push(worst);
    }
    snrs
}

/// Decodes every packet of `e` at `fs` with both decoders. Checks the number
/// of samples and the final range of every packet, and that every channel is
/// within `min_snr` dB of the C decoder. Returns the smallest SNR in dB.
fn compare_float(e: &Encoded, fs: i32, min_snr: f64) -> f64 {
    let (mut c_dec, mut r_dec) = e.decoders(fs);
    let max_frame = fs as usize / 25 * 3;
    let mut out = vec![0f32; max_frame * e.channels];
    let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
    for (i, packet) in e.packets.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, packet {i} ({})", e.label, describe_packet(packet, e.streams));
        let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap_or_else(|err| panic!("{}: C decoder: {err}", what()));
        let n = r_dec.decode(packet, max_frame, &mut out).unwrap_or_else(|err| panic!("{}: opus-rs decoder: {err}", what()));
        assert_eq!(n * e.channels, c.len(), "{}: samples per channel", what());
        assert_eq!(r_dec.final_range(), e.ranges[i], "{}: final range", what());
        assert_eq!(c_dec.final_range(), e.ranges[i], "{}: final range of the C decoder", what());
        c_all.extend_from_slice(&c);
        r_all.extend_from_slice(&out[..n * e.channels]);
    }
    let snrs = channel_snrs(&c_all, &r_all, e.channels);
    let worst = snrs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(worst >= min_snr, "{} at {fs} Hz: {worst:.1} dB from the C decoder (all channels: {snrs:.0?})", e.label);
    worst
}

/// Layouts that the other test files leave out, with noise and a sweep, at
/// 10 and 40 kb/s per channel.
#[test]
fn further_layouts_match_libopus() {
    // (channels, mapping family)
    let layouts = [(3, 1), (5, 1), (7, 1), (9, 2), (11, 2), (16, 2), (2, 255), (8, 255)];
    let mut worst = f64::INFINITY;
    let mut checked = 0;
    for (channels, family) in layouts {
        for (kind, kbps, frame, fs) in [(Input::Noise, 40, 960, 48000), (Input::Sweep, 10, 480, 16000), (Input::Sweep, 40, 1920, 24000)] {
            let enc = CMsEncoder::surround(FS, channels, family, APP_AUDIO).unwrap();
            let e = encode_with(enc, None, kind, &[frame], &[kbps], 400, &format!("{channels} channels, family {family}"));
            if e.has_mono_coded_stereo_stream() {
                continue;
            }
            assert!(!runs_celt_concealment(&e), "{}", e.label);
            worst = worst.min(compare_float(&e, fs, MIN_SNR_DB));
            checked += 1;
        }
    }
    println!("{checked} encoded signals, smallest SNR {worst:.0} dB");
    assert!(checked >= 20, "only {checked} cases without a mono-coded packet");
}

/// The frame size and the bitrate change from packet to packet, the encoder
/// runs at 16, 24 or 48 kHz, with the low-delay application or with low
/// complexity.
///
/// Many of these signals run the CELT concealment although nothing is lost
/// (see `celt_concealment_packets`): 2.5 and 5 ms frames are always CELT, so
/// a stream that codes its longer frames with SILK or Hybrid keeps changing
/// its mode, and at 9 kb/s per channel a 2.5 ms packet has no byte left for
/// some streams. Every packet must be the C decoder's, except such a packet
/// and the packet after it.
#[test]
fn changing_frame_sizes_and_bitrates_match_libopus() {
    // Frame sizes in samples at 48 kHz: all six, and those SILK can code.
    const ALL: &[usize] = &[960, 120, 2880, 480, 240, 1920, 960];
    const FROM_10_MS: &[usize] = &[960, 2880, 480, 1920, 960];
    // (encoder rate, application, complexity, frame sizes, kb/s per channel)
    let cases: [(i32, i32, i32, &[usize], [i32; 5]); 6] = [
        (48000, APP_LOWDELAY, 10, ALL, [48, 9, 20, 64, 14]),
        (48000, APP_LOWDELAY, 3, ALL, [48, 24, 32, 64, 40]),
        (48000, APP_AUDIO, 0, ALL, [48, 9, 20, 64, 14]),
        (24000, APP_VOIP, 5, ALL, [48, 9, 20, 64, 14]),
        (16000, APP_AUDIO, 10, FROM_10_MS, [48, 9, 20, 64, 14]),
        (16000, APP_VOIP, 2, FROM_10_MS, [12, 9, 20, 32, 14]),
    ];
    let (mut exact, mut near_concealment, mut inexact) = (0usize, 0usize, 0usize);
    let (mut worst, mut worst_inexact) = (f64::INFINITY, f64::INFINITY);
    for (enc_fs, application, complexity, frames, kbps) in cases {
        for (channels, family) in [(1, 1), (6, 1), (3, 255), (4, 2)] {
            for kind in [Input::Sweep, Input::Speech] {
                let mut enc = CMsEncoder::surround(enc_fs, channels, family, application).unwrap();
                enc.ctl(OPUS_SET_COMPLEXITY_REQUEST, complexity);
                let frames: Vec<usize> = frames.iter().map(|f| f * enc_fs as usize / 48000).collect();
                let label = format!("{channels} channels, family {family}, encoder at {enc_fs} Hz, application {application}, complexity {complexity}");
                let e = encode_with(enc, None, kind, &frames, &kbps, 900, &label);
                assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
                let conceals = celt_concealment_packets(&e);
                for fs in [8000, 48000] {
                    for (i, snr) in packet_snrs(&e, fs).into_iter().enumerate() {
                        let near = conceals[i] || (i > 0 && conceals[i - 1]);
                        near_concealment += near as usize;
                        if snr >= MIN_SNR_DB {
                            exact += 1;
                            worst = worst.min(snr);
                            continue;
                        }
                        let what = format!("{} at {fs} Hz, packet {i} ({}): {snr:.1} dB from the C decoder", e.label, describe_packet(&e.packets[i], e.streams));
                        assert!(near, "{what}, and no CELT concealment in this packet or the one before");
                        // A concealed part still has the level of libopus's.
                        assert!(snr >= 20.0, "{what}");
                        inexact += 1;
                        worst_inexact = worst_inexact.min(snr);
                    }
                }
            }
        }
    }
    println!(
        "{exact} packets within {MIN_SNR_DB} dB of the C decoder (smallest SNR {worst:.0} dB); {inexact} packets below, all of the {near_concealment} at or after a CELT concealment (smallest SNR {worst_inexact:.0} dB)"
    );
    assert!(exact >= 3000, "only {exact} packets match the C decoder");
}

/// A layout without a mapping family leaves the coding mode of every stream
/// to the stream encoder. Here the mode is forced per stream and changes
/// every 8 packets, and the stereo streams are forced to code two channels.
/// The packets then hold SILK and Hybrid stereo streams, and streams that
/// change their mode at different times.
#[test]
fn forced_modes_per_stream_match_libopus() {
    let modes = [MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY];
    let mut worst = f64::INFINITY;
    let mut seen = std::collections::BTreeSet::new();
    for (frame, kbps) in [(960usize, 24), (480, 32), (1920, 16)] {
        // Two stereo streams and two mono streams.
        let mut enc = CMsEncoder::generic(FS, 6, 4, 2, &[0, 1, 2, 3, 4, 5], APP_VOIP).unwrap();
        enc.set_bitrate(kbps * 1000 * 6);
        for s in 0..2 {
            assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_CHANNELS_REQUEST, 2), 0);
        }
        let pcm = input(Input::Speech, FS, 6, FS as usize * 3);
        let (mut packets, mut ranges) = (Vec::new(), Vec::new());
        for (i, chunk) in pcm.chunks_exact(frame * 6).enumerate() {
            for s in 0..4 {
                // Stream `s` changes its mode every 8 packets, 2 packets
                // after stream `s - 1`.
                let turn = (i + 2 * s) / 8 + s;
                let mode = modes[turn % 3];
                // SILK alone codes up to wideband, and Hybrid starts above
                // it, so the bandwidth has to fit the mode.
                let bandwidth = match mode {
                    MODE_SILK_ONLY => [sys::OPUS_BANDWIDTH_WIDEBAND, sys::OPUS_BANDWIDTH_NARROWBAND, sys::OPUS_BANDWIDTH_MEDIUMBAND][(turn / 3) % 3],
                    MODE_HYBRID => [sys::OPUS_BANDWIDTH_FULLBAND, sys::OPUS_BANDWIDTH_SUPERWIDEBAND][(turn / 3) % 2],
                    _ => sys::OPUS_AUTO,
                };
                assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_MODE_REQUEST, mode), 0);
                assert_eq!(enc.stream_ctl(s, sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth), 0);
            }
            packets.push(enc.encode_float(chunk, frame, max_packet_bytes(4)).unwrap());
            ranges.push(enc.final_range());
        }
        for p in &packets {
            for (s, part) in split_streams(p, 4).unwrap().iter().enumerate() {
                let t = toc(part.toc);
                seen.insert((s < 2, format!("{:?}", t.mode)));
            }
        }
        let e = Encoded {
            channels: 6,
            streams: 4,
            coupled: 2,
            mapping: vec![0, 1, 2, 3, 4, 5],
            family: 255,
            input_channels: 6,
            packets,
            ranges,
            label: format!("forced modes, {} ms frames, {kbps} kb/s per channel", frame as f64 / 48.0),
        };
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for fs in [16000, 48000] {
            worst = worst.min(compare_float(&e, fs, MIN_SNR_DB));
        }
    }
    println!("stream kinds (stereo, mode): {seen:?}; smallest SNR {worst:.0} dB");
    // Every mode was seen in a stereo and in a mono stream.
    assert_eq!(seen.len(), 6, "{seen:?}");
}

/// With discontinuous transmission the encoder stops sending data for a
/// stream that is silent: the stream is then a single TOC byte, which a
/// decoder treats as a lost frame of that stream.
#[test]
fn streams_without_data_from_dtx_match_libopus() {
    let mut empty_streams = 0;
    let mut worst = f64::INFINITY;
    for (channels, family, application, kbps) in [(3, 255, APP_VOIP, 16), (1, 1, APP_VOIP, 12), (6, 1, APP_AUDIO, 24), (5, 255, APP_AUDIO, 32)] {
        let mut enc = CMsEncoder::surround(FS, channels, family, application).unwrap();
        enc.ctl(OPUS_SET_DTX_REQUEST, 1);
        let e = encode_with(enc, None, Input::SpeechWithPause, &[960], &[kbps], 3000, &format!("DTX, {channels} channels, family {family}"));
        if e.has_mono_coded_stereo_stream() {
            continue;
        }
        for p in &e.packets {
            empty_streams += split_streams(p, e.streams).unwrap().iter().filter(|s| s.packet.len() <= 1).count();
        }
        for fs in [16000, 48000] {
            worst = worst.min(compare_float(&e, fs, MIN_SNR_DB));
        }
    }
    println!("{empty_streams} streams without data, smallest SNR {worst:.0} dB");
    assert!(empty_streams >= 50, "only {empty_streams} streams without data");
}

/// `reset_state` in the middle of a stream gives what `OPUS_RESET_STATE`
/// gives in the C decoder: silence when a packet is lost right after it, and
/// the same samples for the packets that follow.
#[test]
fn reset_in_the_middle_of_a_stream_matches_libopus() {
    for (channels, family, application, kbps) in [(6, 1, APP_AUDIO, 32), (3, 255, APP_VOIP, 12), (4, 2, APP_AUDIO, 24)] {
        let enc = CMsEncoder::surround(FS, channels, family, application).unwrap();
        let e = encode_with(enc, None, Input::Speech, &[960], &[kbps], 1200, &format!("reset, {channels} channels, family {family}"));
        for fs in [12000, 48000] {
            let mut c_dec = RawCDecoder::for_packets(&e, fs);
            let (_, mut r_dec) = e.decoders(fs);
            let max_frame = fs as usize / 25 * 3;
            let frame = fs as usize / 50;
            let mut out = vec![0f32; max_frame * channels];
            let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
            for (i, packet) in e.packets.iter().enumerate() {
                let what = format!("{} at {fs} Hz, packet {i}", e.label);
                // Reset before packets 13, 26, 39 and so on. After every
                // second reset a packet is lost before the next one arrives.
                if i > 0 && i % 13 == 0 {
                    c_dec.reset_state();
                    r_dec.reset_state();
                    assert_eq!(c_dec.final_range(), 0);
                    assert_eq!(r_dec.final_range(), 0, "{what}: final range after a reset");
                    if i % 26 == 0 {
                        let c = c_dec.decode_float(&[], frame, false).unwrap();
                        out.fill(0.5);
                        assert_eq!(r_dec.decode(&[], frame, &mut out), Ok(frame), "{what}");
                        assert!(c.iter().all(|&v| v == 0.0), "{what}: the C decoder is not silent after a reset");
                        assert!(out[..frame * channels].iter().all(|&v| v == 0.0), "{what}: not silent after a reset");
                        // The same for a rebuilt frame: a new decoder has no
                        // mode yet, so the FEC data of the packet is used
                        // where there is some.
                        let c = c_dec.decode_float(packet, frame, true).unwrap();
                        assert_eq!(r_dec.decode_fec(packet, frame, &mut out), Ok(frame), "{what}");
                        let snr = min_channel_snr(&c, &out[..frame * channels], channels);
                        assert!(snr >= MIN_SNR_DB, "{what}: FEC after a reset is {snr:.1} dB from the C decoder");
                        assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range of FEC after a reset");
                    }
                }
                let c = c_dec.decode_float(packet, max_frame, false).unwrap();
                let n = r_dec.decode(packet, max_frame, &mut out).unwrap();
                assert_eq!(c.len(), n * channels, "{what}");
                assert_eq!(r_dec.final_range(), e.ranges[i], "{what}: final range");
                c_all.extend_from_slice(&c);
                r_all.extend_from_slice(&out[..n * channels]);
            }
            let snrs = channel_snrs(&c_all, &r_all, channels);
            assert!(snrs.iter().all(|&snr| snr >= MIN_SNR_DB), "{} at {fs} Hz: {snrs:.0?} dB", e.label);
        }
    }
}

/// 16-bit output of a stream with lost packets, against the C decoder's
/// 16-bit output. The streams are SILK-only, which both decoders conceal and
/// rebuild alike, and the signal goes beyond full scale. Lost packets are
/// concealed and rebuilt from FEC data in turn.
#[test]
fn sixteen_bit_output_with_losses_matches_libopus() {
    let mut enc = CMsEncoder::surround(FS, 3, 255, APP_VOIP).unwrap();
    enc.set_inband_fec(true).set_packet_loss_perc(20);
    for s in 0..3 {
        // SILK alone codes up to wideband.
        assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK_ONLY), 0);
        assert_eq!(enc.stream_ctl(s, sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, sys::OPUS_BANDWIDTH_WIDEBAND), 0);
    }
    let channels = 3;
    let samples = FS as usize * 2;
    let pcm: Vec<f32> = voiced(FS, channels, samples).iter().map(|v| v * 6.0).collect();
    enc.set_bitrate(16_000 * 3);
    let packets: Vec<Vec<u8>> = pcm.chunks_exact(960 * channels).map(|c| enc.encode_float(c, 960, max_packet_bytes(3)).unwrap()).collect();
    for fs in [8000, 16000, 48000] {
        let mut c_dec = CMsDecoder::new(fs, channels, 3, 0, &[0, 1, 2]).unwrap();
        let mut r_dec = MultistreamDecoder::with_layout(fs, channels, 3, 0, &[0, 1, 2]).unwrap();
        let frame = fs as usize / 50;
        let max_frame = fs as usize / 25 * 3;
        let mut out = vec![0i16; max_frame * channels];
        let (mut differing, mut total, mut at_full_scale, mut lost_calls) = (0usize, 0usize, 0usize, 0usize);
        let mut check = |c: &[i16], r: &[i16], what: &str| {
            assert_eq!(c.len(), r.len(), "{what}: samples");
            for (&c, &r) in c.iter().zip(r) {
                let d = (c as i32 - r as i32).abs();
                assert!(d <= 1, "{what}: C {c}, opus-rs {r}");
                differing += (d != 0) as usize;
                at_full_scale += (c == i16::MAX || c == i16::MIN) as usize;
            }
            total += c.len();
        };
        for (i, packet) in packets.iter().enumerate() {
            let what = format!("{fs} Hz, packet {i}");
            assert!(split_streams(packet, 3).unwrap().iter().all(|s| toc(s.toc).mode == Mode::Silk), "{what}: not SILK");
            // Packets 4, 9, 14, ... are lost.
            if i % 5 == 4 {
                continue;
            }
            if i > 0 && i % 5 == 0 {
                lost_calls += 1;
                // The lost packet before this one: concealed or rebuilt.
                if i % 10 == 0 {
                    let c = c_dec.decode_i16(None, frame, false).unwrap();
                    let n = r_dec.decode_i16(&[], frame, &mut out).unwrap();
                    check(&c, &out[..n * channels], &format!("{what}, concealed"));
                } else {
                    let c = c_dec.decode_i16(Some(packet), frame, true).unwrap();
                    let n = r_dec.decode_fec_i16(packet, frame, &mut out).unwrap();
                    check(&c, &out[..n * channels], &format!("{what}, rebuilt"));
                }
                assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range after a loss");
            }
            let c = c_dec.decode_i16(Some(packet), max_frame, false).unwrap();
            let n = r_dec.decode_i16(packet, max_frame, &mut out).unwrap();
            check(&c, &out[..n * channels], &what);
            assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range");
        }
        println!("{fs} Hz: {differing} of {total} samples differ by one step, {at_full_scale} at full scale, {lost_calls} lost packets");
        assert!(lost_calls >= 15);
        assert!(at_full_scale > 0, "{fs} Hz: the signal did not reach full scale");
        assert!(differing * 100 <= total, "{fs} Hz: {differing} of {total} samples differ from the C decoder");
    }
}

/// 16-bit output with a layout in which channels are silent and repeated:
/// a silent channel is written as zeros over what the buffer held, repeated
/// channels are equal, and nothing is written behind the decoded samples.
/// The layouts with a stereo stream that `sixteen_bit_output_matches_libopus`
/// leaves out are decoded here, with the stereo stream forced to two
/// channels.
#[test]
fn sixteen_bit_output_routes_and_silences_like_libopus() {
    for (kbps, gain) in [(12, 1.0f32), (48, 6.0)] {
        // One stereo stream and one mono stream.
        let mut enc = CMsEncoder::generic(FS, 3, 2, 1, &[0, 1, 2], APP_AUDIO).unwrap();
        assert_eq!(enc.stream_ctl(0, OPUS_SET_FORCE_CHANNELS_REQUEST, 2), 0);
        enc.set_bitrate(kbps * 1000 * 3);
        let pcm: Vec<f32> = sweep_with_clicks(FS, 3, FS as usize / 2).iter().map(|v| v * gain).collect();
        let packets: Vec<Vec<u8>> = pcm.chunks_exact(480 * 3).map(|c| enc.encode_float(c, 480, max_packet_bytes(2)).unwrap()).collect();
        // Outputs: right, silent, mono, left, mono, silent, right.
        let mapping = [1u8, 255, 2, 0, 2, 255, 1];
        let ch = mapping.len();
        for fs in [24000, 48000] {
            let mut c_dec = CMsDecoder::new(fs, ch, 2, 1, &mapping).unwrap();
            let mut r_dec = MultistreamDecoder::with_layout(fs, ch, 2, 1, &mapping).unwrap();
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0i16; max_frame * ch];
            let mut loud = 0usize;
            for (i, packet) in packets.iter().enumerate() {
                let what = format!("{kbps} kb/s per channel, gain {gain}, {fs} Hz, packet {i}");
                out.fill(1234);
                let c = c_dec.decode_i16(Some(packet), max_frame, false).unwrap();
                let n = r_dec.decode_i16(packet, max_frame, &mut out).unwrap();
                assert_eq!(c.len(), n * ch, "{what}");
                for (c_frame, r_frame) in c.chunks_exact(ch).zip(out[..n * ch].chunks_exact(ch)) {
                    for k in 0..ch {
                        assert!((c_frame[k] as i32 - r_frame[k] as i32).abs() <= 1, "{what}: channel {k}: C {}, opus-rs {}", c_frame[k], r_frame[k]);
                    }
                    assert_eq!((r_frame[1], r_frame[5]), (0, 0), "{what}: a silent channel is not zero");
                    assert_eq!(r_frame[0], r_frame[6], "{what}: the right channel differs between its outputs");
                    assert_eq!(r_frame[2], r_frame[4], "{what}: the mono stream differs between its outputs");
                    loud += (r_frame[3].unsigned_abs() >= 32767) as usize;
                }
                assert!(out[n * ch..].iter().all(|&v| v == 1234), "{what}: written behind the decoded samples");
            }
            if gain > 1.0 {
                assert!(loud > 0, "{kbps} kb/s per channel, {fs} Hz: the signal did not reach full scale");
            }
        }
    }
}

/// One decoder may be asked for float and for 16-bit output in turn. The
/// soft clip of the 16-bit output keeps a memory per stream, which a float
/// call clears, as in libopus.
#[test]
fn float_and_sixteen_bit_calls_alternate_like_libopus() {
    // Two stereo streams, forced to two channels, with a loud signal.
    let mut enc = CMsEncoder::generic(FS, 4, 2, 2, &[0, 1, 2, 3], APP_AUDIO).unwrap();
    for s in 0..2 {
        assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_CHANNELS_REQUEST, 2), 0);
    }
    enc.set_bitrate(4 * 40_000);
    let pcm: Vec<f32> = voiced(FS, 4, FS as usize).iter().map(|v| v * 7.0).collect();
    let packets: Vec<Vec<u8>> = pcm.chunks_exact(960 * 4).map(|c| enc.encode_float(c, 960, max_packet_bytes(2)).unwrap()).collect();
    for fs in [16000, 48000] {
        let mut c_dec = CMsDecoder::new(fs, 4, 2, 2, &[0, 1, 2, 3]).unwrap();
        let mut r_dec = MultistreamDecoder::with_layout(fs, 4, 2, 2, &[0, 1, 2, 3]).unwrap();
        let max_frame = fs as usize / 25 * 3;
        let mut out = vec![0f32; max_frame * 4];
        let mut out16 = vec![0i16; max_frame * 4];
        let (mut c_float, mut r_float) = (Vec::new(), Vec::new());
        let (mut differing, mut total, mut at_full_scale) = (0usize, 0usize, 0usize);
        for (i, packet) in packets.iter().enumerate() {
            let what = format!("{fs} Hz, packet {i}");
            // Float for packets 0, 3, 4, 7, 8, ..., 16 bits for the others.
            if i % 4 == 0 || i % 4 == 3 {
                let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap();
                let n = r_dec.decode(packet, max_frame, &mut out).unwrap();
                assert_eq!(c.len(), n * 4, "{what}");
                c_float.extend_from_slice(&c);
                r_float.extend_from_slice(&out[..n * 4]);
            } else {
                let c = c_dec.decode_i16(Some(packet), max_frame, false).unwrap();
                let n = r_dec.decode_i16(packet, max_frame, &mut out16).unwrap();
                assert_eq!(c.len(), n * 4, "{what}");
                for (&c, &r) in c.iter().zip(&out16[..n * 4]) {
                    let d = (c as i32 - r as i32).abs();
                    assert!(d <= 1, "{what}: C {c}, opus-rs {r}");
                    differing += (d != 0) as usize;
                    at_full_scale += (c == i16::MAX || c == i16::MIN) as usize;
                }
                total += c.len();
            }
            assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range");
        }
        let snr = min_channel_snr(&c_float, &r_float, 4);
        println!("{fs} Hz: float {snr:.0} dB, {differing} of {total} 16-bit samples differ by one step, {at_full_scale} at full scale");
        assert!(snr >= MIN_SNR_DB, "{fs} Hz: float output {snr:.1} dB from the C decoder");
        assert!(at_full_scale > 0, "{fs} Hz: the signal did not reach full scale");
        assert!(differing * 100 <= total, "{fs} Hz: {differing} of {total} samples differ from the C decoder");
    }
}

/// The widest layout with stereo streams: 127 stereo streams and one mono
/// stream give 255 decoded channels.
#[test]
fn layout_with_127_stereo_streams_matches_libopus() {
    let mapping: Vec<u8> = (0..255).map(|c| c as u8).collect();
    assert!(CMsDecoder::new(48000, 255, 128, 127, &mapping).is_ok());
    // One stream more is one decoded channel too many.
    assert!(CMsDecoder::new(48000, 255, 129, 127, &mapping).is_err());
    assert!(MultistreamDecoder::with_layout(48000, 255, 129, 127, &mapping).is_err());
    assert!(CMsDecoder::new(48000, 255, 128, 128, &mapping).is_err());
    assert!(MultistreamDecoder::with_layout(48000, 255, 128, 128, &mapping).is_err());

    let mut enc = CMsEncoder::generic(FS, 255, 128, 127, &mapping, APP_AUDIO).unwrap();
    for s in 0..127 {
        assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_CHANNELS_REQUEST, 2), 0);
    }
    // The decoders put the channels in reverse order.
    let reversed: Vec<u8> = mapping.iter().rev().copied().collect();
    let e = encode_with(enc, Some(&reversed), Input::Sweep, &[960], &[16], 120, "127 stereo streams and 1 mono stream");
    assert_eq!((e.streams, e.coupled, e.packets.len()), (128, 127, 6));
    assert!(!e.has_mono_coded_stereo_stream());
    let worst = compare_float(&e, 24000, MIN_SNR_DB);
    println!("smallest SNR {worst:.0} dB");

    // A lost packet, a rebuilt packet and a malformed packet on this layout.
    let (mut c_dec, mut r_dec) = e.decoders(24000);
    let mut out = vec![0f32; 2880 * 255];
    for packet in &e.packets[..3] {
        c_dec.decode_float(Some(packet), 2880, false).unwrap();
        r_dec.decode(packet, 2880, &mut out).unwrap();
    }
    assert_eq!(c_dec.decode_float(None, 480, false).map(|pcm| pcm.len()), Ok(480 * 255));
    assert_eq!(r_dec.decode(&[], 480, &mut out), Ok(480));
    assert_eq!(c_dec.decode_float(Some(&e.packets[4]), 480, true).map(|pcm| pcm.len()), Ok(480 * 255));
    assert_eq!(r_dec.decode_fec(&e.packets[4], 480, &mut out), Ok(480));
    // 254 bytes are one byte less than 128 streams need.
    assert_eq!(c_dec.decode_float(Some(&[0xFCu8; 254]), 2880, false), Err(OPUS_INVALID_PACKET));
    assert!(r_dec.decode(&[0xFCu8; 254], 2880, &mut out).is_err());
    // 127 stereo CELT streams and a mono CELT stream, all without data.
    let mut empty = Vec::new();
    for _ in 0..127 {
        empty.extend_from_slice(&[0xFC, 0]);
    }
    empty.push(0xF8);
    assert_eq!(empty.len(), 255);
    assert_eq!(c_dec.decode_float(Some(&empty), 2880, false).map(|pcm| pcm.len()), Ok(480 * 255));
    assert_eq!(r_dec.decode(&empty, 2880, &mut out), Ok(480));
}

/// Damage inside the frame data leaves the framing intact, so both decoders
/// accept the packet. They must then read the same bits: the final range is
/// equal, and so is the decoded signal.
#[test]
fn packets_with_damaged_frame_data_decode_like_libopus() {
    let sources = [
        (6, 1, APP_AUDIO, 32, 960usize),
        (3, 255, APP_VOIP, 12, 960),
        (5, 255, APP_VOIP, 24, 480),
        (4, 2, APP_AUDIO, 24, 240),
        (8, 1, APP_AUDIO, 16, 1920),
    ];
    let mut rng = Rng(0xBAD5EED);
    let (mut damaged, mut range_equal, mut exact) = (0usize, 0usize, 0usize);
    let mut worst = f64::INFINITY;
    for (channels, family, application, kbps, frame) in sources {
        let enc = CMsEncoder::surround(FS, channels, family, application).unwrap();
        let e = encode_with(enc, None, Input::Speech, &[frame], &[kbps], 1500, &format!("damaged, {channels} channels, family {family}"));
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for fs in [16000, 48000] {
            let (mut c_dec, mut r_dec) = e.decoders(fs);
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0f32; max_frame * channels];
            for (i, packet) in e.packets.iter().enumerate() {
                let mut p = packet.clone();
                // Every third packet: flip 1 to 3 bits in the frame data of
                // some streams. The TOC bytes, the size fields and the
                // padding stay as they are.
                let damage = i % 3 == 2;
                if damage {
                    let parts = split_streams(packet, e.streams).unwrap();
                    let mut start = 0;
                    let mut spans = Vec::new();
                    for part in &parts {
                        let frames_end = start + part.used - part.padding;
                        if part.payload_len >= 2 {
                            spans.push((frames_end - part.payload_len, frames_end));
                        }
                        start += part.used;
                    }
                    for _ in 0..1 + rng.below(3) {
                        let (from, to) = spans[rng.below(spans.len())];
                        p[from + rng.below(to - from)] ^= 1 << rng.below(8);
                    }
                }
                let what = format!("{} at {fs} Hz, packet {i}", e.label);
                let c = c_dec.decode_float(Some(&p), max_frame, false);
                let r = r_dec.decode(&p, max_frame, &mut out);
                match (c, r) {
                    (Ok(c), Ok(n)) => {
                        assert_eq!(c.len(), n * channels, "{what}: samples");
                        if damage {
                            damaged += 1;
                            let snr = min_channel_snr(&c, &out[..n * channels], channels);
                            worst = worst.min(snr);
                            exact += (snr >= MIN_SNR_DB) as usize;
                            range_equal += (r_dec.final_range() == c_dec.final_range()) as usize;
                        }
                    }
                    (Err(_), Err(_)) => {}
                    (c, r) => panic!("{what}: C {:?}, opus-rs {r:?}", c.map(|pcm| pcm.len())),
                }
            }
        }
    }
    println!("{damaged} damaged packets accepted: {range_equal} with the C decoder's final range, {exact} within {MIN_SNR_DB} dB, smallest SNR {worst:.1} dB");
    assert!(damaged >= 300, "only {damaged} damaged packets were accepted");
    assert_eq!(range_equal, damaged, "damaged packets with another final range than the C decoder's");
    // Measured: all of them. A damaged SILK or CELT frame may switch on a
    // part of the decoder that is not sample-exact, so 2 % may differ.
    assert!(exact * 100 >= damaged * 98, "only {exact} of {damaged} damaged packets decode like the C decoder");
}

/// Random bytes as frame data, behind valid framing and with the channel
/// count of each stream's decoder. Every such packet is a valid Opus packet,
/// so both decoders accept it, and they must read the same bits: the final
/// range is equal after every packet.
///
/// This depends on the range decoder, not on the multistream code: libopus
/// `ec_dec_uint` (entdec.c) returns `ft - 1` and sets the error flag when
/// the bits it read give a larger value. Until `RangeCoder::dec_uint`
/// (src/range_coder.rs) did the same, it returned the larger value, and
/// about 1 packet in 200 ended with another final range. An encoder never
/// writes such a value, so encoded streams were not affected.
#[test]
fn random_frame_data_has_the_final_range_of_libopus() {
    // (channels, streams, coupled streams, mapping)
    let layouts: [(usize, usize, usize, &[u8]); 3] = [(6, 4, 2, &[0, 4, 1, 2, 3, 5]), (3, 3, 0, &[0, 1, 2]), (2, 1, 1, &[0, 1])];
    let mut rng = Rng(0x5EED_1234);
    let (mut packets, mut other_range) = (0usize, 0usize);
    for (channels, streams, coupled, mapping) in layouts {
        for fs in [16000, 48000] {
            let mut c_dec = CMsDecoder::new(fs, channels, streams, coupled, mapping).unwrap();
            let mut r_dec = MultistreamDecoder::with_layout(fs, channels, streams, coupled, mapping).unwrap();
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0f32; max_frame * channels];
            for round in 0..2000 {
                // One mode, bandwidth and frame size for all streams of a
                // packet, so that they have the same duration.
                let config = rng.below(32) as u8;
                let mut packet = Vec::new();
                for s in 0..streams {
                    let toc_byte = (config << 3) | (((s < coupled) as u8) << 2);
                    let long = rng.below(4) == 0;
                    let len = 2 + rng.below(if long { 300 } else { 60 });
                    let frame: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
                    let mut buf = vec![0u8; len + 8];
                    let n = opus_rs::packet::write(toc_byte, &[&frame], &mut buf, s + 1 != streams, false).unwrap();
                    packet.extend_from_slice(&buf[..n]);
                }
                let what = format!("{streams} streams at {fs} Hz, packet {round}, {}", toc(config << 3));
                let c = c_dec.decode_float(Some(&packet), max_frame, false).unwrap_or_else(|err| panic!("{what}: C decoder: {err}"));
                let n = r_dec.decode(&packet, max_frame, &mut out).unwrap_or_else(|err| panic!("{what}: {err}"));
                assert_eq!(c.len(), n * channels, "{what}: samples");
                assert!(out[..n * channels].iter().all(|v| v.is_finite()), "{what}: output is not finite");
                packets += 1;
                other_range += (r_dec.final_range() != c_dec.final_range()) as usize;
            }
        }
    }
    println!("{packets} packets of random frame data, {other_range} with another final range than the C decoder");
    assert_eq!(other_range, 0, "of {packets} packets");
}

/// Every packet of 1 byte, and many of 2 and 3 bytes, on layouts with one
/// and two streams, through all four decoding functions and with several
/// frame sizes: the verdict and the number of samples are libopus's.
///
/// A layout with one stream gets the packets of 1 and 2 bytes, the layout
/// with two streams gets those of 3 bytes as well. Every frame then has at
/// most 1 byte. A decoder conceals such a frame, so the channel count in the
/// TOC byte does not matter.
#[test]
fn shortest_packets_get_the_verdict_of_libopus() {
    // (channels, streams, coupled streams, mapping)
    let layouts: [(usize, usize, usize, &[u8]); 3] = [(1, 1, 0, &[0]), (2, 1, 1, &[0, 1]), (3, 2, 1, &[0, 1, 2])];
    // Second bytes: frame sizes, frame counts with and without the padding
    // and VBR flags, and a 2-byte size field cut short.
    let seconds = [0u8, 1, 2, 6, 7, 0x41, 0x86, 252];
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (channels, streams, coupled, mapping) in layouts {
        let fs = 16000;
        let mut c_dec = CMsDecoder::new(fs, channels, streams, coupled, mapping).unwrap();
        let mut r_dec = MultistreamDecoder::with_layout(fs, channels, streams, coupled, mapping).unwrap();
        let mut out = vec![0f32; 1920 * channels];
        let mut out16 = vec![0i16; 1920 * channels];
        let mut packets: Vec<Vec<u8>> = (0..=255u8).map(|toc| vec![toc]).collect();
        for toc in 0..=255u8 {
            for second in seconds {
                packets.push(vec![toc, second]);
                // Three bytes: the same TOC byte for a second stream, and a
                // few others.
                for third in [toc, toc ^ 0x08, 0x00, 0xFF] {
                    packets.push(vec![toc, second, third]);
                }
            }
        }
        for packet in packets.iter().filter(|p| p.len() <= 2 || streams == 2) {
            // 2.5, 10, 20 and 120 ms, and a size that is no multiple of 2.5 ms.
            for frame_size in [40usize, 160, 320, 1920, 100] {
                for kind in 0..4 {
                    let (fec, sixteen) = (kind & 1 != 0, kind & 2 != 0);
                    let (c, r) = if sixteen {
                        let c = c_dec.decode_i16(Some(packet), frame_size, fec).map(|pcm| pcm.len() / channels);
                        let r = if fec { r_dec.decode_fec_i16(packet, frame_size, &mut out16) } else { r_dec.decode_i16(packet, frame_size, &mut out16) };
                        (c, r)
                    } else {
                        let c = c_dec.decode_float(Some(packet), frame_size, fec).map(|pcm| pcm.len() / channels);
                        let r = if fec { r_dec.decode_fec(packet, frame_size, &mut out) } else { r_dec.decode(packet, frame_size, &mut out) };
                        (c, r)
                    };
                    match (c, r) {
                        (Ok(c), Ok(r)) => {
                            assert_eq!(c, r, "{streams} streams, packet {packet:02X?}, frame_size {frame_size}, call {kind}: samples");
                            accepted += 1;
                        }
                        (Err(_), Err(_)) => rejected += 1,
                        (c, r) => panic!("{streams} streams, packet {packet:02X?}, frame_size {frame_size}, call {kind}: C {c:?}, opus-rs {r:?}"),
                    }
                }
            }
        }
    }
    println!("{accepted} calls accepted, {rejected} rejected");
    assert!(accepted >= 10_000 && rejected >= 10_000, "{accepted} accepted, {rejected} rejected");
}

/// Arguments at the limits of their types never panic.
#[test]
fn extreme_arguments_are_errors() {
    let big = [usize::MAX, usize::MAX / 2 + 1, u32::MAX as usize, 65536, 256];
    for &a in &big {
        for &b in &big {
            assert!(MultistreamDecoder::with_layout(48000, a, b, 0, &[0; 4]).is_err());
            assert!(MultistreamDecoder::with_layout(48000, 2, a, b, &[0, 1]).is_err());
            assert!(MultistreamDecoder::with_layout(48000, a, 1, b, &[0, 1]).is_err());
        }
    }
    // The mapping is shorter than the channel count.
    assert!(MultistreamDecoder::with_layout(48000, 3, 2, 1, &[0, 1]).is_err());
    assert!(MultistreamDecoder::with_layout(48000, 1, 1, 0, &[]).is_err());

    let mut dec = MultistreamDecoder::with_layout(8000, 3, 2, 1, &[0, 1, 2]).unwrap();
    let mut out = vec![0f32; 960 * 3];
    let mut out16 = vec![0i16; 960 * 3];
    let packets: [&[u8]; 6] = [&[], &[0], &[0xFF], &[0xFF; 2], &[0x04, 0, 0x00], &[0xFF; 4000]];
    for packet in packets {
        for frame_size in [0usize, 1, 19, 20, 960, 961, 5760, usize::MAX / 3, usize::MAX] {
            // No room at all, room for less than a channel set, enough room.
            for room in [0usize, 2, 59, 60, 960 * 3] {
                let _ = dec.decode(packet, frame_size, &mut out[..room]);
                let _ = dec.decode_fec(packet, frame_size, &mut out[..room]);
                let _ = dec.decode_i16(packet, frame_size, &mut out16[..room]);
                let _ = dec.decode_fec_i16(packet, frame_size, &mut out16[..room]);
            }
        }
    }
    // The decoder still works: 20 samples are 2.5 ms at 8 kHz.
    assert_eq!(dec.decode(&[], 20, &mut out), Ok(20));
    // Room for 20 samples per channel is too little for a lost 5 ms.
    assert!(dec.decode(&[], 40, &mut out[..60]).is_err());
    // More than 120 ms counts as 120 ms, which is 960 samples at 8 kHz.
    assert_eq!(dec.decode(&[], usize::MAX, &mut out), Ok(960));
    assert!(dec.decode(&[], usize::MAX, &mut out[..960 * 3 - 1]).is_err());
}
