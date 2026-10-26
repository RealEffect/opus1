//! Packets of `OpusEncoder` against the fixed-point build of libopus 1.6.1.
//!
//! With the `Voip` application and 16-bit input, the SILK-only packets of
//! opus1 are byte for byte those of the fixed-point build: its SILK
//! encoder is a port of `silk/fixed`, and at 8 to 24 kHz it filters the
//! input with the integer `hp_cutoff` of that build. At 48 kHz opus1 gives
//! SILK the float filter's output instead, so a test at 48 kHz first sets
//! `internals::silk_input::set_integer_hp_at_48k`.
//!
//! The libopus that the tests link is the float build. The reference here
//! is the fixed-point build of the official libopus 1.6.1 release that the
//! dev-dependency `libopus-fixed` compiles (tools/libopus-fixed). Both
//! encoders get the same 16-bit input and the same controls before every
//! packet (`Both`), and their packets are compared one by one.
//!
//! The fixed-point build runs its speech and music analysis at complexity
//! 10 only, so every opus1 encoder here uses that gate
//! (`AnalysisGate::FixedPointBuild`). At complexity 7 to 9 opus1 would
//! otherwise run the analysis, as the float build does.
//!
//! The libopus source that the `opus-head-sys` dependency vendors is
//! libopus's main branch, not 1.6.1. Its SILK encoder adds 1 to the frame
//! energy of the Burg analysis (line 72 of silk/fixed/burg_modified_FIX.c).
//! It writes other packets in every stream of this file; the first
//! difference is at packet 0 to 13. This file first held hashes recorded
//! from that source. Figures in the comments below that describe opus1
//! before a fix were measured against it.

mod k24_common;

use libopus_fixed::consts::*;
use opus1::internals::analysis::{AnalysisGate, set_analysis_gate};
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::dec_api::SilkDecoder;
use opus1::packet::Framing;
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, InbandFec, OpusEncoder, SampleRate};
use std::num::NonZeroU32;

/// opus1's encoder and an encoder of the fixed-point libopus, which get
/// the same controls and the same input.
struct Both {
    rs: OpusEncoder,
    c: libopus_fixed::Encoder,
}

impl Both {
    /// VoIP encoders; opus1's with the analysis gate of the fixed-point
    /// build.
    fn new(rate: SampleRate, channels: usize) -> Both {
        let mut rs = OpusEncoder::new(rate, Channels::try_from(channels).unwrap(), Application::Voip).unwrap();
        set_analysis_gate(&mut rs, AnalysisGate::FixedPointBuild);
        let c = libopus_fixed::Encoder::new(rate as i32, channels, OPUS_APPLICATION_VOIP);
        Both { rs, c }
    }

    fn set_bitrate(&mut self, bitrate: i32) {
        self.rs.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
        self.c.set(OPUS_SET_BITRATE_REQUEST, bitrate);
    }

    fn set_complexity(&mut self, complexity: i32) {
        self.rs.set_complexity(complexity as u8).unwrap();
        self.c.set(OPUS_SET_COMPLEXITY_REQUEST, complexity);
    }

    fn set_force_channels(&mut self, channels: Option<Channels>) {
        self.rs.set_force_channels(channels);
        self.c.set(OPUS_SET_FORCE_CHANNELS_REQUEST, channels.map_or(OPUS_AUTO, |c| c as i32));
    }

    fn set_bandwidth(&mut self, bandwidth: Option<Bandwidth>) {
        self.rs.set_bandwidth(bandwidth);
        self.c.set(OPUS_SET_BANDWIDTH_REQUEST, bandwidth.map_or(OPUS_AUTO, |b| b as i32));
    }

    fn set_force_mode(&mut self, mode: Option<ForcedMode>) {
        self.rs.set_force_mode(mode);
        self.c.set(OPUS_SET_FORCE_MODE_REQUEST, mode.map_or(OPUS_AUTO, |m| m as i32));
    }

    fn set_vbr(&mut self, vbr: bool) {
        self.rs.set_vbr(vbr);
        self.c.set(OPUS_SET_VBR_REQUEST, vbr as i32);
    }

    fn set_inband_fec(&mut self, fec: InbandFec) {
        self.rs.set_inband_fec(fec);
        self.c.set(OPUS_SET_INBAND_FEC_REQUEST, fec as i32);
    }

    fn set_packet_loss_perc(&mut self, percent: i32) {
        self.rs.set_packet_loss_perc(percent as u8).unwrap();
        self.c.set(OPUS_SET_PACKET_LOSS_PERC_REQUEST, percent);
    }

    fn set_prediction_disabled(&mut self, disabled: bool) {
        self.rs.set_prediction_disabled(disabled);
        self.c.set(OPUS_SET_PREDICTION_DISABLED_REQUEST, disabled as i32);
    }

    fn set_dtx(&mut self, dtx: bool) {
        self.rs.set_dtx(dtx);
        self.c.set(OPUS_SET_DTX_REQUEST, dtx as i32);
    }

    /// One packet of `frame` samples per channel from each encoder, with
    /// room for `max_bytes`: (opus1, libopus).
    fn encode(&mut self, pcm: &[i16], frame: usize, max_bytes: usize) -> (Vec<u8>, Vec<u8>) {
        let mut buf = vec![0u8; max_bytes];
        let n = self.rs.encode_i16(pcm, frame, &mut buf).unwrap();
        buf.truncate(n);
        (buf, self.c.encode(pcm, frame, max_bytes).unwrap())
    }

    /// `in_dtx` (`OPUS_GET_IN_DTX`) of each encoder: (opus1, libopus).
    fn in_dtx(&mut self) -> (bool, bool) {
        (self.rs.in_dtx(), self.c.get(OPUS_GET_IN_DTX_REQUEST) != 0)
    }
}

/// The 16 kHz speech of the fixture as 16-bit samples: the samples of the
/// file.
fn speech_16k(seconds: f64) -> Vec<i16> {
    k24_common::speech(16000, seconds).iter().map(|&v| (v * 32768.0).round() as i16).collect()
}

/// What a packet is coded with: bitrate, complexity and the channel count
/// asked for (`None`: the bitrate decides).
#[derive(Clone, Copy)]
struct Packet {
    bitrate: i32,
    complexity: i32,
    channels: Option<Channels>,
}

/// SILK-only wideband VoIP packets of 20 ms at 16 kHz with a variable
/// bitrate, one per entry of `script`. Returns the index of the first
/// packet that is not libopus's, if any.
fn first_difference(channels: usize, pcm: &[i16], script: &[Packet]) -> Option<usize> {
    let frame = 320;
    let mut enc = Both::new(SampleRate::Hz16000, channels);
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_bandwidth(Some(Bandwidth::Wideband));
    for (i, (p, input)) in script.iter().zip(pcm.chunks_exact(frame * channels)).enumerate() {
        enc.set_bitrate(p.bitrate);
        enc.set_complexity(p.complexity);
        enc.set_force_channels(p.channels);
        let (rs, c) = enc.encode(input, frame, 4000);
        if rs != c {
            return Some(i);
        }
    }
    None
}

/// libopus `silk_Encode` calls `silk_control_encoder` for every packet
/// (enc_API.c:253-260), so a new complexity takes effect with the next
/// packet. Here it changes every 20 packets: 10, 2, 0, 6, 1, 9, 4.
/// opus1 called the control function only when SILK's rate or packet
/// duration changed: from packet 20 on its packets differed.
#[test]
fn a_new_complexity_takes_effect_with_the_next_packet_like_libopus() {
    let complexities = [10, 2, 0, 6, 1, 9, 4];
    let script: Vec<Packet> = (0..140)
        .map(|i| Packet { bitrate: 24000, complexity: complexities[i / 20], channels: None })
        .collect();
    let first = first_difference(1, &speech_16k(3.0), &script);
    assert_eq!(first, None, "first packet that is not libopus's");
}

/// Speech with four stretches of digital silence, 150 to 500 ms long, as
/// 16-bit samples.
fn speech_with_silences(channels: usize) -> Vec<i16> {
    let mut mono = k24_common::speech(16000, 3.0);
    for (from, to) in [(0.30, 0.55), (0.90, 1.40), (1.75, 1.95), (2.30, 2.75)] {
        mono[(16000.0 * from) as usize..(16000.0 * to) as usize].fill(0.0);
    }
    let x = if channels == 2 { k24_common::to_stereo(&mono, 16000) } else { mono };
    x.iter().map(|&v| (v * 32768.0).round() as i16).collect()
}

/// In a frame of digital silence libopus tells SILK that there is no voice
/// activity (`is_digital_silence`, opus_encoder.c:1246 and 1909-1914).
/// SILK then lowers its own speech activity below the threshold of its
/// voice activity flag (`silk_encode_do_VAD_FIX`). Right after speech
/// stops SILK's detector still finds activity, so without this the flag
/// and the frame's coding differ: the mono stream of opus1 differed from
/// packet 15 on. Every packet is the fixed-point build's.
#[test]
fn digital_silence_tells_silk_there_is_no_voice_activity_like_libopus() {
    let script: Vec<Packet> = (0..150).map(|_| Packet { bitrate: 24000, complexity: 9, channels: None }).collect();
    let first = first_difference(1, &speech_with_silences(1), &script);
    assert_eq!(first, None, "mono: first packet that is not libopus's");
    let first = first_difference(2, &speech_with_silences(2), &script);
    assert_eq!(first, None, "stereo: first packet that is not libopus's");
}

/// Stereo input, 40 kb/s and 12 kb/s in turns every 20 packets: libopus
/// codes the 12 kb/s stretches as mono (opus_encoder.c:1428-1451). The
/// packet before each mono stretch still codes two channels, with SILK's
/// side faded out (`toMono`, C 1560-1571). The mono packets code the
/// downmix through the mid resampler, the first of them averaged with the
/// side resampler, and the first stereo packet after them starts the side
/// resampler from the mid's (enc_API.c:296-332). Every packet is the
/// fixed-point build's.
#[test]
fn stereo_input_with_mono_stretches_codes_the_packets_of_libopus() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 3.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let script: Vec<Packet> = (0..140)
        .map(|i| Packet { bitrate: if (i / 20) % 2 == 0 { 40000 } else { 12000 }, complexity: 9, channels: None })
        .collect();
    let first = first_difference(2, &pcm, &script);
    assert_eq!(first, None, "first packet that is not libopus's");
}

/// Stereo speech at 32 kb/s with a variable bitrate, 8 s. opus1 computed
/// the weight of the last NLSF coefficient from its distance to 32767
/// (`silk_NLSF_VQ_weights_laroia` uses `1 << 15`). The weight was 1 too
/// large for some vectors, and from packet 274 on the NLSF indices of the
/// mid channel were not libopus's.
#[test]
fn stereo_speech_at_32_kbps_codes_the_packets_of_libopus() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 8.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let script = [Packet { bitrate: 32000, complexity: 9, channels: None }; 400];
    let first = first_difference(2, &pcm, &script);
    assert_eq!(first, None, "first packet that is not libopus's");
}

/// The SILK layer of each packet, decoded at 16 kHz by the SILK decoder of
/// this crate: the decoded samples and the number of bits the layer takes.
/// A CELT layer or a redundant CELT frame is not compared: the fixed-point
/// build's CELT encoder is not the float one of opus1.
///
/// `channels` is the number of channels the packets code. A packet must
/// hold one 20 ms SILK frame at 16 kHz.
fn silk_layers(packets: &[Vec<u8>], channels: usize) -> Vec<(Vec<i16>, u32)> {
    let mut dec = SilkDecoder::new();
    dec.init(16000, channels as i32);
    dec.channel_state[0].fs_api_hz = 16000;
    let mut pcm = [0i16; 2 * 320];
    packets
        .iter()
        .map(|p| {
            let mut rc = RangeCoder::new_decoder(&p[1..]);
            let n = dec.decode(&mut rc, &mut pcm, 0, true, 20, 16000);
            let n = (n.max(0) as usize * channels).min(pcm.len());
            (pcm[..n].to_vec(), rc.tell() as u32)
        })
        .collect()
}

/// The index of the first packet whose SILK layer (`silk_layers`) differs
/// between `rs` and `c`, if any.
fn first_silk_layer_difference(rs: &[Vec<u8>], c: &[Vec<u8>], channels: usize) -> Option<usize> {
    assert_eq!(rs.len(), c.len());
    let (rs, c) = (silk_layers(rs, channels), silk_layers(c, channels));
    (0..rs.len()).find(|&i| rs[i] != c[i])
}

/// Hybrid with a constant bitrate: libopus codes the SILK layer at a
/// variable bitrate with a cap, and clears SILK's `useCBR`
/// (opus_encoder.c:2163-2177). SILK's noise shaping reads that flag: with
/// a variable bitrate it lowers the quality of frames with little speech
/// activity (noise_shape_analysis_FIX.c:171). opus1 passed the cap but
/// left the flag set: from packet 32 on, in 108 of 140 packets, the SILK
/// layer was not libopus's.
///
/// Input: the 16 kHz speech with every sample repeated three times, which
/// is 48 kHz input with images above 8 kHz for the CELT layer, made
/// without arithmetic that could round differently elsewhere. Fullband
/// Hybrid at 32 kb/s, VoIP, complexity 9. SILK reads the integer
/// `hp_cutoff`, as in the fixed-point build; without that control the
/// first SILK layer differs (packet 0).
#[test]
fn hybrid_at_a_constant_bitrate_codes_the_silk_layer_of_libopus() {
    let pcm: Vec<i16> = speech_16k(3.0).iter().flat_map(|&v| [v, v, v]).collect();
    let mut enc = Both::new(SampleRate::Hz48000, 1);
    opus1::internals::silk_input::set_integer_hp_at_48k(&mut enc.rs, true);
    enc.set_bitrate(32000);
    enc.set_complexity(9);
    enc.set_vbr(false);
    enc.set_force_mode(Some(ForcedMode::Hybrid));
    enc.set_bandwidth(Some(Bandwidth::Fullband));
    let (mut rs, mut c) = (Vec::new(), Vec::new());
    for input in pcm.as_chunks::<960>().0.iter().take(140) {
        let (r, l) = enc.encode(input, 960, 4000);
        assert_eq!(r[0] >> 3, 15, "Hybrid fullband 20 ms");
        rs.push(r);
        c.push(l);
    }
    let first = first_silk_layer_difference(&rs, &c, 1);
    assert_eq!(first, None, "first packet whose SILK layer is not libopus's");
}

/// Stereo SILK-only VoIP at 64 kb/s into 100-byte packets, wideband, then
/// narrowband asked for from packet 150. SILK closes its input low-pass
/// filter in the pauses of the speech, and at packet 317 both channels
/// report that they are ready to switch. libopus runs
/// `silk_control_audio_bandwidth` for each channel on the same
/// `maxBits`, so each takes room for the redundant frame of the switch out
/// of it (control_audio_bandwidth.c:88-116). opus1 took that room once:
/// its SILK layer of packet 317 was not libopus's, and the packet had 100
/// bytes where libopus's has 89 (libopus 1.6.1: 91).
///
/// Packets 0 to 316 are byte for byte the fixed-point build's; packet 317
/// ends with a CELT frame, so the SILK layers are compared. The packets
/// after the switch are not: the prefill of SILK at the new rate reads the
/// float delay buffer in opus1 (see `OpusEncoder::encode_impl`).
#[test]
fn both_stereo_channels_make_room_for_the_redundant_frame_of_a_rate_switch() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 7.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let mut enc = Both::new(SampleRate::Hz16000, 2);
    enc.set_bitrate(64000);
    enc.set_complexity(9);
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_force_channels(Some(Channels::Stereo));
    let (mut rs, mut c) = (Vec::new(), Vec::new());
    for (i, input) in pcm.as_chunks::<{ 2 * 320 }>().0.iter().take(318).enumerate() {
        enc.set_bandwidth(Some(if i < 150 { Bandwidth::Wideband } else { Bandwidth::Narrowband }));
        let (r, l) = enc.encode(input, 320, 100);
        rs.push(r);
        c.push(l);
    }
    // Wideband to the end: the packet after 317 is the first at the new
    // rate.
    assert!(rs.iter().chain(&c).all(|p| p[0] == 0x4c), "stereo wideband 20 ms SILK-only");
    let first = first_silk_layer_difference(&rs, &c, 2);
    assert_eq!(first, None, "first packet whose SILK layer is not libopus's");
}

/// Stereo input coded as mono at 7 kb/s, then at 40 kb/s again: the first
/// stereo packet (171) is also the first at SILK's new internal rate. Its
/// redundant CELT frame was announced by packet 170, which coded one
/// channel at 14 kb/s in a pause of the speech. libopus's silk_Encode first
/// starts the side encoder afresh (enc_API.c:194-211), then resets both
/// encoders for the rate switch and gives each the mid's low-pass state
/// (enc_API.c:222-236). Both channels then run the same low-pass
/// transition.
///
/// In opus1 the side encoder is started afresh after that reset, inside
/// `silk_encode_prefill` (`enter_stereo`). Without the copy of the
/// mid's low-pass state after `enter_stereo`, its side signal skips the
/// transition filter, and from packet 171 on 430 of the 600 packets are
/// not libopus's. With the copy only packets 170 to 172 differ:
/// - 170: its redundant CELT frame (the fixed-point build's CELT encoder
///   is not the float one).
/// - 171 and 172: the prefill of 171 reads the float delay buffer (see
///   `OpusEncoder::encode_impl`). With a temporary change that gave the
///   prefill the 16-bit samples and the integer fade of the fixed-point
///   build, packet 172 was libopus's. Packet 171 still was not; that cause
///   was not found.
///
/// The packets after them are compared.
///
/// 16 kHz VoIP, 10 ms, complexity 5, the bitrate deciding the channels and
/// the bandwidth.
#[test]
fn stereo_that_starts_with_a_silk_rate_switch_runs_the_side_through_the_transition_filter() {
    let mut mono = k24_common::speech(16000, 6.0);
    // A pause of 1.2 s, in which SILK may change its rate.
    for v in &mut mono[16000 * 14 / 10..16000 * 26 / 10] {
        *v *= 0.001;
    }
    let pcm: Vec<i16> = k24_common::to_stereo(&mono, 16000)
        .iter()
        .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let frame = 160;
    let mut enc = Both::new(SampleRate::Hz16000, 2);
    enc.set_complexity(5);
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(272) {
        let bitrate = match i {
            0..40 => 40000,
            40..170 => 7000,
            170 => 14000,
            _ => 40000,
        };
        enc.set_bitrate(bitrate);
        let (rs, c) = enc.encode(input, frame, 4000);
        if i == 170 || i == 171 {
            // One channel at 12 kHz, then two at 16 kHz.
            let want = if i == 170 { 0x20 } else { 0x44 };
            assert_eq!(rs[0] & 0xE4, want, "packet {i}: first byte {:02x}", rs[0]);
            assert_eq!(c[0] & 0xE4, want, "libopus, packet {i}: first byte {:02x}", c[0]);
        }
        if i >= 173 && rs != c {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// `OPUS_SET_PREDICTION_DISABLED` in SILK: libopus's silk_Encode marks
/// every channel of the input as just reset before each packet
/// (enc_API.c:172-177). Each packet then codes its first frame without
/// prediction from the packets before, and the LBRR frames that the
/// previous packet left are dropped (enc_API.c:268). Stereo VoIP at
/// 32 kb/s with in-band FEC at 5 % loss, prediction disabled in packets
/// 20-39, 60-79 and 100-119. Every packet is the fixed-point build's.
/// Without the reset, packets 20 to 139 were not.
#[test]
fn prediction_disabled_codes_every_silk_packet_alone_like_libopus() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 3.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let frame = 320;
    let mut enc = Both::new(SampleRate::Hz16000, 2);
    enc.set_bitrate(32000);
    enc.set_complexity(9);
    enc.set_inband_fec(InbandFec::On);
    enc.set_packet_loss_perc(5);
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(140) {
        enc.set_prediction_disabled((i / 20) % 2 == 1);
        let (rs, c) = enc.encode(input, frame, 4000);
        assert_eq!(rs[0] & 0xE4, 0x44, "packet {i}: not stereo SILK-only wideband");
        if rs != c {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// A switch of SILK's internal rate in stereo: libopus resets both
/// encoders and gives each the mid's low-pass state, with the rate it ran
/// at, so that both go on with the same transition (enc_API.c:222-236).
/// Stereo VoIP at 40 kb/s, SILK-only, narrowband for 1.5 s, then wideband
/// asked for. The input is digital silence from 1.4 to 3 s, where SILK
/// switches: there the prefill reads the same zeros in both builds (see
/// `stereo_that_starts_with_a_silk_rate_switch_runs_the_side_through_the_transition_filter`).
/// Every packet is the fixed-point build's. With the side keeping its own
/// low-pass state, packets from 77 on were not.
#[test]
fn a_stereo_silk_rate_switch_gives_the_side_the_mids_low_pass_state() {
    let mut mono = k24_common::speech(16000, 3.2);
    mono[16000 * 14 / 10..16000 * 30 / 10].fill(0.0);
    let pcm: Vec<i16> = k24_common::to_stereo(&mono, 16000)
        .iter()
        .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let frame = 320;
    let mut enc = Both::new(SampleRate::Hz16000, 2);
    enc.set_bitrate(40000);
    enc.set_complexity(5);
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_force_channels(Some(Channels::Stereo));
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(160) {
        enc.set_bandwidth(Some(if i * 20 < 1500 { Bandwidth::Narrowband } else { Bandwidth::Wideband }));
        let (rs, c) = enc.encode(input, frame, 4000);
        if rs != c {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// Speech with three pauses, as 16-bit samples: digital silence (0.7 s),
/// noise at -60 dBFS (0.9 s), and noise below the 16-bit step, which the
/// conversion to 16 bits turns into digital silence (0.7 s). 4.8 s in all.
fn speech_with_pauses(channels: usize) -> Vec<i16> {
    let n = |s: f64| (s * 16000.0) as usize;
    let mut seed = 7u32;
    let mut noise = |len: usize, level: f32| -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                level * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
            })
            .collect()
    };
    let talk = k24_common::speech(16000, 2.5);
    let mut x = talk[..n(1.0)].to_vec();
    x.extend(std::iter::repeat_n(0.0, n(0.7)));
    x.extend_from_slice(&talk[n(1.0)..n(1.6)]);
    x.extend(noise(n(0.9), 1e-3));
    x.extend_from_slice(&talk[n(1.6)..n(2.1)]);
    x.extend(noise(n(0.7), 1e-5));
    x.extend_from_slice(&talk[n(2.1)..n(2.5)]);
    let x = if channels == 2 { k24_common::to_stereo(&x, 16000) } else { x };
    x.iter().map(|&v| (v * 32768.0).round() as i16).collect()
}

/// Discontinuous transmission (DTX) decided by SILK, as below the gate of
/// the analysis: SILK counts the frames without speech, and from the 11th
/// on returns no bytes for up to 20 frames (enc_API.c:273, 550-553;
/// encode_frame_FIX.c:62-79). The packet is then its TOC byte, and the
/// Opus layer skips the rest of the frame (opus_encoder.c:2242-2249).
/// With the `Voip` application libopus codes speech with SILK when SILK's
/// DTX is on (opus_encoder.c:1521).
///
/// Every packet is the fixed-point build's, and so is `in_dtx`
/// (`OPUS_GET_IN_DTX`) after it. The streams cover 20 to 120 ms, mono and
/// stereo, VBR and CBR, and complexity 0 to 9. In a packet of several
/// frames each frame is left out on its own, and with CBR the packet is
/// padded unless every frame was left out (opus_encoder.c:1818-1831).
/// Packets of several frames are compared with CBR only: with VBR opus1
/// leaves out libopus's limit of each frame to its share of the bitrate on
/// purpose.
#[test]
fn silk_dtx_leaves_out_the_frames_of_libopus() {
    /// One stream: channels, frame duration in ms, bitrate, complexity,
    /// VBR.
    struct Stream(usize, usize, i32, i32, bool);
    let streams = [
        ("mono 20 ms VBR c5", Stream(1, 20, 16000, 5, true)),
        ("stereo 20 ms VBR c9", Stream(2, 20, 24000, 9, true)),
        ("mono 60 ms CBR c2", Stream(1, 60, 20000, 2, false)),
        ("mono 120 ms CBR c6", Stream(1, 120, 16000, 6, false)),
        ("stereo 100 ms CBR c8", Stream(2, 100, 20000, 8, false)),
        ("mono 40 ms VBR c0", Stream(1, 40, 12000, 0, true)),
    ];
    for (name, Stream(channels, frame_ms, bitrate, complexity, vbr)) in streams {
        let pcm = speech_with_pauses(channels);
        let frame = 16 * frame_ms;
        let mut enc = Both::new(SampleRate::Hz16000, channels);
        enc.set_bitrate(bitrate);
        enc.set_complexity(complexity);
        enc.set_vbr(vbr);
        enc.set_dtx(true);
        let (mut differ, mut left_out) = (Vec::new(), 0);
        for (i, input) in pcm.chunks_exact(frame * channels).enumerate() {
            let (rs, c) = enc.encode(input, frame, 4000);
            let parsed = opus1::packet::parse(&rs, Framing::Normal).unwrap();
            left_out += parsed.frames(&rs).filter(|f| f.is_empty()).count();
            let (rs_dtx, c_dtx) = enc.in_dtx();
            if rs != c || rs_dtx != c_dtx {
                differ.push(i);
            }
        }
        assert!(differ.is_empty(), "{name}: packets that are not libopus's: {differ:?}");
        assert!(left_out >= 14, "{name}: only {left_out} frames left out");
    }
}
