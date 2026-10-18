//! Shared helpers for the `silkdec_*` tests.
//!
//! These tests check the SILK decoder of opus1 (`SilkDecoder`) against
//! C libopus, without going through `OpusDecoder`. The stream always comes
//! from the C encoder with the coding mode forced to SILK-only, so that
//! `opus_decode` in C is the SILK decoder and nothing else. SILK is
//! fixed-point: the tests require identical 16-bit samples.
//!
//! The helpers do three things:
//!
//! 1. Build test signals from the speech recording in `fixtures/`.
//! 2. Wrap the C encoder and decoder (`audiopus_sys`).
//! 3. Drive `SilkDecoder` the way libopus `opus_decode_native` and
//!    `opus_decode_frame` (src/opus_decoder.c) drive `silk_Decode`: for a
//!    received packet, for a lost packet, and for a lost packet that is
//!    recovered from the forward error correction data of the next one.

#![allow(dead_code)]

use audiopus_sys as sys;
use opus1::internals::silk::SilkResampler;
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::dec_api::SilkDecoder;
use opus1::internals::silk::decode_frame::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST};
use opus1::internals::silk::decoder_structs::SilkDecControl;

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// `fixtures/answer_16k.wav`: 18 s of 16 kHz mono 16-bit speech.
pub fn load_speech_16k() -> Vec<i16> {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/answer_16k.wav"
    ))
    .expect("fixtures/answer_16k.wav");
    // Find the "data" chunk instead of assuming a 44-byte header.
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            return data
                .as_chunks::<2>().0.iter()
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk in answer_16k.wav");
}

/// `seconds` of mono speech at 16 kHz, starting `start` seconds into the
/// recording.
pub fn speech(start: f64, seconds: f64) -> Vec<i16> {
    static ALL: std::sync::OnceLock<Vec<i16>> = std::sync::OnceLock::new();
    let all = ALL.get_or_init(load_speech_16k);
    let first = (start * 16000.0) as usize;
    all[first..first + (seconds * 16000.0) as usize].to_vec()
}

/// Interleaved stereo with a different talker position per channel: the
/// right channel is the left one 3 ms later and 4.4 dB lower. The side
/// channel (left minus right) carries real signal.
pub fn stereo_delayed(mono: &[i16]) -> Vec<i16> {
    let delay = 48;
    let mut out = Vec::with_capacity(mono.len() * 2);
    for i in 0..mono.len() {
        out.push(mono[i]);
        out.push(if i >= delay { (mono[i - delay] as i32 * 6 / 10) as i16 } else { 0 });
    }
    out
}

/// Interleaved stereo from two unrelated mono signals.
pub fn stereo_from(left: &[i16], right: &[i16]) -> Vec<i16> {
    left.iter().zip(right).flat_map(|(&l, &r)| [l, r]).collect()
}

// ---------------------------------------------------------------------------
// Synthetic signals
// ---------------------------------------------------------------------------

/// A small generator of repeatable pseudo-random numbers.
pub struct Lcg(pub u32);

impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    pub fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

pub fn to_i16(x: f64) -> i16 {
    x.round().clamp(-32768.0, 32767.0) as i16
}

/// A low voice: 12 harmonics of a fundamental that moves between 58 and
/// 82 Hz, loud. The pitch period is close to the longest SILK can code.
pub fn low_voice(rate: usize, n: usize) -> Vec<i16> {
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let f0 = 70.0 + 12.0 * (2.0 * std::f64::consts::PI * 1.3 * t).sin();
            phase += 2.0 * std::f64::consts::PI * f0 / rate as f64;
            let sum: f64 = (1..=12).map(|h| (phase * h as f64).sin() / h as f64).sum();
            to_i16(14000.0 * sum)
        })
        .collect()
}

/// A tone that rises from 100 Hz to 7 kHz, twice.
pub fn sweep(rate: usize, n: usize) -> Vec<i16> {
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let pos = (2 * i % n) as f64 / n as f64;
            let f = 100.0 * (70.0f64).powf(pos);
            phase += 2.0 * std::f64::consts::PI * f / rate as f64;
            to_i16(20000.0 * phase.sin())
        })
        .collect()
}

/// 80 ms of full-scale white noise, then 120 ms of silence, repeated.
pub fn noise_bursts(rate: usize, n: usize) -> Vec<i16> {
    let mut rng = Lcg(7);
    (0..n)
        .map(|i| {
            let v = (rng.next() & 0xffff) as i32 - 32768;
            if (i * 1000 / rate) % 200 < 80 { v as i16 } else { 0 }
        })
        .collect()
}

/// A 440 Hz tone at four times full scale, clipped: nearly a square wave.
/// Its level swells and fades once per second.
pub fn clipped(rate: usize, n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let level = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos();
            to_i16(131072.0 * level * (2.0 * std::f64::consts::PI * 440.0 * t).sin())
        })
        .collect()
}

/// One full-scale sample 55 times per second, alternating in sign.
pub fn clicks(rate: usize, n: usize) -> Vec<i16> {
    let period = rate / 55;
    (0..n)
        .map(|i| match (i % period, (i / period) % 2) {
            (0, 0) => 32767,
            (0, _) => -32768,
            _ => 0,
        })
        .collect()
}

/// Digital silence with a 100 ms tone in the middle.
pub fn mostly_silence(rate: usize, n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| {
            let ms = i * 1000 / rate;
            let mid = n * 500 / rate;
            if ms >= mid && ms < mid + 100 {
                to_i16(8000.0 * (2.0 * std::f64::consts::PI * 300.0 * i as f64 / rate as f64).sin())
            } else {
                0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// C libopus
// ---------------------------------------------------------------------------

pub const MODE_SILK: i32 = 1000;
pub const MODE_CELT: i32 = 1002;
/// `OPUS_SET_FORCE_MODE` (opus_private.h): not part of the public header.
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;

pub const BW_NB: i32 = sys::OPUS_BANDWIDTH_NARROWBAND;
pub const BW_MB: i32 = sys::OPUS_BANDWIDTH_MEDIUMBAND;
pub const BW_WB: i32 = sys::OPUS_BANDWIDTH_WIDEBAND;

/// SILK sampling rate of a bandwidth.
pub fn silk_rate(bandwidth: i32) -> usize {
    match bandwidth {
        BW_NB => 8000,
        BW_MB => 12000,
        _ => 16000,
    }
}

pub fn bandwidth_name(bandwidth: i32) -> &'static str {
    match bandwidth {
        BW_NB => "NB",
        BW_MB => "MB",
        _ => "WB",
    }
}

/// C libopus encoder with the coding mode forced. The input rate is 16 kHz
/// unless the encoder was made with `CEnc::with`.
pub struct CEnc {
    st: *mut sys::OpusEncoder,
    pub channels: usize,
    /// Input sampling rate in Hz.
    pub rate: usize,
}

impl CEnc {
    /// A SILK-only encoder at `bandwidth` and `bitrate`.
    pub fn silk(channels: usize, bandwidth: i32, bitrate: i32) -> Self {
        Self::forced(MODE_SILK, channels, bandwidth, bitrate)
    }

    /// A CELT-only encoder at `bandwidth` and `bitrate`.
    pub fn celt(channels: usize, bandwidth: i32, bitrate: i32) -> Self {
        Self::forced(MODE_CELT, channels, bandwidth, bitrate)
    }

    fn forced(mode: i32, channels: usize, bandwidth: i32, bitrate: i32) -> Self {
        let mut enc = Self::with(16000, channels, sys::OPUS_APPLICATION_VOIP);
        enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, mode);
        enc.ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth);
        enc.ctl(sys::OPUS_SET_BITRATE_REQUEST, bitrate);
        enc
    }

    /// An encoder for input at `rate` Hz and the given application
    /// (`OPUS_APPLICATION_VOIP` or `OPUS_APPLICATION_AUDIO`). Nothing is
    /// forced yet: the caller sets the mode, bandwidth and bitrate with
    /// `ctl`.
    pub fn with(rate: usize, channels: usize, application: i32) -> Self {
        let mut err = 0;
        let st = unsafe {
            sys::opus_encoder_create(rate as i32, channels as i32, application, &mut err)
        };
        assert_eq!(err, 0, "opus_encoder_create");
        Self { st, channels, rate }
    }

    /// Any "set" control that takes one `opus_int32`.
    pub fn ctl(&mut self, request: i32, value: i32) -> &mut Self {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value}) = {ret}");
        self
    }

    /// In-band forward error correction for an expected loss of
    /// `loss_perc` percent.
    pub fn fec(&mut self, loss_perc: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_INBAND_FEC_REQUEST, 1);
        self.ctl(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, loss_perc)
    }

    /// Code the input as `channels` channels from the next frame on.
    pub fn force_channels(&mut self, channels: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, channels)
    }

    /// Encode one frame of interleaved 16-bit PCM.
    pub fn encode(&mut self, pcm: &[i16]) -> Vec<u8> {
        let frame = pcm.len() / self.channels;
        let mut out = vec![0u8; 4000];
        let n = unsafe {
            sys::opus_encode(self.st, pcm.as_ptr(), frame as i32, out.as_mut_ptr(), out.len() as i32)
        };
        assert!(n > 0, "opus_encode failed: {n}");
        out.truncate(n as usize);
        out
    }

    /// Encode a whole signal in frames of `frame_ms`. A trailing partial
    /// frame is dropped.
    pub fn encode_all(&mut self, pcm: &[i16], frame_ms: usize) -> Vec<Vec<u8>> {
        pcm.chunks_exact(self.rate / 1000 * frame_ms * self.channels)
            .map(|frame| self.encode(frame))
            .collect()
    }
}

impl Drop for CEnc {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// C libopus decoder with 16-bit output.
pub struct CDec {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
    pub rate: usize,
}

impl CDec {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        Self { st, channels, rate }
    }

    /// `opus_decode`: `packet` is `None` for a lost packet. With `fec`,
    /// `packet` is the packet after the lost one. `frame_size` is in
    /// samples per channel.
    pub fn decode(&mut self, packet: Option<&[u8]>, frame_size: usize, fec: bool) -> Vec<i16> {
        let mut pcm = vec![0i16; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_decode(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as i32, fec as i32)
        };
        assert!(n >= 0, "opus_decode failed: {n}");
        pcm.truncate(n as usize * self.channels);
        pcm
    }
}

impl CDec {
    /// `OPUS_GET_PITCH`: after a SILK frame, the pitch lag that
    /// `silk_Decode` reported for it, in samples at 48 kHz (0 if the frame
    /// was not voiced).
    pub fn pitch(&mut self) -> i32 {
        let mut value = 0i32;
        let ret = unsafe {
            sys::opus_decoder_ctl(self.st, sys::OPUS_GET_PITCH_REQUEST, &mut value as *mut i32)
        };
        assert_eq!(ret, 0, "OPUS_GET_PITCH");
        value
    }
}

impl Drop for CDec {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

// ---------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------

/// What the first byte of a SILK-only packet says (RFC 6716 section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Toc {
    /// SILK sampling rate in Hz.
    pub silk_rate: usize,
    /// Duration of one Opus frame in ms: 10, 20, 40 or 60.
    pub frame_ms: usize,
    pub channels: usize,
}

/// Whether a packet is SILK-only.
pub fn is_silk(packet: &[u8]) -> bool {
    packet[0] >> 3 < 12
}

/// Parse the first byte of a SILK-only packet. Panics for any other packet.
pub fn toc(packet: &[u8]) -> Toc {
    let config = packet[0] >> 3;
    assert!(config < 12, "not a SILK-only packet: configuration {config}");
    Toc {
        silk_rate: [8000, 12000, 16000][(config / 4) as usize],
        frame_ms: [10, 20, 40, 60][(config % 4) as usize],
        channels: if packet[0] & 4 != 0 { 2 } else { 1 },
    }
}

/// The frames inside a packet, found by libopus's packet parser.
pub fn frames(packet: &[u8]) -> Vec<&[u8]> {
    let mut ptrs = [std::ptr::null::<u8>(); 48];
    let mut sizes = [0i16; 48];
    let count = unsafe {
        sys::opus_packet_parse(
            packet.as_ptr(),
            packet.len() as i32,
            std::ptr::null_mut(),
            ptrs.as_mut_ptr(),
            sizes.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    assert!(count > 0, "opus_packet_parse = {count}");
    (0..count as usize)
        .map(|i| {
            let offset = unsafe { ptrs[i].offset_from(packet.as_ptr()) } as usize;
            &packet[offset..offset + sizes[i] as usize]
        })
        .collect()
}

/// Join packets of the same configuration into one packet with several
/// frames (libopus `opus_repacketizer_cat` and `opus_repacketizer_out`).
pub fn repacketize(packets: &[Vec<u8>]) -> Vec<u8> {
    let mut out = vec![0u8; packets.iter().map(|p| p.len() + 4).sum::<usize>() + 8];
    unsafe {
        let rp = sys::opus_repacketizer_create();
        assert!(!rp.is_null());
        for p in packets {
            let ret = sys::opus_repacketizer_cat(rp, p.as_ptr(), p.len() as i32);
            assert_eq!(ret, 0, "opus_repacketizer_cat");
        }
        let n = sys::opus_repacketizer_out(rp, out.as_mut_ptr(), out.len() as i32);
        sys::opus_repacketizer_destroy(rp);
        assert!(n > 0, "opus_repacketizer_out = {n}");
        out.truncate(n as usize);
    }
    out
}

/// Add `extra` bytes of padding to a packet (libopus `opus_packet_pad`).
pub fn pad(packet: &[u8], extra: usize) -> Vec<u8> {
    let mut out = packet.to_vec();
    out.resize(packet.len() + extra, 0);
    let ret = unsafe {
        sys::opus_packet_pad(out.as_mut_ptr(), packet.len() as i32, out.len() as i32)
    };
    assert_eq!(ret, 0, "opus_packet_pad");
    out
}

/// How the packets of a stream are configured, for the tests to check that
/// the C encoder produced what the test is about.
pub fn stream_summary(packets: &[Vec<u8>]) -> String {
    let mut seen: Vec<(Toc, usize)> = Vec::new();
    for p in packets {
        let t = toc(p);
        match seen.iter_mut().find(|(s, _)| *s == t) {
            Some((_, n)) => *n += 1,
            None => seen.push((t, 1)),
        }
    }
    seen.iter()
        .map(|(t, n)| format!("{} Hz {} ms {} ch x{n}", t.silk_rate, t.frame_ms, t.channels))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// opus1 SILK decoder, driven like libopus drives silk_Decode
// ---------------------------------------------------------------------------

/// Counts of what the decoder did, so that a test can check that the case
/// it is about really occurred in its stream.
#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    /// SILK frames decoded from an LBRR copy (mid channel).
    pub lbrr_mid: usize,
    /// SILK frames whose side channel was decoded from an LBRR copy.
    pub lbrr_side: usize,
    /// LBRR copies whose own mid-only flag was set.
    pub lbrr_mid_only: usize,
    /// SILK frames asked for with the FEC flag that had no LBRR copy and
    /// were concealed instead.
    pub lbrr_missing: usize,
    /// Normally decoded stereo SILK frames without a coded side channel.
    pub mid_only: usize,
    /// Normally decoded stereo SILK frames with a coded side channel.
    pub with_side: usize,
    /// SILK frames concealed as lost.
    pub concealed: usize,
}

impl Stats {
    /// Add the counts of another run.
    pub fn add(&mut self, other: &Stats) {
        self.lbrr_mid += other.lbrr_mid;
        self.lbrr_side += other.lbrr_side;
        self.lbrr_mid_only += other.lbrr_mid_only;
        self.lbrr_missing += other.lbrr_missing;
        self.mid_only += other.mid_only;
        self.with_side += other.with_side;
        self.concealed += other.concealed;
    }
}

/// `SilkDecoder` behind the interface of `opus_decode`, for SILK-only
/// streams. It follows libopus `opus_decode_native` and
/// `opus_decode_frame` step by step.
pub struct RsDec {
    pub dec: SilkDecoder,
    rate: usize,
    channels: usize,
    /// Control values of the last packet. libopus keeps them in
    /// `st->DecControl` and uses them unchanged for a lost packet.
    control: Option<SilkDecControl>,
    /// Samples per channel of one frame of the last packet (libopus
    /// `st->frame_size`).
    last_frame_size: usize,
    /// The packet before was CELT-only (libopus `st->prev_mode`).
    after_celt: bool,
    /// For the last frame that carried data: the number of bits the SILK
    /// decoder read, and the size of the frame in bytes.
    pub last_bits: (i32, usize),
    pub stats: Stats,
}

impl RsDec {
    pub fn new(rate: usize, channels: usize) -> Self {
        Self {
            dec: SilkDecoder::new(),
            rate,
            channels,
            control: None,
            last_frame_size: 0,
            after_celt: false,
            last_bits: (0, 0),
            stats: Stats::default(),
        }
    }

    /// A decoder set up the way `OpusDecoder::new` in src/lib.rs sets up
    /// its SILK decoder today, with a call to `SilkDecoder::init`.
    /// `decode_with_control` does not need that call, but must work after
    /// it.
    pub fn with_init(rate: usize, channels: usize) -> Self {
        let mut this = Self::new(rate, channels);
        assert_eq!(this.dec.init(rate.min(16000) as i32, channels as i32), 0);
        this.dec.channel_state[0].fs_api_hz = rate as i32;
        this
    }

    /// Tell the decoder that the stream had a CELT-only packet here. Before
    /// the next SILK frame libopus resets the SILK decoder
    /// (`opus_decode_frame`: `if (st->prev_mode==MODE_CELT_ONLY)
    /// silk_ResetDecoder(silk_dec)`), and so does this helper.
    pub fn celt_packet(&mut self) {
        self.after_celt = true;
    }

    fn samples(&self, ms: usize) -> usize {
        self.rate * ms / 1000
    }

    /// `opus_decode(packet, decode_fec = 0)`.
    pub fn decode(&mut self, packet: &[u8]) -> Vec<i16> {
        let t = toc(packet);
        let frame_size = self.samples(t.frame_ms);
        self.last_frame_size = frame_size;
        let mut out = Vec::new();
        for payload in frames(packet) {
            if payload.len() <= 1 {
                // opus_decode_frame: `if (len<=1) data = NULL`.
                out.extend(self.conceal(frame_size));
            } else {
                out.extend(self.frame(t, payload, FLAG_DECODE_NORMAL));
            }
        }
        out
    }

    /// `opus_decode(NULL)`: conceal `frame_size` samples per channel.
    pub fn lost(&mut self, frame_size: usize) -> Vec<i16> {
        self.conceal(frame_size)
    }

    /// `opus_decode(next, decode_fec = 1)`: `next` is the packet that
    /// followed the lost one, `frame_size` the duration to produce.
    pub fn fec(&mut self, next: &[u8], frame_size: usize) -> Vec<i16> {
        let t = toc(next);
        let packet_frame_size = self.samples(t.frame_ms);
        // opus_decode_native: "If no FEC can be present, run the PLC".
        if frame_size < packet_frame_size {
            return self.conceal(frame_size);
        }
        // Conceal everything except the part the FEC data can cover.
        let mut out = self.conceal(frame_size - packet_frame_size);
        self.last_frame_size = packet_frame_size;
        let first = frames(next)[0];
        if first.len() <= 1 {
            out.extend(self.conceal(packet_frame_size));
        } else {
            out.extend(self.frame(t, first, FLAG_DECODE_LBRR));
        }
        out
    }

    /// One Opus frame that carries data (`opus_decode_frame`, the SILK
    /// part): as many `silk_Decode` calls as the frame has SILK frames.
    fn frame(&mut self, t: Toc, payload: &[u8], lost_flag: i32) -> Vec<i16> {
        let control = SilkDecControl {
            n_channels_api: self.channels as i32,
            n_channels_internal: t.channels as i32,
            api_sample_rate: self.rate as i32,
            internal_sample_rate: t.silk_rate as i32,
            payload_size_ms: t.frame_ms as i32,
        };
        self.control = Some(control);
        if self.after_celt {
            self.dec.reset();
            self.after_celt = false;
        }
        let frame_size = self.samples(t.frame_ms);
        let mut out = vec![0i16; frame_size * self.channels];
        let mut rc = RangeCoder::new_decoder(payload);
        let mut decoded = 0;
        while decoded < frame_size {
            let n = self
                .dec
                .decode_with_control(
                    &control,
                    &mut rc,
                    &mut out[decoded * self.channels..],
                    lost_flag,
                    decoded == 0,
                )
                .expect("decode_with_control");
            // The frame counter was already advanced; look at the frame
            // just decoded.
            let index = self.dec.channel_state[0].n_frames_decoded as usize - 1;
            let mid_lbrr = self.dec.channel_state[0].lbrr_flags[index] == 1;
            let side_lbrr = t.channels == 2 && self.dec.channel_state[1].lbrr_flags[index] == 1;
            if lost_flag == FLAG_DECODE_LBRR {
                self.stats.lbrr_mid += mid_lbrr as usize;
                self.stats.lbrr_side += side_lbrr as usize;
                self.stats.lbrr_mid_only +=
                    (mid_lbrr && self.dec.prev_decode_only_middle == 1) as usize;
                self.stats.lbrr_missing += (!mid_lbrr) as usize;
            } else if t.channels == 2 {
                let mid_only = self.dec.prev_decode_only_middle == 1;
                self.stats.mid_only += mid_only as usize;
                self.stats.with_side += (!mid_only) as usize;
            }
            decoded += n;
        }
        assert_eq!(decoded, frame_size);
        self.last_bits = (rc.tell(), payload.len());
        out
    }

    /// `opus_decode_native` and `opus_decode_frame` with `data == NULL`, for
    /// any multiple of 2.5 ms.
    ///
    /// One call to SILK conceals at most one frame of the last packet
    /// (`frame_size = IMIN(frame_size, st->frame_size)`), and at most
    /// 20 ms. A piece between 10 and 20 ms is cut down to 10 ms. For a piece
    /// shorter than 10 ms SILK still conceals 10 ms, and libopus returns the
    /// start of it (`pcm_too_small`): the SILK decoder is then 10 ms
    /// further, although the caller got less.
    fn conceal(&mut self, frame_size: usize) -> Vec<i16> {
        let mut out = vec![0i16; frame_size * self.channels];
        // "If we haven't got any packet yet, all we can do is return zeros".
        let Some(mut control) = self.control else {
            return out;
        };
        // After a CELT-only packet libopus conceals with CELT.
        assert!(!self.after_celt, "this helper cannot conceal after a CELT-only packet");
        let (f10, f20) = (self.samples(10), self.samples(20));
        assert_eq!(frame_size % (f10 / 4), 0, "libopus conceals multiples of 2.5 ms");
        let mut scratch = vec![0i16; f10 * self.channels];
        let mut done = 0;
        while done < frame_size {
            let mut piece = (frame_size - done).min(self.last_frame_size).min(f20);
            if piece > f10 && piece < f20 {
                piece = f10;
            }
            control.payload_size_ms = if piece == f20 { 20 } else { 10 };
            let mut rc = RangeCoder::new_decoder(&[]);
            let target = &mut out[done * self.channels..(done + piece) * self.channels];
            let buffer = if piece < f10 { &mut scratch[..] } else { &mut target[..] };
            let n = self
                .dec
                .decode_with_control(&control, &mut rc, buffer, FLAG_PACKET_LOST, true)
                .expect("decode_with_control (lost)");
            assert_eq!(n, piece.max(f10));
            if piece < f10 {
                target.copy_from_slice(&scratch[..piece * self.channels]);
            }
            self.stats.concealed += 1;
            done += piece;
        }
        out
    }
}

/// The older entry point `SilkDecoder::decode`, driven the way
/// `OpusDecoder::decode_silk_frames` in src/lib.rs drives it: the decoder
/// returns one channel after the other at the SILK rate, and the caller
/// owns one resampler per channel. Stream and decoder have the same
/// channel count.
pub struct RsLegacyDec {
    pub dec: SilkDecoder,
    rate: usize,
    channels: usize,
    resamplers: [SilkResampler; 2],
    /// SILK rate of the last packet, 0 before the first.
    silk_rate: usize,
    pub lbrr_frames: usize,
}

impl RsLegacyDec {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut dec = SilkDecoder::new();
        assert_eq!(dec.init(rate.min(16000) as i32, channels as i32), 0);
        dec.channel_state[0].fs_api_hz = rate as i32;
        Self {
            dec,
            rate,
            channels,
            resamplers: [SilkResampler::default(), SilkResampler::default()],
            silk_rate: 0,
            lbrr_frames: 0,
        }
    }

    pub fn decode(&mut self, packet: &[u8]) -> Vec<i16> {
        let t = toc(packet);
        let mut out = Vec::new();
        for payload in frames(packet) {
            out.extend(self.frame(Some(t), payload, t.frame_ms, FLAG_DECODE_NORMAL));
        }
        out
    }

    /// Conceal `ms` milliseconds, 20 or 10 ms at a time.
    pub fn lost(&mut self, ms: usize) -> Vec<i16> {
        if self.silk_rate == 0 {
            return vec![0i16; self.rate * ms / 1000 * self.channels];
        }
        let mut out = Vec::new();
        let mut left = ms;
        while left > 0 {
            let step = if left >= 20 { 20 } else { 10 };
            out.extend(self.frame(None, &[], step, FLAG_PACKET_LOST));
            left -= step;
        }
        out
    }

    /// Recover a lost packet of the same duration as `next` from the LBRR
    /// data of `next`.
    pub fn fec(&mut self, next: &[u8]) -> Vec<i16> {
        let t = toc(next);
        self.frame(Some(t), frames(next)[0], t.frame_ms, FLAG_DECODE_LBRR)
    }

    fn frame(&mut self, t: Option<Toc>, payload: &[u8], ms: usize, lost_flag: i32) -> Vec<i16> {
        if let Some(t) = t {
            assert_eq!(t.channels, self.channels, "SilkDecoder::decode cannot convert channels");
            if t.silk_rate != self.silk_rate {
                for r in &mut self.resamplers {
                    assert_eq!(r.init(t.silk_rate as i32, self.rate as i32), 0);
                }
            }
            self.silk_rate = t.silk_rate;
        }
        let silk_rate = self.silk_rate;
        let internal = silk_rate * ms / 1000;
        let mut planar = vec![0i16; internal * self.channels];
        let mut out = vec![0i16; self.rate * ms / 1000 * self.channels];
        let mut rc = RangeCoder::new_decoder(payload);
        let (mut pos, mut new_packet) = (0, true);
        while pos < internal {
            let n = self.dec.decode(
                &mut rc,
                &mut planar,
                lost_flag,
                new_packet,
                ms as i32,
                silk_rate as i32,
            );
            assert!(n > 0, "SilkDecoder::decode = {n}");
            let n = n as usize;
            if lost_flag == FLAG_DECODE_LBRR {
                let index = self.dec.channel_state[0].n_frames_decoded as usize - 1;
                self.lbrr_frames += (self.dec.channel_state[0].lbrr_flags[index] == 1) as usize;
            }
            new_packet = false;
            let api_n = n * self.rate / silk_rate;
            let api_pos = pos * self.rate / silk_rate;
            let mut resampled = vec![0i16; api_n];
            for ch in 0..self.channels {
                self.resamplers[ch].process(&mut resampled, &planar[ch * n..(ch + 1) * n], n as i32);
                for (i, &v) in resampled.iter().enumerate() {
                    out[(api_pos + i) * self.channels + ch] = v;
                }
            }
            pos += n;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Where two decoded signals differ.
#[derive(Default, Debug)]
pub struct Diff {
    /// Number of samples that differ.
    pub samples: usize,
    /// Largest absolute difference.
    pub max: i32,
    /// Label of the first packet with a difference.
    pub first: Option<String>,
    /// Packets compared.
    pub packets: usize,
    /// Packets after which the two decoders reported different pitch lags.
    pub pitch: usize,
    /// Packets after which both reported a voiced frame (a pitch lag that
    /// is not 0).
    pub voiced: usize,
}

impl Diff {
    /// Compare the output of one packet. `label` describes the packet.
    pub fn add(&mut self, reference: &[i16], ours: &[i16], label: impl FnOnce() -> String) {
        assert_eq!(reference.len(), ours.len(), "output length of {}", label());
        self.packets += 1;
        let mut differing = 0;
        for (a, b) in reference.iter().zip(ours) {
            if a != b {
                differing += 1;
                self.max = self.max.max((*a as i32 - *b as i32).abs());
            }
        }
        if differing > 0 && self.first.is_none() {
            self.first = Some(label());
        }
        self.samples += differing;
    }

    /// Compare the pitch lag the two decoders report after a packet.
    pub fn add_pitch(&mut self, reference: i32, ours: i32, label: impl FnOnce() -> String) {
        if reference != ours {
            self.pitch += 1;
            if self.first.is_none() {
                self.first = Some(format!("{} (pitch lag {ours} against {reference})", label()));
            }
        } else if reference != 0 {
            self.voiced += 1;
        }
    }

    /// A line for a list of failures, or `None` if every sample and every
    /// pitch lag was identical.
    pub fn failure(&self, what: &str) -> Option<String> {
        (self.samples > 0 || self.pitch > 0).then(|| {
            format!(
                "{what}: {} samples differ (largest difference {}), {} pitch lags differ, first in {}",
                self.samples,
                self.max,
                self.pitch,
                self.first.as_deref().unwrap_or("?")
            )
        })
    }

    /// Panic unless every sample and every pitch lag was identical.
    pub fn assert_identical(&self, what: &str) {
        if let Some(failure) = self.failure(what) {
            panic!("{failure}");
        }
    }
}

/// Panic with the list of failed cases, if there are any.
pub fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} cases differ from libopus:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// How a packet of a stream reaches the decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    /// Received and decoded.
    Received,
    /// Lost and concealed.
    Lost,
    /// Lost, then recovered from the FEC data of the next packet. The last
    /// packet of a stream has no next packet and is concealed.
    Recovered,
}

/// The fate of packet `i` when the packets for which `lost` is true do not
/// arrive and the receiver uses FEC: a lost packet is recovered from the
/// next packet if that one arrived. Otherwise it is concealed, as the
/// earlier packets of a burst are.
pub fn fate_with_fec(lost: impl Fn(usize) -> bool) -> impl Fn(usize) -> Fate {
    move |i| {
        if !lost(i) {
            Fate::Received
        } else if lost(i + 1) {
            Fate::Lost
        } else {
            Fate::Recovered
        }
    }
}

/// The fate of packet `i` when lost packets are concealed.
pub fn fate_without_fec(lost: impl Fn(usize) -> bool) -> impl Fn(usize) -> Fate {
    move |i| if lost(i) { Fate::Lost } else { Fate::Received }
}

/// A loss pattern drawn from `seed`: about one packet in four is lost, in
/// bursts of up to four. With `fec`, two out of three lost packets that
/// are followed by a received one are recovered from it.
pub fn random_fates(seed: u32, packets: usize, fec: bool) -> Vec<Fate> {
    let mut rng = Lcg(seed);
    let mut lost = vec![false; packets + 1];
    let mut i = 0;
    while i < packets {
        if rng.below(8) == 0 {
            for _ in 0..1 + rng.below(4) {
                if i < packets {
                    lost[i] = true;
                    i += 1;
                }
            }
        }
        i += 1;
    }
    // Every stream also starts with a loss half of the time.
    lost[0] = seed.is_multiple_of(2);
    (0..packets)
        .map(|i| {
            if !lost[i] {
                Fate::Received
            } else if fec && !lost[i + 1] && rng.below(3) != 0 {
                Fate::Recovered
            } else {
                Fate::Lost
            }
        })
        .collect()
}

/// Decode `packets` with C libopus and with opus1 through
/// `decode_with_control`, treating packet `i` as `fate(i)` says, and compare
/// every packet's output. `rate` and `channels` are the decoder's.
pub fn run(
    packets: &[Vec<u8>],
    rate: usize,
    channels: usize,
    fate: impl Fn(usize) -> Fate,
) -> (Diff, Stats) {
    run_with(RsDec::new(rate, channels), packets, rate, channels, fate)
}

/// `run` with a given opus1 decoder, which must be new.
///
/// The stream may contain CELT-only packets. The C decoder decodes them and
/// the SILK decoder of opus1 is reset after them, as libopus resets its
/// own. In the first SILK frame after a CELT-only packet libopus cross-fades
/// the first 5 ms with CELT concealment, so those 5 ms are not compared.
/// A CELT-only packet and the packet after it are always received.
pub fn run_with(
    mut rs: RsDec,
    packets: &[Vec<u8>],
    rate: usize,
    channels: usize,
    fate: impl Fn(usize) -> Fate,
) -> (Diff, Stats) {
    let mut c = CDec::new(rate, channels);
    let mut diff = Diff::default();
    let mut after_celt = false;
    for (i, p) in packets.iter().enumerate() {
        if !is_silk(p) {
            c.decode(Some(p), rate / 50, false);
            rs.celt_packet();
            after_celt = true;
            continue;
        }
        let frame_size = rate * toc(p).frame_ms / 1000 * frames(p).len();
        let last = i + 1 == packets.len();
        let before_celt = !last && !is_silk(&packets[i + 1]);
        let f = match fate(i) {
            _ if after_celt || before_celt => Fate::Received,
            Fate::Recovered if last => Fate::Lost,
            f => f,
        };
        let (reference, ours) = match f {
            Fate::Received => (c.decode(Some(p), frame_size, false), rs.decode(p)),
            Fate::Lost => (c.decode(None, frame_size, false), rs.lost(frame_size)),
            Fate::Recovered => {
                let next = &packets[i + 1];
                (c.decode(Some(next), frame_size, true), rs.fec(next, frame_size))
            }
        };
        let skip = if after_celt { rate / 200 * channels } else { 0 };
        after_celt = false;
        diff.add(&reference[skip..], &ours[skip..], || format!("packet {i} ({f:?})"));
        diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || format!("packet {i} ({f:?})"));
    }
    (diff, rs.stats)
}
