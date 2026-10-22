//! The multistream decoder against libopus's `opus_multistream_decoder`.
//!
//! The input chooses a sampling rate and a channel layout: the number of
//! output channels, streams and coupled (stereo) streams, and the mapping.
//! Both libopus and opus1 must accept or refuse the layout alike. For an
//! accepted layout the input then gives a sequence of decoder calls:
//!
//! - packets made by libopus encoders, one per stream, joined into a
//!   multistream packet with self-delimited framing, sometimes damaged;
//! - packets taken from the input as they are;
//! - lost packets, forward error correction (FEC) calls, 16-bit output,
//!   frame sizes that do not fit, and resets;
//! - output buffers of opus1 with more or less room than the frame size,
//!   where with too little room opus1 may refuse and must keep its state.
//!
//! A decoder built from the layout as the channel mapping table of an Ogg
//! Opus header must be accepted too.
//!
//! After every call both decoders must accept or refuse alike. When they
//! accept, they must return the same number of samples and end in the same
//! final range (the state of the range decoders, combined over the
//! streams). opus1 must write only the samples it returns, with `f32` and
//! with 16-bit output, and they must fit its buffer. Decoded `f32` samples
//! must be finite. While the decoders have decoded only whole packets as
//! the encoders wrote them, with no frame of 0 or 1 bytes and no stream
//! that switched from CELT-only to SILK or Hybrid, the output of every
//! channel must be within 50 dB of libopus's.

#![no_main]

mod common;

use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::SampleRate;
use opus1::multistream::{ChannelMappingTable, MultistreamDecoder};

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});

/// What opus1's output buffer holds before a call.
const MARK_I16: i16 = 0x5a5a;
const MARK_F32: f32 = 1234.5;

/// The packets of one frame duration from one libopus encoder per stream,
/// joined as libopus's multistream encoder joins them.
struct Source {
    encoders: Vec<CEncoder>,
    signal: Signal,
    frame_size: usize,
    fs: i32,
}

impl Source {
    fn new(u: &mut Unstructured, streams: usize, coupled: usize) -> Result<Self> {
        let fs = *u.choose(&RATES)?;
        let app = *u.choose(&[2048, 2049, 2051])?;
        let mut encoders = Vec::new();
        for s in 0..streams {
            let channels = if s < coupled { 2 } else { 1 };
            let mut enc = CEncoder::new(fs, channels, app);
            setup(u, &mut enc)?;
            encoders.push(enc);
        }
        let frame_size = *u.choose(&encoder_frame_sizes(fs))?;
        Ok(Self { encoders, signal: Signal::new(u)?, frame_size, fs })
    }

    /// One multistream packet, or `None` when an encoder fails.
    fn packet(&mut self, u: &mut Unstructured) -> Result<Option<Vec<u8>>> {
        if u.ratio(1, 8)? {
            self.signal.change(u)?;
        }
        if u.ratio(1, 8)? {
            for enc in &mut self.encoders {
                setup(u, enc)?;
            }
        }
        let n = self.encoders.len();
        let mut out = Vec::new();
        let mut pcm = Vec::new();
        let mut buf = vec![0u8; 1500];
        let mut joined = vec![0u8; 1600];
        for (s, enc) in self.encoders.iter_mut().enumerate() {
            self.signal.frame(self.fs, enc.channels, self.frame_size, &[], &mut pcm);
            let len = enc.encode(&pcm, self.frame_size, &mut buf);
            if len <= 0 {
                return Ok(None);
            }
            let p = &buf[..len as usize];
            if s + 1 == n {
                out.extend_from_slice(p);
            } else {
                // Self-delimited framing, through the libopus repacketizer.
                let mut rp = CRepacketizer::new();
                if rp.cat(p) != 0 {
                    return Ok(None);
                }
                let m = rp.out(&mut joined, true, false);
                if m <= 0 {
                    return Ok(None);
                }
                out.extend_from_slice(&joined[..m as usize]);
            }
        }
        Ok(Some(out))
    }
}

/// Encoder settings from the input. A value libopus refuses is ignored.
fn setup(u: &mut Unstructured, enc: &mut CEncoder) -> Result<()> {
    use audiopus_sys as sys;
    enc.set(sys::OPUS_SET_BITRATE_REQUEST, u.int_in_range(500..=300_000)?);
    enc.set(sys::OPUS_SET_VBR_REQUEST, u.arbitrary::<bool>()? as i32);
    enc.set(sys::OPUS_SET_COMPLEXITY_REQUEST, u.int_in_range(0..=10)?);
    enc.set(sys::OPUS_SET_INBAND_FEC_REQUEST, u.int_in_range(0..=2)?);
    enc.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, u.int_in_range(0..=100)?);
    enc.set(sys::OPUS_SET_DTX_REQUEST, u.ratio(1, 4)? as i32);
    let mode = *u.choose(&[-1000, MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY])?;
    enc.set(OPUS_SET_FORCE_MODE_REQUEST, mode);
    let bw = *u.choose(&[-1000, 1101, 1102, 1103, 1104, 1105])?;
    enc.set(sys::OPUS_SET_BANDWIDTH_REQUEST, bw);
    Ok(())
}

fn run(data: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(data);
    let fs = *u.choose(&RATES)?;
    // Mostly valid layouts, sometimes counts that libopus refuses.
    let channels = u.int_in_range(0..=10)?;
    let streams = u.int_in_range(0..=6)?;
    let coupled = u.int_in_range(0..=streams + 1)?;
    let mut mapping = vec![0u8; channels.max(1)];
    for m in mapping.iter_mut() {
        *m = match u.int_in_range(0..=9)? {
            0 => 255,
            1 => u.arbitrary()?,
            _ => u.int_in_range(0..=(streams + coupled).max(1) as u8 - 1)?,
        };
    }

    let rust = MultistreamDecoder::new(
        SampleRate::try_from(fs).unwrap(),
        channels,
        streams,
        coupled,
        &mapping,
    );
    let c = CMsDecoder::new(fs, channels, streams, coupled, &mapping);
    let (mut rust, mut c) = match (rust, c) {
        (Ok(r), Ok(c)) => (r, c),
        (Err(_), Err(_)) => return Ok(()),
        (r, c) => panic!(
            "layout accepted differently: opus1 {:?}, libopus {:?} for {channels} channels, \
             {streams} streams, {coupled} coupled, mapping {mapping:?}",
            r.err(),
            c.err()
        ),
    };

    // The same layout as the channel mapping table of an Ogg Opus header.
    let mut tail = vec![streams as u8, coupled as u8];
    tail.extend_from_slice(&mapping[..channels]);
    let family = *u.choose(&[1u8, 2, 255])?;
    let table = ChannelMappingTable::parse(family, channels as u8, &tail).expect("a whole table");
    assert!(MultistreamDecoder::from_mapping_table(SampleRate::try_from(fs).unwrap(), &table).is_ok(), "the decoder refuses the table of a valid layout");
    assert!(ChannelMappingTable::parse(0, channels as u8, &tail).is_none());
    assert!(ChannelMappingTable::parse(family, channels as u8, &tail[..tail.len() - 1]).is_none());

    let mut source = Source::new(&mut u, streams, coupled)?;
    let max = fs as usize / 25 * 3; // 120 ms
    let f2_5 = fs as usize / 400;
    // Since the decoders were created or reset, they have decoded only
    // whole packets as the encoders wrote them. Their output must then be
    // close (see `assert_close`).
    let mut clean = true;
    let mut modes = Modes::default();
    let steps = u.int_in_range(1..=12)?;
    for step in 0..steps {
        if u.is_empty() {
            break;
        }
        let mut fec = false;
        let packet: Vec<u8> = match u.int_in_range(0..=9)? {
            0..=4 => match source.packet(&mut u)? {
                Some(mut p) => {
                    if u.ratio(1, 4)? {
                        mutate(&mut u, &mut p, true)?;
                        clean = false;
                    }
                    p
                }
                None => continue,
            },
            5 => {
                let n = u.int_in_range(0..=200)?;
                clean = false;
                u.bytes(n)?.to_vec()
            }
            6 => Vec::new(),
            7 => {
                // FEC from the next packet of the source.
                fec = true;
                match source.packet(&mut u)? {
                    Some(p) => p,
                    None => continue,
                }
            }
            8 => {
                rust.reset_state();
                c.reset();
                clean = true;
                modes.reset();
                continue;
            }
            _ => {
                // A new source: other rate, frame size, settings.
                source = Source::new(&mut u, streams, coupled)?;
                continue;
            }
        };
        // The room in the output, in samples per channel.
        let frame_size = match u.int_in_range(0..=5)? {
            0..=2 => source.frame_size * fs as usize / source.fs as usize,
            3 => max,
            4 => f2_5 * u.int_in_range(1..=48)?,
            _ => u.int_in_range(0..=max + 3 * f2_5)?,
        };
        let i16_out = u.arbitrary::<bool>()?;
        let ch = channels;
        // opus1's output buffer: mostly `frame_size` samples per channel,
        // sometimes more or less. With less room opus1 may refuse; it then
        // keeps its state, and libopus is not called.
        let full = frame_size * ch;
        let rust_len = match u.int_in_range(0..=15)? {
            0 => full + u.int_in_range(1..=100)?,
            1 => u.int_in_range(0..=full)?,
            _ => full,
        };
        let short = rust_len < full;
        clean &= !fec && whole_frames(&packet, streams) && !modes.leaves_celt(&packet, streams);
        let what = format!(
            "step {step}: fec={fec}, i16={i16_out}, frame_size={frame_size}, packet {packet:02x?}"
        );
        // opus1's buffer starts with a marker value: the samples after the
        // ones a call returns must keep it.
        let (r, cr) = if i16_out {
            let mut a = vec![MARK_I16; rust_len];
            let mut b = vec![0i16; full];
            let r = if fec {
                rust.decode_fec_i16(&packet, frame_size, &mut a)
            } else {
                rust.decode_i16(&packet, frame_size, &mut a)
            };
            if short && r.is_err() {
                continue;
            }
            let cr = c.decode_i16(&packet, &mut b, frame_size, fec);
            if let Ok(n) = r {
                assert!(n * ch <= rust_len, "{what}: {n} samples per channel for a buffer of {rust_len}");
                assert!(a[n * ch..].iter().all(|&x| x == MARK_I16), "{what}: samples after the {n} returned were written");
                if clean && cr == n as i32 {
                    assert_close(&what, &a[..n * ch], &b[..n * ch], 2.0);
                }
            }
            (r, cr)
        } else {
            let mut a = vec![MARK_F32; rust_len];
            let mut b = vec![0f32; full];
            let r = if fec {
                rust.decode_fec(&packet, frame_size, &mut a)
            } else {
                rust.decode(&packet, frame_size, &mut a)
            };
            if short && r.is_err() {
                continue;
            }
            let cr = c.decode_float(&packet, &mut b, frame_size, fec);
            if let Ok(n) = r {
                assert!(n * ch <= rust_len, "{what}: {n} samples per channel for a buffer of {rust_len}");
                assert!(a[n * ch..].iter().all(|&x| x == MARK_F32), "{what}: samples after the {n} returned were written");
                if let Some(bad) = a[..n * ch].iter().position(|x| !x.is_finite()) {
                    panic!("step {step}: sample {bad} is {}", a[bad]);
                }
                if clean && cr == n as i32 {
                    assert_close(&what, &a[..n * ch], &b[..n * ch], 1e-10);
                }
            }
            (r, cr)
        };
        match (r, cr) {
            (Ok(n), c_n) if c_n >= 0 => {
                assert_eq!(n as i32, c_n, "step {step}: sample count (fec={fec}, frame_size={frame_size})");
                modes.decoded(&packet, streams);
                assert_eq!(
                    rust.final_range(),
                    c.final_range(),
                    "step {step}: final range (fec={fec}, i16={i16_out}, frame_size={frame_size}, packet {packet:02x?})"
                );
            }
            (Err(_), c_n) if c_n < 0 => {}
            (r, c_n) => panic!(
                "step {step}: opus1 {r:?}, libopus {c_n} (fec={fec}, i16={i16_out}, \
                 frame_size={frame_size}, packet {packet:02x?})"
            ),
        }
    }
    Ok(())
}
