//! A stream of packets into the single-stream decoder, against libopus's
//! decoder.
//!
//! The input chooses the decoder's sampling rate and channel count, and a
//! libopus encoder with its own rate, channel count and settings that
//! makes valid packets (any frame duration up to 120 ms, discontinuous
//! transmission, in-band forward error correction (FEC), forced modes and
//! bandwidths). Then it gives a sequence of steps:
//!
//! - decode the next packet with `decode` or `decode_i16`, sometimes
//!   damaged;
//! - lose it: conceal its duration, then rebuild it with `decode_fec` or
//!   `decode_fec_i16` from the packet after it, which is decoded next;
//! - decode bytes from the input as a packet;
//! - conceal a duration from the input, or reset both decoders;
//! - change the encoder's settings or signal.
//!
//! opus1's output buffer sometimes has more or less room than the frame
//! size asks for. More room must not change the result, and `decode` sets
//! it to zero. With too little room opus1 refuses and keeps its state.
//!
//! After every call both decoders must accept or refuse alike. When they
//! accept, they must return the same number of samples and end in the same
//! final range (the state of the range decoder). Decoded `f32` samples
//! must be finite. While the decoders have decoded only whole packets as
//! the encoder wrote them (no loss, no damage, no frame of 0 or 1 bytes,
//! no switch from CELT-only to SILK or Hybrid), the output must be within
//! 50 dB of libopus's.

#![no_main]

mod common;

use audiopus_sys as sys;
use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});

/// Settings of the libopus encoder that makes the packets. A value libopus
/// refuses is ignored.
fn setup(u: &mut Unstructured, enc: &mut CEncoder) -> Result<()> {
    match u.int_in_range(0..=9)? {
        0 => enc.set(sys::OPUS_SET_BITRATE_REQUEST, u.int_in_range(500..=512_000)?),
        1 => enc.set(sys::OPUS_SET_VBR_REQUEST, u.arbitrary::<bool>()? as i32),
        2 => enc.set(sys::OPUS_SET_COMPLEXITY_REQUEST, u.int_in_range(0..=10)?),
        3 => {
            enc.set(sys::OPUS_SET_INBAND_FEC_REQUEST, u.int_in_range(0..=2)?);
            enc.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, u.int_in_range(0..=100)?)
        }
        4 => enc.set(sys::OPUS_SET_DTX_REQUEST, u.arbitrary::<bool>()? as i32),
        5 => {
            let mode = *u.choose(&[-1000, MODE_SILK_ONLY, MODE_HYBRID, MODE_CELT_ONLY])?;
            enc.set(OPUS_SET_FORCE_MODE_REQUEST, mode)
        }
        6 => {
            let bw = *u.choose(&[-1000, 1101, 1102, 1103, 1104, 1105])?;
            enc.set(sys::OPUS_SET_BANDWIDTH_REQUEST, bw)
        }
        7 => enc.set(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, *u.choose(&[-1000, 1, 2])?),
        8 => enc.set(sys::OPUS_SET_PREDICTION_DISABLED_REQUEST, u.arbitrary::<bool>()? as i32),
        _ => enc.set(sys::OPUS_SET_SIGNAL_REQUEST, *u.choose(&[-1000, 3001, 3002])?),
    };
    Ok(())
}

/// Encoder frame durations in units of 2.5 ms: 2.5 to 120 ms.
const DURATIONS: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];

/// The libopus encoder that makes the packets, and its input.
struct Source {
    enc: CEncoder,
    signal: Signal,
    /// Samples per channel in 2.5 ms at the encoder's rate.
    f2_5: usize,
    /// The encoder's frame size in samples per channel.
    frame: usize,
    pcm: Vec<f32>,
    buf: Vec<u8>,
}

impl Source {
    fn new(u: &mut Unstructured) -> Result<Self> {
        let fs = *u.choose(&RATES)?;
        let channels = u.int_in_range(1..=2)?;
        let app = *u.choose(&[2048, 2049, 2051])?;
        let mut enc = CEncoder::new(fs, channels, app);
        for _ in 0..u.int_in_range(0..=4)? {
            setup(u, &mut enc)?;
        }
        let f2_5 = fs as usize / 400;
        let frame = f2_5 * *u.choose(&DURATIONS)?;
        Ok(Self { enc, signal: Signal::new(u)?, f2_5, frame, pcm: Vec::new(), buf: vec![0; 4000] })
    }

    /// The next packet, sometimes damaged, or `None` when the encoder
    /// fails (for example with an output buffer that is too small). The
    /// flag says whether the packet is as the encoder wrote it.
    fn next(&mut self, u: &mut Unstructured) -> Result<Option<(Vec<u8>, bool)>> {
        let extra = if u.ratio(1, 16)? {
            let n = u.int_in_range(0..=64)?;
            u.bytes(n)?
        } else {
            &[][..]
        };
        let (fs, ch) = (self.enc.fs, self.enc.channels);
        self.signal.frame(fs, ch, self.frame, extra, &mut self.pcm);
        let max_bytes = if u.ratio(1, 8)? { u.int_in_range(1..=200)? } else { self.buf.len() };
        let n = self.enc.encode(&self.pcm, self.frame, &mut self.buf[..max_bytes]);
        if n <= 0 {
            return Ok(None);
        }
        let mut p = self.buf[..n as usize].to_vec();
        let mut intact = true;
        if u.ratio(1, 6)? {
            let keep_toc = u.arbitrary()?;
            mutate(u, &mut p, keep_toc)?;
            intact = false;
        }
        Ok(Some((p, intact)))
    }
}

struct Pair {
    rust: OpusDecoder,
    c: CDecoder,
    fs: i32,
    channels: usize,
    /// Since the decoders were created or reset, they have only decoded
    /// whole packets as the encoder wrote them, with no frame left out
    /// (a frame of 0 or 1 bytes is concealed). Their output must then be
    /// close: the decoder of opus1 differs from libopus's only in
    /// rounding. After a loss or a damaged packet the concealment can take
    /// another course on a rounding difference, so the output is not
    /// compared any more.
    clean: bool,
    modes: Modes,
}

impl Pair {
    /// Run one call on both decoders and compare the results.
    ///
    /// libopus's output buffer holds `frame_size` samples per channel.
    /// opus1's holds `rust_len` samples. With more room the result must be
    /// the same, and `decode` sets the rest to zero. With less room opus1
    /// may refuse; it then must not change its state, which the next calls
    /// show, and libopus is not called.
    fn call(&mut self, step: usize, packet: &[u8], frame_size: usize, fec: bool, i16_out: bool, rust_len: usize) {
        let ch = self.channels;
        let full = frame_size * ch;
        let short = rust_len < full;
        let compare =
            self.clean && !fec && whole_frames(packet, 1) && !self.modes.leaves_celt(packet, 1);
        let what = format!(
            "step {step}: {} Hz {ch} ch, fec={fec}, i16={i16_out}, frame_size={frame_size}, \
             room {rust_len}, packet {packet:02x?}",
            self.fs
        );
        let (r, c) = if i16_out {
            let mut a = vec![0i16; rust_len];
            let r = if fec {
                self.rust.decode_fec_i16(packet, frame_size, &mut a)
            } else {
                self.rust.decode_i16(packet, frame_size, &mut a)
            };
            if short && r.is_err() {
                return;
            }
            let mut b = vec![0i16; full];
            let c = self.c.decode_i16(packet, &mut b, frame_size, fec);
            if let Ok(n) = r {
                assert!(n * ch <= rust_len, "{what}: {n} samples written");
                if compare && c == n as i32 {
                    assert_close(&what, &a[..n * ch], &b[..n * ch], 2.0);
                }
            }
            (r, c)
        } else {
            let mut a = vec![f32::NAN; rust_len];
            let r = if fec {
                self.rust.decode_fec(packet, frame_size, &mut a)
            } else {
                self.rust.decode(packet, frame_size, &mut a)
            };
            if short && r.is_err() {
                return;
            }
            let mut b = vec![0f32; full];
            let c = self.c.decode_float(packet, &mut b, frame_size, fec);
            if let Ok(n) = r {
                assert!(n * ch <= rust_len, "{what}: {n} samples written");
                if let Some(bad) = a[..n * ch].iter().position(|x| !x.is_finite()) {
                    panic!("{what}: sample {bad} is {}", a[bad]);
                }
                if !fec {
                    assert!(a[n * ch..].iter().all(|&x| x == 0.0), "{what}: the room after the samples is not zero");
                }
                if compare && c == n as i32 {
                    assert_close(&what, &a[..n * ch], &b[..n * ch], 1e-10);
                }
            }
            (r, c)
        };
        if !compare {
            self.clean = false;
        }
        match (r, c) {
            (Ok(n), c) if c >= 0 => {
                assert_eq!(n as i32, c, "{what}: sample count");
                assert_eq!(self.rust.final_range(), self.c.final_range(), "{what}: final range");
                self.modes.decoded(packet, 1);
            }
            (Err(_), c) if c < 0 => {}
            (r, c) => panic!("{what}: opus1 {r:?}, libopus {c}"),
        }
    }
}

/// The room in opus1's output buffer, in samples: mostly exactly
/// `frame_size` per channel, sometimes more or less.
fn room(u: &mut Unstructured, frame_size: usize, channels: usize) -> Result<usize> {
    let full = frame_size * channels;
    Ok(match u.int_in_range(0..=15)? {
        0 => full + u.int_in_range(1..=100)?,
        1 => u.int_in_range(0..=full)?,
        _ => full,
    })
}

fn run(data: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(data);
    let fs = *u.choose(&RATES)?;
    let channels = u.int_in_range(1..=2)?;
    let mut pair = Pair {
        rust: OpusDecoder::new(fs, channels).unwrap(),
        c: CDecoder::new(fs, channels),
        fs,
        channels,
        clean: true,
        modes: Modes::default(),
    };

    let mut src = Source::new(&mut u)?;
    let max = fs as usize / 25 * 3; // 120 ms at the decoder's rate
    let dec_f2_5 = fs as usize / 400;

    let steps = u.int_in_range(1..=24)?;
    for step in 0..steps {
        if u.is_empty() {
            break;
        }
        // The room in the output, in samples per channel. Most calls get
        // the packet's duration or 120 ms; some get sizes that do not fit.
        let choose_frame_size = |u: &mut Unstructured, packet: &[u8]| -> Result<usize> {
            let own = c_nb_samples(packet, fs).max(0) as usize;
            Ok(match u.int_in_range(0..=7)? {
                0..=3 => own,
                4 | 5 => max,
                6 => dec_f2_5 * u.int_in_range(1..=48)?,
                _ => u.int_in_range(0..=max + 3 * dec_f2_5)?,
            })
        };
        match u.int_in_range(0..=11)? {
            0..=4 => {
                // The next packet, decoded.
                let Some((p, intact)) = src.next(&mut u)? else { continue };
                pair.clean &= intact;
                let frame_size = choose_frame_size(&mut u, &p)?;
                pair.call(step, &p, frame_size, false, u.arbitrary()?, room(&mut u, frame_size, channels)?);
            }
            5 | 6 => {
                // The next packet is lost. The one after it brings FEC.
                let Some((lost, _)) = src.next(&mut u)? else { continue };
                let lost_size = match u.int_in_range(0..=3)? {
                    0 => choose_frame_size(&mut u, &lost)?,
                    _ => c_nb_samples(&lost, fs).max(0) as usize,
                };
                let Some((p, _)) = src.next(&mut u)? else { continue };
                if u.arbitrary()? {
                    pair.call(step, &p, lost_size, true, u.arbitrary()?, room(&mut u, lost_size, channels)?);
                } else {
                    pair.call(step, &[], lost_size, false, u.arbitrary()?, room(&mut u, lost_size, channels)?);
                }
                let frame_size = choose_frame_size(&mut u, &p)?;
                pair.call(step, &p, frame_size, false, u.arbitrary()?, room(&mut u, frame_size, channels)?);
            }
            7 => {
                // Bytes from the input as a packet.
                let n = u.int_in_range(0..=300)?;
                let p = u.bytes(n)?.to_vec();
                pair.clean = false;
                let frame_size = choose_frame_size(&mut u, &p)?;
                let fec = u.ratio(1, 4)?;
                pair.call(step, &p, frame_size, fec, u.arbitrary()?, room(&mut u, frame_size, channels)?);
            }
            8 => {
                // A loss of a duration from the input.
                let frame_size = if u.arbitrary()? {
                    dec_f2_5 * u.int_in_range(1..=96)?
                } else {
                    u.int_in_range(0..=2 * max)?
                };
                pair.call(step, &[], frame_size, u.arbitrary()?, u.arbitrary()?, room(&mut u, frame_size, channels)?);
            }
            9 => {
                pair.rust.reset_state();
                pair.c.reset();
                pair.clean = true;
                pair.modes.reset();
            }
            10 => {
                for _ in 0..u.int_in_range(1..=3)? {
                    setup(&mut u, &mut src.enc)?;
                }
                if u.arbitrary()? {
                    src.frame = src.f2_5 * *u.choose(&DURATIONS)?;
                }
            }
            _ => {
                if u.arbitrary()? {
                    src.signal.change(&mut u)?;
                } else {
                    src.enc.reset();
                }
            }
        }
    }
    Ok(())
}
