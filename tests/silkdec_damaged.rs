//! Damaged data: packets cut short, packets with flipped bits, packets
//! whose payload is random, all zeros or all ones.
//!
//! The SILK decoder cannot tell damaged data from good data: every bit
//! pattern decodes to something. libopus and opus1 must decode it to the
//! same thing, sample for sample, and neither may fail. That is a far
//! stronger check than "it does not panic": damaged data reaches parameter
//! values and filter states that no encoder produces.
//!
//! One step of libopus needs care. After the SILK data of a SILK-only
//! frame, at least 17 unread bits mean that the frame ends with a CELT
//! frame (redundancy, written by the encoder when it changes mode).
//! Damaged data often leaves that many bits unread, and the C decoder then
//! mixes a CELT frame into its output. The SILK decoder is not involved in
//! that, so the output of such a frame is not compared. Neither is the
//! frame after it: CELT output can exceed full scale, and the soft clipping
//! that `opus_decode` applies to its 16-bit output then also changes the
//! first samples of the next frame. The SILK state is unaffected by all
//! this, and the frames after that are compared again.
//!
//! No packet is lost here. A frame the C decoder took for one with
//! redundancy would be followed by CELT concealment, which this test
//! cannot follow. Some packets are decoded a second time instead, with the
//! flag that asks for their forward error correction (FEC) data, the
//! low-bitrate copy of the packet before. That path never looks for
//! redundancy.
//!
//! The last two tests are about arguments instead of data: sizes at the
//! edge, and every public call in random order.

mod silkdec_common;
use silkdec_common::*;

use opus1::range_coder::RangeCoder;
use opus1::silk::dec_api::SilkDecoder;
use opus1::silk::decode_frame::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST};
use opus1::silk::decoder_structs::SilkDecControl;

/// The ways a packet is damaged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Damage {
    None,
    /// The last bytes are missing.
    CutShort,
    /// One to three bits are flipped.
    BitFlips,
    /// The payload is replaced by random bytes.
    Random,
    Zeros,
    Ones,
    /// Random bytes follow the payload.
    Extended,
}

/// Damage the payload of a packet with a single frame. The first byte,
/// which says how to read the packet, stays. So do at least two bytes of
/// payload: a shorter payload means "no data" to libopus.
fn damage(rng: &mut Lcg, packet: &[u8], kind: Damage) -> Vec<u8> {
    let mut out = packet.to_vec();
    let payload = packet.len() - 1;
    if payload < 3 {
        return out;
    }
    match kind {
        Damage::None => {}
        Damage::CutShort => out.truncate(1 + 2 + rng.below(payload as u32 - 2) as usize),
        Damage::BitFlips => {
            for _ in 0..1 + rng.below(3) {
                let bit = rng.below(8 * payload as u32) as usize;
                out[1 + bit / 8] ^= 1 << (bit % 8);
            }
        }
        Damage::Random => out[1..].iter_mut().for_each(|b| *b = rng.next() as u8),
        Damage::Zeros => out[1..].fill(0),
        Damage::Ones => out[1..].fill(0xff),
        Damage::Extended => out.extend((0..1 + rng.below(40)).map(|_| rng.next() as u8)),
    }
    out
}

/// What one run over a stream found.
#[derive(Default)]
struct Outcome {
    diff: Diff,
    /// Frames compared, per kind of damage.
    compared: [usize; 7],
    /// Frames decoded with the FEC flag from a damaged packet. They are
    /// always compared.
    fec_from_damaged: usize,
    /// Frames not compared because libopus read a CELT frame after the
    /// SILK data, in this frame or in the one before.
    redundancy: usize,
}

/// Decode a stream in which `kinds` of damage hit about half of the
/// packets, with C libopus and with opus1.
fn run_damaged(
    packets: &[Vec<u8>],
    rate: usize,
    channels: usize,
    seed: u32,
    kinds: &[Damage],
) -> Outcome {
    let mut rng = Lcg(seed);
    let damaged: Vec<(Damage, Vec<u8>)> = packets
        .iter()
        .map(|p| {
            let kind = if rng.below(2) == 0 {
                Damage::None
            } else {
                kinds[rng.below(kinds.len() as u32) as usize]
            };
            (kind, damage(&mut rng, p, kind))
        })
        .collect();

    let mut c = CDec::new(rate, channels);
    let mut rs = RsDec::new(rate, channels);
    let mut out = Outcome::default();
    let mut after_redundancy = false;
    for (i, (kind, p)) in damaged.iter().enumerate() {
        let frame_size = rate * toc(p).frame_ms / 1000;
        // Every fifth packet is first used to recover the "lost" audio
        // before it, damaged or not.
        if i % 5 == 3 {
            let reference = c.decode(Some(p), frame_size, true);
            let ours = rs.fec(p, frame_size);
            out.fec_from_damaged += (*kind != Damage::None) as usize;
            out.diff.add(&reference, &ours, || format!("FEC from packet {i} ({kind:?})"));
            out.diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || {
                format!("FEC from packet {i} ({kind:?})")
            });
        }
        let reference = c.decode(Some(p), frame_size, false);
        let ours = rs.decode(p);
        assert_eq!(reference.len(), ours.len(), "packet {i} ({kind:?})");
        let (bits_read, bytes) = rs.last_bits;
        let redundancy = bits_read + 17 <= 8 * bytes as i32;
        if redundancy || after_redundancy {
            out.redundancy += 1;
            after_redundancy = redundancy;
        } else {
            out.diff.add(&reference, &ours, || format!("packet {i} ({kind:?})"));
            out.compared[*kind as usize] += 1;
        }
        out.diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || format!("packet {i} ({kind:?})"));
    }
    out
}

fn stream(channels: usize, bandwidth: i32, frame_ms: usize) -> Vec<Vec<u8>> {
    let mono = speech(2.0, 3.0);
    let pcm = if channels == 2 { stereo_from(&mono, &speech(9.0, 3.0)) } else { mono };
    let base = match bandwidth {
        BW_NB => 20000,
        BW_MB => 24000,
        _ => 30000,
    } + if frame_ms == 10 { 6000 } else { 0 };
    let mut enc = CEnc::silk(channels, bandwidth, base * channels as i32);
    enc.fec(20).force_channels(channels as i32);
    let packets = enc.encode_all(&pcm, frame_ms);
    assert!(packets.iter().all(|p| is_silk(p) && frames(p).len() == 1 && p.len() > 3));
    packets
}

const STREAMS: [(usize, i32, usize); 8] = [
    (1, BW_WB, 20),
    (2, BW_WB, 20),
    (1, BW_NB, 10),
    (2, BW_MB, 10),
    (1, BW_MB, 40),
    (2, BW_NB, 60),
    (1, BW_WB, 60),
    (2, BW_WB, 40),
];

/// Packets that lost their last bytes. The decoder reads zeros past the
/// end, so it never sees unread bits, and every frame is compared.
#[test]
fn packets_cut_short_decode_like_libopus() {
    let mut failures = Vec::new();
    let mut compared = 0;
    for (channels, bandwidth, frame_ms) in STREAMS {
        let packets = stream(channels, bandwidth, frame_ms);
        for decoder_channels in [1, 2] {
            for rate in [silk_rate(bandwidth), 48000] {
                let out = run_damaged(&packets, rate, decoder_channels, 100 + frame_ms as u32, &[Damage::CutShort]);
                compared += out.compared[Damage::CutShort as usize];
                failures.extend(out.diff.failure(&format!(
                    "cut short, {} {frame_ms} ms {channels} ch stream, {decoder_channels} ch decoder at {rate} Hz",
                    bandwidth_name(bandwidth)
                )));
            }
        }
    }
    println!("{compared} frames from packets cut short compared");
    assert_no_failures(&failures);
    assert!(compared > 500, "only {compared} damaged frames were compared");
}

const KINDS: [Damage; 5] =
    [Damage::BitFlips, Damage::Random, Damage::Zeros, Damage::Ones, Damage::Extended];

/// Every stream with every other kind of damage, three times with
/// different random choices. Returns the cases whose samples differ, the
/// number of pitch lags that differ, and what was compared.
fn damaged_matrix() -> (Vec<String>, usize, [usize; 7], usize) {
    let mut failures = Vec::new();
    let mut compared = [0usize; 7];
    let (mut pitch, mut redundancy, mut fec) = (0, 0, 0);
    for (channels, bandwidth, frame_ms) in STREAMS {
        let packets = stream(channels, bandwidth, frame_ms);
        for decoder_channels in [1, 2] {
            for rate in [silk_rate(bandwidth), 48000] {
                for seed in [1u32, 2, 3] {
                    let out = run_damaged(&packets, rate, decoder_channels, seed * 1000 + frame_ms as u32, &KINDS);
                    for (total, n) in compared.iter_mut().zip(out.compared) {
                        *total += n;
                    }
                    redundancy += out.redundancy;
                    fec += out.fec_from_damaged;
                    pitch += out.diff.pitch;
                    failures.extend(out.diff.failure(&format!(
                        "damaged, {} {frame_ms} ms {channels} ch stream, {decoder_channels} ch decoder at {rate} Hz, seed {seed}",
                        bandwidth_name(bandwidth)
                    )));
                }
            }
        }
    }
    println!(
        "frames compared per kind of damage: {compared:?}, {fec} with the FEC flag; {redundancy} not compared (redundancy)"
    );
    // Payloads of zeros, and payloads with bytes added, nearly always
    // leave bits unread. Those are compared through the FEC flag.
    for kind in [Damage::BitFlips, Damage::Random, Damage::Ones] {
        assert!(compared[kind as usize] > 100, "{kind:?}: only {} frames compared", compared[kind as usize]);
    }
    assert!(fec > 500, "only {fec} damaged frames were decoded with the FEC flag");
    (failures, pitch, compared, redundancy)
}

/// Flipped bits, random payloads, payloads of all zeros or all ones, and
/// random bytes after the payload: libopus and opus1 both decode every
/// packet, to the same number of samples and with the same pitch lag.
///
/// The samples are not compared here. Damaged data makes the synthesis
/// filter overflow, and opus1 then drifts away from libopus until
/// `silk_lshift_sat32` is corrected (see tests/silkdec_saturation.rs). The
/// next test compares them.
#[test]
fn damaged_packets_are_accepted_like_libopus() {
    let (_, pitch, _, _) = damaged_matrix();
    assert_eq!(pitch, 0, "pitch lags differ from libopus");
}

/// The same streams, with every sample compared.
#[test]
fn damaged_packets_decode_like_libopus() {
    let (failures, _, _, _) = damaged_matrix();
    assert_no_failures(&failures);
}

/// The lengths and arguments at the edge: no data at all, one byte, an
/// output buffer of exactly the right size, and one sample less.
#[test]
fn empty_and_tiny_input_is_handled() {
    let good = stream(2, BW_WB, 20);
    for (api, internal) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
        for rate in [8000, 12000, 16000, 24000, 48000] {
            for silk in [8000, 12000, 16000] {
                for ms in [0, 10, 20, 40, 60] {
                    let control = SilkDecControl {
                        n_channels_api: api,
                        n_channels_internal: internal,
                        api_sample_rate: rate,
                        internal_sample_rate: silk,
                        payload_size_ms: ms,
                    };
                    let silk_frames = [1usize, 1, 1, 2, 3][[0, 10, 20, 40, 60].iter().position(|&m| m == ms).unwrap()];
                    let n = (rate as usize) * if ms <= 10 { 10 } else { 20 } / 1000;
                    for data in [&[][..], &[0x00][..], &[0xff][..], &[0xff, 0xff][..], &good[7][1..3]] {
                        for flag in [FLAG_DECODE_NORMAL, FLAG_DECODE_LBRR, FLAG_PACKET_LOST] {
                            let mut dec = SilkDecoder::new();
                            let mut rc = RangeCoder::new_decoder(data);
                            let mut out = vec![0i16; n * api as usize];
                            for k in 0..silk_frames {
                                let got = dec
                                    .decode_with_control(&control, &mut rc, &mut out, flag, k == 0)
                                    .unwrap_or_else(|e| panic!("{control:?} flag {flag} frame {k}: {e}"));
                                assert_eq!(got, n, "{control:?} flag {flag}");
                            }
                            // One sample less is refused, and the decoder
                            // goes on as if the call had not happened.
                            let mut rc = RangeCoder::new_decoder(data);
                            let short = n * api as usize - 1;
                            assert!(
                                dec.decode_with_control(&control, &mut rc, &mut out[..short], flag, true).is_err(),
                                "{control:?}: a buffer of {short} samples was accepted"
                            );
                            assert_eq!(dec.decode_with_control(&control, &mut rc, &mut out, flag, true), Ok(n));
                        }
                    }
                }
            }
        }
    }
}

/// Every public call of `SilkDecoder` in random order on one decoder, with
/// valid and invalid arguments: both decode entry points, `init`, `reset`
/// and `decode_bytes`, on random data and on real packets. Nothing may
/// panic, whatever state the calls before left behind, and the decoder
/// must stay usable.
#[test]
fn mixed_calls_in_random_order_do_not_panic() {
    let real: Vec<Vec<u8>> = [(1usize, BW_NB, 10usize), (2, BW_WB, 60), (1, BW_MB, 40), (2, BW_NB, 20)]
        .into_iter()
        .flat_map(|(channels, bandwidth, frame_ms)| stream(channels, bandwidth, frame_ms).into_iter().take(40))
        .collect();
    // Most values are valid, so that most calls get through and build up
    // state for the ones after them.
    let rates = [8000, 12000, 16000, 24000, 48000, 16000, 48000, 8000, 0, 44100, -1, i32::MAX];
    let mut rng = Lcg(4711);
    let mut out = vec![0i16; 2 * 960];
    let (mut ok, mut refused) = (0usize, 0usize);
    for _ in 0..600 {
        let mut dec = SilkDecoder::new();
        let mut payload: Vec<u8> = Vec::new();
        for _ in 0..150 {
            // New data for about half of the calls.
            match rng.below(4) {
                0 => payload = (0..rng.below(200)).map(|_| rng.next() as u8).collect(),
                1 => payload = real[rng.below(real.len() as u32) as usize][1..].to_vec(),
                _ => {}
            }
            let flag = [
                FLAG_DECODE_NORMAL,
                FLAG_DECODE_NORMAL,
                FLAG_DECODE_NORMAL,
                FLAG_PACKET_LOST,
                FLAG_PACKET_LOST,
                FLAG_DECODE_LBRR,
                FLAG_DECODE_LBRR,
                5,
            ][rng.below(8) as usize];
            let new_packet = rng.below(3) != 0;
            let ms = [0, 10, 20, 40, 60, 10, 20, 20, 40, 60, 30, -10][rng.below(12) as usize];
            let silk = [8000, 12000, 16000, 16000, 8000, 12000, 16000, 24000, 0][rng.below(9) as usize];
            let len = [out.len(), out.len(), out.len(), out.len(), 960, 320, 80, 0][rng.below(8) as usize];
            let mut rc = RangeCoder::new_decoder(&payload);
            let accepted = match rng.below(8) {
                0 => {
                    dec.reset();
                    true
                }
                1 => dec.init(rates[rng.below(12) as usize], rng.below(4) as i32) == 0,
                2 => dec.decode_bytes(&payload, &mut out[..len], new_packet) >= 0,
                3 | 4 => dec.decode(&mut rc, &mut out[..len], flag, new_packet, ms, silk) >= 0,
                _ => {
                    let control = SilkDecControl {
                        n_channels_api: [1, 2, 1, 2, 1, 2, 0, 3][rng.below(8) as usize],
                        n_channels_internal: [1, 2, 1, 2, 1, 2, 2, 3][rng.below(8) as usize],
                        api_sample_rate: rates[rng.below(10) as usize],
                        internal_sample_rate: silk,
                        payload_size_ms: ms,
                    };
                    dec.decode_with_control(&control, &mut rc, &mut out[..len], flag, new_packet).is_ok()
                }
            };
            if accepted {
                ok += 1;
            } else {
                refused += 1;
            }
            // The getters must work in every state.
            let _ = (dec.prev_pitch_lag(), dec.frame_length(), dec.sample_rate());
            // No call sequence may leave the decoder unusable: a valid
            // first frame of a packet is always decoded.
            if rng.below(10) == 0 {
                let control = SilkDecControl {
                    n_channels_api: 1 + rng.below(2) as i32,
                    n_channels_internal: 1 + rng.below(2) as i32,
                    api_sample_rate: rates[rng.below(5) as usize],
                    internal_sample_rate: [8000, 12000, 16000][rng.below(3) as usize],
                    payload_size_ms: [10, 20, 40, 60][rng.below(4) as usize],
                };
                let flag = [FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, FLAG_DECODE_LBRR][rng.below(3) as usize];
                let mut rc = RangeCoder::new_decoder(&payload);
                let n = dec.decode_with_control(&control, &mut rc, &mut out, flag, true);
                let ms = if control.payload_size_ms == 10 { 10 } else { 20 };
                assert_eq!(n, Ok((control.api_sample_rate * ms / 1000) as usize), "{control:?} flag {flag}");
            }
        }
    }
    println!("{ok} calls accepted, {refused} refused");
    assert!(ok > 20000 && refused > 20000);
}
