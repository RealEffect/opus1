//! In-band FEC of the first SILK packet after CELT-only packets.
//!
//! SILK starts afresh after CELT-only packets, and libopus first feeds it
//! the 10 ms before the packet in a call to `silk_Encode` of its own
//! (opus_encoder.c:2190-2208). That call runs `silk_control_encoder`, and
//! with it `silk_setup_LBRR` (control_codec.c:403-423), like the call that
//! codes the packet. The first coded packet is then the second with FEC
//! for `silk_setup_LBRR`: its redundant (LBRR) frame has its first gain
//! raised by `max(7 - silk_SMULWB(loss, 0.2 in Q16), 3)` quantization
//! steps, not by the 7 of a first FEC packet.
//!
//! The LBRR frame of a packet travels in the next packet. These tests read
//! the first gain index of the frame and of its LBRR copy with this
//! crate's SILK decoder, from the packets of both encoders.

mod core_common;
mod k24_common;

use core_common::*;
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::dec_api::SilkDecoder;
use opus1::internals::silk::decode_frame::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL};
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusEncoder};

/// The first gain index of the SILK frame of a 20 ms wideband mono packet,
/// decoded as the normal frame or as the LBRR copy it carries.
fn first_gain_index(packet: &[u8], flag: i32) -> i8 {
    let mut dec = SilkDecoder::new();
    dec.init(16000, 1);
    dec.channel_state[0].fs_api_hz = 16000;
    let mut pcm = [0i16; 320];
    let mut rc = RangeCoder::new_decoder(&packet[1..]);
    dec.decode(&mut rc, &mut pcm, flag, true, 20, 16000);
    dec.channel_state[0].indices.gains_indices[0]
}

/// Whether a SILK packet carries an LBRR frame: the LBRR flag after the
/// voice activity flag of its one frame.
fn has_lbrr(packet: &[u8]) -> bool {
    packet[1] & 0x40 != 0
}

/// For each SILK-only packet that follows CELT-only packets: how far the
/// first gain index of its LBRR copy (in the next packet) lies above the
/// frame's own, for opus1 and for libopus.
fn gain_increases(application: (i32, Application), loss: i32) -> Vec<(usize, i32, i32)> {
    let rate = 16000;
    let frame = 320;
    let pcm = k24_common::speech(rate, 6.0);
    let mut c_enc = CEncoder::new(rate, 1, application.0);
    c_enc.complexity(5).bitrate(32000).bandwidth(BW_WB).fec(true, loss);
    let mut enc = OpusEncoder::new(rate as i32, 1, application.1).unwrap();
    enc.set_complexity(5).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(32000)).unwrap();
    enc.set_bandwidth(Bandwidth::Wideband);
    enc.set_inband_fec(true);
    enc.set_packet_loss_perc(loss).unwrap();
    let mut buf = vec![0u8; 1500];
    let (mut packets, mut c_packets) = (Vec::new(), Vec::new());
    for (i, input) in pcm.chunks_exact(frame).enumerate() {
        // 5 CELT-only packets, then 15 SILK-only packets.
        let celt = i % 20 < 5;
        c_enc.force_mode(if celt { MODE_CELT } else { MODE_SILK });
        enc.set_force_mode(Some(if celt { ForcedMode::CeltOnly } else { ForcedMode::SilkOnly }));
        let n = enc.encode(input, frame, &mut buf).unwrap();
        packets.push(buf[..n].to_vec());
        c_packets.push(c_enc.encode(input));
    }
    let mut out = Vec::new();
    for i in (5..packets.len() - 1).step_by(20) {
        let increase = |p: &[Vec<u8>]| {
            assert!(p[i][0] & 0x80 == 0 && p[i + 1][0] & 0x80 == 0, "packet {i}: SILK-only");
            assert!(has_lbrr(&p[i + 1]), "packet {}: no LBRR frame", i + 1);
            (first_gain_index(&p[i + 1], FLAG_DECODE_LBRR) - first_gain_index(&p[i], FLAG_DECODE_NORMAL)) as i32
        };
        out.push((i, increase(&packets), increase(&c_packets)));
    }
    out
}

/// At 20 % loss the increase is `7 - (20 * 13107 >> 16) = 4` for the first
/// SILK packet after CELT-only packets, in libopus's packets and in
/// opus1's. opus1 ran `silk_setup_LBRR` once for the packet and raised
/// the gain by 7: its LBRR frame was coded about 4 dB coarser than
/// libopus's. At 10 % loss the increase is 6. A gain index of 63 is the
/// largest, so an increase that would go beyond it is not compared.
#[test]
fn the_first_silk_packet_after_celt_has_the_lbrr_gain_of_libopus() {
    for application in [(APP_AUDIO, Application::Audio), (APP_VOIP, Application::Voip)] {
        for (loss, want) in [(20, 4), (10, 6)] {
            let increases = gain_increases(application, loss);
            assert!(increases.len() >= 10);
            for (i, got, c) in increases {
                let what = format!("{:?}, {loss} % loss, packet {i}", application.1);
                if c < want {
                    // The index was within `want` of the largest.
                    continue;
                }
                assert_eq!(c, want, "{what}: libopus");
                assert_eq!(got, want, "{what}: opus1");
            }
        }
    }
}
