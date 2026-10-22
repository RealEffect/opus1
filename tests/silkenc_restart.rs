//! The first frames of a SILK encoder that was restarted, against libopus.
//!
//! After CELT-only packets libopus clears the SILK encoder and feeds it the
//! 10 ms before the frame with a call of `silk_Encode` that codes nothing
//! (`prefillFlag`). That call goes through `silk_encode_frame_FIX`, which
//! takes a seed for the frame and advances the frame counter before it
//! leaves (silk/fixed/encode_frame_FIX.c:120). The first coded frame after a
//! restart therefore has seed 1. opus1 did not count the prefill frame
//! and coded seed 0.
//!
//! The seed is written into the packet. At complexity 0 the plain quantizer
//! is used, which keeps the seed of the frame counter, so the seeds of the
//! two encoders can be compared packet by packet.

mod silkenc_common;
use silkenc_common::*;

use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::decode_indices::{
    silk_decode_indices, silk_stereo_decode_mid_only, silk_stereo_decode_pred,
};
use opus1::internals::silk::decode_pulses::silk_decode_pulses;
use opus1::internals::silk::define::{CODE_CONDITIONALLY, CODE_INDEPENDENTLY};
use opus1::internals::silk::init_decoder::{silk_create_decoder, silk_decoder_set_fs};
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

const MODE_CELT: i32 = 1002;

/// Whether the packet is SILK-only (configurations 0 to 11).
fn is_silk(packet: &[u8]) -> bool {
    packet[0] >> 3 < 12
}

/// The seed of every frame of the first channel of a SILK-only packet
/// without FEC data: the mono channel, or the mid of a 10 or 20 ms stereo
/// packet.
fn seeds(packet: &[u8]) -> Vec<i8> {
    let config = (packet[0] >> 3) as usize;
    assert!(config < 12, "not a SILK-only packet");
    let stereo = packet[0] & 4 != 0;
    let fs_khz = [8, 12, 16][config / 4];
    let frame_ms = [10, 20, 40, 60][config % 4];
    let n_frames = (frame_ms / 20).max(1);
    assert!(!stereo || n_frames == 1, "stereo packets of one frame only");

    let mut rc = RangeCoder::new_decoder(&packet[1..]);
    let mut dec = silk_create_decoder();
    dec.nb_subfr = if frame_ms == 10 { 2 } else { 4 };
    assert_eq!(silk_decoder_set_fs(&mut dec, fs_khz, fs_khz * 1000), 0);
    // The header: each channel's voice activity flags, then its FEC flag.
    let vad: Vec<bool> = (0..n_frames).map(|_| rc.decode_bit_logp(1)).collect();
    assert!(!rc.decode_bit_logp(1), "the packet carries FEC data");
    let mut side_vad = false;
    if stereo {
        side_vad = rc.decode_bit_logp(1);
        assert!(!rc.decode_bit_logp(1), "the packet carries FEC data");
    }

    let mut out = Vec::new();
    for (i, &voiced) in vad.iter().enumerate() {
        if stereo {
            silk_stereo_decode_pred(&mut rc);
            if !side_vad {
                silk_stereo_decode_mid_only(&mut rc);
            }
        }
        dec.vad_flags[i] = voiced as i32;
        let cond = if i == 0 { CODE_INDEPENDENTLY } else { CODE_CONDITIONALLY };
        silk_decode_indices(&mut dec, &mut rc, i as i32, 0, cond);
        out.push(dec.indices.seed);
        // The excitation lies between this frame's indices and the next's.
        let mut pulses = [0i16; 320 + 16];
        silk_decode_pulses(
            &mut rc,
            &mut pulses,
            dec.indices.signal_type as i32,
            dec.indices.quant_offset_type as i32,
            dec.frame_length,
        );
    }
    out
}

struct Case {
    fs: usize,
    channels: usize,
    bandwidth: (Bandwidth, i32),
    /// Duration of the SILK packets.
    silk_ms: usize,
    bitrate: i32,
}

/// SILK packets, then CELT-only packets of 20 ms, then SILK packets again:
/// the seeds in the packets of both encoders.
fn seeds_of_both(case: &Case) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let Case { fs, channels, bandwidth, silk_ms, bitrate } = *case;
    let mut rs = OpusEncoder::new(
        SampleRate::try_from(fs as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
        Application::Audio,
    )
    .unwrap();
    rs.set_bandwidth(bandwidth.0);
    rs.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
    rs.set_complexity(0).unwrap();
    let mut c = CEncoder::new(fs, channels, APP_AUDIO);
    c.bandwidth(bandwidth.1).bitrate(bitrate).complexity(0).force_channels(channels as i32);

    // 6 SILK packets, 4 CELT packets, 9 SILK packets.
    let silk_frame = fs * silk_ms / 1000;
    let celt_frame = fs / 50;
    let plan: Vec<(bool, usize)> = (0..19)
        .map(|i| if (6..10).contains(&i) { (false, celt_frame) } else { (true, silk_frame) })
        .collect();
    let total: usize = plan.iter().map(|p| p.1).sum();
    let pcm = speech(fs, channels, total, 0.5);

    let mut out = vec![0u8; 1500];
    let (mut rs_seeds, mut c_seeds) = (Vec::new(), Vec::new());
    let mut pos = 0;
    for (i, &(silk, frame)) in plan.iter().enumerate() {
        rs.set_force_mode(Some(if silk { ForcedMode::SilkOnly } else { ForcedMode::CeltOnly }));
        c.set(OPUS_SET_FORCE_MODE_REQUEST, if silk { MODE_SILK } else { MODE_CELT });
        let chunk = &pcm[pos * channels..(pos + frame) * channels];
        pos += frame;
        let n = rs.encode(chunk, frame, &mut out).unwrap();
        let rs_packet = &out[..n];
        let c_packet = c.encode(chunk);
        // Both encoders finish the SILK packets with one more SILK packet
        // before they change to CELT.
        assert_eq!(is_silk(rs_packet), is_silk(&c_packet), "packet {i}: SILK or CELT");
        if is_silk(&c_packet) {
            assert_eq!(rs_packet[0], c_packet[0], "packet {i}: first byte");
            rs_seeds.push(seeds(rs_packet));
            c_seeds.push(seeds(&c_packet));
        } else {
            // A marker for the CELT packets.
            rs_seeds.push(Vec::new());
            c_seeds.push(Vec::new());
        }
    }
    (rs_seeds, c_seeds)
}

/// The frame counter of a restarted encoder has counted the prefill frame:
/// the first SILK frame after CELT-only packets has seed 1, as in libopus.
/// The first frame of a new encoder, which is not prefilled, has seed 0.
#[test]
fn the_prefill_frame_takes_a_seed_as_in_libopus() {
    const NB: (Bandwidth, i32) = (Bandwidth::Narrowband, BW_NB);
    const MB: (Bandwidth, i32) = (Bandwidth::Mediumband, BW_MB);
    const WB: (Bandwidth, i32) = (Bandwidth::Wideband, BW_WB);
    let cases = [
        Case { fs: 16_000, channels: 1, bandwidth: WB, silk_ms: 20, bitrate: 24_000 },
        Case { fs: 16_000, channels: 1, bandwidth: WB, silk_ms: 10, bitrate: 24_000 },
        Case { fs: 48_000, channels: 1, bandwidth: MB, silk_ms: 20, bitrate: 18_000 },
        Case { fs: 8_000, channels: 1, bandwidth: NB, silk_ms: 20, bitrate: 14_000 },
        Case { fs: 16_000, channels: 1, bandwidth: WB, silk_ms: 60, bitrate: 24_000 },
        Case { fs: 16_000, channels: 1, bandwidth: NB, silk_ms: 40, bitrate: 14_000 },
        Case { fs: 16_000, channels: 2, bandwidth: WB, silk_ms: 20, bitrate: 40_000 },
        Case { fs: 48_000, channels: 2, bandwidth: WB, silk_ms: 10, bitrate: 40_000 },
    ];
    for case in &cases {
        let what = format!("{} Hz, {} ch, {} ms", case.fs, case.channels, case.silk_ms);
        let (rs, c) = seeds_of_both(case);

        // What libopus's packets show, to check that they are read correctly
        // and that the stream holds what is tested.
        assert_eq!(c[0][0], 0, "{what}: libopus, first frame of the stream");
        let celt_at = c.iter().position(|s| s.is_empty()).expect("no CELT packet");
        let restart = celt_at + c[celt_at..].iter().position(|s| !s.is_empty()).expect("no restart");
        assert_eq!(c[restart][0], 1, "{what}: libopus, first frame after the restart");
        assert!(c.len() - restart >= 5, "{what}: too few packets after the restart");

        assert_eq!(rs, c, "{what}: seeds per packet (CELT packets are empty)");
    }
}
