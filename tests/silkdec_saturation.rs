//! Streams that drive the SILK decoder's 32-bit filter state into
//! saturation.
//!
//! The synthesis filter of the decoder keeps its state in 32 bits. When the
//! state would overflow, libopus limits it with `silk_LSHIFT_SAT32`. That
//! macro first limits the value to the range that can be shifted and then
//! shifts, so the largest result of a shift by 4 is `0x7FFFFFF0`, not
//! `0x7FFFFFFF`.
//!
//! `silk_lshift_sat32` in src/silk/macros.rs returns `0x7FFFFFFF` instead.
//! The 16-bit output of the frame in which this happens is the same, because
//! it is clipped. The filter state is not the same, and the following
//! samples and frames drift away from libopus: by one step at first, by the
//! full range after a lost frame.
//!
//! The tests here are ignored until `silk_lshift_sat32` is corrected (the
//! file is not part of the SILK decoder package). With the function body
//!
//! ```text
//! a.clamp(i32::MIN >> shift, i32::MAX >> shift) << shift
//! ```
//!
//! all of them pass. Run them with
//! `cargo test --release --test silkdec_saturation -- --ignored`.
//!
//! The streams come from the unmodified C encoder. Its input is loud and
//! tonal, and the bitrate is low, which is enough to make the decoded
//! signal overshoot.

mod silkdec_common;
use silkdec_common::*;

use audiopus_sys as sys;

/// A stereo tone sweep with the two channels in opposite phase: the mid
/// channel is silent and the side channel carries everything.
fn opposite_phase(seconds: f64) -> Vec<i16> {
    let tone = sweep(16000, (16000.0 * seconds) as usize);
    let negated: Vec<i16> = tone.iter().map(|&v| v.saturating_neg()).collect();
    stereo_from(&tone, &negated)
}

/// Decode `packets` in a mono and a stereo decoder at two rates, without
/// loss and with random losses. In one of the two loss runs, lost packets
/// are recovered from the forward error correction (FEC) data of the next
/// packet where that is possible.
fn check(name: &str, packets: &[Vec<u8>], failures: &mut Vec<String>) {
    let silk_rate = toc(&packets[0]).silk_rate;
    for channels in [2, 1] {
        for rate in [silk_rate, 48000] {
            let what = format!("{name}, {channels} ch decoder at {rate} Hz");
            let (diff, _) = run(packets, rate, channels, |_| Fate::Received);
            failures.extend(diff.failure(&format!("{what}, no loss")));
            for (seed, fec) in [(52, true), (1052, false)] {
                let fates = random_fates(seed, packets.len(), fec);
                let (diff, _) = run(packets, rate, channels, |i| fates[i]);
                failures.extend(diff.failure(&format!("{what}, seed {seed}")));
            }
        }
    }
}

/// The side channel at full scale, lost frames included.
#[test]
fn a_side_channel_at_full_scale_matches_libopus() {
    let pcm = opposite_phase(1.5);
    let mut failures = Vec::new();
    for (bandwidth, frame_ms, vbr, bitrate, loss, dtx) in [
        (BW_NB, 40usize, true, 60000, 25, true),
        (BW_MB, 20, false, 80000, 60, false),
        (BW_WB, 60, true, 12000, 0, false),
        (BW_WB, 10, true, 56000, 20, false),
    ] {
        let mut enc = CEnc::with(16000, 2, sys::OPUS_APPLICATION_VOIP);
        enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK)
            .ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth)
            .ctl(sys::OPUS_SET_BITRATE_REQUEST, bitrate)
            .ctl(sys::OPUS_SET_VBR_REQUEST, vbr as i32)
            .ctl(sys::OPUS_SET_DTX_REQUEST, dtx as i32)
            .force_channels(2);
        if loss > 0 {
            enc.fec(loss);
        }
        let packets = enc.encode_all(&pcm, frame_ms);
        assert!(packets.iter().all(|p| is_silk(p)));
        let name = format!("opposite phase, {} {frame_ms} ms", bandwidth_name(bandwidth));
        check(&name, &packets, &mut failures);
    }
    assert_no_failures(&failures);
}

/// A stream whose bitrate drops to 8 kbit/s again and again while one
/// channel carries a loud tone sweep. Here the decoder differs from libopus
/// without any loss.
#[test]
fn a_loud_sweep_at_changing_bitrate_matches_libopus() {
    let n = 16000 * 5;
    let pcm = stereo_from(&low_voice(16000, n), &sweep(16000, n));
    let mut failures = Vec::new();
    for (frame_ms, bandwidth) in [(60usize, BW_WB), (20, BW_NB), (40, BW_MB)] {
        let mut enc = CEnc::with(16000, 2, sys::OPUS_APPLICATION_VOIP);
        enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK)
            .ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth)
            .fec(20);
        let mut packets = Vec::new();
        for (i, frame) in pcm.chunks_exact(16 * frame_ms * 2).enumerate() {
            let bitrate = [64000, 8000, 40000, 12000, 56000][(i * frame_ms / 480) % 5];
            enc.ctl(sys::OPUS_SET_BITRATE_REQUEST, bitrate);
            packets.push(enc.encode(frame));
        }
        assert!(packets.iter().all(|p| is_silk(p)));
        let name = format!("loud sweep, {} {frame_ms} ms", bandwidth_name(bandwidth));
        check(&name, &packets, &mut failures);
    }
    assert_no_failures(&failures);
}
