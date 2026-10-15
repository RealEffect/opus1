//! Settings that the SILK encoder derives per packet (silk/control_codec.c),
//! against libopus.

mod silkenc_common;
use silkenc_common::*;

use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::decode_indices::silk_decode_indices;
use opus_rs::silk::decode_pulses::silk_decode_pulses;
use opus_rs::silk::define::CODE_INDEPENDENTLY;
use opus_rs::silk::control_codec::{
    SilkBandwidthControl, silk_control_audio_bandwidth, silk_control_encoder,
};
use opus_rs::silk::define::TRANSITION_FRAMES;
use opus_rs::silk::enc_api::{
    silk_encode, silk_encode_do_vad, silk_encode_prefill, silk_encode_prefill_stereo,
    silk_encode_stereo_packet,
};
use opus_rs::silk::stereo_lr_to_ms::silk_stereo_lr_to_ms;
use opus_rs::silk::init_decoder::{silk_create_decoder, silk_decoder_set_fs};
use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::structs::SilkEncoderState;
use opus_rs::{Application, Bandwidth, ForcedMode, OpusEncoder};

const FS: usize = 16_000;
const FRAME: usize = 320;

/// The gain index of the first subframe of the FEC copy (if the packet has
/// one) and of the frame itself, read from a mono wideband 20 ms SILK packet.
fn first_gain_indices(packet: &[u8]) -> (Option<i32>, i32) {
    assert_eq!(packet[0] >> 3, 9, "not a SILK wideband 20 ms packet");
    let mut rc = RangeCoder::new_decoder(&packet[1..]);
    let mut dec = silk_create_decoder();
    assert_eq!(silk_decoder_set_fs(&mut dec, 16, FS as i32), 0);
    let vad = rc.decode_bit_logp(1);
    let has_lbrr = rc.decode_bit_logp(1);
    let mut lbrr = None;
    if has_lbrr {
        silk_decode_indices(&mut dec, &mut rc, 0, 1, CODE_INDEPENDENTLY);
        lbrr = Some(dec.indices.gains_indices[0] as i32);
        let mut pulses = [0i16; FRAME + 16];
        silk_decode_pulses(
            &mut rc,
            &mut pulses,
            dec.indices.signal_type as i32,
            dec.indices.quant_offset_type as i32,
            FRAME as i32,
        );
    }
    dec.vad_flags[0] = vad as i32;
    silk_decode_indices(&mut dec, &mut rc, 0, 0, CODE_INDEPENDENTLY);
    (lbrr, dec.indices.gains_indices[0] as i32)
}

/// How many gain index steps the FEC copy of a frame lies above the frame
/// itself, for each frame that has a copy in the next packet. The first
/// frame after FEC is switched on is left out: its step is always 7.
fn lbrr_gain_steps(packets: &[Vec<u8>]) -> Vec<i32> {
    let indices: Vec<(Option<i32>, i32)> = packets.iter().map(|p| first_gain_indices(p)).collect();
    let mut steps = Vec::new();
    let mut seen_first = false;
    for w in indices.windows(2) {
        if let Some(lbrr) = w[1].0 {
            // 63 is the top of the scale; a copy clipped there says nothing.
            if seen_first && lbrr < 63 {
                steps.push(lbrr - w[0].1);
            }
            seen_first = true;
        }
    }
    steps
}

/// The value that occurs most often.
fn most_common(values: &[i32]) -> i32 {
    let mut best = (0, 0);
    for &v in values {
        let n = values.iter().filter(|&&x| x == v).count();
        if n > best.1 {
            best = (v, n);
        }
    }
    best.0
}

/// The FEC copy of a frame is coded with a larger gain, so that it takes
/// fewer bits. libopus 1.6 raises the gain index by
/// `max(7 - 0.2 * loss, 3)` (silk/control_codec.c, `silk_setup_LBRR`).
/// opus-rs used the constants of older releases, `max(7 - 0.4 * loss, 2)`:
/// its copies took more bits than libopus's and left fewer for the frame.
#[test]
fn fec_copies_are_coded_with_libopus_gain_increase() {
    let pcm = speech(FS, 1, 150 * FRAME, 0.5);
    let mut failures = Vec::new();
    for loss in [3, 8, 10, 15, 20, 30, 60] {
        let mut rs = OpusEncoder::new(FS as i32, 1, Application::Audio).unwrap();
        rs.set_force_mode(Some(ForcedMode::SilkOnly));
        rs.set_bandwidth(Bandwidth::Wideband);
        rs.bitrate_bps = 32_000;
        rs.use_inband_fec = true;
        rs.packet_loss_perc = loss;
        let mut c = CEncoder::new(FS, 1, APP_AUDIO);
        c.silk_only().bandwidth(BW_WB).bitrate(32_000).fec(true, loss);
        let mut out = vec![0u8; 1500];
        let (mut rs_packets, mut c_packets) = (Vec::new(), Vec::new());
        for frame in pcm.as_chunks::<FRAME>().0 {
            let n = rs.encode(frame, FRAME, &mut out).unwrap();
            rs_packets.push(out[..n].to_vec());
            c_packets.push(c.encode(frame));
        }
        let (rs_steps, c_steps) = (lbrr_gain_steps(&rs_packets), lbrr_gain_steps(&c_packets));
        assert!(c_steps.len() > 40 && rs_steps.len() > 40, "{loss} %: too few FEC copies");
        // What libopus's formula gives, to check that the packets are read
        // correctly.
        let formula = (7 - ((loss * 13107) >> 16)).max(3);
        assert_eq!(most_common(&c_steps), formula, "{loss} % loss: libopus packets");
        if most_common(&rs_steps) != formula {
            failures.push(format!(
                "{loss} % loss: opus-rs raises the gain index by {}, libopus by {formula}",
                most_common(&rs_steps)
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A SILK encoder running at 16 kHz with every field that a change of the
/// internal rate touches set to a value that a reset would not leave.
fn running_encoder() -> Box<SilkEncoderState> {
    let mut enc = Box::new(SilkEncoderState::default());
    assert_eq!(silk_init_encoder(&mut enc, 0), 0);
    assert_eq!(silk_control_encoder(&mut enc, 16, 20, 20_000, 9), 0);
    enc.s_shape.last_gain_index = 33;
    enc.s_shape.harm_boost_smth_q16 = 111;
    enc.s_shape.harm_shape_gain_smth_q16 = 222;
    enc.s_shape.tilt_smth_q16 = -333;
    enc.s_nsq.s_lf_ar_q14 = 444;
    enc.s_nsq.lag_prev = 77;
    enc.s_nsq.prev_gain_q16 = 5 << 16;
    enc.s_cmn.prev_nlsf_q15 = [1000; 16];
    enc.s_cmn.s_lp.in_lp_state = [555, -666];
    enc.s_cmn.s_lp.transition_frame_no = 40;
    enc.s_cmn.s_lp.mode = 1;
    enc.s_cmn.prev_lag = 77;
    enc.s_cmn.prev_signal_type = 2;
    enc.s_cmn.first_frame_after_reset = 0;
    enc
}

/// A change of the internal sampling rate resets the part of the encoder
/// state that libopus's `silk_setup_fs` resets (silk/control_codec.c:254-271):
/// the noise shaping smoothers (the whole `sShape`), the quantizer state, the
/// previous NLSFs and the state of the input low-pass filter. opus-rs kept
/// the smoothers and the filter state of the old rate.
#[test]
fn a_rate_change_resets_what_libopus_resets() {
    for (new_khz, order) in [(12, 10), (8, 10)] {
        let mut enc = running_encoder();
        assert_eq!(silk_control_encoder(&mut enc, new_khz, 20, 20_000, 9), 0);
        assert_eq!(enc.s_cmn.fs_khz, new_khz);
        assert_eq!(enc.s_cmn.predict_lpc_order, order);
        // memset( &psEnc->sShape, 0, ... ), then LastGainIndex = 10.
        assert_eq!(enc.s_shape.last_gain_index, 10);
        assert_eq!(enc.s_shape.harm_boost_smth_q16, 0);
        assert_eq!(enc.s_shape.harm_shape_gain_smth_q16, 0);
        assert_eq!(enc.s_shape.tilt_smth_q16, 0);
        // memset( &psEnc->sCmn.sNSQ, 0, ... ), then lagPrev and prev_gain_Q16.
        assert_eq!(enc.s_nsq.s_lf_ar_q14, 0);
        assert_eq!(enc.s_nsq.lag_prev, 100);
        assert_eq!(enc.s_nsq.prev_gain_q16, 65536);
        assert_eq!(enc.s_cmn.prev_nlsf_q15, [0; 16]);
        // memset( &psEnc->sCmn.sLP.In_LP_State, 0, ... ): the filter state
        // only. The transition itself goes on.
        assert_eq!(enc.s_cmn.s_lp.in_lp_state, [0, 0]);
        assert_eq!(enc.s_cmn.s_lp.transition_frame_no, 40);
        assert_eq!(enc.s_cmn.s_lp.mode, 1);
        assert_eq!(enc.s_cmn.prev_lag, 100);
        assert_eq!(enc.s_cmn.prev_signal_type, 0);
        assert_eq!(enc.s_cmn.first_frame_after_reset, 1);
    }
}

/// Without a change of the rate nothing is reset, also when the packet size
/// changes.
#[test]
fn the_same_rate_resets_nothing() {
    for packet_ms in [20, 10, 60] {
        let mut enc = running_encoder();
        assert_eq!(silk_control_encoder(&mut enc, 16, packet_ms, 20_000, 9), 0);
        assert_eq!(enc.s_cmn.packet_size_ms, packet_ms);
        assert_eq!(enc.s_shape.last_gain_index, 33);
        assert_eq!(enc.s_shape.harm_shape_gain_smth_q16, 222);
        assert_eq!(enc.s_shape.tilt_smth_q16, -333);
        assert_eq!(enc.s_nsq.s_lf_ar_q14, 444);
        assert_eq!(enc.s_cmn.s_lp.in_lp_state, [555, -666]);
        assert_eq!(enc.s_cmn.prev_lag, 77);
        assert_eq!(enc.s_cmn.first_frame_after_reset, 0);
    }
}

/// One or two SILK encoders at 16 kHz coding packets of `packet_ms`.
fn encoders(packet_ms: i32, channels: usize) -> Vec<Box<SilkEncoderState>> {
    (0..channels)
        .map(|_| {
            let mut enc = Box::new(SilkEncoderState::default());
            assert_eq!(silk_init_encoder(&mut enc, 0), 0);
            assert_eq!(silk_control_encoder(&mut enc, 16, packet_ms, 24_000, 9), 0);
            enc.s_cmn.n_channels = channels as i32;
            enc
        })
        .collect()
}

/// Code `pcm` (16 kHz, one or two de-interleaved channels) in packets of
/// `packet_ms` and return the pitch-tracking state behind the cutoff of the
/// input high-pass filter (libopus `variable_HP_smth1_Q15`) after every
/// 60 ms.
fn hp_tracker_every_60_ms(pcm: &[Vec<i16>], packet_ms: i32) -> Vec<i32> {
    let mut enc = encoders(packet_ms, pcm.len());
    let n = 16 * packet_ms as usize;
    let mut track = Vec::new();
    for (i, start) in (0..pcm[0].len() - n + 1).step_by(n).enumerate() {
        let mut rc = RangeCoder::new_encoder(1275);
        let mut bytes = 0;
        let ret = if let [mid, side] = &mut enc[..] {
            let (l, r) = (&pcm[0][start..start + n], &pcm[1][start..start + n]);
            silk_encode_stereo_packet(mid, side, l, r, n, &mut rc, &mut bytes, 40_000, 8000, 0, 1)
        } else {
            let x = &pcm[0][start..start + n];
            silk_encode(&mut enc[0], x, n, &mut rc, &mut bytes, 24_000, 8000, 0, 1)
        };
        assert_eq!(ret, 0);
        if ((i + 1) * packet_ms as usize).is_multiple_of(60) {
            track.push(enc[0].s_cmn.variable_hp_smth1_q15);
        }
    }
    track
}

/// libopus updates the pitch tracker behind the high-pass cutoff once per
/// SILK frame (silk/enc_API.c:392, inside the frame loop), so a 40 or 60 ms
/// packet updates it two or three times. opus-rs updated it once per packet.
///
/// The update depends on the pitch analysis and the voice activity alone,
/// not on how frames are grouped into packets. So the same input coded as
/// 60 ms, 40 ms and 20 ms packets must leave the same tracker state.
#[test]
fn the_high_pass_tracker_is_updated_once_per_frame() {
    let n = 16_000 * 6;
    let stereo = speech(16_000, 2, n, 0.5);
    let to_i16 = |x: f32| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
    let left: Vec<i16> = stereo.as_chunks::<2>().0.iter().map(|lr| to_i16(lr[0])).collect();
    let right: Vec<i16> = stereo.as_chunks::<2>().0.iter().map(|lr| to_i16(lr[1])).collect();
    for pcm in [vec![left.clone()], vec![left, right]] {
        let per_frame = hp_tracker_every_60_ms(&pcm, 20);
        // The tracker must move for the comparison to mean something.
        let moves = per_frame.windows(2).filter(|w| w[0] != w[1]).count();
        println!("{} channel(s): the tracker moves in {moves} of {} steps", pcm.len(), per_frame.len());
        assert!(moves >= per_frame.len() / 10, "{moves} of {} steps", per_frame.len());
        for packet_ms in [40, 60] {
            let got = hp_tracker_every_60_ms(&pcm, packet_ms);
            // 40 ms packets end on a 60 ms boundary every second time.
            let want: Vec<i32> = if packet_ms == 40 {
                per_frame.iter().skip(1).step_by(2).copied().collect()
            } else {
                per_frame.clone()
            };
            assert_eq!(got, want, "{} channel(s), {packet_ms} ms packets", pcm.len());
        }
    }
}

fn bandwidth_control(desired_khz: i32, max_bits: i32) -> SilkBandwidthControl {
    SilkBandwidthControl {
        api_fs_hz: 16_000,
        max_internal_fs_hz: 16_000,
        min_internal_fs_hz: 8_000,
        desired_internal_fs_hz: desired_khz * 1000,
        payload_size_ms: 20,
        allow_bandwidth_switch: true,
        opus_can_switch: false,
        max_bits,
        switch_ready: false,
    }
}

/// In a stereo packet libopus runs the rate switch state machine for the mid
/// and for the side (silk/control_codec.c:101). Each channel that reports
/// that it is ready takes room for the redundant frame out of `maxBits`
/// (silk/control_audio_bandwidth.c:86-88, 114-116).
///
/// The figures are those of the fixed-point build of libopus for a 48 kb/s
/// constant bitrate stereo stream with 20 ms packets, on the packet before a
/// switch from 16 to 12 kHz: 952 bits become 762, then 610. opus-rs, whose
/// caller reduces the bits once, gave SILK 762.
#[test]
fn each_channel_that_is_ready_to_switch_takes_room_out_of_max_bits() {
    let mut channels = encoders(20, 2);
    // Both channels have closed their low-pass filter for a switch down.
    for enc in &mut channels {
        enc.s_cmn.s_lp.mode = -2;
        enc.s_cmn.s_lp.transition_frame_no = 0;
    }
    let mut ctl = bandwidth_control(12, 952);
    assert_eq!(silk_control_audio_bandwidth(&mut channels[0].s_cmn, &mut ctl), 16);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (true, 762));
    assert_eq!(silk_control_audio_bandwidth(&mut channels[1].s_cmn, &mut ctl), 16);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (true, 610));

    // A side whose filter is still closing does not: one reduction.
    channels[1].s_cmn.s_lp.transition_frame_no = 7;
    let mut ctl = bandwidth_control(12, 952);
    silk_control_audio_bandwidth(&mut channels[0].s_cmn, &mut ctl);
    silk_control_audio_bandwidth(&mut channels[1].s_cmn, &mut ctl);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (true, 762));

    // 10 ms packets keep a third for the redundant frame: 5 / (10 + 5).
    let mut ctl = SilkBandwidthControl { payload_size_ms: 10, ..bandwidth_control(12, 480) };
    silk_control_audio_bandwidth(&mut channels[0].s_cmn, &mut ctl);
    assert_eq!(ctl.max_bits, 320);
}

/// The steps of a switch down and of a switch up, as in
/// silk/control_audio_bandwidth.c.
#[test]
fn the_rate_switch_state_machine_follows_libopus() {
    let mut enc = encoders(20, 1).remove(0);
    let cmn = &mut enc.s_cmn;

    // Down. A new transition starts with a full counter and a cleared
    // filter, and the filter closes at double speed.
    cmn.s_lp.in_lp_state = [5, 6];
    let mut ctl = bandwidth_control(8, 1000);
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 16);
    assert_eq!((cmn.s_lp.mode, cmn.s_lp.transition_frame_no), (-2, TRANSITION_FRAMES));
    assert_eq!(cmn.s_lp.in_lp_state, [0, 0]);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (false, 1000));
    // Not while SILK does not allow a switch.
    cmn.s_lp.transition_frame_no = 0;
    let mut ctl =
        SilkBandwidthControl { allow_bandwidth_switch: false, ..bandwidth_control(8, 1000) };
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 16);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (false, 1000));
    // The filter is closed: ready. The caller then switches one step.
    let mut ctl = bandwidth_control(8, 1000);
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 16);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (true, 800));
    let mut ctl = SilkBandwidthControl { opus_can_switch: true, ..bandwidth_control(8, 1000) };
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 12);
    assert_eq!((cmn.s_lp.mode, ctl.switch_ready, ctl.max_bits), (0, false, 1000));

    // Up, from 12 kHz. Ready at once when no transition runs. The switch
    // opens the filter from a cleared state.
    cmn.fs_khz = 12;
    cmn.s_lp.transition_frame_no = 0;
    let mut ctl = bandwidth_control(16, 1000);
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 12);
    assert_eq!((ctl.switch_ready, ctl.max_bits), (true, 800));
    cmn.s_lp.in_lp_state = [5, 6];
    let mut ctl = SilkBandwidthControl { opus_can_switch: true, ..bandwidth_control(16, 1000) };
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 16);
    assert_eq!((cmn.s_lp.mode, cmn.s_lp.transition_frame_no), (1, 0));
    assert_eq!(cmn.s_lp.in_lp_state, [0, 0]);
    // The transition ends when the counter is full.
    cmn.fs_khz = 16;
    cmn.s_lp.transition_frame_no = TRANSITION_FRAMES;
    let mut ctl = bandwidth_control(16, 1000);
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 16);
    assert_eq!(cmn.s_lp.mode, 0);

    // A new encoder takes the desired rate, at most the input's. One that
    // was reset for a switch goes on from the rate it had.
    cmn.fs_khz = 0;
    let mut ctl = SilkBandwidthControl { api_fs_hz: 12_000, ..bandwidth_control(16, 1000) };
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 12);
    cmn.s_lp.saved_fs_khz = 16;
    cmn.s_lp.mode = 0;
    let mut ctl = SilkBandwidthControl { opus_can_switch: true, ..bandwidth_control(12, 1000) };
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 12);
    // A rate above the maximum allowed drops to it at once.
    cmn.fs_khz = 16;
    let mut ctl =
        SilkBandwidthControl { max_internal_fs_hz: 12_000, ..bandwidth_control(12, 1000) };
    assert_eq!(silk_control_audio_bandwidth(cmn, &mut ctl), 12);
}

/// Everything the voice activity detector leaves in the encoder state.
fn vad_state(enc: &SilkEncoderState) -> impl PartialEq + std::fmt::Debug {
    let v = &enc.s_cmn.s_vad;
    (
        (v.ana_state, v.ana_state1, v.ana_state2, v.hp_state, v.counter),
        (v.xnrg_subfr, v.nrg_ratio_smth_q8, v.nl, v.inv_nl, v.noise_level_bias),
        (enc.s_cmn.speech_activity_q8, enc.s_cmn.input_quality_bands_q15, enc.s_cmn.input_tilt_q15),
        (enc.s_cmn.vad_flags[0], enc.s_cmn.indices.signal_type, enc.s_cmn.no_speech_counter),
    )
}

/// libopus prefills a restarted SILK encoder with a call of `silk_Encode`
/// that codes nothing (silk/enc_API.c, `prefillFlag`). That call runs the
/// voice activity detector on the 10 ms it is given, with the encoder set up
/// for 10 ms packets. opus-rs did not run the detector when one channel is
/// coded, so the detector's noise estimates lagged one frame behind
/// libopus's for the first frames after a restart.
///
/// Measured against the fixed-point build of libopus on quiet speech, 8 kHz,
/// for the frames 3 to 5 after the restart: speech activity 20, 26, 6 in
/// libopus; 5, 9, 3 in opus-rs before and 21, 27, 7 now.
///
/// Here the state that a prefill leaves must be the state that the detector
/// leaves when it runs on the same 10 ms in an encoder set up for 10 ms.
#[test]
fn a_prefill_runs_the_voice_activity_detector() {
    let pcm = speech(16_000, 1, 16_000, 0.5);
    let frame: Vec<i16> =
        pcm[8000..8160].iter().map(|&x| (x * 3000.0).round() as i16).collect();
    for packet_ms in [20, 10, 60] {
        let mut prefilled = encoders(packet_ms, 1).remove(0);
        silk_encode_prefill(&mut prefilled, &frame, 1);

        // The detector reads the frame behind one sample of history, which
        // is 0 in a new encoder.
        let mut direct = encoders(10, 1).remove(0);
        let mut shifted = vec![0i16];
        shifted.extend_from_slice(&frame[..159]);
        silk_encode_do_vad(&mut direct, &shifted, 1);

        assert_eq!(vad_state(&prefilled), vad_state(&direct), "{packet_ms} ms packets");
        // The detector did run: a new one has seen no frame.
        assert_eq!(prefilled.s_cmn.s_vad.counter, encoders(10, 1)[0].s_cmn.s_vad.counter + 1);
        // The packet size the encoder is set up for is untouched.
        assert_eq!(prefilled.s_cmn.frame_length, 16 * packet_ms.min(20));
    }
}

/// The stereo prefill splits the bits between mid and side from the target
/// rate that libopus computes for the prefill frame (silk/enc_API.c:395-423):
/// the bitrate less the payback of the bits that earlier packets used beyond
/// their share. opus-rs passed the bitrate itself.
///
/// With 10000 bits to pay back, 36 kb/s become 16 kb/s. That is under what
/// a wideband mid alone needs, so the prefill frame has no stereo width.
#[test]
fn the_stereo_prefill_pays_back_the_bit_reservoir() {
    let stereo = speech(16_000, 2, 16_000, 0.5);
    let to_i16 = |x: f32| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
    let left: Vec<i16> = stereo.as_chunks::<2>().0.iter().skip(8000).take(160).map(|lr| to_i16(lr[0])).collect();
    let right: Vec<i16> = stereo.as_chunks::<2>().0.iter().skip(8000).take(160).map(|lr| to_i16(lr[1])).collect();
    for (exceeded, target) in [(0, 36_000), (10_000, 16_000), (3_000, 30_000)] {
        let mut enc = encoders(20, 2);
        enc[0].n_bits_exceeded = exceeded;
        let [mid, side] = &mut enc[..] else { unreachable!() };
        silk_encode_prefill_stereo(mid, side, &left, &right, 36_000, 1);

        // What silk_stereo_LR_to_MS makes of the frame at libopus's target.
        let mut want = encoders(20, 1).remove(0);
        let (mut x1, mut x2) = (vec![0i16; 162], vec![0i16; 162]);
        x1[2..].copy_from_slice(&left);
        x2[2..].copy_from_slice(&right);
        let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
        silk_stereo_lr_to_ms(
            &mut want.stereo, &mut x1, &mut x2, &mut ix, &mut mid_only, &mut rates, target, 0, false, 16, 160,
        );
        let state = |e: &SilkEncoderState| {
            let s = &e.stereo;
            (s.pred_prev_q13, s.width_prev_q14, s.smth_width_q14, s.mid_side_amp_q0, s.silent_side_len)
        };
        assert_eq!(state(&enc[0]), state(&want), "{exceeded} bits to pay back");
        assert_eq!(enc[0].stereo.pred_ix[0], ix);
        assert_eq!(enc[0].n_bits_exceeded, exceeded, "the prefill leaves the reservoir alone");
    }
    // The payback changes the outcome: full width at 36 kb/s, none at 16.
    let width = |exceeded: i32| {
        let mut enc = encoders(20, 2);
        enc[0].n_bits_exceeded = exceeded;
        let [mid, side] = &mut enc[..] else { unreachable!() };
        silk_encode_prefill_stereo(mid, side, &left, &right, 36_000, 1);
        enc[0].stereo.width_prev_q14
    };
    assert_eq!((width(0), width(10_000)), (1 << 14, 0));
}
