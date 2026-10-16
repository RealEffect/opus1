//! Decoder behaviour that used to differ from libopus. Every test here
//! states what libopus does. Each one failed, and was ignored with the
//! reason, until the gap it describes was closed; all of them run now.
//! None of them is specific to 24 kHz. Run them with
//! `cargo test --release --test k24_known_gaps -- --nocapture`
//! to see the measured numbers.

mod k24_common;
use k24_common::*;

fn rms_dbfs(x: &[f32]) -> f64 {
    let p: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * p.max(1e-20).log10()
}

/// Decode with one frame in ten lost. Returns, for libopus and opus1,
/// the mean level of the concealed frames, and the lowest SNR of opus1
/// against libopus in the concealed frames, in the first good frame after a
/// loss and in the fifth.
fn loss_run(mode: i32, bw: i32, bitrate: i32) -> (f64, f64, f64, f64, f64) {
    let pcm = speech_fullband(24000, 3.0);
    let mut enc = CEnc::voip(24000, 1);
    enc.force_mode(mode).bandwidth(bw).bitrate(bitrate);
    let packets = enc.encode_all(&pcm, 480);
    let lost: Vec<bool> = (0..packets.len()).map(|i| i % 10 == 7).collect();
    let mut c_dec = CDec::new(24000, 1);
    let mut rs_dec = opus1::OpusDecoder::new(24000, 1).unwrap();
    let (mut c, mut r) = (Vec::new(), Vec::new());
    let mut buf = vec![0f32; 480];
    for (i, p) in packets.iter().enumerate() {
        if lost[i] {
            c.extend(c_dec.decode(None, 480, false).unwrap());
            rs_dec.decode(&[], 480, &mut buf).unwrap();
        } else {
            c.extend(c_dec.decode(Some(p), 480, false).unwrap());
            rs_dec.decode(p, 480, &mut buf).unwrap();
        }
        r.extend_from_slice(&buf);
    }
    let s = frame_snrs(&c, &r, 1, 480);
    let (mut level_c, mut level_r, mut n) = (0.0, 0.0, 0.0);
    let (mut in_loss, mut after1, mut after5) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for i in 0..packets.len() {
        if lost[i] {
            level_c += rms_dbfs(&c[i * 480..(i + 1) * 480]);
            level_r += rms_dbfs(&r[i * 480..(i + 1) * 480]);
            n += 1.0;
            in_loss = in_loss.min(s[i].0);
        } else if i >= 1 && lost[i - 1] {
            after1 = after1.min(s[i].0);
        } else if i >= 5 && lost[i - 5] {
            after5 = after5.min(s[i].0);
        }
    }
    (level_c / n, level_r / n, in_loss, after1, after5)
}

/// CELT-only stream, one frame in ten lost.
///
/// libopus `celt_decode_lost` conceals by repeating the last pitch period
/// (or with shaped noise after 100 ms), and leaves the band energies in
/// place, so the next good frame decodes at the right level.
///
/// This was a gap until `CeltDecoder::decode_lost` (src/celt.rs) ported
/// that function. The test stays here as the first check of it;
/// tests/celtdec_plc.rs covers the concealment in depth.
#[test]
fn celt_only_loss_is_concealed_like_libopus() {
    let (level_c, level_r, in_loss, after1, after5) = loss_run(MODE_CELT, BW_SWB, 48000);
    println!("CELT-SWB: concealed-frame level libopus {level_c:.1} dBFS, opus1 {level_r:.1} dBFS; against libopus: concealed {in_loss:.1} dB, first good frame {after1:.1} dB, fifth {after5:.1} dB");
    assert!((level_c - level_r).abs() < 3.0, "concealed level differs by {:.1} dB", level_r - level_c);
    assert!(after5 > 40.0, "fifth good frame after a loss is {after5:.1} dB against libopus");
}

/// Hybrid stream, one frame in ten lost. The SILK layer is concealed
/// bit-exactly. The CELT layer (above 8 kHz) is concealed with noise drawn
/// from the decoder's random seed, which libopus and opus1 share.
#[test]
fn hybrid_loss_is_concealed_like_libopus() {
    let (level_c, level_r, in_loss, after1, after5) = loss_run(MODE_HYBRID, BW_SWB, 32000);
    println!("Hybrid-SWB: concealed-frame level libopus {level_c:.1} dBFS, opus1 {level_r:.1} dBFS; against libopus: concealed {in_loss:.1} dB, first good frame {after1:.1} dB, fifth {after5:.1} dB");
    assert!(in_loss > 40.0 && after1 > 40.0, "concealed {in_loss:.1} dB, next {after1:.1} dB against libopus");
}

/// libopus decodes any stream with any decoder channel count: a mono
/// stream is copied to both channels of a stereo decoder, a stereo stream
/// is mixed down in a mono decoder (`CELT_SET_CHANNELS`, and
/// `nChannelsAPI` / `nChannelsInternal` in SILK). libopus's own encoder
/// codes stereo input as mono at low bitrates, so a stereo decoder meets
/// mono packets in ordinary streams.
///
/// This was a gap until `OpusDecoder` (src/lib.rs) passed the packet's
/// channel count down to both layers; before, it returned "Channel count
/// mismatch between packet and decoder". tests/core_channels.rs covers the
/// conversion in depth.
#[test]
fn decoder_channel_count_may_differ_from_the_stream() {
    let mono = speech_fullband(24000, 0.5);
    let stereo = to_stereo(&mono, 24000);
    let mut failures = Vec::new();
    for (stream_ch, dec_ch) in [(1usize, 2usize), (2, 1)] {
        for (mode, bw, name, bitrate) in [(MODE_SILK, BW_WB, "SILK-WB", 24000), (MODE_HYBRID, BW_SWB, "Hybrid-SWB", 32000), (MODE_CELT, BW_SWB, "CELT-SWB", 48000)] {
            let mut enc = CEnc::voip(24000, stream_ch);
            enc.force_mode(mode).bandwidth(bw).bitrate(bitrate * stream_ch as i32).force_channels(stream_ch as i32);
            let packets = enc.encode_all(if stream_ch == 1 { &mono } else { &stereo }, 480);
            let reference = c_decode_all(&packets, 24000, dec_ch);
            match rs_decode_all(&packets, 24000, dec_ch) {
                Ok(ours) => {
                    let cmp = compare(&reference, &ours, dec_ch, 480, 24000);
                    if cmp.worst_frame_snr_db < 80.0 {
                        failures.push(format!("{name}, {stream_ch}-channel stream, {dec_ch}-channel decoder: {cmp}"));
                    }
                }
                Err((i, e)) => failures.push(format!("{name}, {stream_ch}-channel stream, {dec_ch}-channel decoder: packet {i}: {e}")),
            }
        }
    }
    println!("{}", failures.join("\n"));
    assert!(failures.is_empty(), "{} of 6 combinations fail", failures.len());
}

/// In-band forward error correction: a SILK or Hybrid packet can carry a
/// low-bitrate copy of the previous frame. `opus_decode(..., decode_fec=1)`
/// on the packet that follows a lost one returns that copy.
///
/// This was a gap until `OpusDecoder::decode_fec` existed: a lost frame
/// could only be concealed. The test measures both decoders against the
/// loss-free decode, and against each other. tests/core_fec.rs covers FEC
/// decoding in depth.
#[test]
fn lost_frames_are_recovered_from_in_band_fec() {
    let pcm = speech_fullband(24000, 3.0);
    let mut enc = CEnc::voip(24000, 1);
    enc.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(32000).fec(true, 20);
    let packets = enc.encode_all(&pcm, 480);
    let lost: Vec<bool> = (0..packets.len()).map(|i| i % 10 == 7).collect();
    let clean = c_decode_all(&packets, 24000, 1);

    let mut c_dec = CDec::new(24000, 1);
    let mut rs_dec = opus1::OpusDecoder::new(24000, 1).unwrap();
    let (mut c, mut r) = (Vec::new(), Vec::new());
    let mut buf = vec![0f32; 480];
    for (i, p) in packets.iter().enumerate() {
        if lost[i] {
            // Recover frame i from the FEC data in packet i + 1.
            c.extend(c_dec.decode(Some(&packets[i + 1]), 480, true).unwrap());
            assert_eq!(rs_dec.decode_fec(&packets[i + 1], 480, &mut buf), Ok(480));
        } else {
            c.extend(c_dec.decode(Some(p), 480, false).unwrap());
            rs_dec.decode(p, 480, &mut buf).unwrap();
        }
        r.extend_from_slice(&buf);
    }
    let lost_snr = |x: &[f32]| -> f64 {
        let (mut sig, mut err) = (0f64, 0f64);
        for i in (0..packets.len()).filter(|&i| lost[i]) {
            for k in i * 480..(i + 1) * 480 {
                sig += (clean[k] as f64).powi(2);
                err += ((clean[k] - x[k]) as f64).powi(2);
            }
        }
        snr_db(sig, err)
    };
    let (fec, ours) = (lost_snr(&c), lost_snr(&r));
    let against_c = compare(&c, &r, 1, 480, 24000);
    println!("lost frames against the loss-free decode: libopus with FEC {fec:.1} dB, opus1 with FEC {ours:.1} dB; opus1 against libopus: {against_c}");
    assert!(ours > fec - 0.1, "opus1 {ours:.1} dB, libopus with FEC {fec:.1} dB");
    assert!(against_c.worst_frame_snr_db > 80.0, "{against_c}");
}

/// libopus `opus_decode_float` does not clip: a full-scale signal comes
/// back with peaks above 1.0. This was a gap while opus1 limited every
/// output sample to [-1, 1]. It only showed on signals near full scale.
#[test]
fn float_output_is_not_clipped() {
    let pcm = full_scale(24000, 24000);
    let mut enc = CEnc::audio(24000, 1);
    enc.force_mode(MODE_CELT).bandwidth(BW_SWB).bitrate(48000);
    let packets = enc.encode_all(&pcm, 480);
    let reference = c_decode_all(&packets, 24000, 1);
    let ours = rs_decode_all(&packets, 24000, 1).unwrap();
    let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let cmp = compare(&reference, &ours, 1, 480, 24000);
    println!("peak libopus {:.4}, opus1 {:.4}; {cmp}", peak(&reference), peak(&ours));
    assert!(cmp.worst_frame_snr_db > 80.0, "{cmp}");
}

/// Discontinuous transmission (DTX) when the pause is not digital silence.
///
/// With DTX the encoder sends nothing while the talker pauses, apart from
/// one frame every 400 ms, and the decoder conceals the missing frames. In
/// a SILK-only stream opus1 does that bit-exactly, comfort noise
/// included (k24_silk_loss_cng.rs). In a CELT-only or Hybrid stream the
/// missing frames go through the CELT concealment
/// (`CeltDecoder::decode_lost`), which keeps the background noise going
/// during the pause as libopus does.
///
/// `dtx_in_hybrid_and_celt_streams` in k24_decoder_coverage.rs covers a
/// pause of digital silence, where there is no noise to keep.
#[test]
fn dtx_pause_with_background_noise_is_concealed_like_libopus() {
    let mut pcm = speech_fullband(24000, 1.0);
    let mut seed = 9u32;
    pcm.extend((0..24000 * 3).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.001
    }));
    pcm.extend(speech_fullband(24000, 1.0));
    let mut failures = Vec::new();
    for (mode, bw, name, bitrate) in [(MODE_CELT, BW_SWB, "CELT-SWB", 40000), (MODE_HYBRID, BW_SWB, "Hybrid-SWB", 24000)] {
        let mut enc = CEnc::voip(24000, 1);
        enc.force_mode(mode).bandwidth(bw).bitrate(bitrate).dtx(true);
        let packets = enc.encode_all(&pcm, 480);
        let reference = c_decode_all(&packets, 24000, 1);
        let ours = rs_decode_all(&packets, 24000, 1).unwrap();
        let s = frame_snrs(&reference, &ours, 1, 480);
        let (mut level_c, mut level_r, mut n) = (0.0, 0.0, 0.0);
        let mut first_after = f64::INFINITY;
        for i in 1..packets.len() {
            if packets[i].len() <= 2 {
                level_c += rms_dbfs(&reference[i * 480..(i + 1) * 480]);
                level_r += rms_dbfs(&ours[i * 480..(i + 1) * 480]);
                n += 1.0;
            } else if packets[i - 1].len() <= 2 {
                first_after = first_after.min(s[i].0);
            }
        }
        assert!(n > 50.0, "{name}: expected DTX packets, got {n}");
        let (level_c, level_r) = (level_c / n, level_r / n);
        println!("{name}: {n} missing frames; level in them libopus {level_c:.1} dBFS, opus1 {level_r:.1} dBFS; first frame after a pause {first_after:.1} dB against libopus");
        if (level_c - level_r).abs() > 3.0 || first_after < 40.0 {
            failures.push(format!("{name}: pause level {level_r:.1} dBFS against {level_c:.1} dBFS, first frame after a pause {first_after:.1} dB"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
