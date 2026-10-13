//! Decoder behaviour that still differs from libopus. Every test here
//! states what libopus does, fails on the current code, and is ignored
//! with the reason. None of them is specific to 24 kHz. Run them with
//! `cargo test --release --test k24_known_gaps -- --ignored --nocapture`
//! to see the measured numbers.

mod k24_common;
use k24_common::*;

fn rms_dbfs(x: &[f32]) -> f64 {
    let p: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * p.max(1e-20).log10()
}

/// Decode with one frame in ten lost. Returns, for libopus and opus-rs,
/// the mean level of the concealed frames, and the lowest SNR of opus-rs
/// against libopus in the concealed frames, in the first good frame after a
/// loss and in the fifth.
fn loss_run(mode: i32, bw: i32, bitrate: i32) -> (f64, f64, f64, f64, f64) {
    let pcm = speech_fullband(24000, 3.0);
    let mut enc = CEnc::voip(24000, 1);
    enc.force_mode(mode).bandwidth(bw).bitrate(bitrate);
    let packets = enc.encode_all(&pcm, 480);
    let lost: Vec<bool> = (0..packets.len()).map(|i| i % 10 == 7).collect();
    let mut c_dec = CDec::new(24000, 1);
    let mut rs_dec = opus_rs::OpusDecoder::new(24000, 1).unwrap();
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
/// opus-rs `OpusDecoder::conceal_celt` (src/lib.rs) decodes an empty CELT
/// frame instead. That takes the decoder's silence branch, which outputs
/// only the fading tail of the previous frame and sets every band energy
/// to -28 (src/celt.rs, `if silence`). The concealed frame is about 10 dB
/// quieter than libopus's, and because CELT predicts each frame's energies
/// from the previous frame's, the following good frames come out at the
/// wrong level for roughly 100 ms.
#[test]
#[ignore = "CELT loss concealment (libopus celt_decode_lost) is not ported: concealed frames fade to silence and the following ~5 good frames decode at the wrong level"]
fn celt_only_loss_is_concealed_like_libopus() {
    let (level_c, level_r, in_loss, after1, after5) = loss_run(MODE_CELT, BW_SWB, 48000);
    println!("CELT-SWB: concealed-frame level libopus {level_c:.1} dBFS, opus-rs {level_r:.1} dBFS; against libopus: concealed {in_loss:.1} dB, first good frame {after1:.1} dB, fifth {after5:.1} dB");
    assert!((level_c - level_r).abs() < 3.0, "concealed level differs by {:.1} dB", level_r - level_c);
    assert!(after5 > 40.0, "fifth good frame after a loss is {after5:.1} dB against libopus");
}

/// Hybrid stream, one frame in ten lost. The SILK layer is concealed
/// bit-exactly; the CELT layer (above 8 kHz) has the fault described above.
#[test]
#[ignore = "the CELT layer of a lost Hybrid frame is not concealed like libopus (celt_decode_lost is not ported)"]
fn hybrid_loss_is_concealed_like_libopus() {
    let (level_c, level_r, in_loss, after1, after5) = loss_run(MODE_HYBRID, BW_SWB, 32000);
    println!("Hybrid-SWB: concealed-frame level libopus {level_c:.1} dBFS, opus-rs {level_r:.1} dBFS; against libopus: concealed {in_loss:.1} dB, first good frame {after1:.1} dB, fifth {after5:.1} dB");
    assert!(in_loss > 40.0 && after1 > 40.0, "concealed {in_loss:.1} dB, next {after1:.1} dB against libopus");
}

/// libopus decodes any stream with any decoder channel count: a mono
/// stream is copied to both channels of a stereo decoder, a stereo stream
/// is mixed down in a mono decoder (`CELT_SET_CHANNELS`, and
/// `nChannelsAPI` / `nChannelsInternal` in SILK). libopus's own encoder
/// codes stereo input as mono at low bitrates, so a stereo decoder meets
/// mono packets in ordinary streams.
///
/// opus-rs `OpusDecoder::decode` (src/lib.rs) returns
/// "Channel count mismatch between packet and decoder".
#[test]
#[ignore = "a decoder whose channel count differs from the packet's returns an error; libopus converts"]
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
/// opus-rs has no such call: `OpusDecoder::decode` takes no FEC flag, and
/// the redundant copy is skipped (src/silk/dec_api.rs, "Skip LBRR data").
/// A lost frame can only be concealed. This test measures both against the
/// loss-free decode: libopus recovers the lost frames from the FEC data,
/// opus-rs conceals them.
#[test]
#[ignore = "no FEC decode path: OpusDecoder has no decode_fec, lost frames can only be concealed"]
fn lost_frames_are_recovered_from_in_band_fec() {
    let pcm = speech_fullband(24000, 3.0);
    let mut enc = CEnc::voip(24000, 1);
    enc.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(32000).fec(true, 20);
    let packets = enc.encode_all(&pcm, 480);
    let lost: Vec<bool> = (0..packets.len()).map(|i| i % 10 == 7).collect();
    let clean = c_decode_all(&packets, 24000, 1);

    let mut c_dec = CDec::new(24000, 1);
    let mut rs_dec = opus_rs::OpusDecoder::new(24000, 1).unwrap();
    let (mut c, mut r) = (Vec::new(), Vec::new());
    let mut buf = vec![0f32; 480];
    for (i, p) in packets.iter().enumerate() {
        if lost[i] {
            // libopus: recover frame i from the FEC data in packet i + 1.
            c.extend(c_dec.decode(Some(&packets[i + 1]), 480, true).unwrap());
            rs_dec.decode(&[], 480, &mut buf).unwrap();
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
    let (fec, plc) = (lost_snr(&c), lost_snr(&r));
    println!("lost frames against the loss-free decode: libopus with FEC {fec:.1} dB, opus-rs with concealment {plc:.1} dB");
    assert!(plc > fec - 1.0, "opus-rs {plc:.1} dB, libopus with FEC {fec:.1} dB");
}

/// libopus `opus_decode_float` does not clip: a full-scale signal comes
/// back with peaks above 1.0. opus-rs clamps every output sample to
/// [-1, 1] (src/lib.rs, `decode_frame`). This is the only difference left
/// in good-packet decoding, and it only shows on signals near full scale.
#[test]
#[ignore = "opus-rs clamps float output to [-1, 1]; libopus opus_decode_float does not clip"]
fn float_output_is_not_clipped() {
    let pcm = full_scale(24000, 24000);
    let mut enc = CEnc::audio(24000, 1);
    enc.force_mode(MODE_CELT).bandwidth(BW_SWB).bitrate(48000);
    let packets = enc.encode_all(&pcm, 480);
    let reference = c_decode_all(&packets, 24000, 1);
    let ours = rs_decode_all(&packets, 24000, 1).unwrap();
    let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let cmp = compare(&reference, &ours, 1, 480, 24000);
    println!("peak libopus {:.4}, opus-rs {:.4}; {cmp}", peak(&reference), peak(&ours));
    assert!(cmp.worst_frame_snr_db > 80.0, "{cmp}");
}

/// Discontinuous transmission (DTX) when the pause is not digital silence.
///
/// With DTX the encoder sends nothing while the talker pauses, apart from
/// one frame every 400 ms, and the decoder conceals the missing frames. In
/// a SILK-only stream opus-rs does that bit-exactly, comfort noise
/// included (k24_silk_loss_cng.rs). In a CELT-only or Hybrid stream the
/// missing frames go through the CELT concealment that is not ported (see
/// `celt_only_loss_is_concealed_like_libopus` above):
///
/// - CELT-only: libopus keeps the background noise going during the pause;
///   opus-rs outputs digital silence, and the frame that arrives every
///   400 ms comes out up to 15 dB too quiet.
/// - Hybrid: the level is right (the SILK layer supplies the noise), but
///   the samples differ from libopus's in the pause and in the first frame
///   after it.
///
/// `dtx_in_hybrid_and_celt_streams` in k24_decoder_coverage.rs passes
/// because its pause is digital silence, where there is no noise to keep.
#[test]
#[ignore = "DTX pauses with background noise: CELT-only streams go silent and Hybrid streams differ from libopus (CELT loss concealment is not ported)"]
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
        println!("{name}: {n} missing frames; level in them libopus {level_c:.1} dBFS, opus-rs {level_r:.1} dBFS; first frame after a pause {first_after:.1} dB against libopus");
        if (level_c - level_r).abs() > 3.0 || first_after < 40.0 {
            failures.push(format!("{name}: pause level {level_r:.1} dBFS against {level_c:.1} dBFS, first frame after a pause {first_after:.1} dB"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
