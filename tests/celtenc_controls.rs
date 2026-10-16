//! The controls of the CELT encoder that the Opus layer uses, against the
//! same requests of libopus (celt_encoder.c, `opus_custom_encoder_ctl`):
//! `reset_state` (OPUS_RESET_STATE), `set_prediction` (CELT_SET_PREDICTION)
//! and `set_stream_channels` (CELT_SET_CHANNELS).

mod celtenc_common;

use celtenc_common::c_celt::{CELT_SET_CHANNELS, Stream};
use celtenc_common::*;

fn headers_of(st: &Stream, payloads: &[Vec<u8>]) -> Vec<Header> {
    headers_raw(payloads, st.stream_channels, st.end_band, st.format().lm())
}

fn headers_raw(payloads: &[Vec<u8>], channels: usize, end: usize, lm: usize) -> Vec<Header> {
    payloads
        .iter()
        .map(|p| parse_header(p, channels, end, lm))
        .collect()
}

/// After `reset_state` the encoder codes exactly what a new encoder with
/// the same settings codes. The settings stay, among them the prediction
/// mode and the number of coded channels.
#[test]
fn reset_state_leaves_the_state_of_a_new_encoder() {
    let first = stereo_of(&music(48_000, 1, 0), &music(48_000, 1, 1), 0.3);
    let second = stereo_of(&speech(48_000, 1), &music(48_000, 1, 1), 0.2);
    for (prediction, stream_channels, vbr) in [(2, 2, true), (1, 2, false), (2, 1, true)] {
        let mut st = Stream::new(48_000, 2, 20, 96_000);
        st.complexity = 9;
        st.vbr = vbr;
        st.prediction = prediction;
        st.stream_channels = stream_channels;
        let mut used = st.rs_encoder();
        for frame in first.chunks_exact(st.frame * 2) {
            st.rs_frame(&mut used, frame);
        }
        used.reset_state();
        let mut fresh = st.rs_encoder();
        for (k, frame) in second.chunks_exact(st.frame * 2).enumerate() {
            assert_eq!(
                st.rs_frame(&mut used, frame),
                st.rs_frame(&mut fresh, frame),
                "prediction {prediction}, {stream_channels} channels: frame {k} after the reset"
            );
        }
    }
}

/// The first frames after `OPUS_RESET_STATE` have the header that libopus
/// writes after its reset.
#[test]
fn reset_state_matches_libopus() {
    let pcm = stereo_of(&music(48_000, 2, 0), &music(48_000, 2, 1), 0.3);
    let mut st = Stream::new(48_000, 2, 20, 96_000);
    st.complexity = 5;
    let at = 100;
    let (c, rs) = st.enc_both_with(&pcm, |k, c_enc, rs_enc| {
        if k == at {
            c_enc.reset();
            rs_enc.reset_state();
        }
    });
    let (hc, hr) = (headers_of(&st, &c), headers_of(&st, &rs));
    assert_eq!(hr[at], hc[at], "first frame after the reset");
}

/// `CELT_SET_PREDICTION` 0 makes every frame intra and switches the
/// pre-filter off. 1 switches the pre-filter off and keeps the prediction.
/// Values outside 0 to 2 are refused.
#[test]
fn prediction_modes() {
    let pcm = speech(48_000, 3);
    for prediction in [0, 1, 2] {
        let mut st = Stream::new(48_000, 1, 40, 48_000);
        st.complexity = 5;
        st.prediction = prediction;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let (hc, hr) = (headers_of(&st, &c), headers_of(&st, &rs));
        let count = |h: &[Header], f: fn(&Header) -> bool| h.iter().filter(|x| f(x)).count();
        let intra = |x: &Header| x.intra;
        let filtered = |x: &Header| x.postfilter.is_some();
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        println!(
            "prediction {prediction}: intra {} (libopus {}), filtered {} (libopus {}) of {}; \
             opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            count(&hr, intra),
            count(&hc, intra),
            count(&hr, filtered),
            count(&hc, filtered),
            hr.len()
        );
        match prediction {
            0 => {
                assert_eq!(count(&hc, intra), hc.len());
                assert_eq!(count(&hr, intra), hr.len());
            }
            _ => {
                assert!(count(&hc, intra) * 10 < hc.len());
                assert!(count(&hr, intra) * 10 < hr.len());
            }
        }
        if prediction < 2 {
            assert_eq!(count(&hc, filtered), 0);
            assert_eq!(count(&hr, filtered), 0);
        } else {
            assert!(count(&hr, filtered) * 2 > hr.len());
        }
        assert!(snr_rs > snr_c - 0.5, "{snr_rs:.2} dB against {snr_c:.2} dB");
    }
    let mut enc = Stream::new(48_000, 1, 40, 48_000).rs_encoder();
    assert!(enc.set_prediction(3).is_err());
    assert!(enc.set_prediction(-1).is_err());
}

/// `OPUS_SET_PACKET_LOSS_PERC` reaches the CELT encoder of libopus. With
/// loss, the pitch pre-filter gets weaker and is off from 9 %, and the
/// coarse energy is coded intra more often (quant_coarse_energy:
/// `intra_bias`). opus1 had no such setting; its CELT encoder always
/// coded as for no loss.
#[test]
fn packet_loss_changes_filter_and_intra_frames() {
    let pcm = speech(48_000, 4);
    for loss in [0, 5, 10, 25] {
        let mut st = Stream::new(48_000, 1, 40, 32_000);
        st.complexity = 9;
        st.loss = loss;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let (hc, hr) = (headers_of(&st, &c), headers_of(&st, &rs));
        let count = |h: &[Header], f: fn(&Header) -> bool| h.iter().filter(|x| f(x)).count();
        let intra = |x: &Header| x.intra;
        let filtered = |x: &Header| x.postfilter.is_some();
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        println!(
            "loss {loss} %: intra {} (libopus {}), filtered {} (libopus {}) of {}; \
             opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            count(&hr, intra),
            count(&hc, intra),
            count(&hr, filtered),
            count(&hc, filtered),
            hr.len()
        );
        let (ic, ir) = (count(&hc, intra) as i32, count(&hr, intra) as i32);
        assert!((ic - ir).abs() <= 2 + ic / 10, "intra frames {ir} against {ic}");
        if loss > 8 {
            assert_eq!(count(&hc, filtered), 0);
            assert_eq!(count(&hr, filtered), 0);
        }
    }
    let mut enc = Stream::new(48_000, 1, 40, 32_000).rs_encoder();
    assert!(enc.set_packet_loss_perc(101).is_err());
    assert!(enc.set_packet_loss_perc(-1).is_err());
}

/// A stereo encoder set to one coded channel codes the mean of its two
/// channels, as libopus does with `CELT_SET_CHANNELS(1)`. The stream may
/// switch between one and two coded channels from frame to frame.
#[test]
fn a_stereo_encoder_codes_a_mono_stream() {
    let pcm = stereo_of(&speech(48_000, 4), &music(48_000, 4, 1), 0.3);
    for (frame_ms_x2, bitrate) in [(40, 32_000), (10, 48_000)] {
        let mut st = Stream::new(48_000, 2, frame_ms_x2, bitrate);
        st.complexity = 9;
        st.stream_channels = 1;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let (hc, hr) = (headers_of(&st, &c), headers_of(&st, &rs));
        let same = |f: fn(&Header) -> String| {
            hc.iter().zip(&hr).filter(|(a, b)| f(a) == f(b)).count()
        };
        let transient = same(|h| format!("{}", h.transient));
        let filtered = same(|h| format!("{}", h.postfilter.is_some()));
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        println!(
            "{} ms: same transient flag in {transient}, same filter decision in {filtered} \
             of {} frames; opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            frame_ms_x2 as f32 / 2.0,
            hc.len()
        );
        assert!(transient * 100 >= hc.len() * 95);
        // The pitch search itself is in src/pitch.rs. With the spectrum
        // flattening of libopus there, the decisions are the same in every
        // frame; without it, in about 60 % of them.
        assert!(filtered * 100 >= hc.len() * 55);
        assert!((snr_rs - snr_c).abs() < 0.5, "{snr_rs:.2} dB against {snr_c:.2} dB");
    }

    // Switching: two channels, then one for a second, then two again.
    let st = Stream::new(48_000, 2, 40, 64_000);
    let channels_of = |k: usize| if (50..100).contains(&k) { 1 } else { 2 };
    let (c, rs) = st.enc_both_with(&pcm, |k, c_enc, rs_enc| {
        c_enc.ctl(CELT_SET_CHANNELS, channels_of(k) as i32);
        rs_enc.set_stream_channels(channels_of(k)).unwrap();
    });
    let decode = |payloads: &[Vec<u8>]| {
        let packets: Vec<Vec<u8>> = payloads
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let mut pkt = vec![celt_toc(21, channels_of(k))];
                pkt.extend_from_slice(p);
                pkt
            })
            .collect();
        dec_c(&packets, 2)
    };
    // The part after the switch back to two channels.
    let snr = |d: &[f32]| snr_db_delay(&pcm, d, 2, 100 * 960, CELT_DELAY);
    let (snr_rs, snr_c) = (snr(&decode(&rs)), snr(&decode(&c)));
    println!("after switching back: opus1 {snr_rs:.2} dB, libopus {snr_c:.2} dB");
    assert!(snr_rs > snr_c - 0.5, "{snr_rs:.2} dB against {snr_c:.2} dB");
    let mut enc = st.rs_encoder();
    assert!(enc.set_stream_channels(0).is_err());
    assert!(enc.set_stream_channels(3).is_err());
    let mut mono = Stream::new(48_000, 1, 40, 64_000).rs_encoder();
    assert!(mono.set_stream_channels(2).is_err());
}

/// After a frame coded as mono, libopus copies its band energies to the
/// history of the second channel (celt_encoder.c 2782-2784). When the
/// stream goes back to two channels, the second channel's coarse energy is
/// predicted from the mono frame, not from the last stereo frame before
/// the switch.
///
/// The signal is 30 dB quieter while the stream is mono, so an older
/// history is far off.
#[test]
fn the_second_channel_continues_from_the_mono_energies() {
    let mut pcm = stereo_of(&music(48_000, 3, 0), &music(48_000, 3, 1), 0.3);
    let frame = 960;
    for (i, s) in pcm.iter_mut().enumerate() {
        if (50 * frame * 2..100 * frame * 2).contains(&i) {
            *s *= 0.03;
        }
    }
    let channels_of = |k: usize| if (50..100).contains(&k) { 1 } else { 2 };
    let mut st = Stream::new(48_000, 2, 40, 96_000);
    st.prediction = 1;
    let (c, rs) = st.enc_both_with(&pcm, |k, c_enc, rs_enc| {
        c_enc.ctl(CELT_SET_CHANNELS, channels_of(k) as i32);
        rs_enc.set_stream_channels(channels_of(k)).unwrap();
    });
    // The first stereo frame after the mono part.
    let (hc, hr) = (parse_header(&c[100], 2, 21, 3), parse_header(&rs[100], 2, 21, 3));
    println!("libopus: intra {}, second channel {:?}", hc.intra, &hc.coarse[21..]);
    println!("opus1: intra {}, second channel {:?}", hr.intra, &hr.coarse[21..]);
    assert_eq!(hr.intra, hc.intra);
    let d: i32 = hc.coarse[21..].iter().zip(&hr.coarse[21..]).map(|(a, b)| (a - b).abs()).sum();
    assert!(d <= 2, "the second channel differs by {d} steps in all");
}
