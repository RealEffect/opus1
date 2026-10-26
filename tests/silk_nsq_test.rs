//! The SILK quantizers `silk_nsq` and `silk_nsq_del_dec` on simple frames
//! (noise, a pulse train, silence; narrowband and wideband), against
//! `silk_NSQ_c` and `silk_NSQ_del_dec_c` of libopus (silk/NSQ.c,
//! silk/NSQ_del_dec.c).
//!
//! The C functions take the encoder state, whose layout depends on how
//! libopus was built, so they are not called in the float build the tests
//! link. Each test calls them in the fixed-point build of the official
//! libopus 1.6.1 release that the dev-dependency `libopus-fixed` compiles
//! (tools/libopus-fixed), from the same state and on the same frame. The
//! quantizers are fixed-point: the pulses, the whole quantizer state after
//! the frame and the seed must be identical.

mod silk_nsq_common;

use opus1::internals::silk::define::*;
use opus1::internals::silk::nsq::silk_nsq;
use opus1::internals::silk::nsq_del_dec::silk_nsq_del_dec;
use opus1::internals::silk::structs::*;

/// A 16 kHz wideband encoder state.
fn create_wb_encoder_state() -> SilkEncoderStateCommon {
    SilkEncoderStateCommon {
        fs_khz: 16,
        nb_subfr: 4,
        subfr_length: 80,    // 5ms * 16kHz
        frame_length: 320,   // 20ms * 16kHz
        ltp_mem_length: 320, // PE_LTP_MEM_LENGTH_MS * fs_khz = 20 * 16
        predict_lpc_order: 16,
        shaping_lpc_order: 16,
        first_frame_after_reset: 1,
        n_states_delayed_decision: 1,
        indices: SideInfoIndices {
            signal_type: TYPE_UNVOICED as i8,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// An 8 kHz narrowband encoder state.
fn create_nb_encoder_state() -> SilkEncoderStateCommon {
    SilkEncoderStateCommon {
        fs_khz: 8,
        nb_subfr: 4,
        subfr_length: 40,    // 5ms * 8kHz
        frame_length: 160,   // 20ms * 8kHz
        ltp_mem_length: 160, // PE_LTP_MEM_LENGTH_MS * fs_khz = 20 * 8
        predict_lpc_order: 10,
        shaping_lpc_order: 16,
        first_frame_after_reset: 1,
        n_states_delayed_decision: 1,
        indices: SideInfoIndices {
            signal_type: TYPE_UNVOICED as i8,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Noise-like (unvoiced) input.
fn generate_unvoiced_input(length: usize) -> Vec<i16> {
    let mut rng: u32 = 12345;
    (0..length)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng >> 16) as i16) >> 2 // small amplitude noise
        })
        .collect()
}

/// A pulse train (voiced input) with period `pitch`.
fn generate_voiced_input(length: usize, pitch: usize) -> Vec<i16> {
    (0..length)
        .map(|i| {
            let phase = (i % pitch) as f32 / pitch as f32;
            if phase < 0.1 { 3000 } else { -200 }
        })
        .collect()
}

/// The parameters of one frame.
struct Frame {
    pred_coef_q12: Vec<i16>,
    ltp_coef_q14: Vec<i16>,
    ar_q13: Vec<i16>,
    harm_shape_gain_q14: Vec<i32>,
    tilt_q14: Vec<i32>,
    lf_shp_q14: Vec<i32>,
    gains_q16: Vec<i32>,
    pitch_l: Vec<i32>,
    lambda_q10: i32,
    ltp_scale_q14: i32,
}

impl Frame {
    /// Unvoiced: a first-order predictor (0.9, -0.45), a mild shaping tilt,
    /// no long-term prediction, unit gains.
    fn unvoiced(nb_subfr: usize) -> Frame {
        let mut pred_coef_q12 = vec![0i16; 2 * MAX_LPC_ORDER];
        for set in 0..2 {
            pred_coef_q12[set * MAX_LPC_ORDER] = 3686; // 0.9 in Q12
            pred_coef_q12[set * MAX_LPC_ORDER + 1] = -1843; // -0.45 in Q12
        }
        let mut ar_q13 = vec![0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
        for k in 0..nb_subfr {
            ar_q13[k * MAX_SHAPE_LPC_ORDER] = 4096; // 0.5 in Q13
            ar_q13[k * MAX_SHAPE_LPC_ORDER + 1] = 2048;
        }
        Frame {
            pred_coef_q12,
            ltp_coef_q14: vec![0; MAX_NB_SUBFR * LTP_ORDER],
            ar_q13,
            harm_shape_gain_q14: vec![0; MAX_NB_SUBFR],
            tilt_q14: vec![0; MAX_NB_SUBFR],
            lf_shp_q14: vec![0; MAX_NB_SUBFR],
            gains_q16: vec![65536; MAX_NB_SUBFR], // 1.0 in Q16
            pitch_l: vec![0; MAX_NB_SUBFR],
            lambda_q10: 1024, // 1.0 in Q10
            ltp_scale_q14: 16384,
        }
    }

    /// Voiced: as `unvoiced`, with the pitch lag `pitch`, a long-term
    /// predictor of 0.5 in the centre tap and mild harmonic shaping.
    fn voiced(nb_subfr: usize, pitch: i32) -> Frame {
        let mut f = Frame::unvoiced(nb_subfr);
        for k in 0..nb_subfr {
            f.ltp_coef_q14[k * LTP_ORDER + 2] = 8192; // 0.5 in Q14
        }
        f.harm_shape_gain_q14 = vec![4096; MAX_NB_SUBFR];
        f.pitch_l = vec![pitch; MAX_NB_SUBFR];
        f
    }
}

/// A fresh quantizer state with a previous gain of 1.0.
fn fresh_state(signal_type: i32) -> SilkNSQState {
    SilkNSQState {
        prev_gain_q16: 65536,
        prev_sig_type: signal_type as i8,
        ..Default::default()
    }
}

/// Quantizes `input` with opus1 (delayed decision when `del_dec`) and
/// with libopus from `nsq`, checks that both give the same pulses, state
/// and seed, and returns opus1's pulses and state.
fn quantize(
    s_cmn: &SilkEncoderStateCommon,
    nsq: &SilkNSQState,
    input: &[i16],
    f: &Frame,
    del_dec: bool,
) -> (Vec<i8>, SilkNSQState) {
    let n = s_cmn.frame_length as usize;
    let mut c_state = silk_nsq_common::state_of(nsq);
    let mut c_frame = silk_nsq_common::frame_of(s_cmn, &s_cmn.indices);
    let mut c_pulses = vec![0i8; n];
    let params = libopus_fixed::NsqParams {
        pred_coef_q12: &f.pred_coef_q12,
        ltp_coef_q14: &f.ltp_coef_q14,
        ar_q13: &f.ar_q13,
        harm_shape_gain_q14: &f.harm_shape_gain_q14,
        tilt_q14: &f.tilt_q14,
        lf_shp_q14: &f.lf_shp_q14,
        gains_q16: &f.gains_q16,
        pitch_l: &f.pitch_l,
        lambda_q10: f.lambda_q10,
        ltp_scale_q14: f.ltp_scale_q14,
    };
    libopus_fixed::nsq(del_dec, &mut c_frame, &mut c_state, input, &mut c_pulses, &params);

    let mut nsq = *nsq;
    let mut pulses = vec![0i8; n];
    let (pred, ltp, ar) = (&f.pred_coef_q12, &f.ltp_coef_q14, &f.ar_q13);
    let (harm, tilt, lf, gains, pitch) = (&f.harm_shape_gain_q14, &f.tilt_q14, &f.lf_shp_q14, &f.gains_q16, &f.pitch_l);
    let ind = &s_cmn.indices;
    // The delayed-decision quantizer returns the seed of the path it chose
    // (libopus writes it to the side information); the plain one keeps it.
    let seed = if del_dec {
        silk_nsq_del_dec(
            s_cmn, &mut nsq, ind, input, &mut pulses, pred, ltp, ar, harm, tilt, lf, gains, pitch, f.lambda_q10,
            f.ltp_scale_q14,
        )
    } else {
        silk_nsq(
            s_cmn, &mut nsq, ind, input, &mut pulses, pred, ltp, ar, harm, tilt, lf, gains, pitch, f.lambda_q10,
            f.ltp_scale_q14,
        );
        ind.seed as i32
    };

    assert_eq!(pulses, c_pulses, "pulses");
    assert_eq!(silk_nsq_common::differences(&nsq, &c_state), Vec::<&str>::new(), "state");
    assert_eq!(seed, c_frame.seed, "seed");
    (pulses, nsq)
}

fn non_zero(pulses: &[i8]) -> usize {
    pulses.iter().filter(|&&p| p != 0).count()
}

fn energy(pulses: &[i8]) -> i64 {
    pulses.iter().map(|&p| p as i64 * p as i64).sum()
}

#[test]
fn test_nsq_unvoiced_basic() {
    let s_cmn = create_wb_encoder_state();
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    let (pulses, nsq) = quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &Frame::unvoiced(4), false);
    assert!(non_zero(&pulses) > 0, "non-silent input gives pulses");
    assert_ne!(nsq.s_ltp_shp_buf_idx, 0, "the shaping buffer index moves on");
}

#[test]
fn test_nsq_voiced_basic() {
    let mut s_cmn = create_wb_encoder_state();
    s_cmn.indices.signal_type = TYPE_VOICED as i8;
    s_cmn.first_frame_after_reset = 0;
    let input = generate_voiced_input(s_cmn.frame_length as usize, 100);
    let (pulses, nsq) = quantize(&s_cmn, &fresh_state(TYPE_VOICED), &input, &Frame::voiced(4, 100), false);
    assert!(non_zero(&pulses) > 0, "voiced input gives pulses");
    assert_eq!(nsq.lag_prev, 100);
}

#[test]
fn test_nsq_del_dec_basic() {
    let mut s_cmn = create_wb_encoder_state();
    s_cmn.n_states_delayed_decision = 2;
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    let (pulses, _) = quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &Frame::unvoiced(4), true);
    assert!(non_zero(&pulses) > 0, "non-silent input gives pulses");
}

/// The delayed-decision quantizer with 1 to 4 states, unvoiced and voiced,
/// in both bandwidths, with a seed that is not 0.
#[test]
fn del_dec_matches_libopus_for_every_number_of_states() {
    for wideband in [false, true] {
        for voiced in [false, true] {
            for states in 1..=4 {
                let mut s_cmn = if wideband { create_wb_encoder_state() } else { create_nb_encoder_state() };
                s_cmn.n_states_delayed_decision = states;
                s_cmn.indices.seed = 3;
                let n = s_cmn.frame_length as usize;
                let (input, f, signal_type) = if voiced {
                    s_cmn.first_frame_after_reset = 0;
                    let pitch = if wideband { 100 } else { 50 };
                    (generate_voiced_input(n, pitch), Frame::voiced(4, pitch as i32), TYPE_VOICED)
                } else {
                    (generate_unvoiced_input(n), Frame::unvoiced(4), TYPE_UNVOICED)
                };
                s_cmn.indices.signal_type = signal_type as i8;
                quantize(&s_cmn, &fresh_state(signal_type), &input, &f, true);
            }
        }
    }
}

#[test]
fn test_nsq_silent_input() {
    let s_cmn = create_wb_encoder_state();
    let input = vec![0i16; s_cmn.frame_length as usize];
    // Silence may still give small pulses from the dither.
    quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &Frame::unvoiced(4), false);
}

#[test]
fn test_nsq_gain_scaling() {
    let s_cmn = create_wb_encoder_state();
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    let mut low = Frame::unvoiced(4);
    low.gains_q16 = vec![32768; MAX_NB_SUBFR]; // 0.5 in Q16
    let mut high = Frame::unvoiced(4);
    high.gains_q16 = vec![131072; MAX_NB_SUBFR]; // 2.0 in Q16
    let (pulses_low, _) = quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &low, false);
    let (pulses_high, _) = quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &high, false);
    assert!(energy(&pulses_low) > energy(&pulses_high), "a larger step gives smaller pulses");
}

#[test]
fn unvoiced_wb_matches_libopus() {
    let s_cmn = create_wb_encoder_state();
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &Frame::unvoiced(4), false);
}

#[test]
fn voiced_wb_matches_libopus() {
    let mut s_cmn = create_wb_encoder_state();
    s_cmn.indices.signal_type = TYPE_VOICED as i8;
    s_cmn.first_frame_after_reset = 0;
    let input = generate_voiced_input(s_cmn.frame_length as usize, 100);
    quantize(&s_cmn, &fresh_state(TYPE_VOICED), &input, &Frame::voiced(4, 100), false);
}

#[test]
fn unvoiced_nb_matches_libopus() {
    let s_cmn = create_nb_encoder_state();
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    quantize(&s_cmn, &fresh_state(TYPE_UNVOICED), &input, &Frame::unvoiced(4), false);
}

#[test]
fn voiced_nb_matches_libopus() {
    let mut s_cmn = create_nb_encoder_state();
    s_cmn.indices.signal_type = TYPE_VOICED as i8;
    s_cmn.first_frame_after_reset = 0;
    let input = generate_voiced_input(s_cmn.frame_length as usize, 50);
    quantize(&s_cmn, &fresh_state(TYPE_VOICED), &input, &Frame::voiced(4, 50), false);
}
