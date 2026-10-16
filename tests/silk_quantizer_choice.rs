//! Which noise shaping quantizer `silk_encode_frame` runs.
//!
//! libopus 1.6.1 silk/fixed/encode_frame_FIX.c, line 207:
//!
//! ```c
//! if( psEnc->sCmn.nStatesDelayedDecision > 1 || psEnc->sCmn.warping_Q16 > 0 ) {
//!     silk_NSQ_del_dec( ... );
//! } else {
//!     silk_NSQ( ... );
//! }
//! ```
//!
//! The plain quantizer has no warped noise shaping, so a frame analysed with
//! warping must go to the delayed-decision quantizer even with one state.
//! The complexity table of libopus never gives one state together with
//! warping, so the standard settings do not show the second half of the
//! condition. The test sets that combination by hand.
//!
//! How the test tells the two apart: only the plain quantizer keeps its
//! random seed in the quantizer state (NSQ.c lines 105, 229 and 361 write
//! `NSQ->rand_seed`; NSQ_del_dec.c never does). The state starts with a seed
//! of 0, so after one frame the seed is still 0 exactly when the
//! delayed-decision quantizer ran.

use opus1::range_coder::RangeCoder;
use opus1::silk::control_codec::silk_control_encoder;
use opus1::silk::define::CODE_INDEPENDENTLY;
use opus1::silk::enc_api::silk_encode_frame;
use opus1::silk::init_encoder::silk_init_encoder;
use opus1::silk::structs::SilkEncoderState;

/// Encodes one 20 ms frame of a 200 Hz tone at 16 kHz with the given number
/// of delayed-decision states and warping, and returns the random seed left
/// in the quantizer state.
fn seed_after_one_frame(complexity: i32, states: i32, warping_q16: i32) -> i32 {
    let mut enc = Box::new(SilkEncoderState::default());
    silk_init_encoder(&mut enc, 0);
    silk_control_encoder(&mut enc, 16, 20, 20000, complexity);
    enc.s_cmn.snr_db_q7 = 25 * 128;
    enc.s_cmn.n_states_delayed_decision = states;
    enc.s_cmn.warping_q16 = warping_q16;
    assert_eq!(enc.s_nsq.rand_seed, 0, "the quantizer state starts with seed 0");

    let input: Vec<i16> = (0..enc.s_cmn.frame_length as usize)
        .map(|i| (10000.0 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin()) as i16)
        .collect();
    let mut rc = RangeCoder::new_encoder(1275);
    let mut n_bytes_out = 0;
    let ret = silk_encode_frame(&mut enc, &input, &mut rc, &mut n_bytes_out, CODE_INDEPENDENTLY, 8000, 0);
    assert_eq!(ret, 0);
    assert!(enc.pulses.iter().any(|&p| p != 0), "the frame has pulses");
    enc.s_nsq.rand_seed
}

#[test]
fn one_state_without_warping_runs_the_plain_quantizer() {
    // Complexity 0: one state, no warping.
    assert_ne!(seed_after_one_frame(0, 1, 0), 0);
}

#[test]
fn several_states_run_the_delayed_decision_quantizer() {
    // Complexity 2 has two states and no warping, complexity 5 two states
    // and warping.
    assert_eq!(seed_after_one_frame(2, 2, 0), 0);
    assert_eq!(seed_after_one_frame(5, 2, 16 * 983), 0);
}

#[test]
fn one_state_with_warping_runs_the_delayed_decision_quantizer() {
    // The analysis settings of complexity 5 (warping on), with one state.
    assert_eq!(seed_after_one_frame(5, 1, 16 * 983), 0);
}
