//! `silk_setup_complexity` against the table in libopus 1.6.1
//! silk/control_codec.c (lines 316 to 391).
//!
//! The expected numbers are the ones the C source gives. Its fixed-point
//! constants are written with the macro
//! `SILK_FIX_CONST( C, Q ) = (opus_int32)( C * ( 1 << Q ) + 0.5 )`
//! (silk/SigProc_FIX.h), which rounds to the nearest integer. Compiled from
//! the 1.6.1 source, the macro gives 52429 for 0.8, 49807 for 0.76, 48497
//! for 0.74, 47186 for 0.72 and 45875 for 0.7 in Q16, and 983 for the
//! warping multiplier 0.015.

use opus_rs::silk::control_codec::{silk_control_encoder, silk_setup_lbrr};
use opus_rs::silk::structs::SilkEncoderState;

/// The gain increase of the low-bitrate redundancy (LBRR, the in-band FEC)
/// frames, against libopus 1.6.1 silk/control_codec.c, silk_setup_LBRR,
/// lines 414 to 419:
///
/// ```c
/// if( LBRR_in_previous_packet == 0 ) {
///     psEncC->LBRR_GainIncreases = 7;
/// } else {
///     psEncC->LBRR_GainIncreases = silk_max_int( 7 - silk_SMULWB( (opus_int32)psEncC->PacketLoss_perc, SILK_FIX_CONST( 0.2, 16 ) ), 3 );
/// }
/// ```
///
/// `SILK_FIX_CONST( 0.2, 16 )` is 13107. The expected values below were
/// printed by these C lines for every loss percentage: 7 up to 5 %, 6 up to
/// 10 %, 5 up to 15 %, 4 up to 20 %, and 3 from 21 %.
#[test]
fn lbrr_gain_increase_follows_the_packet_loss_as_in_libopus() {
    for loss in 0..=100 {
        let mut enc = Box::new(SilkEncoderState::default());
        enc.s_cmn.packet_loss_perc = loss;
        // The first packet with LBRR: the packet before had none.
        silk_setup_lbrr(&mut enc, 1);
        assert_eq!(enc.s_cmn.lbrr_enabled, 1);
        assert_eq!(enc.s_cmn.lbrr_gain_increases, 7, "loss {loss} %, first packet with LBRR");
        // The packets after it.
        silk_setup_lbrr(&mut enc, 1);
        let c_value = match loss {
            0..=5 => 7,
            6..=10 => 6,
            11..=15 => 5,
            16..=20 => 4,
            _ => 3,
        };
        assert_eq!(enc.s_cmn.lbrr_gain_increases, c_value, "loss {loss} %, later packets");
        // Without LBRR the value is left alone, and the next packet with
        // LBRR starts from 7 again.
        silk_setup_lbrr(&mut enc, 0);
        assert_eq!((enc.s_cmn.lbrr_enabled, enc.s_cmn.lbrr_gain_increases), (0, c_value));
        silk_setup_lbrr(&mut enc, 1);
        assert_eq!(enc.s_cmn.lbrr_gain_increases, 7, "loss {loss} %, LBRR switched on again");
    }
}

/// One row of the C table: what `silk_setup_complexity` sets for a range of
/// complexity values.
struct Row {
    /// libopus `pitchEstimationComplexity`.
    pitch_complexity: i32,
    /// libopus `pitchEstimationThreshold_Q16`.
    pitch_threshold_q16: i32,
    /// libopus `pitchEstimationLPCOrder`, before the limit to the prediction
    /// order.
    pitch_lpc_order: i32,
    /// libopus `shapingLPCOrder`.
    shaping_lpc_order: i32,
    /// libopus `la_shape`, in milliseconds.
    la_shape_ms: i32,
    /// libopus `nStatesDelayedDecision`.
    states: i32,
    /// libopus `useInterpolatedNLSFs`.
    interpolated_nlsfs: i32,
    /// libopus `NLSF_MSVQ_Survivors`.
    nlsf_survivors: i32,
    /// libopus `warping_Q16` per kHz of the internal rate.
    warping_q16_per_khz: i32,
}

/// The C table for the complexity values 0 to 10.
fn c_row(complexity: i32) -> Row {
    let row = |pitch_complexity, pitch_threshold_q16, pitch_lpc_order, shaping_lpc_order, la_shape_ms, states, interpolated_nlsfs, nlsf_survivors, warping_q16_per_khz| Row {
        pitch_complexity,
        pitch_threshold_q16,
        pitch_lpc_order,
        shaping_lpc_order,
        la_shape_ms,
        states,
        interpolated_nlsfs,
        nlsf_survivors,
        warping_q16_per_khz,
    };
    match complexity {
        0 => row(0, 52429, 6, 12, 3, 1, 0, 2, 0),
        1 => row(1, 49807, 8, 14, 5, 1, 0, 3, 0),
        2 => row(0, 52429, 6, 12, 3, 2, 0, 2, 0),
        3 => row(1, 49807, 8, 14, 5, 2, 0, 4, 0),
        4 | 5 => row(1, 48497, 10, 16, 5, 2, 1, 6, 983),
        6 | 7 => row(1, 47186, 12, 20, 5, 3, 1, 8, 983),
        _ => row(2, 45875, 16, 24, 5, 4, 1, 16, 983),
    }
}

#[test]
fn setup_complexity_gives_the_values_of_libopus() {
    // (internal rate in kHz, prediction order at that rate)
    for (fs_khz, predict_order) in [(8, 10), (12, 10), (16, 16)] {
        for complexity in 0..=10 {
            let mut enc = Box::new(SilkEncoderState::default());
            silk_control_encoder(&mut enc, fs_khz, 20, 20000, complexity);
            let c = c_row(complexity);
            let what = format!("{fs_khz} kHz, complexity {complexity}");
            let cmn = &enc.s_cmn;
            assert_eq!(cmn.pitch_estimation_complexity, c.pitch_complexity, "{what}: pitch estimation complexity");
            assert_eq!(cmn.pitch_estimation_threshold_q16, c.pitch_threshold_q16, "{what}: pitch estimation threshold");
            assert_eq!(enc.pitch_estimation_lpc_order, c.pitch_lpc_order.min(predict_order), "{what}: pitch estimation LPC order");
            assert_eq!(cmn.shaping_lpc_order, c.shaping_lpc_order, "{what}: shaping LPC order");
            assert_eq!(cmn.la_shape, c.la_shape_ms * fs_khz, "{what}: shaping look-ahead");
            assert_eq!(cmn.n_states_delayed_decision, c.states, "{what}: delayed-decision states");
            assert_eq!(cmn.use_interpolated_nlsfs, c.interpolated_nlsfs, "{what}: interpolated NLSFs");
            assert_eq!(cmn.n_nlsf_survivors, c.nlsf_survivors, "{what}: NLSF survivors");
            assert_eq!(cmn.warping_q16, c.warping_q16_per_khz * fs_khz, "{what}: warping");
            // C 390: SUB_FRAME_LENGTH_MS * fs_kHz + 2 * la_shape.
            assert_eq!(cmn.shape_win_length, 5 * fs_khz + 2 * c.la_shape_ms * fs_khz, "{what}: shaping window");
            assert_eq!(cmn.complexity, complexity, "{what}: complexity");
        }
    }
}
