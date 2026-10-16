//! `silk_nsq`, the quantizer of complexity 0 and 1, against libopus
//! `silk_NSQ_c` (silk/NSQ.c).
//!
//! The C function takes the encoder state, whose layout depends on how
//! libopus was built, so it is not called through the linked library. The
//! reference values in `perfk_silk_nsq_plain.expected` were computed once by
//! the C function itself (the libopus 1.6.1 source, compiled) on the frames
//! that `Generator` below produces: 60 streams of 20 frames, the quantizer
//! state carried from frame to frame. The generator uses integers only, so
//! the C program and this test produce the same frames; each row carries a
//! checksum of its input signal to make sure of that.
//!
//! The quantizer is fixed-point: the pulses and the whole state after every
//! frame must be identical.

use opus1::silk::nsq::silk_nsq;
use opus1::silk::structs::{SideInfoIndices, SilkEncoderStateCommon, SilkNSQState};

const MAX_LPC_ORDER: usize = 16;
const MAX_SHAPE_LPC_ORDER: usize = 24;
const LTP_ORDER: usize = 5;
const MAX_NB_SUBFR: usize = 4;
const STREAMS: usize = 60;
const FRAMES: usize = 20;

/// (checksum of the input signal, checksum of pulses and state, sum of the
/// absolute pulses, signal type).
type Expected = (u64, u64, i32, i32);

static EXPECTED: [Expected; STREAMS * FRAMES] = include!("perfk_silk_nsq_plain.expected");

/// The generator of the C program that produced `EXPECTED`.
struct Generator(u32);

impl Generator {
    fn rnd(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }

    /// `low + rnd() % span`.
    fn range(&mut self, low: i32, span: u32) -> i32 {
        low + (self.rnd() % span) as i32
    }
}

struct Checksum(u64);

impl Checksum {
    fn add(&mut self, v: i64) {
        self.0 = self.0.wrapping_mul(1000003).wrapping_add(v as u64);
    }
}

#[test]
fn plain_quantizer_matches_libopus() {
    let mut g = Generator(0x600d_cafe);
    let mut failures = Vec::new();
    let mut unvoiced_with_lag_and_shaping = 0;
    'streams: for stream in 0..STREAMS {
        let fs_khz = [8, 12, 16][(g.rnd() % 3) as usize];
        let nb_subfr = if g.rnd().is_multiple_of(4) { 2usize } else { 4 };
        let pred_order = if fs_khz == 16 { 16usize } else { 10 };
        let shp_order = [12usize, 14, 16][(g.rnd() % 3) as usize];
        let mut enc = Box::new(SilkEncoderStateCommon::default());
        enc.fs_khz = fs_khz;
        enc.nb_subfr = nb_subfr as i32;
        enc.subfr_length = 5 * fs_khz;
        enc.frame_length = enc.subfr_length * nb_subfr as i32;
        enc.ltp_mem_length = 20 * fs_khz;
        enc.predict_lpc_order = pred_order as i32;
        enc.shaping_lpc_order = shp_order as i32;
        enc.n_states_delayed_decision = 1;
        let frame_length = enc.frame_length as usize;
        let ltp_mem_length = enc.ltp_mem_length as usize;
        let (min_lag, max_lag) = (2 * fs_khz, 18 * fs_khz);
        let mut nsq = Box::new(SilkNSQState::default());
        nsq.prev_gain_q16 = 65536;

        for frame in 0..FRAMES {
            let expected = &EXPECTED[stream * FRAMES + frame];
            let mut ind = SideInfoIndices::default();
            let kind = g.rnd() % 5;
            let voiced = kind < 2;
            ind.signal_type = if voiced { 2 } else if kind < 4 { 1 } else { 0 };
            ind.quant_offset_type = (g.rnd() % 2) as i8;
            ind.nlsf_interp_coef_q2 = if g.rnd().is_multiple_of(4) { (g.rnd() % 4) as i8 } else { 4 };
            ind.seed = (frame & 3) as i8;
            let amp_shift = g.rnd() % 6;
            let noise_shift = 3 + g.rnd() % 8;
            let base_lag = if voiced { g.range(min_lag, (max_lag - min_lag - 8) as u32) } else { 0 };
            let period = if voiced { base_lag as i64 } else { g.range(40, 100) as i64 };

            // The signal: noise, plus a pulse train when voiced, through a
            // resonator.
            let (mut y1, mut y2, mut phase) = (0i64, 0i64, 0i64);
            let mut x16 = [0i16; 320];
            let mut input_checksum = 0u64;
            for x in x16[..frame_length].iter_mut() {
                let noise = (g.rnd() & 0xffff) as i64 - 32768;
                let mut e = noise >> noise_shift;
                phase += 1;
                if phase >= period {
                    phase = 0;
                    if voiced {
                        e += 12000;
                    }
                }
                let y = (e + ((22000 * y1) >> 14) + ((-9000 * y2) >> 14)).clamp(-30000, 30000);
                y2 = y1;
                y1 = y;
                *x = (y >> amp_shift) as i16;
                input_checksum = input_checksum.wrapping_mul(31).wrapping_add(*x as i64 as u64);
            }
            assert_eq!(
                input_checksum & 0xffff_ffff_ffff,
                expected.0,
                "stream {stream}, frame {frame}: the input differs from the C program's"
            );

            // The parameters of the frame, in the ranges the encoder gives
            // them. Both filters are stable.
            let mut pred = [0i16; 2 * MAX_LPC_ORDER];
            for k in 0..2 {
                pred[k * MAX_LPC_ORDER] = g.range(800, 601) as i16;
                for i in 1..pred_order {
                    pred[k * MAX_LPC_ORDER + i] = g.range(-150, 301) as i16;
                }
            }
            let mut ltp = [0i16; MAX_NB_SUBFR * LTP_ORDER];
            let mut ar = [0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
            let mut harm = [0i32; MAX_NB_SUBFR];
            let mut tilt = [0i32; MAX_NB_SUBFR];
            let mut lf = [0i32; MAX_NB_SUBFR];
            let mut gains = [65536i32; MAX_NB_SUBFR];
            let mut pitch = [0i32; MAX_NB_SUBFR];
            for k in 0..nb_subfr {
                if voiced {
                    ltp[k * LTP_ORDER] = g.range(-800, 1601) as i16;
                    ltp[k * LTP_ORDER + 1] = g.range(-1500, 3001) as i16;
                    ltp[k * LTP_ORDER + 2] = g.range(4000, 5001) as i16;
                    ltp[k * LTP_ORDER + 3] = g.range(-1500, 3001) as i16;
                    ltp[k * LTP_ORDER + 4] = g.range(-800, 1601) as i16;
                }
                ar[k * MAX_SHAPE_LPC_ORDER] = g.range(2000, 4001) as i16;
                for i in 1..shp_order {
                    ar[k * MAX_SHAPE_LPC_ORDER + i] = g.range(-80, 161) as i16;
                }
                // Unvoiced frames keep a shaping gain too: in the encoder it
                // is the smoothed gain, which is not zero after a voiced
                // frame.
                harm[k] = if g.rnd().is_multiple_of(3) {
                    0
                } else {
                    (g.rnd() % if voiced { 8001 } else { 3001 }) as i32
                };
                tilt[k] = -g.range(2000, 7001);
                let b = g.range(200, 3000);
                let hi = 16384 - b - (g.rnd() % 500) as i32;
                lf[k] = (((hi as u32) << 16) | ((b - 16384) as u32 & 0xffff)) as i32;
                let mut gain = 1u32 << (16 + g.rnd() % 9);
                gain += g.rnd() % 65536;
                gains[k] = gain as i32;
                pitch[k] = if voiced { base_lag + (g.rnd() % 5) as i32 } else { 0 };
            }
            let lambda = if g.rnd().is_multiple_of(5) { g.range(2100, 600) } else { g.range(600, 800) };
            let ltp_scale = if voiced { [15565, 12288, 8192][(g.rnd() % 3) as usize] } else { 0 };
            if !voiced && nsq.lag_prev > 0 && harm[..nb_subfr].iter().any(|&h| h > 3) {
                unvoiced_with_lag_and_shaping += 1;
            }

            let mut pulses = [0i8; 320];
            silk_nsq(
                &enc, &mut nsq, &ind, &x16, &mut pulses, &pred, &ltp, &ar, &harm, &tilt, &lf, &gains, &pitch, lambda,
                ltp_scale,
            );

            let mut h = Checksum(0);
            let mut sum_abs = 0i32;
            for &p in &pulses[..frame_length] {
                h.add(p as i64);
                sum_abs += (p as i32).abs();
            }
            nsq.xq[..ltp_mem_length].iter().for_each(|&v| h.add(v as i64));
            nsq.s_ltp_shp_q14[..ltp_mem_length].iter().for_each(|&v| h.add(v as i64));
            nsq.s_lpc_q14[..16].iter().for_each(|&v| h.add(v as i64));
            nsq.s_ar2_q14.iter().for_each(|&v| h.add(v as i64));
            for v in [
                nsq.s_lf_ar_q14,
                nsq.s_diff_shp_q14,
                nsq.lag_prev,
                nsq.s_ltp_buf_idx,
                nsq.s_ltp_shp_buf_idx,
                nsq.rand_seed,
                nsq.prev_gain_q16,
                nsq.rewhite_flag,
            ] {
                h.add(v as i64);
            }
            let got: Expected = (expected.0, h.0 & 0xffff_ffff_ffff, sum_abs, ind.signal_type as i32);
            if got != *expected {
                failures.push(format!(
                    "stream {stream} ({fs_khz} kHz, {nb_subfr} subframes, shaping order {shp_order}), frame {frame}: opus1 {:?}, libopus {:?}",
                    got, expected
                ));
                // The state has left libopus's and the generator would get
                // out of step: stop at the first difference.
                break 'streams;
            }
        }
    }
    assert!(failures.is_empty(), "first difference from libopus: {}", failures.join("\n"));
    // The case libopus handles and opus1 did not: an unvoiced frame after
    // a voiced one, with harmonic shaping still on.
    assert!(unvoiced_with_lag_and_shaping > 50, "{unvoiced_with_lag_and_shaping}");
}
