use crate::range_coder::RangeCoder;
use crate::silk::define::*;
use crate::silk::shell_coder::silk_shell_encoder;
use crate::silk::tables::*;

const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
const _: () = assert!(1 << LOG2_SHELL_CODEC_FRAME_LENGTH == SHELL_CODEC_FRAME_LENGTH);

fn combine_and_check(
    pulses_comb: &mut [i32],
    pulses_in: &[i32],
    max_pulses: i32,
    len: usize,
) -> i32 {
    for k in 0..len {
        let sum = pulses_in[2 * k] + pulses_in[2 * k + 1];
        if sum > max_pulses {
            return 1;
        }
        pulses_comb[k] = sum;
    }
    0
}

/// Code the excitation `pulses[..frame_length]` of one frame. Port of
/// `silk_encode_pulses` (silk/encode_pulses.c).
///
/// The frame is coded in shell blocks of 16 samples. A frame of 120 samples
/// (10 ms at 12 kHz) fills 7.5 blocks: the last block is padded with zeros.
/// As in libopus, the padding is written into `pulses`, which must therefore
/// be longer than the frame; the 16 samples after the frame are cleared.
pub fn silk_encode_pulses(
    ps_range_enc: &mut RangeCoder,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &mut [i8],
    frame_length: usize,
) {
    let mut pulses_comb = [0i32; 8];

    let mut abs_pulses = [0i32; MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH];

    // encode_pulses.c:86-91: "Calculate number of shell blocks". A partial
    // last block only happens for 10 ms at 12 kHz. The loops for the least
    // significant bits and the signs read whole blocks of `pulses`, so the
    // tail of the last block must be zero there too.
    let frame_length = frame_length.min(pulses.len()).min(MAX_FRAME_LENGTH);
    let mut iter = frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        iter += 1;
        let pad_end = (frame_length + SHELL_CODEC_FRAME_LENGTH).min(pulses.len());
        pulses[frame_length..pad_end].fill(0);
    }
    // The samples of all blocks. They lie within `pulses`, except when the
    // caller's buffer ends inside the last block.
    let coded_length = (iter * SHELL_CODEC_FRAME_LENGTH).min(pulses.len());
    let pulses = &pulses[..coded_length];

    // encode_pulses.c:93-101: "Take the absolute value of the pulses".
    for (abs, &p) in abs_pulses.iter_mut().zip(pulses) {
        *abs = (p as i32).abs();
    }

    let mut sum_pulses = [0i32; MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH];
    let mut n_rshifts = [0i32; MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH];

    for i in 0..iter {
        let abs_pulses_ptr = &mut abs_pulses[i * SHELL_CODEC_FRAME_LENGTH..];
        n_rshifts[i] = 0;

        loop {
            let mut scale_down = combine_and_check(
                &mut pulses_comb,
                abs_pulses_ptr,
                SILK_MAX_PULSES_TABLE[0] as i32,
                8,
            );

            let pulses_temp8 = pulses_comb;
            scale_down += combine_and_check(
                &mut pulses_comb,
                &pulses_temp8,
                SILK_MAX_PULSES_TABLE[1] as i32,
                4,
            );

            let pulses_temp4 = pulses_comb;
            scale_down += combine_and_check(
                &mut pulses_comb,
                &pulses_temp4,
                SILK_MAX_PULSES_TABLE[2] as i32,
                2,
            );

            let pulses_temp2 = pulses_comb;
            let mut sum_pulse_val = [sum_pulses[i]];
            scale_down += combine_and_check(
                &mut sum_pulse_val,
                &pulses_temp2,
                SILK_MAX_PULSES_TABLE[3] as i32,
                1,
            );
            sum_pulses[i] = sum_pulse_val[0];

            if scale_down > 0 {
                n_rshifts[i] += 1;
                for k in 0..SHELL_CODEC_FRAME_LENGTH {
                    abs_pulses_ptr[k] >>= 1;
                }
            } else {
                break;
            }
        }
    }

    let mut min_sum_bits_q5 = i32::MAX;
    let mut rate_level_index = 0;
    for k in 0..N_RATE_LEVELS - 1 {
        let n_bits_ptr = &SILK_PULSES_PER_BLOCK_BITS_Q5[k];
        let mut sum_bits_q5 = SILK_RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize][k] as i32;
        for i in 0..iter {
            if n_rshifts[i] > 0 {
                sum_bits_q5 += n_bits_ptr[SILK_MAX_PULSES + 1] as i32;
            } else {
                sum_bits_q5 += n_bits_ptr[sum_pulses[i] as usize] as i32;
            }
        }
        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = k;
        }
    }
    ps_range_enc.encode_icdf(
        rate_level_index as i32,
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    );

    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        if n_rshifts[i] == 0 {
            ps_range_enc.encode_icdf(sum_pulses[i], cdf_ptr, 8);
        } else {
            ps_range_enc.encode_icdf(SILK_MAX_PULSES as i32 + 1, cdf_ptr, 8);
            for _ in 0..n_rshifts[i] - 1 {
                ps_range_enc.encode_icdf(
                    SILK_MAX_PULSES as i32 + 1,
                    &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                    8,
                );
            }
            ps_range_enc.encode_icdf(
                sum_pulses[i],
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                8,
            );
        }
    }

    for i in 0..iter {
        if sum_pulses[i] > 0 {
            silk_shell_encoder(
                ps_range_enc,
                &abs_pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH],
            );
        }
    }

    // encode_pulses.c:181-196: "LSB Encoding".
    for (i, block) in pulses.chunks(SHELL_CODEC_FRAME_LENGTH).enumerate() {
        if n_rshifts[i] > 0 {
            let n_ls = n_rshifts[i] - 1;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                // C: `abs_q = (opus_int8)silk_abs( pulses_ptr[ k ] )`.
                let abs_q = (block.get(k).copied().unwrap_or(0) as i32).abs() as i8 as i32;
                for j in (1..=n_ls).rev() {
                    let bit = (abs_q >> j) & 1;
                    ps_range_enc.encode_icdf(bit, &SILK_LSB_ICDF, 8);
                }
                let bit = abs_q & 1;
                ps_range_enc.encode_icdf(bit, &SILK_LSB_ICDF, 8);
            }
        }
    }

    // silk_encode_signs (silk/code_signs.c:40-72), over the same blocks:
    // `(length + SHELL_CODEC_FRAME_LENGTH / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH`
    // is `iter` for every frame length SILK codes.
    let icdf_offset = (7 * (quant_offset_type + signal_type * 2)) as usize;
    for (i, block) in pulses.chunks(SHELL_CODEC_FRAME_LENGTH).enumerate() {
        let p = sum_pulses[i];
        if p > 0 {
            let icdf0 = SILK_SIGN_ICDF[icdf_offset + ((p & 0x1F) as usize).min(6)];
            let icdf = [icdf0, 0u8];
            for &pulse in block {
                if pulse != 0 {
                    let mapped = if pulse > 0 { 1i32 } else { 0i32 };
                    ps_range_enc.encode_icdf(mapped, &icdf, 8);
                }
            }
        }
    }
}
