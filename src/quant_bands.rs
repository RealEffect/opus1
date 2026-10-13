use crate::modes::CeltMode;
use crate::range_coder::{BITRES, RANGE_BUF_MAX, RangeCoder};

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;

pub const PRED_COEF: [f32; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
pub const BETA_COEF: [f32; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
pub const BETA_INTRA: f32 = 4915.0 / 32768.0;

pub const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

pub const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

fn loss_distortion(
    e_bands: &[f32],
    old_e_bands: &[f32],
    start: usize,
    end: usize,
    len: usize,
    channels: usize,
) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..channels {
        let off = c * len;
        for i in start..end.min(len) {
            let d = e_bands[off + i] - old_e_bands[off + i];
            dist += d * d;
        }
    }
    dist.min(200.0)
}

#[allow(clippy::too_many_arguments)]
fn quant_coarse_energy_impl(
    m: &CeltMode,
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    tell_start: i32,
    prob_model: &[u8; 42],
    error: &mut [f32],
    enc: &mut RangeCoder,
    channels: usize,
    lm: usize,
    intra: bool,
    max_decay: f32,
    lfe: bool,
) -> i32 {
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    let mut prev = [0.0f32; 2];
    let mut badness = 0i32;

    if tell_start + 3 <= budget as i32 {
        enc.encode_bit_logp(intra, 3);
    }

    for i in start..end {
        for c in 0..channels {
            let x = e_bands[c * m.nb_ebands + i];
            let old_e_val = old_e_bands[c * m.nb_ebands + i];
            let old_e = old_e_val.max(-9.0);
            let f = x - coef * old_e - prev[c];

            let mut qi = ((f + 0.5).floor() as i32).clamp(-32767, 32767);
            let qi0 = qi;

            let decay_bound = old_e_val.max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi = qi.saturating_add(((decay_bound - x) as i32).max(0));
                if qi > 0 {
                    qi = 0;
                }
            }

            let tell = enc.tell();
            let bits_left = budget as i32 - tell - 3 * channels as i32 * (end - i) as i32;
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if lfe && i >= 2 {
                qi = qi.min(0);
            }

            if tell + 15 <= budget as i32 {
                let prob_idx = 2 * i.min(20);
                let fs = (prob_model[prob_idx] as u32) << 7;
                let decay = (prob_model[prob_idx + 1] as i32) << 6;
                enc.laplace_encode(&mut qi, fs, decay);
            } else if tell + 2 <= budget as i32 {
                qi = qi.clamp(-1, 1);
                enc.encode_icdf(
                    (2 * qi) ^ (if qi < 0 { -1 } else { 0 }),
                    &SMALL_ENERGY_ICDF,
                    2,
                );
            } else if tell < budget as i32 {
                qi = qi.min(0);
                enc.encode_bit_logp(qi != 0, 1);
            } else {
                qi = -1;
            }

            badness = badness.saturating_add(qi0.saturating_sub(qi).saturating_abs());

            let q = qi as f32;
            error[c * m.nb_ebands + i] = f - q;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[c * m.nb_ebands + i] = tmp;
            prev[c] = prev[c] + q - beta * q;
        }
    }

    if lfe { 0 } else { badness }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy_advanced(
    m: &CeltMode,
    start: usize,
    end: usize,
    eff_end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut RangeCoder,
    channels: usize,
    lm: usize,
    nb_available_bytes: usize,
    force_intra: bool,
    delayed_intra: &mut f32,
    mut two_pass: bool,
    loss_rate: i32,
    lfe: bool,
) {
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > 2.0 * channels as f32 * (end.saturating_sub(start)) as f32
            && nb_available_bytes > (end.saturating_sub(start)) * channels);

    let intra_bias = ((budget as f32) * (*delayed_intra) * (loss_rate as f32)
        / ((channels as f32) * 512.0)) as i32;
    let new_distortion =
        loss_distortion(e_bands, old_e_bands, start, eff_end, m.nb_ebands, channels);

    let tell = enc.tell();
    if tell + 3 > budget as i32 {
        two_pass = false;
        intra = false;
    }

    let mut max_decay = if end - start > 10 {
        16.0f32.min(0.125 * nb_available_bytes as f32)
    } else {
        16.0f32
    };
    if lfe {
        max_decay = 3.0;
    }

    if intra || !two_pass {
        // Only one pass runs, so it codes straight into the caller's arrays
        // and into `enc`. Nothing has to be saved.
        quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_e_bands,
            budget,
            tell,
            &E_PROB_MODEL[lm][intra as usize],
            error,
            enc,
            channels,
            lm,
            intra,
            max_decay,
            lfe,
        );
    } else {
        // Two passes: the intra pass first, then the inter pass from the same
        // coder state. The better one stays (C: quant_bands.c:295-341).
        //
        // The coder is not cloned for this, because a clone copies the byte
        // buffer too. Both passes only add range coder bytes from
        // `start_offs` on. So it is enough to save the 44 bytes of coder
        // words and, for the intra pass, the bytes it wrote.
        //
        // Compared with full copies, the coder ends with the same words and
        // the same bytes below `offs`. Above `offs` the buffer may still hold
        // bytes of the pass that lost. Later symbols overwrite them and
        // `done()` clears the rest. `done()` skips that only when the coder
        // has an error, and then the packet is not valid anyway.
        let start_state = enc.save_state();
        let start_offs = enc.offs as usize;
        let mut old_e_bands_intra: FixedVec<f32, 42> = FixedVec::from_slice(old_e_bands);
        let mut error_intra: FixedVec<f32, 42> = FixedVec::from_slice(error);

        let badness1 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            &mut old_e_bands_intra,
            budget,
            tell,
            &E_PROB_MODEL[lm][1],
            &mut error_intra,
            enc,
            channels,
            lm,
            true,
            max_decay,
            lfe,
        );
        let tell_intra = crate::tell_frac_inline!(enc);

        let intra_state = enc.save_state();
        let intra_offs = enc.offs as usize;
        let mut intra_bytes: FixedVec<u8, RANGE_BUF_MAX> = FixedVec::new();
        intra_bytes.extend_from_slice(&enc.buf[start_offs..intra_offs]);

        enc.restore_state(&start_state);
        let badness2 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_e_bands,
            budget,
            tell,
            &E_PROB_MODEL[lm][0],
            error,
            enc,
            channels,
            lm,
            false,
            max_decay,
            lfe,
        );

        if badness1 < badness2
            || (badness1 == badness2 && crate::tell_frac_inline!(enc) + intra_bias > tell_intra)
        {
            enc.restore_state(&intra_state);
            enc.buf[start_offs..intra_offs].copy_from_slice(&intra_bytes);
            old_e_bands.copy_from_slice(&old_e_bands_intra);
            error.copy_from_slice(&error_intra);
            intra = true;
        }
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        let pred2 = PRED_COEF[lm] * PRED_COEF[lm];
        *delayed_intra = pred2 * *delayed_intra + new_distortion;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut RangeCoder,
    channels: usize,
    lm: usize,
    force_intra: bool,
    nb_available_bytes: usize,
) {
    let mut delayed_intra = 0.0f32;
    quant_coarse_energy_advanced(
        m,
        start,
        end,
        end,
        e_bands,
        old_e_bands,
        budget,
        error,
        enc,
        channels,
        lm,
        nb_available_bytes,
        force_intra,
        &mut delayed_intra,
        false,
        0,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn unquant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    intra: bool,
    dec: &mut RangeCoder,
    channels: usize,
    lm: usize,
) {
    let prob_model = &E_PROB_MODEL[lm][if intra { 1 } else { 0 }];
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    debug_assert!(channels <= 2);
    let mut prev = [0.0f32; 2];
    let budget = (dec.storage * 8) as i32;

    for i in start..end {
        for c in 0..channels {
            let qi;
            let tell = dec.tell();
            if budget - tell >= 15 {
                let prob_idx = 2 * i.min(20);
                let fs = (prob_model[prob_idx] as u32) << 7;
                let decay = (prob_model[prob_idx + 1] as i32) << 6;
                qi = dec.laplace_decode(fs, decay);
            } else if budget - tell >= 2 {
                let s = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2);
                qi = (s >> 1) ^ -(s & 1);
            } else if budget - tell >= 1 {
                qi = if dec.decode_bit_logp(1) { -1 } else { 0 };
            } else {
                qi = -1;
            }

            // Clamp in-place, matching C: oldEBands[i] = MAXG(-GCONST(9.f), oldEBands[i])
            old_e_bands[c * m.nb_ebands + i] = old_e_bands[c * m.nb_ebands + i].max(-9.0);
            let old_e = old_e_bands[c * m.nb_ebands + i];

            let q = qi as f32;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[c * m.nb_ebands + i] = tmp;
            prev[c] = prev[c] + q - beta * q;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_fine_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    enc: &mut RangeCoder,
    channels: usize,
) {
    for i in start..end {
        for c in 0..channels {
            let bits = fine_quant[i];
            if bits <= 0 {
                continue;
            }
            let mut q = ((error[c * m.nb_ebands + i] + 0.5) * (1 << bits) as f32).floor() as i32;
            q = q.max(0).min((1 << bits) - 1);
            enc.enc_bits(q as u32, bits as u32);
            // C (quant_bands.c): offset = (q+.5)*(1<<(14-bits))*(1/16384) - .5;
            // (multiply-then-scale, not /2^bits; matters for byte-exactness)
            let offset = (q as f32 + 0.5) * (1i32 << (14 - bits)) as f32 * (1.0 / 16384.0) - 0.5;
            old_e_bands[c * m.nb_ebands + i] += offset;
            error[c * m.nb_ebands + i] -= offset;
        }
    }
}

pub fn unquant_fine_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    fine_quant: &[i32],
    dec: &mut RangeCoder,
    channels: usize,
) {
    for i in start..end {
        for c in 0..channels {
            let bits = fine_quant[i];
            if bits <= 0 {
                continue;
            }
            let q = dec.dec_bits(bits as u32);
            // C (quant_bands.c): offset = (q+.5)*(1<<(14-bits))*(1/16384) - .5;
            // (multiply-then-scale, not /2^bits; matters for byte-exactness)
            let offset = (q as f32 + 0.5) * (1i32 << (14 - bits)) as f32 * (1.0 / 16384.0) - 0.5;
            old_e_bands[c * m.nb_ebands + i] += offset;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_energy_finalise(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    bits_left: i32,
    enc: &mut RangeCoder,
    channels: usize,
) {
    let mut bits_left = bits_left;
    for priority in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= 8 || fine_priority[i] != priority {
                i += 1;
                continue;
            }
            let mut c = 0;
            while c < channels {
                let q2 = if error[i + c * m.nb_ebands] < 0.0 {
                    0
                } else {
                    1
                };
                enc.enc_bits(q2 as u32, 1);
                let offset =
                    (q2 as f32 - 0.5) * (1i32 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old_e_bands[i + c * m.nb_ebands] += offset;
                error[i + c * m.nb_ebands] -= offset;
                bits_left -= 1;
                c += 1;
            }
            i += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn unquant_energy_finalise(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    bits_left: i32,
    dec: &mut RangeCoder,
    channels: usize,
) {
    let mut bits_left = bits_left;
    for priority in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= 8 || fine_priority[i] != priority {
                i += 1;
                continue;
            }
            let mut c = 0;
            while c < channels {
                let q2 = dec.dec_bits(1);
                let offset =
                    (q2 as f32 - 0.5) * (1i32 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old_e_bands[i + c * m.nb_ebands] += offset;
                bits_left -= 1;
                c += 1;
            }
            i += 1;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::range_coder::RangeCoder;

    #[test]
    fn test_coarse_fine_energy() {
        let mode = crate::modes::default_mode();
        let mut e_bands = vec![0.0; mode.nb_ebands];
        for (i, v) in e_bands.iter_mut().enumerate() {
            *v = 5.0 + (i as f32 * 0.5).sin() * 2.0;
        }

        let mut old_e_bands = vec![0.0; mode.nb_ebands];
        let mut error = vec![0.0; mode.nb_ebands];
        let mut enc = RangeCoder::new_encoder(1000);

        quant_coarse_energy(
            mode,
            0,
            mode.nb_ebands,
            &e_bands,
            &mut old_e_bands,
            10000,
            &mut error,
            &mut enc,
            1,
            3,
            false,
            80,
        );

        let mut fine_quant = vec![0; mode.nb_ebands];
        for (i, v) in fine_quant.iter_mut().enumerate() {
            *v = (i % 3) as i32;
        }

        quant_fine_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut old_e_bands,
            &mut error,
            &fine_quant,
            &mut enc,
            1,
        );

        let mut fine_priority = vec![0i32; mode.nb_ebands];
        for (i, v) in fine_priority.iter_mut().enumerate() {
            *v = (i % 2) as i32;
        }

        quant_energy_finalise(
            mode,
            0,
            mode.nb_ebands,
            &mut old_e_bands,
            &mut error,
            &fine_quant,
            &fine_priority,
            10,
            &mut enc,
            1,
        );

        enc.done();
        let _compressed = &enc.buf;

        let mut dec = RangeCoder::new_decoder(&enc.buf);

        let mut decoded_old_e_bands = vec![0.0; mode.nb_ebands];
        let intra = dec.decode_bit_logp(3);
        unquant_coarse_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            intra,
            &mut dec,
            1,
            3,
        );

        unquant_fine_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            &fine_quant,
            &mut dec,
            1,
        );

        unquant_energy_finalise(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            &fine_quant,
            &fine_priority,
            10,
            &mut dec,
            1,
        );

        for i in 0..mode.nb_ebands {
            if (decoded_old_e_bands[i] - old_e_bands[i]).abs() >= 1e-5 {
                println!(
                    "Mismatch at band {}: enc={} dec={} diff={}",
                    i,
                    old_e_bands[i],
                    decoded_old_e_bands[i],
                    (decoded_old_e_bands[i] - old_e_bands[i]).abs()
                );
            }
            assert!((decoded_old_e_bands[i] - old_e_bands[i]).abs() < 1e-5);
        }
    }

    /// `quant_coarse_energy_advanced` as it was before it stopped cloning the
    /// coder: every pass works on a full copy of the coder and of the arrays.
    /// The reference for `coarse_energy_matches_full_copy_reference`.
    #[allow(clippy::too_many_arguments)]
    fn reference_quant_coarse_energy_advanced(
        m: &CeltMode,
        start: usize,
        end: usize,
        eff_end: usize,
        e_bands: &[f32],
        old_e_bands: &mut [f32],
        budget: u32,
        error: &mut [f32],
        enc: &mut RangeCoder,
        channels: usize,
        lm: usize,
        nb_available_bytes: usize,
        force_intra: bool,
        delayed_intra: &mut f32,
        mut two_pass: bool,
        loss_rate: i32,
        lfe: bool,
    ) {
        let mut intra = force_intra
            || (!two_pass
                && *delayed_intra > 2.0 * channels as f32 * (end.saturating_sub(start)) as f32
                && nb_available_bytes > (end.saturating_sub(start)) * channels);
        let intra_bias = ((budget as f32) * (*delayed_intra) * (loss_rate as f32)
            / ((channels as f32) * 512.0)) as i32;
        let new_distortion =
            loss_distortion(e_bands, old_e_bands, start, eff_end, m.nb_ebands, channels);
        let tell = enc.tell();
        if tell + 3 > budget as i32 {
            two_pass = false;
            intra = false;
        }
        let mut max_decay = if end - start > 10 {
            16.0f32.min(0.125 * nb_available_bytes as f32)
        } else {
            16.0f32
        };
        if lfe {
            max_decay = 3.0;
        }

        let enc_start_state = enc.clone();
        let mut old_e_bands_intra = old_e_bands.to_vec();
        let mut error_intra = error.to_vec();
        let mut badness1 = 0i32;
        let mut tell_intra = 0i32;
        if two_pass || intra {
            badness1 = quant_coarse_energy_impl(
                m,
                start,
                end,
                e_bands,
                &mut old_e_bands_intra,
                budget,
                tell,
                &E_PROB_MODEL[lm][1],
                &mut error_intra,
                enc,
                channels,
                lm,
                true,
                max_decay,
                lfe,
            );
            tell_intra = enc.tell_frac();
        }
        if !intra {
            let enc_intra_state = enc.clone();
            *enc = enc_start_state.clone();
            let badness2 = quant_coarse_energy_impl(
                m,
                start,
                end,
                e_bands,
                old_e_bands,
                budget,
                tell,
                &E_PROB_MODEL[lm][0],
                error,
                enc,
                channels,
                lm,
                false,
                max_decay,
                lfe,
            );
            if two_pass
                && (badness1 < badness2
                    || (badness1 == badness2 && enc.tell_frac() + intra_bias > tell_intra))
            {
                *enc = enc_intra_state;
                old_e_bands.copy_from_slice(&old_e_bands_intra);
                error.copy_from_slice(&error_intra);
                intra = true;
            }
        } else {
            old_e_bands.copy_from_slice(&old_e_bands_intra);
            error.copy_from_slice(&error_intra);
        }
        if intra {
            *delayed_intra = new_distortion;
        } else {
            let pred2 = PRED_COEF[lm] * PRED_COEF[lm];
            *delayed_intra = pred2 * *delayed_intra + new_distortion;
        }
    }

    /// Small deterministic generator (xorshift64*).
    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            (self.next() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    /// The words of a coder, for comparing two coders.
    fn coder_words(rc: &RangeCoder) -> [i64; 11] {
        [
            rc.storage as i64,
            rc.end_offs as i64,
            rc.end_window as i64,
            rc.nend_bits as i64,
            rc.nbits_total as i64,
            rc.offs as i64,
            rc.rng as i64,
            rc.val as i64,
            rc.ext as i64,
            rc.rem as i64,
            rc.error as i64,
        ]
    }

    /// The coarse energy coder saves the coder words and the bytes of the
    /// intra pass instead of cloning the coder. It must leave the same coder
    /// words, the same range coder bytes, the same arrays and the same packet
    /// as the version that clones.
    ///
    /// The cases cover mono and stereo, all four frame sizes, one and two
    /// passes, forced intra, the low-frequency channel, a band range that
    /// starts at band 17, budgets from "no room for the intra flag" to
    /// "everything fits", and band energies from steady to jumping by 20
    /// steps. The coder already holds symbols when the coarse energy starts;
    /// in variant 2 it also holds raw bits and a run of 0xFF bytes whose
    /// carry is still open. The buffer starts with bytes of an earlier frame.
    /// Six frames run in a row, so the prediction state and `delayed_intra`
    /// of one frame feed the next.
    #[test]
    fn coarse_energy_matches_full_copy_reference() {
        let mode = crate::modes::default_mode();
        let nb = mode.nb_ebands;
        let mut rng = TestRng(0x1234_5678_9ABC_DEF1);
        // How often each pass won where both ran, read back from the packet.
        let mut intra_wins = 0;
        let mut inter_wins = 0;

        for channels in 1..=2usize {
            for lm in 0..4usize {
                for &two_pass in &[true, false] {
                    for &(storage, budget_bits) in &[
                        (1275u32, 1275 * 8u32),
                        (160, 160 * 8),
                        (40, 40 * 8),
                        (12, 12 * 8),
                        (6, 40),
                        (3, 24),
                        (2, 10),
                        (160, 2),
                    ] {
                        for variant in 0..6u32 {
                            let force_intra = variant == 4;
                            let lfe = variant == 5;
                            let (start, end) = if variant == 3 { (17, nb) } else { (0, nb) };
                            let open_carry = variant == 2 && storage >= 40;
                            let n = channels * nb;

                            let mut enc_new = RangeCoder::new_encoder(storage);
                            enc_new.buf.fill(0x5A);
                            let mut old_new = [0f32; 42];
                            let mut old_ref = [0f32; 42];
                            let mut err_new = [0f32; 42];
                            let mut err_ref = [0f32; 42];
                            let mut delayed_new = 0f32;
                            let mut delayed_ref = 0f32;
                            let mut level = [0f32; 42];
                            for (i, l) in level.iter_mut().enumerate() {
                                *l = 8.0 - 0.3 * (i % nb) as f32 + 2.0 * rng.unit();
                            }

                            for frame in 0..6 {
                                let step = if frame == 3 { 20.0 } else { 0.7 };
                                let mut e_bands = [0f32; 42];
                                for i in 0..n {
                                    level[i] = (level[i] + step * rng.unit()).clamp(-28.0, 30.0);
                                    e_bands[i] = level[i];
                                }
                                let loss_rate = if frame == 4 { 20 } else { 0 };

                                // What the frame codes before the coarse energy.
                                enc_new.reset_for_encode(storage);
                                enc_new.encode_bit_logp(false, 15);
                                enc_new.encode_bit_logp(frame & 1 == 1, 1);
                                if open_carry {
                                    for _ in 0..12 {
                                        enc_new.encode(255, 256, 256);
                                    }
                                    enc_new.enc_bits(0x2B, 6);
                                    assert!(enc_new.ext > 0, "no open carry");
                                }
                                let mut enc_ref = enc_new.clone();
                                let flag_coded = enc_ref.tell() + 3 <= budget_bits as i32;

                                quant_coarse_energy_advanced(
                                    mode,
                                    start,
                                    end,
                                    end,
                                    &e_bands[..n],
                                    &mut old_new[..n],
                                    budget_bits,
                                    &mut err_new[..n],
                                    &mut enc_new,
                                    channels,
                                    lm,
                                    storage as usize,
                                    force_intra,
                                    &mut delayed_new,
                                    two_pass,
                                    loss_rate,
                                    lfe,
                                );
                                reference_quant_coarse_energy_advanced(
                                    mode,
                                    start,
                                    end,
                                    end,
                                    &e_bands[..n],
                                    &mut old_ref[..n],
                                    budget_bits,
                                    &mut err_ref[..n],
                                    &mut enc_ref,
                                    channels,
                                    lm,
                                    storage as usize,
                                    force_intra,
                                    &mut delayed_ref,
                                    two_pass,
                                    loss_rate,
                                    lfe,
                                );

                                let what = format!(
                                    "channels {channels} lm {lm} two_pass {two_pass} storage \
                                     {storage} budget {budget_bits} variant {variant} frame {frame}"
                                );
                                assert_eq!(coder_words(&enc_new), coder_words(&enc_ref), "{what}");
                                let offs = enc_ref.offs as usize;
                                assert_eq!(enc_new.buf[..offs], enc_ref.buf[..offs], "{what}");
                                let raw = (enc_ref.storage - enc_ref.end_offs) as usize;
                                assert_eq!(enc_new.buf[raw..], enc_ref.buf[raw..], "{what}");
                                assert_eq!(
                                    old_new.map(f32::to_bits),
                                    old_ref.map(f32::to_bits),
                                    "{what}"
                                );
                                assert_eq!(
                                    err_new.map(f32::to_bits),
                                    err_ref.map(f32::to_bits),
                                    "{what}"
                                );
                                assert_eq!(delayed_new.to_bits(), delayed_ref.to_bits(), "{what}");

                                // The finished packets are the same as well.
                                // `enc_new` itself is finished, so that the
                                // next frame starts on the bytes of this one.
                                enc_new.done();
                                enc_ref.done();
                                assert_eq!(enc_new.error, enc_ref.error, "{what}");
                                if enc_ref.error != 0 {
                                    continue;
                                }
                                let s = enc_ref.storage as usize;
                                assert_eq!(enc_new.buf[..s], enc_ref.buf[..s], "{what}");

                                // Read the intra flag back from the packet.
                                if two_pass && !force_intra && flag_coded {
                                    let mut dec = RangeCoder::new_decoder(&enc_new.buf[..s]);
                                    assert!(!dec.decode_bit_logp(15), "{what}");
                                    assert_eq!(dec.decode_bit_logp(1), frame & 1 == 1, "{what}");
                                    if open_carry {
                                        for _ in 0..12 {
                                            assert_eq!(dec.decode(256), 255, "{what}");
                                            dec.update(255, 256, 256);
                                        }
                                        assert_eq!(dec.dec_bits(6), 0x2B, "{what}");
                                    }
                                    if dec.decode_bit_logp(3) {
                                        intra_wins += 1;
                                    } else {
                                        inter_wins += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Both outcomes of the two-pass decision must have been exercised,
        // because they restore the coder in different ways.
        assert!(intra_wins > 100, "intra chosen in only {intra_wins} cases");
        assert!(inter_wins > 100, "inter chosen in only {inter_wins} cases");
    }

    /// Regression test: extreme/corrupt energy values must not cause an
    /// "attempt to add with overflow" panic in `quant_coarse_energy_impl`.
    /// Previously `(qi0 - qi).abs()` overflowed i32 when the float->int cast
    /// of `qi` saturated near i32::MAX/MIN.
    #[test]
    fn test_coarse_energy_extreme_no_overflow() {
        let mode = crate::modes::default_mode();
        let n = mode.nb_ebands;

        for &extreme in &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0e30, -1.0e30] {
            let e_bands = vec![extreme; n];
            let mut old_e_bands = vec![0.0; n];
            let mut error = vec![0.0; n];
            let mut enc = RangeCoder::new_encoder(1000);

            // tiny budget forces the `else { qi = -1 }` path, which combined
            // with a saturated qi0 triggered the overflow at the badness line.
            quant_coarse_energy(
                mode,
                0,
                n,
                &e_bands,
                &mut old_e_bands,
                0,
                &mut error,
                &mut enc,
                1,
                3,
                false,
                80,
            );

            // large budget exercises the laplace-encode path with extreme qi.
            let mut old_e_bands2 = vec![0.0; n];
            let mut error2 = vec![0.0; n];
            let mut enc2 = RangeCoder::new_encoder(1000);
            quant_coarse_energy(
                mode,
                0,
                n,
                &e_bands,
                &mut old_e_bands2,
                10000,
                &mut error2,
                &mut enc2,
                1,
                3,
                false,
                80,
            );
        }
    }
}
