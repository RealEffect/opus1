//! The quantizer schedule of DRED, shared by the encoder and the decoder
//! (libopus `dnn/dred_coding.c`).

/// The quantizer level of latent vector `i` (libopus `compute_quantizer`).
///
/// The level starts at `q0` for the newest vector and rises by
/// `DQ_TABLE[dq] / 16` per vector, rounded, up to `qmax`. `dq` is the
/// 3-bit step index that the packet carries, so it must be below 8.
pub fn compute_quantizer(q0: i32, dq: usize, qmax: i32, i: i32) -> i32 {
    const DQ_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    let quant = q0 + (DQ_TABLE[dq] * i + 8) / 16;
    quant.min(qmax)
}

#[cfg(test)]
mod tests {
    use super::compute_quantizer;

    #[test]
    fn quantizer_rises_with_the_step_and_stops_at_qmax() {
        assert_eq!(compute_quantizer(6, 0, 15, 25), 6);
        // Step 16/16: one level per vector.
        assert_eq!(compute_quantizer(6, 7, 15, 3), 9);
        assert_eq!(compute_quantizer(6, 7, 15, 20), 15);
        // Step 3/16, rounded: (3*2 + 8)/16 = 0, (3*3 + 8)/16 = 1.
        assert_eq!(compute_quantizer(4, 2, 15, 2), 4);
        assert_eq!(compute_quantizer(4, 2, 15, 3), 5);
    }
}
