//! The software FRECPE (`vec_portable::frecpe_bits`) against the ARM
//! instruction `vrecpeq_f32`, which the NEON kernels use.
//!
//! The default run checks every combination of sign, exponent and top 8
//! fraction bits (the bits that the estimate reads), each with random low
//! bits, plus the special values: about 2^17 inputs. The ignored test
//! checks all 2^32 inputs (about 10 s on an Apple M4):
//!
//! ```text
//! cargo test --release --features dnn --test dnn_frecpe -- --ignored
//! ```
#![cfg(all(feature = "dnn", target_arch = "aarch64"))]

use core::arch::aarch64::*;
use opus1::internals::dnn::vec_portable::frecpe_bits;

/// `vrecpeq_f32` on 4 inputs.
fn hardware(x: [u32; 4]) -> [u32; 4] {
    let mut out = [0u32; 4];
    // SAFETY: NEON is part of every 64-bit ARM processor; the pointers
    // point to 4 values each.
    unsafe {
        let v = vreinterpretq_f32_u32(vld1q_u32(x.as_ptr()));
        vst1q_u32(out.as_mut_ptr(), vreinterpretq_u32_f32(vrecpeq_f32(v)));
    }
    out
}

fn check(x: [u32; 4]) -> usize {
    let hw = hardware(x);
    let mut bad = 0;
    for k in 0..4 {
        let sw = frecpe_bits(x[k]);
        if sw != hw[k] {
            if bad < 5 {
                eprintln!("frecpe {:08x}: instruction {:08x}, software {sw:08x}", x[k], hw[k]);
            }
            bad += 1;
        }
    }
    bad
}

#[test]
fn software_frecpe_matches_the_instruction_on_every_class() {
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng as u32
    };
    let mut bad = 0;
    let mut n = 0;
    for sign in [0u32, 1] {
        for exp in 0..256u32 {
            for top in 0..256u32 {
                let base = sign << 31 | exp << 23 | top << 15;
                let x = [base, base | (next() & 0x7fff), base | 0x7fff, base | (next() & 0x7fff)];
                bad += check(x);
                n += 4;
            }
        }
    }
    let specials = [
        0u32, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001, 0xffc1_2345, 0x0000_0001,
        0x001f_ffff, 0x0020_0000, 0x0040_0000, 0x007f_ffff, 0x3f80_0000, 0x7e80_0000, 0x7f00_0000, 0x7f7f_ffff,
    ];
    for c in specials.chunks(4) {
        bad += check([c[0], c[1], c[2], c[3]]);
        n += 4;
    }
    assert_eq!(bad, 0, "{bad} of {n} inputs differ");
}

#[test]
#[ignore = "all 2^32 inputs; about 10 s on an Apple M4, minutes on a phone"]
fn software_frecpe_matches_the_instruction_on_all_inputs() {
    let mut bad = 0usize;
    let mut u = 0u64;
    while u < 1 << 32 {
        let b = u as u32;
        bad += check([b, b + 1, b + 2, b + 3]);
        u += 4;
    }
    assert_eq!(bad, 0, "{bad} of 2^32 inputs differ");
}
