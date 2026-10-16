//! `silk_sum_sqr_shift` against the function inside C libopus
//! (silk/sum_sqr_shift.c). libopus is linked as a static library, so the
//! internal function can be called directly. The function is fixed-point:
//! energy and shift must be identical.

use opus1::internals::silk::sigproc_fix::silk_sum_sqr_shift;
use std::os::raw::c_int;

unsafe extern "C" {
    #[link_name = "silk_sum_sqr_shift"]
    fn c_silk_sum_sqr_shift(energy: *mut i32, shift: *mut c_int, x: *const i16, len: c_int);
}

/// Makes sure the test binary links libopus: the function above is found in
/// the static library that `audiopus_sys` brings along.
fn link_libopus() {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
}

/// libopus `silk_sum_sqr_shift`, statement by statement. A second reference,
/// independent of the linked library.
fn transcription(x: &[i16]) -> (i32, i32) {
    let len = x.len() as i32;
    let clz32 = |v: i32| if v == 0 { 32 } else { v.leading_zeros() as i32 };
    let mut shft = 31 - clz32(len);
    let mut nrg: i32 = len;
    let mut i = 0;
    while i < len - 1 {
        let (a, b) = (x[i as usize] as i32, x[i as usize + 1] as i32);
        let mut nrg_tmp = (a * a) as u32;
        nrg_tmp = nrg_tmp.wrapping_add((b * b) as u32);
        nrg = (nrg as u32).wrapping_add(nrg_tmp >> shft) as i32;
        i += 2;
    }
    if i < len {
        let a = x[i as usize] as i32;
        nrg = (nrg as u32).wrapping_add(((a * a) as u32) >> shft) as i32;
    }
    shft = (shft + 3 - clz32(nrg)).max(0);
    nrg = 0;
    i = 0;
    while i < len - 1 {
        let (a, b) = (x[i as usize] as i32, x[i as usize + 1] as i32);
        let mut nrg_tmp = (a * a) as u32;
        nrg_tmp = nrg_tmp.wrapping_add((b * b) as u32);
        nrg = (nrg as u32).wrapping_add(nrg_tmp >> shft) as i32;
        i += 2;
    }
    if i < len {
        let a = x[i as usize] as i32;
        nrg = (nrg as u32).wrapping_add(((a * a) as u32) >> shft) as i32;
    }
    (nrg, shft)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Test signals. Kinds 0 to 5 are loud: with them the second pass of the
/// function shifts, and the result depends on whether a pair of squares is
/// shifted as a sum or square by square.
fn signal(rng: &mut Lcg, len: usize, kind: usize) -> Vec<i16> {
    let level = 1 + rng.next() % 12;
    (0..len)
        .map(|i| match kind {
            0 => rng.next() as i16,
            1 => {
                if rng.next().is_multiple_of(2) { i16::MAX } else { i16::MIN }
            }
            2 => i16::MIN,
            3 => i16::MAX,
            4 => {
                if i % 2 == 0 { i16::MIN } else { i16::MAX }
            }
            // A loud tone: every second pair has two odd squares.
            5 => (30000.0 * (i as f64 * 0.37).sin()) as i16 | 1,
            6 => (rng.next() as i16) >> level,
            _ => 0,
        })
        .collect()
}

#[test]
fn sum_sqr_shift_matches_libopus_for_every_length_and_level() {
    link_libopus();
    let mut rng = Lcg(0x5ee_d0fa_c0de);
    let mut shifted_cases = 0;
    // Every length up to 100 (all remainders of a vector loop, with and
    // without a last sample that has no partner), then the lengths of real
    // calls up to a whole 20 ms frame with history at 16 kHz.
    let lengths = (1..=100usize).chain([160, 161, 320, 480, 640, 656, 959, 960]);
    for len in lengths {
        for kind in 0..8 {
            for _ in 0..4 {
                let x = signal(&mut rng, len, kind);
                let (mut c_energy, mut c_shift) = (-1i32, -1 as c_int);
                unsafe {
                    c_silk_sum_sqr_shift(&mut c_energy, &mut c_shift, x.as_ptr(), len as c_int);
                }
                assert_eq!(
                    transcription(&x),
                    (c_energy, c_shift),
                    "transcription against libopus: len={len} kind={kind}"
                );

                // Samples after `len` must not be read.
                let mut padded = x.clone();
                padded.extend_from_slice(&[i16::MIN; 3]);
                let (mut energy, mut shift) = (-1i32, -1i32);
                silk_sum_sqr_shift(&mut energy, &mut shift, &padded, len);
                assert_eq!(
                    (energy, shift),
                    (c_energy, c_shift),
                    "opus1 against libopus: len={len} kind={kind}"
                );
                if c_shift > 0 {
                    shifted_cases += 1;
                }
            }
        }
    }
    // Most of the cases are loud enough for the second pass to shift.
    assert!(shifted_cases > 1500, "{shifted_cases} cases with a shift");
}
