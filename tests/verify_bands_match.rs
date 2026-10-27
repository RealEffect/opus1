//! Band energies and band normalisation (`compute_band_energies`,
//! `normalise_bands`, src/bands.rs) against the functions of the same name
//! in libopus (celt/bands.c), called in the static library the tests link.
//!
//! One short block (120 values) of a sine with a slow amplitude change.
//! On 64-bit ARM the energies and the normalised values must be equal. On
//! other processors opus1 and libopus may sum the squares in another order,
//! which changes the last bit of some energies.

use opus1::internals::bands::{compute_band_energies, normalise_bands};
use opus1::internals::modes::default_mode;
use std::os::raw::{c_int, c_void};

unsafe extern "C" {
    fn opus_custom_mode_create(fs: i32, frame_size: c_int, error: *mut c_int) -> *const c_void;
    #[link_name = "compute_band_energies"]
    fn c_compute_band_energies(
        m: *const c_void,
        x: *const f32,
        band_e: *mut f32,
        end: c_int,
        channels: c_int,
        lm: c_int,
        arch: c_int,
    );
    #[link_name = "normalise_bands"]
    fn c_normalise_bands(
        m: *const c_void,
        freq: *const f32,
        x: *mut f32,
        band_e: *const f32,
        end: c_int,
        channels: c_int,
        big_m: c_int,
    );
}

/// The 48 kHz mode of libopus. The call also makes sure the test binary
/// links the static library that `audiopus_sys` brings along.
fn c_mode() -> *const c_void {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
    let mut err: c_int = -1;
    let mode = unsafe { opus_custom_mode_create(48000, 960, &mut err) };
    assert!(err == 0 && !mode.is_null(), "libopus has no 48 kHz mode: {err}");
    mode
}

/// Band energies and normalised values.
type Bands = (Vec<f32>, Vec<f32>);

/// The bands of opus1 and of libopus: (opus1, libopus).
fn both() -> (Bands, Bands) {
    let mode = default_mode();
    let nb_ebands = mode.nb_ebands;
    let lm: usize = 0;
    let n = mode.short_mdct_size << lm;
    let x: Vec<f32> =
        (0..n).map(|i| (i as f32 * 0.1).sin() * (1.0 + 0.5 * (i as f32 * 0.01).cos())).collect();

    let mut band_e = vec![0.0f32; nb_ebands];
    compute_band_energies(mode, &x, &mut band_e, nb_ebands, 1, lm);
    let mut x_norm = vec![0.0f32; n];
    normalise_bands(mode, &x, &mut x_norm, &band_e, nb_ebands, 1, 1 << lm);

    let mut c_band_e = vec![0.0f32; nb_ebands];
    let mut c_norm = vec![0.0f32; n];
    // SAFETY: `x` and `c_norm` hold one block of `short_mdct_size << lm`
    // values, `c_band_e` one energy per band, and the mode is libopus's.
    unsafe {
        let m = c_mode();
        c_compute_band_energies(m, x.as_ptr(), c_band_e.as_mut_ptr(), nb_ebands as c_int, 1, lm as c_int, 0);
        c_normalise_bands(m, x.as_ptr(), c_norm.as_mut_ptr(), c_band_e.as_ptr(), nb_ebands as c_int, 1, 1 << lm);
    }
    // The signal reaches every band.
    assert!(c_band_e[1..].iter().all(|&e| e > 0.1), "{c_band_e:?}");
    ((band_e, x_norm), (c_band_e, c_norm))
}

/// Every value within a few units of the last place of libopus's: the
/// processors and libopus builds that sum in another order.
#[test]
fn test_bands_match() {
    let ((band_e, x_norm), (c_band_e, c_norm)) = both();
    for (what, a, b) in [("band energy", &band_e, &c_band_e), ("normalised value", &x_norm, &c_norm)] {
        for (i, (&a, &b)) in a.iter().zip(b.iter()).enumerate() {
            assert!((a - b).abs() <= 4.0 * f32::EPSILON * b.abs().max(1e-6), "{what} {i}: {a} where libopus has {b}");
        }
    }
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn bands_are_those_of_libopus_bit_for_bit() {
    let ((band_e, x_norm), (c_band_e, c_norm)) = both();
    assert_eq!(band_e, c_band_e, "band energies");
    assert_eq!(x_norm, c_norm, "normalised values");
}
