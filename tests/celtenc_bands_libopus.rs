//! The band functions of the CELT encoder against the same functions of C
//! libopus, called directly on the same input.
//!
//! The tests link libopus as a static library, which keeps the names of its
//! internal functions. Both sides get identical values, so their results
//! must be identical too, to the bit.
//!
//! Run with `--nocapture` to see the measured numbers.

mod celtenc_common;
use celtenc_common::Noise;

use opus1::internals::bands;
use opus1::internals::modes::{CeltMode, default_mode};
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::rate::clt_compute_allocation;
use std::os::raw::{c_int, c_uchar, c_void};

/// libopus `ec_ctx` (celt/entcode.h).
#[repr(C)]
struct EcCtx {
    buf: *mut c_uchar,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: c_int,
    nbits_total: c_int,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: c_int,
    error: c_int,
}

unsafe extern "C" {
    fn opus_custom_mode_create(fs: i32, frame_size: c_int, error: *mut c_int) -> *const c_void;
    fn ec_enc_init(enc: *mut EcCtx, buf: *mut c_uchar, size: u32);
    fn ec_enc_done(enc: *mut EcCtx);
    fn ec_dec_init(dec: *mut EcCtx, buf: *mut c_uchar, size: u32);
    fn quant_all_bands(
        encode: c_int,
        m: *const c_void,
        start: c_int,
        end: c_int,
        x: *mut f32,
        y: *mut f32,
        collapse_masks: *mut c_uchar,
        band_e: *const f32,
        pulses: *mut c_int,
        short_blocks: c_int,
        spread: c_int,
        dual_stereo: c_int,
        intensity: c_int,
        tf_res: *mut c_int,
        total_bits: i32,
        balance: i32,
        ec: *mut EcCtx,
        lm: c_int,
        coded_bands: c_int,
        seed: *mut u32,
        complexity: c_int,
        arch: c_int,
        disable_inv: c_int,
    );
    fn alg_quant(
        x: *mut f32,
        n: c_int,
        k: c_int,
        spread: c_int,
        b: c_int,
        enc: *mut EcCtx,
        gain: f32,
        resynth: c_int,
        arch: c_int,
    ) -> u32;
    fn alg_unquant(
        x: *mut f32,
        n: c_int,
        k: c_int,
        spread: c_int,
        b: c_int,
        dec: *mut EcCtx,
        gain: f32,
    ) -> u32;
    fn exp_rotation(x: *mut f32, len: c_int, dir: c_int, stride: c_int, k: c_int, spread: c_int);
    fn op_pvq_search_c(x: *mut f32, iy: *mut c_int, k: c_int, n: c_int, arch: c_int) -> f32;
    fn anti_collapse(
        m: *const c_void,
        x: *mut f32,
        collapse_masks: *mut c_uchar,
        lm: c_int,
        c: c_int,
        size: c_int,
        start: c_int,
        end: c_int,
        log_e: *const f32,
        prev1_log_e: *const f32,
        prev2_log_e: *const f32,
        pulses: *const c_int,
        seed: u32,
        encode: c_int,
        arch: c_int,
    );
    fn denormalise_bands(
        m: *const c_void,
        x: *const f32,
        freq: *mut f32,
        band_log_e: *const f32,
        start: c_int,
        end: c_int,
        lm_m: c_int,
        downsample: c_int,
        silence: c_int,
    );
    fn stereo_itheta(x: *const f32, y: *const f32, stereo: c_int, n: c_int, arch: c_int) -> i32;
    fn compute_band_energies(
        m: *const c_void,
        x: *const f32,
        band_e: *mut f32,
        end: c_int,
        c: c_int,
        lm: c_int,
        arch: c_int,
    );
}

/// The 48 kHz mode of libopus, the one its encoder uses for every rate.
fn c_mode() -> *const c_void {
    let mut err = 0;
    let m = unsafe { opus_custom_mode_create(48_000, 960, &mut err) };
    assert!(!m.is_null() && err == 0, "opus_custom_mode_create");
    m
}

/// `stereo_itheta` gives the angle libopus gives, to the bit, in Q30.
///
/// libopus 1.6 starts both energies at 0 and returns
/// `floor(0.5 + 2^30 * theta)`, which `compute_theta` shifts down by 16.
/// opus1 started at 1e-15 and rounded `0.5 + 16384 * theta`: a silent
/// band got 8192 instead of 0, and one angle in two was one step too high.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn stereo_angles_are_the_ones_libopus_computes() {
    let mut noise = Noise(3);
    let mut checked = 0;
    for n in (1..=40).chain([44, 48, 64, 72, 88, 96, 128, 144, 176]) {
        for stereo in [false, true] {
            for round in 0..40 {
                let mut x: Vec<f32> = (0..n).map(|_| noise.next()).collect();
                let mut y: Vec<f32> = (0..n).map(|_| noise.next()).collect();
                match round % 5 {
                    // A weak second half or channel: small angles.
                    1 => y.iter_mut().for_each(|v| *v *= 1e-3),
                    // The same in both: mid only, or equal energies.
                    2 => y.copy_from_slice(&x),
                    // Silent.
                    3 => {
                        x.fill(0.0);
                        y.fill(0.0);
                    }
                    4 => x.iter_mut().for_each(|v| *v *= 1e-4),
                    _ => {}
                }
                let want = unsafe { stereo_itheta(x.as_ptr(), y.as_ptr(), stereo as c_int, n as c_int, 0) };
                let got = bands::stereo_itheta(&x, &y, stereo, n);
                assert_eq!(got, want, "n {n} stereo {stereo} round {round}");
                checked += 1;
            }
        }
    }
    println!("{checked} angles, all equal to libopus's");
}


/// The pulse search gives the pulses of libopus's `op_pvq_search_c`, on
/// the band sizes and pulse counts the encoder uses and on inputs that
/// take every path of the search: the projection for many pulses, the
/// pulse search after it, bands that are silent, tiny or too large, and
/// bands with many equal values.
#[test]
fn pulses_are_searched_as_libopus_does() {
    let mut noise = Noise(31);
    let (mut checked, mut differ) = (0, 0);
    for n in (1usize..=24).chain([32, 36, 44, 48, 64, 72, 88, 96, 144, 176]) {
        for k in [1, 2, 3, 4, 5, 7, 10, 16, 24, 40, 64, 128] {
            for round in 0..12 {
                let mut x: Vec<f32> = (0..n).map(|_| noise.next()).collect();
                match round % 6 {
                    // A few large values among small ones.
                    1 => {
                        x.iter_mut().for_each(|v| *v *= 1e-3);
                        let at = (noise.next().abs() * n as f32) as usize % n;
                        x[at] = 0.9;
                    }
                    // Silent, or so small that libopus puts one pulse at 0.
                    2 => x.fill(0.0),
                    3 => x.iter_mut().for_each(|v| *v *= 1e-18),
                    // So large that the sum reaches 64.
                    4 => x.iter_mut().for_each(|v| *v *= 64.0),
                    // Equal magnitudes.
                    5 => {
                        for (i, v) in x.iter_mut().enumerate() {
                            *v = if i % 2 == 0 { 0.25 } else { -0.25 };
                        }
                    }
                    _ => {}
                }
                let mut xc = x.clone();
                let mut yc = vec![0 as c_int; n];
                unsafe { op_pvq_search_c(xc.as_mut_ptr(), yc.as_mut_ptr(), k, n as c_int, 0) };
                let mut yr = vec![0i32; n];
                opus1::internals::pvq::pvq_search(&x, &mut yr, k, n);
                checked += 1;
                differ += (yc != yr) as usize;
            }
        }
    }
    println!("{differ} of {checked} pulse vectors differ from libopus's");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differ, 0);
    }
}

/// `compute_band_energies` gives the band amplitudes libopus gives, to the
/// bit.
///
/// On 64-bit ARM libopus sums each band with `celt_inner_prod_neon`: one
/// accumulator of four lanes with fused multiply-adds, the lanes added low
/// half plus high half, a fused tail, then `1e-27 + sum`. opus1 summed in
/// four accumulators and started at 1e-27; about one amplitude in five was
/// one step off, enough to change the dynamic allocation and the
/// time-frequency decisions of some frames.
#[test]
fn band_energies_are_the_ones_libopus_computes() {
    let m = default_mode();
    let mut noise = Noise(9);
    let (mut checked, mut differ) = (0, 0);
    for lm in 0..=3usize {
        let n = 120 << lm;
        for round in 0..50 {
            // MDCT values of very different sizes, as in real frames.
            let scale = [1.0f32, 1e-3, 3e2, 1e-6, 7.0][round % 5];
            let x: Vec<f32> = (0..2 * n)
                .map(|i| scale * noise.next() * (1.0 + (i % 7) as f32))
                .collect();
            let mut want = vec![0f32; 2 * 21];
            let mut got = vec![0f32; 2 * 21];
            unsafe { compute_band_energies(c_mode(), x.as_ptr(), want.as_mut_ptr(), 21, 2, lm as c_int, 0) };
            bands::compute_band_energies(m, &x, &mut got, 21, 2, lm);
            for (g, w) in got.iter().zip(&want) {
                checked += 1;
                differ += (g.to_bits() != w.to_bits()) as usize;
            }
        }
    }
    println!("{differ} of {checked} band amplitudes differ from libopus's");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differ, 0);
    }
}

/// Everything `quant_all_bands` needs besides the shapes.
#[derive(Clone, Debug)]
struct Frame {
    channels: usize,
    lm: usize,
    start: usize,
    end: usize,
    short_blocks: bool,
    spread: i32,
    dual_stereo: bool,
    intensity: usize,
    tf_res: Vec<i32>,
    pulses: Vec<i32>,
    total_bits: i32,
    balance: i32,
    coded_bands: i32,
    band_e: Vec<f32>,
    seed: u32,
    complexity: i32,
    bytes: usize,
    disable_inv: bool,
}

/// What one side returns: the coded bytes, the bands after coding (the
/// decoded bands when the encoder resynthesises), the collapse masks and
/// the seed.
#[derive(Debug, PartialEq)]
struct Coded {
    bytes: Vec<u8>,
    x: Vec<f32>,
    masks: Vec<u8>,
    seed: u32,
}

fn run_c(f: &Frame, x_in: &[f32]) -> Coded {
    let m = c_mode();
    let mut x = x_in.to_vec();
    let n = 120 << f.lm;
    let mut buf = vec![0u8; f.bytes];
    let mut ec: EcCtx = unsafe { std::mem::zeroed() };
    let mut masks = vec![0u8; 21 * f.channels];
    let mut pulses = f.pulses.clone();
    let mut tf_res = f.tf_res.clone();
    let mut seed = f.seed;
    unsafe {
        ec_enc_init(&mut ec, buf.as_mut_ptr(), f.bytes as u32);
        let (xp, yp) = if f.channels == 2 {
            let p = x.as_mut_ptr();
            (p, p.add(n))
        } else {
            (x.as_mut_ptr(), std::ptr::null_mut())
        };
        quant_all_bands(
            1,
            m,
            f.start as c_int,
            f.end as c_int,
            xp,
            yp,
            masks.as_mut_ptr(),
            f.band_e.as_ptr(),
            pulses.as_mut_ptr(),
            f.short_blocks as c_int,
            f.spread,
            f.dual_stereo as c_int,
            f.intensity as c_int,
            tf_res.as_mut_ptr(),
            f.total_bits,
            f.balance,
            &mut ec,
            f.lm as c_int,
            f.coded_bands,
            &mut seed,
            f.complexity,
            0,
            f.disable_inv as c_int,
        );
        ec_enc_done(&mut ec);
    }
    assert_eq!(ec.error, 0, "libopus range coder error");
    Coded {
        bytes: buf,
        x,
        masks,
        seed,
    }
}

fn run_rs(f: &Frame, x_in: &[f32]) -> Coded {
    let m = default_mode();
    let mut x = x_in.to_vec();
    let n = 120 << f.lm;
    let mut rc = RangeCoder::new_encoder(f.bytes as u32);
    let mut masks = vec![0u32; 21 * f.channels];
    let mut dual_stereo = f.dual_stereo;
    let mut balance = f.balance;
    let mut seed = f.seed;
    {
        let (xs, ys) = x.split_at_mut(n);
        let y = if f.channels == 2 { Some(ys) } else { None };
        rs_quant_all_bands(
            m,
            f.start,
            f.end,
            xs,
            y,
            &mut masks,
            &f.band_e,
            &f.pulses,
            f.short_blocks,
            f.spread,
            &mut dual_stereo,
            f.intensity,
            &f.tf_res,
            f.total_bits,
            &mut balance,
            &mut rc,
            f.lm as i32,
            f.coded_bands,
            &mut seed,
            f.complexity,
            f.disable_inv,
        );
    }
    rc.done();
    assert_eq!(rc.error, 0, "opus1 range coder error");
    Coded {
        bytes: rc.buf[..f.bytes].to_vec(),
        x,
        masks: masks.iter().map(|&v| v as u8).collect(),
        seed,
    }
}

/// Band shapes of unit norm, as `normalise_bands` leaves them, with band
/// energies to match. Some bands are silent; some are pure tones; in stereo
/// some bands are the same in both channels or nearly opposite.
fn shapes(noise: &mut Noise, channels: usize, lm: usize, frame: usize) -> (Vec<f32>, Vec<f32>) {
    let m = default_mode();
    let n = 120 << lm;
    let mut x = vec![0f32; n * channels];
    let mut band_e = vec![0f32; 21 * channels];
    for i in 0..21 {
        let lo = (m.e_bands[i] as usize) << lm;
        let hi = (m.e_bands[i + 1] as usize) << lm;
        let kind = (noise.next().abs() * 8.0) as usize + frame;
        for c in 0..channels {
            let band = &mut x[c * n + lo..c * n + hi];
            match kind % 8 {
                // Silent: `normalise_bands` divides by 1e-27 plus the energy.
                0 => band.fill(0.0),
                // A tone.
                1 => {
                    let at = (noise.next().abs() * band.len() as f32) as usize % band.len();
                    band[at] = 1.0 + 0.1 * noise.next();
                    for v in band.iter_mut() {
                        *v += 0.01 * noise.next();
                    }
                }
                _ => {
                    for v in band.iter_mut() {
                        *v = noise.next();
                    }
                }
            }
            if c == 1 && kind % 8 == 2 {
                // The same as the left channel.
                let (l, r) = x.split_at_mut(n);
                r[lo..hi].copy_from_slice(&l[lo..hi]);
            } else if c == 1 && kind % 8 == 3 {
                // Nearly the opposite of the left channel.
                let (l, r) = x.split_at_mut(n);
                for j in lo..hi {
                    r[j] = -l[j] + 0.05 * noise.next();
                }
            }
            let band = &mut x[c * n + lo..c * n + hi];
            let e: f32 = band.iter().map(|v| v * v).sum::<f32>().sqrt();
            let g = 1.0 / (1e-27 + e);
            for v in band.iter_mut() {
                *v *= g;
            }
            band_e[c * 21 + i] = if e > 0.0 {
                (0.5 + noise.next().abs()) * 1000.0
            } else {
                0.0
            };
        }
    }
    (x, band_e)
}

/// An allocation as the encoder computes it, for `bytes` bytes per frame.
/// `stereo` is the encoder's choice of the intensity band and of dual
/// stereo; `clt_compute_allocation` may lower the intensity band.
#[allow(clippy::too_many_arguments)]
fn frame_settings(
    channels: usize,
    lm: usize,
    bytes: usize,
    short_blocks: bool,
    spread: i32,
    complexity: i32,
    tf: &[i32],
    band_e: Vec<f32>,
    seed: u32,
    stereo: (i32, bool),
) -> Frame {
    frame_settings_in(channels, lm, bytes, short_blocks, spread, complexity, tf, band_e, seed, stereo, 0, 21)
}

/// `frame_settings` for the bands `start..end`.
#[allow(clippy::too_many_arguments)]
fn frame_settings_in(
    channels: usize,
    lm: usize,
    bytes: usize,
    short_blocks: bool,
    spread: i32,
    complexity: i32,
    tf: &[i32],
    band_e: Vec<f32>,
    seed: u32,
    stereo: (i32, bool),
    start: usize,
    end: usize,
) -> Frame {
    let m: &CeltMode = default_mode();
    let mut cap = vec![0i32; 21];
    for (i, c) in cap.iter_mut().enumerate() {
        let n = (m.e_bands[i + 1] - m.e_bands[i]) as i32;
        let caps = m.cache.caps[21 * (2 * lm + channels - 1) + i] as i32;
        *c = ((caps + 64) * channels as i32 * n) >> 2;
    }
    let offsets = vec![0i32; 21];
    let mut intensity = stereo.0;
    let mut dual_stereo = stereo.1 as i32;
    let mut balance = 0i32;
    let mut pulses = vec![0i32; 21];
    let mut ebits = vec![0i32; 21 * channels];
    let mut fine_priority = vec![0i32; 21 * channels];
    let mut rc = RangeCoder::new_encoder(bytes as u32);
    // Some bits for the header and the energies, as in a real frame.
    let used = (bytes as i32 * 8) / 6;
    let total = ((bytes as i32 * 8) << 3) - (used << 3) - 1;
    let coded_bands = clt_compute_allocation(
        m,
        start,
        end,
        &offsets,
        &cap,
        5,
        &mut intensity,
        &mut dual_stereo,
        total,
        &mut balance,
        &mut pulses,
        &mut ebits,
        &mut fine_priority,
        channels as i32,
        lm as i32,
        &mut rc,
        true,
        0,
        20,
    );
    Frame {
        channels,
        lm,
        start,
        end,
        short_blocks,
        spread,
        dual_stereo: dual_stereo != 0,
        intensity: intensity as usize,
        tf_res: tf.to_vec(),
        pulses,
        total_bits: total,
        balance,
        coded_bands,
        band_e,
        seed,
        complexity,
        bytes,
        disable_inv: false,
    }
}

/// The time-frequency changes of frame `k`: values libopus can choose for
/// this frame size, transient or not (`tf_select_table`, celt/celt.c).
fn tf_choice(lm: usize, short_blocks: bool, k: usize) -> Vec<i32> {
    const TF_SELECT: [[i32; 8]; 4] = [
        [0, -1, 0, -1, 0, -1, 0, -1],
        [0, -1, 0, -2, 1, 0, 1, -1],
        [0, -2, 0, -3, 2, 0, 1, -1],
        [0, -2, 0, -3, 3, 0, 1, -1],
    ];
    let tf_select = k / 2 % 2;
    (0..21)
        .map(|i| {
            let bit = (i * 7 + k * 3).is_multiple_of(5) as usize;
            TF_SELECT[lm][4 * short_blocks as usize + 2 * tf_select + bit]
        })
        .collect()
}

/// The intensity band and dual stereo flag of frame `k`: mostly every band
/// coded as mid and side, sometimes intensity stereo from a band on, now and
/// then dual stereo.
fn stereo_choice(k: usize) -> (i32, bool) {
    ([21, 21, 21, 17, 12, 6, 0][k % 7], k % 5 == 3)
}

/// How many of `frames` random frames give the same bytes on both sides,
/// and the first frame that does not.
/// With `compare_decoded` the bands that `quant_all_bands` leaves must be
/// the same too; they are the decoded bands when the encoder resynthesises.
fn agreement(
    channels: usize,
    lm: usize,
    bytes: usize,
    complexity: i32,
    frames: usize,
    seed: u32,
    compare_decoded: bool,
) -> (usize, Option<String>) {
    let mut noise = Noise(seed);
    let mut same = 0;
    let mut first = None;
    for k in 0..frames {
        let (x, band_e) = shapes(&mut noise, channels, lm, k);
        let short_blocks = lm > 0 && k % 3 == 1;
        let spread = [2, 1, 0, 3][k % 4];
        let tf = tf_choice(lm, short_blocks, k);
        let f = frame_settings(
            channels,
            lm,
            bytes,
            short_blocks,
            spread,
            complexity,
            &tf,
            band_e,
            noise.0,
            stereo_choice(k),
        );
        let c = run_c(&f, &x);
        let r = run_rs(&f, &x);
        // Only an encoder that resynthesises (theta_rdo) leaves decoded bands.
        let resynth = channels == 2 && complexity >= 8 && !f.dual_stereo;
        let same_x = !(compare_decoded && resynth) || same_bits(&c.x, &r.x);
        if c.bytes == r.bytes && c.masks == r.masks && c.seed == r.seed && same_x {
            same += 1;
        } else if first.is_none() {
            let byte = c.bytes.iter().zip(&r.bytes).position(|(a, b)| a != b);
            first = Some(format!(
                "frame {k}: first differing byte {byte:?}, masks {:?} and {:?}, seeds {} and {}, \
                 short blocks, spread, dual stereo, intensity, coded bands {:?}",
                c.masks,
                r.masks,
                c.seed,
                r.seed,
                (f.short_blocks, f.spread, f.dual_stereo, f.intensity, f.coded_bands)
            ));
        }
    }
    (same, first)
}

/// `quant_all_bands` of a mono frame writes the bytes libopus writes, with
/// the same collapse masks and the same noise seed, for every frame size.
///
/// The inputs are random band shapes of unit norm (some silent, some
/// tones) with an allocation from `clt_compute_allocation`, and the
/// time-frequency changes and spreading decisions libopus can choose.
/// Without resynthesis libopus leaves the collapse mask of a band without
/// pulses at 0; opus1 gave it the mask of a filled band.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn mono_bands_are_coded_as_libopus_codes_them() {
    for (lm, bytes) in [(3usize, 160usize), (3, 60), (2, 80), (1, 40), (0, 30)] {
        let (same, first) = agreement(1, lm, bytes, 9, 200, 11 + lm as u32, false);
        println!("mono LM {lm}, {bytes} bytes: {same} of 200 frames equal; {first:?}");
        assert_eq!(same, 200, "{first:?}");
    }
}


/// `quant_all_bands` of a stereo frame writes the bytes libopus writes, with
/// the same collapse masks and the same noise seed, at complexity 5 and at
/// complexity 9.
///
/// At complexity 8 and above libopus codes each stereo band below the
/// intensity band twice, with the angle rounded down and up, and keeps the
/// result with the smaller distortion (theta_rdo); for that it
/// resynthesises every band, and the decoded bands must be the same too.
/// The frames mix bands coded as mid and side, intensity stereo and dual
/// stereo (`stereo_choice`).
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn stereo_bands_are_coded_as_libopus_codes_them() {
    for complexity in [5, 9] {
        for (lm, bytes) in [(3usize, 160usize), (3, 60), (3, 400), (2, 80), (1, 40), (0, 30)] {
            let (same, first) = agreement(2, lm, bytes, complexity, 200, 21 + lm as u32, true);
            println!("stereo c{complexity} LM {lm}, {bytes} bytes: {same} of 200 frames equal; {first:?}");
            assert_eq!(same, 200, "{first:?}");
        }
    }
}

/// `quant_all_bands` writes libopus's bytes, collapse masks, noise seed and
/// decoded bands for frames that do not start at band 0 or end at band 21:
/// hybrid frames (start 17, where the band after the first one folds from a
/// copy, `special_hybrid_folding`) and narrower bandwidths (end 13, 15, 17,
/// 19, 20). Also with phase inversion disabled, and with bands whose energy
/// in one channel is below `MIN_STEREO_ENERGY`.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn bands_of_hybrid_and_narrow_frames_are_coded_as_libopus_codes_them() {
    let mut noise = Noise(77);
    let mut frames = 0;
    for channels in [1usize, 2] {
        for complexity in [5, 9, 10] {
            for (start, end) in [(17usize, 21usize), (17, 19), (17, 20), (0, 13), (0, 15), (0, 17), (0, 19), (0, 20)] {
                for (lm, bytes) in [(3usize, 160usize), (3, 50), (2, 70), (1, 40), (0, 25)] {
                    for k in 0..12 {
                        let (x, mut band_e) = shapes(&mut noise, channels, lm, k);
                        if k % 4 == 2 {
                            // Some bands nearly silent in one channel.
                            for i in (k % 3..21).step_by(3) {
                                band_e[(channels - 1) * 21 + i] = 1e-12;
                            }
                        }
                        let short_blocks = lm > 0 && k % 3 == 1;
                        let tf = tf_choice(lm, short_blocks, k);
                        let mut f = frame_settings_in(
                            channels,
                            lm,
                            bytes,
                            short_blocks,
                            [2, 1, 0, 3][k % 4],
                            complexity,
                            &tf,
                            band_e,
                            noise.0,
                            stereo_choice(k),
                            start,
                            end,
                        );
                        f.disable_inv = k % 2 == 1;
                        let c = run_c(&f, &x);
                        let r = run_rs(&f, &x);
                        let what = format!(
                            "C {channels} c{complexity} bands {start}..{end} LM {lm} {bytes} bytes frame {k} \
                             (short {short_blocks}, dual {}, intensity {}, inv off {})",
                            f.dual_stereo, f.intensity, f.disable_inv
                        );
                        assert_eq!(c.bytes, r.bytes, "{what}: bytes");
                        assert_eq!(c.masks, r.masks, "{what}: collapse masks");
                        assert_eq!(c.seed, r.seed, "{what}: seed");
                        let resynth = channels == 2 && complexity >= 8 && !f.dual_stereo;
                        if resynth {
                            assert!(same_bits(&c.x, &r.x), "{what}: decoded bands");
                        }
                        frames += 1;
                    }
                }
            }
        }
    }
    println!("{frames} frames, all equal to libopus's");
}

/// Stereo bands whose two channels are scaled copies of each other, at very
/// low levels. This is what the DC filter of the Opus layer leaves after a
/// stereo signal stops: each channel decays at the same rate, so the right
/// channel is the left one times a constant.
///
/// The normalized shapes of the two channels are then the same, the angle
/// of every band is 0, and libopus mixes the channels with
/// `intensity_stereo`. Its weights add `EPSILON` (1e-15) under the square
/// root of the energies and to the root. At band energies of 1e-6 and
/// below this changes the weights, and so the pulses.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn stereo_bands_that_are_scaled_copies_at_low_levels_are_coded_as_libopus_codes_them() {
    let m = default_mode();
    let mut noise = Noise(91);
    let mut frames = 0;
    for complexity in [5, 9] {
        for (lm, bytes) in [(3usize, 1200usize), (3, 160), (3, 50), (2, 80), (1, 40), (0, 25)] {
            for k in 0..60 {
                let n = 120 << lm;
                let mut x = vec![0f32; 2 * n];
                let mut band_e = vec![0f32; 42];
                // The level of the frame: band energies from 1e-4 down to
                // 1e-9, the range of the decaying output of the DC filter.
                let level = [1e-4f32, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9][k % 6];
                let ratio = 0.2 + 1.5 * noise.next().abs();
                for i in 0..21 {
                    let lo = (m.e_bands[i] as usize) << lm;
                    let hi = (m.e_bands[i + 1] as usize) << lm;
                    let band = &mut x[lo..hi];
                    for v in band.iter_mut() {
                        *v = noise.next();
                    }
                    let e: f32 = band.iter().map(|v| v * v).sum::<f32>().sqrt();
                    for v in band.iter_mut() {
                        *v /= 1e-27 + e;
                    }
                    let (left, right) = x.split_at_mut(n);
                    right[lo..hi].copy_from_slice(&left[lo..hi]);
                    // Higher bands are quieter, as for a decaying DC offset.
                    let e = level * (0.5 + noise.next().abs()) / (1 + i * i) as f32;
                    band_e[i] = e;
                    band_e[21 + i] = ratio * e;
                }
                let short_blocks = lm > 0 && k % 3 == 1;
                let tf = tf_choice(lm, short_blocks, k);
                let f = frame_settings(
                    2,
                    lm,
                    bytes,
                    short_blocks,
                    [2, 1, 0, 3][k % 4],
                    complexity,
                    &tf,
                    band_e,
                    noise.0,
                    stereo_choice(k),
                );
                let c = run_c(&f, &x);
                let r = run_rs(&f, &x);
                let what = format!(
                    "c{complexity} LM {lm} {bytes} bytes frame {k}, level {level:e} \
                     (dual {}, intensity {})",
                    f.dual_stereo, f.intensity
                );
                assert_eq!(c.bytes, r.bytes, "{what}: bytes");
                assert_eq!(c.masks, r.masks, "{what}: collapse masks");
                assert_eq!(c.seed, r.seed, "{what}: seed");
                if complexity >= 8 && !f.dual_stereo {
                    assert!(same_bits(&c.x, &r.x), "{what}: decoded bands");
                }
                frames += 1;
            }
        }
    }
    println!("{frames} frames, all equal to libopus's");
}

/// Decodes `bytes` with libopus's `quant_all_bands` and with opus1's, for
/// the frame `f`. Returns the decoded bands, collapse masks and seeds.
fn decode_both(f: &Frame, bytes: &[u8]) -> (Coded, Coded) {
    let n = 120 << f.lm;
    let mut buf = bytes.to_vec();
    // libopus.
    let mut xc = vec![0f32; n * f.channels];
    let mut masks_c = vec![0u8; 21 * f.channels];
    let mut seed_c = f.seed;
    let mut pulses = f.pulses.clone();
    let mut tf_res = f.tf_res.clone();
    unsafe {
        let mut dec: EcCtx = std::mem::zeroed();
        ec_dec_init(&mut dec, buf.as_mut_ptr(), buf.len() as u32);
        let (xp, yp) = if f.channels == 2 {
            let p = xc.as_mut_ptr();
            (p, p.add(n))
        } else {
            (xc.as_mut_ptr(), std::ptr::null_mut())
        };
        quant_all_bands(
            0,
            c_mode(),
            f.start as c_int,
            f.end as c_int,
            xp,
            yp,
            masks_c.as_mut_ptr(),
            f.band_e.as_ptr(),
            pulses.as_mut_ptr(),
            f.short_blocks as c_int,
            f.spread,
            f.dual_stereo as c_int,
            f.intensity as c_int,
            tf_res.as_mut_ptr(),
            f.total_bits,
            f.balance,
            &mut dec,
            f.lm as c_int,
            f.coded_bands,
            &mut seed_c,
            0,
            0,
            0,
        );
    }
    // opus1.
    let mut xr = vec![0f32; n * f.channels];
    let mut masks_r = vec![0u32; 21 * f.channels];
    let mut seed_r = f.seed;
    let mut dual = f.dual_stereo;
    let mut balance = f.balance;
    let mut rd = RangeCoder::new_decoder(bytes);
    {
        let (xs, ys) = xr.split_at_mut(n);
        let y = if f.channels == 2 { Some(ys) } else { None };
        bands::quant_all_bands(
            false,
            default_mode(),
            f.start,
            f.end,
            xs,
            y,
            &mut masks_r,
            &f.band_e,
            &f.pulses,
            f.short_blocks,
            f.spread,
            &mut dual,
            f.intensity,
            &f.tf_res,
            f.total_bits,
            &mut balance,
            &mut rd,
            f.lm as i32,
            f.coded_bands,
            true,
            false,
            &mut seed_r,
        );
    }
    (
        Coded {
            bytes: Vec::new(),
            x: xc,
            masks: masks_c,
            seed: seed_c,
        },
        Coded {
            bytes: Vec::new(),
            x: xr,
            masks: masks_r.iter().map(|&v| v as u8).collect(),
            seed: seed_r,
        },
    )
}

/// The decoder's `quant_all_bands` gives libopus's decoded bands, collapse
/// masks and noise seed, mono and stereo.
///
/// libopus merges mid and side (stereo_merge) with energies from
/// `celt_inner_prod` (on arm64 the NEON form) and `mid * mid + side` fused;
/// opus1 summed one value at a time without fusing.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn bands_are_decoded_as_libopus_decodes_them() {
    for channels in [1usize, 2] {
        for (lm, bytes) in [(3usize, 160usize), (3, 60), (2, 80), (1, 40), (0, 30)] {
            let mut noise = Noise(31 + lm as u32);
            let mut same = 0;
            for k in 0..200 {
                let (x, band_e) = shapes(&mut noise, channels, lm, k);
                let short_blocks = lm > 0 && k % 3 == 1;
                let tf = tf_choice(lm, short_blocks, k);
                let f = frame_settings(
                    channels,
                    lm,
                    bytes,
                    short_blocks,
                    [2, 1, 0, 3][k % 4],
                    5,
                    &tf,
                    band_e,
                    noise.0,
                    stereo_choice(k),
                );
                let coded = run_c(&f, &x);
                let (c, r) = decode_both(&f, &coded.bytes);
                let ok = same_bits(&c.x, &r.x) && c.masks == r.masks && c.seed == r.seed;
                same += ok as usize;
            }
            println!("{channels} channels, LM {lm}, {bytes} bytes: {same} of 200 frames decoded alike");
            assert_eq!(same, 200);
        }
    }
}

/// `anti_collapse` fills collapsed blocks with the noise of libopus, at the
/// level of libopus, and renormalises as libopus does, to the bit.
///
/// The level is `min(0.5 * 2^(-depth / 8), 2 * 2^(-Ediff))` with libopus's
/// `celt_exp2`: in the float build with `FLOAT_APPROX` a polynomial whose
/// steps clang fuses on arm64. opus1 used the exact `exp2` of the
/// standard library.
#[test]
fn anti_collapse_is_the_one_of_libopus() {
    let m = default_mode();
    let mut noise = Noise(41);
    let (mut checked, mut differ) = (0, 0);
    for lm in 0..=3usize {
        for channels in [1usize, 2] {
            for round in 0..100 {
                let n = 120 << lm;
                let mut x: Vec<f32> = (0..n * channels).map(|_| noise.next()).collect();
                let masks: Vec<u8> =
                    (0..21 * channels).map(|_| (noise.0 >> 7) as u8 & noise.next().to_bits() as u8).collect();
                let log_e: Vec<f32> = (0..42).map(|_| 8.0 * noise.next()).collect();
                let prev1: Vec<f32> = (0..42).map(|_| 8.0 * noise.next()).collect();
                let prev2: Vec<f32> = (0..42).map(|_| 8.0 * noise.next()).collect();
                let pulses: Vec<i32> = (0..21).map(|_| (noise.next().abs() * 400.0) as i32).collect();
                let seed = noise.0 ^ round;
                let mut xc = x.clone();
                let mut mc = masks.clone();
                unsafe {
                    anti_collapse(
                        c_mode(),
                        xc.as_mut_ptr(),
                        mc.as_mut_ptr(),
                        lm as c_int,
                        channels as c_int,
                        n as c_int,
                        0,
                        21,
                        log_e.as_ptr(),
                        prev1.as_ptr(),
                        prev2.as_ptr(),
                        pulses.as_ptr(),
                        seed,
                        0,
                        0,
                    )
                };
                let mr: Vec<u32> = masks.iter().map(|&v| v as u32).collect();
                bands::anti_collapse(
                    m, &mut x, &mr, lm as i32, channels, n, 0, 21, &log_e, &prev1, &prev2, &pulses, seed,
                );
                checked += 1;
                differ += !same_bits(&xc, &x) as usize;
            }
        }
    }
    println!("{differ} of {checked} frames differ from libopus's");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differ, 0);
    }
}

/// `denormalise_bands` scales the decoded bands by the gains libopus
/// computes, to the bit.
///
/// The gain is `celt_exp2(min(32, log energy + mean))`, with the
/// polynomial `celt_exp2` of the float build with `FLOAT_APPROX`. opus1
/// took `2^x` from `powf`.
#[test]
fn bands_are_denormalised_as_libopus_does() {
    let m = default_mode();
    let mut noise = Noise(43);
    let (mut checked, mut differ) = (0, 0);
    for lm in 0..=3usize {
        for _ in 0..100 {
            let n = 120 << lm;
            let x: Vec<f32> = (0..n).map(|_| noise.next()).collect();
            let log_e: Vec<f32> = (0..21).map(|_| 12.0 * noise.next()).collect();
            let mut want = vec![0f32; n];
            unsafe {
                denormalise_bands(
                    c_mode(),
                    x.as_ptr(),
                    want.as_mut_ptr(),
                    log_e.as_ptr(),
                    0,
                    21,
                    1 << lm,
                    1,
                    0,
                )
            };
            let mut amp = vec![0f32; 21];
            bands::log2amp(m, 21, &mut amp, &log_e, 1);
            let mut got = vec![0f32; n];
            bands::denormalise_bands(m, &x, &mut got, &amp, 0, 21, 1, 1 << lm);
            // libopus also clears the bins above the last band.
            let bound = (m.e_bands[21] as usize) << lm;
            checked += 1;
            differ += !same_bits(&want[..bound], &got[..bound]) as usize;
        }
    }
    println!("{differ} of {checked} frames differ from libopus's");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differ, 0);
    }
}

/// The number of pulse vectors of `n` values and `k` pulses fits in 32
/// bits, as for every band that CELT codes.
fn codable(n: usize, k: usize) -> bool {
    // V(n, k) = V(n - 1, k) + V(n, k - 1) + V(n - 1, k - 1).
    let mut v = vec![vec![0u128; k + 1]; n + 1];
    for row in v.iter_mut() {
        row[0] = 1;
    }
    for i in 1..=n {
        for j in 1..=k {
            v[i][j] = (v[i - 1][j] + v[i][j - 1] + v[i - 1][j - 1]).min(1 << 40);
        }
    }
    v[n][k] < 1 << 32
}

/// A band of `n` random values with unit norm.
fn unit_band(noise: &mut Noise, n: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..n).map(|_| noise.next()).collect();
    let e: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    x.iter_mut().for_each(|v| *v /= e);
    x
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// `exp_rotation` spreads and gathers a band as libopus does, to the bit,
/// in both directions.
///
/// libopus computes the rotation angle with `cos` in double precision for
/// `theta` and for `1 - theta` (celt_cos_norm), and clang fuses the products
/// with `c` into the adds of each step on 64-bit ARM. opus1 used `cos` and
/// `sin` in single precision, and fused other products in some passes.
#[test]
fn spreading_rotation_is_the_one_of_libopus() {
    let mut noise = Noise(5);
    let (mut checked, mut differ) = (0, 0);
    for n in [3usize, 4, 6, 8, 12, 16, 22, 24, 32, 36, 48, 64, 72, 88, 96, 144, 176] {
        for k in [1, 2, 3, 5, 8, 13] {
            for spread in 1..4 {
                for b in [1usize, 2, 4, 8] {
                    if n % b != 0 {
                        continue;
                    }
                    for dir in [1, -1] {
                        let x = unit_band(&mut noise, n);
                        let (mut xc, mut xr) = (x.clone(), x.clone());
                        unsafe {
                            exp_rotation(xc.as_mut_ptr(), n as c_int, dir, b as c_int, k, spread)
                        };
                        opus1::internals::pvq::exp_rotation(&mut xr, n, dir, b, k, spread);
                        checked += 1;
                        differ += !same_bits(&xc, &xr) as usize;
                    }
                }
            }
        }
    }
    println!("{differ} of {checked} rotated bands differ from libopus's");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differ, 0);
    }
}

/// `alg_quant` with resynthesis and `alg_unquant` give libopus's bytes,
/// collapse masks and bands, to the bit.
///
/// libopus scales the pulses by `(1 / sqrt(Ryy)) * gain`
/// (normalise_residual); opus1 divided the gain by the square root, which
/// rounds differently when the gain is not 1.
#[test]
fn pulses_are_resynthesised_as_libopus_does() {
    let mut noise = Noise(6);
    let (mut checked, mut differ) = (0, 0);
    for n in [2usize, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 176] {
        for k in [1usize, 2, 3, 5, 8, 13, 20, 40] {
            if !codable(n, k) {
                continue;
            }
            for spread in 0..4 {
                for b in [1usize, 2, 4, 8] {
                    if n % b != 0 {
                        continue;
                    }
                    let x = unit_band(&mut noise, n);
                    let gain = [1.0f32, 0.75, 0.3, 0.9][spread as usize];
                    let k = k as i32;
                    // libopus: encode with resynthesis, then decode.
                    let mut buf = vec![0u8; 200];
                    let mut ec: EcCtx = unsafe { std::mem::zeroed() };
                    let mut xc = x.clone();
                    let cm_c = unsafe {
                        ec_enc_init(&mut ec, buf.as_mut_ptr(), 200);
                        let cm =
                            alg_quant(xc.as_mut_ptr(), n as c_int, k, spread, b as c_int, &mut ec, gain, 1, 0);
                        ec_enc_done(&mut ec);
                        cm
                    };
                    let mut dc: EcCtx = unsafe { std::mem::zeroed() };
                    let mut yc = vec![0f32; n];
                    let dm_c = unsafe {
                        ec_dec_init(&mut dc, buf.as_mut_ptr(), 200);
                        alg_unquant(yc.as_mut_ptr(), n as c_int, k, spread, b as c_int, &mut dc, gain)
                    };
                    // opus1: the same.
                    let mut rc = RangeCoder::new_encoder(200);
                    let mut xr = x.clone();
                    let cm_r = opus1::internals::pvq::alg_quant(&mut xr, n, k, spread, b, &mut rc, gain, true);
                    rc.done();
                    let mut rd = RangeCoder::new_decoder(&rc.buf[..200]);
                    let mut yr = vec![0f32; n];
                    let dm_r = opus1::internals::pvq::alg_unquant(&mut yr, n, k, spread, b, &mut rd, gain);
                    checked += 1;
                    let ok = cm_c == cm_r
                        && dm_c == dm_r
                        && buf[..] == rc.buf[..200]
                        && same_bits(&xc, &xr)
                        && same_bits(&yc, &yr);
                    differ += !ok as usize;
                }
            }
        }
    }
    println!("{differ} of {checked} bands differ from libopus's");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differ, 0);
    }
}

/// Writes the signals of `celtenc_gap_report` as raw interleaved floats,
/// for an instrumented libopus build: `QAB_SIGNALS=dir`.
#[test]
#[ignore = "tool"]
fn write_signals() {
    use celtenc_common::{music, speech, stereo_of};
    let Ok(dir) = std::env::var("QAB_SIGNALS") else {
        return;
    };
    for fs in [48_000usize, 16_000] {
        let s = speech(fs, 6);
        let m0 = music(fs, 6, 0);
        let lag = 12 * fs / 48_000;
        let late: Vec<f32> = (0..s.len()).map(|i| if i >= lag { 0.6 * s[i - lag] } else { 0.0 }).collect();
        let m1 = music(fs, 6, 1);
        let sigs: Vec<(&str, usize, Vec<f32>)> = vec![
            ("speech", 1, s.clone()),
            ("music", 1, m0.clone()),
            ("speech", 2, stereo_of(&s, &late, 0.0)),
            ("music", 2, stereo_of(&m0, &m1, 0.3)),
        ];
        for (name, ch, pcm) in sigs {
            let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/{name}_{fs}_{ch}.f32"), bytes).unwrap();
        }
    }
}

/// Reads the records of an instrumented libopus build.
struct Reader {
    data: Vec<u8>,
    at: usize,
}

impl Reader {
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let s = self.data[self.at..self.at + n].to_vec();
        self.at += n;
        s
    }
    fn ints(&mut self, n: usize) -> Vec<i32> {
        self.bytes(4 * n).chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        self.bytes(4 * n).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    /// A range coder: its eleven words, then its buffer.
    fn coder(&mut self) -> RangeCoder {
        let w = self.ints(11);
        let mut rc = RangeCoder::new_encoder(w[0] as u32);
        let buf = self.bytes(w[0] as usize);
        rc.buf[..w[0] as usize].copy_from_slice(&buf);
        rc.storage = w[0] as u32;
        rc.end_offs = w[1] as u32;
        rc.end_window = w[2] as u32;
        rc.nend_bits = w[3];
        rc.nbits_total = w[4];
        rc.offs = w[5] as u32;
        rc.rng = w[6] as u32;
        rc.val = w[7] as u32;
        rc.ext = w[8] as u32;
        rc.rem = w[9];
        rc.error = w[10];
        rc
    }
}

/// The words and the bytes of a range coder: the bytes written so far from
/// the start (up to `offs`) and from the end (`end_offs`). The bytes in
/// between are not part of the result: the coder writes them again later or
/// clears them when it finishes.
fn coder_state(rc: &RangeCoder) -> (Vec<i64>, Vec<u8>) {
    (
        vec![
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
        ],
        [
            &rc.buf[..rc.offs as usize],
            &rc.buf[(rc.storage - rc.end_offs) as usize..rc.storage as usize],
        ]
        .concat(),
    )
}

/// Replays the calls of `quant_all_bands` that an instrumented libopus
/// encoder wrote to `QAB_REPLAY` (inputs, range coder before and after),
/// through opus1, and counts the calls with the same result.
#[test]
#[ignore = "tool"]
fn replay_libopus_calls() {
    let Ok(path) = std::env::var("QAB_REPLAY") else {
        return;
    };
    let mut r = Reader {
        data: std::fs::read(&path).unwrap(),
        at: 0,
    };
    let (mut calls, mut same, mut shown) = (0, 0, 0);
    let m = default_mode();
    while r.at < r.data.len() {
        assert_eq!(r.bytes(4), b"QAB1");
        let h = r.ints(16);
        let (start, end, c, lm) = (h[0] as usize, h[1] as usize, h[2] as usize, h[3]);
        let n = h[14] as usize;
        let nb = h[15] as usize;
        let tf = r.ints(nb);
        let pulses = r.ints(nb);
        let band_e = r.floats(c * nb);
        let mut x = r.floats(c * n);
        let mut rc = r.coder();
        let rc_c = r.coder();
        let masks_c = r.bytes(c * nb);
        let seed_c = r.ints(1)[0] as u32;

        let mut masks = vec![0u32; c * nb];
        let mut dual = h[6] != 0;
        let mut balance = h[9];
        let mut seed = h[11] as u32;
        {
            let (xs, ys) = x.split_at_mut(n);
            let y = if c == 2 { Some(ys) } else { None };
            rs_quant_all_bands(
                m, start, end, xs, y, &mut masks, &band_e, &pulses, h[4] != 0, h[5], &mut dual,
                h[7] as usize, &tf, h[8], &mut balance, &mut rc, lm, h[10], &mut seed, h[12],
                h[13] != 0,
            );
        }
        let masks_rs: Vec<u8> = masks.iter().map(|&v| v as u8).collect();
        let (state_c, bytes_c) = coder_state(&rc_c);
        let (state_rs, bytes_rs) = coder_state(&rc);
        let ok = state_c == state_rs && bytes_c == bytes_rs && masks_rs == masks_c && seed == seed_c;
        calls += 1;
        if ok {
            same += 1;
        } else if shown < 3 {
            shown += 1;
            let byte = bytes_c.iter().zip(&bytes_rs).position(|(a, b)| a != b);
            println!(
                "call {}: LM {lm} C {c} short {} spread {} intensity {} dual {} cx {}: state equal {}, \
                 first byte {byte:?}, masks equal {}, seed equal {}",
                calls - 1,
                h[4],
                h[5],
                h[7],
                h[6],
                h[12],
                state_c == state_rs,
                masks_rs == masks_c,
                seed == seed_c
            );
        }
    }
    println!("{same} of {calls} calls of quant_all_bands give libopus's result");
}

/// opus1's `quant_all_bands` as the encoder calls it.
#[allow(clippy::too_many_arguments)]
fn rs_quant_all_bands(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    seed: &mut u32,
    complexity: i32,
    disable_inv: bool,
) {
    bands::quant_all_bands_enc(
        m,
        start,
        end,
        x,
        y,
        masks,
        band_e,
        pulses,
        short_blocks,
        spread,
        dual_stereo,
        intensity,
        tf_res,
        total_bits,
        balance,
        rc,
        lm,
        coded_bands,
        seed,
        complexity,
        disable_inv,
    );
}

