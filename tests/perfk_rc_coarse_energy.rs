//! The coarse energy encoder (`quant_coarse_energy_advanced`,
//! src/quant_bands.rs) against `quant_coarse_energy` of libopus
//! (celt/quant_bands.c), called directly from the static library.
//!
//! Both encoders get the same band energies, the same previous energies and
//! a range coder in the same state. Afterwards the bytes and words of the
//! two coders, the quantized energies and the quantization errors must be
//! equal bit for bit, and the state for the next frame (`delayed_intra`)
//! equal to within rounding.
//!
//! With two passes the encoder codes the energies twice, with and without
//! prediction from the previous frame, and keeps the pass whose values are
//! closer to the wanted ones (`badness`). The inputs here make the energy
//! drop fast in many bands, which is where the limit on the decay applies
//! and where the count of `badness` matters.

use opus1::modes::default_mode;
use opus1::quant_bands::quant_coarse_energy_advanced;
use opus1::range_coder::RangeCoder;
use std::os::raw::{c_int, c_uint, c_void};

/// `ec_ctx` of libopus (celt/entcode.h).
#[repr(C)]
struct EcCtx {
    buf: *mut u8,
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
    fn ec_enc_init(this: *mut EcCtx, buf: *mut u8, size: u32);
    fn ec_enc_bit_logp(this: *mut EcCtx, val: c_int, logp: c_uint);
    #[link_name = "quant_coarse_energy"]
    fn c_quant_coarse_energy(
        m: *const c_void,
        start: c_int,
        end: c_int,
        eff_end: c_int,
        e_bands: *const f32,
        old_e_bands: *mut f32,
        budget: u32,
        error: *mut f32,
        enc: *mut EcCtx,
        channels: c_int,
        lm: c_int,
        nb_available_bytes: c_int,
        force_intra: c_int,
        delayed_intra: *mut f32,
        two_pass: c_int,
        loss_rate: c_int,
        lfe: c_int,
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

/// Small deterministic pseudo-random generator.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        ((self.next() as u64 * n as u64) >> 32) as u32
    }
    /// Uniform in [lo, hi), a multiple of 1/64: exact in `f32`.
    fn level(&mut self, lo: i32, hi: i32) -> f32 {
        lo as f32 + self.below(((hi - lo) * 64) as u32) as f32 / 64.0
    }
}

struct Case {
    start: usize,
    end: usize,
    eff_end: usize,
    channels: usize,
    lm: usize,
    size: u32,
    budget: u32,
    nb_available_bytes: usize,
    force_intra: bool,
    two_pass: bool,
    loss_rate: i32,
    lfe: bool,
    delayed_intra: f32,
    header_bits: Vec<(bool, u32)>,
    e_bands: Vec<f32>,
    old_e_bands: Vec<f32>,
}

const NB_EBANDS: usize = 21;

fn random_case(rng: &mut Lcg, two_pass: bool) -> Case {
    let channels = 1 + rng.below(2) as usize;
    let (start, end) = match rng.below(4) {
        0 => (17, 21),
        1 => (0, 13 + rng.below(9) as usize),
        _ => (0, 21),
    };
    let size = [8u32, 20, 40, 80, 160, 320, 1275][rng.below(7) as usize];
    let old_e_bands: Vec<f32> = (0..channels * NB_EBANDS).map(|_| rng.level(-30, 20)).collect();
    // Half of the cases: the level of the previous frame with small
    // changes and, in every third band, a drop of up to 40 (240 dB). The
    // other half: unrelated levels.
    let related = rng.below(2) == 0;
    let e_bands: Vec<f32> = old_e_bands
        .iter()
        .map(|&old| {
            if !related {
                rng.level(-25, 20)
            } else if rng.below(3) == 0 {
                old - rng.level(2, 40)
            } else {
                old + rng.level(-3, 3)
            }
        })
        .collect();
    Case {
        start,
        end,
        eff_end: end - rng.below(3) as usize,
        channels,
        lm: rng.below(4) as usize,
        size,
        // Usually the whole packet, sometimes less than the energies need.
        budget: if rng.below(4) == 0 { rng.below(size * 8 + 1) } else { size * 8 },
        nb_available_bytes: (size - rng.below(size.min(4))) as usize,
        force_intra: rng.below(6) == 0,
        two_pass,
        loss_rate: [0, 0, 5, 20, 100][rng.below(5) as usize],
        lfe: rng.below(10) == 0,
        delayed_intra: if rng.below(2) == 0 { 0.0 } else { rng.level(0, 200) },
        header_bits: (0..rng.below(6)).map(|_| (rng.below(2) == 0, 1 + rng.below(4))).collect(),
        e_bands,
        old_e_bands,
    }
}

/// Runs both encoders on `case`. Returns a description of the first
/// difference, or `None`, and whether libopus coded the frame as intra
/// (`None` when the case cannot tell).
fn compare(mode: *const c_void, case: &Case) -> (Option<String>, Option<bool>) {
    let n = case.channels * NB_EBANDS;

    // libopus.
    let mut c_buf = vec![0u8; case.size as usize];
    let mut c_enc = EcCtx {
        buf: std::ptr::null_mut(),
        storage: 0,
        end_offs: 0,
        end_window: 0,
        nend_bits: 0,
        nbits_total: 0,
        offs: 0,
        rng: 0,
        val: 0,
        ext: 0,
        rem: 0,
        error: 0,
    };
    let mut c_old = case.old_e_bands.clone();
    let mut c_error = vec![0f32; n];
    let mut c_delayed = case.delayed_intra;
    // SAFETY: the coder writes into `c_buf`, which has `size` bytes and
    // outlives the calls. The arrays have `channels * 21` elements, which
    // is what the mode's 21 bands need.
    unsafe {
        ec_enc_init(&mut c_enc, c_buf.as_mut_ptr(), case.size);
        for &(bit, logp) in &case.header_bits {
            ec_enc_bit_logp(&mut c_enc, bit as c_int, logp);
        }
        c_quant_coarse_energy(
            mode,
            case.start as c_int,
            case.end as c_int,
            case.eff_end as c_int,
            case.e_bands.as_ptr(),
            c_old.as_mut_ptr(),
            case.budget,
            c_error.as_mut_ptr(),
            &mut c_enc,
            case.channels as c_int,
            case.lm as c_int,
            case.nb_available_bytes as c_int,
            case.force_intra as c_int,
            &mut c_delayed,
            case.two_pass as c_int,
            case.loss_rate,
            case.lfe as c_int,
        );
    }

    // opus1.
    let mut enc = RangeCoder::new_encoder(case.size);
    for &(bit, logp) in &case.header_bits {
        enc.encode_bit_logp(bit, logp);
    }
    let mut old = case.old_e_bands.clone();
    let mut error = vec![0f32; n];
    let mut delayed = case.delayed_intra;
    quant_coarse_energy_advanced(
        default_mode(),
        case.start,
        case.end,
        case.eff_end,
        &case.e_bands,
        &mut old,
        case.budget,
        &mut error,
        &mut enc,
        case.channels,
        case.lm,
        case.nb_available_bytes,
        case.force_intra,
        &mut delayed,
        case.two_pass,
        case.loss_rate,
        case.lfe,
    );

    let c_words = [
        c_enc.offs as i64,
        c_enc.rng as i64,
        c_enc.val as i64,
        c_enc.rem as i64,
        c_enc.ext as i64,
        c_enc.nbits_total as i64,
        c_enc.error as i64,
    ];
    let words = [
        enc.offs as i64,
        enc.rng as i64,
        enc.val as i64,
        enc.rem as i64,
        enc.ext as i64,
        enc.nbits_total as i64,
        enc.error as i64,
    ];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    // The errors of the coded bands only. When the intra pass wins, libopus
    // copies its whole scratch array of errors, and the entries of the
    // bands outside `start..end` were never written there.
    let coded = |v: &[f32]| -> Vec<f32> {
        (0..case.channels)
            .flat_map(|c| v[c * NB_EBANDS + case.start..c * NB_EBANDS + case.end].to_vec())
            .collect()
    };
    let (error, c_error) = (coded(&error), coded(&c_error));
    let diff = if words != c_words {
        Some(format!("coder words {words:?}, libopus {c_words:?}"))
    } else if enc.buf[..enc.offs as usize] != c_buf[..c_enc.offs as usize] {
        Some("coder bytes".to_string())
    } else if bits(&old) != bits(&c_old) {
        Some(format!("quantized energies {old:?}, libopus {c_old:?}"))
    } else if bits(&error) != bits(&c_error) {
        Some(format!("errors {error:?}, libopus {c_error:?}"))
    } else if (delayed - c_delayed).abs() > 4e-7 * c_delayed.abs() {
        // Not compared bit for bit: for a frame that is not intra the value
        // is `pred * pred * old + new`, and a C compiler may compute the
        // product and the sum in one fused step. The libopus build this
        // test links does. It is then one unit in the last place away in
        // about 1 % of the cases.
        Some(format!("delayed_intra {delayed}, libopus {c_delayed}"))
    } else {
        None
    };
    // Whether the frame was coded as intra shows in `delayed_intra`: for an
    // intra frame it is the new distortion alone, otherwise a part of the
    // old value is added. A case that starts from zero cannot tell.
    let c_intra = (case.delayed_intra > 0.0).then(|| {
        let mut dist = 0.0f32;
        for c in 0..case.channels {
            for i in case.start..case.eff_end.min(NB_EBANDS) {
                let d = case.e_bands[c * NB_EBANDS + i] - case.old_e_bands[c * NB_EBANDS + i];
                dist += d * d;
            }
        }
        c_delayed.to_bits() == dist.min(200.0).to_bits()
    });
    (diff, c_intra)
}

/// Returns the number of cases that can tell whether the frame was intra,
/// and how many of them were.
fn run(seed: u64, cases: usize, two_pass: bool) -> (usize, usize) {
    let mode = c_mode();
    let mut rng = Lcg(seed);
    let (mut known, mut intra) = (0, 0);
    let mut differ = Vec::new();
    for k in 0..cases {
        let case = random_case(&mut rng, two_pass);
        let (diff, c_intra) = compare(mode, &case);
        known += usize::from(c_intra.is_some());
        intra += usize::from(c_intra == Some(true));
        if let Some(d) = diff {
            differ.push(format!(
                "case {k} (bands {}..{}, {} ch, LM {}, {} bytes, budget {}, lfe {}): {d}",
                case.start, case.end, case.channels, case.lm, case.size, case.budget, case.lfe
            ));
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {cases} cases differ from libopus; the first:\n{}",
        differ.len(),
        differ[0]
    );
    (known, intra)
}

/// One pass, as the encoder runs it below complexity 4.
#[test]
fn one_pass_coarse_energy_matches_libopus() {
    let (known, intra) = run(11, 20_000, false);
    assert!(known > 5000 && intra > known / 50, "{intra} of {known} intra");
}

/// Two passes, as the encoder runs them from complexity 4 on. The choice
/// between the passes must be that of libopus.
#[test]
fn two_pass_coarse_energy_matches_libopus() {
    let (known, intra) = run(12, 40_000, true);
    // Both outcomes must occur, or the choice was not tested.
    assert!(
        known > 10_000 && intra > known / 20 && intra < known * 19 / 20,
        "{intra} of {known} intra"
    );
}
