//! The surround mask against `surround_analysis` of C libopus, called
//! directly.
//!
//! libopus has no API call that returns the mask. But `surround_analysis`
//! (opus_multistream_encoder.c) is not a `static` function, and libopus is
//! linked as a static library, so the test can call it like
//! `tests/k24_silk_internals.rs` calls the SILK internals. The test keeps the
//! two memories the function needs (the window memory and the pre-emphasis
//! memory of every channel), as the C multistream encoder does.
//!
//! The opus1 analysis and the C function compute in `f32` with different
//! MDCT code, so their results differ by rounding. `TOLERANCE` says how much
//! when every band holds signal. The test
//! `bands_that_hold_only_rounding_noise_differ_more` covers the other case.
//!
//! The signals and several settings here are not the ones of
//! `multistream_surround_reference.rs` and `multistream_surround_c.rs`:
//! sweeps, clicks, one source on all channels, a constant offset, gated
//! noise, 16-bit input, a signal that fades to nothing, and channels at the
//! energy where libopus starts to drop them.
// The multistream API needs the `std` feature.
#![cfg(feature = "std")]

mod ms_support;

use audiopus_sys as sys;
use ms_support::{APP_AUDIO, CMsEncoder, max_packet_bytes, noise, silence, to_i16, tones, voiced};
use opus1::internals::multistream::surround::SurroundProbe;
use std::f64::consts::PI;
use std::os::raw::{c_int, c_void};

const NB_BANDS: usize = 21;
/// `celt_mode->overlap` of the one CELT mode libopus has.
const OVERLAP: usize = 120;
/// Private request `CELT_GET_MODE` (celt.h). `opus_encoder_ctl` hands it to
/// the CELT encoder. Its argument is a pointer to `*const CELTMode`.
const CELT_GET_MODE_REQUEST: c_int = 10015;

/// libopus `CELTMode`. Only pointers to it are used here.
#[repr(C)]
struct CeltMode {
    _private: [u8; 0],
}

/// libopus `opus_copy_channel_in_func` (opus_private.h), float build.
type CopyChannelIn = unsafe extern "C" fn(
    dst: *mut f32,
    dst_stride: c_int,
    src: *const c_void,
    src_stride: c_int,
    src_channel: c_int,
    frame_size: c_int,
    user_data: *mut c_void,
);

unsafe extern "C" {
    /// opus_multistream_encoder.c 230. `arch` selects the vector code of
    /// libopus. 0 is always valid: it is the value of a build without
    /// run-time CPU detection, and the plain C code in a build with it.
    fn surround_analysis(
        celt_mode: *const CeltMode,
        pcm: *const c_void,
        band_log_e: *mut f32,
        mem: *mut f32,
        preemph_mem: *mut f32,
        len: c_int,
        overlap: c_int,
        channels: c_int,
        rate: c_int,
        copy_channel_in: CopyChannelIn,
        arch: c_int,
    );
}

/// `opus_copy_channel_in_float` of opus_multistream_encoder.c.
unsafe extern "C" fn copy_channel_in_float(
    dst: *mut f32,
    dst_stride: c_int,
    src: *const c_void,
    src_stride: c_int,
    src_channel: c_int,
    frame_size: c_int,
    _user_data: *mut c_void,
) {
    let src = src as *const f32;
    for i in 0..frame_size as usize {
        unsafe {
            *dst.add(i * dst_stride as usize) =
                *src.add(i * src_stride as usize + src_channel as usize);
        }
    }
}

/// `opus_copy_channel_in_short` of opus_multistream_encoder.c: a 16-bit
/// sample becomes `x / 32768`.
unsafe extern "C" fn copy_channel_in_short(
    dst: *mut f32,
    dst_stride: c_int,
    src: *const c_void,
    src_stride: c_int,
    src_channel: c_int,
    frame_size: c_int,
    _user_data: *mut c_void,
) {
    let src = src as *const i16;
    for i in 0..frame_size as usize {
        unsafe {
            *dst.add(i * dst_stride as usize) =
                (1.0 / 32768.0) * *src.add(i * src_stride as usize + src_channel as usize) as f32;
        }
    }
}

/// The CELT mode of libopus, read from an encoder as the C multistream
/// encoder reads it. The mode is a constant table inside libopus.
fn c_celt_mode() -> *const CeltMode {
    let mut err = 0;
    let mut mode: *const CeltMode = std::ptr::null();
    unsafe {
        let enc = sys::opus_encoder_create(48_000, 1, sys::OPUS_APPLICATION_AUDIO, &mut err);
        assert!(!enc.is_null() && err == 0, "opus_encoder_create = {err}");
        let ret = sys::opus_encoder_ctl(
            enc,
            CELT_GET_MODE_REQUEST,
            &mut mode as *mut *const CeltMode,
        );
        sys::opus_encoder_destroy(enc);
        assert_eq!(ret, 0, "CELT_GET_MODE = {ret}");
    }
    assert!(!mode.is_null());
    mode
}

/// C `surround_analysis` with the memories it keeps from frame to frame.
struct CSurround {
    mode: *const CeltMode,
    channels: usize,
    rate: i32,
    /// `ms_get_window_mem`: 120 values per channel.
    mem: Vec<f32>,
    /// `ms_get_preemph_mem`: one value per channel.
    preemph_mem: Vec<f32>,
}

impl CSurround {
    fn new(channels: usize, rate: i32) -> Self {
        Self {
            mode: c_celt_mode(),
            channels,
            rate,
            mem: vec![0.0; channels * OVERLAP],
            preemph_mem: vec![0.0; channels],
        }
    }

    /// `OPUS_RESET_STATE` of the C multistream encoder clears both memories.
    fn reset(&mut self) {
        self.mem.fill(0.0);
        self.preemph_mem.fill(0.0);
    }

    fn call(&mut self, pcm: *const c_void, frame_size: usize, copy: CopyChannelIn) -> Vec<f32> {
        let mut band_smr = vec![f32::NAN; NB_BANDS * self.channels];
        unsafe {
            surround_analysis(
                self.mode,
                pcm,
                band_smr.as_mut_ptr(),
                self.mem.as_mut_ptr(),
                self.preemph_mem.as_mut_ptr(),
                frame_size as c_int,
                OVERLAP as c_int,
                self.channels as c_int,
                self.rate,
                copy,
                0,
            );
        }
        band_smr
    }

    /// One frame of interleaved float samples.
    fn run(&mut self, pcm: &[f32], frame_size: usize) -> Vec<f32> {
        assert_eq!(pcm.len(), frame_size * self.channels);
        self.call(
            pcm.as_ptr() as *const c_void,
            frame_size,
            copy_channel_in_float,
        )
    }

    /// One frame of interleaved 16-bit samples.
    fn run_i16(&mut self, pcm: &[i16], frame_size: usize) -> Vec<f32> {
        assert_eq!(pcm.len(), frame_size * self.channels);
        self.call(
            pcm.as_ptr() as *const c_void,
            frame_size,
            copy_channel_in_short,
        )
    }
}

/// Largest difference expected between the two masks when every band holds
/// signal, in log2 units (1.0 is 6 dB).
///
/// Both sides round to `f32` in the MDCT. That leaves an error of about
/// 1e-7 of the strongest coefficient of a frame. A band 70 dB below the
/// strongest one is then off by 3e-4 of its level, which is 5e-4 in log2
/// units.
///
/// A few values lie further apart. With frames of 2.5 ms each of the 8
/// lowest bands is one MDCT coefficient. Such a coefficient comes out close
/// to zero now and then, when the parts it is summed from cancel. Rounding
/// then decides its level. Measured on aarch64 over the 1.2 million values
/// of `mask_is_the_mask_of_libopus_for_every_layout_rate_and_frame_size`:
/// 9 values differ by more than this tolerance, never two of them in one
/// case, and the largest difference is 3.6e-3. Every one of them is in a
/// frame of 2.5 ms. In longer frames the largest difference is 1.7e-3.
const TOLERANCE: f32 = 2e-3;

/// Largest difference allowed for the few values `TOLERANCE` describes as
/// decided by rounding.
const TOLERANCE_OUTLIER: f32 = 0.5;

/// Checks the differences of one case. Every value must be within
/// `TOLERANCE_OUTLIER`. One value, plus one for every 1000 values of the
/// case, may differ by more than `TOLERANCE`.
///
/// A mistake in the analysis moves whole bands or channels in every frame,
/// which is far more than that.
fn check(what: &str, s: &Stats) {
    assert!(s.above <= 1 + s.values / 1000, "{what}: {s}");
    assert!(s.worst < TOLERANCE_OUTLIER, "{what}: {s}");
}

/// The differences between the opus1 mask and the C mask over a signal.
#[derive(Default, Clone, Copy)]
struct Stats {
    /// Number of values compared.
    values: usize,
    /// Largest difference.
    worst: f32,
    /// Where it is: frame, channel and band.
    at: (usize, usize, usize),
    /// Number of values that differ by more than `TOLERANCE`.
    above: usize,
    /// Number of values that differ by more than 1e-4.
    above_1e4: usize,
    /// Number of values that are the same bits.
    exact: usize,
}

impl Stats {
    fn add(&mut self, other: &Stats) {
        self.values += other.values;
        self.above += other.above;
        self.above_1e4 += other.above_1e4;
        self.exact += other.exact;
        if other.worst > self.worst {
            self.worst = other.worst;
            self.at = other.at;
        }
    }

    /// Adds the values of one frame.
    fn tally(&mut self, frame: usize, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.is_finite() && w.is_finite(),
                "frame {frame}, value {k}: opus1 {g}, libopus {w}"
            );
            let d = (g - w).abs();
            self.values += 1;
            self.exact += (g.to_bits() == w.to_bits()) as usize;
            self.above += (d > TOLERANCE) as usize;
            self.above_1e4 += (d > 1e-4) as usize;
            if d > self.worst {
                self.worst = d;
                self.at = (frame, k / NB_BANDS, k % NB_BANDS);
            }
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (frame, channel, band) = self.at;
        write!(
            f,
            "{} values, {} the same bits, {} differ by more than 1e-4, {} by more than {TOLERANCE:e}, largest difference {:.2e} (frame {frame}, channel {channel}, band {band})",
            self.values, self.exact, self.above_1e4, self.above, self.worst
        )
    }
}

/// Runs both analyses over `pcm` and compares every value. Frame `f` has
/// `frame_sizes[f % frame_sizes.len()]` samples per channel. Both keep
/// their memories from frame to frame.
fn compare(channels: usize, rate: i32, frame_sizes: &[usize], pcm: &[f32]) -> Stats {
    let mut probe = SurroundProbe::new(channels, rate);
    let mut c = CSurround::new(channels, rate);
    let mut stats = Stats::default();
    let (mut start, mut f) = (0, 0);
    loop {
        let frame_size = frame_sizes[f % frame_sizes.len()];
        let end = start + frame_size * channels;
        if end > pcm.len() {
            break;
        }
        let frame = &pcm[start..end];
        let mut got = vec![f32::NAN; NB_BANDS * channels];
        assert!(probe.run(frame, frame_size, &mut got), "frame {f}: no mask");
        stats.tally(f, &got, &c.run(frame, frame_size));
        start = end;
        f += 1;
    }
    assert!(f >= 2, "need at least two frames to use the memories");
    stats
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// Adds `b` to `a`, sample by sample.
fn mix(mut a: Vec<f32>, b: &[f32]) -> Vec<f32> {
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
    a
}

/// A sine on every channel whose frequency rises over the signal. Every
/// channel starts at another frequency and has another level.
fn sweeps(rate: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let f0 = 60.0 + 170.0 * c as f64;
        let f1 = (0.45 * rate as f64).min(3000.0 + 2500.0 * c as f64);
        let amp = 0.5 / (1.0 + c as f64);
        let mut phase = 0.0f64;
        for i in 0..samples {
            let f = f0 + (f1 - f0) * i as f64 / samples as f64;
            phase += 2.0 * PI * f / rate as f64;
            pcm[i * channels + c] = (amp * phase.sin()) as f32;
        }
    }
    pcm
}

/// Clicks: single samples followed by a short burst that changes sign from
/// sample to sample. They come at other times on every channel.
fn clicks(rate: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let period = rate as usize / 50 + 37 * c + 11;
        let mut i = 13 * c + 5;
        while i < samples {
            pcm[i * channels + c] += 0.8 - 0.1 * c as f32;
            for k in 1..24.min(samples - i) {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                pcm[(i + k) * channels + c] += sign * 0.4 * (-(k as f32) / 6.0).exp();
            }
            i += period;
        }
    }
    pcm
}

/// One source on all channels: every channel carries the same speech-like
/// signal with another gain and another delay, plus a little noise of its
/// own.
fn one_source_on_all_channels(rate: i32, channels: usize, samples: usize) -> Vec<f32> {
    const GAINS: [f32; 8] = [1.0, 0.7, 0.05, 0.3, 0.01, 0.5, 0.2, 0.9];
    let source = voiced(rate, 1, samples + 64);
    let mut pcm = noise(channels, samples, 0.002, 103);
    for c in 0..channels {
        for i in 0..samples {
            pcm[i * channels + c] += GAINS[c] * source[i + 5 * c];
        }
    }
    pcm
}

/// A constant offset with a little noise on every channel, and a square
/// wave at full scale on channel 1. The offset makes the result depend on
/// the pre-emphasis memory: without the memory the first sample of a frame
/// is a step.
fn offset_and_square_wave(rate: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = noise(channels, samples, 0.01, 107);
    let half_period = (rate as usize / 880).max(2);
    for i in 0..samples {
        for c in 0..channels {
            pcm[i * channels + c] += 0.3 - 0.1 * c as f32;
        }
        pcm[i * channels + 1] = if (i / half_period).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
    }
    pcm
}

/// Noise that is on for 7 ms and off for 14 ms, at other times on every
/// channel. A frame often starts or ends inside a block of noise. The
/// window memory then holds signal while the frame holds none, or the
/// reverse.
fn gated_noise(rate: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = noise(channels, samples, 0.25, 109);
    let block = rate as usize * 7 / 1000;
    for i in 0..samples {
        for c in 0..channels {
            if !((i + c * block / 3) / block + c).is_multiple_of(3) {
                pcm[i * channels + c] = 0.0;
            }
        }
    }
    pcm
}

/// Signal on the last channel and on one more channel, silence on the
/// others. With 6 or more channels the last one is the
/// low-frequency-effects channel.
fn last_channel_and_one_more(rate: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = silence(channels, samples);
    let source = mix(
        tones(rate, channels, 1, samples),
        &noise(channels, samples, 0.05, 113),
    );
    for i in 0..samples {
        for c in [channels - 1, channels / 2] {
            pcm[i * channels + c] = source[i * channels + c];
        }
    }
    pcm
}

/// The signals in which every band holds signal or exact silence, by name.
fn signals(rate: i32, channels: usize, samples: usize) -> Vec<(&'static str, Vec<f32>)> {
    vec![
        (
            "sweeps over a noise floor",
            mix(
                sweeps(rate, channels, samples),
                &noise(channels, samples, 0.003, 97),
            ),
        ),
        (
            "clicks over a noise floor",
            mix(
                clicks(rate, channels, samples),
                &noise(channels, samples, 0.003, 101),
            ),
        ),
        (
            "one source on all channels",
            one_source_on_all_channels(rate, channels, samples),
        ),
        (
            "offset and square wave",
            offset_and_square_wave(rate, channels, samples),
        ),
        ("gated noise", gated_noise(rate, channels, samples)),
        (
            "last channel and one more",
            last_channel_and_one_more(rate, channels, samples),
        ),
    ]
}

const RATES: [i32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];
/// Frame durations in units of 2.5 ms: 2.5, 5, 10, 20, 40, 60, 80, 100 and
/// 120 ms. These are all the durations the C multistream encoder accepts.
const FRAMES_400: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn mask_is_the_mask_of_libopus_for_every_layout_rate_and_frame_size() {
    // 3 to 8 channels, 5 rates, 9 frame durations, 6 signals: 1620 cases.
    let mut all = Stats::default();
    for which in 0..signals(48_000, 3, 1).len() {
        let mut total = Stats::default();
        let mut name = "";
        for channels in 3..=8usize {
            for rate in RATES {
                for frame_400 in FRAMES_400 {
                    let frame_size = rate as usize / 400 * frame_400;
                    let frames = (24 / frame_400).max(3);
                    let (n, pcm) = signals(rate, channels, frames * frame_size).swap_remove(which);
                    name = n;
                    let s = compare(channels, rate, &[frame_size], &pcm);
                    let ms = frame_400 as f64 * 2.5;
                    check(
                        &format!("{name}, {channels} channels, {rate} Hz, {ms} ms"),
                        &s,
                    );
                    total.add(&s);
                }
            }
        }
        println!("{name}: {total}");
        all.add(&total);
    }
    println!("all: {all}");
    // Rounding is all that differs. Few values differ by more than 1e-4,
    // and next to none by more than the tolerance. Measured on aarch64:
    // no value differs by more than 1e-4, and 98.7 % of the values are the
    // same bits (1,231,583 of 1,247,400).
    assert!(all.above_1e4 * 20 < all.values, "{all}");
    assert!(all.above * 10_000 < all.values, "{all}");
    // On aarch64 opus1 takes the band levels and the channel offset with
    // libopus's approximation of the logarithm (`celt_log2`), as libopus
    // does there. With the exact logarithm for the band levels 78.5 % of the
    // values are the same bits, and with it for the offset 89.2 %.
    if cfg!(target_arch = "aarch64") {
        assert!(all.exact * 1000 >= all.values * 985, "{all}");
    }
}

#[test]
fn mask_of_16_bit_input_is_the_mask_of_libopus() {
    // libopus turns a 16-bit sample into `x / 32768` before the analysis
    // (`opus_copy_channel_in_short`). The opus1 analysis expects the
    // caller to do the same.
    let mut total = Stats::default();
    for (channels, rate, frame_size) in [
        (6usize, 48_000, 960usize),
        (8, 48_000, 240),
        (3, 8_000, 160),
        (7, 12_000, 480),
        (5, 16_000, 40),
        (4, 24_000, 1440),
    ] {
        for (name, pcm) in signals(rate, channels, 6 * frame_size) {
            let mut pcm16 = to_i16(&pcm);
            // Both ends of the 16-bit range, next to each other.
            pcm16[channels * 7] = i16::MIN;
            pcm16[channels * 8] = i16::MAX;
            let as_float: Vec<f32> = pcm16.iter().map(|&v| v as f32 / 32768.0).collect();
            let mut probe = SurroundProbe::new(channels, rate);
            let mut c = CSurround::new(channels, rate);
            let mut stats = Stats::default();
            let frames16 = pcm16.chunks_exact(frame_size * channels);
            let frames = as_float.chunks_exact(frame_size * channels);
            for (f, (frame16, frame)) in frames16.zip(frames).enumerate() {
                let mut got = vec![f32::NAN; NB_BANDS * channels];
                assert!(probe.run(frame, frame_size, &mut got));
                stats.tally(f, &got, &c.run_i16(frame16, frame_size));
            }
            check(
                &format!("{name}, {channels} channels, {rate} Hz, frame {frame_size}"),
                &stats,
            );
            total.add(&stats);
        }
    }
    println!("16-bit input: {total}");
}

#[test]
fn memories_follow_libopus_when_the_frame_size_changes_and_after_a_reset() {
    // Every frame duration at every rate, in an order that puts short
    // frames next to long ones, twice. Before frame 5 and before frame 12
    // both sides clear their memories, as `OPUS_RESET_STATE` does in the C
    // multistream encoder.
    const ORDER: [usize; 9] = [8, 1, 48, 2, 24, 4, 40, 16, 32];
    let mut total = Stats::default();
    for (channels, rate) in [
        (6usize, 48_000),
        (8, 16_000),
        (3, 8_000),
        (4, 12_000),
        (5, 24_000),
        (7, 48_000),
    ] {
        let unit = rate as usize / 400;
        let samples = 2 * ORDER.iter().sum::<usize>() * unit;
        for (name, pcm) in signals(rate, channels, samples) {
            let mut probe = SurroundProbe::new(channels, rate);
            let mut c = CSurround::new(channels, rate);
            let mut stats = Stats::default();
            let mut start = 0;
            for f in 0..2 * ORDER.len() {
                let frame_size = ORDER[f % ORDER.len()] * unit;
                let frame = &pcm[start..start + frame_size * channels];
                start += frame_size * channels;
                if f == 5 || f == 12 {
                    probe.reset();
                    c.reset();
                }
                let mut got = vec![f32::NAN; NB_BANDS * channels];
                assert!(probe.run(frame, frame_size, &mut got));
                stats.tally(f, &got, &c.run(frame, frame_size));
            }
            check(&format!("{name}, {channels} channels, {rate} Hz"), &stats);
            total.add(&stats);
        }
    }
    println!("changing frame sizes and resets: {total}");
}

/// Energy of `x` after the scaling to the 16-bit range and the pre-emphasis
/// of CELT, as `surround_analysis` sums it. `mem` is the pre-emphasis
/// memory before the first sample.
fn preemphasized_energy(x: &[f32], mut mem: f64) -> f64 {
    let mut sum = 0.0;
    for &v in x {
        let v = 32768.0 * v as f64;
        sum += (v - mem) * (v - mem);
        mem = 0.850_006_1 * v;
    }
    sum
}

/// Channel `c` of frame `f` of an interleaved signal.
fn channel_of_frame(
    pcm: &[f32],
    channels: usize,
    frame_size: usize,
    f: usize,
    c: usize,
) -> Vec<f32> {
    pcm[f * frame_size * channels..(f + 1) * frame_size * channels]
        .iter()
        .skip(c)
        .step_by(channels)
        .copied()
        .collect()
}

#[test]
fn a_channel_is_dropped_at_the_energy_libopus_drops_it() {
    // libopus clears a channel when the energy of its frame, together with
    // the 120 samples kept from the frame before, is not below 1e18. One
    // channel is put on either side of that limit, a factor of 3 away.
    let (channels, rate, frame_size) = (6usize, 48_000, 960usize);
    let loud = 3usize;
    for (target, dropped) in [(0.3e18f64, false), (3e18, true)] {
        let mut pcm = noise(channels, 5 * frame_size, 0.1, 127);
        let unit =
            preemphasized_energy(&channel_of_frame(&pcm, channels, frame_size, 1, loud), 0.0)
                * (frame_size + OVERLAP) as f64
                / frame_size as f64;
        let gain = (target / unit).sqrt() as f32;
        for v in pcm.iter_mut().skip(loud).step_by(channels) {
            *v *= gain;
        }
        // The energy of every frame alone is on the same side of the limit
        // by a factor of 2 or more. The 120 samples before it only add.
        for f in 0..5 {
            let e =
                preemphasized_energy(&channel_of_frame(&pcm, channels, frame_size, f, loud), 0.0);
            assert!(
                if dropped { e > 2e18 } else { e < 0.5e18 },
                "frame {f}: energy {e:e}"
            );
        }
        let stats = compare(channels, rate, &[frame_size], &pcm);
        println!("energy {target:e}: {stats}");
        check(&format!("energy {target:e}"), &stats);
        // The other left channel shows whether the loud one counted: it is
        // far below its mask only when the loud channel was kept.
        let mut probe = SurroundProbe::new(channels, rate);
        let mut mask = vec![0f32; NB_BANDS * channels];
        assert!(probe.run(&pcm[..frame_size * channels], frame_size, &mut mask));
        let front_left = &mask[..NB_BANDS];
        if dropped {
            assert!(front_left.iter().all(|&v| v > -1.0), "{front_left:?}");
        } else {
            assert!(front_left.iter().all(|&v| v < -8.0), "{front_left:?}");
        }
    }

    // The 120 samples kept from the frame before count. With frames of
    // 2.5 ms they weigh as much as the frame. A tone whose frames have
    // 0.7e18 each is kept in the first frame. In the second frame the sum
    // is 1.4e18, so the channel is cleared, and its memories with it. The
    // third frame is then like the first. The tone has 8 periods in a
    // frame, so every frame starts at a zero crossing and the energy of a
    // frame does not depend on the pre-emphasis memory.
    let frame_size = 120usize;
    let frames = 12;
    let mut pcm = noise(channels, frames * frame_size, 0.1, 131);
    let tone: Vec<f32> = (0..frames * frame_size)
        .map(|i| (2.0 * PI * 3200.0 * i as f64 / rate as f64).sin() as f32)
        .collect();
    let gain =
        (0.7e18 / preemphasized_energy(&tone[frame_size..2 * frame_size], 0.0)).sqrt() as f32;
    for (i, &t) in tone.iter().enumerate() {
        pcm[i * channels] = gain * t;
    }
    for f in 0..frames {
        let x = channel_of_frame(&pcm, channels, frame_size, f, 0);
        let before = if f == 0 {
            0.0
        } else {
            pcm[(f * frame_size - 1) * channels]
        };
        for mem in [0.0, 0.850_006_1 * 32768.0 * before as f64] {
            let e = preemphasized_energy(&x, mem);
            assert!((0.6e18..0.8e18).contains(&e), "frame {f}: energy {e:e}");
        }
    }
    let mut probe = SurroundProbe::new(channels, rate);
    let mut c = CSurround::new(channels, rate);
    let mut stats = Stats::default();
    for (f, frame) in pcm.chunks_exact(frame_size * channels).enumerate() {
        let mut got = vec![f32::NAN; NB_BANDS * channels];
        assert!(probe.run(frame, frame_size, &mut got));
        stats.tally(f, &got, &c.run(frame, frame_size));
        // Band 12 starts at 3200 Hz. A kept tone stands above its mask by
        // the offset for 6 channels, 0.66. A cleared channel lies far
        // below the noise of the other left channel.
        let level = got[12];
        if f % 2 == 0 {
            assert!(level > 0.5, "frame {f}: kept, but band 12 is at {level}");
        } else {
            assert!(
                level < -10.0,
                "frame {f}: cleared, but band 12 is at {level}"
            );
        }
    }
    println!("tone at 0.7e18 per 2.5 ms: {stats}");
    check("tone at 0.7e18 per 2.5 ms", &stats);
}

#[test]
fn absurd_samples_are_handled_as_by_libopus() {
    // Samples that are not a number, infinite, far outside any audio range
    // or too small to count. libopus clears the channel for the frame when
    // its energy is 1e18 or more, or not a number. It then carries on with
    // empty memories. Every third frame holds one kind of such samples. The
    // frames between them show that the memories are those of libopus.
    for (channels, rate, frame_size) in [
        (6usize, 48_000, 960usize),
        (8, 16_000, 320),
        (5, 8_000, 20),
        (7, 24_000, 1440),
    ] {
        let frames = 30;
        let mut pcm = mix(
            offset_and_square_wave(rate, channels, frames * frame_size),
            &noise(channels, frames * frame_size, 0.05, 151),
        );
        let at = |f: usize, i: usize, c: usize| (f * frame_size + i) * channels + c;
        // One bad sample in the middle of a frame.
        pcm[at(1, frame_size / 2, 0)] = f32::NAN;
        pcm[at(4, frame_size / 3, 2)] = f32::INFINITY;
        pcm[at(7, 3, channels - 1)] = f32::MAX;
        // The last sample of a frame: it is also the pre-emphasis memory
        // of the next frame.
        pcm[at(10, frame_size - 1, 1)] = f32::NEG_INFINITY;
        pcm[at(13, frame_size - 1, 0)] = -1e6;
        // The first sample of a frame.
        pcm[at(16, 0, 2)] = f32::NAN;
        // A whole frame of every channel.
        for v in &mut pcm[at(19, 0, 0)..at(20, 0, 0)] {
            *v = f32::NAN;
        }
        // Samples whose squares are 0 in `f32`: like silence.
        for i in 0..frame_size {
            pcm[at(22, i, 0)] = 1e-40;
            pcm[at(22, i, 1)] = -f32::MIN_POSITIVE;
            pcm[at(22, i, 2)] = -0.0;
        }
        // Two bad frames in a row on one channel.
        pcm[at(25, 5, 1)] = f32::NAN;
        pcm[at(26, 5, 1)] = f32::INFINITY;
        let stats = compare(channels, rate, &[frame_size], &pcm);
        println!("{channels} channels, {rate} Hz, absurd samples: {stats}");
        check(
            &format!("{channels} channels, {rate} Hz, absurd samples"),
            &stats,
        );
    }
}

#[test]
fn a_signal_that_fades_to_nothing_matches_libopus() {
    // Noise that loses 20 dB from frame to frame, from full scale down to
    // 1e-30. On the way the band levels reach the two floors of the
    // analysis: 1e-27 under the square root of a band energy, and -28 as
    // the level of an empty side of the mix.
    for (channels, rate, frame_size) in
        [(6usize, 48_000, 960usize), (8, 16_000, 160), (4, 8_000, 20)]
    {
        let frames = 31;
        let mut pcm = noise(channels, frames * frame_size, 1.0, 137);
        for (f, frame) in pcm.chunks_exact_mut(frame_size * channels).enumerate() {
            let gain = 10f32.powi(-(f as i32));
            for (i, v) in frame.iter_mut().enumerate() {
                // Another level on every channel.
                *v *= gain / (1 + i % channels) as f32;
            }
        }
        let stats = compare(channels, rate, &[frame_size], &pcm);
        println!("{channels} channels, {rate} Hz, fading: {stats}");
        check(&format!("{channels} channels, {rate} Hz, fading"), &stats);
        // The last frames are silence for the analysis: the same values as
        // for samples that are exactly 0.
        let mut faded = SurroundProbe::new(channels, rate);
        let mut silent = SurroundProbe::new(channels, rate);
        let (mut a, mut b) = (
            vec![0f32; NB_BANDS * channels],
            vec![1f32; NB_BANDS * channels],
        );
        let last = &pcm[(frames - 1) * frame_size * channels..];
        assert!(last.iter().any(|&v| v != 0.0));
        assert!(faded.run(last, frame_size, &mut a));
        assert!(silent.run(&silence(channels, frame_size), frame_size, &mut b));
        assert_eq!(a, b);
    }
}

#[test]
fn bands_that_hold_only_rounding_noise_differ_more() {
    // A sine without a noise floor leaves most bands empty. What the MDCT
    // puts there is its own rounding error, about 1e-7 of the sine. libopus
    // and opus1 round differently, so the level of such a band, and with
    // it the mask, is not the same on both sides. Neither side is the
    // exact one. Measured against the `f64` reference of
    // `multistream_surround_reference.rs` on signals of this kind, libopus
    // is off by up to 0.21 and opus1 by up to 0.27.
    //
    // This is why a few packets differ in `multistream_surround_c.rs` on
    // the speech-like signal, which has no noise floor either.
    let mut total = Stats::default();
    for channels in [3usize, 6, 8] {
        for rate in RATES {
            for frame_400 in [1usize, 8, 24] {
                let frame_size = rate as usize / 400 * frame_400;
                let frames = (24 / frame_400).max(3);
                let pcm = sweeps(rate, channels, frames * frame_size);
                total.add(&compare(channels, rate, &[frame_size], &pcm));
            }
        }
    }
    println!("sweeps without a noise floor: {total}");
    // Measured on aarch64: 4 % of the values differ by more than 2e-3, and
    // the largest difference is 0.38.
    assert!(total.above * 5 < total.values, "{total}");
    assert!(total.worst < 2.0, "{total}");
    // The bands that do hold signal still agree. Measured: 13 % of the
    // values differ by more than 1e-4.
    assert!(total.above_1e4 * 4 < total.values, "{total}");
}

#[test]
fn frame_sizes_are_accepted_as_by_the_c_surround_encoder() {
    // The C surround encoder runs the analysis for every frame size it
    // accepts. The opus1 analysis gives a mask for exactly those sizes.
    for (rate, channels) in RATES
        .into_iter()
        .flat_map(|r| (3..=8usize).map(move |c| (r, c)))
    {
        let unit = rate as usize / 400;
        let mut ms = CMsEncoder::surround(rate, channels, 1, APP_AUDIO).unwrap();
        let mut probe = SurroundProbe::new(channels, rate);
        let pcm = noise(channels, 52 * unit, 0.1, 139);
        let mut mask = vec![0f32; NB_BANDS * channels];
        let mut accepted = Vec::new();
        let mut sizes: Vec<usize> = (0..=50).map(|k| k * unit).collect();
        sizes.extend((1..=48).flat_map(|k| [k * unit - 1, k * unit + 1]));
        sizes.extend([1, unit / 2, 3 * unit / 2, 5 * unit / 2]);
        for frame_size in sizes {
            let c_ok = ms
                .encode_float(&pcm, frame_size, max_packet_bytes(ms.streams))
                .is_ok();
            let rs_ok = probe.run(&pcm, frame_size, &mut mask);
            assert_eq!(
                rs_ok, c_ok,
                "{channels} channels, {rate} Hz, frame size {frame_size}: opus1 {rs_ok}, libopus {c_ok}"
            );
            if c_ok {
                accepted.push(frame_size / unit);
            }
        }
        assert_eq!(accepted, FRAMES_400, "{channels} channels, {rate} Hz");
    }
}

#[test]
fn a_call_without_a_mask_leaves_the_memories_alone() {
    // A call that gives no mask must not count as a frame. The next frame
    // has to be analysed with the memories of the frame before the call.
    let (channels, rate, frame_size) = (7usize, 24_000, 240usize);
    let pcm = offset_and_square_wave(rate, channels, 4 * frame_size);
    let other = noise(channels, 6000, 0.9, 149);
    let mut plain = SurroundProbe::new(channels, rate);
    let mut disturbed = SurroundProbe::new(channels, rate);
    let mut c = CSurround::new(channels, rate);
    let mut stats = Stats::default();
    for (f, frame) in pcm.chunks_exact(frame_size * channels).enumerate() {
        let mut want = vec![0f32; NB_BANDS * channels];
        assert!(plain.run(frame, frame_size, &mut want));
        let mut got = vec![7f32; NB_BANDS * channels];
        // A frame size that is not valid, no room for the mask, and too
        // few samples.
        assert!(!disturbed.run(&other, frame_size + 1, &mut got));
        assert!(!disturbed.run(&other, 0, &mut got));
        assert!(!disturbed.run(&other, frame_size, &mut got[..NB_BANDS * channels - 1]));
        assert!(!disturbed.run(&other[..frame_size * channels - 1], frame_size, &mut got));
        assert!(
            got.iter().all(|&v| v == 7.0),
            "frame {f}: the mask was written"
        );
        assert!(disturbed.run(frame, frame_size, &mut got));
        assert_eq!(got, want, "frame {f}");
        stats.tally(f, &got, &c.run(frame, frame_size));
    }
    check("frames between calls without a mask", &stats);
}
