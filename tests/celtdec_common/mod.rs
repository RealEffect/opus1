//! Shared helpers for the `celtdec_*` tests.
//!
//! These tests compare opus-rs's CELT decoder with C libopus on the same
//! packets: loss concealment, and decoding with a channel count that
//! differs from the packet's. The signals and the C encoder wrapper come
//! from `k24_common`. This module adds what those tests need on top.

#![allow(dead_code)]

use audiopus_sys as sys;

/// C libopus decoder with access to the final range of the last packet.
pub struct CDecoder {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
}

impl CDecoder {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        Self { st, channels }
    }

    /// Decode a packet, or conceal a lost one (`None`), into `frame`
    /// samples per channel. Returns the interleaved samples.
    pub fn decode(&mut self, packet: Option<&[u8]>, frame: usize) -> Vec<f32> {
        let mut pcm = vec![0f32; frame * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n =
            unsafe { sys::opus_decode_float(self.st, ptr, len, pcm.as_mut_ptr(), frame as i32, 0) };
        assert!(n >= 0, "opus_decode_float failed: {n}");
        pcm.truncate(n as usize * self.channels);
        pcm
    }

    /// `OPUS_GET_FINAL_RANGE`: the range coder state after the last packet.
    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_decoder_ctl(
                self.st,
                sys::OPUS_GET_FINAL_RANGE_REQUEST,
                &mut value as *mut u32,
            )
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE");
        value
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// Mean power of a block of samples.
pub fn power(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64
}

/// Level of a block of samples in dB relative to full scale.
pub fn dbfs(x: &[f32]) -> f64 {
    10.0 * power(x).max(1e-20).log10()
}

/// Signal-to-noise ratio in dB of `test` against `reference`. 200 means
/// identical.
pub fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (a, b) in reference.iter().zip(test) {
        sig += (*a as f64).powi(2);
        err += ((*a - *b) as f64).powi(2);
    }
    if err <= 0.0 {
        200.0
    } else if sig <= 0.0 {
        -200.0
    } else {
        (10.0 * (sig / err).log10()).min(200.0)
    }
}

/// Planar samples (`[c * n + i]`, what `CeltDecoder` writes) to interleaved
/// (`[i * channels + c]`, what libopus writes).
pub fn interleave(planar: &[f32], channels: usize) -> Vec<f32> {
    let n = planar.len() / channels;
    let mut out = vec![0f32; planar.len()];
    for c in 0..channels {
        for i in 0..n {
            out[i * channels + c] = planar[c * n + i];
        }
    }
    out
}

/// SNR of a group of frames against libopus, in dB.
#[derive(Debug, Clone, Copy)]
pub struct Group {
    pub frames: usize,
    pub worst: f64,
    pub median: f64,
    /// Number of frames below 40 dB.
    pub poor: usize,
}

impl Group {
    fn of(mut snrs: Vec<f64>) -> Self {
        snrs.sort_by(|a, b| a.total_cmp(b));
        Self {
            frames: snrs.len(),
            worst: snrs.first().copied().unwrap_or(200.0),
            median: snrs.get(snrs.len() / 2).copied().unwrap_or(200.0),
            poor: snrs.iter().filter(|s| **s < 40.0).count(),
        }
    }

    /// Share of the frames that are below 40 dB.
    pub fn poor_share(&self) -> f64 {
        self.poor as f64 / self.frames.max(1) as f64
    }
}

impl std::fmt::Display for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "median {:.1} dB, worst {:.1} dB, {} of {} below 40 dB",
            self.median, self.worst, self.poor, self.frames
        )
    }
}

/// How opus-rs compares with libopus over a stream with lost packets.
#[derive(Debug, Clone, Copy)]
pub struct LossStats {
    /// Number of lost frames.
    pub lost: usize,
    /// The concealed frames.
    pub concealed: Group,
    /// The first good frame after each loss. Its first 2.5 ms overlap with
    /// the concealed signal.
    pub first_good: Group,
    /// Good frames that start 40 ms to 120 ms after a loss. The concealed
    /// samples have left the post-filter's memory by then (21 ms), so only
    /// the decoder state that the concealment left behind still matters.
    pub recovered: Group,
    /// Largest difference in level between the two decoders in a concealed
    /// frame, in dB.
    pub level_diff_db: f64,
    /// Loudest concealed frame of libopus, in dBFS. It shows that the test
    /// compares real concealment and not silence.
    pub loudest_concealed_dbfs: f64,
}

impl std::fmt::Display for LossStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} lost; concealed: {}; first good frame: {}; recovered: {}; level differs by up to {:.2} dB; loudest concealed frame {:.1} dBFS",
            self.lost,
            self.concealed,
            self.first_good,
            self.recovered,
            self.level_diff_db,
            self.loudest_concealed_dbfs
        )
    }
}

/// Decode `packets` with libopus and with opus-rs, treating the packets
/// marked in `lost` as lost. Both decoders produce `frame` samples per
/// channel for every packet. Returns the two outputs, interleaved.
///
/// Both outputs are limited to [-1, 1]. `opus_decode_float` does not clip
/// and `OpusDecoder::decode` does (see `float_output_is_not_clipped` in
/// k24_known_gaps.rs). The first frames after a loss can overshoot full
/// scale, and without the limit this known difference would hide what the
/// tests are about. The limit acts on the output only, not on the state.
pub fn decode_with_loss(
    packets: &[Vec<u8>],
    lost: &[bool],
    rate: usize,
    channels: usize,
    frame: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut c_dec = CDecoder::new(rate, channels);
    let mut rs_dec = opus_rs::OpusDecoder::new(rate as i32, channels).unwrap();
    let (mut c, mut r) = (Vec::new(), Vec::new());
    let mut buf = vec![0f32; frame * channels];
    for (i, p) in packets.iter().enumerate() {
        let n = if lost[i] {
            c.extend(c_dec.decode(None, frame));
            rs_dec.decode(&[], frame, &mut buf).unwrap()
        } else {
            c.extend(c_dec.decode(Some(p), frame));
            rs_dec.decode(p, frame, &mut buf).unwrap()
        };
        assert_eq!(n, frame, "packet {i}");
        r.extend_from_slice(&buf);
    }
    assert_eq!(c.len(), r.len());
    for v in c.iter_mut().chain(r.iter_mut()) {
        *v = v.clamp(-1.0, 1.0);
    }
    (c, r)
}

/// Compare the two outputs of `decode_with_loss` frame by frame. `rate` is
/// the decoder's sampling rate and `frame` the frame length in samples per
/// channel.
///
/// A frame that is close to silent in both outputs (below -100 dBFS, with an
/// error below -140 dBFS) is left out: its SNR says nothing.
pub fn loss_stats(
    c: &[f32],
    r: &[f32],
    lost: &[bool],
    rate: usize,
    channels: usize,
    frame: usize,
) -> LossStats {
    let len = frame * channels;
    let (mut concealed, mut first_good, mut recovered) = (Vec::new(), Vec::new(), Vec::new());
    let (mut level_diff_db, mut loudest) = (0f64, -200f64);
    let mut n_lost = 0;
    // Number of good frames since the last loss.
    let mut since_loss = usize::MAX;
    for i in 0..lost.len() {
        let (cf, rf) = (&c[i * len..(i + 1) * len], &r[i * len..(i + 1) * len]);
        let err: Vec<f32> = cf.iter().zip(rf).map(|(a, b)| a - b).collect();
        let quiet = power(cf) < 1e-10 && power(&err) < 1e-14;
        let db = snr(cf, rf);
        if lost[i] {
            n_lost += 1;
            since_loss = 0;
            loudest = loudest.max(dbfs(cf));
            if !quiet {
                concealed.push(db);
                level_diff_db = level_diff_db.max((dbfs(cf) - dbfs(rf)).abs());
            }
        } else if since_loss != usize::MAX {
            since_loss += 1;
            // Time from the end of the loss to the start of this frame.
            let ms = (since_loss - 1) * frame * 1000 / rate;
            if quiet {
                continue;
            }
            if since_loss == 1 {
                first_good.push(db);
            } else if (40..120).contains(&ms) {
                recovered.push(db);
            }
        }
    }
    LossStats {
        lost: n_lost,
        concealed: Group::of(concealed),
        first_good: Group::of(first_good),
        recovered: Group::of(recovered),
        level_diff_db,
        loudest_concealed_dbfs: loudest,
    }
}
