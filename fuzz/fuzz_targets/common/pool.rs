//! Packets from opus1's encoder for the targets with the neural network
//! features (`fuzz_dred`, `fuzz_decoder_dnn`), made once per process.
//!
//! A made-up voice (a few harmonics of a gliding pitch, in syllables, with
//! some noise) is encoded with settings that reach the neural code of the
//! decoder: SILK wideband in 20 ms frames (LACE, NoLACE and the bandwidth
//! extension), Hybrid, SILK narrowband and CELT, with in-band FEC and with
//! DRED data in the padding.

use opus1::extensions::ExtensionIterator;
use opus1::packet;
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};
use std::sync::OnceLock;

/// The extension ID of DRED data.
pub const DRED_ID: u8 = 126;

pub struct Pool {
    /// Every packet, stream by stream.
    pub packets: Vec<Vec<u8>>,
    /// The packets that carry DRED data.
    pub dred: Vec<Vec<u8>>,
}

/// Whether `data` carries an extension with the DRED ID.
pub fn has_dred(data: &[u8]) -> bool {
    let Ok(p) = packet::parse(data, false) else { return false };
    matches!(ExtensionIterator::new(p.padding(data), p.count).find(DRED_ID), Ok(Some(_)))
}

/// `n` samples per channel of the made-up voice at `fs`.
fn voice(fs: i32, channels: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut rng = seed;
    let mut out = Vec::with_capacity(n * channels);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / fs as f32;
        let f0 = 110.0 + 60.0 * (t * 1.3).sin() + 30.0 * (t * 4.1).sin();
        phase += f0 / fs as f32;
        let env = (0.5 + 0.5 * (t * 2.0 * std::f32::consts::PI * 3.0).sin()).powi(2);
        let mut s = 0.0;
        for h in 1..8 {
            s += (2.0 * std::f32::consts::PI * phase * h as f32).sin() / h as f32;
        }
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (rng >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
        let s = 0.25 * env * s + 0.02 * noise;
        for c in 0..channels {
            out.push(if c == 0 { s } else { 0.8 * s });
        }
    }
    out
}

struct Setting {
    fs: i32,
    channels: usize,
    app: Application,
    bitrate: i32,
    ms2: usize,
    force: Option<ForcedMode>,
    dred: i32,
}

fn make() -> Pool {
    let settings = [
        // SILK wideband, 20 ms, FEC and DRED.
        Setting { fs: 16000, channels: 1, app: Application::Voip, bitrate: 32000, ms2: 40, force: None, dred: 100 },
        Setting { fs: 48000, channels: 1, app: Application::Voip, bitrate: 24000, ms2: 40, force: Some(ForcedMode::SilkOnly), dred: 40 },
        Setting { fs: 16000, channels: 2, app: Application::Voip, bitrate: 40000, ms2: 40, force: None, dred: 60 },
        // Hybrid.
        Setting { fs: 48000, channels: 2, app: Application::Voip, bitrate: 40000, ms2: 40, force: Some(ForcedMode::Hybrid), dred: 20 },
        // SILK in 10, 40 and 60 ms frames.
        Setting { fs: 16000, channels: 1, app: Application::Voip, bitrate: 14000, ms2: 20, force: None, dred: 0 },
        Setting { fs: 8000, channels: 1, app: Application::Voip, bitrate: 10000, ms2: 80, force: None, dred: 0 },
        Setting { fs: 48000, channels: 1, app: Application::Voip, bitrate: 20000, ms2: 120, force: Some(ForcedMode::SilkOnly), dred: 0 },
        // CELT.
        Setting { fs: 48000, channels: 2, app: Application::Audio, bitrate: 64000, ms2: 40, force: Some(ForcedMode::CeltOnly), dred: 0 },
        Setting { fs: 48000, channels: 1, app: Application::RestrictedLowDelay, bitrate: 48000, ms2: 10, force: None, dred: 0 },
    ];
    let mut packets = Vec::new();
    for (k, s) in settings.iter().enumerate() {
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(s.fs).unwrap(),
            Channels::try_from(s.channels).unwrap(),
            s.app,
        )
        .expect("encoder");
        enc.set_bitrate(Bitrate::BitsPerSecond(s.bitrate)).unwrap();
        enc.set_inband_fec(true);
        enc.set_packet_loss_perc(20).unwrap();
        enc.set_force_mode(s.force);
        if s.dred > 0 {
            enc.set_dred_duration(s.dred).unwrap();
        }
        let frame = s.fs as usize * s.ms2 / 2000;
        // 600 ms.
        let pcm = voice(s.fs, s.channels, frame * (1200 / s.ms2), 7 + k as u32);
        let mut out = vec![0u8; 4000];
        for chunk in pcm.chunks_exact(frame * s.channels) {
            let n = enc.encode(chunk, frame, &mut out).expect("encode");
            packets.push(out[..n].to_vec());
        }
    }
    let dred: Vec<Vec<u8>> = packets.iter().filter(|p| has_dred(p)).cloned().collect();
    assert!(dred.len() >= 20, "only {} packets carry DRED data", dred.len());
    Pool { packets, dred }
}

pub fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(make)
}
