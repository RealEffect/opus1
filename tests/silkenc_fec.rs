//! In-band forward error correction (FEC) written by the SILK encoder,
//! against libopus.
//!
//! With FEC on, SILK keeps a low-bitrate copy of each frame (LBRR, low
//! bit-rate redundancy) and writes it at the start of the next packet.
//! libopus writes whatever copies it has stored, whether or not FEC is
//! still on for the packet that carries them (silk/enc_API.c:355-390).
//! opus1 wrote them only while FEC was on, so the last frame before FEC
//! was switched off, or before the bitrate dropped under the FEC threshold,
//! could not be recovered.

mod silkenc_common;
use silkenc_common::*;

use opus1::{Application, Bitrate, ForcedMode, OpusEncoder};

/// A change of settings before packet `at`.
#[derive(Clone, Copy)]
struct Step {
    at: usize,
    fec: bool,
    loss: i32,
    bitrate: i32,
}

struct Stream {
    fs: usize,
    channels: usize,
    /// Samples per channel in a packet.
    frame: usize,
    /// Constant bitrate: every packet has the size the bitrate gives.
    cbr: bool,
    steps: &'static [Step],
}

impl Stream {
    fn n_frames(&self) -> usize {
        // SILK frames per packet: 10 and 20 ms packets hold one.
        (self.frame * 50 / self.fs).max(1)
    }

    fn pcm(&self, packets: usize) -> Vec<f32> {
        speech(self.fs, self.channels, packets * self.frame, 0.5)
    }

    fn encode_rs(&self, pcm: &[f32]) -> Vec<Vec<u8>> {
        let mut enc = OpusEncoder::new(self.fs as i32, self.channels, Application::Audio).unwrap();
        enc.set_force_mode(Some(ForcedMode::SilkOnly));
        // The channel count libopus is forced to in `encode_c`.
        enc.set_force_channels(Some(self.channels)).unwrap();
        enc.set_vbr(!self.cbr);
        let mut out = vec![0u8; 1500];
        let mut packets = Vec::new();
        for (i, chunk) in pcm.chunks_exact(self.frame * self.channels).enumerate() {
            for s in self.steps.iter().filter(|s| s.at == i) {
                enc.set_inband_fec(s.fec);
                enc.set_packet_loss_perc(s.loss).unwrap();
                enc.set_bitrate(Bitrate::BitsPerSecond(s.bitrate)).unwrap();
            }
            let n = enc.encode(chunk, self.frame, &mut out).unwrap();
            packets.push(out[..n].to_vec());
        }
        packets
    }

    fn encode_c(&self, pcm: &[f32]) -> Vec<Vec<u8>> {
        let mut enc = CEncoder::new(self.fs, self.channels, APP_AUDIO);
        enc.silk_only().force_channels(self.channels as i32).vbr(!self.cbr);
        let mut packets = Vec::new();
        for (i, chunk) in pcm.chunks_exact(self.frame * self.channels).enumerate() {
            for s in self.steps.iter().filter(|s| s.at == i) {
                enc.fec(s.fec, s.loss).bitrate(s.bitrate);
            }
            packets.push(enc.encode(chunk));
        }
        packets
    }

    /// The LBRR flag of each channel in the header of a SILK packet: the
    /// packet starts with the frames' voice activity flags and the LBRR flag
    /// of the first channel, then those of the second.
    fn lbrr_flags(&self, packet: &[u8]) -> Vec<bool> {
        let n = self.n_frames();
        let silk = silk_payload(packet);
        let header = u16::from_be_bytes([*silk.first().unwrap_or(&0), *silk.get(1).unwrap_or(&0)]);
        (0..self.channels)
            .map(|c| (header >> (15 - (c * (n + 1) + n))) & 1 == 1)
            .collect()
    }

    /// The frame of packet `lost` as the libopus decoder rebuilds it from
    /// the next packet, and as it decodes it when nothing is lost.
    fn recover(&self, packets: &[Vec<u8>], lost: usize) -> (Vec<f32>, Vec<f32>) {
        let mut dec = CDecoder::new(self.fs, self.channels);
        let mut clean_dec = CDecoder::new(self.fs, self.channels);
        let mut out = (Vec::new(), Vec::new());
        for (i, p) in packets.iter().enumerate().take(lost + 2) {
            let clean = clean_dec.decode(Some(p), self.frame, false).unwrap();
            if i == lost {
                out.1 = clean;
                out.0 = dec.decode(Some(&packets[i + 1]), self.frame, true).unwrap();
            } else {
                dec.decode(Some(p), self.frame, false).unwrap();
            }
        }
        out
    }
}

/// The SILK data of a packet with one frame: what follows the first byte,
/// or, in a packet that was padded to a constant size (code 3), what lies
/// between the padding length and the padding.
fn silk_payload(packet: &[u8]) -> &[u8] {
    if packet[0] & 3 != 3 {
        return &packet[1..];
    }
    let mut start = 2;
    let mut padding = 0;
    if packet[1] & 0x40 != 0 {
        loop {
            let b = packet[start] as usize;
            start += 1;
            padding += if b == 255 { 254 } else { b };
            if b != 255 {
                break;
            }
        }
    }
    &packet[start..packet.len() - padding]
}

/// FEC off, on at packet 10, off at 30, on at 45, then a bitrate too low for
/// FEC at 5 % loss from packet 60, and back up at 75.
const STEPS_MONO: [Step; 6] = [
    Step { at: 0, fec: false, loss: 0, bitrate: 24_000 },
    Step { at: 10, fec: true, loss: 20, bitrate: 24_000 },
    Step { at: 30, fec: false, loss: 20, bitrate: 24_000 },
    Step { at: 45, fec: true, loss: 5, bitrate: 24_000 },
    Step { at: 60, fec: true, loss: 5, bitrate: 9_000 },
    Step { at: 75, fec: true, loss: 5, bitrate: 28_000 },
];
const STEPS_STEREO: [Step; 6] = [
    Step { at: 0, fec: false, loss: 0, bitrate: 40_000 },
    Step { at: 10, fec: true, loss: 20, bitrate: 40_000 },
    Step { at: 30, fec: false, loss: 20, bitrate: 40_000 },
    Step { at: 45, fec: true, loss: 5, bitrate: 40_000 },
    Step { at: 60, fec: true, loss: 5, bitrate: 14_000 },
    Step { at: 75, fec: true, loss: 5, bitrate: 48_000 },
];
const PACKETS: usize = 90;

fn streams() -> Vec<Stream> {
    let mut all = Vec::new();
    for (channels, steps) in [(1, &STEPS_MONO[..]), (2, &STEPS_STEREO[..])] {
        for ms in [10, 20, 40, 60] {
            all.push(Stream { fs: 16_000, channels, frame: 16 * ms, cbr: false, steps });
        }
    }
    all
}

/// Each packet says in its header whether it carries FEC data for the
/// previous packet. opus1 and libopus say the same for every packet of a
/// stream in which FEC is switched on and off and the bitrate changes.
#[test]
fn packets_carry_fec_data_where_libopus_packets_do() {
    let mut failures = Vec::new();
    let mut after_stop = 0;
    for s in streams() {
        let pcm = s.pcm(PACKETS);
        let rs = s.encode_rs(&pcm);
        let c = s.encode_c(&pcm);
        let flags = |packets: &[Vec<u8>]| -> Vec<Vec<bool>> {
            packets.iter().map(|p| s.lbrr_flags(p)).collect()
        };
        let (rs_flags, c_flags) = (flags(&rs), flags(&c));
        // The first channel (mono, or the mid of stereo) must agree in every
        // packet: its voice activity depends on the input alone.
        let differ: Vec<usize> =
            (0..PACKETS).filter(|&i| rs_flags[i][0] != c_flags[i][0]).collect();
        // The side of a stereo stream is what is left of left minus right
        // after prediction from the mid. That depends on how many bits the
        // earlier packets took, which is not the same in two encoders, so
        // its voice activity may differ in a few packets.
        let side_differ = (0..PACKETS).filter(|&i| rs_flags[i] != c_flags[i]).count();
        // Packets 30 and 60 are the first after FEC was switched off and
        // after the bitrate dropped under what FEC needs.
        after_stop += [30, 60].iter().filter(|&&i| c_flags[i][0]).count();
        if !differ.is_empty() || side_differ > differ.len() + 3 {
            failures.push(format!(
                "{} ch, {} ms: {} packets differ ({} with the side), first {:?} (libopus {:?}, opus1 {:?})",
                s.channels,
                s.frame / 16,
                differ.len(),
                side_differ,
                &differ[..differ.len().min(8)],
                differ.first().map(|&i| &c_flags[i]),
                differ.first().map(|&i| &rs_flags[i]),
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    // The streams must show what is tested.
    assert!(after_stop >= 5, "libopus wrote FEC data after FEC stopped in {after_stop} packets");
}

/// FEC on at packet 10, off at 30, on at 45, with 49 or 99 bytes for SILK in
/// every packet.
const STEPS_CBR_MONO: [Step; 4] = [
    Step { at: 0, fec: false, loss: 0, bitrate: 20_000 },
    Step { at: 10, fec: true, loss: 10, bitrate: 20_000 },
    Step { at: 30, fec: false, loss: 10, bitrate: 20_000 },
    Step { at: 45, fec: true, loss: 25, bitrate: 20_000 },
];
const STEPS_CBR_STEREO: [Step; 4] = [
    Step { at: 0, fec: false, loss: 0, bitrate: 40_000 },
    Step { at: 10, fec: true, loss: 10, bitrate: 40_000 },
    Step { at: 30, fec: false, loss: 10, bitrate: 40_000 },
    Step { at: 45, fec: true, loss: 25, bitrate: 40_000 },
];

/// With a constant bitrate the FEC data shares a packet of fixed size with
/// the frame. libopus writes it whatever room that leaves for the frame
/// (silk/enc_API.c:365-389). When the two do not fit, the packet carries no
/// audio and the decoder conceals the frame (opus_encoder.c:2582-2594).
/// opus1 left the FEC data out instead when less than 8 bytes would have
/// been left for the frame.
///
/// The two encoders do not exceed the size in the same packets, so the
/// packets are counted: those with FEC data and those without audio.
#[test]
fn fec_data_is_written_into_small_packets_as_in_libopus() {
    let mut failures = Vec::new();
    for (channels, steps) in [(1, &STEPS_CBR_MONO[..]), (2, &STEPS_CBR_STEREO[..])] {
        for ms in [10, 20, 40, 60] {
            let s = Stream { fs: 16_000, channels, frame: 16 * ms, cbr: true, steps };
            let pcm = s.pcm(PACKETS);
            let (rs, c) = (s.encode_rs(&pcm), s.encode_c(&pcm));
            let what = format!("{channels} ch, {ms} ms");
            let mut dec = CDecoder::new(s.fs, channels);
            // Packets with FEC data and packets without audio, opus1 and
            // libopus.
            let (mut with_fec, mut empty) = ([0, 0], [0, 0]);
            for (i, (r, c)) in rs.iter().zip(&c).enumerate() {
                assert_eq!(r.len(), c.len(), "{what}: size of packet {i}");
                dec.decode(Some(r), s.frame, false).unwrap();
                for (k, packet) in [r, c].into_iter().enumerate() {
                    with_fec[k] += s.lbrr_flags(packet)[0] as usize;
                    empty[k] += (silk_payload(packet).len() <= 1) as usize;
                }
            }
            // Shown with `--nocapture`.
            println!(
                "{what}: packets with FEC data opus1 {}, libopus {}; without audio {}, {}",
                with_fec[0], with_fec[1], empty[0], empty[1]
            );
            assert!(with_fec[1] >= 15, "{what}: libopus wrote FEC data in {} packets", with_fec[1]);
            if with_fec[0] * 10 < with_fec[1] * 8 || empty[0] > empty[1] + 4 {
                failures.push(format!(
                    "{what}: packets with FEC data opus1 {}, libopus {}; without audio {}, {}",
                    with_fec[0], with_fec[1], empty[0], empty[1]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Signal and error energy of recovered frames, summed over frames.
#[derive(Default, Clone, Copy)]
struct Energy {
    signal: f64,
    error: f64,
    frames: usize,
}

impl Energy {
    fn add(&mut self, clean: &[f32], recovered: &[f32]) {
        for (a, b) in clean.iter().zip(recovered) {
            self.signal += (*a as f64).powi(2);
            self.error += ((*a - *b) as f64).powi(2);
        }
        self.frames += 1;
    }

    fn snr_db(&self) -> f64 {
        10.0 * (self.signal / self.error.max(1e-30)).log10()
    }
}

/// Lost frames are recovered from the next packet as well as libopus's own
/// packets allow: in the middle of a stretch with FEC, right after FEC is
/// switched on, and for the last frame before FEC stops.
///
/// The measure is the signal-to-noise ratio of the frame that the libopus
/// decoder rebuilds from the next packet's FEC data, against the frame it
/// decodes when nothing is lost. A frame without FEC data is concealed and
/// does not follow the waveform. Single frames differ by several dB between
/// the two encoders, so the figures are taken over many frames.
#[test]
fn lost_frames_are_recovered_as_well_as_libopus_allows() {
    let mut failures = Vec::new();
    // Over all streams: the last frame before FEC stops (packets 29 and 59).
    let mut last = [Energy::default(); 2];
    for s in streams() {
        let pcm = s.pcm(PACKETS);
        let rs = s.encode_rs(&pcm);
        let c = s.encode_c(&pcm);
        let mut all = [Energy::default(); 2];
        for lost in 5..PACKETS - 2 {
            // Every frame for which libopus wrote FEC data.
            if !s.lbrr_flags(&c[lost + 1])[0] {
                continue;
            }
            for (k, packets) in [&rs, &c].into_iter().enumerate() {
                let (recovered, clean) = s.recover(packets, lost);
                all[k].add(&clean, &recovered);
                if lost == 29 || lost == 59 {
                    last[k].add(&clean, &recovered);
                }
            }
        }
        assert!(all[1].frames >= 10, "{} frames with FEC data", all[1].frames);
        // Shown with `--nocapture`.
        println!(
            "{} ch, {} ms, {} frames: opus1 {:.2} dB, libopus {:.2} dB",
            s.channels,
            s.frame / 16,
            all[1].frames,
            all[0].snr_db(),
            all[1].snr_db()
        );
        if all[0].snr_db() < all[1].snr_db() - 1.5 {
            failures.push(format!(
                "{} ch, {} ms, {} frames: opus1 {:.2} dB, libopus {:.2} dB",
                s.channels,
                s.frame / 16,
                all[1].frames,
                all[0].snr_db(),
                all[1].snr_db()
            ));
        }
    }
    assert!(last[1].frames >= 5, "{} frames before FEC stops", last[1].frames);
    println!(
        "last frame before FEC stops, {} frames: opus1 {:.2} dB, libopus {:.2} dB",
        last[1].frames,
        last[0].snr_db(),
        last[1].snr_db()
    );
    if last[0].snr_db() < last[1].snr_db() - 2.0 {
        failures.push(format!(
            "last frame before FEC stops, {} frames: opus1 {:.2} dB, libopus {:.2} dB",
            last[1].frames,
            last[0].snr_db(),
            last[1].snr_db()
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
