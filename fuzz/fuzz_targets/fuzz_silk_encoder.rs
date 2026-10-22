#![no_main]

use std::num::NonZeroU32;
use libfuzzer_sys::fuzz_target;
use opus1::{Application, Bitrate, Channels, InbandFec, OpusEncoder, SampleRate};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // SILK supports 8, 12, and 16 kHz
    let sampling_rate = match data[0] % 3 {
        0 => 8000,
        1 => 12000,
        _ => 16000,
    };

    let channels = if data[1] % 2 == 0 { 1 } else { 2 };

    // Use VoIP mode for SILK
    let mut encoder = match OpusEncoder::new(SampleRate::try_from(sampling_rate).unwrap(), Channels::try_from(channels).unwrap(), Application::Voip) {
        Ok(e) => e,
        Err(_) => return,
    };

    // SILK frame sizes: 10, 20, 40, or 60 ms
    let frame_ms = match data[2] % 4 {
        0 => 10,
        1 => 20,
        2 => 40,
        _ => 60,
    };
    let frame_size = (sampling_rate as i64 * frame_ms / 1000) as usize;

    // Test various bitrates
    let bitrate = match data[3] % 8 {
        0 => 6000,
        1 => 10000,
        2 => 20000,
        3 => 32000,
        4 => 64000,
        5 => 96000,
        6 => 128000,
        _ => 510000,
    };
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate).unwrap()));

    // Test CBR/VBR
    encoder.set_vbr(data[4] % 2 != 0);

    // Test complexity
    let _ = encoder.set_complexity((data[5] % 11).wrapping_sub(1)); // 255 (refused), then 0 to 9

    // Test FEC settings
    encoder.set_inband_fec(if data[6] % 2 == 0 { InbandFec::On } else { InbandFec::Off });
    let _ = encoder.set_packet_loss_perc(data[7] % 101);

    // Generate input with various patterns
    let samples_needed = frame_size * channels;
    let mut input = vec![0.0f32; samples_needed];

    let pattern = data[0] % 6;
    for i in 0..samples_needed {
        input[i] = match pattern {
            // Sine wave
            0 => (2.0 * std::f32::consts::PI * (i as f32 % 100.0) / 100.0).sin() * 0.5,
            // White noise
            1 => {
                if data.len() > 8 {
                    ((data[8 + (i % (data.len() - 8))] as f32 - 128.0) / 128.0).clamp(-1.0, 1.0)
                } else {
                    ((i % 2) as f32 - 0.5) * 0.2
                }
            }
            // Impulses
            2 => {
                if i % 100 == 0 {
                    0.9
                } else {
                    0.0
                }
            }
            // DC offset
            3 => 0.5,
            // Alternating
            4 => {
                if i % 2 == 0 {
                    0.3
                } else {
                    -0.3
                }
            }
            // Random from data
            _ => {
                let idx = (i + 8) % data.len();
                (data[idx] as f32 - 128.0) / 128.0
            }
        };
    }

    // Encode
    let mut output = vec![0u8; 4000];
    let _ = encoder.encode(&input, frame_size, &mut output);
});
