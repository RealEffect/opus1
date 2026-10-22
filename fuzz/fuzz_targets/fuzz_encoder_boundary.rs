#![no_main]

use std::num::NonZeroU32;
use libfuzzer_sys::fuzz_target;
use opus1::{Application, Bitrate, Channels, InbandFec, OpusEncoder, SampleRate};

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    // Test all supported sample rates
    let sample_rates = [8000, 12000, 16000, 24000, 48000];

    for &sampling_rate in &sample_rates {
        for &channels in &[1, 2] {
            for &application in &[
                Application::Voip,
                Application::Audio,
                Application::RestrictedLowDelay,
                Application::RestrictedSilk,
                Application::RestrictedCelt,
            ] {
                let mut encoder = match OpusEncoder::new(SampleRate::try_from(sampling_rate).unwrap(), Channels::try_from(channels).unwrap(), application) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                // Test extreme bitrate values
                let bitrates: [u32; 8] = [
                    500,  // Very low
                    6000, // Minimum recommended
                    16000, 32000, 64000, 128000, 256000, 510000, // Maximum
                ];
                let bitrate = bitrates[data[0] as usize % bitrates.len()];
                encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate).unwrap()));

                // Test complexity values (0-10, with edge cases)
                let _ = encoder.set_complexity(match data[1] % 5 {
                    0 => 0,
                    1 => 5,
                    2 => 10,
                    3 => 11, // Invalid, should be handled
                    _ => data[1] % 11,
                });

                // Test packet loss percentage
                let _ = encoder.set_packet_loss_perc(match data[2] % 5 {
                    0 => 0,
                    1 => 50,
                    2 => 100,
                    3 => u8::MAX, // Invalid
                    _ => 101,     // Invalid
                });

                // Test FEC settings
                encoder.set_inband_fec(if data[3] % 2 == 0 { InbandFec::On } else { InbandFec::Off });
                encoder.set_vbr(data[4] % 2 != 0);

                // Test frame sizes based on sample rate
                let frame_sizes: Vec<usize> = match sampling_rate {
                    8000 => vec![80, 160, 320, 480, 640],
                    12000 => vec![120, 240, 480, 720, 960],
                    16000 => vec![160, 320, 640, 960, 1280],
                    24000 => vec![240, 480, 960, 1440, 1920],
                    _ => vec![120, 240, 480, 960, 1920, 2880],
                };

                for frame_size in &frame_sizes {
                    let samples_needed = frame_size * channels;
                    let mut input = vec![0.0f32; samples_needed];

                    // Generate various input patterns
                    let pattern = data[5] % 12;
                    for i in 0..samples_needed {
                        input[i] = match pattern {
                            // Silence
                            0 => 0.0,
                            // Maximum amplitude
                            1 => 1.0,
                            // Minimum amplitude
                            2 => -1.0,
                            // DC offset positive
                            3 => 0.5,
                            // DC offset negative
                            4 => -0.5,
                            // Sine wave
                            5 => {
                                let t = i as f32 / sampling_rate as f32;
                                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                            }
                            // High frequency
                            6 => {
                                let t = i as f32 / sampling_rate as f32;
                                (2.0 * std::f32::consts::PI * (sampling_rate as f32 / 4.0) * t)
                                    .sin()
                                    * 0.3
                            }
                            // Random from data
                            7 => {
                                let idx = (i + 6) % data.len();
                                (data[idx] as f32 - 128.0) / 128.0
                            }
                            // Impulses
                            8 => {
                                if i % 50 == 0 {
                                    0.9
                                } else {
                                    0.0
                                }
                            }
                            // Sawtooth
                            9 => ((i as f32 / 100.0) % 1.0 - 0.5) * 2.0,
                            // Square wave
                            10 => {
                                if i % 100 < 50 {
                                    0.5
                                } else {
                                    -0.5
                                }
                            }
                            // Near overflow
                            _ => {
                                let idx = (i + 6) % data.len();
                                let val = (data[idx] as f32 - 128.0) / 127.0;
                                val.clamp(-0.999, 0.999)
                            }
                        };

                        // Ensure valid values
                        if !input[i].is_finite() {
                            input[i] = 0.0;
                        }
                        input[i] = input[i].clamp(-1.0, 1.0);
                    }

                    // Encode with various output buffer sizes
                    let output_sizes = [1, 10, 100, 1000, 4000];
                    for &out_size in &output_sizes {
                        let mut output = vec![0u8; out_size];
                        let _ = encoder.encode(&input, *frame_size, &mut output);
                    }
                }
            }
        }
    }
});
