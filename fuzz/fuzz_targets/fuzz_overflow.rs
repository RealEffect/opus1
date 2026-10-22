#![no_main]

use std::num::NonZeroU32;
use libfuzzer_sys::fuzz_target;
use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    // Test with extreme values
    let sampling_rate = match data[0] % 5 {
        0 => 8000,
        1 => 12000,
        2 => 16000,
        3 => 24000,
        _ => 48000,
    };

    let channels = if data[1] % 2 == 0 { 1 } else { 2 };
    let application = match data[2] % 3 {
        0 => Application::Voip,
        1 => Application::Audio,
        _ => Application::RestrictedLowDelay,
    };

    // Create encoder
    let mut encoder = match OpusEncoder::new(SampleRate::try_from(sampling_rate).unwrap(), Channels::try_from(channels).unwrap(), application) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Extreme settings. As in libopus, a bitrate is limited to 500 b/s at
    // least and 750,000 b/s per channel at most, and a complexity above 10
    // and a loss above 100 are refused: the encoder keeps its setting.
    let bitrate = match data[3] % 8 {
        0 => NonZeroU32::MIN,
        1 => NonZeroU32::new(2).unwrap(),
        2 => NonZeroU32::new(499).unwrap(),
        3 => NonZeroU32::new(500).unwrap(),
        4 => NonZeroU32::new(64000).unwrap(),
        5 => NonZeroU32::new(510000).unwrap(),
        6 => NonZeroU32::new(i32::MAX as u32).unwrap(),
        _ => NonZeroU32::MAX,
    };
    encoder.set_bitrate(Bitrate::BitsPerSecond(bitrate));
    // At least 500 b/s, and at most 750,000 b/s per channel.
    let limited = bitrate.get().clamp(500, 750_000 * channels as u32);
    assert_eq!(encoder.bitrate_setting(), Bitrate::BitsPerSecond(NonZeroU32::new(limited).unwrap()));

    let complexity = match data[4] % 6 {
        0 => 0,
        1 => 5,
        2 => 10,
        3 => 11,
        4 => 200,
        _ => u8::MAX,
    };
    let accepted = encoder.set_complexity(complexity).is_ok();
    assert_eq!(accepted, complexity <= 10, "set_complexity({complexity})");
    assert_eq!(encoder.complexity(), if accepted { complexity } else { 9 });

    let loss = match data[5] % 5 {
        0 => 0,
        1 => 50,
        2 => 100,
        3 => 101,
        _ => u8::MAX,
    };
    let accepted = encoder.set_packet_loss_perc(loss).is_ok();
    assert_eq!(accepted, loss <= 100, "set_packet_loss_perc({loss})");
    assert_eq!(encoder.packet_loss_perc(), if accepted { loss } else { 0 });

    // Determine frame size - test edge cases
    // Use valid frame sizes based on sampling rate
    let frame_size = match data[6] % 8 {
        0 => sampling_rate as usize / 400, // 2.5ms
        1 => sampling_rate as usize / 200, // 5ms
        2 => sampling_rate as usize / 100, // 10ms
        3 => sampling_rate as usize / 50,  // 20ms
        4 => sampling_rate as usize / 25,  // 40ms
        5 => sampling_rate as usize / 16,  // 60ms
        6 => sampling_rate as usize / 100, // 10ms (valid)
        _ => sampling_rate as usize / 50,  // 20ms (valid)
    };

    // Skip if frame_size is 0 or too small (SILK needs at least 80 samples for 8kHz)
    let min_frame_size = sampling_rate as usize / 100; // 10ms minimum
    if frame_size < min_frame_size {
        return;
    }

    // Generate extreme input values
    let samples_needed = frame_size * channels;
    let mut input = vec![0.0f32; samples_needed];

    let extreme_pattern = data[7] % 10;
    for i in 0..samples_needed {
        input[i] = match extreme_pattern {
            // All zeros
            0 => 0.0,
            // Maximum positive
            1 => 1.0,
            // Maximum negative
            2 => -1.0,
            // Almost overflow (f32)
            3 => 3.4e38,
            // Very small positive
            4 => f32::MIN_POSITIVE,
            // Very small negative
            5 => -f32::MIN_POSITIVE,
            // NaN (should be handled)
            6 => f32::NAN,
            // Infinity
            7 => f32::INFINITY,
            // Negative infinity
            8 => f32::NEG_INFINITY,
            // Random extreme
            _ => {
                let idx = (i + 8) % data.len().max(1);
                let raw = f32::from_le_bytes([
                    data[idx % data.len()],
                    data[(idx + 1) % data.len()],
                    data[(idx + 2) % data.len()],
                    data[(idx + 3) % data.len()],
                ]);
                raw
            }
        };
    }

    // Clamp non-finite values before encoding
    for sample in input.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        }
        *sample = sample.clamp(-10.0, 10.0);
    }

    // Encode with small output buffer to test buffer handling
    let output_size = match data[8] % 5 {
        0 => 1, // Very small
        1 => 10,
        2 => 100,
        3 => 1000,
        _ => 4000, // Normal
    };
    let mut output = vec![0u8; output_size];

    // This should not panic even with extreme values
    let encode_result = encoder.encode(&input, frame_size, &mut output);

    // If encoding succeeded, try decoding
    if let Ok(encoded_len) = encode_result {
        if encoded_len > 0 && encoded_len <= output_size {
            let mut decoder = match OpusDecoder::new(SampleRate::try_from(sampling_rate).unwrap(), Channels::try_from(channels).unwrap()) {
                Ok(d) => d,
                Err(_) => return,
            };

            // Whatever the settings, the packet decodes to one frame and
            // the decoder ends in the encoder's final range.
            let mut decoded = vec![0.0f32; frame_size * channels];
            let got = decoder.decode(&output[..encoded_len], frame_size, &mut decoded);
            assert_eq!(got, Ok(frame_size));
            assert_eq!(decoder.final_range(), encoder.final_range());
        }
    }
});
