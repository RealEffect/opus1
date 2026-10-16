#![no_main]

use libfuzzer_sys::fuzz_target;
use opus1::{Application, OpusDecoder, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Parse parameters
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

    let frame_size = match data[3] % 4 {
        0 => sampling_rate as usize / 400, // 2.5ms
        1 => sampling_rate as usize / 200, // 5ms
        2 => sampling_rate as usize / 100, // 10ms
        _ => sampling_rate as usize / 50,  // 20ms
    };

    if frame_size < 40 {
        return;
    }

    let mut encoder = match OpusEncoder::new(sampling_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    let samples_needed = frame_size * channels;
    let mut input = vec![0.0f32; samples_needed];

    for (i, sample) in input.iter_mut().enumerate() {
        let offset = 4 + (i * 4) % (data.len() - 4).max(1);
        if offset + 4 <= data.len() {
            let raw = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            *sample = raw.clamp(-1.0, 1.0);
        }
    }

    let mut encoded = vec![0u8; 4000];
    // A valid frame size and enough room: the frame must be encoded.
    let encoded_len = encoder
        .encode(&input, frame_size, &mut encoded)
        .expect("a valid frame was not encoded");

    // The decoder reads the whole frame, and exactly the bits the encoder
    // wrote: both end in the same range coder state.
    let mut decoded = vec![0.0f32; frame_size * channels];
    let decoded_len = decoder.decode(&encoded[..encoded_len], frame_size, &mut decoded);
    assert_eq!(decoded_len, Ok(frame_size));
    assert_eq!(decoder.final_range(), encoder.final_range());
    assert!(decoded.iter().all(|x| x.is_finite()));
});
