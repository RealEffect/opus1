//! Encodes two seconds of a 5.1 test signal with the multistream encoder,
//! decodes the packets with the multistream decoder, and prints the layout,
//! the packet sizes and the signal-to-noise ratio of every channel.
//!
//! Run it with `cargo run --release --example multistream`.

/// The multistream module needs the `std` feature, which is on by default.
#[cfg(not(feature = "std"))]
fn main() {
    println!("this example needs the `std` feature");
}

#[cfg(feature = "std")]
fn main() -> Result<(), &'static str> {
    use opus1::{Application, Bitrate};
    use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};

    const SAMPLE_RATE: i32 = 48_000;
    const CHANNELS: usize = 6;
    /// Samples per channel in one frame: 20 ms.
    const FRAME: usize = 960;
    /// Two seconds.
    const FRAMES: usize = 100;
    const BITRATE: i32 = 256_000;
    /// The channels of 5.1 in the order of mapping family 1.
    const NAMES: [&str; CHANNELS] = [
        "front left",
        "centre",
        "front right",
        "rear left",
        "rear right",
        "LFE",
    ];

    // Mapping family 1 is the surround family of the Ogg Opus header. With
    // 6 channels the encoder knows that the last one is the LFE channel.
    let mut encoder =
        MultistreamEncoder::new_surround(SAMPLE_RATE, CHANNELS, 1, Application::Audio)?;
    encoder.set_bitrate(Bitrate::BitsPerSecond(BITRATE))?;

    // The layout the encoder chose. A file format stores these numbers in
    // its header; here they go straight to the decoder.
    let layout = encoder.layout().clone();
    println!(
        "layout: {} channels in {} streams, the first {} of them stereo",
        layout.channels(),
        layout.streams(),
        layout.coupled_streams()
    );
    println!("mapping (channel -> coded channel): {:?}", layout.mapping());
    if let Some(stream) = encoder.lfe_stream() {
        println!("stream {stream} carries the LFE channel");
    }
    let mut decoder = MultistreamDecoder::with_layout(
        SAMPLE_RATE,
        layout.channels(),
        layout.streams(),
        layout.coupled_streams(),
        layout.mapping(),
    )?;

    // The test signal, interleaved: sample `i` of channel `c` is
    // `pcm[i * CHANNELS + c]`. Every channel has a note of its own with
    // three overtones that swells and fades. The LFE channel has a 50 Hz
    // tone.
    let mut pcm = vec![0f32; FRAMES * FRAME * CHANNELS];
    for (i, frame) in pcm.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        let turn = 2.0 * std::f32::consts::PI * t;
        for (c, sample) in frame.iter_mut().enumerate() {
            *sample = if c == CHANNELS - 1 {
                0.3 * (turn * 50.0).sin()
            } else {
                let note = 196.0 * 1.25f32.powi(c as i32);
                let loudness = 0.6 + 0.4 * (turn * (0.7 + 0.2 * c as f32)).sin();
                let tone = (turn * note).sin()
                    + 0.5 * (turn * 2.0 * note).sin()
                    + 0.25 * (turn * 3.0 * note).sin();
                0.15 * loudness * tone
            };
        }
    }

    // Encode and decode frame by frame. The length of `packet` is the
    // largest packet the encoder may write.
    let mut packet = vec![0u8; 1500 * layout.streams()];
    let mut decoded = vec![0f32; pcm.len()];
    let mut sizes = Vec::with_capacity(FRAMES);
    let step = FRAME * CHANNELS;
    for (input, output) in pcm.chunks_exact(step).zip(decoded.chunks_exact_mut(step)) {
        let len = encoder.encode(input, FRAME, &mut packet)?;
        let samples = decoder.decode(&packet[..len], FRAME, output)?;
        assert_eq!(samples, FRAME);
        sizes.push(len);
    }

    let total: usize = sizes.iter().sum();
    let seconds = (FRAMES * FRAME) as f64 / SAMPLE_RATE as f64;
    println!(
        "packets: {} of {} to {} bytes, {} bytes on average",
        sizes.len(),
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        total / sizes.len()
    );
    println!(
        "bitrate: {:.0} b/s, asked for {BITRATE} b/s",
        8.0 * total as f64 / seconds
    );

    // The decoded signal is the input, late by the lookahead of the
    // encoder. The first 0.2 s are left out: the codec starts from silence.
    let delay = encoder.lookahead();
    let skip = SAMPLE_RATE as usize / 5;
    println!("signal-to-noise ratio at a delay of {delay} samples:");
    for (c, name) in NAMES.iter().enumerate() {
        let (mut signal, mut noise) = (0f64, 0f64);
        for i in skip..FRAMES * FRAME - delay {
            let x = pcm[i * CHANNELS + c] as f64;
            let y = decoded[(i + delay) * CHANNELS + c] as f64;
            signal += x * x;
            noise += (x - y) * (x - y);
        }
        println!("  {name:<12} {:5.1} dB", 10.0 * (signal / noise).log10());
    }
    Ok(())
}
