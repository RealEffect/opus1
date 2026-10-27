//! `OpusEncoder::init`: an encoder built in memory that the caller
//! provides, as `OpusDecoder::init` builds a decoder (see
//! `decoder_init.rs`). An encoder built with `init` must encode exactly like
//! one built with `new`, and building it must need only a little stack,
//! even without the `heap` feature.

use core::mem::MaybeUninit;
use opus1::{Application, Channels, OpusEncoder, SampleRate};

/// `frames` frames of 20 ms: a sine per channel, with another pitch on the
/// second channel, and some noise so that SILK and the analysis have work.
fn input(sampling_rate: SampleRate, channels: Channels, frames: usize) -> Vec<Vec<f32>> {
    let rate = sampling_rate.hz() as usize;
    let frame_size = rate / 50;
    let channels = channels.count();
    let mut seed = 1u32;
    (0..frames)
        .map(|frame| {
            (0..frame_size * channels)
                .map(|i| {
                    let t = (frame * frame_size + i / channels) as f32 / rate as f32;
                    let pitch = 220.0 + 110.0 * (i % channels) as f32;
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let noise = (seed >> 16) as f32 / 65536.0 - 0.5;
                    0.3 * (2.0 * core::f32::consts::PI * pitch * t).sin() + 0.02 * noise
                })
                .collect()
        })
        .collect()
}

/// Encodes every frame of `input` and returns each packet followed by the
/// four bytes of its final range.
fn encode_all(encoder: &mut OpusEncoder, input: &[Vec<f32>]) -> Vec<Vec<u8>> {
    let frame_size = encoder.sampling_rate().hz() as usize / 50;
    input
        .iter()
        .map(|pcm| {
            let mut packet = vec![0u8; 1276];
            let len = encoder.encode(pcm, frame_size, &mut packet).unwrap();
            packet.truncate(len);
            packet.extend_from_slice(&encoder.final_range().to_le_bytes());
            packet
        })
        .collect()
}

/// The same settings on both encoders.
fn configure(encoder: &mut OpusEncoder, complexity: u8) {
    encoder.set_complexity(complexity).unwrap();
    encoder.set_packet_loss_perc(10).unwrap();
    // DRED runs only with an expected loss, which is set above.
    #[cfg(feature = "dred")]
    encoder.set_dred_duration(20).unwrap();
}

#[test]
fn init_encodes_like_new() {
    for &(sampling_rate, channels, application) in &[
        (SampleRate::Hz8000, Channels::Mono, Application::Voip),
        (SampleRate::Hz16000, Channels::Stereo, Application::Voip),
        (SampleRate::Hz24000, Channels::Mono, Application::Audio),
        (SampleRate::Hz48000, Channels::Mono, Application::Audio),
        (SampleRate::Hz48000, Channels::Stereo, Application::Audio),
        (SampleRate::Hz48000, Channels::Stereo, Application::RestrictedLowDelay),
    ] {
        let input = input(sampling_rate, channels, 50);
        // Complexity 10 runs the speech and music analysis in the builds
        // that have it.
        for complexity in [0, 10] {
            let mut expected = Box::new(OpusEncoder::new(sampling_rate, channels, application).unwrap());
            let mut slot = Box::<OpusEncoder>::new_uninit();
            let actual = OpusEncoder::init(&mut slot, sampling_rate, channels, application).unwrap();
            configure(&mut expected, complexity);
            configure(actual, complexity);
            assert_eq!(
                encode_all(actual, &input),
                encode_all(&mut expected, &input),
                "{sampling_rate:?}, {channels:?}, {application:?}, complexity {complexity}"
            );
            // After a reset both go on alike.
            actual.reset_state();
            expected.reset_state();
            assert_eq!(encode_all(actual, &input[..10]), encode_all(&mut expected, &input[..10]));
            // SAFETY: `init` initialized the slot, and `actual` is not used
            // any more.
            unsafe { slot.assume_init_drop() };
        }
    }
}

/// Builds an encoder with `init` and resets it on a thread with a 32 KB
/// stack (in a release build). Without `heap` that stack is much smaller
/// than the encoder itself (150 KB, 210 KB with `dred`), so `new`, which
/// returns the encoder through the stack, would overflow it; it needs more
/// than 256 KB there.
///
/// The encoder is then used on the main thread, because encoding needs
/// more stack than building: about 100 KB, see docs/guide/06.
///
/// The first encoder of a program also builds the shared tables (the CELT
/// mode, and the built-in models of DRED with that feature); if this test
/// runs first, that happens on the small stack too.
#[test]
fn init_needs_little_stack() {
    // An unoptimized build keeps every temporary on the stack.
    const STACK: usize = if cfg!(debug_assertions) { 1 << 20 } else { 32 << 10 };
    #[cfg(not(feature = "heap"))]
    if !cfg!(debug_assertions) {
        assert!(core::mem::size_of::<OpusEncoder>() > STACK);
    }
    let input = input(SampleRate::Hz48000, Channels::Stereo, 20);
    let mut expected = Box::new(OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap());
    let expected = encode_all(&mut expected, &input);
    let mut slot = std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(|| {
            let mut slot = Box::<OpusEncoder>::new_uninit();
            let encoder =
                OpusEncoder::init(&mut slot, SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
            // A reset writes the new state in place too.
            encoder.reset_state();
            slot
        })
        .unwrap()
        .join()
        .unwrap();
    // SAFETY: `init` initialized the slot on the other thread.
    let encoder = unsafe { slot.assume_init_mut() };
    assert_eq!(encode_all(encoder, &input), expected);
    // SAFETY: `init` initialized the slot, and `encoder` is not used any
    // more.
    unsafe { slot.assume_init_drop() };
}

/// The way a program without a heap keeps its encoder: in a `static`.
#[test]
fn init_in_a_static() {
    static mut ENCODER: MaybeUninit<OpusEncoder> = MaybeUninit::uninit();
    let slot = &raw mut ENCODER;
    // SAFETY: only this test uses `ENCODER`, once.
    let encoder =
        OpusEncoder::init(unsafe { &mut *slot }, SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    let input = input(SampleRate::Hz16000, Channels::Mono, 20);
    let mut expected = Box::new(OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap());
    assert_eq!(encode_all(encoder, &input), encode_all(&mut expected, &input));
}
