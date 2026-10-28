//! `OpusDecoder::init`: a decoder built in memory that the caller provides
//! (the idea of upstream opus-rs PR #65). A decoder built with `init` must
//! decode exactly like one built with `new`, and building it must need
//! only a little stack, even without the `heap` feature.

use core::mem::MaybeUninit;
use opus1::{Application, Channels, ErrorKind, OpusDecoder, OpusEncoder, SampleRate};

/// 50 packets of 20 ms: a sine per channel, with another pitch on the
/// second channel.
fn packets(sampling_rate: i32, channels: usize, application: Application) -> Vec<Vec<u8>> {
    let frame_size = sampling_rate as usize / 50;
    let mut encoder = Box::new(OpusEncoder::new(SampleRate::try_from(sampling_rate).unwrap(), Channels::try_from(channels).unwrap(), application).unwrap());
    let mut packets = Vec::new();
    for frame in 0..50 {
        let input: Vec<f32> = (0..frame_size * channels)
            .map(|i| {
                let t = (frame * frame_size + i / channels) as f32 / sampling_rate as f32;
                let pitch = 220.0 + 110.0 * (i % channels) as f32;
                0.3 * (2.0 * core::f32::consts::PI * pitch * t).sin()
            })
            .collect();
        let mut output = vec![0u8; 1276];
        let length = encoder.encode(&input, frame_size, &mut output).unwrap();
        packets.push(output[..length].to_vec());
    }
    packets
}

/// Decodes `packets` with every fifth one lost, and returns the bits of
/// the output samples.
fn decode_all(
    decoder: &mut OpusDecoder,
    packets: &[Vec<u8>],
    frame_size: usize,
    channels: usize,
) -> Vec<u32> {
    let mut out = Vec::new();
    let mut pcm = vec![0.0f32; frame_size * channels];
    for (i, packet) in packets.iter().enumerate() {
        let packet: &[u8] = if i % 5 == 3 { &[] } else { packet };
        let n = decoder.decode(packet, frame_size, &mut pcm).unwrap();
        out.extend(pcm[..n * channels].iter().map(|v| v.to_bits()));
    }
    out
}

#[test]
fn init_decodes_like_new() {
    for &(sampling_rate, channels, application) in &[
        (8000, 1, Application::Voip),
        (16000, 2, Application::Voip),
        (48000, 1, Application::Audio),
        (48000, 2, Application::Audio),
        (48000, 2, Application::Voip),
    ] {
        let frame_size = sampling_rate as usize / 50;
        let packets = packets(sampling_rate, channels, application);
        // Complexity 10 runs deep PLC and OSCE in the builds that have them.
        for complexity in [0, 10] {
            let mut expected = Box::new(OpusDecoder::new(SampleRate::try_from(sampling_rate).unwrap(), Channels::try_from(channels).unwrap()).unwrap());
            let mut slot = Box::<OpusDecoder>::new_uninit();
            let actual = OpusDecoder::init(
                &mut slot,
                SampleRate::try_from(sampling_rate).unwrap(),
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            for decoder in [&mut *expected, &mut *actual] {
                decoder.set_complexity(complexity).unwrap();
                #[cfg(feature = "osce-bwe")]
                decoder.set_osce_bwe(true);
            }
            assert_eq!(
                decode_all(actual, &packets, frame_size, channels),
                decode_all(&mut expected, &packets, frame_size, channels),
                "{sampling_rate} Hz, {channels} channels, complexity {complexity}"
            );
            // SAFETY: `init` initialized the slot, and `actual` is not used
            // any more.
            unsafe { slot.assume_init_drop() };
        }
    }
}

/// `init` takes a `SampleRate` and a `Channels`, so a rate or a channel
/// count that Opus does not have cannot reach it: the conversions refuse
/// them, as libopus `opus_decoder_init` does.
#[test]
fn invalid_settings_do_not_convert() {
    assert_eq!(SampleRate::try_from(44100u32).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
    assert_eq!(Channels::try_from(3).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
}

/// Builds a decoder with `init` and decodes on a thread with a 32 KB stack
/// (in a release build). Without `heap` that stack is smaller than the
/// decoder itself (52 KB, 153 KB with the neural network features), so
/// `new`, which returns the decoder through the stack, would overflow it;
/// it needs 256 KB there (512 KB with the neural network features).
///
/// The first decoder of a program also builds the shared tables (the CELT
/// mode and the built-in neural network models); if this test runs first,
/// that happens on the small stack too.
#[test]
fn init_needs_little_stack() {
    // An unoptimized build keeps every temporary on the stack.
    const STACK: usize = if cfg!(debug_assertions) { 1 << 20 } else { 32 << 10 };
    #[cfg(not(feature = "heap"))]
    if !cfg!(debug_assertions) {
        assert!(core::mem::size_of::<OpusDecoder>() > STACK);
    }
    let packets = packets(48000, 2, Application::Audio);
    let expected = decode_all(&mut Box::new(OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap()), &packets, 960, 2);
    let actual = std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || {
            let mut slot = Box::<OpusDecoder>::new_uninit();
            let decoder =
                OpusDecoder::init(&mut slot, SampleRate::Hz48000, Channels::Stereo).unwrap();
            let out = decode_all(decoder, &packets, 960, 2);
            // SAFETY: `init` initialized the slot, and `decoder` is not used
            // any more.
            unsafe { slot.assume_init_drop() };
            out
        })
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(actual, expected);
}

/// The way a program without a heap keeps its decoder: in a `static`.
#[test]
fn init_in_a_static() {
    static mut DECODER: MaybeUninit<OpusDecoder> = MaybeUninit::uninit();
    let slot = &raw mut DECODER;
    // SAFETY: only this test uses `DECODER`, once.
    let decoder =
        OpusDecoder::init(unsafe { &mut *slot }, SampleRate::Hz16000, Channels::Mono).unwrap();
    let packets = packets(16000, 1, Application::Voip);
    let expected = decode_all(&mut Box::new(OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap()), &packets, 320, 1);
    assert_eq!(decode_all(decoder, &packets, 320, 1), expected);
}
