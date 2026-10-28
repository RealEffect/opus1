//! The multistream encoder and decoder build each stream codec in place,
//! in the memory of their `Vec` (`OpusEncoder::init`, `OpusDecoder::init`).
//! So building them needs only a little stack, also without the `heap`
//! feature, where each stream codec holds 52 KB to 210 KB inline.
//!
//! These tests build them on a thread with a 32 KB stack (in a release
//! build), then use them on the main thread: they must code exactly like
//! codecs built on the main thread. `scripts/check_no_std.sh` runs this file
//! in a release build without `heap` (features `alloc,libm`), where the
//! test matters most.

#![cfg(feature = "alloc")]

use opus1::multistream::{ChannelMappingTable, MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, SampleRate};

/// The stack of the thread that builds the codecs. An unoptimized build
/// keeps every temporary on the stack, so it gets more.
const STACK: usize = if cfg!(debug_assertions) { 1 << 20 } else { 32 << 10 };

/// The 5.1 layout of mapping family 1: 4 streams, 2 of them coupled, in
/// Vorbis channel order.
const MAPPING_51: [u8; 6] = [0, 4, 1, 2, 3, 5];

/// Runs `build` on a thread with a stack of [`STACK`] bytes and returns
/// its result.
fn on_small_stack<T: Send + 'static>(build: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(STACK).spawn(build).unwrap().join().unwrap()
}

/// Every encoder constructor, with the layouts that need the most streams
/// of each kind.
fn encoders() -> Vec<MultistreamEncoder> {
    let rate = SampleRate::Hz48000;
    let table = ChannelMappingTable::new(6, 4, 2, &MAPPING_51);
    let layout = MultistreamEncoder::new_surround(rate, 6, 1, Application::Audio).unwrap().layout().clone();
    vec![
        MultistreamEncoder::new(rate, 2, 1, 1, &[0, 1], Application::Audio).unwrap(),
        MultistreamEncoder::new(rate, 6, 4, 2, &MAPPING_51, Application::Audio).unwrap(),
        MultistreamEncoder::from_mapping_table(rate, &table, Application::Voip).unwrap(),
        MultistreamEncoder::from_layout(rate, &layout, Application::Audio).unwrap(),
        // 5.1 and 7.1 surround, with an LFE stream and the surround
        // analysis.
        MultistreamEncoder::new_surround(rate, 6, 1, Application::Audio).unwrap(),
        MultistreamEncoder::new_surround(rate, 8, 1, Application::Audio).unwrap(),
        // One mono stream per channel.
        MultistreamEncoder::new_surround(rate, 16, 255, Application::Audio).unwrap(),
    ]
}

/// Every decoder constructor, for the layouts of [`encoders`].
fn decoders(encoders: &[MultistreamEncoder]) -> Vec<MultistreamDecoder> {
    let rate = SampleRate::Hz48000;
    let table = ChannelMappingTable::new(6, 4, 2, &MAPPING_51);
    let mut decoders = vec![
        MultistreamDecoder::new(rate, 2, 1, 1, &[0, 1]).unwrap(),
        MultistreamDecoder::new(rate, 6, 4, 2, &MAPPING_51).unwrap(),
        MultistreamDecoder::from_mapping_table(rate, &table).unwrap(),
    ];
    decoders.extend(encoders[3..].iter().map(|enc| MultistreamDecoder::from_layout(rate, enc.layout()).unwrap()));
    decoders
}

/// A tone with another pitch on each channel, `frames` frames of 20 ms at
/// 48 kHz.
fn input(channels: usize, frames: usize) -> Vec<f32> {
    (0..960 * frames * channels)
        .map(|i| {
            let t = (i / channels) as f32 / 48000.0;
            let pitch = 110.0 + 55.0 * (i % channels) as f32;
            0.2 * (2.0 * core::f32::consts::PI * pitch * t).sin()
        })
        .collect()
}

/// Encodes 10 frames with `encoder`, decodes them with `decoder`, and
/// returns the packets and the bits of the decoded samples.
fn code(encoder: &mut MultistreamEncoder, decoder: &mut MultistreamDecoder) -> (Vec<Vec<u8>>, Vec<u32>) {
    let channels = encoder.channels();
    assert_eq!(decoder.channels(), channels);
    let pcm = input(channels, 10);
    let mut packets = Vec::new();
    let mut samples = Vec::new();
    let mut out = vec![0.0f32; 960 * channels];
    for frame in pcm.chunks_exact(960 * channels) {
        let mut packet = vec![0u8; 1500 * channels];
        let len = encoder.encode(frame, 960, &mut packet).unwrap();
        packet.truncate(len);
        assert_eq!(decoder.decode(&packet, 960, &mut out).unwrap(), 960);
        samples.extend(out.iter().map(|s| s.to_bits()));
        packets.push(packet);
    }
    (packets, samples)
}

#[test]
fn constructors_need_little_stack() {
    // Without `heap` a stream codec is far larger than the stack of the
    // thread that builds it (in a release build), so building it on that
    // stack first would overflow.
    #[cfg(not(feature = "heap"))]
    if !cfg!(debug_assertions) {
        assert!(core::mem::size_of::<opus1::OpusEncoder>() > STACK);
        assert!(core::mem::size_of::<opus1::OpusDecoder>() > STACK);
    }
    let (mut small_encoders, mut small_decoders) = on_small_stack(|| {
        let encoders = encoders();
        let decoders = decoders(&encoders);
        (encoders, decoders)
    });
    let mut encoders = encoders();
    let mut decoders = decoders(&encoders);
    assert_eq!(small_encoders.len(), small_decoders.len());
    for (i, (small, (encoder, decoder))) in small_encoders
        .iter_mut()
        .zip(&mut small_decoders)
        .zip(encoders.iter_mut().zip(&mut decoders))
        .enumerate()
    {
        let label = format!("codec pair {i}, {} channels", encoder.channels());
        assert_eq!(small.0.layout(), encoder.layout(), "{label}");
        assert_eq!(small.0.lfe_stream(), encoder.lfe_stream(), "{label}");
        assert_eq!(code(small.0, small.1), code(encoder, decoder), "{label}");
    }
}

#[test]
fn reset_needs_little_stack() {
    // Two sets of codecs with the same history. One is reset on the small
    // stack, the other on the main thread.
    let mut encoders = encoders();
    let mut decoders = decoders(&encoders);
    let mut other_encoders = self::encoders();
    let mut other_decoders = self::decoders(&other_encoders);
    for (encoder, decoder) in encoders.iter_mut().zip(&mut decoders) {
        code(encoder, decoder);
    }
    for (encoder, decoder) in other_encoders.iter_mut().zip(&mut other_decoders) {
        code(encoder, decoder);
        encoder.reset_state();
        decoder.reset_state();
    }
    // A reset writes every stream codec anew, in place.
    let (mut encoders, mut decoders) = on_small_stack(move || {
        for encoder in &mut encoders {
            encoder.reset_state();
        }
        for decoder in &mut decoders {
            decoder.reset_state();
        }
        (encoders, decoders)
    });
    for (i, ((encoder, decoder), (other_encoder, other_decoder))) in encoders
        .iter_mut()
        .zip(&mut decoders)
        .zip(other_encoders.iter_mut().zip(&mut other_decoders))
        .enumerate()
    {
        assert_eq!(code(encoder, decoder), code(other_encoder, other_decoder), "codec pair {i} after a reset");
    }
}
