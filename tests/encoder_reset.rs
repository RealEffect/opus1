//! `OpusEncoder::reset_state` in the middle of a stream.
//!
//! - After a reset the encoder writes exactly the packets that a new
//!   encoder with the same settings writes for the same input.
//! - A reset allocates nothing, as libopus `OPUS_RESET_STATE`: with the
//!   `heap` feature the new state goes into the heap memory the encoder
//!   already has. The same holds for the decoder and for the multistream
//!   encoder and decoder.
//!
//! The file runs with and without `heap`.

use opus1::{Application, Bitrate, Channels, InbandFec, OpusDecoder, OpusEncoder, SampleRate};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::num::NonZeroU32;

thread_local! {
    /// Whether this thread counts its allocator calls.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Calls that take memory: `alloc`, `alloc_zeroed`, `realloc`.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    /// Calls that give memory back: `dealloc`.
    static FREES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count(counter: &'static std::thread::LocalKey<Cell<usize>>) {
    // `try_with`: the allocator also runs while a thread shuts down, when
    // its thread-local values are gone.
    if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
        let _ = counter.try_with(|n| n.set(n.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(&FREES);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` with counting on and returns the allocations and the frees it
/// made.
fn allocator_calls(f: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.with(|n| n.set(0));
    FREES.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    (ALLOCATIONS.with(|n| n.get()), FREES.with(|n| n.get()))
}

/// `seconds` of a voice-like signal at `rate` Hz with `channels` channels:
/// a harmonic tone whose pitch glides, with pauses and some noise, and
/// another level on the second channel.
fn signal(rate: usize, channels: usize, seconds: f32) -> Vec<f32> {
    let mut seed = 7u32;
    let samples = (rate as f32 * seconds) as usize;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(samples * channels);
    for i in 0..samples {
        let t = i as f32 / rate as f32;
        let pitch = 140.0 + 60.0 * (2.0 * core::f32::consts::PI * 0.7 * t).sin();
        phase += 2.0 * core::f32::consts::PI * pitch / rate as f32;
        // Voiced for 0.4 s, then 0.1 s of a quiet pause.
        let voiced = (t * 2.0).fract() < 0.8;
        let tone: f32 = (1..6).map(|h| (phase * h as f32).sin() / h as f32).sum();
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let noise = (seed >> 16) as f32 / 65536.0 - 0.5;
        let x = if voiced { 0.25 * tone + 0.02 * noise } else { 0.002 * noise };
        for c in 0..channels {
            out.push(x * if c == 0 { 1.0 } else { 0.6 });
        }
    }
    out
}

/// Encodes `pcm` in frames of 20 ms and returns each packet followed by
/// the four bytes of its final range.
fn encode_all(encoder: &mut OpusEncoder, pcm: &[f32]) -> Vec<Vec<u8>> {
    let channels = encoder.channels().count();
    let frame_size = encoder.sampling_rate().hz() as usize / 50;
    pcm.chunks_exact(frame_size * channels)
        .map(|frame| {
            let mut packet = vec![0u8; 1500];
            let len = encoder.encode(frame, frame_size, &mut packet).unwrap();
            packet.truncate(len);
            packet.extend_from_slice(&encoder.final_range().to_le_bytes());
            packet
        })
        .collect()
}

/// The encoders of the tests: SILK, Hybrid and CELT, mono and stereo (a
/// stereo SILK encoder has a second channel state), with the speech and
/// music analysis (complexity 10) in the builds that have it.
const CASES: [(SampleRate, Channels, Application, u32, u8); 5] = [
    (SampleRate::Hz16000, Channels::Mono, Application::Voip, 20_000, 10),
    (SampleRate::Hz16000, Channels::Stereo, Application::Voip, 32_000, 5),
    (SampleRate::Hz48000, Channels::Stereo, Application::Voip, 40_000, 10),
    (SampleRate::Hz48000, Channels::Stereo, Application::Audio, 96_000, 10),
    (SampleRate::Hz24000, Channels::Mono, Application::RestrictedLowDelay, 32_000, 7),
];

/// The settings of every encoder of a case. A reset keeps them, except the
/// DRED duration (feature `dred`), which it sets to 0 as libopus does; so
/// the encoder that was reset gets them again.
///
/// In-band FEC stays off: libopus keeps across a reset whether the last
/// packet carried FEC, and with FEC on the first packets after a reset can
/// then differ from those of a new encoder (`core_encoder.rs` tests that
/// case against libopus).
fn configure(encoder: &mut OpusEncoder, bitrate: u32, complexity: u8) {
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate).unwrap()));
    encoder.set_complexity(complexity).unwrap();
    encoder.set_inband_fec(InbandFec::Off);
    encoder.set_packet_loss_perc(10).unwrap();
    // DRED runs only with an expected loss, which is set above.
    #[cfg(feature = "dred")]
    encoder.set_dred_duration(20).unwrap();
}

#[test]
fn a_reset_in_the_middle_of_a_stream_gives_a_new_encoder() {
    for (rate, channels, application, bitrate, complexity) in CASES {
        let label = format!("{rate:?}, {channels:?}, {application:?}");
        let pcm = signal(rate.hz() as usize, channels.count(), 2.0);
        // The halves have a whole number of frames of 20 ms.
        let (first, second) = pcm.split_at(pcm.len() / 2);
        let mut used = Box::new(OpusEncoder::new(rate, channels, application).unwrap());
        configure(&mut used, bitrate, complexity);
        encode_all(&mut used, first);
        used.reset_state();
        assert_eq!(used.final_range(), 0, "{label}");
        configure(&mut used, bitrate, complexity);
        let mut fresh = Box::new(OpusEncoder::new(rate, channels, application).unwrap());
        configure(&mut fresh, bitrate, complexity);
        assert_eq!(encode_all(&mut used, second), encode_all(&mut fresh, second), "{label}");
    }
}

#[test]
fn a_reset_allocates_nothing() {
    for (rate, channels, application, bitrate, complexity) in CASES {
        let label = format!("{rate:?}, {channels:?}, {application:?}");
        let pcm = signal(rate.hz() as usize, channels.count(), 0.4);
        let mut encoder = Box::new(OpusEncoder::new(rate, channels, application).unwrap());
        configure(&mut encoder, bitrate, complexity);
        encode_all(&mut encoder, &pcm);
        assert_eq!(allocator_calls(|| encoder.reset_state()), (0, 0), "{label}: encoder");
        // The encoder works on after the reset.
        assert_eq!(encode_all(&mut encoder, &pcm).len(), 20, "{label}");

        let mut decoder = Box::new(OpusDecoder::new(rate, channels).unwrap());
        let mut out = vec![0.0f32; rate.max_frame_size() * channels.count()];
        for packet in encode_all(&mut encoder, &pcm) {
            let packet = &packet[..packet.len() - 4];
            decoder.decode(packet, rate.max_frame_size(), &mut out).unwrap();
        }
        assert_eq!(allocator_calls(|| decoder.reset_state()), (0, 0), "{label}: decoder");
    }
}

/// The multistream codecs reset every stream codec, which allocates
/// nothing either.
#[cfg(feature = "std")]
#[test]
fn a_multistream_reset_allocates_nothing() {
    use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
    let rate = SampleRate::Hz48000;
    for (channels, family) in [(2, 0), (6, 1), (8, 1), (4, 255)] {
        let mut encoder = MultistreamEncoder::new_surround(rate, channels, family, Application::Audio).unwrap();
        let layout = encoder.layout();
        let mut decoder = MultistreamDecoder::with_layout(
            rate,
            layout.channels(),
            layout.streams(),
            layout.coupled_streams(),
            layout.mapping(),
        )
        .unwrap();
        let pcm = signal(48_000, channels, 0.2);
        let mut packet = vec![0u8; 1500 * channels];
        let mut out = vec![0.0f32; 960 * channels];
        for frame in pcm.chunks_exact(960 * channels) {
            let len = encoder.encode(frame, 960, &mut packet).unwrap();
            decoder.decode(&packet[..len], 960, &mut out).unwrap();
        }
        let label = format!("{channels} channels, mapping family {family}");
        assert_eq!(allocator_calls(|| encoder.reset_state()), (0, 0), "{label}: encoder");
        assert_eq!(allocator_calls(|| decoder.reset_state()), (0, 0), "{label}: decoder");
    }
}
