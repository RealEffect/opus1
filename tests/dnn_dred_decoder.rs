//! The DRED decoder without libopus: malformed DRED data does not panic,
//! the argument rules, the loading of weights, and that parsing,
//! processing and decoding allocate nothing.
//!
//! The comparisons with libopus are in `dnn_dred_decode_libopus.rs`. This
//! binary has its own global allocator, which counts the calls a thread
//! makes while that thread has counting switched on (as
//! `dnn_no_alloc.rs` does).
#![cfg(all(feature = "dred", feature = "std"))]

use opus1::extensions::Extension;
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::dred::{DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID};
use opus1::{Application, Bitrate, DnnBlob, OpusDecoder, OpusDred, OpusDredDecoder, OpusEncoder, packet};
use std::alloc::{GlobalAlloc, Layout as MemoryLayout, System};
use std::cell::Cell;

thread_local! {
    /// Whether this thread counts its allocator calls.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Calls that take memory: `alloc`, `alloc_zeroed`, `realloc`.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count() {
    if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: MemoryLayout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: MemoryLayout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: MemoryLayout, new_size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: MemoryLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` with counting on and returns the number of allocations.
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(|n| n.get())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Packets of opus1's encoder (without DRED): `frame` samples per channel
/// each.
fn encode(fs: i32, channels: usize, application: Application, bitrate: i32, frame: usize, n: usize) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(fs, channels, application).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
    let mut out = Vec::new();
    for k in 0..n {
        let pcm: Vec<f32> = (0..frame * channels)
            .map(|i| {
                let t = (k * frame + i / channels) as f32 / fs as f32;
                0.3 * (2.0 * std::f32::consts::PI * (180.0 + 40.0 * (3.0 * t).sin()) * t).sin()
            })
            .collect();
        let mut buf = vec![0u8; 1500];
        let len = enc.encode(&pcm, frame, &mut buf).unwrap();
        buf.truncate(len);
        out.push(buf);
    }
    out
}

/// `plain` with `payload` as the DRED data of frame `frame`, after the
/// DRED header.
fn with_dred(plain: &[u8], payload: &[u8], frame: usize) -> Vec<u8> {
    let mut data = vec![b'D', DRED_EXPERIMENTAL_VERSION];
    data.extend_from_slice(payload);
    let mut buf = plain.to_vec();
    buf.resize(plain.len() + data.len() + 64, 0);
    let ext = [Extension { id: DRED_EXTENSION_ID, frame, data: &data }];
    let new_len = buf.len();
    let n = packet::pad_with_extensions(&mut buf, plain.len(), new_len, false, &ext).unwrap();
    buf.truncate(n);
    buf
}

/// A random DRED payload. Its first bytes are often small, so that the
/// header asks for many latent vectors.
fn random_payload(rng: &mut Rng) -> Vec<u8> {
    let len = rng.next() as usize % 700;
    let mut p: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
    if len > 2 && rng.next().is_multiple_of(2) {
        p[0] &= 0x0F;
        p[1] &= 0x0F;
    }
    p
}

#[test]
fn malformed_dred_data_does_not_panic() {
    let r = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut rng = Rng(7);
    let streams = [
        (16000, 1, encode(16000, 1, Application::Voip, 24000, 320, 12)),
        (48000, 2, encode(48000, 2, Application::Audio, 64000, 480, 12)),
        (16000, 1, encode(16000, 1, Application::Voip, 24000, 640, 12)),
    ];
    let mut with_latents = 0;
    for (fs, channels, packets) in &streams {
        let (fs, channels) = (*fs, *channels);
        let mut dec = OpusDecoder::new(fs, channels).unwrap();
        dec.set_complexity(10).unwrap();
        let mut pcm = vec![0f32; 48000 * 2];
        let mut pcm16 = vec![0i16; 48000 * 2];
        let mut pcm24 = vec![0i32; 48000 * 2];
        for n in 0..300 {
            let plain = &packets[n % packets.len()];
            let frames = packet::parse(plain, false).unwrap().count;
            let data = with_dred(plain, &random_payload(&mut rng), rng.next() as usize % frames);
            let max = (rng.next() % 100_000) as i32 - 20_000;
            let rate = [8000, 16000, 44100, 48000, 1, i32::MAX][rng.next() as usize % 6];
            let defer = rng.next().is_multiple_of(2);
            let got = r.parse(&mut dred, &data, max, rate, defer).unwrap();
            assert!(got.samples >= 0 && got.end >= 0);
            if dred.is_parsed() {
                r.process_in_place(&mut dred).unwrap();
                assert!(dred.features().iter().all(|x| x.is_finite()));
                if dred.nb_latents() > 0 {
                    with_latents += 1;
                }
            }
            let f2_5 = fs as usize / 400;
            let frame = f2_5 * (1 + rng.next() as usize % 160);
            let offset = (rng.next() % 200_000) as i32 - 50_000;
            let samples = match n % 3 {
                0 => dec.dred_decode_float(&dred, offset, frame, &mut pcm).unwrap(),
                1 => dec.dred_decode(&dred, offset, frame, &mut pcm16).unwrap(),
                _ => dec.dred_decode24(&dred, offset, frame, &mut pcm24).unwrap(),
            };
            assert_eq!(samples, frame);
            if n % 3 == 0 {
                assert!(pcm[..frame * channels].iter().all(|x| x.is_finite()));
            }
            dec.decode(plain, 2880, &mut pcm).unwrap();
        }
        // Random packets: errors or no DRED, never a panic.
        for _ in 0..3000 {
            let len = rng.next() as usize % 80;
            let mut data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            if len > 2 && rng.next().is_multiple_of(2) {
                data[0] |= 3;
                data[1] |= 0x40;
            }
            if let Ok(got) = r.parse(&mut dred, &data, 48000, 48000, false) {
                assert!(got.samples >= 0);
            }
        }
    }
    assert!(with_latents > 100, "{with_latents}");
}

#[test]
fn dred_decode_checks_its_arguments() {
    let packets = encode(16000, 1, Application::Voip, 24000, 320, 10);
    let r = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut rng = Rng(3);
    // A payload that gives latent vectors up to the start of the packet.
    let data = loop {
        let data = with_dred(&packets[9], &random_payload(&mut rng), 0);
        let got = r.parse(&mut dred, &data, 16000, 16000, false).unwrap();
        if got.samples > 0 && got.end == 0 && dred.nb_latents() > 2 {
            break data;
        }
    };
    assert!(dred.is_processed());
    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    let mut plain = OpusDecoder::new(16000, 1).unwrap();
    let mut a = vec![0f32; 960];
    let mut b = vec![0f32; 960];
    for p in &packets[..9] {
        dec.decode(p, 320, &mut a).unwrap();
        plain.decode(p, 320, &mut b).unwrap();
    }
    assert_eq!(dec.dred_decode_float(&dred, 320, 0, &mut a), Err("Invalid argument"));
    for frame in [1, 39, 41, 330] {
        assert!(dec.dred_decode_float(&dred, 320, frame, &mut a).is_err(), "{frame}");
    }
    assert!(dec.dred_decode_float(&dred, 320, 320, &mut a[..319]).is_err());
    let mut small = [0i16; 100];
    assert!(dec.dred_decode(&dred, 320, 320, &mut small).is_err());
    // The errors changed nothing: an unprocessed state conceals as a lost
    // packet does.
    let unprocessed = OpusDred::new();
    assert_eq!(dec.dred_decode_float(&unprocessed, 320, 320, &mut a), Ok(320));
    assert_eq!(plain.decode(&[], 320, &mut b), Ok(320));
    assert_eq!(a[..320], b[..320]);
    // With DRED features the concealment differs.
    assert_eq!(dec.dred_decode_float(&dred, 320, 320, &mut a), Ok(320));
    assert_eq!(plain.decode(&[], 320, &mut b), Ok(320));
    assert_ne!(a[..320], b[..320]);
    assert_eq!(dec.last_packet_duration(), 320);

    // Processing.
    let mut dst = OpusDred::new();
    assert_eq!(r.process(&OpusDred::new(), &mut dst), Err("Invalid argument"));
    assert_eq!(r.process_in_place(&mut OpusDred::new()), Err("Invalid argument"));
    let mut deferred = OpusDred::new();
    r.parse(&mut deferred, &data, 16000, 16000, true).unwrap();
    assert!(deferred.is_parsed() && !deferred.is_processed());
    r.process(&deferred, &mut dst).unwrap();
    assert!(dst.is_processed() && !deferred.is_processed());
    assert_eq!(dst.features(), dred.features());

    // Parsing: an empty packet, a packet without DRED, a bad rate.
    assert_eq!(r.parse(&mut dred, &[], 16000, 16000, false), Err("Invalid packet"));
    assert!(!dred.is_parsed());
    let got = r.parse(&mut dred, &packets[0], 16000, 16000, false).unwrap();
    assert_eq!((got.samples, got.end), (0, 0));
    assert_eq!(r.parse(&mut dred, &data, 16000, 0, false), Err("Invalid argument"));
    assert_eq!(r.parse(&mut dred, &data, 16000, -16000, false), Err("Invalid argument"));
    // A DRED extension with another version is skipped.
    let mut other = data.clone();
    let at = other.windows(2).position(|w| w == [b'D', DRED_EXPERIMENTAL_VERSION]).unwrap();
    other[at + 1] ^= 1;
    assert_eq!(r.parse(&mut dred, &other, 16000, 16000, false).unwrap().samples, 0);
}

#[test]
fn dred_decoder_loads_weights() {
    let mut r = OpusDredDecoder::new();
    assert!(r.is_loaded());
    // A blob without the RDOVAE decoder is refused and changes nothing.
    let pitch = DnnBlob::from_static(builtin::pitchdnn()).unwrap();
    assert_eq!(r.set_dnn_blob(&pitch), Err("DNN blob holds no model of this build"));
    assert!(r.is_loaded());
    r.set_dnn_blob(&DnnBlob::builtin()).unwrap();
    let own = DnnBlob::from_bytes(builtin::dred_rdovae_dec()).unwrap();
    r.set_dnn_blob(&own).unwrap();
    drop(own);
    // The decoder keeps the blob alive.
    let packets = encode(16000, 1, Application::Voip, 24000, 320, 2);
    let mut rng = Rng(11);
    let mut a = OpusDred::new();
    let mut b = OpusDred::new();
    let builtin = OpusDredDecoder::new();
    for _ in 0..50 {
        let data = with_dred(&packets[1], &random_payload(&mut rng), 0);
        r.parse(&mut a, &data, 48000, 48000, false).unwrap();
        builtin.parse(&mut b, &data, 48000, 48000, false).unwrap();
        assert_eq!(a.features(), b.features());
    }
}

#[test]
fn parse_process_and_decode_allocate_nothing() {
    let packets = encode(48000, 2, Application::Voip, 48000, 960, 12);
    let r = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut other = OpusDred::new();
    let mut rng = Rng(5);
    let data = loop {
        let data = with_dred(&packets[11], &random_payload(&mut rng), 0);
        if r.parse(&mut dred, &data, 48000, 48000, false).unwrap().samples > 0 && dred.nb_latents() > 2 {
            break data;
        }
    };
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    dec.set_complexity(10).unwrap();
    let mut pcm = vec![0f32; 2 * 48000];
    let mut pcm16 = vec![0i16; 2 * 48000];
    let mut pcm24 = vec![0i32; 2 * 48000];
    for p in &packets[..11] {
        dec.decode(p, 960, &mut pcm).unwrap();
    }
    // Losses longer than 120 ms, which the integer variants conceal in
    // pieces.
    let n = allocations(|| {
        r.parse(&mut dred, &data, 48000, 48000, true).unwrap();
        r.process(&dred, &mut other).unwrap();
        r.process_in_place(&mut dred).unwrap();
        dec.dred_decode_float(&dred, 9600, 9600, &mut pcm).unwrap();
        dec.dred_decode(&dred, 9600, 9600, &mut pcm16).unwrap();
        dec.dred_decode24(&dred, 9600, 9600, &mut pcm24).unwrap();
    });
    assert_eq!(n, 0);
}

#[test]
fn dred_types_can_move_between_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<OpusDredDecoder>();
    send_sync::<OpusDred>();
    assert_eq!(size_of::<OpusDred>(), 11_236);
}
