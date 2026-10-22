//! The DRED encoder without libopus: the controls, the loading of the
//! models, DRED data that opus1's own DRED decoder reads, random settings
//! and inputs that do not panic, and encoding that allocates nothing.
//!
//! The comparisons with libopus are in `dnn_dred_encode_libopus.rs`. This
//! binary has its own global allocator, which counts the calls a thread
//! makes while that thread has counting switched on (as
//! `dnn_no_alloc.rs` does).
#![cfg(all(feature = "dred", feature = "std"))]

use std::num::NonZeroU32;
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::dred::{DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID};
use opus1::{
    Application, Bitrate, Channels, DnnBlob, DnnModels, DredProcessing, ErrorKind, ForcedMode,
    InbandFec, OpusDred, OpusDredDecoder, OpusEncoder, SampleRate, packet,
};
use opus1::packet::Framing;
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

/// A test signal: a gliding tone with pauses and some noise, as floats in
/// [-1, 1). The second channel is shifted in time.
fn signal(fs: i32, channels: usize, n: usize) -> Vec<f32> {
    let mut rng = Rng(3);
    let mut out = Vec::with_capacity(n * channels);
    for i in 0..n {
        for c in 0..channels {
            let t = (i + 29 * c) as f32 / fs as f32;
            let f = 180.0 + 60.0 * (t * 2.0).sin();
            let level = if (t * 0.7).fract() < 0.75 { 0.3 } else { 0.0 };
            let noise = (rng.next() % 1000) as f32 / 1e6;
            out.push(level * (2.0 * std::f32::consts::PI * f * t).sin() + noise);
        }
    }
    out
}

/// The DRED extension of a packet: its frame and its data.
fn dred_of(data: &[u8]) -> Option<(usize, Vec<u8>)> {
    let parsed = packet::parse(data, Framing::Normal).unwrap();
    let mut it = opus1::extensions::ExtensionIterator::new(parsed.padding(data), parsed.nb_frames());
    let ext = it.find_id(DRED_EXTENSION_ID).unwrap()?;
    Some((ext.frame, ext.data.to_vec()))
}

fn encoder(fs: i32, channels: usize, bitrate: i32, loss: i32, dred: i32) -> OpusEncoder {
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(fs).unwrap(),
        Channels::try_from(channels).unwrap(),
        Application::Voip,
    )
    .unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
    enc.set_packet_loss_perc(loss as u8).unwrap();
    enc.set_dred_duration(dred as u8).unwrap();
    enc
}

#[test]
fn the_dred_duration_takes_0_to_104_and_a_reset_clears_it() {
    let mut enc =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
    assert_eq!(enc.dred_duration(), 0);
    for v in [0, 1, 52, 104] {
        enc.set_dred_duration(v).unwrap();
        assert_eq!(enc.dred_duration(), v);
    }
    for v in [105, 200, u8::MAX] {
        assert_eq!(enc.set_dred_duration(v).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
        assert_eq!(enc.dred_duration(), 104);
    }
    enc.reset_state();
    assert_eq!(enc.dred_duration(), 0);
}

/// The DRED data starts with 'D' and the experimental version, and
/// opus1's DRED decoder reads it: at every rate, mono and stereo, in every
/// mode, and in packets of several frames.
#[test]
fn dred_data_is_written_and_read_back() {
    let dec = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    for (k, &(fs, channels, frame_ms, mode)) in [
        (48000, 1usize, 20usize, None),
        (48000, 2, 10, Some(ForcedMode::CeltOnly)),
        (48000, 2, 60, Some(ForcedMode::CeltOnly)),
        (24000, 1, 40, Some(ForcedMode::SilkOnly)),
        (16000, 2, 60, Some(ForcedMode::SilkOnly)),
        (16000, 1, 120, Some(ForcedMode::SilkOnly)),
        (12000, 1, 20, None),
        (8000, 2, 20, None),
        (48000, 1, 20, Some(ForcedMode::Hybrid)),
    ]
    .iter()
    .enumerate()
    {
        let frame = fs as usize * frame_ms / 1000;
        let mut enc = encoder(fs, channels, 32000 * channels as i32, 30, 104);
        enc.set_force_mode(mode);
        enc.set_vbr(k % 3 != 1);
        let pcm = signal(fs, channels, 3 * fs as usize);
        let mut with_dred = 0;
        let mut out = vec![0u8; 1500];
        for f in pcm.chunks_exact(frame * channels) {
            let n = enc.encode(f, frame, &mut out).unwrap();
            let p = &out[..n];
            if let Some((ext_frame, data)) = dred_of(p) {
                with_dred += 1;
                assert_eq!(&data[..2], &[b'D', DRED_EXPERIMENTAL_VERSION]);
                assert!(ext_frame < packet::nb_frames(p).unwrap());
                let got =
                    dec.parse(&mut dred, p, fs, SampleRate::try_from(fs).unwrap(), DredProcessing::Now).unwrap();
                assert!(got.samples > 0, "{fs} {channels} {frame_ms} {mode:?}");
                assert!(dred.nb_latents() > 0);
                assert!(dred.features().iter().all(|x| x.is_finite()));
            }
        }
        let packets = pcm.len() / (frame * channels);
        assert!(with_dred * 2 > packets, "{fs} {channels} {frame_ms} {mode:?}: {with_dred} of {packets}");
    }
}

/// With VBR and no expected loss DRED takes no bits and writes nothing;
/// with CBR the packets keep their size.
#[test]
fn no_dred_data_without_loss_with_vbr_and_cbr_sizes_stay() {
    let pcm = signal(48000, 1, 48000);
    let mut enc = encoder(48000, 1, 32000, 0, 104);
    let mut out = vec![0u8; 1500];
    for f in pcm.as_chunks::<960>().0 {
        let n = enc.encode(f, 960, &mut out).unwrap();
        assert!(dred_of(&out[..n]).is_none());
    }
    for loss in [0, 10, 40] {
        let mut enc = encoder(48000, 1, 32000, loss, 104);
        enc.set_vbr(false);
        let mut any = false;
        for f in pcm.as_chunks::<960>().0 {
            let n = enc.encode(f, 960, &mut out).unwrap();
            assert_eq!(n, 80);
            any |= dred_of(&out[..n]).is_some();
        }
        assert!(any, "loss {loss}");
    }
}

/// A duration of 0 gives the packets of an encoder that never had DRED on.
#[test]
fn a_duration_of_0_gives_the_packets_without_dred() {
    let pcm = signal(48000, 2, 2 * 48000);
    let mut a = encoder(48000, 2, 48000, 30, 0);
    let mut b = encoder(48000, 2, 48000, 30, 80);
    b.set_dred_duration(0).unwrap();
    let mut c = encoder(48000, 2, 48000, 30, 80);
    let (mut pa, mut pb, mut pc) = (vec![0u8; 1500], vec![0u8; 1500], vec![0u8; 1500]);
    let mut differ = 0;
    for f in pcm.as_chunks::<{ 960 * 2 }>().0 {
        let na = a.encode(f, 960, &mut pa).unwrap();
        let nb = b.encode(f, 960, &mut pb).unwrap();
        let nc = c.encode(f, 960, &mut pc).unwrap();
        assert_eq!(pa[..na], pb[..nb]);
        assert!(dred_of(&pa[..na]).is_none());
        differ += (pa[..na] != pc[..nc]) as usize;
    }
    assert!(differ > 50, "{differ}");
}

/// `set_dnn_blob`: a blob without both models of the DRED encoder is
/// refused and the models stay; a blob with both, and the built-in
/// weights, load. A reset keeps the loaded models.
#[test]
fn set_dnn_blob_loads_both_models_or_keeps_the_old_ones() {
    let pcm = signal(16000, 1, 16000);
    let run = |enc: &mut OpusEncoder| {
        let mut out = vec![0u8; 1500];
        let mut packets = Vec::new();
        for f in pcm.as_chunks::<320>().0 {
            let n = enc.encode(f, 320, &mut out).unwrap();
            packets.push(out[..n].to_vec());
        }
        packets
    };
    let reference = run(&mut encoder(16000, 1, 32000, 30, 100));
    assert!(reference.iter().filter(|p| dred_of(p).is_some()).count() > 20);

    for only in [builtin::pitchdnn(), builtin::dred_rdovae_enc(), builtin::plc()] {
        let mut enc = encoder(16000, 1, 32000, 30, 100);
        let blob = DnnBlob::from_static(only).unwrap();
        assert_eq!(
            enc.set_dnn_blob(&blob).map_err(|e| (e.kind(), e.message())),
            Err((ErrorKind::BadArg, "DNN blob holds no model of this build"))
        );
        assert_eq!(run(&mut enc), reference);
    }
    let mut both = builtin::dred_rdovae_enc().to_vec();
    both.extend_from_slice(builtin::pitchdnn());
    let mut enc = encoder(16000, 1, 32000, 30, 100);
    assert_eq!(enc.set_dnn_blob(&DnnBlob::from_bytes(&both).unwrap()), Ok(DnnModels::DRED_ENCODER));
    enc.reset_state();
    enc.set_dred_duration(100).unwrap();
    assert_eq!(run(&mut enc), reference);
    let mut enc = encoder(16000, 1, 32000, 30, 100);
    assert_eq!(enc.set_dnn_blob(&DnnBlob::builtin()), Ok(DnnModels::DRED_ENCODER));
    assert_eq!(run(&mut enc), reference);
}

/// Random settings and inputs, NaN and huge values included, do not panic,
/// and every packet parses.
#[test]
fn random_settings_and_inputs_do_not_panic() {
    let mut rng = Rng(11);
    let rates = [8000, 12000, 16000, 24000, 48000];
    let mut out = vec![0u8; 1500];
    for round in 0..120 {
        let fs = rates[rng.next() as usize % 5];
        let channels = 1 + rng.next() as usize % 2;
        let app = [Application::Voip, Application::Audio, Application::RestrictedLowDelay][rng.next() as usize % 3];
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(fs).unwrap(),
            Channels::try_from(channels).unwrap(),
            app,
        )
        .unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(6000 + rng.next() % 200_000).unwrap()));
        enc.set_packet_loss_perc((rng.next() % 101) as u8).unwrap();
        enc.set_dred_duration((rng.next() % 105) as u8).unwrap();
        enc.set_inband_fec(if rng.next().is_multiple_of(2) { InbandFec::On } else { InbandFec::Off });
        enc.set_vbr(!rng.next().is_multiple_of(3));
        enc.set_dtx(rng.next().is_multiple_of(4));
        enc.set_complexity((rng.next() % 11) as u8).unwrap();
        let ms10 = [1usize, 2, 4, 8, 16, 24, 32, 40, 48][rng.next() as usize % 9];
        let frame = fs as usize * ms10 / 400;
        let buf_len = [1500usize, 40, 200, 1275][rng.next() as usize % 4];
        for i in 0..40 {
            let kind = rng.next() % 10;
            let pcm: Vec<f32> = (0..frame * channels)
                .map(|j| match kind {
                    0 => 0.0,
                    1 if j == 7 => f32::NAN,
                    2 => 1e9,
                    3 => ((rng.next() % 2001) as f32 - 1000.0) / 1000.0,
                    _ => ((i * frame + j) as f32 * 0.05).sin() * 0.5,
                })
                .collect();
            match enc.encode(&pcm, frame, &mut out[..buf_len]) {
                Ok(n) => {
                    packet::parse(&out[..n], Framing::Normal).unwrap_or_else(|e| panic!("round {round}: {e}"));
                    let _ = dred_of(&out[..n]);
                }
                Err(e) => assert!(buf_len < 1500 || frame * 400 / fs as usize > 48, "round {round}: {e}"),
            }
        }
    }
}

/// Encoding with DRED allocates nothing, once the encoder exists.
#[test]
fn encoding_with_dred_allocates_nothing() {
    let pcm = signal(48000, 2, 48000);
    let mut enc = encoder(48000, 2, 64000, 30, 104);
    let mut out = vec![0u8; 1500];
    let n = allocations(|| {
        for f in pcm.as_chunks::<{ 960 * 2 }>().0 {
            enc.encode(f, 960, &mut out).unwrap();
        }
        enc.set_vbr(false);
        for f in pcm.as_chunks::<{ 2880 * 2 }>().0 {
            enc.encode(f, 2880, &mut out).unwrap();
        }
    });
    assert_eq!(n, 0);
}
