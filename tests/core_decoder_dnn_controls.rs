//! The decoder controls of the neural network features: the decoder
//! complexity (`OPUS_SET_COMPLEXITY`), the bandwidth extension switch
//! (`OPUS_SET_OSCE_BWE`) and the weight blob (`OPUS_SET_DNN_BLOB`), on
//! `OpusDecoder` and `MultistreamDecoder`.
//!
//! The return values are compared with the libopus that the suite links
//! (Homebrew 1.6.1, built without the neural network features). Its
//! decoder takes every complexity from 0 to 10 and refuses the others with
//! `OPUS_BAD_ARG`, and it answers the BWE requests with
//! `OPUS_UNIMPLEMENTED`. `dnn_decoder_controls_libopus.rs` compares the
//! same requests with a libopus built with the features.
//!
//! In a build without the features `deep-plc` and `osce`, the controls
//! must not change the decoded output. With deep PLC the complexity
//! changes the concealment; `dnn_plc_decoder_libopus.rs` compares that
//! with a libopus that has it.

use audiopus_sys as sys;
use opus1::{ErrorKind, OpusDecoder};
use opus1::multistream::MultistreamDecoder;
use std::os::raw::c_int;

const OPUS_GET_COMPLEXITY_REQUEST: c_int = 4011;
#[cfg(not(feature = "osce-bwe"))]
const OPUS_SET_OSCE_BWE_REQUEST: c_int = 4054;
#[cfg(not(feature = "osce-bwe"))]
const OPUS_GET_OSCE_BWE_REQUEST: c_int = 4055;

/// Values that libopus refuses, and the edges of the range it takes.
const BAD_COMPLEXITIES: [i32; 6] = [-1, 11, 12, 100, i32::MIN, i32::MAX];

struct CDec(*mut sys::OpusDecoder);

impl CDec {
    fn new(rate: i32, channels: i32) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate, channels, &mut err) };
        assert_eq!(err, sys::OPUS_OK);
        CDec(st)
    }

    fn set(&mut self, request: c_int, value: i32) -> c_int {
        unsafe { sys::opus_decoder_ctl(self.0, request, value) }
    }

    fn get(&mut self, request: c_int) -> (c_int, i32) {
        let mut value = -12345;
        let ret = unsafe { sys::opus_decoder_ctl(self.0, request, &mut value as *mut i32) };
        (ret, value)
    }

    fn reset(&mut self) {
        assert_eq!(unsafe { sys::opus_decoder_ctl(self.0, sys::OPUS_RESET_STATE) }, sys::OPUS_OK);
    }

}

impl Drop for CDec {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.0) }
    }
}

/// opus1's result as a libopus return code.
fn code<T>(r: &Result<T, opus1::Error>) -> c_int {
    match r {
        Ok(_) => sys::OPUS_OK,
        Err(e) => e.to_raw(),
    }
}

#[test]
fn complexity_starts_at_0_and_takes_0_to_10_like_libopus() {
    for rate in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            let mut d = OpusDecoder::new(rate, channels).unwrap();
            let mut c = CDec::new(rate, channels as i32);
            assert_eq!(d.complexity(), 0);
            assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, 0));
            for value in 0..=10 {
                assert_eq!(code(&d.set_complexity(value)), c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, value));
                assert_eq!(d.complexity(), value);
                assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, value));
            }
            d.set_complexity(6).unwrap();
            c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, 6);
            for value in BAD_COMPLEXITIES {
                let r = d.set_complexity(value);
                assert_eq!(r.unwrap_err().kind(), ErrorKind::BadArg);
                assert_eq!(code(&r), c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, value), "complexity {value}");
                // A refused value leaves the setting as it was.
                assert_eq!(d.complexity(), 6);
                assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, 6));
            }
        }
    }
}

#[test]
fn a_reset_keeps_the_complexity_like_libopus() {
    let mut d = OpusDecoder::new(48000, 2).unwrap();
    let mut c = CDec::new(48000, 2);
    d.set_complexity(7).unwrap();
    c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, 7);
    d.reset_state();
    c.reset();
    assert_eq!(d.complexity(), 7);
    assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, 7));
}

#[cfg(not(feature = "osce-bwe"))]
#[test]
fn bwe_requests_are_unimplemented_without_the_feature_like_libopus() {
    let mut d = OpusDecoder::new(48000, 1).unwrap();
    let mut c = CDec::new(48000, 1);
    for on in [false, true] {
        let r = d.set_osce_bwe(on);
        assert_eq!(r.unwrap_err().kind(), ErrorKind::Unimplemented);
        assert_eq!(code(&r), c.set(OPUS_SET_OSCE_BWE_REQUEST, on as i32));
    }
    let r = d.osce_bwe();
    assert_eq!(r.unwrap_err().kind(), ErrorKind::Unimplemented);
    assert_eq!(code(&r), c.get(OPUS_GET_OSCE_BWE_REQUEST).0);
}

#[cfg(feature = "osce-bwe")]
#[test]
fn the_bwe_switch_starts_off_and_survives_a_reset() {
    let mut d = OpusDecoder::new(48000, 1).unwrap();
    assert_eq!(d.osce_bwe(), Ok(false));
    d.set_osce_bwe(true).unwrap();
    assert_eq!(d.osce_bwe(), Ok(true));
    d.reset_state();
    assert_eq!(d.osce_bwe(), Ok(true));
    d.set_osce_bwe(false).unwrap();
    assert_eq!(d.osce_bwe(), Ok(false));
}

/// A mono 16 kHz speech-like signal: a vowel-like pulse train with noise.
#[cfg(not(any(feature = "deep-plc", feature = "osce")))]
fn speech(len: usize) -> Vec<f32> {
    let mut seed = 1u32;
    (0..len)
        .map(|i| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let t = i as f32 / 16000.0;
            let f0 = 120.0 + 30.0 * (t * 1.3).sin();
            let voiced = (0..12).map(|h| ((h + 1) as f32 * f0 * t * std::f32::consts::TAU).sin() / (h + 1) as f32).sum::<f32>();
            let envelope = 0.5 + 0.5 * (t * 2.1).sin();
            0.15 * envelope * voiced + 0.02 * noise
        })
        .collect()
}

/// Packets of 20 ms from the libopus encoder, at the given bitrate and
/// with the mode left to the encoder, from 16 kHz input: SILK at
/// 12 kb/s, CELT at 24 kb/s.
#[cfg(not(any(feature = "deep-plc", feature = "osce")))]
fn packets(bitrate: i32) -> Vec<Vec<u8>> {
    let mut err = 0;
    let enc = unsafe { sys::opus_encoder_create(16000, 1, sys::OPUS_APPLICATION_VOIP, &mut err) };
    assert_eq!(err, sys::OPUS_OK);
    unsafe { sys::opus_encoder_ctl(enc, sys::OPUS_SET_BITRATE_REQUEST, bitrate) };
    let pcm = speech(16000 * 3);
    let mut out = Vec::new();
    for frame in pcm.as_chunks::<320>().0 {
        let mut buf = vec![0u8; 1500];
        let n = unsafe { sys::opus_encode_float(enc, frame.as_ptr(), 320, buf.as_mut_ptr(), 1500) };
        assert!(n > 0);
        buf.truncate(n as usize);
        out.push(buf);
    }
    unsafe { sys::opus_encoder_destroy(enc) };
    out
}

/// Every 7th packet is lost.
#[cfg(not(any(feature = "deep-plc", feature = "osce")))]
fn lost(i: usize) -> bool {
    i % 7 == 3
}

#[cfg(not(any(feature = "deep-plc", feature = "osce")))]
fn decode_all(d: &mut OpusDecoder, packets: &[Vec<u8>], rate: i32) -> Vec<f32> {
    let frame = rate as usize / 50;
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let mut pcm = vec![0f32; frame];
        let packet: &[u8] = if lost(i) { &[] } else { p };
        let n = d.decode(packet, frame, &mut pcm).unwrap();
        out.extend_from_slice(&pcm[..n]);
    }
    out
}

/// Without deep PLC and OSCE compiled in, the complexity must not change
/// the output, as in a libopus built without them. (The BWE switch and
/// built-in blobs need those features.)
#[cfg(not(any(feature = "deep-plc", feature = "osce")))]
#[test]
fn the_controls_do_not_change_the_output_yet() {
    for bitrate in [12_000, 24_000] {
        let packets = packets(bitrate);
        for rate in [16000, 48000] {
            let mut plain = OpusDecoder::new(rate, 1).unwrap();
            let want = decode_all(&mut plain, &packets, rate);
            for complexity in [0, 4, 5, 6, 7, 10] {
                let mut d = OpusDecoder::new(rate, 1).unwrap();
                d.set_complexity(complexity).unwrap();
                let got = decode_all(&mut d, &packets, rate);
                assert!(
                    want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()) && want.len() == got.len(),
                    "{bitrate} b/s at {rate} Hz, complexity {complexity}: the output changed"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multistream decoder
// ---------------------------------------------------------------------------

struct CMsDec(*mut sys::OpusMSDecoder);

impl CMsDec {
    fn new(rate: i32, channels: i32, streams: i32, coupled: i32, mapping: &[u8]) -> Self {
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(rate, channels, streams, coupled, mapping.as_ptr(), &mut err)
        };
        assert_eq!(err, sys::OPUS_OK);
        CMsDec(st)
    }

    fn set(&mut self, request: c_int, value: i32) -> c_int {
        unsafe { sys::opus_multistream_decoder_ctl(self.0, request, value) }
    }

    fn get(&mut self, request: c_int) -> (c_int, i32) {
        let mut value = -12345;
        let ret = unsafe { sys::opus_multistream_decoder_ctl(self.0, request, &mut value as *mut i32) };
        (ret, value)
    }
}

impl Drop for CMsDec {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_decoder_destroy(self.0) }
    }
}

/// A 5.1 decoder (4 streams, 2 of them coupled) on both sides.
fn surround() -> (MultistreamDecoder, CMsDec) {
    let mapping = [0u8, 4, 1, 2, 3, 5];
    let table = opus1::multistream::ChannelMappingTable {
        channels: 6,
        stream_count: 4,
        coupled_count: 2,
        mapping: mapping.to_vec(),
    };
    (MultistreamDecoder::new(48000, table).unwrap(), CMsDec::new(48000, 6, 4, 2, &mapping))
}

#[test]
fn the_multistream_decoder_sets_the_complexity_of_every_stream_like_libopus() {
    let (mut d, mut c) = surround();
    assert_eq!(d.complexity(), 0);
    assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, 0));
    for value in 0..=10 {
        assert_eq!(code(&d.set_complexity(value)), c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, value));
        assert_eq!(d.complexity(), value);
        assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, value));
        for s in 0..4 {
            assert_eq!(d.stream_decoder(s).unwrap().complexity(), value);
        }
    }
    d.set_complexity(5).unwrap();
    c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, 5);
    for value in BAD_COMPLEXITIES {
        let r = d.set_complexity(value);
        assert_eq!(code(&r), c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, value), "complexity {value}");
        assert_eq!(d.complexity(), 5);
        assert_eq!(c.get(OPUS_GET_COMPLEXITY_REQUEST), (sys::OPUS_OK, 5));
        for s in 0..4 {
            assert_eq!(d.stream_decoder(s).unwrap().complexity(), 5);
        }
    }
    // libopus answers OPUS_GET_COMPLEXITY with the first stream's value.
    d.stream_decoder_mut(0).unwrap().set_complexity(9).unwrap();
    assert_eq!(d.complexity(), 9);
    d.reset_state();
    assert_eq!(d.complexity(), 9);
}

#[test]
fn the_multistream_decoder_forwards_the_bwe_switch() {
    let (mut d, mut c) = surround();
    #[cfg(not(feature = "osce-bwe"))]
    {
        assert_eq!(d.set_osce_bwe(true).unwrap_err().kind(), ErrorKind::Unimplemented);
        // libopus's multistream decoder does not know the request at all.
        assert_eq!(c.set(OPUS_SET_OSCE_BWE_REQUEST, 1), sys::OPUS_UNIMPLEMENTED);
    }
    #[cfg(feature = "osce-bwe")]
    {
        let _ = &mut c;
        d.set_osce_bwe(true).unwrap();
        for s in 0..4 {
            assert_eq!(d.stream_decoder(s).unwrap().osce_bwe(), Ok(true));
        }
        d.reset_state();
        assert_eq!(d.stream_decoder(3).unwrap().osce_bwe(), Ok(true));
    }
}

// ---------------------------------------------------------------------------
// Weight blobs
// ---------------------------------------------------------------------------

#[cfg(feature = "dnn")]
mod blobs {
    use opus1::multistream::{ChannelMappingTable, MultistreamDecoder};
    use opus1::{DnnBlob, DnnModels, ErrorKind, OpusDecoder};

    const NO_MODEL: &str = "DNN blob holds no model of this build";

    /// The kind and the message of an error. Every blob error is
    /// `BadArg`; the message tells them apart.
    fn kind_message<T>(r: Result<T, opus1::Error>) -> Result<T, (ErrorKind, &'static str)> {
        r.map_err(|e| (e.kind(), e.message()))
    }

    /// The groups this build contains.
    fn all() -> DnnModels {
        #[allow(unused_mut)]
        let mut m = DnnModels::NONE;
        #[cfg(feature = "deep-plc")]
        {
            m = m.union(DnnModels::DEEP_PLC);
        }
        #[cfg(feature = "osce")]
        {
            m = m.union(DnnModels::LACE).union(DnnModels::NOLACE);
        }
        #[cfg(feature = "osce-bwe")]
        {
            m = m.union(DnnModels::BBWENET);
        }
        m
    }

    /// The blob file of a model in `dnn-weights/`.
    #[cfg_attr(not(feature = "deep-plc"), allow(dead_code))]
    fn file(name: &str) -> Vec<u8> {
        std::fs::read(format!("{}/dnn-weights/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u8; N]);

    /// An empty blob, which libopus parses as a list without arrays.
    static EMPTY: Aligned<0> = Aligned([]);

    #[test]
    fn a_new_decoder_has_the_builtin_models_of_its_build() {
        let d = OpusDecoder::new(48000, 2).unwrap();
        assert_eq!(d.dnn_models(), all());
    }

    #[test]
    fn a_blob_without_a_model_of_this_build_changes_nothing() {
        let mut d = OpusDecoder::new(16000, 1).unwrap();
        let empty = DnnBlob::from_static(&EMPTY.0).unwrap();
        assert_eq!(kind_message(d.set_dnn_blob(&empty)), Err((ErrorKind::BadArg, NO_MODEL)));
        assert_eq!(d.dnn_models(), all());
        #[cfg(feature = "heap")]
        {
            let blob = DnnBlob::from_bytes(&[]).unwrap();
            assert_eq!(kind_message(d.set_dnn_blob(&blob)), Err((ErrorKind::BadArg, NO_MODEL)));
        }
    }

    #[cfg(feature = "heap")]
    #[test]
    fn malformed_and_misaligned_blobs_are_refused() {
        assert_eq!(kind_message(DnnBlob::from_bytes(&[1, 2, 3])).err(), Some((ErrorKind::BadArg, "invalid DNN blob")));
        assert_eq!(kind_message(DnnBlob::from_bytes(&[0u8; 64])).err(), Some((ErrorKind::BadArg, "invalid DNN blob")));
        #[cfg(feature = "deep-plc")]
        {
            // A static blob whose arrays start 1 byte after a 4-byte
            // boundary: the bytes sit at offset 1 of `u32` storage.
            let src = file("pitchdnn.bin");
            let words: &'static mut [u32] = Box::leak(vec![0u32; src.len() / 4 + 2].into_boxed_slice());
            // SAFETY: the `u32` storage holds `4 · words.len() > src.len() + 1`
            // bytes, and any bytes are valid `u8` values.
            let bytes: &'static mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 4) };
            bytes[1..1 + src.len()].copy_from_slice(&src);
            let shifted: &'static [u8] = &bytes[1..1 + src.len()];
            assert_eq!(
                kind_message(DnnBlob::from_static(shifted)).err(),
                Some((ErrorKind::BadArg, "DNN blob is not aligned"))
            );
            // A copy is aligned whatever the source.
            assert!(DnnBlob::from_bytes(shifted).is_ok());
        }
    }

    #[cfg(all(feature = "heap", feature = "deep-plc"))]
    #[test]
    fn each_group_loads_on_its_own_and_a_reset_keeps_it() {
        let mut d = OpusDecoder::new(48000, 1).unwrap();
        // libopus's blob writer order: PitchDNN, FARGAN, PLC, (LACE, NoLACE).
        let mut bytes = file("pitchdnn.bin");
        bytes.extend(file("fargan.bin"));
        bytes.extend(file("plc.bin"));
        let blob = DnnBlob::from_bytes(&bytes).unwrap();
        assert_eq!(d.set_dnn_blob(&blob), Ok(DnnModels::DEEP_PLC));
        assert_eq!(d.dnn_models(), all());
        // Two of the three deep PLC models are not a group: nothing loads.
        let partial = DnnBlob::from_bytes(&bytes[..file("pitchdnn.bin").len()]).unwrap();
        assert_eq!(kind_message(d.set_dnn_blob(&partial)), Err((ErrorKind::BadArg, NO_MODEL)));
        d.reset_state();
        assert_eq!(d.dnn_models(), all());
        assert_eq!(d.set_dnn_blob(&DnnBlob::builtin()), Ok(all()));
    }

    /// The blob of libopus's `dump_weights_blob` has no BBWENet. In a
    /// libopus built with the bandwidth extension it leaves LACE and
    /// NoLACE unloaded too; opus1 loads them and keeps its BBWENet.
    #[cfg(all(feature = "heap", feature = "osce"))]
    #[test]
    fn a_blob_without_bbwenet_still_loads_lace_and_nolace() {
        let mut bytes = Vec::new();
        for name in ["pitchdnn.bin", "fargan.bin", "plc.bin", "lace.bin", "nolace.bin"] {
            bytes.extend(file(name));
        }
        let blob = DnnBlob::from_bytes(&bytes).unwrap();
        let mut d = OpusDecoder::new(48000, 1).unwrap();
        let want = DnnModels::DEEP_PLC.union(DnnModels::LACE).union(DnnModels::NOLACE);
        assert_eq!(d.set_dnn_blob(&blob), Ok(want));
        assert_eq!(d.dnn_models(), all());
        // Only LACE.
        let lace = DnnBlob::from_bytes(&file("lace.bin")).unwrap();
        assert_eq!(d.set_dnn_blob(&lace), Ok(DnnModels::LACE));
    }

    /// Blobs and decoders that hold one can move to other threads and be
    /// shared, like decoders without the features. A copied blob lives on
    /// in the decoder that loaded it after its last other clone is gone.
    #[test]
    fn blobs_and_decoders_can_move_to_other_threads() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<DnnBlob>();
        send_sync::<OpusDecoder>();
        send_sync::<MultistreamDecoder>();
        #[cfg(all(feature = "heap", feature = "deep-plc"))]
        {
            let mut bytes = file("pitchdnn.bin");
            bytes.extend(file("fargan.bin"));
            bytes.extend(file("plc.bin"));
            let blob = DnnBlob::from_bytes(&bytes).unwrap();
            drop(bytes);
            let mut d = OpusDecoder::new(16000, 1).unwrap();
            assert_eq!(d.set_dnn_blob(&blob), Ok(DnnModels::DEEP_PLC));
            let other = blob.clone();
            drop(blob);
            let t = std::thread::spawn(move || (d.dnn_models(), format!("{other:?}")));
            let (models, text) = t.join().unwrap();
            assert_eq!(models, all());
            assert!(text.contains("copied"), "{text}");
        }
    }

    #[test]
    fn the_multistream_decoder_loads_a_blob_into_every_stream() {
        let table = ChannelMappingTable { channels: 3, stream_count: 2, coupled_count: 1, mapping: vec![0, 2, 1] };
        let mut d = MultistreamDecoder::new(48000, table).unwrap();
        let empty = DnnBlob::from_static(&EMPTY.0).unwrap();
        assert_eq!(kind_message(d.set_dnn_blob(&empty)), Err((ErrorKind::BadArg, NO_MODEL)));
        #[cfg(feature = "deep-plc")]
        {
            assert_eq!(d.set_dnn_blob(&DnnBlob::builtin()), Ok(all()));
            for s in 0..2 {
                assert_eq!(d.stream_decoder(s).unwrap().dnn_models(), all());
            }
        }
    }
}
