#![cfg(all(feature = "test-libopus-dnn", feature = "dred"))]
//! Deep redundancy (DRED) against libopus 1.6.1 built with DRED.
//!
//! - The tables of DRED's quantizer equal the arrays of libopus's
//!   `dnn/dred_rdovae_stats_data.c`.
//! - The packets of a libopus encoder with DRED carry the DRED extension:
//!   opus1's extension parser finds what libopus's finds, and opus1's
//!   repacketizer, `pad` and `unpad` give libopus's bytes on them.
//!
//! The layers of the RDOVAE encoder and decoder are compared with libopus
//! in `dnn_weights_libopus.rs`. These tests need the build with DRED, whose
//! library holds the DRED code and tables:
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dred-novec LIBOPUS_NO_PKG=1 \
//!   LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dred-ref cargo test --release \
//!   --features test-libopus-dnn,dred --test dnn_dred_libopus
//! ```

use audiopus_sys as sys;
use opus1::extensions::{self, Extension};
use opus1::internals::dnn::dred::{DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID};
use opus1::internals::dnn::models::dred_rdovae_stats as stats;
use opus1::packet::{self, Repacketizer};
use std::ffi::CStr;
use std::os::raw::{c_int, c_uchar};

/// libopus `opus_extension_data` (src/opus_private.h).
#[repr(C)]
#[derive(Clone, Copy)]
struct CExt {
    id: c_int,
    frame: c_int,
    data: *const c_uchar,
    len: i32,
}

/// `OPUS_SET_DRED_DURATION_REQUEST` of include/opus_defines.h.
const OPUS_SET_DRED_DURATION_REQUEST: c_int = 4050;
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;

unsafe extern "C" {
    fn opus_packet_extensions_parse(data: *const c_uchar, len: i32, exts: *mut CExt, nb: *mut i32, nb_frames: c_int) -> i32;
}

unsafe extern "C" {
    static dred_latent_quant_scales_q8: [u8; 400];
    static dred_latent_dead_zone_q8: [u8; 400];
    static dred_latent_r_q8: [u8; 400];
    static dred_latent_p0_q8: [u8; 400];
    static dred_state_quant_scales_q8: [u8; 800];
    static dred_state_dead_zone_q8: [u8; 800];
    static dred_state_r_q8: [u8; 800];
    static dred_state_p0_q8: [u8; 800];
}

/// Makes sure the test binary links libopus 1.6.1.
fn linked() {
    let version = unsafe { CStr::from_ptr(audiopus_sys::opus_get_version_string()) };
    assert!(version.to_string_lossy().contains("1.6.1"), "{version:?}");
}

#[test]
fn quantizer_tables_equal_those_of_libopus() {
    linked();
    // SAFETY: the C arrays are constant and have these sizes in
    // dnn/dred_rdovae_stats_data.h.
    let c: [(&str, &[u8], &[u8]); 8] = unsafe {
        [
            ("dred_latent_quant_scales_q8", &stats::DRED_LATENT_QUANT_SCALES_Q8, &dred_latent_quant_scales_q8),
            ("dred_latent_dead_zone_q8", &stats::DRED_LATENT_DEAD_ZONE_Q8, &dred_latent_dead_zone_q8),
            ("dred_latent_r_q8", &stats::DRED_LATENT_R_Q8, &dred_latent_r_q8),
            ("dred_latent_p0_q8", &stats::DRED_LATENT_P0_Q8, &dred_latent_p0_q8),
            ("dred_state_quant_scales_q8", &stats::DRED_STATE_QUANT_SCALES_Q8, &dred_state_quant_scales_q8),
            ("dred_state_dead_zone_q8", &stats::DRED_STATE_DEAD_ZONE_Q8, &dred_state_dead_zone_q8),
            ("dred_state_r_q8", &stats::DRED_STATE_R_Q8, &dred_state_r_q8),
            ("dred_state_p0_q8", &stats::DRED_STATE_P0_Q8, &dred_state_p0_q8),
        ]
    };
    for (name, ours, theirs) in c {
        assert_eq!(ours, theirs, "{name}");
    }
    assert_eq!(stats::DRED_LATENT_P0_Q8.len(), stats::DRED_NUM_QUANTIZATION_LEVELS * stats::DRED_LATENT_DIM);
    assert_eq!(stats::DRED_STATE_P0_Q8.len(), stats::DRED_NUM_QUANTIZATION_LEVELS * stats::DRED_STATE_DIM);
}

/// Speech-like test signal at 16 kHz: a gliding tone with a varying level
/// and some noise.
fn signal(n: usize) -> Vec<i16> {
    let mut seed = 1u32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let t = i as f64 / 16000.0;
            let f = 150.0 + 50.0 * (t * 3.0).sin();
            let v = 6000.0 * (2.0 * std::f64::consts::PI * f * t).sin() * (0.6 + 0.4 * (t * 7.0).sin());
            (v + ((seed >> 16) % 1000) as f64 - 500.0) as i16
        })
        .collect()
}

/// 100 packets of 20 ms of a libopus encoder with DRED: 16 kHz VoIP at
/// 32 kb/s, 20 % expected loss, `OPUS_SET_DRED_DURATION(100)`.
fn dred_packets() -> Vec<Vec<u8>> {
    let mut err = 0;
    let enc = unsafe { sys::opus_encoder_create(16000, 1, sys::OPUS_APPLICATION_VOIP as c_int, &mut err) };
    assert_eq!(err, 0);
    unsafe {
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 32000 as c_int), 0);
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, 20 as c_int), 0);
        assert_eq!(sys::opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION_REQUEST, 100 as c_int), 0, "a libopus without DRED");
    }
    let pcm = signal(100 * 320);
    let packets = pcm
        .as_chunks::<320>()
        .0
        .iter()
        .map(|frame| {
            let mut buf = vec![0u8; 1500];
            let n = unsafe { sys::opus_encode(enc, frame.as_ptr(), 320, buf.as_mut_ptr(), buf.len() as i32) };
            assert!(n > 0);
            buf.truncate(n as usize);
            buf
        })
        .collect();
    unsafe { sys::opus_encoder_destroy(enc) };
    packets
}

#[test]
fn dred_packets_of_libopus_carry_the_extension() {
    linked();
    let packets = dred_packets();
    let mut with_dred = 0;
    for (k, p) in packets.iter().enumerate() {
        let parsed = packet::parse(p, false).unwrap();
        let padding = parsed.padding(p);
        let mut ours = [Extension::default(); 8];
        let n = extensions::parse(padding, &mut ours, parsed.count).unwrap();
        let mut c = [CExt { id: 0, frame: 0, data: std::ptr::null(), len: 0 }; 8];
        let mut c_n = 8;
        let ret = unsafe { opus_packet_extensions_parse(padding.as_ptr(), padding.len() as i32, c.as_mut_ptr(), &mut c_n, parsed.count as c_int) };
        assert_eq!((ret, c_n as usize), (0, n), "packet {k}");
        for (a, b) in ours[..n].iter().zip(&c) {
            let data = unsafe { std::slice::from_raw_parts(b.data, b.len as usize) };
            assert_eq!((a.id as c_int, a.frame as c_int, a.data), (b.id, b.frame, data), "packet {k}");
        }
        let mut iter = extensions::ExtensionIterator::new(padding, parsed.count);
        if let Some(ext) = iter.find(DRED_EXTENSION_ID).unwrap() {
            assert_eq!(&ext.data[..2], &[b'D', DRED_EXPERIMENTAL_VERSION], "packet {k}");
            assert_eq!(ext.frame, 0, "packet {k}");
            with_dred += 1;
        }
    }
    assert!(with_dred >= 95, "{with_dred} packets with DRED");

    // The repacketizer, pad and unpad on these packets.
    for (k, group) in packets.chunks(3).enumerate() {
        let c_rp = unsafe { sys::opus_repacketizer_create() };
        let mut rp = Repacketizer::new();
        for p in group {
            assert_eq!(unsafe { sys::opus_repacketizer_cat(c_rp, p.as_ptr(), p.len() as i32) }, 0);
            rp.cat(p).unwrap();
        }
        let mut c_out = vec![0u8; 4000];
        let mut out = vec![0u8; 4000];
        let c_n = unsafe { sys::opus_repacketizer_out(c_rp, c_out.as_mut_ptr(), 4000) };
        let n = rp.out(&mut out).unwrap();
        assert_eq!(n as i32, c_n, "group {k}");
        assert_eq!(out[..n], c_out[..n], "group {k}");
        let parsed = packet::parse(&out[..n], false).unwrap();
        let frames: Vec<usize> = {
            let mut v = Vec::new();
            let mut iter = extensions::ExtensionIterator::new(parsed.padding(&out[..n]), parsed.count);
            while let Some(e) = iter.find(DRED_EXTENSION_ID).unwrap() {
                v.push(e.frame);
            }
            v
        };
        assert!(frames.windows(2).all(|w| w[0] < w[1]) && frames.len() + 1 >= group.len(), "group {k}: {frames:?}");
        // One frame of the middle.
        if group.len() == 3 {
            let c_n = unsafe { sys::opus_repacketizer_out_range(c_rp, 1, 2, c_out.as_mut_ptr(), 4000) };
            let n = rp.out_range(1, 2, &mut out).unwrap();
            assert_eq!(out[..n], c_out[..c_n as usize], "group {k}: frame 1");
        }
        unsafe { sys::opus_repacketizer_destroy(c_rp) };

        let p = &group[0];
        let mut c_buf = p.clone();
        c_buf.resize(p.len() + 100, 0);
        let mut buf = c_buf.clone();
        assert_eq!(unsafe { sys::opus_packet_pad(c_buf.as_mut_ptr(), p.len() as i32, p.len() as i32 + 100) }, 0);
        packet::pad(&mut buf, p.len(), p.len() + 100).unwrap();
        assert_eq!(buf, c_buf, "group {k}: pad");
        let mut c_buf = p.clone();
        let mut buf = p.clone();
        let c_n = unsafe { sys::opus_packet_unpad(c_buf.as_mut_ptr(), p.len() as i32) };
        let n = packet::unpad(&mut buf, p.len()).unwrap();
        assert_eq!(buf[..n], c_buf[..c_n as usize], "group {k}: unpad");
    }
}
