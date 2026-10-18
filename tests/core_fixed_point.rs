//! Packets of `OpusEncoder` against the fixed-point build of libopus 1.6.1.
//!
//! With the `Voip` application and 16-bit input, the SILK-only packets of
//! opus1 are byte for byte those of the fixed-point build: its SILK
//! encoder is a port of `silk/fixed`, and at 8 to 24 kHz it filters the
//! input with the integer `hp_cutoff` of that build. At 48 kHz opus1 gives
//! SILK the float filter's output instead, so a test at 48 kHz first sets
//! `internals::silk_input::set_integer_hp_at_48k`. The tests here hold a
//! 32-bit hash (FNV-1a) of every packet that build wrote for the same input
//! and the same settings, packet by packet.
//!
//! The packets were made with the source of the official libopus 1.6.1
//! release, built with `cc` and `-O2 -DFIXED_POINT -DOPUS_BUILD
//! -DVAR_ARRAYS -DHAVE_LRINTF -DHAVE_LRINT` and no other options, driven by
//! `opus_encode` with the controls named in each test. Each control is set
//! before the first packet and again whenever its value changes. The input
//! is the one the test makes, as 16-bit samples.
//!
//! The libopus source that the `opus-head-sys` dependency vendors is
//! libopus's main branch, not 1.6.1. Its SILK encoder adds 1 to the frame
//! energy of the Burg analysis (line 72 of silk/fixed/burg_modified_FIX.c).
//! It writes other packets in every stream of this file; the first
//! difference is at packet 0 to 13. This file first held hashes recorded
//! from that source. Figures in the comments below that describe opus1
//! before a fix were measured against it.

mod k24_common;

use opus1::internals::range_coder::RangeCoder;
use opus1::internals::silk::dec_api::SilkDecoder;
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusEncoder};

/// FNV-1a, 32 bits.
fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c9dc5u32, |h, &b| (h ^ b as u32).wrapping_mul(0x01000193))
}

/// The 16 kHz speech of the fixture as 16-bit samples: the samples of the
/// file.
fn speech_16k(seconds: f64) -> Vec<i16> {
    k24_common::speech(16000, seconds).iter().map(|&v| (v * 32768.0).round() as i16).collect()
}

/// What a packet is coded with: bitrate, complexity and the channel count
/// asked for (`None`: the bitrate decides).
#[derive(Clone, Copy)]
struct Packet {
    bitrate: i32,
    complexity: i32,
    channels: Option<usize>,
}

/// SILK-only wideband VoIP packets of 20 ms at 16 kHz with a variable
/// bitrate, one per entry of `script`. Returns the index of the first
/// packet whose hash is not `want`'s, if any.
fn first_difference(channels: usize, pcm: &[i16], script: &[Packet], want: &[u32]) -> Option<usize> {
    let frame = 320;
    let mut enc = OpusEncoder::new(16000, channels, Application::Voip).unwrap();
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_bandwidth(Bandwidth::Wideband);
    let mut buf = vec![0u8; 4000];
    for (i, (p, input)) in script.iter().zip(pcm.chunks_exact(frame * channels)).enumerate() {
        enc.set_bitrate(Bitrate::BitsPerSecond(p.bitrate)).unwrap();
        enc.set_complexity(p.complexity).unwrap();
        enc.set_force_channels(p.channels).unwrap();
        let n = enc.encode_i16(input, frame, &mut buf).unwrap();
        if fnv1a(&buf[..n]) != want[i] {
            return Some(i);
        }
    }
    None
}

/// libopus `silk_Encode` calls `silk_control_encoder` for every packet
/// (enc_API.c:253-260), so a new complexity takes effect with the next
/// packet. Here it changes every 20 packets: 10, 2, 0, 6, 1, 9, 4.
/// opus1 called the control function only when SILK's rate or packet
/// duration changed: from packet 20 on its packets differed.
#[test]
fn a_new_complexity_takes_effect_with_the_next_packet_like_libopus() {
    let complexities = [10, 2, 0, 6, 1, 9, 4];
    let script: Vec<Packet> = (0..140)
        .map(|i| Packet { bitrate: 24000, complexity: complexities[i / 20], channels: None })
        .collect();
    let first = first_difference(1, &speech_16k(3.0), &script, &COMPLEXITY_WALK);
    assert_eq!(first, None, "first packet that is not libopus's");
}

/// `a_new_complexity_takes_effect_with_the_next_packet_like_libopus`.
const COMPLEXITY_WALK: [u32; 140] = [
    0x748c5126, 0xcc9c1cab, 0x70822472, 0xaa92e8c4, 0x98b37de4, 0xfa252609, 0x757d3fea, 0x684c772c,
    0xac63206a, 0x8a7d456e, 0x392a11cd, 0x8ae1dcb4, 0x6b985835, 0xe200b441, 0xcbb43206, 0x9a20188b,
    0x81683a42, 0xba5487f3, 0xa8d27cb3, 0x7f9e618c, 0x494d39c9, 0x2042abcc, 0xb3eeba3a, 0xb0f75585,
    0x28f0125f, 0x6f3b5d08, 0x6bfc5a6a, 0x257cc19b, 0x16feecc7, 0xe8e53ec9, 0x6de31cc3, 0x89a44a6c,
    0xfee4734e, 0x146cbbe3, 0xee72ab93, 0x3753baeb, 0x6c41418d, 0x95bd0ee0, 0x85f5b788, 0x30e36dc2,
    0xb0de6d4d, 0xb9bc2e47, 0xa1b96d2b, 0xce6d1914, 0xc16e3b21, 0x3fb9aba5, 0x02f71600, 0xc72d0774,
    0x84ee6082, 0x423ad517, 0x025c669c, 0x5ea6d846, 0xb1e2c4a0, 0xfbcc3e8c, 0x63ba0818, 0x8b8a346e,
    0x7291040c, 0x6482ea8b, 0x92981cb6, 0xffb6b0a5, 0x9854568f, 0x5578e50b, 0x54d171e9, 0x75e44296,
    0xc26c7eee, 0x1a764b58, 0xef9b0896, 0x2e5840bf, 0xa60b5e85, 0x114daaab, 0x0012365a, 0xeaca8186,
    0x55db1df8, 0xada9a4d9, 0x2d1a7535, 0x42be174e, 0x19e92382, 0x80d8d75c, 0x1763a646, 0x30d3838f,
    0xcac6f9e6, 0xca890b95, 0x926808d7, 0xaff2341f, 0xde798e87, 0xf9327db9, 0xf1aab7aa, 0xbe5c1449,
    0xa909025a, 0x4e628bb9, 0x6741de9b, 0xd13ffcd5, 0xf45f30d3, 0xbf0f6800, 0x288c2dba, 0xa219353e,
    0x03991874, 0x266739d3, 0xe613288e, 0x4f534e55, 0xdfd7e626, 0x9e7663b1, 0xc4ca51e1, 0x7dedb462,
    0xcc0b32c2, 0x96be8a5e, 0x04fea5a0, 0x74d9ae0a, 0x52274e01, 0x3f8ae778, 0x8e445593, 0x3bd30f25,
    0x9ddb5a1a, 0x2429ce9a, 0xf8740329, 0xec45f0b4, 0x2e126a07, 0xb323bbcf, 0x23428e95, 0xe457b287,
    0x5ed8619d, 0x0e0d6133, 0x5a99afb4, 0x0037bb52, 0x5bf22488, 0x39752d1a, 0x29bb055b, 0x014b7fcc,
    0xe0bfbebd, 0xb9a1ae38, 0x0a9f1626, 0x5d683ba7, 0xa653c8aa, 0x93b7231f, 0xcf66c437, 0xd17c16d5,
    0x5738a14d, 0xd02c4170, 0x24847081, 0xf1aa42ad,
];

/// Speech with four stretches of digital silence, 150 to 500 ms long, as
/// 16-bit samples.
fn speech_with_silences(channels: usize) -> Vec<i16> {
    let mut mono = k24_common::speech(16000, 3.0);
    for (from, to) in [(0.30, 0.55), (0.90, 1.40), (1.75, 1.95), (2.30, 2.75)] {
        mono[(16000.0 * from) as usize..(16000.0 * to) as usize].fill(0.0);
    }
    let x = if channels == 2 { k24_common::to_stereo(&mono, 16000) } else { mono };
    x.iter().map(|&v| (v * 32768.0).round() as i16).collect()
}

/// In a frame of digital silence libopus tells SILK that there is no voice
/// activity (`is_digital_silence`, opus_encoder.c:1246 and 1909-1914).
/// SILK then lowers its own speech activity below the threshold of its
/// voice activity flag (`silk_encode_do_VAD_FIX`). Right after speech
/// stops SILK's detector still finds activity, so without this the flag
/// and the frame's coding differ: the mono stream of opus1 differed from
/// packet 15 on. Every packet is the fixed-point build's.
///
/// The hashes were recorded with the same build as the others in this
/// file.
#[test]
fn digital_silence_tells_silk_there_is_no_voice_activity_like_libopus() {
    let script: Vec<Packet> = (0..150).map(|_| Packet { bitrate: 24000, complexity: 9, channels: None }).collect();
    let first = first_difference(1, &speech_with_silences(1), &script, &MONO_WITH_SILENCES);
    assert_eq!(first, None, "mono: first packet that is not libopus's");
    let first = first_difference(2, &speech_with_silences(2), &script, &STEREO_WITH_SILENCES);
    assert_eq!(first, None, "stereo: first packet that is not libopus's");
}

/// `digital_silence_tells_silk_there_is_no_voice_activity_like_libopus`, mono.
const MONO_WITH_SILENCES: [u32; 150] = [
    0x748c5126, 0xcc9c1cab, 0x70822472, 0xaa92e8c4, 0x98b37de4, 0xfa252609, 0x757d3fea, 0x684c772c,
    0xac63206a, 0x8a7d456e, 0x392a11cd, 0x8ae1dcb4, 0x6b985835, 0xe200b441, 0xcbb43206, 0x7e148261,
    0x79b41d6d, 0x85a27670, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xebe5c197, 0x2ced0102, 0x9f938e37, 0x24f0513a, 0xbb104d08,
    0x94b52715, 0x844b7c2f, 0xa61f75fd, 0xc05b9ea8, 0x7635958e, 0xaf7ba209, 0x1edbbad3, 0xd2cbc5aa,
    0x0a9ab1fd, 0x2b504d8a, 0xe82009d6, 0x59407e38, 0xdfd39912, 0x4a360e44, 0x670b2f98, 0x10cf5b0d,
    0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xb6c0b323, 0xef68232d,
    0xa9aa1520, 0x91ec136d, 0x723670b4, 0xf376e1e9, 0x00e5897c, 0x16ce186c, 0x271ad80b, 0xb24eba54,
    0x3a985d92, 0x4074d4ff, 0xd1626f96, 0x925fda88, 0x54625bed, 0x2bd5979d, 0x8d98da1e, 0xd0f98197,
    0x50c37bd0, 0x20afd498, 0xc53d8815, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0x8cdaf838, 0xb1b4074f, 0x3c2ae977, 0x598d4952, 0xabc9a8df, 0xa85f1eda, 0xa03ec76c,
    0x12d3fb49, 0xb9900ad5, 0x8f7f80f3, 0xb6d3e591, 0x5bc5b7c6, 0xb5ce28c5, 0x36a3efd4, 0x3f65bee7,
    0x6f5b67bf, 0x471edfad, 0x991d05af, 0x7f011ad6, 0xe9e72cab, 0xb45d1e42, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef, 0xfb39b4ef,
    0xfb39b4ef, 0xe8af3638, 0xd5c5540b, 0xf5be734a, 0x1e6042f1, 0x85809ff7, 0x13c69b2e, 0x5843cb30,
    0x9a21c3bf, 0x30c0341c, 0x103ae190, 0xda5ca49e, 0xdf579008, 0x19902e0c,
];

/// `digital_silence_tells_silk_there_is_no_voice_activity_like_libopus`, stereo.
const STEREO_WITH_SILENCES: [u32; 150] = [
    0xf1bdfbc3, 0xb86602e5, 0xa9887701, 0x221c844e, 0xf971eeb8, 0xb68e925a, 0x1e427cf4, 0x27165a7c,
    0xc50332d9, 0xf567da8e, 0xab9dd0c6, 0xeddf6163, 0xd3bddad0, 0x96be1098, 0x649e127d, 0x4662473a,
    0x93caa991, 0x2e60b139, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0x4c4d3e31, 0x62c75595, 0xd17943dc, 0xd03590ae, 0x4d3f04b4,
    0x9f5f07c4, 0xefaa19e8, 0x6a5fae42, 0x04161b57, 0x22816286, 0xa6e4caea, 0x28421d82, 0xe335e4bd,
    0xd3ac291d, 0xfefac726, 0x91f4ece0, 0x579cfb07, 0xd198a6c6, 0x827bb4c1, 0x9452e573, 0xadc07205,
    0xc612ca75, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0x496e73a6, 0x955c95cc,
    0xde1923aa, 0x1aff8a6a, 0xc191ffd4, 0x02d96c55, 0x891e9b77, 0x56bcc24a, 0xf2e12312, 0x90efd1ff,
    0x02ab96df, 0x68b50581, 0x89052be6, 0x6a27e87a, 0xff639e97, 0xd9f1d215, 0xf937ab24, 0x849aa370,
    0x6baf9b4c, 0x77532491, 0x973c882a, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0x8f2505b7, 0xa0d0e1ed, 0x53fe187d, 0x6af47e7d, 0x70e8126e, 0xb93be0eb, 0xd47315c3,
    0xe0e01c3e, 0xd819f0ee, 0xbc1cdb76, 0x6382de41, 0xd8924235, 0xaeccef91, 0x4d245094, 0x04f9d4e0,
    0xe637bd32, 0x7be7bab9, 0xe56ad23a, 0x07a4fc9f, 0x52b5281a, 0xebff0ab3, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb, 0xe3a9dddb,
    0xe3a9dddb, 0x86d1a9c4, 0x064e4e84, 0x728ca2ea, 0xaaaf644a, 0x36e011ef, 0xb11564dd, 0x79c13677,
    0xce283011, 0xdd4da78d, 0x816e67cd, 0x40edb887, 0x0f15848a, 0x1c41c753,
];

/// Stereo input, 40 kb/s and 12 kb/s in turns every 20 packets: libopus
/// codes the 12 kb/s stretches as mono (opus_encoder.c:1428-1451). The
/// packet before each mono stretch still codes two channels, with SILK's
/// side faded out (`toMono`, C 1560-1571). The mono packets code the
/// downmix through the mid resampler, the first of them averaged with the
/// side resampler, and the first stereo packet after them starts the side
/// resampler from the mid's (enc_API.c:296-332). Every packet is the
/// fixed-point build's.
#[test]
fn stereo_input_with_mono_stretches_codes_the_packets_of_libopus() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 3.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let script: Vec<Packet> = (0..140)
        .map(|i| Packet { bitrate: if (i / 20) % 2 == 0 { 40000 } else { 12000 }, complexity: 9, channels: None })
        .collect();
    let first = first_difference(2, &pcm, &script, &STEREO_WITH_MONO_STRETCHES);
    assert_eq!(first, None, "first packet that is not libopus's");
}

/// `stereo_input_with_mono_stretches_codes_the_packets_of_libopus`.
const STEREO_WITH_MONO_STRETCHES: [u32; 140] = [
    0xad70a6fa, 0x52d71d40, 0xf28d5153, 0xc7b096e8, 0xd0ab14b8, 0x3b09aaa5, 0x5639bb44, 0xf6fc1fa5,
    0xb900be23, 0xe4838251, 0xe5e3376d, 0x26db4711, 0xa4082d3c, 0x577f224d, 0xbe27f93c, 0x99527d8c,
    0x1debd612, 0x9857a865, 0x81f1ebed, 0x5f52dc6d, 0x96fb9667, 0xb6e749ad, 0x42f8c48c, 0xd6c0b759,
    0xc9117e39, 0x638757fb, 0xacced9a2, 0x819fc628, 0x1b74e35c, 0xbc2f73be, 0xc1671c57, 0x4cfedbea,
    0x7f44f0cc, 0x98fda3b3, 0x6b2d4ae6, 0x6db2efd8, 0xea5c40cf, 0x34cb4802, 0x56b6c477, 0xab28b0a1,
    0xbd024fea, 0xf6e4c57c, 0xc2d70d4c, 0xa4f8cb4f, 0xa1528cdd, 0x1cb58d7c, 0x3dd952ce, 0xb2d697ac,
    0xe654dc6e, 0x7331683a, 0xa9930a79, 0x2eac8d54, 0xc2d5797b, 0xf24d75c0, 0x53f28fbe, 0xc716eef3,
    0xacade1dd, 0xb972dc6b, 0xe4f1670f, 0x7affc3da, 0xf1fe7049, 0x17d23cb8, 0x015d6021, 0x305c6345,
    0xfe7400a8, 0x1972a1aa, 0x400972b5, 0x231a97be, 0xa563d2cd, 0x8d2142e4, 0x19ef69a9, 0x40b5300b,
    0x56b8ec6a, 0xc34ccc05, 0xb2ec69b2, 0x2009110d, 0x8b2f37f1, 0x44446a80, 0x6f50cd16, 0x150e37af,
    0xa4aa33e3, 0xc4a46590, 0x74e12a07, 0xa684ff46, 0xd329814a, 0x2425ff12, 0xdc665572, 0xba9f0afe,
    0xd612dd2f, 0x7ce2c9f8, 0x4a1ab649, 0xf8cc3586, 0x3d5c14fc, 0x75f21bae, 0x543d25ed, 0x6e514722,
    0x65a044ad, 0x6ccd2326, 0x1b460858, 0x5fab5962, 0xbbb04627, 0x50ad3b4b, 0xfa92f506, 0x3efaebb8,
    0xa3fb0fb0, 0x3e8f928f, 0x9707c102, 0x7cee86c5, 0xcf9e1925, 0x5eed6744, 0x7bc616ae, 0x2db578b9,
    0xaea242a8, 0x06c42538, 0xf5eb2f98, 0xb249a1af, 0xd19d270f, 0xe38a158d, 0x19926ff4, 0xf3f04494,
    0x08ebb6b2, 0xbe6919d5, 0x40cdee1c, 0xe86762c4, 0xf413438a, 0x703bab3a, 0x3d319aba, 0x59e60adf,
    0x497eccbf, 0x06a7b314, 0x808c2583, 0x1dc2c9e2, 0xce8ce88e, 0x224cefaf, 0x9e846893, 0x5ea02715,
    0xaf572a5d, 0x6d0543d8, 0x8b18db7f, 0xadbf31f5,
];

/// Stereo speech at 32 kb/s with a variable bitrate, 8 s. opus1 computed
/// the weight of the last NLSF coefficient from its distance to 32767
/// (`silk_NLSF_VQ_weights_laroia` uses `1 << 15`). The weight was 1 too
/// large for some vectors, and from packet 274 on the NLSF indices of the
/// mid channel were not libopus's.
#[test]
fn stereo_speech_at_32_kbps_codes_the_packets_of_libopus() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 8.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let script = [Packet { bitrate: 32000, complexity: 9, channels: None }; 400];
    let first = first_difference(2, &pcm, &script, &STEREO_SPEECH_32K);
    assert_eq!(first, None, "first packet that is not libopus's");
}

/// `stereo_speech_at_32_kbps_codes_the_packets_of_libopus`.
const STEREO_SPEECH_32K: [u32; 400] = [
    0x7e3b4254, 0xfb616cc3, 0x2dfc07b1, 0xecede0eb, 0xa7a32d4a, 0x7c755246, 0x3f861397, 0x265d65d7,
    0xad6207dc, 0x971a3053, 0x730fdfa3, 0xc2792040, 0x6ce5e5ce, 0x0be5f208, 0x795e8090, 0x36e4ba33,
    0x1f528af6, 0x1ce0758e, 0x93d44072, 0x350c5036, 0x3c1f43e5, 0x7e0f74b0, 0xecd52868, 0x943cf334,
    0x0270e3f9, 0x3d337ea0, 0x301d5544, 0xd80d9f36, 0x01a5b8eb, 0xf2be1ff6, 0x79cf3b20, 0x1b711adc,
    0xfa34bbbe, 0xa2cd4e42, 0x6a37d81d, 0x345cb608, 0xd9232d48, 0xa655ebad, 0xa59e72a3, 0x1d958199,
    0x496b9af3, 0x40caef71, 0x89b9923e, 0x0d204784, 0x6c10a3c8, 0x95434aba, 0x1e1762e9, 0xf4f6b0ca,
    0xccbd74eb, 0x70d4c64b, 0x3df45953, 0x0bb501bb, 0x42434108, 0x5a3cb006, 0x5ede0c77, 0x3bd60b99,
    0xf8717df1, 0xd41b2ef5, 0xe336c9d7, 0x595a5367, 0x0be6209e, 0xc2c3720e, 0xf23cd6e6, 0x14def784,
    0x86fded5e, 0xbcf81f3d, 0x8e0270c0, 0xad218825, 0x4e7cf4f8, 0x80578c8e, 0x9326ed53, 0x0473d638,
    0xa13c5e25, 0x107c9718, 0x41df61d7, 0xbd98c154, 0x2ab14683, 0xddb924a0, 0x7b665755, 0x07ef4252,
    0x6ccb1353, 0xb6dbe7cf, 0x743d50c7, 0x6f10d29a, 0x6d0b2e9b, 0x413d6d9c, 0xda26ab98, 0x8de396c2,
    0xe696b3f4, 0x4ebf8300, 0x42b42e76, 0x27c05b60, 0x292d3545, 0xe5060845, 0xc53dd070, 0xebcca233,
    0x7ae08e90, 0x46cc2cea, 0x29cef329, 0xc66f2502, 0x80208c08, 0x4be16f5c, 0x5e4bc781, 0xabb40c6a,
    0xec951325, 0xed1ff184, 0x129a149a, 0x3d13ad98, 0x08b93cfc, 0xb352d396, 0xbe2fcd22, 0x5add3f53,
    0xc36a573a, 0x5f058b2f, 0xe68422a2, 0x2a964317, 0x68dd8ec5, 0xb0d18e77, 0x8b54ed53, 0x90a5a71e,
    0x76c2311a, 0x51178d15, 0x15c18606, 0x7dd4e7d0, 0x80e26a6a, 0xf6bb43c5, 0x50c372a2, 0x7fcaa188,
    0xba95e134, 0x6cacfc04, 0x583fdbbf, 0x43f71642, 0x7917b170, 0x115cc353, 0xf66b08ee, 0xecf44e3c,
    0xd40b8445, 0x145e2a65, 0xbf71577f, 0x625604e7, 0x5a013417, 0xe52fe938, 0x32a35c07, 0x8e3e18b2,
    0xb4324983, 0x7bd07521, 0xd770d5dc, 0x50448a54, 0x69430f20, 0xf26b9741, 0x7d730600, 0x815296e4,
    0xaf1f2cb2, 0x61cb285e, 0xa6843733, 0xd39a38d1, 0x352cfdab, 0xd73b3934, 0x47e5f16c, 0xccc51949,
    0x7890ba3f, 0x2c17e102, 0xe6752664, 0xa101a640, 0x386522c1, 0x3a415e98, 0x90974f7d, 0x477ef538,
    0x0a0505f5, 0x0bb0f9db, 0x998ad0da, 0xe2a6ac1d, 0xbdfabe00, 0x7054b22f, 0x9d619842, 0x2a92a597,
    0x08245873, 0xf59e87bc, 0xe3b9322d, 0xfd6b229f, 0xa380c6f0, 0x53cd9965, 0xe71e03be, 0xf7f2d268,
    0x43e57b41, 0x2a1476d4, 0x87a47cbe, 0xb547b4a1, 0x134b596a, 0xd05d0fa8, 0x021010b5, 0xbc790a70,
    0xd414dba5, 0x9b01c258, 0x5fdcf066, 0x8665890a, 0x5a0c1804, 0x8c0895ba, 0x2b6a9a41, 0xa6ec9fd9,
    0xf6ffdd35, 0x6bdefb24, 0xb9a4b681, 0x2a52ff29, 0x6e866c6e, 0xdbae9287, 0x70d1ddcf, 0xa42b89d2,
    0x523ed5b7, 0x4013cf5d, 0x924f13b0, 0x67a5701c, 0x2a3ad875, 0x84642993, 0xc9fdbd0a, 0x2db17bdd,
    0xee90ccf1, 0xabfeecd6, 0x902209f5, 0x7dd5b8b1, 0xced07ba9, 0x1d22bc99, 0xc62af3bc, 0x77985d0c,
    0x7755902c, 0xe3e6fc92, 0x5f8f953d, 0xb3b73da6, 0x2a884951, 0x3bd07771, 0xb28b85dd, 0x11c470e4,
    0xb1b49bc1, 0xe9d32d3a, 0x40650341, 0x128132a1, 0x090a1446, 0xcd2fc7d6, 0x97de2eb6, 0xe21005e8,
    0x534a4593, 0x675bf0d0, 0x5719a44e, 0xb031cb8b, 0x6e709f5f, 0xe0a65afc, 0xd8313cb4, 0xe70dd8b3,
    0x3928e5b5, 0x0f4c3d36, 0x16e3ad1a, 0x4aa15e7c, 0x856a157e, 0xadcb53a0, 0xd70130ba, 0x312a82df,
    0xb623c750, 0xf88a04fd, 0x8731b18f, 0x05220e64, 0x1bacc022, 0x404b820b, 0xfcfa9c86, 0x61b4e126,
    0x6470eb08, 0xd132b9f0, 0xea064cf5, 0x85f28dd6, 0x68512a99, 0x9e9f69bc, 0x55b75304, 0x99adcbc8,
    0x05e31173, 0x64729f5a, 0x61553e2c, 0x40edb1fc, 0x651a6898, 0xd41eb149, 0x77dbae87, 0xbda9b7c7,
    0xf2740694, 0xb88e3898, 0x5f6965ac, 0x6e7d45b3, 0x1311f3c3, 0x9bfe758f, 0x1dc9b690, 0xa310fff1,
    0x00a45532, 0xa18b338e, 0x9aa5d72c, 0xb17828f7, 0xdc0ed139, 0xbb5dd624, 0xa751b690, 0xf8553de0,
    0xa54ddea8, 0x9f83e1a8, 0xe96e94c9, 0x9247aab1, 0xf9fe504e, 0x4b20eb66, 0x7375af55, 0x379428ad,
    0xbab2842d, 0x44ff192c, 0xaa060d0d, 0x0f76cc85, 0x6ef9d7c1, 0x73e0ae6f, 0x59a18f1f, 0x0f1c554a,
    0x5b21be5c, 0xe8d4f51a, 0xb343dad4, 0xdfee3791, 0x22a99ae5, 0x4dbedd75, 0x806d1c99, 0x22c75fa6,
    0xd5cc8563, 0x7b37077b, 0x56f97864, 0xa6bc0960, 0xbefb5449, 0x6a8f2209, 0x2bddcd8e, 0x1b186129,
    0x3a6e395b, 0xf60d41c4, 0xac447545, 0xd339d90f, 0x234c0c4b, 0x94f55efb, 0x55efb8e0, 0x011ee841,
    0xa9321cff, 0x6fc654af, 0x6441834f, 0x8e29fa1a, 0x7444f5a0, 0xf2f3e766, 0x76d844f4, 0x8bc6d1e2,
    0xed5bc0b2, 0x6e6d63ff, 0x6b615d0b, 0x8d9fda49, 0xabf991c6, 0x9b7edd37, 0x3e17db77, 0x7f00ab22,
    0x40b37d35, 0x3a7bcccb, 0xa44cc3ee, 0x17bab31e, 0x96a7f667, 0x6d6f1441, 0xe7f48527, 0x2c8f06e9,
    0x01d46de0, 0xa23573c6, 0x4b2e5f6e, 0xce8577cb, 0x133fc743, 0x99aa4e69, 0x1531ad46, 0xffdd7848,
    0x013745dc, 0xab3968ab, 0x9aaa8ccc, 0x26a2d945, 0xace0202d, 0xbc9f88e7, 0x9dd4a3a2, 0x4a0e98b9,
    0xc9737c02, 0x41c4a171, 0x66e981b0, 0x87ac1da5, 0x53ff5c08, 0xf241105d, 0xf2aae8f1, 0x33e70b90,
    0xe2c1189e, 0x88adede7, 0x593682e7, 0x4d1c3aa4, 0xab1b0852, 0x8d0f3035, 0xc18f6fa8, 0xbbf01ef7,
    0x6dfd8f74, 0x798d16a4, 0x76d42d01, 0x299b8ae8, 0xbc1d48d4, 0x1d9c972c, 0xa84c97b6, 0x2a93e176,
];

/// The SILK layer of each packet, decoded at 16 kHz by the SILK decoder of
/// this crate: a hash of the decoded samples and of the number of bits the
/// layer takes. A CELT layer or a redundant CELT frame is not compared: the
/// fixed-point build's CELT encoder is not the float one of opus1.
///
/// `channels` is the number of channels the packets code. A packet must
/// hold one 20 ms SILK frame at 16 kHz.
fn silk_layers(packets: &[Vec<u8>], channels: usize) -> Vec<u32> {
    let mut dec = SilkDecoder::new();
    dec.init(16000, channels as i32);
    dec.channel_state[0].fs_api_hz = 16000;
    let mut pcm = [0i16; 2 * 320];
    packets
        .iter()
        .map(|p| {
            let mut rc = RangeCoder::new_decoder(&p[1..]);
            let n = dec.decode(&mut rc, &mut pcm, 0, true, 20, 16000);
            let n = (n.max(0) as usize * channels).min(pcm.len());
            let samples: Vec<u8> = pcm[..n].iter().flat_map(|s| s.to_le_bytes()).collect();
            (fnv1a(&samples) ^ rc.tell() as u32).wrapping_mul(0x01000193)
        })
        .collect()
}

/// Hybrid with a constant bitrate: libopus codes the SILK layer at a
/// variable bitrate with a cap, and clears SILK's `useCBR`
/// (opus_encoder.c:2163-2177). SILK's noise shaping reads that flag: with
/// a variable bitrate it lowers the quality of frames with little speech
/// activity (noise_shape_analysis_FIX.c:171). opus1 passed the cap but
/// left the flag set: from packet 32 on, in 108 of 140 packets, the SILK
/// layer was not libopus's.
///
/// Input: the 16 kHz speech with every sample repeated three times, which
/// is 48 kHz input with images above 8 kHz for the CELT layer, made
/// without arithmetic that could round differently elsewhere. Fullband
/// Hybrid at 32 kb/s, VoIP, complexity 9. SILK reads the integer
/// `hp_cutoff`, as in the fixed-point build; without that control the
/// first SILK layer differs (packet 0).
#[test]
fn hybrid_at_a_constant_bitrate_codes_the_silk_layer_of_libopus() {
    let pcm: Vec<i16> = speech_16k(3.0).iter().flat_map(|&v| [v, v, v]).collect();
    let mut enc = OpusEncoder::new(48000, 1, Application::Voip).unwrap();
    opus1::internals::silk_input::set_integer_hp_at_48k(&mut enc, true);
    enc.set_bitrate(Bitrate::BitsPerSecond(32000)).unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_vbr(false);
    enc.set_force_mode(Some(ForcedMode::Hybrid));
    enc.set_bandwidth(Bandwidth::Fullband);
    let mut buf = vec![0u8; 4000];
    let packets: Vec<Vec<u8>> = pcm
        .as_chunks::<960>()
        .0
        .iter()
        .take(140)
        .map(|input| {
            let n = enc.encode_i16(input, 960, &mut buf).unwrap();
            assert_eq!(buf[0] >> 3, 15, "Hybrid fullband 20 ms");
            buf[..n].to_vec()
        })
        .collect();
    let layers = silk_layers(&packets, 1);
    let first = (0..layers.len()).find(|&i| layers[i] != HYBRID_CBR_SILK_LAYERS[i]);
    assert_eq!(first, None, "first packet whose SILK layer is not libopus's");
}

/// `hybrid_at_a_constant_bitrate_codes_the_silk_layer_of_libopus`.
const HYBRID_CBR_SILK_LAYERS: [u32; 140] = [
    0x95791c72, 0x77a59ab8, 0x23349952, 0xca76eed7, 0xd6133db1, 0x01cbca16, 0x082976fe, 0xee4d12a5,
    0xd01bb46e, 0x25469c8a, 0xa6de1c46, 0x1fc065cf, 0x42b662da, 0x347f78ac, 0x6f87f244, 0x7eac16cf,
    0xe40eacab, 0xba4906af, 0x710e5c5f, 0x3fe7a50f, 0x9a31e7c2, 0x7eb0170e, 0x622c6c9e, 0xa1530511,
    0xe11682dc, 0xfc8ab958, 0xb202bd79, 0xb02eafe9, 0x13170c9f, 0x7c5494e3, 0x735c66f9, 0x24e164ea,
    0x98746050, 0x6374a32c, 0x09c69cb9, 0xa7e0a388, 0x54043dfc, 0x4ce7073d, 0xb0d247af, 0x755f84cf,
    0xe17b2e52, 0xf43f3db9, 0x75f8f6c3, 0xb490ff0a, 0x4fff24d7, 0x8a9a88cd, 0x1df80ca2, 0xb50fa1ac,
    0xb2dddb52, 0xf43879e7, 0xff972d12, 0x7335f2e3, 0x5f3b2069, 0x73bbfd74, 0xe0d20515, 0x86deae2c,
    0x2689d450, 0x026d37e5, 0x1eb1244d, 0x6a361a64, 0xab9a6200, 0x2e88f9a9, 0x26453f31, 0xe4f3ad27,
    0x1c8616c5, 0xdf607f91, 0xf107e13a, 0x5b1da523, 0xf5e9a289, 0xe892d588, 0x03a0e4d1, 0x325a8757,
    0xb19f2891, 0x4934581f, 0x6320ef9f, 0x0a458ec4, 0x4e866dbd, 0x0eab742b, 0x18615f23, 0xbf83f987,
    0x5642b699, 0x13b6695e, 0x32b22909, 0x383695b3, 0xf2b5b912, 0xfad170fa, 0x2a6bf96e, 0xfe6facf3,
    0xc5d62fac, 0x58ad0d0a, 0x19e7c014, 0x30be7bf9, 0xd5e57902, 0xbf695247, 0x80f5a888, 0x42679217,
    0x8c3926b9, 0x6493aabf, 0xb69dd208, 0x6aaa67d4, 0x71fdb698, 0xfd2667f6, 0x0e4a5079, 0x3f3e62e6,
    0xf2ad8091, 0x470df0ac, 0x499a4a11, 0x53b0ca68, 0xef525ca7, 0xcda773a4, 0x6947947b, 0x805fb2da,
    0xf2862453, 0x03de734d, 0x09c0e4f7, 0xf434049b, 0xff1b9217, 0x49965cd8, 0x37a50a50, 0xe22eb345,
    0xd8a7c7c9, 0x7b07b29e, 0x0019a45e, 0xd5176be3, 0x3c973ff6, 0xea4e5c97, 0xdbb62782, 0x39d9bede,
    0xfd159ba4, 0x50e8cf03, 0xd9cb247f, 0x9dc13ea6, 0x15f06427, 0x4601f126, 0xb5eff796, 0x733ee48e,
    0x8eb302ff, 0x8c29ed2e, 0x6fb12db8, 0x9dc48e4f,
];

/// Stereo SILK-only VoIP at 64 kb/s into 100-byte packets, wideband, then
/// narrowband asked for from packet 150. SILK closes its input low-pass
/// filter in the pauses of the speech, and at packet 317 both channels
/// report that they are ready to switch. libopus runs
/// `silk_control_audio_bandwidth` for each channel on the same
/// `maxBits`, so each takes room for the redundant frame of the switch out
/// of it (control_audio_bandwidth.c:88-116). opus1 took that room once:
/// its SILK layer of packet 317 was not libopus's, and the packet had 100
/// bytes where libopus's has 89 (libopus 1.6.1: 91).
///
/// Packets 0 to 316 are byte for byte the fixed-point build's; packet 317
/// ends with a CELT frame, so the SILK layers are compared. The packets
/// after the switch are not: the prefill of SILK at the new rate reads the
/// float delay buffer in opus1 (see `OpusEncoder::encode_impl`).
#[test]
fn both_stereo_channels_make_room_for_the_redundant_frame_of_a_rate_switch() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 7.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(64000)).unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_force_channels(Some(2)).unwrap();
    let mut buf = vec![0u8; 100];
    let packets: Vec<Vec<u8>> = pcm
        .as_chunks::<{ 2 * 320 }>()
        .0
        .iter()
        .take(318)
        .enumerate()
        .map(|(i, input)| {
            enc.set_bandwidth(if i < 150 { Bandwidth::Wideband } else { Bandwidth::Narrowband });
            let n = enc.encode_i16(input, 320, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .collect();
    // Wideband to the end: the packet after 317 is the first at the new
    // rate.
    assert!(packets.iter().all(|p| p[0] == 0x4c), "stereo wideband 20 ms SILK-only");
    let layers = silk_layers(&packets, 2);
    let first = (0..layers.len()).find(|&i| layers[i] != STEREO_SWITCH_SILK_LAYERS[i]);
    assert_eq!(first, None, "first packet whose SILK layer is not libopus's");
}

/// `both_stereo_channels_make_room_for_the_redundant_frame_of_a_rate_switch`.
const STEREO_SWITCH_SILK_LAYERS: [u32; 318] = [
    0xfed3f602, 0xa6d88c39, 0xfd760e5a, 0x0dc28f5e, 0x70363ace, 0x59ec6565, 0x73a3b587, 0x6264d00e,
    0x061af642, 0x07cd50a5, 0x2c00024c, 0x8a668b84, 0xa4a7dd86, 0x446a2ecb, 0xa109a93b, 0xd642e19b,
    0x49ec51c2, 0xe7eaf3cf, 0xe1511561, 0xf66931c7, 0xd6a68735, 0x1ea81e40, 0x287c3862, 0x0cc50ec7,
    0xf989df6d, 0x63a27603, 0xeb318e02, 0x67e9e1fa, 0x883ff370, 0x17911cef, 0xa640ef48, 0x05929519,
    0xa54c8c94, 0x41f460d2, 0x283337a8, 0x54d45773, 0xfa8a72d2, 0x0c029078, 0x337fdb11, 0x12623827,
    0x671b00d5, 0xf2f9ef45, 0x57422408, 0xf4bbf138, 0x97f74a78, 0x1fc058ac, 0xec5b24b6, 0xd6510076,
    0xc1eacfd5, 0x6231acb6, 0x63c1d8cd, 0xd9d20891, 0xb6c72d85, 0x71a5b46e, 0x5b46afbc, 0xc206e303,
    0x19a0f776, 0xec100906, 0xf96c150a, 0x6823dcc8, 0x0aa8e73b, 0x0769d773, 0x9341e807, 0x81c8a715,
    0x30404686, 0xc38d9768, 0xc4ec18bc, 0xdaf96ed4, 0x555bef94, 0xe305948c, 0xc8ac859c, 0x72d58476,
    0x80915c8c, 0x5c8ccdfc, 0x6bcec543, 0xf7ae0fcf, 0x63bcb9ae, 0x7f71d4fa, 0xa6bcce2a, 0xef9a84ae,
    0xd953a7eb, 0xe0fb698d, 0xbc2d3944, 0x19b911df, 0xee777b2a, 0xb3f1bc7e, 0x2a550233, 0x6e2a2b71,
    0xa12f2f48, 0x1c6c402a, 0xda72bd37, 0x19903610, 0x341d9536, 0x589f2c6a, 0xead32381, 0xe9790701,
    0xd8e1f147, 0x0e3b619b, 0x6552ba84, 0x6173786c, 0x5a28372e, 0x3054eaf4, 0xae9306e9, 0x2a85b951,
    0xaad5105a, 0xb8d16e66, 0x951e69cf, 0xedf6cfec, 0x6dd85f02, 0xc732b7da, 0xed6ea8c8, 0x2f4640c9,
    0x8f1af18f, 0xf9f77cde, 0x383da7f7, 0x512cb458, 0x92ccee56, 0x89d8c482, 0x4ebd7d4e, 0x78758a75,
    0x26f9eb20, 0x03053718, 0x0b00fbce, 0xabd11ba4, 0x42e1ffdc, 0x2a11045d, 0xb5d1bbea, 0x55adbb18,
    0x1d7473c4, 0xd891d08e, 0x401d15b0, 0x53d69c02, 0x5ba8ec7e, 0xa2f77582, 0x2df4f6dd, 0x179d7785,
    0xd3e90643, 0x72c14174, 0x04f73795, 0x34da099c, 0x5c87dadf, 0xa26a8336, 0x91159196, 0x2d529de3,
    0xba1b59b9, 0x21a4fc3a, 0xff6c8565, 0x315e9416, 0x92d9095e, 0xb594b1ec, 0x1dffd1b0, 0x5635c41b,
    0xa386120c, 0x80267ad2, 0x83a63393, 0x2e9a3f2b, 0x9e4904fc, 0x724bea9c, 0xd7202ef1, 0x801c69cd,
    0xe946ba3a, 0x4b8a17cc, 0xb1e67c5d, 0xb8ad0c37, 0x4b3f4fe0, 0xe3b0cac3, 0x1c82bf37, 0xe0c2c6c6,
    0x6d92c382, 0x79b163f5, 0x71b10124, 0x2ce5f3d5, 0xb3561626, 0x53326529, 0xe5150b6d, 0x5202d0ef,
    0x21633011, 0x873426fe, 0xd7fefbd3, 0x1f1f52f3, 0xbac99806, 0xece7a916, 0xc866ffd4, 0x919665c0,
    0xe2093b68, 0x5017a6c2, 0x71e2b764, 0x6192314c, 0x5ed9787d, 0xebcc665f, 0x2d599899, 0x74e82f50,
    0x3a8963d1, 0x69c8321f, 0xaec0672b, 0x64f5483e, 0x5a53c6df, 0xbead8fac, 0x48c35d2a, 0x152c97f3,
    0x1b537408, 0x05ffd953, 0xcc673828, 0x1331b69e, 0x855b04bd, 0x6328708e, 0x8e4f4964, 0x5d1b19cd,
    0xe695576e, 0x975f9d82, 0x19e37ad7, 0x34052e9c, 0x26f5e299, 0xea76886f, 0x2482c8a9, 0xd027a53d,
    0x5b921d72, 0x22a3dc7e, 0xfebe3224, 0x951cee1e, 0x2f3ab5dd, 0x3f225e02, 0x9c595263, 0xe14b5c01,
    0xac008bd1, 0x82bf9313, 0xdeb769d4, 0x705c933f, 0x0f203895, 0x8dc76d08, 0x291adfb5, 0x2ea8c895,
    0x646f3f34, 0x13f8a3ca, 0xf5e1dfc6, 0x55f723fc, 0xe72ecb92, 0xa1c75f55, 0xb5ba0c64, 0x53a6fa4f,
    0x1a981ccd, 0x436df229, 0x04b7d94a, 0xdce459f4, 0x600a0763, 0x0e9c9768, 0x049b69bd, 0xca2a9686,
    0xf56f9152, 0xbf98f2ca, 0xd1370252, 0xc4526abc, 0x3b3cb241, 0x7e2374a5, 0xa1e41307, 0x7c5386de,
    0xbc4df6f0, 0xad8ebc5e, 0x17dd490f, 0x95763526, 0xc510a8ec, 0xa96aa28b, 0xbad25e29, 0x53376555,
    0x0d683dc8, 0x7116f6e3, 0x266cb4bf, 0x438c0afd, 0x9b7dcd9c, 0xfdf5703b, 0x21b30fbd, 0x8e1754b9,
    0xe2328b92, 0x96b27482, 0x8fc43c86, 0x5b5d3bb1, 0xd272f12f, 0xd1cae595, 0x194a830a, 0x5fde735d,
    0xe07d6eec, 0x14b27292, 0xbbb240a7, 0xd085794f, 0xe5fa8363, 0x516b559e, 0x15cecc65, 0x0b72a593,
    0x8084da55, 0x443e47f5, 0xf74337b8, 0xefc42571, 0xdd2de517, 0xe2e2a757, 0x4527395d, 0xd94ed16c,
    0x7519bb19, 0xb95c0160, 0x6532c0ca, 0x0c6df758, 0xbdd347ec, 0x75ad802a, 0xe0e61ddc, 0x0f586f1f,
    0xe9ad0a99, 0x72dbb770, 0xef22bce3, 0x6eee6f14, 0x320a95ec, 0x964be9bf, 0x9413374f, 0x84dc119e,
    0xe99e5758, 0x0fe76b43, 0x013aea4f, 0xbeb125ee, 0x81a861ab, 0xf8418b6f,
];

/// Stereo input coded as mono at 7 kb/s, then at 40 kb/s again: the first
/// stereo packet (171) is also the first at SILK's new internal rate. Its
/// redundant CELT frame was announced by packet 170, which coded one
/// channel at 14 kb/s in a pause of the speech. libopus's silk_Encode first
/// starts the side encoder afresh (enc_API.c:194-211), then resets both
/// encoders for the rate switch and gives each the mid's low-pass state
/// (enc_API.c:222-236). Both channels then run the same low-pass
/// transition.
///
/// In opus1 the side encoder is started afresh after that reset, inside
/// `silk_encode_prefill` (`enter_stereo`). Without the copy of the
/// mid's low-pass state after `enter_stereo`, its side signal skips the
/// transition filter, and from packet 171 on 430 of the 600 packets are
/// not libopus's. With the copy only packets 170 to 172 differ:
/// - 170: its redundant CELT frame (the fixed-point build's CELT encoder
///   is not the float one).
/// - 171 and 172: the prefill of 171 reads the float delay buffer (see
///   `OpusEncoder::encode_impl`). With a temporary change that gave the
///   prefill the 16-bit samples and the integer fade of the fixed-point
///   build, packet 172 was libopus's. Packet 171 still was not; that cause
///   was not found.
///
/// The packets after them are compared.
///
/// 16 kHz VoIP, 10 ms, complexity 5, the bitrate deciding the channels and
/// the bandwidth.
#[test]
fn stereo_that_starts_with_a_silk_rate_switch_runs_the_side_through_the_transition_filter() {
    let mut mono = k24_common::speech(16000, 6.0);
    // A pause of 1.2 s, in which SILK may change its rate.
    for v in &mut mono[16000 * 14 / 10..16000 * 26 / 10] {
        *v *= 0.001;
    }
    let pcm: Vec<i16> = k24_common::to_stereo(&mono, 16000)
        .iter()
        .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let frame = 160;
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.set_complexity(5).unwrap();
    let mut buf = vec![0u8; 4000];
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(272) {
        enc.set_bitrate(Bitrate::BitsPerSecond(match i {
            0..40 => 40000,
            40..170 => 7000,
            170 => 14000,
            _ => 40000,
        })).unwrap();
        let n = enc.encode_i16(input, frame, &mut buf).unwrap();
        if i == 170 || i == 171 {
            // One channel at 12 kHz, then two at 16 kHz.
            assert_eq!(buf[0] & 0xE4, if i == 170 { 0x20 } else { 0x44 }, "packet {i}: first byte {:02x}", buf[0]);
        }
        if i >= 173 && fnv1a(&buf[..n]) != SIDE_LOW_PASS[i - 173] {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// `stereo_that_starts_with_a_silk_rate_switch_runs_the_side_through_the_transition_filter`:
/// packets 173 to 271.
const SIDE_LOW_PASS: [u32; 99] = [
    0xd9d480ef, 0x1f3dcaf4, 0xd70c52ed, 0xb74fa858, 0x800b933e, 0x98ab5b84, 0x5d53c7f6, 0xd1438718,
    0xc27d478d, 0x76500a3d, 0x3f52c260, 0x1e0a6378, 0x1712bef4, 0x7bebdb27, 0xe04be297, 0x7a741697,
    0xf626b06e, 0x4bc13a8e, 0x38670f9c, 0x66e5c33b, 0xfd4af166, 0x87fb03a9, 0x28a093fd, 0x558fd38b,
    0xde165435, 0x38a74e0a, 0x187a2259, 0xaa78e9f0, 0xf95e133e, 0x64314fb8, 0xf0890f2c, 0xeac230b5,
    0x84a1c473, 0x66eaa555, 0xa6958c91, 0x7b6ab673, 0x39cdbab6, 0x03b0f9d4, 0x4d9d108f, 0x029f1a71,
    0x28f9451a, 0x205cb2d8, 0xec53c538, 0x3961d666, 0xba1ce1a4, 0x8eccfab4, 0x00f5903b, 0x54fa49ee,
    0x61f363c0, 0x6fce7a14, 0xc822c233, 0x658c03e4, 0xdca74556, 0x767aab3d, 0xddf24304, 0x14f889d0,
    0xa5473464, 0x8c878d84, 0xec5a4c01, 0xec0f97b8, 0x59848766, 0x1d359f16, 0x5794f2de, 0xd4ea17ec,
    0xd42bee07, 0x1de6930c, 0x7eb407c6, 0x248575cb, 0x20c74954, 0x6e6435c8, 0x3f239503, 0x3a68957e,
    0x0164c2c7, 0x5f09bf03, 0xa0065158, 0x159f9705, 0x943899b1, 0x08401d9a, 0x7e7a6161, 0xbf8179be,
    0x94d03e7f, 0x55c47646, 0xb5460d65, 0xc070ec02, 0x2e0e87a1, 0xa2303082, 0x260c6fbe, 0x97b27ea6,
    0x0aa3392e, 0xa9d20ab3, 0xeb9a0327, 0x06a1bc05, 0xf475af09, 0x1acdd66c, 0xe78a770b, 0x42613358,
    0x10892df9, 0x94da293d, 0xa1d9c48b,
];

/// `OPUS_SET_PREDICTION_DISABLED` in SILK: libopus's silk_Encode marks
/// every channel of the input as just reset before each packet
/// (enc_API.c:172-177). Each packet then codes its first frame without
/// prediction from the packets before, and the LBRR frames that the
/// previous packet left are dropped (enc_API.c:268). Stereo VoIP at
/// 32 kb/s with in-band FEC at 5 % loss, prediction disabled in packets
/// 20-39, 60-79 and 100-119. Every packet is the fixed-point build's.
/// Without the reset, packets 20 to 139 were not.
#[test]
fn prediction_disabled_codes_every_silk_packet_alone_like_libopus() {
    let pcm: Vec<i16> = k24_common::to_stereo(&k24_common::speech(16000, 3.0), 16000)
        .iter()
        .map(|&v| (v * 32768.0).round() as i16)
        .collect();
    let frame = 320;
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(32000)).unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_inband_fec(true);
    enc.set_packet_loss_perc(5).unwrap();
    let mut buf = vec![0u8; 4000];
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(140) {
        enc.set_prediction_disabled((i / 20) % 2 == 1);
        let n = enc.encode_i16(input, frame, &mut buf).unwrap();
        assert_eq!(buf[0] & 0xE4, 0x44, "packet {i}: not stereo SILK-only wideband");
        if fnv1a(&buf[..n]) != PREDICTION_DISABLED_WALK[i] {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// `prediction_disabled_codes_every_silk_packet_alone_like_libopus`.
const PREDICTION_DISABLED_WALK: [u32; 140] = [
    0x7e3b4254, 0x156ed64e, 0xf60b392d, 0x22509976, 0x8bdf0330, 0xbddceefc, 0x7944af59, 0xe5ca8eb8,
    0x71cd396e, 0xcd90ecb0, 0xde470462, 0x60d3ad41, 0x738e911f, 0x59e33387, 0x49c89c17, 0x4c6e98a5,
    0x5a78dc0a, 0x3482eb9a, 0x8411a47f, 0x83efd4ef, 0x1ff7d9ef, 0xedbe6c47, 0x3ded5162, 0x3cbf3b77,
    0x52dec53f, 0x23be2376, 0x69fa43db, 0x3175bc59, 0x3bb7aed5, 0xf716598f, 0xf1d3dfc4, 0x57e88a76,
    0xff0882d5, 0x51d4b526, 0x5cb7bde2, 0x3f768222, 0x8897318c, 0x5a4a1bdb, 0x011c7117, 0x1a96468e,
    0x357dc672, 0xcbc9e7ea, 0x9d2d936e, 0x6d7d47f3, 0x553b94ca, 0x4520a9c8, 0x3ab07ad5, 0x4cc13a65,
    0x60e88003, 0xe4cbe74b, 0xcc5a94f1, 0xed491ae1, 0xb3fb8ce2, 0x58484aa6, 0x4da0e14c, 0xfcaaea4a,
    0x552b068a, 0x884586a7, 0xa6f3ea88, 0xe8e501d6, 0xa0570a3c, 0xe2d4d17c, 0x3677708b, 0x46b2cbd2,
    0x33cb253b, 0x5f06d6cd, 0x57bd9d29, 0x526c2d66, 0x0974402b, 0x4bb86543, 0xd16cbfb3, 0xda4e596e,
    0x3876d7ad, 0xa268474b, 0xb00bf821, 0x643eb418, 0xc0e9f3f8, 0x65dd4285, 0xe46e19c8, 0x214a17ea,
    0xee65d352, 0xa5c6255f, 0xf609b3be, 0x8fbf8762, 0xc0eddb45, 0x2fdf2460, 0xc6e2c685, 0xe1c5a82d,
    0xba910941, 0x35ff4705, 0x38751d53, 0x4a40eb27, 0x7b234eef, 0x56c4db36, 0x2033bc3a, 0x85e0aa6a,
    0xd29ae0f9, 0x4c45bd38, 0x41ea527c, 0xdbcfd7ca, 0x57f2381e, 0x16125ae3, 0x79250d5c, 0xa3c38f22,
    0x2fdf2510, 0xad5ac6a7, 0x2681e34e, 0xbddd3b77, 0xf18a782f, 0xde59c1fe, 0xae6abaac, 0xf34a74cf,
    0x08edd435, 0x229b9f44, 0xe7f05701, 0x274bbed9, 0x57c4a7dc, 0x15325fa4, 0xad98be4d, 0x65c1eac7,
    0xa44bc374, 0xbedabaa7, 0x5bded5ac, 0xc1f0cd86, 0xcd5d530e, 0x2ddcdaa9, 0x58020de2, 0x5f296724,
    0xae80041f, 0x8e7154b4, 0x0b6a5fa2, 0xbd83faa0, 0x92f92011, 0xaf2986bb, 0x69e26356, 0xfac027a7,
    0xd32efad6, 0x3c38935a, 0xbae7420d, 0x43171931,
];

/// A switch of SILK's internal rate in stereo: libopus resets both
/// encoders and gives each the mid's low-pass state, with the rate it ran
/// at, so that both go on with the same transition (enc_API.c:222-236).
/// Stereo VoIP at 40 kb/s, SILK-only, narrowband for 1.5 s, then wideband
/// asked for. The input is digital silence from 1.4 to 3 s, where SILK
/// switches: there the prefill reads the same zeros in both builds (see
/// `stereo_that_starts_with_a_silk_rate_switch_runs_the_side_through_the_transition_filter`).
/// Every packet is the fixed-point build's. With the side keeping its own
/// low-pass state, packets from 77 on were not.
#[test]
fn a_stereo_silk_rate_switch_gives_the_side_the_mids_low_pass_state() {
    let mut mono = k24_common::speech(16000, 3.2);
    mono[16000 * 14 / 10..16000 * 30 / 10].fill(0.0);
    let pcm: Vec<i16> = k24_common::to_stereo(&mono, 16000)
        .iter()
        .map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let frame = 320;
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(40000)).unwrap();
    enc.set_complexity(5).unwrap();
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_force_channels(Some(2)).unwrap();
    let mut buf = vec![0u8; 4000];
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(160) {
        enc.set_bandwidth(if i * 20 < 1500 { Bandwidth::Narrowband } else { Bandwidth::Wideband });
        let n = enc.encode_i16(input, frame, &mut buf).unwrap();
        if fnv1a(&buf[..n]) != STEREO_RATE_SWITCH[i] {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// `a_stereo_silk_rate_switch_gives_the_side_the_mids_low_pass_state`.
const STEREO_RATE_SWITCH: [u32; 160] = [
    0x5a5fd235, 0x92a6be5e, 0x977a884f, 0x2ac2e5f5, 0x8bf7ccef, 0x063e871b, 0xb43b3080, 0x532dd60d,
    0x6d07e16b, 0xbe6d18cc, 0x8796afde, 0x7b731100, 0x1eb4561a, 0xd3582eca, 0x9c51006a, 0x04019606,
    0x73172cc2, 0x1517afc1, 0x14856657, 0x0fb834a4, 0x00089715, 0xa216b1e3, 0x99636947, 0x04558160,
    0x48883de1, 0x78097cb6, 0xcc582324, 0xc5f42d98, 0xcdeefecf, 0xa811029d, 0x251cb8b8, 0xcb59a9e9,
    0xd5c77077, 0xd8748d4b, 0x6dc8bc62, 0x4d48248e, 0x3cd3889f, 0xec9a4757, 0x6fa271d3, 0xb07e60ee,
    0x38bb696e, 0xfee62507, 0x6f1f56e2, 0x08642ca7, 0xce4ac917, 0xdb74b00c, 0x95c13cc7, 0x184da708,
    0x2f85d042, 0x107141e7, 0xfed6c687, 0x2fecc9e2, 0x2b529006, 0xbb1df9c2, 0x1e2f093a, 0x609b4102,
    0x1cdcf3c5, 0x6a0e8242, 0x941535ff, 0xf77a9860, 0x3244b500, 0x1ea80242, 0x18ad19e6, 0xda4997ba,
    0x9a339ad7, 0x1595d381, 0x9d6d391d, 0xf3ff8df7, 0x89c190dd, 0x14013f22, 0xed58e3ef, 0x288db917,
    0xf2b0756e, 0xa5eeea7d, 0xa5eeea7d, 0x0182ebd0, 0xf01bba00, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5,
    0x56498475, 0x56498475, 0x10cee923, 0x17c5fdc5, 0x56498475, 0x56498475, 0x19fa72d9, 0xa077b81d,
    0xd1f93d49, 0xf817c8cb, 0x0aa733fe, 0x1722dfb7, 0x696d3578, 0xf167616e, 0x61757b3b, 0x2c2388e6,
];
