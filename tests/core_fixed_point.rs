//! Packets of `OpusEncoder` against the fixed-point build of libopus 1.6.
//!
//! With the `Voip` application and 16-bit input, the SILK-only packets of
//! opus1 are byte for byte those of the fixed-point build: its SILK
//! encoder is a port of `silk/fixed`, and it filters the input with the
//! integer `hp_cutoff` (see `OpusEncoder::encode_i16`). The tests here hold
//! a 32-bit hash (FNV-1a) of every packet that build wrote for the same
//! input and the same settings, packet by packet.
//!
//! The packets were made with the libopus 1.6 source of this repository's
//! `opus-head-sys` dependency, built with `cc` and `-DFIXED_POINT
//! -DOPUS_BUILD -DVAR_ARRAYS -DHAVE_LRINTF -DHAVE_LRINT` and no other
//! options, driven by `opus_encode` with the controls named in each test.

mod k24_common;

use opus1::range_coder::RangeCoder;
use opus1::silk::dec_api::SilkDecoder;
use opus1::{Application, Bandwidth, ForcedMode, OpusEncoder};

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
        enc.bitrate_bps = p.bitrate;
        enc.complexity = p.complexity;
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
    0x748c5126, 0xcc9c1cab, 0x70822472, 0xaa92e8c4, 0x98b37de4, 0x4c4a5c08, 0xf970f270, 0xafaba3b3,
    0x002773c9, 0x5db27013, 0x503f2932, 0x0a86937e, 0xe5cddf3a, 0xdd491db3, 0x6ebec421, 0x3621b9bd,
    0xef1dc6a7, 0x5a6fc416, 0xc1587e69, 0x03877f8c, 0xc2920034, 0x23ad66bd, 0xe278d290, 0x6e3fb000,
    0x3c3299bc, 0xbee54d3d, 0xa41bca7d, 0x99847a3b, 0x6d5bc821, 0x10f95fad, 0x582b3d2d, 0xec0c0da9,
    0x374ab574, 0x60838227, 0xedb8b9c1, 0x75c727d7, 0xf3e16d1b, 0x3a251eb6, 0xe6f5d826, 0x30e36dc2,
    0xb0de6d4d, 0xb9bc2e47, 0xa1b96d2b, 0x0dd5dde0, 0x06d13416, 0x1078a0a1, 0xf7bd2afc, 0xa8fa792e,
    0xaa280693, 0xb15b08c7, 0xba154110, 0x3b2ebcfb, 0xf0a5b35e, 0xa8cc122c, 0xe49291ca, 0x68073f95,
    0xcd789b94, 0xbea97fbb, 0x50016a07, 0x365bef87, 0x4d1d5e96, 0x093ee9d9, 0x54d171e9, 0x7d8cc5c3,
    0x3aff23ef, 0xcc4433f2, 0x35e5b898, 0xbe5c78f9, 0x34fb0eab, 0x8a4dbd27, 0x25f0ae84, 0x85fd0e9a,
    0xc44f64c2, 0x719d8913, 0x763e002a, 0xb7d28a4c, 0x2ed3086b, 0x376cc877, 0x7b38c3ba, 0xb119a3f3,
    0x3049dfa0, 0x5553b58d, 0x71332ccd, 0x4ee73bbb, 0x5dcebf9f, 0xeb7e623a, 0x6eb6874e, 0x4d1a2ae1,
    0x37fba88f, 0x05e53ac1, 0xa351ed70, 0x0d2ec0f4, 0x1a799646, 0x72234b96, 0x7239d6e6, 0xa2b4c868,
    0xc3ab4b10, 0x78361dfa, 0xffc2ff49, 0x6f6a2bc1, 0xee0b349e, 0xc3cfd77b, 0xecd27412, 0xb2188d46,
    0x1998760f, 0xf4d7cf7b, 0x7b7c5e23, 0x2246f5e1, 0x73da5c8f, 0x847b420b, 0x9503e0e5, 0x39a0b89d,
    0x952e4973, 0x69506f0a, 0x2d36c029, 0x309b165b, 0x200dc77f, 0x10023b4e, 0x555a6659, 0x2845b351,
    0xd9c418c3, 0x02551a49, 0x43dd89c8, 0x03400857, 0xe49fd032, 0x3a438632, 0xbfedd2c4, 0x0ca172df,
    0xc70b48d4, 0x430c5bf8, 0xed07a7ab, 0x73f2888a, 0x2ca556b6, 0x04659208, 0xa2fdeeea, 0x85e69c99,
    0xbdd25dcc, 0xe567716d, 0xbc5d285c, 0xf0d3dce4,
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
    0xad70a6fa, 0x52d71d40, 0xf28d5153, 0xc7b096e8, 0x637cd15c, 0xee221a5a, 0x3b537ebc, 0xb3e2a18d,
    0xfe38c42d, 0x278ee0e7, 0x912458f4, 0xba196613, 0x7a046ec2, 0x29f34bbe, 0x7f587f4b, 0x1d2038d8,
    0x923a5453, 0x4a53cc32, 0x10384847, 0x919ac997, 0x0347d928, 0xad1dbd7a, 0xb92ca85a, 0x82a0881d,
    0x2034db48, 0x961ddf7d, 0xbab85893, 0x18cbb14a, 0xb11b4856, 0x3c2bc82e, 0xd126306e, 0x0b266629,
    0x57656016, 0x0a571aca, 0x6955f341, 0x4a4fbe95, 0xc001e0c5, 0xa4d77678, 0x2c32c298, 0x8b95f12e,
    0x72c5ce36, 0xde67feea, 0x9d785bdd, 0x60579209, 0x6acc8679, 0x2599ee97, 0x0cd16948, 0x1b1a2c17,
    0xb5850406, 0x53d45243, 0x5ec96913, 0x00007195, 0xcbe5844a, 0xeb210d80, 0x26f39389, 0x443f24c9,
    0xa72a18d2, 0x5cd421d2, 0x3451d7bc, 0x4a324090, 0xf42fd7d4, 0x1eea00e6, 0xc3f021bd, 0xcb0af5cf,
    0x2d010948, 0x57c3c5df, 0x3fe3d1ed, 0x82f27b6e, 0x55320ee3, 0x248c4bab, 0x9f3735b5, 0x6f3bed86,
    0x429c03c4, 0x398d6630, 0x1d36a63a, 0x734b2330, 0xf36ea5e9, 0x93ebe6b0, 0x532e4253, 0x93c46ac3,
    0xef71839c, 0x07fc76ac, 0x88c799bb, 0x88963514, 0xbdd13938, 0xc7ab50cd, 0xf9eb9170, 0x6c776737,
    0x50fd8545, 0x0f5b8001, 0x6914e272, 0x2faac3cf, 0xd0c07e5c, 0xde55a354, 0x5a67d020, 0x606963e0,
    0x4f13a269, 0x2601a4fd, 0x9ac566c2, 0x9048d8a4, 0x9b24f58d, 0x6e4cccab, 0x39397846, 0xbdc7bd7e,
    0xa2eb818e, 0xe5c5e145, 0xc0d108f6, 0x41bd3407, 0x2e07a1ba, 0x0f2eabdc, 0x39e4f676, 0x8d5ba8d5,
    0x07cac9ed, 0x03946df7, 0x10868a47, 0x973d7c10, 0x999674db, 0x53509af1, 0xecc9d946, 0x838612de,
    0xe2645c9a, 0xf99dfe64, 0x117ef8c5, 0xa27f0a30, 0xecfb2636, 0x7a699003, 0xa6003496, 0x48143c47,
    0x0071e7fa, 0x5798b75b, 0x06603a6c, 0xf024e7c1, 0x6e7b0d8d, 0x6a092ee0, 0xe60f6d51, 0x452e6195,
    0x00b78469, 0xe8ea6c4b, 0x4ed10163, 0x7e4e99c0,
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
/// Hybrid at 32 kb/s, VoIP, complexity 9.
#[test]
fn hybrid_at_a_constant_bitrate_codes_the_silk_layer_of_libopus() {
    let pcm: Vec<i16> = speech_16k(3.0).iter().flat_map(|&v| [v, v, v]).collect();
    let mut enc = OpusEncoder::new(48000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;
    enc.complexity = 9;
    enc.use_cbr = true;
    enc.set_force_mode(Some(ForcedMode::Hybrid));
    enc.set_bandwidth(Bandwidth::Fullband);
    let mut buf = vec![0u8; 4000];
    let packets: Vec<Vec<u8>> = pcm
        .chunks_exact(960)
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
    0xd01bb46e, 0x25469c8a, 0xa6de1c46, 0x1fc065cf, 0x42b662da, 0x1c41fd66, 0x230bb95d, 0x17a27416,
    0xc35a2a86, 0x3f10cd4e, 0x92a7c3ea, 0x8f735111, 0xa4dfe89e, 0x3ed6be7a, 0x817933ef, 0x7be63358,
    0x87245787, 0x2293d84b, 0x165a2291, 0x700a57ed, 0xd8752dc9, 0x0ea9d4db, 0xa44a9814, 0xed0c0464,
    0x9ae031e0, 0xc30cfdad, 0xd1e1279b, 0x366415b7, 0xfd865bb6, 0x4760d027, 0x83c43811, 0x9cc79d1c,
    0xbf908657, 0xd46f3d3b, 0x5aeb31a4, 0x5bf4493a, 0xb6f9c2c6, 0x63d32fdf, 0x2e04125e, 0x4cd34d7b,
    0x2b3d5955, 0x30caf810, 0x92e1b9c6, 0x7edaa04b, 0x134f9158, 0x6fb0da35, 0x926af73c, 0x2bd655e8,
    0x9efeea88, 0x607118c5, 0x7dc9fe42, 0xd8940e5b, 0x3e093787, 0x030e5f0c, 0x5927bd77, 0x588cf1d0,
    0xee863afa, 0x0042bdf2, 0xc466e01e, 0xa68f2ff2, 0xe92da141, 0x18d4d9f7, 0x048ca5fd, 0x06de2a44,
    0x7cf9c38d, 0x05b02a6e, 0x1a058350, 0x55a53596, 0x3c653fcf, 0x652662da, 0xb0f0682d, 0xc6fdbd6c,
    0x97215b36, 0x5eba01ee, 0xc0b46b6b, 0x8734543d, 0x57722db1, 0x579c9482, 0x86d839fd, 0xc6c5d148,
    0x98759bb0, 0x84430c39, 0x6d77d80e, 0x01a17265, 0x6b7c422e, 0xb9d212f1, 0x99ada34b, 0x4115e711,
    0xac6186d1, 0xca96050e, 0x2b235256, 0x5ff51af4, 0xc1708f53, 0x3e0d2348, 0xa82aac69, 0x4c59e0c4,
    0x79cbfee6, 0x18d46308, 0x5584851a, 0xcf07712b, 0x12f03b7e, 0x6d4f0f2a, 0x257d8bb5, 0xf278a77a,
    0x840f2604, 0xe1700408, 0x9eab71a4, 0xb51117d2, 0x38dbf6f2, 0xe6cdf6af, 0x266c7656, 0x1421d54f,
    0x95a495f4, 0x78a57d80, 0xe1ed25b2, 0xe503e3c4, 0x64fa96e9, 0xc28765f7, 0xc9e9da59, 0x9598884b,
    0x318e195d, 0x7a524a2c, 0x7a61c4be, 0x3962725f, 0x7810885c, 0xdb32052b, 0x8817f7e7, 0xe20091d8,
    0xd6108f0d, 0x9c54b251, 0x23e76ee6, 0xbf95e41d,
];

/// Stereo SILK-only VoIP at 64 kb/s into 100-byte packets, wideband, then
/// narrowband asked for from packet 150. SILK closes its input low-pass
/// filter in the pauses of the speech, and at packet 317 both channels
/// report that they are ready to switch. libopus runs
/// `silk_control_audio_bandwidth` for each channel on the same
/// `maxBits`, so each takes room for the redundant frame of the switch out
/// of it (control_audio_bandwidth.c:88-116). opus1 took that room once:
/// its SILK layer of packet 317 was not libopus's, and the packet had 100
/// bytes where libopus's has 89.
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
    enc.bitrate_bps = 64000;
    enc.complexity = 9;
    enc.set_force_mode(Some(ForcedMode::SilkOnly));
    enc.set_force_channels(Some(2)).unwrap();
    let mut buf = vec![0u8; 100];
    let packets: Vec<Vec<u8>> = pcm
        .chunks_exact(2 * 320)
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
    0xfed3f602, 0xa6d88c39, 0xfd760e5a, 0x0dc28f5e, 0x8bb210b3, 0x4c15edc3, 0xb75be9c1, 0x98f7b86f,
    0x5be32dd7, 0x6801187f, 0x580278b6, 0x796f75bb, 0x2204ecf4, 0x5c1510e8, 0x057f0151, 0x26b5a3cf,
    0xae5f944a, 0xe874db95, 0x19a40325, 0xd20ca58f, 0x95dd3e9c, 0xf3c83d12, 0xf97e195a, 0xb02ef838,
    0x30ffbb26, 0x395bef5a, 0x61e34522, 0x1b767faa, 0xdd3bf42d, 0x947541c3, 0xe1243c3e, 0xa68e9747,
    0x571240e4, 0xea49ea03, 0x160caa47, 0x08a785dd, 0xd36ffba4, 0x8bb1eb0a, 0x04338fcf, 0xa54ee7d7,
    0x090742f8, 0xb8876589, 0x23b616e4, 0xc3dd892b, 0x39d81b96, 0x2a5e7ef0, 0xfb5721c7, 0xe5af1fbc,
    0x9a349b75, 0xa79e351c, 0x3745a389, 0x4e02eebd, 0xe6c4f393, 0xabb8a956, 0xf4471ff9, 0xd1167e00,
    0x8cd8f03b, 0xee5b8293, 0xa907d27d, 0x67a6b919, 0x3b87f879, 0xdec96bb1, 0x6b4ee32e, 0x58d2a84a,
    0x6e36e8bd, 0xe2e85f2a, 0x953098dd, 0x0fdc1e9a, 0xcf225d39, 0x6bdb784a, 0x56af1fe2, 0x46bd3be8,
    0x24c14ead, 0xeda70a0d, 0x9b5a04ef, 0x591c6dc2, 0x0c628999, 0xc4e261db, 0x477d2dd8, 0xc352d4cc,
    0x95f429e2, 0x7e4f6905, 0x93c81bd4, 0x6d7fdb8f, 0x876c99f0, 0x9616ea81, 0x6f6aac11, 0x26840868,
    0x98c5e46a, 0xcb42ba62, 0x55a4ab91, 0xac4505af, 0xcf6e0a91, 0xba5e8ac3, 0xbdff99ec, 0x8d67ebac,
    0xf315c977, 0x09929cf4, 0xb9f876c7, 0xa47373bf, 0x6c048611, 0xe302374d, 0xd7de818a, 0xecc0cd18,
    0x3679e5f9, 0x05b0aa47, 0x6809e0ef, 0x7104e090, 0x8fbdbe67, 0xacfbdf7b, 0xedc4c45b, 0x4a727e0c,
    0xcf0327e2, 0xd316a57f, 0xb6c06b9a, 0x88ddd45e, 0xe973525f, 0x8666fd01, 0xf1106c4b, 0xc0737024,
    0x0ece01d0, 0xa02f7d8c, 0x31ca0c44, 0x7224a4c8, 0xa6925709, 0x76e8f446, 0xb43ed4fa, 0xaa2fba2f,
    0x5283a0bc, 0xc73a1874, 0x5cdfc61b, 0x58ddfb16, 0x6e3cd075, 0x27367709, 0x19491d16, 0xfbac1163,
    0x24b0bcf2, 0x5139af60, 0xecbfa1a5, 0x25255280, 0xbb9f95f3, 0xf696318c, 0x0e440acf, 0x0ca324e4,
    0xe8a5d348, 0x8d9b4025, 0x16e0450d, 0xb9b0ec3e, 0xd9a0c2ba, 0xcbad54d5, 0xa7e27ca7, 0xde0bd8b1,
    0x3cae04c1, 0x5380e565, 0xf850d3e7, 0xcff81236, 0x75c21505, 0x68e0def6, 0x973fdaa9, 0x515fb13a,
    0xeb3c421b, 0x2eafa01b, 0x907daec3, 0x20c19801, 0x9af1f133, 0xf2889f97, 0x2770aacd, 0x71dc987e,
    0x41a82a9b, 0x8e0e715e, 0x269b3550, 0xb5ce2725, 0x7b3970f8, 0x73e458e5, 0xcb505a67, 0xf0fd2327,
    0x05dc6239, 0x3ae1784f, 0x1be4523b, 0x894fa803, 0xbe32310a, 0xd1a0f627, 0x0f17e2ca, 0xdb23c608,
    0xc06f08bb, 0x3e846cf9, 0x296db634, 0xd0015b25, 0x8c49bef7, 0x82b568d9, 0x3eb0741e, 0x94309e18,
    0xde36e630, 0x92460254, 0xb27ca185, 0x5bf67886, 0xd604d7dc, 0x9fa1ea5a, 0x549b943d, 0x1f08cde1,
    0x25eefc96, 0xa90ab42a, 0xf8cb2622, 0x2652c068, 0xcc366209, 0x436b9df3, 0x033781d8, 0xb98bfc10,
    0xf5056eec, 0x526d84a9, 0x9f756689, 0xb660ef5a, 0x67d69b02, 0xc5d83448, 0xe1e0895c, 0x131ac1ee,
    0x9b6404b8, 0xaffa288d, 0xcb0c2c00, 0xc32ab460, 0x2715e80d, 0x9a20f616, 0x22c82b0d, 0x4a601b42,
    0x2413172e, 0xed7e3a0f, 0x08f7e2fa, 0x8e0b9f46, 0x1edee247, 0x80031814, 0x9f2481b3, 0x633d9d8f,
    0xb44306a5, 0x8bd71e9b, 0x67daae68, 0xcb2dc608, 0x407e16d0, 0xf3b44266, 0x5bd05f11, 0xfe0b2784,
    0x3da2be57, 0x3be27380, 0x2e636e13, 0x3dabe182, 0x89150f24, 0x384d36fc, 0x56ade025, 0x19455998,
    0xcba46b8c, 0x4496c371, 0xab2d2daa, 0x29735692, 0xa92b8e07, 0x6170e44a, 0xc631c2e7, 0x092e33e7,
    0xc194c8c4, 0xdf7e534a, 0x3c8679bc, 0xbec6453e, 0x97cdf3e5, 0xd2ae6243, 0xd476b40a, 0xae0f2da5,
    0x8fc2261e, 0xc367edd3, 0xaab21385, 0x3a764790, 0xfccdb9a2, 0xc64c8f2f, 0x1bfa785f, 0xbe00d976,
    0x8258cc8b, 0x0076ee78, 0x46585a6b, 0x6a8286f3, 0x7e11f792, 0x4861b8ae, 0x33f0b30a, 0x92095655,
    0x90225abf, 0x2d644cbe, 0xab711670, 0xe03986d9, 0xe4a5c0c1, 0x32d4153b, 0x13d88cd5, 0xf24c8059,
    0x80fc86a1, 0x7e6a7932, 0x51e7916b, 0xe0d40765, 0xd3ef0f53, 0x4856f18a, 0x5e121551, 0x5cadb088,
    0x8d99d62d, 0xfd282ca1, 0x62d530c0, 0xc8e176c7, 0xa349bbdf, 0x5262bf45, 0x693f6b7f, 0x5fdfcb43,
    0xb3546c98, 0x9bfb3311, 0xa1431f6e, 0xb404c06d, 0xf1b93484, 0x1c218427, 0xd908e1da, 0x407b63d3,
    0xa98e425b, 0x54fc947c, 0x2d8ea966, 0xa2ef82dd, 0xd93d7ed4, 0x0de6a84f,
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
/// not libopus's. With the copy only packets 170 and 171 differ, for the
/// known reasons: the redundant CELT frame of
/// 170 (the fixed-point build's CELT encoder is not the float one), and
/// the prefill of 171, which reads the float delay buffer (see
/// `OpusEncoder::encode_impl`). The packets after them are compared.
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
    enc.complexity = 5;
    let mut buf = vec![0u8; 4000];
    let mut differ = Vec::new();
    for (i, input) in pcm.chunks_exact(2 * frame).enumerate().take(272) {
        enc.bitrate_bps = match i {
            0..40 => 40000,
            40..170 => 7000,
            170 => 14000,
            _ => 40000,
        };
        let n = enc.encode_i16(input, frame, &mut buf).unwrap();
        if i == 170 || i == 171 {
            // One channel at 12 kHz, then two at 16 kHz.
            assert_eq!(buf[0] & 0xE4, if i == 170 { 0x20 } else { 0x44 }, "packet {i}: first byte {:02x}", buf[0]);
        }
        if i >= 172 && fnv1a(&buf[..n]) != SIDE_LOW_PASS[i - 172] {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "packets that are not libopus's: {differ:?}");
}

/// `stereo_that_starts_with_a_silk_rate_switch_runs_the_side_through_the_transition_filter`:
/// packets 172 to 271.
const SIDE_LOW_PASS: [u32; 100] = [
    0xfa488a70, 0xd9d480ef, 0x1f3dcaf4, 0xd70c52ed, 0xb74fa858, 0x800b933e, 0x98ab5b84, 0x5d53c7f6,
    0x8976b5fb, 0xb81ae492, 0x8a7725b4, 0x3f52c260, 0x1e0a6378, 0x1712bef4, 0x7bebdb27, 0xca943e03,
    0x5f5a7c9f, 0xd5f71b37, 0x4bc13a8e, 0x38670f9c, 0x66e5c33b, 0xfd4af166, 0x87fb03a9, 0x28a093fd,
    0x94d888e7, 0x01361b6a, 0x38a74e0a, 0x187a2259, 0xaa78e9f0, 0x19037cc2, 0x64314fb8, 0xf8dfbbc6,
    0x3d88cc7b, 0x57694b53, 0x2c7b3732, 0xa6958c91, 0xd3c496a6, 0x39cdbab6, 0x03b0f9d4, 0xefd39029,
    0x029f1a71, 0x747fcd6b, 0x3c2f5401, 0xec53c538, 0xb621b90e, 0xba1ce1a4, 0x835ac1eb, 0x00f5903b,
    0xb792d0f0, 0x53d845e2, 0x6fce7a14, 0xc822c233, 0xb81e0010, 0x2192f85b, 0x7e3424e7, 0xd439c566,
    0xe960ba1b, 0x17ae0e6f, 0x1f3edef7, 0x48e9d62f, 0xec0f97b8, 0x2e2ad44d, 0x1d359f16, 0x267162bf,
    0xd3554bfa, 0x648cb54f, 0x05b752d8, 0x7eb407c6, 0x248575cb, 0x20c74954, 0xa93fd718, 0x4307f812,
    0x3a68957e, 0x0f91e411, 0x5f09bf03, 0xa0065158, 0x203ae58d, 0x943899b1, 0x08401d9a, 0x7e7a6161,
    0xbf8179be, 0x94d03e7f, 0x55c47646, 0xb5460d65, 0xc070ec02, 0x2e0e87a1, 0xd7d34728, 0x260c6fbe,
    0x97b27ea6, 0x1b76945b, 0x9705f98e, 0xec20f43e, 0xca61ded7, 0x92a3ca40, 0xa4f95a41, 0x2aa655bc,
    0x7b329ccb, 0x46735df8, 0x2aa2cf4b, 0xda7fa4ea,
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
    enc.bitrate_bps = 32000;
    enc.complexity = 9;
    enc.use_inband_fec = true;
    enc.packet_loss_perc = 5;
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
    0x7e3b4254, 0x0a6b8e7b, 0x9ff13a3d, 0x6847f256, 0xe14c447d, 0x51423cb6, 0xa6a6f899, 0x8b815263,
    0x3ab75638, 0xecbaa8b5, 0xe5781da3, 0xe3e39bde, 0x5e5c2f87, 0x3fcce190, 0x5d8b7fb1, 0x9fac00cd,
    0xbecf85f3, 0x71dc3467, 0xb3d3af55, 0x8a919f5f, 0xc71d4698, 0xd1759cf9, 0x4d085437, 0xd1de5df4,
    0x671b14fe, 0x2a30fcf8, 0xa9d0e908, 0x3772852d, 0x1c1446ea, 0xb46c41a6, 0xf1d3dfc4, 0xdcb95835,
    0xb2bba878, 0xc2085611, 0x2a56fea2, 0x222587a7, 0x135f6063, 0xf454457e, 0xe0b461b4, 0xb13b30dd,
    0x1ef9b39e, 0x6374e50e, 0x36b77691, 0xbc564108, 0xa9ce22bb, 0xdb968d37, 0xb7446d2f, 0x7fec6f84,
    0xd4568935, 0xc9ec2f37, 0xddc0d977, 0xf8780886, 0x73f44571, 0x65769a2b, 0x4599dd52, 0xf0690559,
    0x939e9dd7, 0x339747ba, 0xf9beed4f, 0xa88cbca7, 0x9ed503b9, 0xfccfb8bb, 0x7dca69d7, 0x4510ebee,
    0x40d12cb2, 0xff1b194e, 0x55b76d5d, 0x58567e27, 0x4d9f78d9, 0xd214f54c, 0xdfdaefd0, 0x7c84698f,
    0xcef17616, 0x8e2ab178, 0x452067d9, 0xcfe68aca, 0x9bfecbff, 0xf6a28fbb, 0x2f2fb38e, 0x01abf2ef,
    0xff9ae276, 0x071795c8, 0x1c8e7a27, 0x0698bd01, 0x042f9f88, 0x28f2e270, 0xff6b4c3b, 0x409e74a5,
    0xc81b0ade, 0x003590f1, 0x3604f30e, 0x50c42460, 0x79399846, 0x4564237a, 0x227f93ef, 0x551d3bea,
    0xc762b7be, 0x11695eed, 0xdb961146, 0x8a04bf3d, 0x184d501c, 0x16125ae3, 0x7f85d9c1, 0x8c4adbb4,
    0xec6601e1, 0x411759b9, 0x9beeca35, 0xa32c7f93, 0xddda59fb, 0xde59c1fe, 0xae6abaac, 0x8297860d,
    0x22b62011, 0x4e729806, 0xfd0a3846, 0xcad7e5fb, 0xbf6a944e, 0x7bb69839, 0xde04068d, 0x7fc9b3d3,
    0x8a81b8ed, 0x6ae9b927, 0x8a9f88b5, 0xd597f754, 0x3021d480, 0x87d0d2ba, 0xb0a56854, 0x7653fa68,
    0x6888110c, 0xd0d111bf, 0x24735c91, 0xcbc9b394, 0x50cfb433, 0x6bde9c5d, 0xc80f92c1, 0xa28aa25e,
    0xdb0fa1ef, 0xf0d26a94, 0x1732b7ba, 0xaa75be39,
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
    enc.bitrate_bps = 40000;
    enc.complexity = 5;
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
    0x931086fe, 0xfecdabd4, 0x9ae57237, 0xea5e5222, 0xc3bd9a80, 0x33c4b13f, 0x7462eefb, 0x87df3b91,
    0x7260a5fc, 0x170e9b3a, 0xbb75e1a3, 0x705d73da, 0xbeb28222, 0x00b0bd3a, 0x544846e3, 0xbfd93cf1,
    0x677fb988, 0xb54b1955, 0x1e7bbcd4, 0x17b6052f, 0x933cb81a, 0xd43a0909, 0x9fbbadba, 0xe0e99672,
    0x8c14d354, 0xe8daa661, 0xa0fd9e35, 0x0c425ebe, 0xc3a3cf0d, 0x3d74052b, 0x3e71f40b, 0x03b97c6c,
    0x4c2968d8, 0x520ac751, 0x6e26f6fd, 0x4735c50d, 0x2045c70c, 0xbac7ef99, 0x23b5a82d, 0xd77afe3d,
    0x6f7c2bf0, 0xd1e74cb0, 0x7cc181c9, 0x94bff217, 0x7ce60b08, 0x68eb3b21, 0x1cddd66a, 0x6d66c60c,
    0x3a4e714b, 0x00af8a79, 0xc9b93d0c, 0xc2505ed2, 0x07ced056, 0xf0b19a63, 0x9e1693e7, 0xcf7b4893,
    0x111f457c, 0x9125a176, 0x9a46fc35, 0xffca5e67, 0xc7b96a7c, 0x7e04438e, 0x7fb232ed, 0x8d5f85bb,
    0xb9f47312, 0x53a92278, 0x6bcd2602, 0x7b5da33d, 0xe6c72558, 0x4fda5db2, 0x58ba44e0, 0xf6af57c0,
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
    0xd1f93d49, 0x006cf688, 0xfc66e508, 0x2ca1d6bc, 0xbe591820, 0x9aa78e0b, 0xab2f53b5, 0x721dafb2,
];
