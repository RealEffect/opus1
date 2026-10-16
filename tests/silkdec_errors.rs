//! `SilkDecoder::decode_with_control` with bad arguments and bad data.
//!
//! Bad arguments return an error and leave the decoder as it was. Data that
//! no encoder would write decodes to something, without a panic.

use opus1::range_coder::RangeCoder;
use opus1::silk::dec_api::SilkDecoder;
use opus1::silk::decode_frame::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST};
use opus1::silk::decoder_structs::SilkDecControl;

mod silkdec_common;
use silkdec_common::*;

fn control(api_channels: i32, packet_channels: i32, api_rate: i32, silk_rate: i32, ms: i32) -> SilkDecControl {
    SilkDecControl {
        n_channels_api: api_channels,
        n_channels_internal: packet_channels,
        api_sample_rate: api_rate,
        internal_sample_rate: silk_rate,
        payload_size_ms: ms,
    }
}

/// Every kind of bad argument, tried before each good frame of a stream.
/// The good frames must still decode exactly as in C libopus, which shows
/// that a rejected call changes nothing.
#[test]
fn bad_arguments_are_rejected_and_change_nothing() {
    let pcm = stereo_from(&speech(0.5, 2.0), &speech(7.0, 2.0));
    let mut enc = CEnc::silk(2, BW_WB, 60000);
    enc.fec(20).force_channels(2);
    let packets = enc.encode_all(&pcm, 40);

    let (rate, channels) = (48000usize, 2usize);
    let mut c = CDec::new(rate, channels);
    let mut dec = SilkDecoder::new();
    let good = control(2, 2, 48000, 16000, 40);
    let bad = [
        ("0 output channels", control(0, 2, 48000, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("3 output channels", control(3, 2, 48000, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("0 packet channels", control(2, 0, 48000, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("3 packet channels", control(2, 3, 48000, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("output rate 44100", control(2, 2, 44100, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("output rate 0", control(2, 2, 0, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("output rate -8000", control(2, 2, -8000, 16000, 40), FLAG_DECODE_NORMAL, 4000),
        ("SILK rate 24000", control(2, 2, 48000, 24000, 40), FLAG_DECODE_NORMAL, 4000),
        ("SILK rate 0", control(2, 2, 48000, 0, 40), FLAG_DECODE_NORMAL, 4000),
        ("duration 30 ms", control(2, 2, 48000, 16000, 30), FLAG_DECODE_NORMAL, 4000),
        ("duration -20 ms", control(2, 2, 48000, 16000, -20), FLAG_DECODE_NORMAL, 4000),
        ("lost flag 3", good, 3, 4000),
        ("lost flag -1", good, -1, 4000),
        ("output one sample short", good, FLAG_DECODE_NORMAL, 2 * 960 - 1),
        ("empty output", good, FLAG_PACKET_LOST, 0),
    ];
    let mut diff = Diff::default();
    let mut out = vec![0i16; 4000];
    for (i, p) in packets.iter().enumerate() {
        let payload = frames(p)[0];
        for (what, ctl, flag, out_len) in &bad {
            for new_packet in [true, false] {
                let mut rc = RangeCoder::new_decoder(payload);
                let r = dec.decode_with_control(ctl, &mut rc, &mut out[..*out_len], *flag, new_packet);
                assert!(r.is_err(), "packet {i}: {what} was accepted: {r:?}");
            }
        }
        // The frame itself: two SILK frames of 20 ms.
        let reference = c.decode(Some(p), 1920, false);
        let mut rc = RangeCoder::new_decoder(payload);
        let mut ours = vec![0i16; 2 * 1920];
        for k in 0..2 {
            let n = dec
                .decode_with_control(&good, &mut rc, &mut ours[k * 1920..], FLAG_DECODE_NORMAL, k == 0)
                .unwrap();
            assert_eq!(n, 960);
        }
        diff.add(&reference, &ours, || format!("packet {i}"));
        // The packet holds two SILK frames. A third one is refused, with
        // and without the FEC flag.
        for flag in [FLAG_DECODE_NORMAL, FLAG_DECODE_LBRR] {
            let r = dec.decode_with_control(&good, &mut rc, &mut out, flag, false);
            assert!(r.is_err(), "packet {i}: a third SILK frame was accepted: {r:?}");
        }
    }
    diff.assert_identical("good frames between rejected calls");
}

/// The older entry point reports errors as a negative value.
#[test]
fn decode_rejects_bad_arguments() {
    let mut out = [0i16; 640];
    let mut rc = RangeCoder::new_decoder(&[]);
    for (rate, channels, ms, silk_rate, flag, out_len, what) in [
        (16000, 1, 30, 16000, FLAG_PACKET_LOST, 640, "duration 30 ms"),
        (16000, 1, 20, 24000, FLAG_PACKET_LOST, 640, "SILK rate 24000"),
        (16000, 1, 20, 16000, 7, 640, "lost flag 7"),
        (16000, 2, 20, 16000, FLAG_PACKET_LOST, 639, "output one sample short"),
        (16000, 3, 20, 16000, FLAG_PACKET_LOST, 640, "3 channels"),
        (16000, 0, 20, 16000, FLAG_PACKET_LOST, 640, "0 channels"),
    ] {
        let mut dec = SilkDecoder::new();
        dec.init(rate, channels);
        let n = dec.decode(&mut rc, &mut out[..out_len], flag, true, ms, silk_rate);
        assert!(n < 0, "{what} was accepted: {n}");
    }
    // `decode` also works on a decoder that `init()` never saw.
    let mut dec = SilkDecoder::new();
    assert_eq!(dec.decode(&mut rc, &mut out, FLAG_PACKET_LOST, true, 20, 16000), 320);
    assert_eq!(dec.decode(&mut rc, &mut out, FLAG_PACKET_LOST, true, 10, 8000), 80);
    // More frames than the packet holds.
    let mut dec = SilkDecoder::new();
    assert_eq!(dec.init(16000, 1), 0);
    let packet = CEnc::silk(1, BW_WB, 24000).encode(&speech(0.5, 0.02));
    let mut rc = RangeCoder::new_decoder(frames(&packet)[0]);
    assert_eq!(dec.decode(&mut rc, &mut out, FLAG_DECODE_NORMAL, true, 20, 16000), 320);
    assert!(dec.decode(&mut rc, &mut out, FLAG_DECODE_NORMAL, false, 20, 16000) < 0);
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// Random bytes as SILK payloads, with the channel counts, rates, packet
/// duration and lost flag changing at random from call to call, including
/// in the middle of a packet. Nothing may panic. A call that succeeds
/// returns one SILK frame at the output rate.
#[test]
fn random_data_and_random_call_sequences_do_not_panic() {
    let mut rng = Lcg(2024);
    let rates = [8000, 12000, 16000, 24000, 48000];
    let mut out = vec![0i16; 2 * 960];
    let (mut ok, mut rejected) = (0usize, 0usize);
    for _ in 0..300 {
        let mut dec = SilkDecoder::new();
        // Some decoders start the way OpusDecoder sets its SILK decoder up.
        if rng.below(3) == 0 {
            dec.init(rates[rng.below(3) as usize], 1 + rng.below(2) as i32);
        }
        let mut payload = vec![0u8; 0];
        let mut ctl = control(1, 1, 16000, 16000, 20);
        for _ in 0..80 {
            let new_packet = rng.below(4) != 0;
            if new_packet {
                payload = (0..rng.below(120)).map(|_| rng.next() as u8).collect();
            }
            // Keep the settings for most calls, as a real caller does.
            if rng.below(3) == 0 {
                ctl = control(
                    1 + rng.below(2) as i32,
                    1 + rng.below(2) as i32,
                    rates[rng.below(5) as usize],
                    rates[rng.below(3) as usize],
                    [0, 10, 20, 40, 60][rng.below(5) as usize],
                );
            }
            if rng.below(40) == 0 {
                dec.reset();
            }
            let flag = [FLAG_DECODE_NORMAL, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, FLAG_DECODE_LBRR]
                [rng.below(4) as usize];
            let mut rc = RangeCoder::new_decoder(&payload);
            // Several SILK frames from one range decoder, as for a 40 or
            // 60 ms packet, and sometimes more than the packet holds.
            for k in 0..1 + rng.below(4) {
                match dec.decode_with_control(&ctl, &mut rc, &mut out, flag, new_packet && k == 0) {
                    Ok(n) => {
                        let ms = n * 1000 / ctl.api_sample_rate as usize;
                        assert!(ms == 10 || ms == 20, "{n} samples at {} Hz", ctl.api_sample_rate);
                        ok += 1;
                    }
                    Err(_) => rejected += 1,
                }
            }
        }
    }
    println!("{ok} calls decoded, {rejected} rejected");
    assert!(ok > 10000 && rejected > 100);
}
