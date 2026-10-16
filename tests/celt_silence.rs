use opus1::internals::celt::{CeltDecoder, PcmLayout};
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;

#[test]
fn test_celt_silence() {
    let mode = default_mode();
    let frame_size = 960;
    let channels = 1;

    // Create a proper silence packet
    // In CELT, silence is indicated by the first bit being 1 with logp=15
    let buf_size = 100u32;
    let mut rc_enc = RangeCoder::new_encoder(buf_size);
    // Encode silence bit
    rc_enc.encode_bit_logp(true, 15);
    rc_enc.done();

    // Use the full buffer to preserve the dual-stream layout
    let silence_packet = rc_enc.buf[..buf_size as usize].to_vec();

    println!(
        "Silence packet: {:?}, len={}",
        silence_packet,
        silence_packet.len()
    );

    // Decode silence packet
    let mut decoder = CeltDecoder::new(mode, channels, 48000);
    let mut output = vec![0.0f32; frame_size];
    let bits = (silence_packet.len() * 8) as i32;
    let mut rc_dec = RangeCoder::new_decoder(&silence_packet);
    decoder.decode(&mut rc_dec, bits, frame_size, &mut output, PcmLayout::Planar);

    let energy: f32 = output.iter().map(|&x| x * x).sum();
    println!("Decoded silence energy: {}", energy);

    // Should be very close to zero
    assert!(energy < 1.0, "Silence energy too high: {}", energy);
}
