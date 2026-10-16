//! Bitrate and byte budget of each stream of a multistream encoder.
//!
//! This is a port of the arithmetic of libopus 1.6
//! opus_multistream_encoder.c: `surround_rate_allocation`,
//! `ambisonics_rate_allocation`, `rate_allocation`, the bitrate control of
//! `opus_multistream_encoder_ctl`, and the byte budget that
//! `opus_multistream_encode_native` gives each stream.
//!
//! All functions are pure. They work on integers only and do not allocate.
//! Intermediate values are 64 bits wide, so no argument can overflow. Where
//! libopus stays inside 32 bits the results are the same.
//!
//! libopus leaves 32 bits in one place. In `surround_rate_allocation` it
//! multiplies `channel_rate` by 512 for a coupled stream. The product does
//! not fit when one coupled stream is given more than about 8.4 Mb/s, which
//! needs a layout with at least 12 input channels and few coded ones. The
//! functions here return the exact value in that case. The stream encoder
//! limits it to 750 000 bits per second per channel afterwards.

use super::layout::{ChannelLayout, MappingType};

/// libopus `OPUS_AUTO`: the encoder picks the bitrate from the sampling rate
/// and the channel count.
pub const OPUS_AUTO: i32 = crate::OPUS_AUTO;

/// libopus `OPUS_BITRATE_MAX`: use as many bits as the packet can hold.
pub const OPUS_BITRATE_MAX: i32 = crate::OPUS_BITRATE_MAX;

/// libopus `MS_FRAME_TMP`: the largest packet one stream can produce, in
/// bytes. That is six frames of 1275 bytes (120 ms) plus their headers.
pub const MS_FRAME_TMP: usize = 6 * crate::packet::MAX_FRAME_BYTES + 12;

/// The most a stream encoder accepts per channel, in bits per second
/// (libopus `opus_encoder_ctl`, `OPUS_SET_BITRATE_REQUEST`).
pub(crate) const MAX_BITRATE_PER_CHANNEL: i32 = 750_000;

/// Clamps a 64-bit result to the range of `i32`.
#[inline]
fn to_i32(v: i64) -> i32 {
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// A byte or sample count as `i64`. Counts above the `i32` range are clamped;
/// libopus could not represent them at all.
#[inline]
fn count(v: usize) -> i64 {
    v.min(i32::MAX as usize) as i64
}

/// Checks the two arguments that every function divides by.
#[inline]
fn check_frame(frame_size: usize, fs: i32) -> Result<(i64, i64), &'static str> {
    if frame_size == 0 {
        return Err("multistream rate: the frame size must not be 0");
    }
    if fs <= 0 {
        return Err("multistream rate: the sampling rate must be positive");
    }
    Ok((count(frame_size), fs as i64))
}

/// Checks the arguments of `bitrate_to_bits` and `bits_to_bitrate`: on top of
/// `check_frame`, the frame must be shorter than 6 seconds, because both
/// divide by six times the number of frames per second.
#[inline]
fn check_conversion(frame_size: usize, fs: i32) -> Result<(), &'static str> {
    let (frame_size, fs) = check_frame(frame_size, fs)?;
    if 6 * fs / frame_size == 0 {
        return Err("multistream rate: the frame is longer than 6 seconds");
    }
    Ok(())
}

/// libopus `bitrate_to_bits`: bits per second to bits per frame.
pub fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: usize) -> Result<i32, &'static str> {
    check_conversion(frame_size, fs)?;
    Ok(crate::bitrate_to_bits(bitrate, fs, frame_size))
}

/// libopus `bits_to_bitrate`: bits per frame to bits per second.
pub fn bits_to_bitrate(bits: i32, fs: i32, frame_size: usize) -> Result<i32, &'static str> {
    check_conversion(frame_size, fs)?;
    Ok(crate::bits_to_bitrate(bits, fs, frame_size))
}

/// The bitrate control of the multistream encoder (libopus
/// `opus_multistream_encoder_ctl`, `OPUS_SET_BITRATE_REQUEST`).
///
/// `value` is [`OPUS_AUTO`], [`OPUS_BITRATE_MAX`] or bits per second for all
/// channels together. The two special values pass through unchanged. Any
/// other value must be positive and is clamped to 500 to 750 000 bits per
/// second per channel. `channels` is the channel count of the layout; 0 is
/// an error.
pub fn clamp_user_bitrate(value: i32, channels: usize) -> Result<i32, &'static str> {
    if channels == 0 {
        return Err("multistream rate: the channel count must not be 0");
    }
    if value == OPUS_AUTO || value == OPUS_BITRATE_MAX {
        return Ok(value);
    }
    if value <= 0 {
        return Err("multistream rate: the bitrate must be positive");
    }
    let channels = count(channels);
    Ok(to_i32(
        (value as i64)
            .max(500 * channels)
            .min(i64::from(MAX_BITRATE_PER_CHANNEL) * channels),
    ))
}

/// Common argument checks of the allocation functions. Returns the frame
/// size and the sampling rate as `i64`.
fn check_allocation_args(
    layout: &ChannelLayout,
    user_bitrate: i32,
    frame_size: usize,
    fs: i32,
    rate: &[i32],
) -> Result<(i64, i64), &'static str> {
    if user_bitrate <= 0 && user_bitrate != OPUS_AUTO && user_bitrate != OPUS_BITRATE_MAX {
        return Err("multistream rate: the bitrate must be positive");
    }
    if rate.len() < layout.streams() {
        return Err("multistream rate: the rate buffer has fewer entries than streams");
    }
    check_frame(frame_size, fs)
}

/// libopus `surround_rate_allocation`: splits `user_bitrate` between the
/// streams of `layout` and writes the result to `rate[..layout.streams()]`.
///
/// libopus uses this function for every layout that is not ambisonics.
/// A coupled stream gets about twice the rate of a mono stream. The stream of
/// the low-frequency-effects (LFE) channel, if there is one, gets about one
/// eighth of it. With mono streams only and no LFE stream, the bitrate is
/// split evenly.
///
/// `user_bitrate` is [`OPUS_AUTO`], [`OPUS_BITRATE_MAX`] or bits per second.
/// `lfe_stream` is the index of the LFE stream; it must be a mono stream.
/// `frame_size` is in samples per channel at the sampling rate `fs`.
pub fn surround_rate_allocation(
    layout: &ChannelLayout,
    lfe_stream: Option<usize>,
    user_bitrate: i32,
    frame_size: usize,
    fs: i32,
    rate: &mut [i32],
) -> Result<(), &'static str> {
    let (frame_size, fs) = check_allocation_args(layout, user_bitrate, frame_size, fs, rate)?;
    let streams = layout.streams();
    let coupled = layout.coupled_streams();
    if lfe_stream.is_some_and(|lfe| lfe < coupled || lfe >= streams) {
        return Err("multistream rate: the LFE stream must be one of the mono streams");
    }

    let nb_lfe = lfe_stream.is_some() as i64;
    let nb_coupled = coupled as i64;
    let nb_uncoupled = streams as i64 - nb_coupled - nb_lfe;
    let nb_normal = 2 * nb_coupled + nb_uncoupled;
    if nb_normal == 0 {
        return Err("multistream rate: there is no stream besides the LFE stream");
    }

    // Give each non-LFE channel enough bits per channel for coding band energy.
    let channel_offset = 40 * (fs / frame_size).max(50);

    let bitrate = if user_bitrate == OPUS_AUTO {
        nb_normal * (channel_offset + fs + 10000) + 8000 * nb_lfe
    } else if user_bitrate == OPUS_BITRATE_MAX {
        nb_normal * i64::from(MAX_BITRATE_PER_CHANNEL) + nb_lfe * 128000
    } else {
        user_bitrate as i64
    };

    // Give LFE some basic stream_channel allocation but never exceed 1/20 of
    // the total rate for the non-energy part to avoid problems at really low
    // rate.
    let lfe_offset = (bitrate / 20).min(3000) + 15 * (fs / frame_size).max(50);

    // We give each stream (coupled or uncoupled) a starting bitrate. This
    // models the main saving of coupled channels over uncoupled.
    let stream_offset =
        ((bitrate - channel_offset * nb_normal - lfe_offset * nb_lfe) / nb_normal / 2)
            .clamp(0, 20000);

    // Both ratios are in Q8: 256 is the rate of one mono stream.
    // Coupled streams get twice the mono rate after the offset is allocated.
    let coupled_ratio = 512;
    // LFE gets 1/8 the bits of mono.
    let lfe_ratio = 32;

    let total = (nb_uncoupled << 8) + coupled_ratio * nb_coupled + nb_lfe * lfe_ratio;
    let channel_rate = 256
        * (bitrate
            - lfe_offset * nb_lfe
            - stream_offset * (nb_coupled + nb_uncoupled)
            - channel_offset * nb_normal)
        / total;

    for (i, r) in rate[..streams].iter_mut().enumerate() {
        let v = if i < coupled {
            2 * channel_offset + (stream_offset + ((channel_rate * coupled_ratio) >> 8)).max(0)
        } else if Some(i) != lfe_stream {
            channel_offset + (stream_offset + channel_rate).max(0)
        } else {
            (lfe_offset + ((channel_rate * lfe_ratio) >> 8)).max(0)
        };
        *r = to_i32(v);
    }
    Ok(())
}

/// libopus `ambisonics_rate_allocation`: gives every stream of `layout` the
/// same share of `user_bitrate` and writes it to `rate[..layout.streams()]`.
/// The arguments are those of [`surround_rate_allocation`].
pub fn ambisonics_rate_allocation(
    layout: &ChannelLayout,
    user_bitrate: i32,
    frame_size: usize,
    fs: i32,
    rate: &mut [i32],
) -> Result<(), &'static str> {
    let (frame_size, fs) = check_allocation_args(layout, user_bitrate, frame_size, fs, rate)?;
    let nb_streams = layout.streams() as i64;
    let nb_coupled_streams = layout.coupled_streams() as i64;
    let nb_channels = nb_streams + nb_coupled_streams;

    let total_rate = if user_bitrate == OPUS_AUTO {
        (nb_coupled_streams + nb_streams) * (fs + 60 * fs / frame_size) + nb_streams * 15000
    } else if user_bitrate == OPUS_BITRATE_MAX {
        nb_channels * i64::from(MAX_BITRATE_PER_CHANNEL)
    } else {
        user_bitrate as i64
    };

    // Allocate equal number of bits to Ambisonic (uncoupled) and non-diegetic
    // (coupled) streams.
    let per_stream_rate = to_i32(total_rate / nb_streams);
    rate[..layout.streams()].fill(per_stream_rate);
    Ok(())
}

/// libopus `rate_allocation`: the bitrate of each stream for one frame.
///
/// Writes one value per stream to `rate[..layout.streams()]`, each at least
/// 500 bits per second, and returns their sum.
///
/// - `mapping_type` selects the rule: [`ambisonics_rate_allocation`] for
///   ambisonics, [`surround_rate_allocation`] for everything else.
/// - `lfe_stream` is the LFE stream reported by [`ChannelLayout::surround`].
///   It is used only with [`MappingType::Surround`]. libopus has no LFE
///   stream with the other mapping types
///   (`opus_multistream_encoder_init_impl` sets `lfe_stream` to -1 for them),
///   so the value is ignored there.
/// - `user_bitrate` is the encoder setting after [`clamp_user_bitrate`]:
///   [`OPUS_AUTO`], [`OPUS_BITRATE_MAX`] or bits per second.
/// - `frame_size` is in samples per channel at the sampling rate `fs`.
///
/// A stream encoder limits what it is given to 750 000 bits per second per
/// channel. The values written here are not limited that way.
pub fn rate_allocation(
    layout: &ChannelLayout,
    mapping_type: MappingType,
    lfe_stream: Option<usize>,
    user_bitrate: i32,
    frame_size: usize,
    fs: i32,
    rate: &mut [i32],
) -> Result<i32, &'static str> {
    if mapping_type == MappingType::Ambisonics {
        ambisonics_rate_allocation(layout, user_bitrate, frame_size, fs, rate)?;
    } else {
        // Only a surround encoder has an LFE stream.
        let lfe_stream = if mapping_type == MappingType::Surround {
            lfe_stream
        } else {
            None
        };
        surround_rate_allocation(layout, lfe_stream, user_bitrate, frame_size, fs, rate)?;
    }
    let mut rate_sum = 0i64;
    for r in rate[..layout.streams()].iter_mut() {
        *r = (*r).max(500);
        rate_sum += *r as i64;
    }
    Ok(to_i32(rate_sum))
}

/// Whether the frame lasts 100 ms. libopus keeps one more byte per stream
/// for such a frame, because its table-of-contents needs two bytes. The
/// multistream encoder takes no 100 ms frames, but the functions here follow
/// libopus for every frame size.
#[inline]
fn is_100_ms(frame_size: i64, fs: i64) -> bool {
    fs / frame_size == 10
}

/// The smallest packet a multistream encoder can produce, in bytes (libopus
/// `smallest_packet` in `opus_multistream_encode_native`).
///
/// Every stream needs one byte, and every stream but the last needs one more
/// byte for its length. An output buffer below this size is an error.
pub fn smallest_packet(streams: usize, frame_size: usize, fs: i32) -> Result<usize, &'static str> {
    let (frame_size, fs) = check_frame(frame_size, fs)?;
    if streams == 0 {
        return Err("multistream rate: at least one stream is needed");
    }
    let streams = count(streams);
    let mut smallest = streams * 2 - 1;
    // 100 ms needs an extra byte per stream for the ToC.
    if is_100_ms(frame_size, fs) {
        smallest += streams;
    }
    Ok(smallest as usize)
}

/// Constant bitrate: the size limit of the whole packet, in bytes (libopus
/// `opus_multistream_encode_native`, the update of `max_data_bytes` when
/// variable bitrate is off).
///
/// `max_data_bytes` is the length of the output buffer. The result is never
/// larger. Call this only when variable bitrate is off; with variable bitrate
/// the limit is `max_data_bytes` itself.
///
/// - `user_bitrate` is the encoder setting, as for [`rate_allocation`].
/// - `rate_sum` is the value [`rate_allocation`] returned for this frame. It
///   is used when the bitrate is automatic.
/// - With [`OPUS_BITRATE_MAX`] the buffer length is the only limit.
///
/// In constant bitrate mode the encoder pads the last stream, so the packet
/// has exactly this many bytes.
pub fn cbr_max_data_bytes(
    user_bitrate: i32,
    rate_sum: i32,
    streams: usize,
    frame_size: usize,
    fs: i32,
    max_data_bytes: usize,
) -> Result<usize, &'static str> {
    let smallest = smallest_packet(streams, frame_size, fs)? as i64;
    let max_data_bytes = count(max_data_bytes);
    let limit = if user_bitrate == OPUS_AUTO {
        max_data_bytes.min((bitrate_to_bits(rate_sum, fs, frame_size)? as i64 + 4) / 8)
    } else if user_bitrate != OPUS_BITRATE_MAX {
        max_data_bytes
            .min(smallest.max((bitrate_to_bits(user_bitrate, fs, frame_size)? as i64 + 4) / 8))
    } else {
        max_data_bytes
    };
    Ok(limit.max(0) as usize)
}

/// The largest packet stream `stream` may produce, in bytes (libopus
/// `curr_max` in `opus_multistream_encode_native`). It applies with variable
/// and with constant bitrate.
///
/// - `max_data_bytes` is the size limit of the whole packet: the output
///   buffer length, or the result of [`cbr_max_data_bytes`].
/// - `tot_size` is the number of bytes the streams before `stream` already
///   take in the packet, their length fields included.
///
/// The result leaves room for the streams that follow: one byte for the last
/// stream and two for each other one. It also leaves room for the length
/// field of this stream, which every stream but the last has. The length
/// field takes one byte, or two bytes when the stream is longer than 251
/// bytes. Returns an error when not even one byte is left.
///
/// Check the output buffer against [`smallest_packet`] first, as libopus
/// does. This function alone does not catch every buffer that is too small:
/// a 100 ms frame needs 2 bytes per stream.
pub fn stream_max_bytes(
    max_data_bytes: usize,
    tot_size: usize,
    streams: usize,
    stream: usize,
    frame_size: usize,
    fs: i32,
) -> Result<usize, &'static str> {
    let (frame_size, fs) = check_frame(frame_size, fs)?;
    if stream >= streams {
        return Err("multistream rate: no such stream");
    }
    let streams_after = count(streams - stream - 1);
    // Number of bytes left (+Toc).
    let mut curr_max = count(max_data_bytes) - count(tot_size);
    // Reserve one byte for the last stream and two for the others.
    curr_max -= (2 * streams_after - 1).max(0);
    // For 100 ms, reserve an extra byte per stream for the ToC.
    if is_100_ms(frame_size, fs) {
        curr_max -= streams_after;
    }
    curr_max = curr_max.min(MS_FRAME_TMP as i64);
    // The length field of a self-delimited stream takes one or two bytes.
    if streams_after != 0 {
        curr_max -= if curr_max > 253 { 2 } else { 1 };
    }
    if curr_max < 1 {
        return Err("Output buffer too small");
    }
    Ok(curr_max as usize)
}

/// Constant bitrate: the bitrate of the last stream (libopus
/// `bits_to_bitrate(curr_max * 8, Fs, frame_size)` in
/// `opus_multistream_encode_native`).
///
/// `curr_max` is the result of [`stream_max_bytes`] for the last stream. The
/// last stream is set to the bitrate that fills these bytes exactly, so the
/// packet reaches its constant size whatever the streams before it used.
pub fn cbr_last_stream_bitrate(
    curr_max: usize,
    frame_size: usize,
    fs: i32,
) -> Result<i32, &'static str> {
    bits_to_bitrate(to_i32(count(curr_max) * 8), fs, frame_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multistream::layout::SILENT_CHANNEL;

    fn rates(
        channels: usize,
        family: u8,
        user_bitrate: i32,
        frame_size: usize,
        fs: i32,
    ) -> ([i32; 255], usize, i32) {
        let (layout, mapping_type, lfe) = ChannelLayout::surround(channels, family).unwrap();
        let mut rate = [0i32; 255];
        let sum = rate_allocation(
            &layout,
            mapping_type,
            lfe,
            user_bitrate,
            frame_size,
            fs,
            &mut rate,
        )
        .unwrap();
        (rate, layout.streams(), sum)
    }

    #[test]
    fn uncoupled_streams_share_the_bitrate_evenly() {
        // Measured with libopus 1.6.1: 96 kb/s over 3 mono streams.
        let (rate, streams, sum) = rates(3, 255, 96_000, 960, 48_000);
        assert_eq!(&rate[..streams], &[32_000, 32_000, 32_000]);
        assert_eq!(sum, 96_000);
    }

    #[test]
    fn surround_5_1_matches_libopus() {
        // Measured with libopus 1.6.1, 48 kHz, 20 ms frames.
        let (rate, streams, _) = rates(6, 1, 256_000, 960, 48_000);
        assert_eq!(&rate[..streams], &[95_120, 95_120, 57_560, 8_195]);
        let (rate, streams, _) = rates(6, 1, 48_000, 960, 48_000);
        assert_eq!(&rate[..streams], &[17_005, 17_005, 10_245, 3_745]);
        // Automatic bitrate: libopus reports a total of 307 998.
        let (_, _, sum) = rates(6, 1, OPUS_AUTO, 960, 48_000);
        assert_eq!(sum, 307_998);
        // 7.1: libopus reports a total of 427 993.
        let (_, _, sum) = rates(8, 1, OPUS_AUTO, 960, 48_000);
        assert_eq!(sum, 427_993);
    }

    #[test]
    fn ambisonics_gives_every_stream_the_same_rate() {
        let (rate, streams, sum) = rates(6, 2, 100_000, 960, 48_000);
        assert_eq!(streams, 5);
        assert_eq!(&rate[..streams], &[20_000; 5]);
        assert_eq!(sum, 100_000);
        // Automatic: (1 + 5) * (48000 + 60 * 50) + 5 * 15000 = 381 000.
        let (rate, _, sum) = rates(6, 2, OPUS_AUTO, 960, 48_000);
        assert_eq!(rate[0], 76_200);
        assert_eq!(sum, 381_000);
        // Maximum: 6 coded channels at 750 000 each, split over 5 streams.
        let (rate, _, _) = rates(6, 2, OPUS_BITRATE_MAX, 960, 48_000);
        assert_eq!(rate[0], 900_000);
    }

    #[test]
    fn every_stream_gets_at_least_500() {
        for (channels, family) in [(1usize, 1u8), (6, 1), (8, 1), (5, 255), (6, 2)] {
            let (rate, streams, sum) = rates(channels, family, 1, 960, 48_000);
            assert!(
                rate[..streams].iter().all(|&r| r >= 500),
                "{channels} ch family {family}"
            );
            assert_eq!(sum, rate[..streams].iter().sum::<i32>());
        }
        // Ambisonics at 1 b/s: every stream would get 0 and is raised to
        // exactly 500. An encoder never asks for this, because it clamps the
        // bitrate to 500 per channel first.
        let (rate, streams, sum) = rates(6, 2, 1, 960, 48_000);
        assert_eq!(&rate[..streams], &[500; 5]);
        assert_eq!(sum, 2_500);
    }

    #[test]
    fn bad_arguments_are_errors() {
        let (layout, mapping_type, lfe) = ChannelLayout::surround(6, 1).unwrap();
        let mut rate = [0i32; 8];
        let mut call = |lfe, bitrate, frame_size, fs, n: usize| {
            rate_allocation(
                &layout,
                mapping_type,
                lfe,
                bitrate,
                frame_size,
                fs,
                &mut rate[..n],
            )
        };
        assert!(call(lfe, 64_000, 960, 48_000, 8).is_ok());
        assert!(call(lfe, 64_000, 960, 48_000, 4).is_ok());
        assert!(
            call(lfe, 64_000, 960, 48_000, 3).is_err(),
            "rate buffer too short"
        );
        assert!(call(lfe, 64_000, 0, 48_000, 8).is_err(), "frame size 0");
        assert!(call(lfe, 64_000, 960, 0, 8).is_err(), "sampling rate 0");
        assert!(
            call(lfe, 64_000, 960, -48_000, 8).is_err(),
            "negative sampling rate"
        );
        assert!(call(lfe, 0, 960, 48_000, 8).is_err(), "bitrate 0");
        assert!(call(lfe, -2, 960, 48_000, 8).is_err(), "negative bitrate");
        assert!(
            call(Some(0), 64_000, 960, 48_000, 8).is_err(),
            "LFE on a coupled stream"
        );
        assert!(
            call(Some(4), 64_000, 960, 48_000, 8).is_err(),
            "LFE stream out of range"
        );
        assert!(call(None, 64_000, 960, 48_000, 8).is_ok());

        // A layout whose only stream is the LFE stream has nothing to scale.
        let (mono, _, _) = ChannelLayout::surround(1, 1).unwrap();
        let mut one = [0i32; 1];
        assert!(surround_rate_allocation(&mono, Some(0), 64_000, 960, 48_000, &mut one).is_err());
    }

    #[test]
    fn lfe_stream_counts_only_in_surround_mode() {
        let (layout, mapping_type, lfe) = ChannelLayout::surround(6, 1).unwrap();
        assert_eq!((mapping_type, lfe), (MappingType::Surround, Some(3)));
        let call = |mapping_type, lfe| {
            let mut rate = [0i32; 4];
            rate_allocation(&layout, mapping_type, lfe, 256_000, 960, 48_000, &mut rate).unwrap();
            rate
        };
        // Measured with libopus 1.6.1: the 5.1 layout through the surround
        // constructor, and through the generic constructor, which has no LFE
        // stream.
        let surround = [95_120, 95_120, 57_560, 8_195];
        let generic = [78_666, 78_666, 49_333, 49_333];
        assert_eq!(call(MappingType::Surround, lfe), surround);
        assert_eq!(call(MappingType::Surround, None), generic);
        assert_eq!(call(MappingType::None, lfe), generic);
        assert_eq!(call(MappingType::None, None), generic);
        // Ambisonics: an even split, with or without the value.
        assert_eq!(call(MappingType::Ambisonics, lfe), [64_000; 4]);
        assert_eq!(call(MappingType::Ambisonics, None), [64_000; 4]);
        // An LFE stream that does not exist is an error only where it counts.
        let mut rate = [0i32; 4];
        for (mapping_type, ok) in [
            (MappingType::Surround, false),
            (MappingType::None, true),
            (MappingType::Ambisonics, true),
        ] {
            let result = rate_allocation(
                &layout,
                mapping_type,
                Some(9),
                256_000,
                960,
                48_000,
                &mut rate,
            );
            assert_eq!(result.is_ok(), ok, "{mapping_type:?}");
        }
    }

    #[test]
    fn coupled_stream_rate_does_not_wrap_at_32_bits() {
        // 12 input channels, one coupled stream: the encoder accepts up to
        // 9 Mb/s. The product `channel_rate * coupled_ratio` is then above
        // the range of `i32`. The stream must still get the whole bitrate.
        let mut map = [SILENT_CHANNEL; 12];
        map[0] = 0;
        map[1] = 1;
        let layout = ChannelLayout::new(12, 1, 1, &map).unwrap();
        let mut rate = [0i32; 1];
        let sum = rate_allocation(
            &layout,
            MappingType::None,
            None,
            9_000_000,
            960,
            48_000,
            &mut rate,
        )
        .unwrap();
        assert_eq!((rate[0], sum), (9_000_000, 9_000_000));
    }

    /// Deterministic pseudo-random numbers.
    struct Rng(u32);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as usize % n
        }
    }

    #[test]
    fn no_argument_makes_a_function_panic() {
        // Every public function with arguments at and around every limit. A
        // debug build also stops on an arithmetic overflow, so run this test
        // without `--release` after a change to the arithmetic.
        const I32S: [i32; 22] = [
            i32::MIN,
            i32::MIN + 1,
            -1_000_000,
            -1001,
            OPUS_AUTO,
            -999,
            -2,
            OPUS_BITRATE_MAX,
            0,
            1,
            2,
            499,
            500,
            501,
            8_000,
            48_000,
            750_000,
            4_500_000,
            191_250_000,
            357_913_942,
            i32::MAX - 1,
            i32::MAX,
        ];
        const USIZES: [usize; 22] = [
            0,
            1,
            2,
            3,
            4,
            7,
            10,
            119,
            120,
            253,
            254,
            255,
            256,
            960,
            2880,
            4800,
            5760,
            48_000,
            48_001,
            u32::MAX as usize,
            usize::MAX - 1,
            usize::MAX,
        ];
        let identity: [u8; 255] = core::array::from_fn(|i| i as u8);
        let layouts = [
            ChannelLayout::surround(1, 0).unwrap().0,
            ChannelLayout::surround(6, 1).unwrap().0,
            ChannelLayout::surround(8, 1).unwrap().0,
            ChannelLayout::surround(255, 255).unwrap().0,
            ChannelLayout::surround(227, 2).unwrap().0,
            ChannelLayout::surround(3, 2).unwrap().0,
            ChannelLayout::new(255, 128, 127, &identity).unwrap(),
        ];
        let mut rng = Rng(77);
        let mut rate = [0i32; 255];
        let mut accepted = 0usize;
        for _ in 0..200_000 {
            let layout = &layouts[rng.below(layouts.len())];
            let streams = layout.streams();
            let mapping_type = [
                MappingType::None,
                MappingType::Surround,
                MappingType::Ambisonics,
            ][rng.below(3)];
            let lfe = match rng.below(4) {
                0 => None,
                1 => Some(streams - 1),
                2 => Some(USIZES[rng.below(22)]),
                _ => Some(rng.below(256)),
            };
            let n = [0usize, 1, 3, 4, 5, 254, 255][rng.below(7)];
            let (a, b, c) = (
                I32S[rng.below(22)],
                I32S[rng.below(22)],
                I32S[rng.below(22)],
            );
            let (x, y, z, w) = (
                USIZES[rng.below(22)],
                USIZES[rng.below(22)],
                USIZES[rng.below(22)],
                USIZES[rng.below(22)],
            );
            let _ = bitrate_to_bits(a, b, x);
            let _ = bits_to_bitrate(a, b, x);
            if let Ok(v) = clamp_user_bitrate(a, x) {
                assert!(v == a || v > 0, "clamp_user_bitrate({a}, {x}) = {v}");
            }
            let r = &mut rate[..n];
            if let Ok(sum) = rate_allocation(layout, mapping_type, lfe, a, x, b, r) {
                accepted += 1;
                assert!(n >= streams && sum >= 500);
                assert!(r[..streams].iter().all(|&v| v >= 500));
            }
            if surround_rate_allocation(layout, lfe, a, x, b, r).is_ok() {
                assert!(r[..streams].iter().all(|&v| v >= 0));
            }
            if ambisonics_rate_allocation(layout, a, x, b, r).is_ok() {
                assert!(r[..streams].iter().all(|&v| v == r[0]));
            }
            if let Ok(v) = smallest_packet(x, y, a) {
                // A layout has at most 255 streams.
                assert!(x > 255 || (v >= x && v <= 3 * x));
            }
            if let Ok(v) = cbr_max_data_bytes(a, b, x, y, c, z) {
                assert!(v <= z);
            }
            if let Ok(v) = stream_max_bytes(x, y, z, w, USIZES[rng.below(22)], a) {
                accepted += 1;
                assert!((1..=MS_FRAME_TMP).contains(&v) && v <= x);
            }
            let _ = cbr_last_stream_bitrate(x, y, a);
        }
        // Enough of the calls passed the argument checks to reach the arithmetic.
        assert!(accepted > 10_000, "{accepted}");
    }

    #[test]
    fn extreme_arguments_do_not_overflow() {
        // 255 input channels and one coupled stream: the bitrate limit of the
        // encoder is 255 * 750 000, far above what the stream can use.
        let mut map = [SILENT_CHANNEL; 255];
        map[0] = 0;
        map[1] = 1;
        let layout = ChannelLayout::new(255, 1, 1, &map).unwrap();
        let mut rate = [0i32; 1];
        for bitrate in [1, 500, 191_250_000, i32::MAX, OPUS_AUTO, OPUS_BITRATE_MAX] {
            for (frame_size, fs) in [
                (1usize, i32::MAX),
                (usize::MAX, 1),
                (120, 48_000),
                (5760, 48_000),
            ] {
                for mapping_type in [
                    MappingType::None,
                    MappingType::Surround,
                    MappingType::Ambisonics,
                ] {
                    let sum = rate_allocation(
                        &layout,
                        mapping_type,
                        None,
                        bitrate,
                        frame_size,
                        fs,
                        &mut rate,
                    )
                    .unwrap();
                    assert!(sum >= 500);
                    assert_eq!(sum, rate[0]);
                }
            }
        }
        // 255 streams at the maximum bitrate.
        let (layout, mapping_type, lfe) = ChannelLayout::surround(255, 255).unwrap();
        let mut rate = [0i32; 255];
        let sum = rate_allocation(
            &layout,
            mapping_type,
            lfe,
            OPUS_BITRATE_MAX,
            960,
            48_000,
            &mut rate,
        )
        .unwrap();
        assert_eq!(sum, 255 * 750_000);
    }

    #[test]
    fn user_bitrate_clamp() {
        assert_eq!(clamp_user_bitrate(OPUS_AUTO, 6), Ok(OPUS_AUTO));
        assert_eq!(
            clamp_user_bitrate(OPUS_BITRATE_MAX, 6),
            Ok(OPUS_BITRATE_MAX)
        );
        assert!(clamp_user_bitrate(0, 6).is_err());
        assert!(clamp_user_bitrate(-2, 6).is_err());
        assert_eq!(clamp_user_bitrate(1, 6), Ok(3_000));
        assert_eq!(clamp_user_bitrate(256_000, 6), Ok(256_000));
        assert_eq!(clamp_user_bitrate(i32::MAX, 6), Ok(4_500_000));
        assert_eq!(clamp_user_bitrate(i32::MAX, 255), Ok(191_250_000));
        assert_eq!(clamp_user_bitrate(i32::MAX, usize::MAX), Ok(i32::MAX));
        // Without channels the limits would both be 0, which is no bitrate.
        assert!(clamp_user_bitrate(64_000, 0).is_err());
        assert!(clamp_user_bitrate(OPUS_AUTO, 0).is_err());
    }

    #[test]
    fn bits_and_bitrate_conversions() {
        // 20 ms at 48 kHz: 50 frames per second.
        assert_eq!(bitrate_to_bits(64_000, 48_000, 960), Ok(1280));
        assert_eq!(bits_to_bitrate(1280, 48_000, 960), Ok(64_000));
        // 60 ms: 16.67 frames per second, computed as 100 / 6.
        assert_eq!(bitrate_to_bits(64_000, 48_000, 2880), Ok(3840));
        assert_eq!(bits_to_bitrate(3840, 48_000, 2880), Ok(64_000));
        // 2.5 ms.
        assert_eq!(bitrate_to_bits(64_000, 48_000, 120), Ok(160));
        assert!(bitrate_to_bits(64_000, 48_000, 0).is_err());
        assert!(bitrate_to_bits(64_000, 0, 960).is_err());
        assert!(bits_to_bitrate(1280, 8_000, 48_001).is_err());
        assert_eq!(bits_to_bitrate(i32::MAX, i32::MAX, 1), Ok(i32::MAX));
    }

    #[test]
    fn smallest_packet_sizes() {
        assert_eq!(smallest_packet(1, 960, 48_000), Ok(1));
        assert_eq!(smallest_packet(4, 960, 48_000), Ok(7));
        // 100 ms: one more byte per stream.
        assert_eq!(smallest_packet(4, 4800, 48_000), Ok(11));
        assert!(smallest_packet(0, 960, 48_000).is_err());
        assert!(smallest_packet(4, 0, 48_000).is_err());
    }

    #[test]
    fn cbr_packet_size() {
        // Measured with libopus 1.6.1: 5.1 at 256 kb/s, 20 ms, gives packets
        // of 640 bytes; 96 kb/s gives 240 bytes.
        assert_eq!(
            cbr_max_data_bytes(256_000, 0, 4, 960, 48_000, 4000),
            Ok(640)
        );
        assert_eq!(cbr_max_data_bytes(96_000, 0, 4, 960, 48_000, 4000), Ok(240));
        // The output buffer is the limit when it is smaller.
        assert_eq!(cbr_max_data_bytes(256_000, 0, 4, 960, 48_000, 100), Ok(100));
        // A very low bitrate still gives the smallest packet.
        assert_eq!(cbr_max_data_bytes(2_000, 0, 4, 960, 48_000, 4000), Ok(7));
        // Automatic: from the sum of the stream rates.
        assert_eq!(
            cbr_max_data_bytes(OPUS_AUTO, 307_998, 4, 960, 48_000, 4000),
            Ok(770)
        );
        // Maximum: only the buffer limits the packet.
        assert_eq!(
            cbr_max_data_bytes(OPUS_BITRATE_MAX, 0, 4, 960, 48_000, 4000),
            Ok(4000)
        );
    }

    #[test]
    fn stream_budget_reserves_room_for_later_streams() {
        // 4 streams, 640 bytes. Stream 0: 640 - 5 reserved - 2 length bytes.
        assert_eq!(stream_max_bytes(640, 0, 4, 0, 960, 48_000), Ok(633));
        // Stream 1 after 240 bytes: 400 - 3 reserved - 2 length bytes.
        assert_eq!(stream_max_bytes(640, 240, 4, 1, 960, 48_000), Ok(395));
        // Stream 2 after 480 bytes: 160 - 1 reserved - 1 length byte.
        assert_eq!(stream_max_bytes(640, 480, 4, 2, 960, 48_000), Ok(158));
        // The last stream takes all that is left and has no length field.
        assert_eq!(stream_max_bytes(640, 623, 4, 3, 960, 48_000), Ok(17));
        // 254 bytes left for a stream that is not the last: 2 length bytes.
        assert_eq!(stream_max_bytes(255, 0, 2, 0, 960, 48_000), Ok(252));
        // 253 bytes left: 1 length byte.
        assert_eq!(stream_max_bytes(254, 0, 2, 0, 960, 48_000), Ok(252));
        // One stream can never use more than MS_FRAME_TMP bytes.
        assert_eq!(
            stream_max_bytes(100_000, 0, 1, 0, 960, 48_000),
            Ok(MS_FRAME_TMP)
        );
        assert_eq!(
            stream_max_bytes(100_000, 0, 2, 0, 960, 48_000),
            Ok(MS_FRAME_TMP - 2)
        );
        // The smallest packet gives every stream one byte.
        assert_eq!(stream_max_bytes(7, 0, 4, 0, 960, 48_000), Ok(1));
        assert_eq!(stream_max_bytes(7, 2, 4, 1, 960, 48_000), Ok(1));
        assert_eq!(stream_max_bytes(7, 4, 4, 2, 960, 48_000), Ok(1));
        assert_eq!(stream_max_bytes(7, 6, 4, 3, 960, 48_000), Ok(1));
        // 100 ms: one more byte reserved for each later stream.
        assert_eq!(stream_max_bytes(640, 0, 4, 0, 4800, 48_000), Ok(630));
        // Errors instead of a wrap-around.
        assert!(stream_max_bytes(6, 0, 4, 0, 960, 48_000).is_err());
        assert!(stream_max_bytes(640, 641, 4, 3, 960, 48_000).is_err());
        assert!(stream_max_bytes(640, usize::MAX, 4, 3, 960, 48_000).is_err());
        assert!(stream_max_bytes(640, 0, 4, 4, 960, 48_000).is_err());
        assert!(stream_max_bytes(640, 0, 0, 0, 960, 48_000).is_err());
        assert!(stream_max_bytes(640, 0, 4, 0, 0, 48_000).is_err());
    }

    #[test]
    fn cbr_last_stream_rate() {
        // 17 bytes per 20 ms frame are 6800 bits per second.
        assert_eq!(cbr_last_stream_bitrate(17, 960, 48_000), Ok(6_800));
        assert_eq!(
            cbr_last_stream_bitrate(usize::MAX, 960, 48_000),
            Ok(i32::MAX)
        );
        assert!(cbr_last_stream_bitrate(17, 0, 48_000).is_err());
    }
}
