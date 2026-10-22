//! Channel layout of a multistream Opus stream.
//!
//! A multistream packet holds `streams` ordinary Opus streams. The first
//! `coupled_streams` of them are stereo ("coupled"), the others are mono.
//! Decoding them in order gives `streams + coupled_streams` decoded channels:
//!
//! - decoded channels `2 * s` and `2 * s + 1` are the left and right channel
//!   of coupled stream `s`;
//! - decoded channel `s + coupled_streams` is mono stream `s`.
//!
//! `mapping[j]` is the decoded channel that channel `j` of the application
//! uses, or [`SILENT_CHANNEL`].
//!
//! This is a port of the layout code of libopus 1.6: `validate_layout`,
//! `get_left_channel`, `get_right_channel` and `get_mono_channel`
//! (opus_multistream.c), and `vorbis_mappings`, `validate_ambisonics`,
//! `validate_encoder_layout` and `opus_multistream_surround_encoder_init`
//! (opus_multistream_encoder.c).
//!
//! Nothing here allocates: a layout stores its mapping in a fixed array.

use crate::Error;

/// Mapping value that means "this output channel is silent" for a decoder and
/// "this input channel is not encoded" for an encoder.
pub const SILENT_CHANNEL: u8 = 255;

/// Largest channel count of a layout. It is also the largest value of
/// `streams + coupled_streams`.
pub const MAX_CHANNELS: usize = 255;

/// How a layout was chosen. libopus `MappingType` (opus_private.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingType {
    /// No special handling: the generic constructor, and mapping families 0
    /// and 255. Mapping family 1 with 1 or 2 channels is also this type.
    None,
    /// Mapping family 1 with more than 2 channels.
    Surround,
    /// Mapping family 2.
    Ambisonics,
}

/// libopus `ChannelLayout` (opus_private.h). A value of this type has passed
/// the checks of [`ChannelLayout::new`], so it is always valid for a decoder.
#[derive(Clone, PartialEq, Eq)]
pub struct ChannelLayout {
    nb_channels: u8,
    nb_streams: u8,
    nb_coupled_streams: u8,
    /// Only the first `nb_channels` entries are used. The others stay 0, so
    /// the derived `==` compares two layouts correctly.
    mapping: [u8; MAX_CHANNELS],
}

/// libopus `vorbis_mappings`: the layouts of mapping family 1. The index is
/// the channel count minus 1. Each row holds the number of streams, the
/// number of coupled streams, and the mapping; only the first entries of the
/// mapping, one per channel, are used.
static VORBIS_MAPPINGS: [(u8, u8, [u8; 8]); 8] = [
    (1, 0, [0, 0, 0, 0, 0, 0, 0, 0]), // 1: mono
    (1, 1, [0, 1, 0, 0, 0, 0, 0, 0]), // 2: stereo
    (2, 1, [0, 2, 1, 0, 0, 0, 0, 0]), // 3: 1-d surround
    (2, 2, [0, 1, 2, 3, 0, 0, 0, 0]), // 4: quadraphonic surround
    (3, 2, [0, 4, 1, 2, 3, 0, 0, 0]), // 5: 5-channel surround
    (4, 2, [0, 4, 1, 2, 3, 5, 0, 0]), // 6: 5.1 surround
    (4, 3, [0, 4, 1, 2, 3, 5, 6, 0]), // 7: 6.1 surround
    (5, 3, [0, 6, 1, 2, 3, 4, 5, 7]), // 8: 7.1 surround
];

/// libopus `validate_ambisonics`: the stream counts of an ambisonics layout
/// (mapping family 2) with `channels` channels.
///
/// Returns `(streams, coupled_streams)`, or `None` when `channels` is not a
/// valid ambisonics channel count. A valid count is a square number
/// (1, 4, 9, ...), or a square number plus 2 for one extra stereo stream, and
/// is at most 227.
pub fn validate_ambisonics(channels: usize) -> Option<(usize, usize)> {
    if !(1..=227).contains(&channels) {
        return None;
    }
    // libopus calls isqrt32 here. For 1..=227 this loop gives the same
    // result: the largest number whose square is not above `channels`.
    let mut order_plus_one = 1;
    while (order_plus_one + 1) * (order_plus_one + 1) <= channels {
        order_plus_one += 1;
    }
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic_channels = channels - acn_channels;
    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }
    let coupled_streams = (nondiegetic_channels != 0) as usize;
    Some((acn_channels + coupled_streams, coupled_streams))
}

impl ChannelLayout {
    /// Builds a layout and checks it the way a decoder does: the argument
    /// checks of libopus `opus_multistream_decoder_init`, then
    /// `validate_layout`.
    ///
    /// `mapping` must hold at least `channels` entries; the rest is ignored.
    /// Each entry is a decoded channel index below
    /// `streams + coupled_streams`, or [`SILENT_CHANNEL`].
    pub fn new(
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
    ) -> Result<Self, Error> {
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(Error::bad_arg("multistream layout: the channel count must be 1 to 255"));
        }
        if coupled_streams > streams {
            return Err(Error::bad_arg("multistream layout: more coupled streams than streams"));
        }
        if streams < 1 {
            return Err(Error::bad_arg("multistream layout: at least one stream is needed"));
        }
        // The sum saturates, so huge counts are an error and do not wrap.
        let max_channel = streams.saturating_add(coupled_streams);
        if max_channel > MAX_CHANNELS {
            return Err(Error::bad_arg(
                "multistream layout: streams plus coupled streams must be at most 255",
            ));
        }
        if mapping.len() < channels {
            return Err(Error::bad_arg(
                "multistream layout: the mapping has fewer entries than channels",
            ));
        }
        // validate_layout.
        let mapping = &mapping[..channels];
        if mapping
            .iter()
            .any(|&m| m as usize >= max_channel && m != SILENT_CHANNEL)
        {
            return Err(Error::bad_arg(
                "multistream layout: a mapping entry names a stream channel that does not exist",
            ));
        }
        let mut stored = [0u8; MAX_CHANNELS];
        stored[..channels].copy_from_slice(mapping);
        Ok(Self {
            nb_channels: channels as u8,
            nb_streams: streams as u8,
            nb_coupled_streams: coupled_streams as u8,
            mapping: stored,
        })
    }

    /// The extra checks an encoder needs, from libopus
    /// `opus_multistream_encoder_init_impl` and `validate_encoder_layout`:
    /// there are not more coded channels than input channels, and every
    /// stream gets its input from some channel.
    pub fn validate_for_encoder(&self) -> Result<(), Error> {
        if self.streams() + self.coupled_streams() > self.channels() {
            return Err(Error::bad_arg(
                "multistream layout: an encoder cannot code more channels than it is given",
            ));
        }
        for s in 0..self.streams() {
            if s < self.coupled_streams() {
                if self.left_channel(s, None).is_none() {
                    return Err(Error::bad_arg(
                        "multistream layout: a coupled stream has no left input channel",
                    ));
                }
                if self.right_channel(s, None).is_none() {
                    return Err(Error::bad_arg(
                        "multistream layout: a coupled stream has no right input channel",
                    ));
                }
            } else if self.mono_channel(s, None).is_none() {
                return Err(Error::bad_arg(
                    "multistream layout: a mono stream has no input channel",
                ));
            }
        }
        Ok(())
    }

    /// The layout that libopus `opus_multistream_surround_encoder_init` picks
    /// for a channel count and a mapping family of the Ogg Opus header.
    ///
    /// - Family 0: 1 or 2 channels, one stream.
    /// - Family 1: 1 to 8 channels in Vorbis channel order.
    /// - Family 255: 1 to 255 channels, one mono stream per channel.
    /// - Family 2: ambisonics with one mono stream per ambisonics channel,
    ///   plus one stereo stream when there are 2 extra channels.
    ///
    /// Returns the layout, the mapping type, and the index of the stream
    /// that carries the low-frequency-effects (LFE) channel if there is one.
    /// Only family 1 with 6, 7 or 8 channels has an LFE stream; it is the
    /// last stream.
    pub fn surround(
        channels: usize,
        mapping_family: u8,
    ) -> Result<(Self, MappingType, Option<usize>), Error> {
        if !(1..=MAX_CHANNELS).contains(&channels) {
            return Err(Error::bad_arg("multistream layout: the channel count must be 1 to 255"));
        }
        let mut mapping = [0u8; MAX_CHANNELS];
        let mut lfe_stream = None;
        let (streams, coupled_streams) = match mapping_family {
            0 => match channels {
                1 => {
                    mapping[0] = 0;
                    (1, 0)
                }
                2 => {
                    mapping[0] = 0;
                    mapping[1] = 1;
                    (1, 1)
                }
                _ => {
                    return Err(Error::bad_arg(
                        "multistream layout: mapping family 0 has 1 or 2 channels",
                    ));
                }
            },
            1 if channels <= 8 => {
                let (nb_streams, nb_coupled_streams, table) = VORBIS_MAPPINGS[channels - 1];
                mapping[..channels].copy_from_slice(&table[..channels]);
                if channels >= 6 {
                    lfe_stream = Some(nb_streams as usize - 1);
                }
                (nb_streams as usize, nb_coupled_streams as usize)
            }
            255 => {
                for (i, m) in mapping[..channels].iter_mut().enumerate() {
                    *m = i as u8;
                }
                (channels, 0)
            }
            2 => {
                let (streams, coupled_streams) = validate_ambisonics(channels)
                    .ok_or(Error::bad_arg("multistream layout: not an ambisonics channel count"))?;
                // The mono streams come first in channel order. The stereo
                // stream, which is stream 0, takes the last two channels.
                for i in 0..streams - coupled_streams {
                    mapping[i] = (i + coupled_streams * 2) as u8;
                }
                for i in 0..coupled_streams * 2 {
                    mapping[i + streams - coupled_streams] = i as u8;
                }
                (streams, coupled_streams)
            }
            _ => {
                return Err(Error::bad_arg(
                    "multistream layout: unsupported mapping family or channel count",
                ));
            }
        };
        let mapping_type = if channels > 2 && mapping_family == 1 {
            MappingType::Surround
        } else if mapping_family == 2 {
            MappingType::Ambisonics
        } else {
            MappingType::None
        };
        // libopus runs the generic encoder checks on the layout it built.
        let layout = Self::new(channels, streams, coupled_streams, &mapping[..channels])?;
        layout.validate_for_encoder()?;
        Ok((layout, mapping_type, lfe_stream))
    }

    /// Number of channels the application sees.
    #[inline]
    pub fn channels(&self) -> usize {
        self.nb_channels as usize
    }

    /// Number of Opus streams in one packet.
    #[inline]
    pub fn streams(&self) -> usize {
        self.nb_streams as usize
    }

    /// Number of stereo streams. They are the first streams of a packet.
    #[inline]
    pub fn coupled_streams(&self) -> usize {
        self.nb_coupled_streams as usize
    }

    /// The mapping table. Its length is [`ChannelLayout::channels`].
    #[inline]
    pub fn mapping(&self) -> &[u8] {
        &self.mapping[..self.nb_channels as usize]
    }

    /// First channel after `prev` whose mapping entry is `index`.
    #[inline]
    fn find_channel(&self, index: usize, prev: Option<usize>) -> Option<usize> {
        let start = match prev {
            None => 0,
            Some(p) => p.checked_add(1)?,
        };
        let mapping = self.mapping();
        mapping
            .get(start..)?
            .iter()
            .position(|&m| m as usize == index)
            .map(|i| i + start)
    }

    /// libopus `get_left_channel`: a channel that is the left channel of
    /// coupled stream `stream`.
    ///
    /// `prev` is `None` to get the first such channel, or the previous result
    /// to get the next one. An encoder reads its input from the first one. A
    /// decoder writes its output to all of them. Returns `None` when there is
    /// no further channel, or when `stream` is not a coupled stream.
    #[inline]
    pub fn left_channel(&self, stream: usize, prev: Option<usize>) -> Option<usize> {
        if stream >= self.coupled_streams() {
            return None;
        }
        self.find_channel(stream * 2, prev)
    }

    /// libopus `get_right_channel`: like [`ChannelLayout::left_channel`], for
    /// the right channel of coupled stream `stream`.
    #[inline]
    pub fn right_channel(&self, stream: usize, prev: Option<usize>) -> Option<usize> {
        if stream >= self.coupled_streams() {
            return None;
        }
        self.find_channel(stream * 2 + 1, prev)
    }

    /// libopus `get_mono_channel`: like [`ChannelLayout::left_channel`], for
    /// the single channel of mono stream `stream`. `stream` counts all
    /// streams, so the first mono stream has the index `coupled_streams()`.
    /// Returns `None` when `stream` is not a mono stream.
    #[inline]
    pub fn mono_channel(&self, stream: usize, prev: Option<usize>) -> Option<usize> {
        if stream < self.coupled_streams() || stream >= self.streams() {
            return None;
        }
        self.find_channel(stream + self.coupled_streams(), prev)
    }
}

/// The channels of a layout grouped by mapping entry. A decoder looks up the
/// output channels of a decoded channel here instead of searching the
/// mapping on every frame.
#[derive(Clone)]
pub(crate) struct ChannelsByEntry {
    /// The channel indices sorted by mapping entry. Channels with the same
    /// entry are in increasing order.
    channels: [u8; MAX_CHANNELS],
    /// The channels with mapping entry `m` are `channels[start[m]..start[m + 1]]`.
    start: [u8; 257],
}

impl ChannelsByEntry {
    pub(crate) fn new(layout: &ChannelLayout) -> Self {
        let mapping = layout.mapping();
        // Count the channels of each entry, then turn the counts into the
        // start of each group. A layout has at most 255 channels, so every
        // value fits in a byte.
        let mut start = [0u8; 257];
        for &m in mapping {
            start[m as usize + 1] += 1;
        }
        for m in 0..256 {
            start[m + 1] += start[m];
        }
        let mut next = start;
        let mut channels = [0u8; MAX_CHANNELS];
        for (c, &m) in mapping.iter().enumerate() {
            channels[next[m as usize] as usize] = c as u8;
            next[m as usize] += 1;
        }
        Self { channels, start }
    }

    /// The channels whose mapping entry is `entry`, in increasing order.
    /// [`SILENT_CHANNEL`] gives the silent channels.
    #[inline]
    pub(crate) fn with_entry(&self, entry: usize) -> &[u8] {
        match self.start.get(entry..).and_then(|s| s.get(..2)) {
            Some(&[first, end]) => &self.channels[first as usize..end as usize],
            _ => &[],
        }
    }
}

impl core::fmt::Debug for ChannelLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ChannelLayout")
            .field("channels", &self.nb_channels)
            .field("streams", &self.nb_streams)
            .field("coupled_streams", &self.nb_coupled_streams)
            .field("mapping", &self.mapping())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surround(channels: usize, family: u8) -> (ChannelLayout, MappingType, Option<usize>) {
        ChannelLayout::surround(channels, family).unwrap()
    }

    #[test]
    fn family_1_table_matches_vorbis_mappings() {
        // (channels, streams, coupled, mapping, LFE stream)
        #[allow(clippy::type_complexity)]
        let rows: [(usize, usize, usize, &[u8], Option<usize>); 8] = [
            (1, 1, 0, &[0], None),
            (2, 1, 1, &[0, 1], None),
            (3, 2, 1, &[0, 2, 1], None),
            (4, 2, 2, &[0, 1, 2, 3], None),
            (5, 3, 2, &[0, 4, 1, 2, 3], None),
            (6, 4, 2, &[0, 4, 1, 2, 3, 5], Some(3)),
            (7, 4, 3, &[0, 4, 1, 2, 3, 5, 6], Some(3)),
            (8, 5, 3, &[0, 6, 1, 2, 3, 4, 5, 7], Some(4)),
        ];
        for (channels, streams, coupled, mapping, lfe) in rows {
            let (layout, mapping_type, lfe_stream) = surround(channels, 1);
            assert_eq!(layout.channels(), channels);
            assert_eq!(layout.streams(), streams, "{channels} channels");
            assert_eq!(layout.coupled_streams(), coupled, "{channels} channels");
            assert_eq!(layout.mapping(), mapping, "{channels} channels");
            assert_eq!(lfe_stream, lfe, "{channels} channels");
            let want_type = if channels > 2 {
                MappingType::Surround
            } else {
                MappingType::None
            };
            assert_eq!(mapping_type, want_type, "{channels} channels");
            // Every coded channel is used exactly once.
            assert_eq!(streams + coupled, channels);
        }
        assert!(ChannelLayout::surround(9, 1).is_err());
        assert!(ChannelLayout::surround(0, 1).is_err());
    }

    #[test]
    fn family_0_is_mono_or_stereo() {
        let (mono, t, lfe) = surround(1, 0);
        assert_eq!(
            (mono.streams(), mono.coupled_streams(), mono.mapping()),
            (1, 0, &[0u8][..])
        );
        assert_eq!((t, lfe), (MappingType::None, None));
        let (stereo, t, lfe) = surround(2, 0);
        assert_eq!(
            (stereo.streams(), stereo.coupled_streams(), stereo.mapping()),
            (1, 1, &[0u8, 1][..])
        );
        assert_eq!((t, lfe), (MappingType::None, None));
        for channels in [0, 3, 4, 255, 256] {
            assert!(
                ChannelLayout::surround(channels, 0).is_err(),
                "{channels} channels"
            );
        }
    }

    #[test]
    fn family_255_is_one_mono_stream_per_channel() {
        for channels in 1..=255usize {
            let (layout, t, lfe) = surround(channels, 255);
            assert_eq!(layout.streams(), channels);
            assert_eq!(layout.coupled_streams(), 0);
            assert!(
                layout
                    .mapping()
                    .iter()
                    .enumerate()
                    .all(|(i, &m)| m as usize == i)
            );
            assert_eq!((t, lfe), (MappingType::None, None));
        }
        assert!(ChannelLayout::surround(0, 255).is_err());
        assert!(ChannelLayout::surround(256, 255).is_err());
    }

    #[test]
    fn family_2_ambisonics_layouts() {
        // (channels, streams, coupled, mapping)
        let valid: [(usize, usize, usize, &[u8]); 8] = [
            (1, 1, 0, &[0]),
            (3, 2, 1, &[2, 0, 1]),
            (4, 4, 0, &[0, 1, 2, 3]),
            (6, 5, 1, &[2, 3, 4, 5, 0, 1]),
            (9, 9, 0, &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (11, 10, 1, &[2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]),
            (
                16,
                16,
                0,
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            ),
            (
                18,
                17,
                1,
                &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1],
            ),
        ];
        for (channels, streams, coupled, mapping) in valid {
            let (layout, t, lfe) = surround(channels, 2);
            assert_eq!(layout.streams(), streams, "{channels} channels");
            assert_eq!(layout.coupled_streams(), coupled, "{channels} channels");
            assert_eq!(layout.mapping(), mapping, "{channels} channels");
            assert_eq!((t, lfe), (MappingType::Ambisonics, None));
        }
        for channels in [0, 2, 5, 7, 8, 10, 12, 15, 17, 228, 229, 255, 256] {
            assert!(
                ChannelLayout::surround(channels, 2).is_err(),
                "{channels} channels"
            );
        }
        // 225 = 15 * 15 and 227 = 225 + 2 are the largest valid counts.
        assert_eq!(validate_ambisonics(225), Some((225, 0)));
        assert_eq!(validate_ambisonics(227), Some((226, 1)));
        assert_eq!(validate_ambisonics(226), None);
    }

    #[test]
    fn validate_ambisonics_accepts_squares_and_squares_plus_two() {
        // Build the answer from the definition: order n has (n + 1) squared
        // channels, and one stereo stream may add 2 more.
        let mut want = [None; 301];
        for order_plus_one in 1..=17usize {
            let acn = order_plus_one * order_plus_one;
            if acn <= 227 {
                want[acn] = Some((acn, 0));
            }
            if acn + 2 <= 227 {
                want[acn + 2] = Some((acn + 1, 1));
            }
        }
        for (channels, want) in want.iter().enumerate() {
            assert_eq!(validate_ambisonics(channels), *want, "{channels} channels");
        }
        assert_eq!(want.iter().filter(|w| w.is_some()).count(), 30);
        assert_eq!(validate_ambisonics(usize::MAX), None);
    }

    #[test]
    fn other_families_are_rejected() {
        for family in [3u8, 4, 100, 254] {
            for channels in 1..=20 {
                assert!(ChannelLayout::surround(channels, family).is_err());
            }
        }
    }

    #[test]
    fn new_rejects_each_invalid_argument() {
        let map = [0u8; 300];
        // Rule 1: channel count.
        assert!(ChannelLayout::new(0, 1, 0, &map).is_err());
        assert!(ChannelLayout::new(256, 1, 0, &map).is_err());
        assert!(ChannelLayout::new(255, 1, 0, &map).is_ok());
        // Rule 2: at least one stream.
        assert!(ChannelLayout::new(1, 0, 0, &map).is_err());
        // Rule 3: coupled streams are a subset of the streams.
        assert!(ChannelLayout::new(2, 1, 2, &map).is_err());
        assert!(ChannelLayout::new(2, 1, 1, &map).is_ok());
        // Rule 4: at most 255 decoded channels.
        assert!(ChannelLayout::new(2, 128, 128, &map).is_err());
        assert!(ChannelLayout::new(2, 128, 127, &map).is_ok());
        assert!(ChannelLayout::new(2, 255, 0, &map).is_ok());
        assert!(ChannelLayout::new(2, 256, 0, &map).is_err());
        assert!(ChannelLayout::new(2, usize::MAX, usize::MAX, &map).is_err());
        // Rule 5: the mapping covers every channel.
        assert!(ChannelLayout::new(3, 2, 1, &[0, 1]).is_err());
        assert!(ChannelLayout::new(3, 2, 1, &[0, 1, 2]).is_ok());
        // Rule 6: every entry is a decoded channel or 255.
        assert!(ChannelLayout::new(3, 2, 1, &[0, 1, 3]).is_err());
        assert!(ChannelLayout::new(3, 2, 1, &[0, 1, 254]).is_err());
        assert!(ChannelLayout::new(3, 2, 1, &[0, 1, 255]).is_ok());
        assert!(ChannelLayout::new(1, 1, 0, &[5]).is_err());
    }

    #[test]
    fn extra_mapping_entries_are_ignored() {
        let a = ChannelLayout::new(2, 1, 1, &[0, 1, 77, 78]).unwrap();
        let b = ChannelLayout::new(2, 1, 1, &[0, 1]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.mapping(), &[0, 1]);
    }

    #[test]
    fn encoder_checks() {
        // Rule 7: more coded channels than input channels.
        let l = ChannelLayout::new(2, 2, 1, &[0, 1]).unwrap();
        assert!(l.validate_for_encoder().is_err());
        // Rule 8: stream 1 (mono, decoded channel 2) has no input channel.
        // The C encoder rejects this layout and the C decoder accepts it.
        let l = ChannelLayout::new(3, 2, 1, &[0, 1, 0]).unwrap();
        assert!(l.validate_for_encoder().is_err());
        // A coupled stream without a right channel.
        let l = ChannelLayout::new(3, 2, 1, &[0, 2, 2]).unwrap();
        assert!(l.validate_for_encoder().is_err());
        // A coupled stream without a left channel.
        let l = ChannelLayout::new(3, 2, 1, &[1, 2, 2]).unwrap();
        assert!(l.validate_for_encoder().is_err());
        // Valid: an input channel that is not encoded, and a repeated entry.
        let l = ChannelLayout::new(5, 2, 1, &[0, 1, 2, 255, 2]).unwrap();
        assert!(l.validate_for_encoder().is_ok());
    }

    #[test]
    fn routing_follows_the_mapping() {
        // 5.1: streams 0 and 1 are stereo, 2 and 3 are mono.
        let (l, _, _) = surround(6, 1);
        assert_eq!(l.left_channel(0, None), Some(0));
        assert_eq!(l.right_channel(0, None), Some(2));
        assert_eq!(l.left_channel(1, None), Some(3));
        assert_eq!(l.right_channel(1, None), Some(4));
        assert_eq!(l.mono_channel(2, None), Some(1));
        assert_eq!(l.mono_channel(3, None), Some(5));
        // Each of them appears once.
        assert_eq!(l.left_channel(0, Some(0)), None);
        assert_eq!(l.mono_channel(3, Some(5)), None);
        // A stream of the other kind, or no stream at all, has no channel.
        assert_eq!(l.left_channel(2, None), None);
        assert_eq!(l.right_channel(3, None), None);
        assert_eq!(l.mono_channel(0, None), None);
        assert_eq!(l.mono_channel(4, None), None);
        assert_eq!(l.left_channel(usize::MAX, None), None);
        assert_eq!(l.mono_channel(usize::MAX, None), None);
    }

    #[test]
    fn routing_with_repeated_mapping_values() {
        // 1 stereo stream and 1 mono stream feed 8 output channels.
        let l = ChannelLayout::new(8, 2, 1, &[2, 0, 1, 0, 255, 2, 1, 2]).unwrap();
        let all = |first: Option<usize>, next: &dyn Fn(Option<usize>) -> Option<usize>| {
            let mut found = [usize::MAX; 8];
            let mut n = 0;
            let mut cur = first;
            while let Some(c) = cur {
                found[n] = c;
                n += 1;
                cur = next(Some(c));
            }
            (found, n)
        };
        let (left, n) = all(l.left_channel(0, None), &|p| l.left_channel(0, p));
        assert_eq!(&left[..n], &[1, 3]);
        let (right, n) = all(l.right_channel(0, None), &|p| l.right_channel(0, p));
        assert_eq!(&right[..n], &[2, 6]);
        let (mono, n) = all(l.mono_channel(1, None), &|p| l.mono_channel(1, p));
        assert_eq!(&mono[..n], &[0, 5, 7]);
        // `prev` beyond the last channel, and the largest possible `prev`.
        assert_eq!(l.mono_channel(1, Some(7)), None);
        assert_eq!(l.mono_channel(1, Some(1000)), None);
        assert_eq!(l.mono_channel(1, Some(usize::MAX)), None);
    }

    /// Deterministic pseudo-random numbers.
    struct Rng(u32);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 8) as usize % n
        }
    }

    /// The loop that libopus `get_left_channel`, `get_right_channel` and
    /// `get_mono_channel` share, written as in C: the first channel after
    /// `prev` whose mapping entry is `index`, or -1. `prev` is -1 to start.
    fn c_lookup(mapping: &[u8], index: usize, prev: i32) -> i32 {
        let mut i = if prev < 0 { 0 } else { prev as usize + 1 };
        while i < mapping.len() {
            if mapping[i] as usize == index {
                return i as i32;
            }
            i += 1;
        }
        -1
    }

    #[test]
    fn lookups_follow_the_c_loops_on_random_layouts() {
        let as_c = |r: Option<usize>| r.map_or(-1, |c| c as i32);
        let mut rng = Rng(0x5EED);
        let mut layouts = 0;
        let mut encoder_layouts = 0;
        while layouts < 3000 {
            let channels = if rng.below(20) == 0 {
                1 + rng.below(255)
            } else {
                1 + rng.below(12)
            };
            let streams = 1 + rng.below(6);
            let coupled = rng.below(streams + 1);
            let decoded = streams + coupled;
            let mut mapping = [0u8; 255];
            for m in mapping[..channels].iter_mut() {
                *m = if rng.below(8) == 0 {
                    SILENT_CHANNEL
                } else {
                    rng.below(decoded) as u8
                };
            }
            let mapping = &mapping[..channels];
            let l = ChannelLayout::new(channels, streams, coupled, mapping).unwrap();
            layouts += 1;

            // Every stream index up to a few past the end, and every `prev`
            // up to a few past the last channel.
            for s in 0..streams + 3 {
                for prev in (0..channels + 2).map(Some).chain([None]) {
                    let c_prev = as_c(prev);
                    let (left, right, mono) = (
                        as_c(l.left_channel(s, prev)),
                        as_c(l.right_channel(s, prev)),
                        as_c(l.mono_channel(s, prev)),
                    );
                    if s < coupled {
                        assert_eq!(left, c_lookup(mapping, 2 * s, c_prev));
                        assert_eq!(right, c_lookup(mapping, 2 * s + 1, c_prev));
                        assert_eq!(mono, -1);
                    } else if s < streams {
                        assert_eq!(mono, c_lookup(mapping, s + coupled, c_prev));
                        assert_eq!((left, right), (-1, -1));
                    } else {
                        assert_eq!((left, right, mono), (-1, -1, -1));
                    }
                }
            }

            // validate_encoder_layout and the encoder's argument check.
            let has_inputs = (0..streams).all(|s| {
                if s < coupled {
                    c_lookup(mapping, 2 * s, -1) != -1 && c_lookup(mapping, 2 * s + 1, -1) != -1
                } else {
                    c_lookup(mapping, s + coupled, -1) != -1
                }
            });
            let want = decoded <= channels && has_inputs;
            assert_eq!(l.validate_for_encoder().is_ok(), want, "{l:?}");
            encoder_layouts += want as usize;
        }
        // Both answers of the encoder check were seen often.
        assert!((200..2800).contains(&encoder_layouts), "{encoder_layouts}");
    }

    #[test]
    fn channels_by_entry_lists_what_the_lookups_find() {
        let mut rng = Rng(0xC0FFEE);
        for round in 0..2000 {
            let channels = if round % 10 == 0 { 255 } else { 1 + rng.below(12) };
            let streams = 1 + rng.below(6);
            let coupled = rng.below(streams + 1);
            let mut mapping = [0u8; 255];
            for m in mapping[..channels].iter_mut() {
                *m = if rng.below(8) == 0 {
                    SILENT_CHANNEL
                } else {
                    rng.below(streams + coupled) as u8
                };
            }
            let l = ChannelLayout::new(channels, streams, coupled, &mapping[..channels]).unwrap();
            let by_entry = ChannelsByEntry::new(&l);
            let mut total = 0;
            for entry in 0..=255usize {
                // Every channel after the one before, as the C loop finds them.
                let mut want = Vec::new();
                let mut prev = -1;
                loop {
                    prev = c_lookup(l.mapping(), entry, prev);
                    if prev < 0 {
                        break;
                    }
                    want.push(prev as u8);
                }
                assert_eq!(by_entry.with_entry(entry), &want[..], "{l:?}, entry {entry}");
                total += want.len();
            }
            assert_eq!(total, channels);
            assert!(by_entry.with_entry(256).is_empty());
            assert!(by_entry.with_entry(usize::MAX).is_empty());
        }
    }

    #[test]
    fn no_argument_makes_a_function_panic() {
        const SIZES: [usize; 16] = [
            0,
            1,
            2,
            3,
            8,
            9,
            127,
            128,
            227,
            228,
            254,
            255,
            256,
            u32::MAX as usize,
            usize::MAX - 1,
            usize::MAX,
        ];
        let mut rng = Rng(99);
        let mut mapping = [0u8; 300];
        let mut accepted = 0;
        for _ in 0..100_000 {
            let (channels, streams, coupled, other) = (
                SIZES[rng.below(16)],
                SIZES[rng.below(16)],
                SIZES[rng.below(16)],
                SIZES[rng.below(16)],
            );
            let len = [0usize, 1, 2, 254, 255, 256, 300][rng.below(7)];
            for m in mapping[..len].iter_mut() {
                *m = [0u8, 1, 2, 127, 253, 254, 255][rng.below(7)];
            }
            if let Ok(l) = ChannelLayout::new(channels, streams, coupled, &mapping[..len]) {
                accepted += 1;
                assert_eq!(
                    (l.channels(), l.streams(), l.coupled_streams()),
                    (channels, streams, coupled)
                );
                assert_eq!(l.mapping(), &mapping[..channels]);
                let _ = l.validate_for_encoder();
                let prev = [None, Some(other), Some(rng.below(300))][rng.below(3)];
                for found in [
                    l.left_channel(other, prev),
                    l.right_channel(other, prev),
                    l.mono_channel(other, prev),
                    l.left_channel(rng.below(streams), prev),
                    l.right_channel(rng.below(streams), prev),
                    l.mono_channel(rng.below(streams), prev),
                ]
                .into_iter()
                .flatten()
                {
                    assert!(found < channels && prev.is_none_or(|p| found > p));
                    assert_ne!(l.mapping()[found], SILENT_CHANNEL);
                }
            }
            if let Ok((l, _, lfe)) = ChannelLayout::surround(channels, rng.below(256) as u8) {
                assert_eq!(l.channels(), channels);
                assert!(l.validate_for_encoder().is_ok());
                assert!(lfe.is_none_or(|s| s >= l.coupled_streams() && s < l.streams()));
            }
            let _ = validate_ambisonics(channels);
        }
        assert!(accepted > 1000, "{accepted}");
    }

    #[test]
    fn silent_channel_is_never_a_stream_channel() {
        // 255 decoded channels: 127 coupled streams and 1 mono stream. The
        // right channel of a stream 127 would be decoded channel 255, which
        // is the silent value. Stream 127 is the mono stream, so the lookup
        // must not report a silent channel as its right channel.
        let mut map = [SILENT_CHANNEL; 255];
        map[0] = 254;
        let l = ChannelLayout::new(255, 128, 127, &map).unwrap();
        assert_eq!(l.right_channel(127, None), None);
        assert_eq!(l.mono_channel(127, None), Some(0));
        assert_eq!(l.left_channel(126, None), None);
    }
}
