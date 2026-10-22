//! The error type of opus1.
//!
//! Every call that can fail returns [`Error`]. Its [`ErrorKind`] matches the
//! error code libopus returns at the same place, so code that knows libopus
//! can handle the errors the same way. Its message says in a few words what
//! was wrong, for people reading a log. Compare kinds, not messages: the
//! wording of a message may change in any release.

use core::fmt;

/// What kind of error happened: one per libopus error code.
///
/// More kinds may be added in a later release, so a `match` on it needs a
/// `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// One or more arguments are invalid or out of range (libopus
    /// `OPUS_BAD_ARG`, -1).
    BadArg,
    /// The output buffer is too small for the result (libopus
    /// `OPUS_BUFFER_TOO_SMALL`, -2).
    BufferTooSmall,
    /// The codec found an internal inconsistency (libopus
    /// `OPUS_INTERNAL_ERROR`, -3).
    InternalError,
    /// The compressed data is corrupted or not a valid Opus packet (libopus
    /// `OPUS_INVALID_PACKET`, -4).
    InvalidPacket,
    /// The request is not supported, for example because the cargo feature
    /// it needs is off (libopus `OPUS_UNIMPLEMENTED`, -5).
    Unimplemented,
    /// The encoder or decoder is not in a state that allows the call
    /// (libopus `OPUS_INVALID_STATE`, -6).
    InvalidState,
    /// Memory allocation failed (libopus `OPUS_ALLOC_FAIL`, -7).
    AllocFail,
}

impl ErrorKind {
    /// The libopus error code of this kind, for example -1 for
    /// [`ErrorKind::BadArg`].
    pub const fn to_raw(self) -> i32 {
        match self {
            ErrorKind::BadArg => -1,
            ErrorKind::BufferTooSmall => -2,
            ErrorKind::InternalError => -3,
            ErrorKind::InvalidPacket => -4,
            ErrorKind::Unimplemented => -5,
            ErrorKind::InvalidState => -6,
            ErrorKind::AllocFail => -7,
        }
    }
}

/// An error of opus1: its kind and a message that says what was wrong.
///
/// `Display` writes the message. The message starts with a lowercase
/// letter and has no trailing period, so it can be part of a longer
/// sentence.
///
/// ```
/// use opus1::{ErrorKind, SampleRate};
///
/// let err = SampleRate::try_from(44100u32).unwrap_err();
/// assert_eq!(err.kind(), ErrorKind::BadArg);
/// assert_eq!(err.to_raw(), -1);
/// assert_eq!(err.to_string(), "invalid sampling rate");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Error {
    kind: ErrorKind,
    message: &'static str,
}

impl Error {
    /// libopus `OPUS_BAD_ARG` with the libopus message.
    pub(crate) const BAD_ARG: Error = Error::bad_arg("invalid argument");
    /// libopus `OPUS_BUFFER_TOO_SMALL` with the message used for every
    /// output buffer that is too small.
    pub(crate) const BUFFER_TOO_SMALL: Error = Error::buffer_too_small("output buffer too small");
    /// libopus `OPUS_INTERNAL_ERROR` with the libopus message.
    pub(crate) const INTERNAL_ERROR: Error = Error::internal("internal error");
    /// libopus `OPUS_INVALID_PACKET` with the message used for every
    /// packet that cannot be parsed.
    pub(crate) const INVALID_PACKET: Error = Error::invalid_packet("invalid packet");
    /// libopus `OPUS_UNIMPLEMENTED` with the libopus message. Only some
    /// sets of cargo features use it.
    #[allow(dead_code)]
    pub(crate) const UNIMPLEMENTED: Error =
        Error::new(ErrorKind::Unimplemented, "request not implemented");

    // The errors that several codecs share, so that each has one wording.
    pub(crate) const INVALID_SAMPLING_RATE: Error = Error::bad_arg("invalid sampling rate");
    pub(crate) const INVALID_CHANNELS: Error = Error::bad_arg("invalid number of channels");
    pub(crate) const INVALID_FRAME_SIZE: Error = Error::bad_arg("invalid frame size");
    /// The input holds fewer than `frame_size` samples per channel.
    pub(crate) const INPUT_TOO_SHORT: Error = Error::bad_arg("input buffer too small for frame");
    /// The packet lasts longer than the `frame_size` the caller gave.
    pub(crate) const FRAME_SIZE_TOO_SMALL: Error =
        Error::buffer_too_small("output buffer too small: frame_size is shorter than the packet");
    /// The duration of a lost frame, or of a frame recovered with FEC
    /// (forward error correction), is not a whole number of 2.5 ms.
    pub(crate) const LOST_FRAME_SIZE: Error =
        Error::bad_arg("lost frame: frame_size must be a multiple of 2.5 ms");
    pub(crate) const INVALID_BITRATE: Error = Error::bad_arg("invalid bitrate");
    pub(crate) const INVALID_COMPLEXITY: Error = Error::bad_arg("invalid complexity");
    pub(crate) const INVALID_LSB_DEPTH: Error = Error::bad_arg("invalid LSB depth");
    pub(crate) const INVALID_PACKET_LOSS: Error = Error::bad_arg("invalid packet loss percentage");
    pub(crate) const INVALID_MAX_BANDWIDTH: Error = Error::bad_arg("invalid maximum bandwidth");

    /// An error of kind `kind` with the message `message`.
    pub(crate) const fn new(kind: ErrorKind, message: &'static str) -> Self {
        Error { kind, message }
    }

    /// An error of kind [`ErrorKind::BadArg`].
    pub(crate) const fn bad_arg(message: &'static str) -> Self {
        Error::new(ErrorKind::BadArg, message)
    }

    /// An error of kind [`ErrorKind::BufferTooSmall`].
    pub(crate) const fn buffer_too_small(message: &'static str) -> Self {
        Error::new(ErrorKind::BufferTooSmall, message)
    }

    /// An error of kind [`ErrorKind::InternalError`].
    pub(crate) const fn internal(message: &'static str) -> Self {
        Error::new(ErrorKind::InternalError, message)
    }

    /// An error of kind [`ErrorKind::InvalidPacket`].
    pub(crate) const fn invalid_packet(message: &'static str) -> Self {
        Error::new(ErrorKind::InvalidPacket, message)
    }

    /// What kind of error this is.
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The message that says what was wrong, for example
    /// `"output buffer too small"`.
    pub const fn message(&self) -> &'static str {
        self.message
    }

    /// The libopus error code, for example -1 for [`ErrorKind::BadArg`].
    pub const fn to_raw(&self) -> i32 {
        self.kind.to_raw()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_codes_match_libopus() {
        let kinds = [
            (ErrorKind::BadArg, -1),
            (ErrorKind::BufferTooSmall, -2),
            (ErrorKind::InternalError, -3),
            (ErrorKind::InvalidPacket, -4),
            (ErrorKind::Unimplemented, -5),
            (ErrorKind::InvalidState, -6),
            (ErrorKind::AllocFail, -7),
        ];
        for (kind, raw) in kinds {
            assert_eq!(kind.to_raw(), raw);
            assert_eq!(Error::new(kind, "x").to_raw(), raw);
        }
    }

    #[test]
    fn display_writes_the_message() {
        let err = Error::BUFFER_TOO_SMALL;
        assert_eq!(format!("{err}"), "output buffer too small");
        assert_eq!(err.kind(), ErrorKind::BufferTooSmall);
        assert_eq!(err.message(), "output buffer too small");
    }
}
