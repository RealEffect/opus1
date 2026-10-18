//! Tools that measure decoded audio. They are used by the conformance test
//! of RFC 8251 and by the comparison of opus1 with libopus.
//!
//! - [`compare`]: a port of the libopus test program `opus_compare`
//!   (`src/opus_compare.c`). It gives the same numbers as the C program
//!   built for the same processor, bit for bit.
//! - [`moc`]: the "modified opus_compare" (MOC) distance of the Opus Speech
//!   Coding Enhancement draft, ported from `moc.py` of the libopus git
//!   repository.
//! - [`demo`]: the decoding loop of `opus_demo -d`, which the RFC 8251
//!   check runs before it compares.
//!
//! This module is not part of the stable API. It is reached as
//! `opus1::internals::eval` and needs the `std` feature: it allocates, and
//! it calls the math library of the platform (`cos`, `sin`, `log`, `pow`),
//! as the C program does.

pub mod compare;
pub mod demo;
pub mod moc;

pub use compare::{
    BANDS, CompareError, Comparison, NBANDS, NFREQS, RATES, Reference, TEST_WIN_SIZE,
    TEST_WIN_STEP, TestSignal, opus_compare, opus_compare_many,
};
pub use demo::{
    BitPacket, Bitstream, BitstreamError, DemoDecoder, DemoOutput, MAX_FRAME_SIZE, MAX_PACKET,
    RangeMismatch, decode_bitstream, opus_demo_s16, parse_bitstream, res2int24,
};
pub use moc::{MOC_RATE, MocError, moc, moc_i16, moc_relative};
