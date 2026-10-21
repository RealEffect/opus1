//! Deep redundancy (DRED), libopus 1.5 and later.
//!
//! With DRED, the encoder adds a compressed history of the speech to each
//! packet: up to about one second of features, coded by the RDOVAE encoder
//! (rate-distortion-optimized variational autoencoder). The data travels in
//! a packet extension (see `crate::extensions`). After a loss, the decoder
//! turns it back into features with the RDOVAE decoder and gives them to
//! deep PLC, which synthesizes the lost audio from them.
//!
//! This module holds the constants of `dnn/dred_config.h` and the decoder
//! side of DRED:
//!
//! - `decoder`: `OpusDredDecoder`, `OpusDred` and the reading of the DRED
//!   payload (libopus `opus_dred_parse`, `opus_dred_process`,
//!   `dnn/dred_decoder.c`).
//! - `rdovae_dec`: the RDOVAE decoder network (`dnn/dred_rdovae_dec.c`).
//! - `coding`: the quantizer schedule (`dnn/dred_coding.c`).
//!
//! `OpusDecoder::dred_decode` and its variants (`src/lib.rs`) give the
//! features to deep PLC. The layers of the models are in
//! `models::dred_rdovae_enc` and `models::dred_rdovae_dec`, and the
//! constants of the RDOVAE and the tables of the quantizer in
//! `models::dred_rdovae_stats`.

/// The extension ID that carries DRED data (`DRED_EXTENSION_ID`). libopus
/// 1.6.1 uses 126, an ID for experiments, until DRED gets its own number.
pub const DRED_EXTENSION_ID: u8 = 126;

/// The version of the experimental DRED format (`DRED_EXPERIMENTAL_VERSION`).
/// The DRED payload starts with the byte `b'D'` and then this byte.
pub const DRED_EXPERIMENTAL_VERSION: u8 = 12;

/// The bytes of that header, `b'D'` and the version
/// (`DRED_EXPERIMENTAL_BYTES`).
pub const DRED_EXPERIMENTAL_BYTES: usize = 2;

/// The smallest DRED payload the encoder writes, without the header
/// (`DRED_MIN_BYTES`).
pub const DRED_MIN_BYTES: usize = 8;

/// `DRED_SILK_ENCODER_DELAY`, `(79+12-80)` in C: the samples at 16 kHz that
/// the DRED encoder's input buffer starts with, so that its frames line up
/// with those of the SILK encoder.
pub const DRED_SILK_ENCODER_DELAY: usize = 79 + 12 - 80;

/// The samples at 16 kHz of one DRED frame, 10 ms (`DRED_FRAME_SIZE`).
pub const DRED_FRAME_SIZE: usize = 160;

/// The samples at 16 kHz of two DRED frames, 20 ms (`DRED_DFRAME_SIZE`).
/// The RDOVAE encoder makes one latent vector per 20 ms.
pub const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;

/// The largest DRED payload in bytes (`DRED_MAX_DATA_SIZE`).
pub const DRED_MAX_DATA_SIZE: usize = 1000;

/// The default quantizer level of the first latent vector (`DRED_ENC_Q0`).
pub const DRED_ENC_Q0: i32 = 6;

/// The default quantizer level of the last latent vectors (`DRED_ENC_Q1`).
pub const DRED_ENC_Q1: i32 = 15;

/// The most latent vectors in one packet (`DRED_MAX_LATENTS`). 26 latent
/// vectors cover 1.04 seconds: one second after the lookahead.
pub const DRED_MAX_LATENTS: usize = 26;

/// The most 10 ms frames that one packet's DRED data covers
/// (`DRED_NUM_REDUNDANCY_FRAMES`, `2*DRED_MAX_LATENTS` in C).
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;

/// `DRED_MAX_FRAMES`, `4*DRED_MAX_LATENTS` in C. It is the largest value of
/// `OPUS_SET_DRED_DURATION`, which counts 10 ms frames, and the number of
/// latent vectors and states that the DRED encoder keeps.
pub const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;

pub mod coding;
pub mod decoder;
pub(crate) mod encoder;
pub mod rdovae_dec;
pub(crate) mod rdovae_enc;
