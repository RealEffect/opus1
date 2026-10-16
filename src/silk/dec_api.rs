//! The SILK decoder for one Opus stream, ported from libopus
//! `silk/dec_API.c`.
//!
//! One call decodes one SILK frame (10 or 20 ms) of one or two channels.
//! There are two entry points:
//!
//! - [`SilkDecoder::decode_with_control`] is the full port of `silk_Decode`.
//!   It resamples to the output rate, and it converts when the packet does
//!   not code as many channels as the caller wants.
//! - [`SilkDecoder::decode`] is the older entry point. It returns the
//!   decoded channels at the SILK rate and leaves the resampling to the
//!   caller. It cannot convert between channel counts.
//!
//! Both can decode three ways, chosen with the lost flag:
//! `FLAG_DECODE_NORMAL` decodes the frame, `FLAG_PACKET_LOST` conceals a
//! lost frame, and `FLAG_DECODE_LBRR` decodes the low-bitrate copy of the
//! previous packet that a packet can carry (in-band forward error
//! correction). libopus calls that copy LBRR, "low bit rate redundancy".

use crate::range_coder::RangeCoder;
use crate::silk::decode_frame::{
    FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST, silk_decode_frame,
};
use crate::silk::decode_indices::{
    silk_decode_indices, silk_stereo_decode_mid_only, silk_stereo_decode_pred,
};
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_structs::{SilkDecControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::init_decoder::{silk_decoder_set_fs, silk_init_decoder, silk_reset_decoder};
use crate::silk::stereo_ms_to_lr::{StereoDecState, silk_stereo_ms_to_lr};
use crate::silk::tables::{SILK_LBRR_FLAGS_2_ICDF, SILK_LBRR_FLAGS_3_ICDF};

/// Samples of one 20 ms frame at the highest output rate, 48 kHz.
const MAX_API_FRAME_LENGTH: usize = 960;

pub struct SilkDecoder {
    pub channel_state: [SilkDecoderState; 2],

    /// Channel count the caller asked for in the previous call (libopus
    /// `psDec->nChannelsAPI`).
    pub n_channels_api: i32,

    /// Channel count of the packet decoded in the previous call (libopus
    /// `psDec->nChannelsInternal`). [`SilkDecoder::decode`] also uses it as
    /// the channel count of every packet.
    pub n_channels_internal: i32,

    pub prev_decode_only_middle: i32,

    /// Persistent stereo M/S decoder state.
    pub s_stereo: StereoDecState,

    /// Per-channel temp buffers for M/S → L/R conversion (frame_length + 2).
    w_silk_buf: [[i16; MAX_FRAME_LENGTH + 2]; 2],

    /// One channel of one frame at the output rate, before it is written
    /// into the interleaved stereo output.
    w_resample_buf: [i16; MAX_API_FRAME_LENGTH],
}

impl Default for SilkDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// What the part shared by both entry points hands back.
struct DecodedFrame {
    /// Samples per channel at the SILK rate. They are in
    /// `w_silk_buf[n][1..1 + n_samples]`.
    n_samples: usize,
    /// The packet before this one was stereo and this one is mono, at the
    /// same SILK rate (libopus `stereo_to_mono`).
    stereo_to_mono: bool,
}

impl SilkDecoder {
    pub fn new() -> Self {
        let mut dec = Self {
            channel_state: [SilkDecoderState::default(), SilkDecoderState::default()],
            n_channels_api: 1,
            n_channels_internal: 1,
            prev_decode_only_middle: 0,
            s_stereo: StereoDecState::default(),
            w_silk_buf: [[0; MAX_FRAME_LENGTH + 2]; 2],
            w_resample_buf: [0; MAX_API_FRAME_LENGTH],
        };
        silk_init_decoder(&mut dec.channel_state[0]);
        silk_init_decoder(&mut dec.channel_state[1]);
        dec
    }

    /// Set the rate and the channel count for [`SilkDecoder::decode`].
    ///
    /// [`SilkDecoder::decode_with_control`] does not need this call: it gets
    /// both with every frame.
    pub fn init(&mut self, sample_rate_hz: i32, channels: i32) -> i32 {
        let fs_khz = sample_rate_hz / 1000;
        let ret = silk_decoder_set_fs(&mut self.channel_state[0], fs_khz, sample_rate_hz);
        if ret < 0 {
            return ret;
        }
        if channels == 2 {
            let ret = silk_decoder_set_fs(&mut self.channel_state[1], fs_khz, sample_rate_hz);
            if ret < 0 {
                return ret;
            }
        }

        self.channel_state[0].n_frames_per_packet = 1;
        self.n_channels_api = channels;
        self.n_channels_internal = channels;
        ret
    }

    /// Decode one SILK frame to the SILK rate, without resampling.
    ///
    /// The packet must code as many channels as `init()` was given. The
    /// output is one channel after the other: channel 0 in
    /// `output[..n]`, channel 1 in `output[n..2 * n]`, where `n` is the
    /// return value. A negative return value is an error.
    ///
    /// The output rate that `silk_decoder_set_fs` needs is read from
    /// `channel_state[0].fs_api_hz`.
    pub fn decode(
        &mut self,
        range_dec: &mut RangeCoder,
        output: &mut [i16],
        lost_flag: i32,
        new_packet: bool,
        payload_size_ms: i32,
        internal_sample_rate: i32,
    ) -> i32 {
        // A decoder that never went through `init()` has no output rate.
        // This entry point does not resample, so the SILK rate serves.
        let api_sample_rate = match self.channel_state[0].fs_api_hz {
            0 => ((internal_sample_rate >> 10) + 1) * 1000,
            rate => rate,
        };
        let control = SilkDecControl {
            n_channels_api: self.n_channels_internal,
            n_channels_internal: self.n_channels_internal,
            api_sample_rate,
            internal_sample_rate,
            payload_size_ms,
        };
        let decoded = check_control(&control, lost_flag).and_then(|fs_khz_dec| {
            self.decode_to_silk_rate(&control, fs_khz_dec, range_dec, lost_flag, new_packet)
        });
        let frame = match decoded {
            Ok(frame) => frame,
            Err(_) => return -1,
        };

        // libopus (`silk/dec_API.c`) resamples mono and stereo alike from
        // `samplesOut1_tmp[n][1]`: samples [0]/[1] are `sMid`, so index 1 is
        // the one-sample-delayed position. Mono reads from there too, not
        // from index 2.
        let fl = frame.n_samples;
        let channels = control.n_channels_internal as usize;
        if output.len() < channels * fl {
            return -1;
        }
        for (n, plane) in output.chunks_exact_mut(fl).take(channels).enumerate() {
            plane.copy_from_slice(&self.w_silk_buf[n][1..1 + fl]);
        }
        fl as i32
    }

    /// Decode one SILK frame to the output rate: the port of libopus
    /// `silk_Decode`.
    ///
    /// `output` receives `control.n_channels_api` interleaved channels at
    /// `control.api_sample_rate`. The return value is the number of samples
    /// per channel.
    ///
    /// The packet may code fewer or more channels than the output has:
    ///
    /// - Mono packet, stereo output: both output channels get the decoded
    ///   signal.
    /// - Stereo packet, mono output: the output is the mid channel, which is
    ///   the average of left and right.
    ///
    /// A 40 or 60 ms Opus frame holds 2 or 3 SILK frames. Call this once per
    /// SILK frame with the same range decoder, and set `new_packet` in the
    /// first call only. For a lost packet pass the control values of the
    /// last received packet, with `payload_size_ms` 10 or 20.
    ///
    /// With `FLAG_DECODE_LBRR` the range decoder must be at the start of the
    /// packet that followed the lost one. A SILK frame for which that packet
    /// carries no copy is concealed instead.
    pub fn decode_with_control(
        &mut self,
        control: &SilkDecControl,
        range_dec: &mut RangeCoder,
        output: &mut [i16],
        lost_flag: i32,
        new_packet: bool,
    ) -> Result<usize, &'static str> {
        if !matches!(control.api_sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err("SILK decoder: invalid output sampling rate");
        }
        let fs_khz_dec = check_control(control, lost_flag)?;
        let n_api = control.n_channels_api as usize;
        let n_int = control.n_channels_internal as usize;

        // The frame size is known before decoding. Check the output buffer
        // now, so that a buffer that is too small does not cost a frame.
        let state = &self.channel_state[0];
        let (frame_ms, fs_khz) = if new_packet || state.n_frames_decoded == 0 {
            (if control.payload_size_ms <= 10 { 10 } else { 20 }, fs_khz_dec)
        } else {
            (state.nb_subfr * SUB_FRAME_LENGTH_MS as i32, state.fs_khz)
        };
        let n_out = (frame_ms * control.api_sample_rate / 1000) as usize;
        if output.len() < n_out * n_api {
            return Err("SILK decoder: output buffer too small");
        }

        let frame =
            self.decode_to_silk_rate(control, fs_khz_dec, range_dec, lost_flag, new_packet)?;
        let n_dec = frame.n_samples;
        if n_dec * control.api_sample_rate as usize != n_out * fs_khz as usize * 1000 {
            return Err("SILK decoder: unexpected frame length");
        }

        // Resample to the output rate. The input starts at index 1 of the
        // frame buffer: the buffer begins with the last two samples of the
        // previous frame, and libopus reads from the second of them.
        if n_api == 1 {
            self.channel_state[0].resampler_state.process(
                &mut output[..n_out],
                &self.w_silk_buf[0][1..1 + n_dec],
                n_dec as i32,
            );
        } else {
            for n in 0..n_int {
                self.channel_state[n].resampler_state.process(
                    &mut self.w_resample_buf[..n_out],
                    &self.w_silk_buf[n][1..1 + n_dec],
                    n_dec as i32,
                );
                let channel = output.iter_mut().skip(n).step_by(2);
                for (dst, &v) in channel.zip(&self.w_resample_buf[..n_out]) {
                    *dst = v;
                }
            }
            // Two channel output from a mono packet.
            if n_int == 1 {
                if frame.stereo_to_mono {
                    // The first mono frame after stereo: the right channel
                    // goes through its own resampler once more. That
                    // resampler still holds the last samples of the stereo
                    // right channel, and this lets them out.
                    self.channel_state[1].resampler_state.process(
                        &mut self.w_resample_buf[..n_out],
                        &self.w_silk_buf[0][1..1 + n_dec],
                        n_dec as i32,
                    );
                    let right = output.iter_mut().skip(1).step_by(2);
                    for (dst, &v) in right.zip(&self.w_resample_buf[..n_out]) {
                        *dst = v;
                    }
                } else {
                    for i in 0..n_out {
                        output[2 * i + 1] = output[2 * i];
                    }
                }
            }
        }
        Ok(n_out)
    }

    /// Pitch lag of the last decoded frame in samples at 48 kHz, or 0 if
    /// that frame was not voiced (libopus `decControl->prevPitchLag`).
    pub fn prev_pitch_lag(&self) -> i32 {
        let state = &self.channel_state[0];
        if state.prev_signal_type == TYPE_VOICED {
            let mult = match state.fs_khz {
                8 => 6,
                12 => 4,
                _ => 3,
            };
            state.lag_prev * mult
        } else {
            0
        }
    }

    /// Everything `silk_Decode` does before it resamples: decode or conceal
    /// one frame per coded channel and turn mid/side into left/right.
    ///
    /// `control` and `lost_flag` have passed `check_control`, which gave
    /// `fs_khz_dec`.
    fn decode_to_silk_rate(
        &mut self,
        control: &SilkDecControl,
        fs_khz_dec: i32,
        range_dec: &mut RangeCoder,
        lost_flag: i32,
        new_packet: bool,
    ) -> Result<DecodedFrame, &'static str> {
        let n_api = control.n_channels_api as usize;
        let n_int = control.n_channels_internal as usize;

        // First frame of a packet: start counting its frames.
        if new_packet {
            for state in &mut self.channel_state[..n_int] {
                state.n_frames_decoded = 0;
            }
        }

        // The stream changes from mono to stereo: the second channel starts
        // afresh.
        if control.n_channels_internal > self.n_channels_internal {
            silk_init_decoder(&mut self.channel_state[1]);
        }

        // The stream changes from stereo to mono at the same SILK rate. In
        // libopus the second channel's resampler then always runs at the
        // rates of this frame, because the frame before used it. The last
        // two conditions say so explicitly: a decoder set up with `init()`
        // can get here without ever having decoded a stereo frame.
        let stereo_to_mono = control.n_channels_internal == 1
            && self.n_channels_internal == 2
            && control.internal_sample_rate == 1000 * self.channel_state[0].fs_khz
            && self.channel_state[1].fs_khz == self.channel_state[0].fs_khz
            && self.channel_state[1].resampler_api_hz == control.api_sample_rate;

        if self.channel_state[0].n_frames_decoded == 0 {
            let (n_frames_per_packet, nb_subfr) = match control.payload_size_ms {
                // 0: assuming packet loss, use 10 ms.
                0 | 10 => (1, 2),
                20 => (1, 4),
                40 => (2, 4),
                _ => (3, 4),
            };
            for state in &mut self.channel_state[..n_int] {
                state.n_frames_per_packet = n_frames_per_packet;
                state.nb_subfr = nb_subfr;
                // The frame length and the pitch contour table follow from
                // `nb_subfr` inside silk_decoder_set_fs.
                if silk_decoder_set_fs(state, fs_khz_dec, control.api_sample_rate) < 0 {
                    return Err("SILK decoder: unsupported sampling rate");
                }
            }
        }

        // Stereo output starts, either because the stream was mono or
        // because the caller wanted mono until now. The stereo predictor
        // starts from zero, and the right channel's resampler continues
        // from the state of the left one, which has been running.
        if n_api == 2
            && n_int == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.s_stereo.pred_prev_q13 = [0; 2];
            self.s_stereo.s_side = [0; 2];
            let (first, second) = self.channel_state.split_at_mut(1);
            second[0]
                .resampler_state
                .clone_from(&first[0].resampler_state);
        }
        self.n_channels_api = control.n_channels_api;
        self.n_channels_internal = control.n_channels_internal;

        // What follows indexes the per-frame flags with the frame counter
        // and relies on the frame geometry. libopus asserts these; a caller
        // that asks for more frames than the packet holds gets an error.
        let fs_khz = self.channel_state[0].fs_khz;
        let frame_length = self.channel_state[0].frame_length;
        for state in &self.channel_state[..n_int] {
            if lost_flag != FLAG_PACKET_LOST && state.n_frames_decoded >= state.n_frames_per_packet
            {
                return Err("SILK decoder: more frames requested than the packet holds");
            }
            if state.fs_khz != fs_khz
                || state.frame_length != frame_length
                || frame_length <= 0
                || frame_length as usize > MAX_FRAME_LENGTH
                || state.resampler_api_hz != control.api_sample_rate
            {
                return Err("SILK decoder: inconsistent state");
            }
        }

        if lost_flag != FLAG_PACKET_LOST && self.channel_state[0].n_frames_decoded == 0 {
            // First call for this packet: voice activity flags and LBRR flag.
            for state in &mut self.channel_state[..n_int] {
                for i in 0..state.n_frames_per_packet as usize {
                    state.vad_flags[i] = range_dec.decode_bit_logp(1) as i32;
                }
                state.lbrr_flag = range_dec.decode_bit_logp(1) as i32;
            }
            // Which frames have an LBRR copy.
            for state in &mut self.channel_state[..n_int] {
                state.lbrr_flags.fill(0);
                if state.lbrr_flag != 0 {
                    let n_frames = state.n_frames_per_packet as usize;
                    if n_frames == 1 {
                        state.lbrr_flags[0] = 1;
                    } else {
                        let icdf = if n_frames == 2 {
                            &SILK_LBRR_FLAGS_2_ICDF[..]
                        } else {
                            &SILK_LBRR_FLAGS_3_ICDF[..]
                        };
                        let lbrr_symbol = range_dec.decode_icdf(icdf, 8) + 1;
                        for i in 0..n_frames {
                            state.lbrr_flags[i] = (lbrr_symbol >> i) & 1;
                        }
                    }
                }
            }

            if lost_flag == FLAG_DECODE_NORMAL {
                // Regular decoding: read past all LBRR data.
                for i in 0..self.channel_state[0].n_frames_per_packet as usize {
                    for n in 0..n_int {
                        if self.channel_state[n].lbrr_flags[i] == 0 {
                            continue;
                        }
                        if n_int == 2 && n == 0 {
                            let _ = silk_stereo_decode_pred(range_dec);
                            if self.channel_state[1].lbrr_flags[i] == 0 {
                                let _ = silk_stereo_decode_mid_only(range_dec);
                            }
                        }
                        // Conditional coding if the previous frame has a copy.
                        let cond_coding = if i > 0 && self.channel_state[n].lbrr_flags[i - 1] != 0
                        {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY
                        };
                        let state = &mut self.channel_state[n];
                        silk_decode_indices(state, range_dec, i as i32, 1, cond_coding);
                        // The pulses are not needed. The second frame buffer
                        // is free at this point and serves as scratch space.
                        silk_decode_pulses(
                            range_dec,
                            &mut self.w_silk_buf[1][..MAX_FRAME_LENGTH],
                            state.indices.signal_type as i32,
                            state.indices.quant_offset_type as i32,
                            state.frame_length,
                        );
                    }
                }
            }
        }

        // Stereo predictor and mid-only flag. An LBRR copy has its own, and
        // codes the mid-only flag only when the side channel has no copy.
        let mut ms_pred_q13 = [0i32; 2];
        let mut decode_only_middle = 0i32;
        if n_int == 2 {
            let frame = self.channel_state[0].n_frames_decoded as usize;
            if lost_flag == FLAG_DECODE_NORMAL
                || (lost_flag == FLAG_DECODE_LBRR && self.channel_state[0].lbrr_flags[frame] == 1)
            {
                ms_pred_q13 = silk_stereo_decode_pred(range_dec);
                if (lost_flag == FLAG_DECODE_NORMAL && self.channel_state[1].vad_flags[frame] == 0)
                    || (lost_flag == FLAG_DECODE_LBRR
                        && self.channel_state[1].lbrr_flags[frame] == 0)
                {
                    decode_only_middle = silk_stereo_decode_mid_only(range_dec) as i32;
                }
            } else {
                ms_pred_q13 = self.s_stereo.pred_prev_q13;
            }
        }

        // Reset the side channel's prediction memory for the first frame
        // with a coded side channel.
        if n_int == 2 && decode_only_middle == 0 && self.prev_decode_only_middle == 1 {
            let side = &mut self.channel_state[1];
            side.out_buf.fill(0);
            side.s_lpc_q14_buf.fill(0);
            side.lag_prev = 100;
            side.last_gain_index = 10;
            side.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            side.first_frame_after_reset = 1;
        }

        // A frame that is not decoded normally has a side channel if the
        // last frame had one, or if the side channel has an LBRR copy.
        let has_side = if lost_flag == FLAG_DECODE_NORMAL {
            decode_only_middle == 0
        } else {
            self.prev_decode_only_middle == 0
                || (n_int == 2
                    && lost_flag == FLAG_DECODE_LBRR
                    && self.channel_state[1].lbrr_flags
                        [self.channel_state[1].n_frames_decoded as usize]
                        == 1)
        };

        // Decode one frame per channel into w_silk_buf[n][2..2 + n_samples].
        let mut n_samples: i32 = 0;
        for n in 0..n_int {
            if n == 0 || has_side {
                // The mid channel's counter is already one ahead when the
                // side channel is decoded.
                let frame = self.channel_state[0].n_frames_decoded - n as i32;
                let cond_coding = if frame <= 0 {
                    // No previous frame in this packet.
                    CODE_INDEPENDENTLY
                } else if lost_flag == FLAG_DECODE_LBRR {
                    if self.channel_state[n].lbrr_flags[frame as usize - 1] != 0 {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    }
                } else if n > 0 && self.prev_decode_only_middle != 0 {
                    // A side frame was skipped in this packet: no LTP
                    // scaling is needed, the LTP state is well defined.
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                let ret = silk_decode_frame(
                    &mut self.channel_state[n],
                    range_dec,
                    &mut self.w_silk_buf[n][2..],
                    &mut n_samples,
                    lost_flag,
                    cond_coding,
                );
                if ret < 0 {
                    return Err("SILK decoder: frame decoding failed");
                }
            } else {
                self.w_silk_buf[n][2..2 + n_samples as usize].fill(0);
            }
            let state = &mut self.channel_state[n];
            state.n_frames_decoded = state.n_frames_decoded.saturating_add(1);
        }
        let n_samples = n_samples as usize;

        if n_api == 2 && n_int == 2 {
            // Mid/side to left/right.
            let (mid, side) = self.w_silk_buf.split_at_mut(1);
            silk_stereo_ms_to_lr(
                &mut self.s_stereo,
                &mut mid[0],
                &mut side[0],
                &ms_pred_q13,
                self.channel_state[0].fs_khz,
                n_samples as i32,
            );
        } else {
            // Buffering only: the frame starts with the last two samples of
            // the previous frame. With a stereo packet and mono output this
            // is the mid channel, and the side channel is not used.
            let mid = &mut self.w_silk_buf[0];
            mid[..2].copy_from_slice(&self.s_stereo.s_mid);
            self.s_stereo
                .s_mid
                .copy_from_slice(&mid[n_samples..n_samples + 2]);
        }

        if lost_flag == FLAG_PACKET_LOST {
            // libopus dec_API.c: "On packet loss, remove the gain clamping
            // to prevent having the energy bounce back if we lose packets
            // when the energy is going down". The first gain of the next
            // good frame may not be more than 16 steps below the previous
            // gain index; after a loss that index is stale, so it is put
            // back to its start value.
            for state in &mut self.channel_state[..n_int] {
                state.last_gain_index = 10;
            }
        } else {
            self.prev_decode_only_middle = decode_only_middle;
        }

        Ok(DecodedFrame {
            n_samples,
            stereo_to_mono,
        })
    }

    pub fn decode_bytes(&mut self, data: &[u8], output: &mut [i16], new_packet: bool) -> i32 {
        let mut range_dec = RangeCoder::new_decoder(data);
        let internal_rate = self.channel_state[0].fs_khz * 1000;
        let payload_ms = if self.channel_state[0].nb_subfr == 2 {
            10
        } else {
            20
        };
        self.decode(
            &mut range_dec,
            output,
            FLAG_DECODE_NORMAL,
            new_packet,
            payload_ms,
            internal_rate,
        )
    }

    /// Put the decoder back into its start state (libopus
    /// `silk_ResetDecoder`). The channel counts of the previous call stay,
    /// as in libopus.
    pub fn reset(&mut self) {
        // `decode()` reads the output rate from this field, so it survives.
        let fs_api_hz = self.channel_state[0].fs_api_hz;
        silk_reset_decoder(&mut self.channel_state[0]);
        silk_reset_decoder(&mut self.channel_state[1]);
        self.channel_state[0].fs_api_hz = fs_api_hz;
        self.prev_decode_only_middle = 0;
        self.s_stereo = StereoDecState::default();
    }

    pub fn frame_length(&self) -> i32 {
        self.channel_state[0].frame_length
    }

    pub fn sample_rate(&self) -> i32 {
        self.channel_state[0].fs_khz * 1000
    }
}

/// Check the values every decode call depends on. Returns the SILK rate in
/// kHz.
fn check_control(control: &SilkDecControl, lost_flag: i32) -> Result<i32, &'static str> {
    if !matches!(control.n_channels_api, 1 | 2) || !matches!(control.n_channels_internal, 1 | 2) {
        return Err("SILK decoder: channel count must be 1 or 2");
    }
    if !matches!(control.payload_size_ms, 0 | 10 | 20 | 40 | 60) {
        return Err("SILK decoder: invalid frame duration");
    }
    let fs_khz_dec = (control.internal_sample_rate >> 10) + 1;
    if !matches!(fs_khz_dec, 8 | 12 | 16) {
        return Err("SILK decoder: invalid sampling rate");
    }
    if !matches!(
        lost_flag,
        FLAG_DECODE_NORMAL | FLAG_PACKET_LOST | FLAG_DECODE_LBRR
    ) {
        return Err("SILK decoder: invalid lost flag");
    }
    Ok(fs_khz_dec)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let dec = SilkDecoder::new();
        assert_eq!(dec.n_channels_api, 1);
        assert_eq!(dec.n_channels_internal, 1);
    }

    #[test]
    fn test_decoder_init() {
        let mut dec = SilkDecoder::new();

        let ret = dec.init(16000, 1);
        assert_eq!(ret, 0);
        assert_eq!(dec.sample_rate(), 16000);
    }

    #[test]
    fn test_decoder_16khz() {
        let mut dec = SilkDecoder::new();
        let ret = dec.init(16000, 1);
        assert_eq!(ret, 0);
        assert_eq!(dec.frame_length(), 320);
    }

    /// Conceal one lost 20 ms stereo frame after a good frame whose mid-only
    /// flag was `prev_mid_only`, and return the decoder.
    fn conceal_after(prev_mid_only: i32) -> SilkDecoder {
        let mut dec = SilkDecoder::new();
        assert_eq!(dec.init(16000, 2), 0);
        dec.prev_decode_only_middle = prev_mid_only;
        let mut out = [0i16; 2 * 320];
        let mut rc = RangeCoder::new_decoder(&[]);
        let n = dec.decode(&mut rc, &mut out, FLAG_PACKET_LOST, true, 20, 16000);
        assert_eq!(n, 320);
        dec
    }

    /// libopus dec_API.c: a lost frame conceals the side channel only if
    /// the last good frame coded one, and leaves the mid-only flag alone.
    #[test]
    fn a_lost_frame_after_a_mid_only_frame_conceals_the_mid_only() {
        let dec = conceal_after(1);
        assert_eq!(dec.prev_decode_only_middle, 1);
        assert_eq!(dec.channel_state[0].loss_cnt, 1);
        assert_eq!(dec.channel_state[1].loss_cnt, 0, "side was concealed");
    }

    #[test]
    fn a_lost_frame_after_a_side_coded_frame_conceals_both() {
        let dec = conceal_after(0);
        assert_eq!(dec.prev_decode_only_middle, 0);
        assert_eq!(dec.channel_state[0].loss_cnt, 1);
        assert_eq!(dec.channel_state[1].loss_cnt, 1);
    }
}
