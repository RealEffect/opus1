//! The RDOVAE encoder of DRED (libopus `dnn/dred_rdovae_enc.c`): it turns
//! two 10 ms frames of features into one latent vector and one initial
//! state vector.
//!
//! Each step runs a dense layer, then five stages of a GRU (gated
//! recurrent unit), a dense layer and a 1-D convolution. Every stage reads
//! all outputs before it. Two final layers read all outputs: a linear one
//! gives the latent vector, and two more dense layers give the initial
//! state.
//!
//! The memories of the convolutions start at zero and are kept from one
//! step to the next. (libopus clears the first one in the first step after
//! a reset with `conv1_cond_init`, after `memset` has already cleared it.)

use crate::dnn::Activation;
use crate::dnn::arch::DnnArch;
use crate::dnn::models::dred_rdovae_enc::{
    ENC_CONV1_OUT_SIZE, ENC_CONV1_STATE_SIZE, ENC_CONV2_STATE_SIZE, ENC_DENSE1_OUT_SIZE, ENC_GRU1_OUT_SIZE,
    ENC_GRU1_STATE_SIZE, GDENSE1_OUT_SIZE, RdovaeEnc,
};
use crate::dnn::models::dred_rdovae_stats::{
    DRED_LATENT_DIM, DRED_MAX_CONV_INPUTS, DRED_NUM_FEATURES, DRED_PADDED_LATENT_DIM, DRED_PADDED_STATE_DIM,
    DRED_STATE_DIM,
};
use crate::dnn::nnet::{
    LinearLayer, compute_generic_conv1d, compute_generic_conv1d_dilation, compute_generic_dense, compute_generic_gru,
};

/// The number of GRU and convolution stages.
const STAGES: usize = 5;

/// The size of the buffer that collects the outputs of all layers of a
/// step: the first dense layer, then a GRU and a convolution per stage.
const BUFFER_SIZE: usize = ENC_DENSE1_OUT_SIZE + STAGES * (ENC_GRU1_OUT_SIZE + ENC_CONV1_OUT_SIZE);

// The five stages have the same sizes in libopus 1.6.1, and the last four
// convolutions have a dilation of 2; `encode_dframe` relies on it.
const _: () = {
    use crate::dnn::models::dred_rdovae_enc::*;
    assert!(ENC_GRU2_STATE_SIZE == ENC_GRU1_STATE_SIZE && ENC_GRU5_STATE_SIZE == ENC_GRU1_STATE_SIZE);
    assert!(ENC_GRU3_STATE_SIZE == ENC_GRU1_STATE_SIZE && ENC_GRU4_STATE_SIZE == ENC_GRU1_STATE_SIZE);
    assert!(ENC_GRU1_OUT_SIZE == ENC_GRU1_STATE_SIZE);
    assert!(ENC_GRU2_OUT_SIZE == ENC_GRU1_OUT_SIZE && ENC_GRU5_OUT_SIZE == ENC_GRU1_OUT_SIZE);
    assert!(ENC_GRU3_OUT_SIZE == ENC_GRU1_OUT_SIZE && ENC_GRU4_OUT_SIZE == ENC_GRU1_OUT_SIZE);
    assert!(ENC_CONV2_OUT_SIZE == ENC_CONV1_OUT_SIZE && ENC_CONV5_OUT_SIZE == ENC_CONV1_OUT_SIZE);
    assert!(ENC_CONV3_OUT_SIZE == ENC_CONV1_OUT_SIZE && ENC_CONV4_OUT_SIZE == ENC_CONV1_OUT_SIZE);
    assert!(ENC_CONV2_IN_SIZE == ENC_CONV1_IN_SIZE && ENC_CONV5_IN_SIZE == ENC_CONV1_IN_SIZE);
    assert!(ENC_CONV3_IN_SIZE == ENC_CONV1_IN_SIZE && ENC_CONV4_IN_SIZE == ENC_CONV1_IN_SIZE);
    assert!(ENC_CONV3_STATE_SIZE == ENC_CONV2_STATE_SIZE && ENC_CONV5_STATE_SIZE == ENC_CONV2_STATE_SIZE);
    assert!(ENC_CONV4_STATE_SIZE == ENC_CONV2_STATE_SIZE);
    // C passes `ENC_CONVk_OUT_SIZE` as the input size of each convolution,
    // which reads the output of the dense layer before it.
    assert!(ENC_CONV_DENSE1_OUT_SIZE == ENC_CONV1_OUT_SIZE && ENC_CONV1_IN_SIZE == ENC_CONV1_OUT_SIZE);
    assert!(ENC_CONV1_OUT_SIZE <= crate::dnn::models::dred_rdovae_stats::DRED_MAX_CONV_INPUTS);
    assert!(ENC_ZDENSE_OUT_SIZE == crate::dnn::models::dred_rdovae_stats::DRED_PADDED_LATENT_DIM);
    assert!(GDENSE2_OUT_SIZE == crate::dnn::models::dred_rdovae_stats::DRED_PADDED_STATE_DIM);
};

/// libopus `RDOVAEEncState` (`dnn/dred_rdovae_enc.h`): the GRU states and
/// the convolution memories. The first convolution has a dilation of 1,
/// the other four of 2, so their memories are twice as long.
#[derive(Clone)]
pub(crate) struct RdovaeEncState {
    gru_state: [[f32; ENC_GRU1_STATE_SIZE]; STAGES],
    conv1_state: [f32; ENC_CONV1_STATE_SIZE],
    conv_state: [[f32; 2 * ENC_CONV2_STATE_SIZE]; STAGES - 1],
}

/// The layers of one stage.
struct Stage<'a, 'w> {
    gru_input: &'a LinearLayer<'w>,
    gru_recurrent: &'a LinearLayer<'w>,
    conv_dense: &'a LinearLayer<'w>,
    conv: &'a LinearLayer<'w>,
}

fn stages<'a, 'w>(m: &'a RdovaeEnc<'w>) -> [Stage<'a, 'w>; STAGES] {
    [
        Stage {
            gru_input: &m.enc_gru1_input,
            gru_recurrent: &m.enc_gru1_recurrent,
            conv_dense: &m.enc_conv_dense1,
            conv: &m.enc_conv1,
        },
        Stage {
            gru_input: &m.enc_gru2_input,
            gru_recurrent: &m.enc_gru2_recurrent,
            conv_dense: &m.enc_conv_dense2,
            conv: &m.enc_conv2,
        },
        Stage {
            gru_input: &m.enc_gru3_input,
            gru_recurrent: &m.enc_gru3_recurrent,
            conv_dense: &m.enc_conv_dense3,
            conv: &m.enc_conv3,
        },
        Stage {
            gru_input: &m.enc_gru4_input,
            gru_recurrent: &m.enc_gru4_recurrent,
            conv_dense: &m.enc_conv_dense4,
            conv: &m.enc_conv4,
        },
        Stage {
            gru_input: &m.enc_gru5_input,
            gru_recurrent: &m.enc_gru5_recurrent,
            conv_dense: &m.enc_conv_dense5,
            conv: &m.enc_conv5,
        },
    ]
}

impl RdovaeEncState {
    /// A cleared state (libopus `DRED_rdovae_init_encoder`).
    pub(crate) const fn new() -> Self {
        RdovaeEncState {
            gru_state: [[0.0; ENC_GRU1_STATE_SIZE]; STAGES],
            conv1_state: [0.0; ENC_CONV1_STATE_SIZE],
            conv_state: [[0.0; 2 * ENC_CONV2_STATE_SIZE]; STAGES - 1],
        }
    }

    /// libopus `dred_rdovae_encode_dframe`: one step of the encoder.
    /// `input` holds the `DRED_NUM_FEATURES` features of two 10 ms frames,
    /// older first. Writes `DRED_LATENT_DIM` values to `latents` and
    /// `DRED_STATE_DIM` values to `initial_state`.
    pub(crate) fn encode_dframe(
        &mut self,
        model: &RdovaeEnc,
        latents: &mut [f32],
        initial_state: &mut [f32],
        input: &[f32; 2 * DRED_NUM_FEATURES],
        arch: DnnArch,
    ) {
        let mut buffer = [0.0f32; BUFFER_SIZE];
        let mut conv_tmp = [0.0f32; DRED_MAX_CONV_INPUTS];
        compute_generic_dense(&model.enc_dense1, &mut buffer, input, Activation::Tanh, arch);
        let mut used = ENC_DENSE1_OUT_SIZE;
        for (k, stage) in stages(model).iter().enumerate() {
            compute_generic_gru(stage.gru_input, stage.gru_recurrent, &mut self.gru_state[k], &buffer[..used], arch);
            buffer[used..used + ENC_GRU1_OUT_SIZE].copy_from_slice(&self.gru_state[k]);
            used += ENC_GRU1_OUT_SIZE;

            let (head, tail) = buffer.split_at_mut(used);
            compute_generic_dense(stage.conv_dense, &mut conv_tmp, head, Activation::Tanh, arch);
            let out = &mut tail[..ENC_CONV1_OUT_SIZE];
            let conv_in = &conv_tmp[..ENC_CONV1_OUT_SIZE];
            if k == 0 {
                compute_generic_conv1d(stage.conv, out, &mut self.conv1_state, conv_in, Activation::Tanh, arch);
            } else {
                let mem = &mut self.conv_state[k - 1];
                compute_generic_conv1d_dilation(stage.conv, out, mem, conv_in, 2, Activation::Tanh, arch);
            }
            used += ENC_CONV1_OUT_SIZE;
        }
        debug_assert_eq!(used, BUFFER_SIZE);

        let mut padded_latents = [0.0f32; DRED_PADDED_LATENT_DIM];
        compute_generic_dense(&model.enc_zdense, &mut padded_latents, &buffer, Activation::Linear, arch);
        latents[..DRED_LATENT_DIM].copy_from_slice(&padded_latents[..DRED_LATENT_DIM]);

        // "next, calculate initial state"
        let mut state_hidden = [0.0f32; GDENSE1_OUT_SIZE];
        let mut padded_state = [0.0f32; DRED_PADDED_STATE_DIM];
        compute_generic_dense(&model.gdense1, &mut state_hidden, &buffer, Activation::Tanh, arch);
        compute_generic_dense(&model.gdense2, &mut padded_state, &state_hidden, Activation::Linear, arch);
        initial_state[..DRED_STATE_DIM].copy_from_slice(&padded_state[..DRED_STATE_DIM]);
    }
}
