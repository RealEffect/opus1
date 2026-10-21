//! The RDOVAE decoder of DRED (libopus `dnn/dred_rdovae_dec.c`): it turns
//! the initial state and the latent vectors of a packet back into features.
//!
//! The network has an initial step and one step per latent vector:
//!
//! - The initial step (`init_states`) makes the states of the five GRUs
//!   (gated recurrent units) from the decoded initial state vector, through
//!   two dense layers.
//! - Each step (`decode_qframe`) runs a dense layer, then five stages of a
//!   GRU, a GLU (gated linear unit), a dense layer and a 1-D convolution.
//!   Every stage reads all outputs before it. A final linear layer gives
//!   four frames of 20 features, newest first.
//!
//! The memories of the convolutions start at zero and are kept from one
//! step to the next. (libopus clears them in the first step of a call with
//! `conv1_cond_init`, after `memset` has already cleared them.)

use crate::dnn::Activation;
use crate::dnn::arch::DnnArch;
use crate::dnn::models::dred_rdovae_dec::{
    DEC_CONV1_IN_SIZE, DEC_CONV1_OUT_SIZE, DEC_CONV1_STATE_SIZE, DEC_DENSE1_OUT_SIZE, DEC_GRU1_OUT_SIZE,
    DEC_GRU1_STATE_SIZE, DEC_GRU_INIT_OUT_SIZE, DEC_HIDDEN_INIT_OUT_SIZE, RdovaeDec,
};
use crate::dnn::models::dred_rdovae_stats::{DRED_LATENT_DIM, DRED_MAX_CONV_INPUTS, DRED_NUM_FEATURES};
use crate::dnn::nnet::{LinearLayer, compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu};

/// The number of GRU, GLU and convolution stages.
const STAGES: usize = 5;

/// The size of the buffer that collects the outputs of all layers of a
/// step: the first dense layer, then a GLU and a convolution per stage.
const BUFFER_SIZE: usize = DEC_DENSE1_OUT_SIZE + STAGES * (DEC_GRU1_OUT_SIZE + DEC_CONV1_OUT_SIZE);

// The five stages have the same sizes in libopus 1.6.1; `decode_qframe`
// relies on it.
const _: () = {
    use crate::dnn::models::dred_rdovae_dec::*;
    assert!(DEC_GRU2_STATE_SIZE == DEC_GRU1_STATE_SIZE && DEC_GRU5_STATE_SIZE == DEC_GRU1_STATE_SIZE);
    assert!(DEC_GRU3_STATE_SIZE == DEC_GRU1_STATE_SIZE && DEC_GRU4_STATE_SIZE == DEC_GRU1_STATE_SIZE);
    assert!(DEC_GRU1_OUT_SIZE == DEC_GRU1_STATE_SIZE);
    assert!(DEC_GRU2_OUT_SIZE == DEC_GRU1_OUT_SIZE && DEC_GRU5_OUT_SIZE == DEC_GRU1_OUT_SIZE);
    assert!(DEC_CONV2_OUT_SIZE == DEC_CONV1_OUT_SIZE && DEC_CONV5_OUT_SIZE == DEC_CONV1_OUT_SIZE);
    assert!(DEC_CONV3_IN_SIZE == DEC_CONV1_IN_SIZE && DEC_CONV5_STATE_SIZE == DEC_CONV1_STATE_SIZE);
    assert!(DEC_GRU_INIT_OUT_SIZE == STAGES * DEC_GRU1_STATE_SIZE);
    assert!(DEC_OUTPUT_OUT_SIZE == 4 * DRED_NUM_FEATURES);
};

/// libopus `RDOVAEDecState`: the GRU states and the convolution memories.
struct RdovaeDecState {
    gru_state: [[f32; DEC_GRU1_STATE_SIZE]; STAGES],
    conv_state: [[f32; DEC_CONV1_STATE_SIZE]; STAGES],
}

/// The layers of one stage.
struct Stage<'a, 'w> {
    gru_input: &'a LinearLayer<'w>,
    gru_recurrent: &'a LinearLayer<'w>,
    glu: &'a LinearLayer<'w>,
    conv_dense: &'a LinearLayer<'w>,
    conv: &'a LinearLayer<'w>,
}

fn stages<'a, 'w>(m: &'a RdovaeDec<'w>) -> [Stage<'a, 'w>; STAGES] {
    [
        Stage {
            gru_input: &m.dec_gru1_input,
            gru_recurrent: &m.dec_gru1_recurrent,
            glu: &m.dec_glu1,
            conv_dense: &m.dec_conv_dense1,
            conv: &m.dec_conv1,
        },
        Stage {
            gru_input: &m.dec_gru2_input,
            gru_recurrent: &m.dec_gru2_recurrent,
            glu: &m.dec_glu2,
            conv_dense: &m.dec_conv_dense2,
            conv: &m.dec_conv2,
        },
        Stage {
            gru_input: &m.dec_gru3_input,
            gru_recurrent: &m.dec_gru3_recurrent,
            glu: &m.dec_glu3,
            conv_dense: &m.dec_conv_dense3,
            conv: &m.dec_conv3,
        },
        Stage {
            gru_input: &m.dec_gru4_input,
            gru_recurrent: &m.dec_gru4_recurrent,
            glu: &m.dec_glu4,
            conv_dense: &m.dec_conv_dense4,
            conv: &m.dec_conv4,
        },
        Stage {
            gru_input: &m.dec_gru5_input,
            gru_recurrent: &m.dec_gru5_recurrent,
            glu: &m.dec_glu5,
            conv_dense: &m.dec_conv_dense5,
            conv: &m.dec_conv5,
        },
    ]
}

impl RdovaeDecState {
    /// libopus `dred_rdovae_dec_init_states`: the GRU states from the
    /// initial state vector. The convolution memories start at zero.
    fn new(model: &RdovaeDec, initial_state: &[f32], arch: DnnArch) -> Self {
        let mut hidden = [0.0f32; DEC_HIDDEN_INIT_OUT_SIZE];
        let mut state_init = [0.0f32; DEC_GRU_INIT_OUT_SIZE];
        compute_generic_dense(&model.dec_hidden_init, &mut hidden, initial_state, Activation::Tanh, arch);
        compute_generic_dense(&model.dec_gru_init, &mut state_init, &hidden, Activation::Tanh, arch);
        let mut st = RdovaeDecState {
            gru_state: [[0.0; DEC_GRU1_STATE_SIZE]; STAGES],
            conv_state: [[0.0; DEC_CONV1_STATE_SIZE]; STAGES],
        };
        let (chunks, _) = state_init.as_chunks::<DEC_GRU1_STATE_SIZE>();
        st.gru_state.copy_from_slice(chunks);
        st
    }

    /// libopus `dred_rdovae_decode_qframe`: one latent vector (with its
    /// quantizer level as the last value) to four frames of features,
    /// written to `qframe`, newest first.
    fn decode_qframe(&mut self, model: &RdovaeDec, qframe: &mut [f32], input: &[f32], arch: DnnArch) {
        let mut buffer = [0.0f32; BUFFER_SIZE];
        let mut conv_tmp = [0.0f32; DRED_MAX_CONV_INPUTS];
        compute_generic_dense(&model.dec_dense1, &mut buffer, input, Activation::Tanh, arch);
        let mut used = DEC_DENSE1_OUT_SIZE;
        for (k, stage) in stages(model).iter().enumerate() {
            let (head, tail) = buffer.split_at_mut(used);
            compute_generic_gru(stage.gru_input, stage.gru_recurrent, &mut self.gru_state[k], head, arch);
            compute_glu(stage.glu, &mut tail[..DEC_GRU1_OUT_SIZE], &self.gru_state[k], arch);
            used += DEC_GRU1_OUT_SIZE;

            let (head, tail) = buffer.split_at_mut(used);
            compute_generic_dense(stage.conv_dense, &mut conv_tmp, head, Activation::Tanh, arch);
            compute_generic_conv1d(
                stage.conv,
                &mut tail[..DEC_CONV1_OUT_SIZE],
                &mut self.conv_state[k],
                &conv_tmp[..DEC_CONV1_IN_SIZE],
                Activation::Tanh,
                arch,
            );
            used += DEC_CONV1_OUT_SIZE;
        }
        compute_generic_dense(&model.dec_output, qframe, &buffer[..used], Activation::Linear, arch);
    }
}

/// libopus `DRED_rdovae_decode_all`: decodes `nb_latents` latent vectors
/// into `features`, four frames of `DRED_NUM_FEATURES` per vector.
///
/// `latents` holds `DRED_LATENT_DIM + 1` values per vector, `state` the
/// initial state. Vector `i` writes frames `4i` to `4i + 3`.
pub fn decode_all(
    model: &RdovaeDec,
    features: &mut [f32],
    state: &[f32],
    latents: &[f32],
    nb_latents: usize,
    arch: DnnArch,
) {
    let mut dec = RdovaeDecState::new(model, state, arch);
    for i in 0..nb_latents {
        dec.decode_qframe(
            model,
            &mut features[4 * i * DRED_NUM_FEATURES..(4 * i + 4) * DRED_NUM_FEATURES],
            &latents[i * (DRED_LATENT_DIM + 1)..(i + 1) * (DRED_LATENT_DIM + 1)],
            arch,
        );
    }
}
