/* Entry points into the fixed-point build of libopus 1.6.1 for the tests
 * of opus1 (see build.rs and src/lib.rs).
 *
 * The build script compiles this file like the libopus files, with the
 * header that gives every libopus symbol the prefix `libopus_fixed_`. So
 * the libopus names below call the fixed-point build, not the float build
 * that the tests also link. The functions here keep their own names, which
 * begin with `fixed_ref_`.
 *
 * Several libopus functions take the whole SILK encoder state, whose layout
 * the Rust side does not know. The functions here take the fields those
 * functions read as plain arguments, fill a zeroed state with them and
 * call the libopus function. */

#include <stdlib.h>
#include <string.h>
#include "opus.h"
#include "opus_private.h"
#include "main_FIX.h"

/* ---- Encoder ---- */

OpusEncoder *fixed_ref_encoder_create(opus_int32 fs, int channels, int application, int *error)
{
    return opus_encoder_create(fs, channels, application, error);
}

void fixed_ref_encoder_destroy(OpusEncoder *st)
{
    opus_encoder_destroy(st);
}

/* A control that takes one integer, such as OPUS_SET_BITRATE_REQUEST. */
int fixed_ref_encoder_set(OpusEncoder *st, int request, opus_int32 value)
{
    return opus_encoder_ctl(st, request, value);
}

/* A control that returns one integer, such as OPUS_GET_IN_DTX_REQUEST. */
int fixed_ref_encoder_get(OpusEncoder *st, int request, opus_int32 *value)
{
    return opus_encoder_ctl(st, request, value);
}

opus_int32 fixed_ref_encode(OpusEncoder *st, const opus_int16 *pcm, int frame_size,
                            unsigned char *data, opus_int32 max_data_bytes)
{
    return opus_encode(st, pcm, frame_size, data, max_data_bytes);
}

/* ---- SILK analysis ---- */

void fixed_ref_burg_modified(opus_int32 *res_nrg, int *res_nrg_Q, opus_int32 *A_Q16,
                             const opus_int16 *x, opus_int32 minInvGain_Q30, int subfr_length,
                             int nb_subfr, int D)
{
    silk_burg_modified_c(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, 0);
}

int fixed_ref_pitch_analysis_core(const opus_int16 *frame, int *pitch_out, opus_int16 *lagIndex,
                                  opus_int8 *contourIndex, int *LTPCorr_Q15, int prevLag,
                                  opus_int32 search_thres1_Q16, int search_thres2_Q13, int Fs_kHz,
                                  int complexity, int nb_subfr)
{
    return silk_pitch_analysis_core(frame, pitch_out, lagIndex, contourIndex, LTPCorr_Q15, prevLag,
                                    search_thres1_Q16, search_thres2_Q13, Fs_kHz, complexity, nb_subfr, 0);
}

/* `silk_HP_variable_cutoff`: returns the new `variable_HP_smth1_Q15`. */
opus_int32 fixed_ref_hp_variable_cutoff(int fs_kHz, int prevSignalType, int prevLag,
                                        int input_quality_band0_Q15, int speech_activity_Q8,
                                        opus_int32 variable_HP_smth1_Q15)
{
    opus_int32 result;
    silk_encoder_state_FIX *st = calloc(1, sizeof(*st));
    if (!st) abort();
    st->sCmn.fs_kHz = fs_kHz;
    st->sCmn.prevSignalType = (opus_int8)prevSignalType;
    st->sCmn.prevLag = prevLag;
    st->sCmn.input_quality_bands_Q15[0] = input_quality_band0_Q15;
    st->sCmn.speech_activity_Q8 = speech_activity_Q8;
    st->sCmn.variable_HP_smth1_Q15 = variable_HP_smth1_Q15;
    silk_HP_variable_cutoff(st);
    result = st->sCmn.variable_HP_smth1_Q15;
    free(st);
    return result;
}

/* The fields of the encoder state and of the control structure that
 * `silk_control_audio_bandwidth` reads or writes. */
typedef struct {
    int fs_kHz;
    int allow_bandwidth_switch;
    opus_int32 API_fs_Hz;
    int maxInternal_fs_Hz;
    int minInternal_fs_Hz;
    int desiredInternal_fs_Hz;
    opus_int32 In_LP_State[2];
    opus_int32 transition_frame_no;
    int mode;
    opus_int32 saved_fs_kHz;
    int opusCanSwitch;
    int payloadSize_ms;
    int maxBits;
    int switchReady;
} fixed_ref_bandwidth;

/* `silk_control_audio_bandwidth` (silk/control_audio_bandwidth.c): returns
 * the internal rate in kHz and writes back the low-pass state, `maxBits`
 * and `switchReady`. */
int fixed_ref_control_audio_bandwidth(fixed_ref_bandwidth *b)
{
    int fs_kHz;
    silk_EncControlStruct control;
    silk_encoder_state *st = calloc(1, sizeof(*st));
    if (!st) abort();
    st->fs_kHz = b->fs_kHz;
    st->allow_bandwidth_switch = b->allow_bandwidth_switch;
    st->API_fs_Hz = b->API_fs_Hz;
    st->maxInternal_fs_Hz = b->maxInternal_fs_Hz;
    st->minInternal_fs_Hz = b->minInternal_fs_Hz;
    st->desiredInternal_fs_Hz = b->desiredInternal_fs_Hz;
    st->sLP.In_LP_State[0] = b->In_LP_State[0];
    st->sLP.In_LP_State[1] = b->In_LP_State[1];
    st->sLP.transition_frame_no = b->transition_frame_no;
    st->sLP.mode = b->mode;
    st->sLP.saved_fs_kHz = b->saved_fs_kHz;
    memset(&control, 0, sizeof(control));
    control.opusCanSwitch = b->opusCanSwitch;
    control.payloadSize_ms = b->payloadSize_ms;
    control.maxBits = b->maxBits;
    control.switchReady = b->switchReady;
    fs_kHz = silk_control_audio_bandwidth(st, &control);
    b->In_LP_State[0] = st->sLP.In_LP_State[0];
    b->In_LP_State[1] = st->sLP.In_LP_State[1];
    b->transition_frame_no = st->sLP.transition_frame_no;
    b->mode = st->sLP.mode;
    b->saved_fs_kHz = st->sLP.saved_fs_kHz;
    b->maxBits = control.maxBits;
    b->switchReady = control.switchReady;
    free(st);
    return fs_kHz;
}

/* ---- SILK quantization ---- */

/* The fields of the encoder state and of the side information that the
 * quantizers read. `Seed` is also written: the delayed-decision quantizer
 * returns the seed of the path it chose. */
typedef struct {
    int nb_subfr;
    int subfr_length;
    int frame_length;
    int ltp_mem_length;
    int predictLPCOrder;
    int shapingLPCOrder;
    int nStatesDelayedDecision;
    int warping_Q16;
    int signalType;
    int quantOffsetType;
    int NLSFInterpCoef_Q2;
    int Seed;
} fixed_ref_nsq_frame;

int fixed_ref_nsq_state_size(void)
{
    return (int)sizeof(silk_nsq_state);
}

/* `silk_NSQ_c` (del_dec 0) or `silk_NSQ_del_dec_c` (del_dec 1) for one frame. */
void fixed_ref_nsq(int del_dec, fixed_ref_nsq_frame *f, silk_nsq_state *NSQ, const opus_int16 *x16,
                   opus_int8 *pulses, const opus_int16 *PredCoef_Q12, const opus_int16 *LTPCoef_Q14,
                   const opus_int16 *AR_Q13, const int *HarmShapeGain_Q14, const int *Tilt_Q14,
                   const opus_int32 *LF_shp_Q14, const opus_int32 *Gains_Q16, const int *pitchL,
                   int Lambda_Q10, int LTP_scale_Q14)
{
    SideInfoIndices indices;
    silk_encoder_state *st = calloc(1, sizeof(*st));
    if (!st) abort();
    st->nb_subfr = f->nb_subfr;
    st->subfr_length = f->subfr_length;
    st->frame_length = f->frame_length;
    st->ltp_mem_length = f->ltp_mem_length;
    st->predictLPCOrder = f->predictLPCOrder;
    st->shapingLPCOrder = f->shapingLPCOrder;
    st->nStatesDelayedDecision = f->nStatesDelayedDecision;
    st->warping_Q16 = f->warping_Q16;
    memset(&indices, 0, sizeof(indices));
    indices.signalType = (opus_int8)f->signalType;
    indices.quantOffsetType = (opus_int8)f->quantOffsetType;
    indices.NLSFInterpCoef_Q2 = (opus_int8)f->NLSFInterpCoef_Q2;
    indices.Seed = (opus_int8)f->Seed;
    if (del_dec) {
        silk_NSQ_del_dec_c(st, NSQ, &indices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13,
                           HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,
                           LTP_scale_Q14);
    } else {
        silk_NSQ_c(st, NSQ, &indices, x16, pulses, PredCoef_Q12, LTPCoef_Q14, AR_Q13,
                   HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,
                   LTP_scale_Q14);
    }
    f->Seed = indices.Seed;
    free(st);
}

/* `silk_process_NLSFs`. The codebook is the one `silk_control_encoder`
 * picks for the rate (control_codec.c). `NLSFInterpCoef_Q2` is read and
 * may be changed; `PredCoef_Q12` is [2][MAX_LPC_ORDER]; `NLSFIndices` gets
 * MAX_LPC_ORDER + 1 values. */
void fixed_ref_process_nlsfs(int fs_kHz, int predictLPCOrder, int nb_subfr, int speech_activity_Q8,
                             int NLSF_MSVQ_Survivors, int signalType, int useInterpolatedNLSFs,
                             int *NLSFInterpCoef_Q2, opus_int16 *PredCoef_Q12, opus_int16 *pNLSF_Q15,
                             const opus_int16 *prev_NLSFq_Q15, opus_int8 *NLSFIndices)
{
    silk_encoder_state *st = calloc(1, sizeof(*st));
    if (!st) abort();
    st->fs_kHz = fs_kHz;
    st->psNLSF_CB = fs_kHz == 16 ? &silk_NLSF_CB_WB : &silk_NLSF_CB_NB_MB;
    st->predictLPCOrder = predictLPCOrder;
    st->nb_subfr = nb_subfr;
    st->speech_activity_Q8 = speech_activity_Q8;
    st->NLSF_MSVQ_Survivors = NLSF_MSVQ_Survivors;
    st->useInterpolatedNLSFs = useInterpolatedNLSFs;
    st->indices.signalType = (opus_int8)signalType;
    st->indices.NLSFInterpCoef_Q2 = (opus_int8)*NLSFInterpCoef_Q2;
    silk_process_NLSFs(st, (opus_int16 (*)[MAX_LPC_ORDER])PredCoef_Q12, pNLSF_Q15, prev_NLSFq_Q15);
    *NLSFInterpCoef_Q2 = st->indices.NLSFInterpCoef_Q2;
    memcpy(NLSFIndices, st->indices.NLSFIndices, sizeof(st->indices.NLSFIndices));
    free(st);
}
