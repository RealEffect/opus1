/* A small C library for the tests of opus1's OSCE port (tests/dnn_osce_libopus.rs).
 *
 * libopus's OSCE entry points take the whole SILK decoder state
 * (silk_decoder_state, silk_decoder_control). This shim keeps one of each,
 * built with libopus's own headers, fills the fields that OSCE reads from
 * flat arguments, and calls:
 *
 *   - osce_enhance_frame, osce_reset            (LACE, NoLACE, reset rules)
 *   - osce_calculate_features                   (features alone)
 *   - osce_bwe, osce_bwe_reset                  (bandwidth extension)
 *   - osce_bwe_calculate_features               (BWE features alone)
 *   - osce_cross_fade_10ms, osce_bwe_cross_fade_10ms, compute_overlap_window
 *   - adacomb_process_frame, adaconv_process_frame, adashape_process_frame
 *     with the layers of the built-in models
 *
 * Build it with scripts/dnn/build-osce-shim.sh, which compiles it with the
 * defines and flags of a reference build (its build-flags.txt) and links it
 * with that build's libopus.a into a shared library. The test loads that
 * library at run time.
 *
 * osce_load_models does not set model->loaded; silk_LoadOSCEModels does.
 * osce_shim_create sets it, or osce_enhance_frame would silently fall back
 * to method NONE.
 */

#include <stdlib.h>
#include <string.h>

#include "opus_types.h"
#include "structs.h"
#include "osce.h"
#include "osce_features.h"
#include "osce_structs.h"
#include "nndsp.h"
#include "cpu_support.h"

#if !defined(ENABLE_OSCE) || !defined(ENABLE_OSCE_BWE)
#error "the OSCE test shim needs a build with ENABLE_OSCE and ENABLE_OSCE_BWE"
#endif

#define SHIM_EXPORT __attribute__((visibility("default")))

/* The version of this interface; the test checks it. */
#define OSCE_SHIM_VERSION 1

typedef struct {
    OSCEModel model;
    int arch;
    /* osce_enhance_frame and osce_bwe run on these. */
    silk_decoder_state dec;
    silk_decoder_control ctrl;
    /* osce_calculate_features alone runs on this one. */
    silk_decoder_state fdec;
    /* osce_bwe_calculate_features alone. */
    OSCEBWEFeatureState bwe_features;
    /* The blocks of nndsp.c. */
    AdaCombState comb[2];
    AdaConvState conv[5];
    AdaShapeState shape[3];
} OsceShim;

SHIM_EXPORT int osce_shim_version(void) { return OSCE_SHIM_VERSION; }

/* Creates the shim with the built-in models. arch < 0 uses
 * opus_select_arch(). The OSCE state is that of a new decoder:
 * osce_reset(OSCE_DEFAULT_METHOD) on a cleared state; the BWE state is all
 * zero. */
SHIM_EXPORT void *osce_shim_create(int arch)
{
    OsceShim *s = calloc(1, sizeof(*s));
    if (s == NULL) return NULL;
    if (osce_load_models(&s->model, NULL, 0) != 0) {
        free(s);
        return NULL;
    }
    s->model.loaded = 1;
    s->arch = arch < 0 ? opus_select_arch() : arch;
    osce_reset(&s->dec.osce, OSCE_DEFAULT_METHOD);
    osce_reset(&s->fdec.osce, OSCE_DEFAULT_METHOD);
    return s;
}

SHIM_EXPORT void osce_shim_destroy(void *h) { free(h); }

SHIM_EXPORT int osce_shim_arch(void *h) { return ((OsceShim *)h)->arch; }

SHIM_EXPORT void osce_shim_set_loaded(void *h, int loaded) { ((OsceShim *)h)->model.loaded = loaded; }

SHIM_EXPORT void osce_shim_reset(void *h, int method) { osce_reset(&((OsceShim *)h)->dec.osce, method); }

SHIM_EXPORT int osce_shim_method(void *h) { return ((OsceShim *)h)->dec.osce.method; }

SHIM_EXPORT int osce_shim_bypass_count(void *h) { return ((OsceShim *)h)->dec.osce.features.reset; }

static void fill(silk_decoder_state *dec, silk_decoder_control *ctrl, int fs_khz, int nb_subfr, int lpc_order,
                 int signal_type, const opus_int16 *pred_coef_q12, const int *pitch_l, const opus_int16 *ltp_coef_q14,
                 const opus_int32 *gains_q16)
{
    int k;
    dec->fs_kHz = fs_khz;
    dec->nb_subfr = nb_subfr;
    dec->LPC_order = lpc_order;
    dec->indices.signalType = (opus_int8)signal_type;
    memcpy(ctrl->PredCoef_Q12, pred_coef_q12, sizeof(ctrl->PredCoef_Q12));
    for (k = 0; k < MAX_NB_SUBFR; k++) {
        ctrl->pitchL[k] = pitch_l[k];
        ctrl->Gains_Q16[k] = gains_q16[k];
    }
    memcpy(ctrl->LTPCoef_Q14, ltp_coef_q14, sizeof(ctrl->LTPCoef_Q14));
}

/* osce_enhance_frame on xq (nb_subfr * 80 samples), in place.
 * pred_coef_q12: 2 x 16 values; pitch_l, gains_q16: 4; ltp_coef_q14: 20. */
SHIM_EXPORT void osce_shim_enhance(void *h, int fs_khz, int nb_subfr, int lpc_order, int signal_type,
                                   const opus_int16 *pred_coef_q12, const int *pitch_l,
                                   const opus_int16 *ltp_coef_q14, const opus_int32 *gains_q16, opus_int16 *xq,
                                   int num_bits)
{
    OsceShim *s = h;
    fill(&s->dec, &s->ctrl, fs_khz, nb_subfr, lpc_order, signal_type, pred_coef_q12, pitch_l, ltp_coef_q14,
         gains_q16);
    osce_enhance_frame(&s->model, &s->dec, &s->ctrl, xq, num_bits, s->arch);
}

/* osce_calculate_features alone, on its own feature state. */
SHIM_EXPORT void osce_shim_features(void *h, int lpc_order, int signal_type, const opus_int16 *pred_coef_q12,
                                    const int *pitch_l, const opus_int16 *ltp_coef_q14, const opus_int32 *gains_q16,
                                    const opus_int16 *xq, int num_bits, float *features, float *numbits,
                                    int *periods)
{
    OsceShim *s = h;
    silk_decoder_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    fill(&s->fdec, &ctrl, 16, 4, lpc_order, signal_type, pred_coef_q12, pitch_l, ltp_coef_q14, gains_q16);
    osce_calculate_features(&s->fdec, &ctrl, features, numbits, periods, xq, num_bits);
}

SHIM_EXPORT void osce_shim_features_reset(void *h) { memset(&((OsceShim *)h)->fdec.osce.features, 0, sizeof(OSCEFeatureState)); }

SHIM_EXPORT void osce_shim_bwe_reset(void *h) { osce_bwe_reset(&((OsceShim *)h)->dec.osce_bwe); }

/* The BWE state of a new decoder: all zero (silk_init_decoder). */
SHIM_EXPORT void osce_shim_bwe_clear(void *h) { memset(&((OsceShim *)h)->dec.osce_bwe, 0, sizeof(silk_OSCE_BWE_struct)); }

/* osce_bwe: xq16 holds len (160 or 320) samples, xq48 receives 3 * len. */
SHIM_EXPORT void osce_shim_bwe(void *h, opus_int16 *xq48, const opus_int16 *xq16, int len)
{
    OsceShim *s = h;
    opus_int16 in[320];
    memcpy(in, xq16, len * sizeof(opus_int16));
    osce_bwe(&s->model, &s->dec.osce_bwe, xq48, in, len, s->arch);
}

/* osce_bwe_calculate_features alone, on its own state (cleared by
 * osce_shim_bwe_features_reset). */
SHIM_EXPORT void osce_shim_bwe_features(void *h, const opus_int16 *xq16, int len, float *features)
{
    osce_bwe_calculate_features(&((OsceShim *)h)->bwe_features, features, xq16, len);
}

SHIM_EXPORT void osce_shim_bwe_features_reset(void *h) { memset(&((OsceShim *)h)->bwe_features, 0, sizeof(OSCEBWEFeatureState)); }

SHIM_EXPORT void osce_shim_cross_fade_10ms(float *x_enhanced, float *x_in) { osce_cross_fade_10ms(x_enhanced, x_in, 320); }

SHIM_EXPORT void osce_shim_bwe_cross_fade_10ms(opus_int16 *x_fadein, opus_int16 *x_fadeout)
{
    osce_bwe_cross_fade_10ms(x_fadein, x_fadeout, 480);
}

SHIM_EXPORT void osce_shim_overlap_window(float *window, int n) { compute_overlap_window(window, n); }

SHIM_EXPORT void osce_shim_blocks_reset(void *h)
{
    OsceShim *s = h;
    memset(s->comb, 0, sizeof(s->comb));
    memset(s->conv, 0, sizeof(s->conv));
    memset(s->shape, 0, sizeof(s->shape));
}

/* adacomb_process_frame with the layers of a model. which: 0 = LACE cf1,
 * 1 = NoLACE cf1. x_in and x_out hold 80 samples. */
SHIM_EXPORT void osce_shim_adacomb(void *h, int which, float *x_out, const float *x_in, const float *features,
                                   int pitch_lag)
{
    OsceShim *s = h;
    if (which == 0) {
        LACELayers *l = &s->model.lace.layers;
        adacomb_process_frame(&s->comb[0], x_out, x_in, features, &l->lace_cf1_kernel, &l->lace_cf1_gain,
                              &l->lace_cf1_global_gain, pitch_lag, LACE_COND_DIM, LACE_FRAME_SIZE,
                              LACE_OVERLAP_SIZE, LACE_CF1_KERNEL_SIZE, LACE_CF1_LEFT_PADDING,
                              LACE_CF1_FILTER_GAIN_A, LACE_CF1_FILTER_GAIN_B, LACE_CF1_LOG_GAIN_LIMIT,
                              s->model.lace.window, s->arch);
    } else {
        NOLACELayers *l = &s->model.nolace.layers;
        adacomb_process_frame(&s->comb[1], x_out, x_in, features, &l->nolace_cf1_kernel, &l->nolace_cf1_gain,
                              &l->nolace_cf1_global_gain, pitch_lag, NOLACE_COND_DIM, NOLACE_FRAME_SIZE,
                              NOLACE_OVERLAP_SIZE, NOLACE_CF1_KERNEL_SIZE, NOLACE_CF1_LEFT_PADDING,
                              NOLACE_CF1_FILTER_GAIN_A, NOLACE_CF1_FILTER_GAIN_B, NOLACE_CF1_LOG_GAIN_LIMIT,
                              s->model.nolace.window, s->arch);
    }
}

#define CONV(slot, L, P, FD, window)                                                                        \
    adaconv_process_frame(&s->conv[slot], x_out, x_in, features, &L##_kernel, &L##_gain, FD, P##_FRAME_SIZE, \
                          P##_OVERLAP_SIZE, P##_IN_CHANNELS, P##_OUT_CHANNELS, P##_KERNEL_SIZE,              \
                          P##_LEFT_PADDING, P##_FILTER_GAIN_A, P##_FILTER_GAIN_B, P##_SHAPE_GAIN, window,     \
                          s->arch)

/* adaconv_process_frame with the layers of a model. which: 0 = LACE af1,
 * 1 = NoLACE af2, 2 = BBWENet af1, 3 = BBWENet af2, 4 = BBWENet af3. */
SHIM_EXPORT void osce_shim_adaconv(void *h, int which, float *x_out, const float *x_in, const float *features)
{
    OsceShim *s = h;
    LACELayers *ll = &s->model.lace.layers;
    NOLACELayers *nl = &s->model.nolace.layers;
    BBWENETLayers *bl = &s->model.bbwenet.layers;
    switch (which) {
    case 0: CONV(0, ll->lace_af1, LACE_AF1, LACE_COND_DIM, s->model.lace.window); break;
    case 1: CONV(1, nl->nolace_af2, NOLACE_AF2, NOLACE_COND_DIM, s->model.nolace.window); break;
    case 2: CONV(2, bl->bbwenet_af1, BBWENET_AF1, BBWENET_COND_DIM, s->model.bbwenet.window16); break;
    case 3: CONV(3, bl->bbwenet_af2, BBWENET_AF2, BBWENET_COND_DIM, s->model.bbwenet.window32); break;
    default: CONV(4, bl->bbwenet_af3, BBWENET_AF3, BBWENET_COND_DIM, s->model.bbwenet.window48); break;
    }
}

/* adashape_process_frame in place on x with the layers of a model. which:
 * 0 = NoLACE tdshape1 (interpolation 1, as nolace_process_20ms_frame passes
 * it), 1 = BBWENet tdshape1, 2 = BBWENet tdshape2. */
SHIM_EXPORT void osce_shim_adashape(void *h, int which, float *x, const float *features)
{
    OsceShim *s = h;
    NOLACELayers *nl = &s->model.nolace.layers;
    BBWENETLayers *bl = &s->model.bbwenet.layers;
    switch (which) {
    case 0:
        adashape_process_frame(&s->shape[0], x, x, features, &nl->nolace_tdshape1_alpha1_f,
                               &nl->nolace_tdshape1_alpha1_t, &nl->nolace_tdshape1_alpha2,
                               NOLACE_TDSHAPE1_FEATURE_DIM, NOLACE_TDSHAPE1_FRAME_SIZE, NOLACE_TDSHAPE1_AVG_POOL_K,
                               1, s->arch);
        break;
    case 1:
        adashape_process_frame(&s->shape[1], x, x, features, &bl->bbwenet_tdshape1_alpha1_f,
                               &bl->bbwenet_tdshape1_alpha1_t, &bl->bbwenet_tdshape1_alpha2,
                               BBWENET_TDSHAPE1_FEATURE_DIM, BBWENET_TDSHAPE1_FRAME_SIZE,
                               BBWENET_TDSHAPE1_AVG_POOL_K, BBWENET_TDSHAPE1_INTERPOLATE_K, s->arch);
        break;
    default:
        adashape_process_frame(&s->shape[2], x, x, features, &bl->bbwenet_tdshape2_alpha1_f,
                               &bl->bbwenet_tdshape2_alpha1_t, &bl->bbwenet_tdshape2_alpha2,
                               BBWENET_TDSHAPE2_FEATURE_DIM, BBWENET_TDSHAPE2_FRAME_SIZE,
                               BBWENET_TDSHAPE2_AVG_POOL_K, BBWENET_TDSHAPE2_INTERPOLATE_K, s->arch);
        break;
    }
}

/* Sizes of the libopus structs, for the documentation: 0 silk_OSCE_struct,
 * 1 silk_OSCE_BWE_struct, 2 OSCEModel, 3 silk_decoder_state. */
SHIM_EXPORT long osce_shim_sizeof(int which)
{
    switch (which) {
    case 0: return (long)sizeof(silk_OSCE_struct);
    case 1: return (long)sizeof(silk_OSCE_BWE_struct);
    case 2: return (long)sizeof(OSCEModel);
    default: return (long)sizeof(silk_decoder_state);
    }
}
