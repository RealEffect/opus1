/* Writes a weight blob with every network of libopus 1.6.1, in the format
   that OPUS_SET_DNN_BLOB reads.

   libopus's own writer, dnn/write_lpcnet_weights.c (the program
   dump_weights_blob), writes PitchDNN, FARGAN, the PLC model, the RDOVAE
   encoder and decoder, and with ENABLE_OSCE LACE and NoLACE. It leaves out
   BBWENet, the network of the OSCE bandwidth extension. A library built with
   ENABLE_OSCE_BWE and USE_WEIGHTS_FILE needs BBWENet in the blob, or
   osce_load_models fails and LACE and NoLACE stay unloaded too.

   This program includes that file, so it writes the records with libopus's
   own write_weights(), in the same order, and then adds the records of
   BBWENet. Its output therefore starts with the bytes of weights_blob.bin.

   scripts/android/build-libopus.sh --host --osce (or --dnn) builds it with
   the defines of the library, which include ENABLE_OSCE and
   DISABLE_DEBUG_FLOAT, and with -DDUMP_BINARY_WEIGHTS, as Makefile.am builds
   dump_weights_blob. DISABLE_DEBUG_FLOAT leaves out the float copies of the
   int8 weights of BBWENet (bbwenet_data.c), as in the library; with them,
   linear_init would take the float weights.

   Usage: dump_weights_blob_all [FILE]   (default weights_blob_all.bin) */

#ifndef ENABLE_OSCE
#error "build with -DENABLE_OSCE=1 (build-libopus.sh --osce or --dnn)"
#endif
#ifndef DISABLE_DEBUG_FLOAT
#error "build with -DDISABLE_DEBUG_FLOAT=1, as the library"
#endif

/* The main() of write_lpcnet_weights.c is compiled under another name and
   not called. */
#define main dump_weights_blob_main
#include "write_lpcnet_weights.c"
#undef main

#include "bbwenet_data.c"

int main(int argc, char **argv)
{
  const char *path = argc > 1 ? argv[1] : "weights_blob_all.bin";
  FILE *fout = fopen(path, "wb");
  if (fout == NULL) {
    perror(path);
    return 1;
  }
  /* The order of write_lpcnet_weights.c, then BBWENet. */
  write_weights(pitchdnn_arrays, fout);
  write_weights(fargan_arrays, fout);
  write_weights(plcmodel_arrays, fout);
  write_weights(rdovaeenc_arrays, fout);
  write_weights(rdovaedec_arrays, fout);
  write_weights(lacelayers_arrays, fout);
  write_weights(nolacelayers_arrays, fout);
  write_weights(bbwenetlayers_arrays, fout);
  if (fclose(fout) != 0) {
    perror(path);
    return 1;
  }
  return 0;
}
