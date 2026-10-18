# dnn-gen

`dnn-gen` writes the weights of the libopus neural network models into the opus1 repository. It is a separate Cargo package with its own workspace. It is not part of the opus1 crate and is not published.

It reads the generated data files of a libopus release, `dnn/<model>_data.c` and `dnn/<model>_data.h`, and writes:

| Output | What it holds |
|---|---|
| `dnn-weights/<model>.bin` | The model's weights in libopus's blob format: the records that libopus's `dump_weights_blob` program (`dnn/write_lpcnet_weights.c`) writes for the model's `WeightArray` list |
| `dnn-weights/SHA256SUMS` | The SHA-256 of each blob, in the format of `sha256sum` |
| `dnn-weights/VERSION` | The libopus version, the SHA-256 of its release tarball, and the SHA-256 of each data file read |
| `dnn-weights/COPYING` | libopus's licence, copied from the release |
| `src/dnn/models/<model>.rs` | The constants of the C header, the layer descriptors (`LAYERS`: for each layer, the arrays it reads and its shape, in the order of the C init function), the layer struct and `init` |
| `tests/dnn_gen/mod.rs` | For the tests against a DNN-enabled libopus: `#[repr(C)]` copies of the C model structs, the C symbols of every model, and the descriptors of every model, the DRED models included |

The models opus1 includes, with their feature:

| Model | Data file | Feature | Blob bytes |
|---|---|---|---:|
| PitchDNN | `pitchdnn_data.c` | `deep-plc` | 76,544 |
| FARGAN | `fargan_data.c` | `deep-plc` | 876,608 |
| PLC model | `plc_data.c` | `deep-plc` | 479,616 |
| LACE | `lace_data.c` | `osce` | 435,456 |
| NoLACE | `nolace_data.c` | `osce` | 1,137,984 |
| BBWENet | `bbwenet_data.c` | `osce-bwe` | 685,888 |

The two DRED models (`dred_rdovae_enc_data.c`, 241,152 bytes, and `dred_rdovae_dec_data.c`, 324,288 bytes) are left out until opus1 has a `dred` feature. `--models` writes them too.

## Running

```bash
cargo run --release --manifest-path tools/dnn-gen/Cargo.toml -- \
  --src target/perf/libopus-1.6.1-src/opus-1.6.1 \
  --check-official target/android/dnn-blob-1.6.1/weights_blob.bin \
  --check-all target/android/dnn-blob-1.6.1/weights_blob_all.bin
```

Options:

- `--src DIR`: the libopus source tree, the folder that holds `dnn/` and `package_version`. Required.
- `--repo DIR`: the opus1 repository. The default is two levels above this README.
- `--models a,b,...`: the models to write. The default is the six models above. The others are `dred_rdovae_enc` and `dred_rdovae_dec`.
- `--blobs-only DIR`: write only the blobs of `--models`, into `DIR`.
- `--check-official FILE`: compare `FILE` with the blobs of the seven models that `dump_weights_blob` writes (PitchDNN, FARGAN, PLC model, both DRED models, LACE, NoLACE), concatenated in its order. The generator always computes all eight blobs, so this works without `--models`.
- `--check-all FILE`: the same with BBWENet appended. `scripts/android/build-libopus.sh --host --dnn` builds `bin/dump_weights_blob_all`, which writes this file.
- `--version V`: the libopus version that `--src` must hold (`package_version`). The default is 1.6.1.
- `--tarball-sha256 HEX`: the SHA-256 of the release tarball, for `VERSION`. It is known for 1.6.1.

The generator prints the size of each model and of each feature.

## What the generator checks

It reads only the part of C that the data files use, and stops with the file and line of anything else, so that a change of format in a later libopus is noticed:

- the preprocessor lines `#ifdef`, `#ifndef`, `#else`, `#endif`, `#define`, `#undef` and `#include`. It compiles the files as a default libopus build does: `DISABLE_DEBUG_FLOAT` is defined, which leaves out the float copies of the int8 weights in `bbwenet_data.c`; `USE_WEIGHTS_FILE` and `DUMP_BINARY_WEIGHTS` are not;
- the arrays (`float`, `int`, `opus_int8`). A float value without a suffix is read as a C compiler reads it: as a `double`, then rounded to `float`;
- the `WeightArray` list, the init function and, in the header, the constants and the struct of layers.

Then it checks that:

- the struct and the init function have the same fields, of the same kinds;
- every layer of the init function finds its arrays with the sizes that `linear_init` and `conv2d_init` require, and has weights;
- no int8 weight is -128 (opus1 refuses those);
- every blob reads back as the list it was written from.

With the 1.6.1 release, `--check-official` and `--check-all` find the two blobs of the reference build byte for byte.

## Adding the DRED models to opus1

1. Run the generator with `--models pitchdnn,fargan,plc,dred_rdovae_enc,dred_rdovae_dec,lace,nolace,bbwenet`.
2. Add the modules `dred_rdovae_enc` and `dred_rdovae_dec` to `src/dnn/models/mod.rs` under the `dred` feature, and their blobs to `src/dnn/builtin.rs`.
