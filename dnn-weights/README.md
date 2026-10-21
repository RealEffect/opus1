# Weights of the libopus neural network models

These files hold the weights of the neural network models of libopus, in libopus's blob format (the format that `OPUS_SET_DNN_BLOB` reads). `src/dnn/builtin.rs` embeds them in opus1 with `include_bytes!`, each one under the feature that needs it:

| File | Model | Feature |
|---|---|---|
| `pitchdnn.bin` | PitchDNN, the pitch estimator of deep packet loss concealment (deep PLC) | `deep-plc` |
| `fargan.bin` | FARGAN, the vocoder of deep PLC | `deep-plc` |
| `plc.bin` | the PLC model, which predicts the features of lost frames | `deep-plc` |
| `dred_rdovae_enc.bin` | the RDOVAE encoder of deep redundancy (DRED) | `dred` |
| `dred_rdovae_dec.bin` | the RDOVAE decoder of DRED | `dred` |
| `lace.bin` | LACE, the speech enhancer of SILK at decoder complexity 6 | `osce` |
| `nolace.bin` | NoLACE, the speech enhancer of SILK at decoder complexity 7 and above | `osce` |
| `bbwenet.bin` | BBWENet, the network of the bandwidth extension | `osce-bwe` |

- `VERSION` names the libopus release and the files the blobs were made from.
- `SHA256SUMS` holds the SHA-256 of each blob. `tests/dnn_weights.rs` checks the embedded blobs against it.
- `COPYING` is the licence of libopus, which covers these weights.

Do not edit these files. `tools/dnn-gen` writes them from the data files of a libopus release (see `tools/dnn-gen/README.md`). Each blob equals, byte for byte, the records of its model in the blob of libopus's own program `dump_weights_blob`. That program leaves out BBWENet; `bbwenet.bin` equals what the same writer function writes for the BBWENet list.
