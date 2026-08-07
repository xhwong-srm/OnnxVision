# PP-LCNetV2 seal experiment results

Run date: 2026-08-07  
Branch: `experiment/paddleclas-pplcnetv2-seal-v6.2`  
Framework: PaddleClas `release/2.6`  
Model: `PPLCNetV2_base`, initialized with its ImageNet pretrained weights

## Dataset

The requested local dataset and the remote copy were verified by matching
`merge_manifest.csv` SHA-256:

```text
7695ECC4FA1970DDF23E8A3B35A1BF30396A81F86E6372CC974B21EC959CB0A1
```

Class IDs are `flipped=0` and `normal=1`.

| Split | Flipped | Normal | Total |
| --- | ---: | ---: | ---: |
| train | 3,470 | 3,470 | 6,940 |
| val | 990 | 2,870 | 3,860 |
| test | 500 | 1,430 | 1,930 |

## Training configuration

- Python 3.11.15, PaddlePaddle 2.6.1 GPU, PaddleClas release 2.6
- NVIDIA RTX A1000 8 GB, compute capability 8.6, cuDNN 8.9, single GPU
- 40 epochs, batch size 64, input 224x224
- ImageNet normalization; random crop and horizontal flip for training
- Resize-short 256 and center crop 224 for validation/test
- Momentum 0.9, cosine learning rate 0.08, 3-epoch warmup, L2 0.00004
- AMP O1 enabled

The run took about 23.7 minutes. Warmed training throughput was approximately
255 images/second on the remote GPU.

## Metrics

PaddleClas selected `best_model` at epoch 40 using validation Top-1:

| Evaluation split | Top-1 | Top-2 | Correct |
| --- | ---: | ---: | ---: |
| val | 99.7396% | 100.00% | 3,850 / 3,860 |
| untouched test | 100.00% | 100.00% | 1,930 / 1,930 |

The test result was obtained after training with:

```powershell
python tools/eval.py `
  -c ..\configs\PPLCNetV2_base_seal_v6.2.yaml `
  -o Global.pretrained_model=../../../experiments/paddleclas_pplcnetv2/artifacts/run/best_model `
  -o DataLoader.Eval.dataset.cls_label_path=../../../experiments/paddleclas_pplcnetv2/artifacts/seal_dataset_v6.2/test_list.txt
```

The checkpoint and generated manifests remain on the remote machine under
`experiments/paddleclas_pplcnetv2/artifacts/`; they are intentionally ignored
by Git. This is split-level evidence, not a guarantee of production accuracy;
the next useful check is inference on a separately collected station holdout
or physical-device capture set.

## ONNX and C# results

The remote PaddleClas inference model was converted with Paddle2ONNX 1.0.6 in
an isolated Python 3.10 environment. The current Paddle2ONNX 2.1.0 package
requires Paddle 3.x and could not consume this Paddle 2.6.1 model. The old
converter emitted opset 16; the local export script upgraded it to opset 18
before using the repository's embedded-preprocessing wrapper.

Both artifacts passed ONNX checker, ONNX Runtime contract checks, and the C#
metadata/tensor contract:

| Artifact | Input contract | Size |
| --- | --- | ---: |
| `pplcnetv2_seal_float-bw8.onnx` | `uint8[B,1,H,W]` BW8 NCHW | 21.51 MB |
| `pplcnetv2_seal_float-c24.onnx` | `uint8[B,H,W,3]` raw BGR C24 NHWC | 21.51 MB |

The ONNX wrapper embeds resize-to-224 stretch, RGB conversion/grayscale
replication, `/255`, ImageNet mean/std normalization, and softmax. This is
not identical to the native PaddleClas eval transform (resize-short 256 plus
center crop), so the following ONNX numbers are the deployment-variant
measurements:

| C# CLI provider / split | BW8 | C24 |
| --- | ---: | ---: |
| CPU / val | 3,841 / 3,860 (99.5078%) | 3,841 / 3,860 (99.5078%) |
| CPU / test | 1,929 / 1,930 (99.9482%) | 1,929 / 1,930 (99.9482%) |
| OpenVINO CPU / test | — | 1,929 / 1,930 (99.9482%) |

The C# CLI used dynamic models at batch 1 because source crop dimensions vary.
On the single-run CPU test pass, BW8 measured 17.692 ms/model call per image
(56.52 images/s; 19.398 ms/image end-to-end), while C24 measured 18.018
ms/model call (55.50 images/s; 19.749 ms/image end-to-end). OpenVINO CPU C24
measured 20.644 ms/model call (48.44 images/s; 22.803 ms/image end-to-end).
All three paths selected the same single normal-image error; BW8 and C24 also
had identical per-image predictions on the validation run.

The C# build completed successfully with zero warnings and zero errors:

```text
OnnxVisionCLI -> onnx-vision/bin/Release/net461/win7-x64/OnnxVisionCLI.exe
```
