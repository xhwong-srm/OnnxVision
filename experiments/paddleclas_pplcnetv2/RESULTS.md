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
