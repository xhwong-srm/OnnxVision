# PP-LCNetV2 seal v6.2 experiment decision

Date: 2026-08-07  
Branch: `experiment/paddleclas-pplcnetv2-seal-v6.2`

## Scope

PP-LCNetV2 base was trained on `seal_dataset_v6.2` using PaddleClas
`release/2.6`, then exported to the repository's `onnx-vision-classification`
contract with embedded preprocessing and tested through the C#
`OnnxVisionCLI`.

The dataset split sizes were train `6,940`, validation `3,860`, and test
`1,930`. The remote dataset matched the requested local dataset by
`merge_manifest.csv` SHA-256:

```text
7695ECC4FA1970DDF23E8A3B35A1BF30396A81F86E6372CC974B21EC959CB0A1
```

## Training and export

- `PPLCNetV2_base`, ImageNet pretrained initialization, 40 epochs.
- Input core resolution: `224x224`; ImageNet normalization.
- Remote GPU: NVIDIA RTX A1000, single GPU, approximately 23.7 minutes.
- PaddleClas result: `99.7396%` validation Top-1 and `100%` untouched-test
  Top-1.
- Paddle2ONNX 2.1.0 rejected the Paddle 2.6.1 model. The workable route was
  an isolated Python 3.10 environment with Paddle2ONNX 1.0.6, followed by a
  local opset-16-to-18 conversion.
- Both BW8 (`uint8[B,1,H,W]` NCHW) and C24 (`uint8[B,H,W,3]` raw BGR NHWC)
  artifacts passed ONNX checker, ONNX Runtime checks, contract validation,
  and C# loading.

The embedded graph resizes raw input to `224x224`, converts/replicates the
input channels, applies `/255` and ImageNet mean/std normalization, and emits
probabilities. Its stretch resize differs from PaddleClas evaluation's
resize-short-256 plus center-crop preprocessing, so those are separate
deployment measurements.

## C# speed comparison

The following uses the same C# Release executable, CPU provider, test set,
batch 1, three measured repeats, and 1,930 logical images. The MobileNetV4
reference is `mnv4-s-050-v6.2-a-fixed-bw8.onnx`.

| Model variant | Batch | Model call | End-to-end |
| --- | ---: | ---: | ---: |
| MobileNetV4 reference | fixed 1 | `2.026 ms/image`, `493.6 img/s` | `2.724 ms/image` |
| PP-LCNetV2 original | dynamic, run at 1 | `15.471 ms/image`, `64.64 img/s` | `15.972 ms/image` |
| PP-LCNetV2 fixed batch 1 | fixed 1 | `15.327 ms/image`, `65.24 img/s` | `15.794 ms/image` |
| PP-LCNetV2 fixed 1 + simplified | fixed 1 | `14.774 ms/image`, `67.69 img/s` | `15.208 ms/image` |

The PP-LCNetV2 fixed/simplified artifact is therefore about `4-5%` faster
than the original PP-LCNetV2 graph, but remains about `7.3x` slower than the
MobileNetV4 reference for model calls. Fixing batch alone contributed only
about `0.9%`.

Source crops have varying dimensions, so the C# CLI cannot form larger raw
image batches without an external resize/grouping stage. Both the BW8 and C24
variants produced the same C# timing class and prediction behavior in the
earlier contract run.

## Optimization analysis

After shape specialization at `224x224`, the cores contain the same 46 Conv
nodes, but their workload is not comparable:

| Core | Convolution MACs | Weight bytes |
| --- | ---: | ---: |
| MobileNetV4 | `63.5M` | `3.81 MB` |
| PP-LCNetV2 | `592.6M` | `21.35 MB` |

PP-LCNetV2 has approximately `9.3x` the convolution MACs and `5.6x` the
weights, plus squeeze-excitation-style global-pool, sigmoid, and channel
multiply paths. The graph simplifier reduced the embedded BW8 graph from 325
nodes to 117, but weight storage remained about 21.38 MB. A Python ONNX
Runtime isolation check measured roughly `10.8 ms` for the simplified core
and `15.5 ms` for the raw embedded wrapper, confirming that preprocessing is
non-trivial but not the primary gap.

Removing the wrapper's duplicate softmax passed validation but did not show a
reliable speed improvement; it is not a useful optimization lever.

## Decision

Do not implement PP-LCNetV2 as a supported model in `vision_workflows`.

The model is accurate and contract-compatible, but its speed advantage is
not competitive with the existing MobileNetV4 model under the same C# input,
resolution, batch, and CPU conditions. The remaining cost is primarily
architectural, so ONNX simplification and batch specialization are not enough
to justify a production backend/catalog integration.

Keep this as an experiment and revisit only if a smaller PP-LCNetV2 variant,
lower input resolution, quantized model, or a suitable accelerator/provider
changes the deployment trade-off.

Detailed commands and intermediate artifacts are recorded in
[`experiments/paddleclas_pplcnetv2/RESULTS.md`](../../experiments/paddleclas_pplcnetv2/RESULTS.md).
