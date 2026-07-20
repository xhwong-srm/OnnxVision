# YOLO26 Seal Classification Backend Benchmark

Benchmark date: 2026-07-20

## Scope

This experiment compared inference backends for the trained seal classification model:

- Model: `runs/classify/yolo26-seal/weights/best.pt`
- Test dataset: `images/seal_dataset_v2/test`
- Input size: 224 x 224
- Test images: 400
  - Flipped: 43
  - Normal: 357
- Inference pattern: one image at a time (`batch=1`)
- Warm-up: 20 images
- Measured runs: 3 complete passes over the test set

The benchmark includes image loading, Ultralytics preprocessing, model inference, and result conversion in the end-to-end measurement. Model loading and the initial warm-up are excluded.

## Test environment

- CPU: Intel Core 7 150U
- Integrated GPU: Intel Graphics, driver 32.0.101.7085
- NVIDIA CUDA device: unavailable
- Python: 3.12.11
- Ultralytics: 8.4.98
- PyTorch: 2.13.0+cpu
- OpenVINO: 2026.2.1
- ONNX Runtime: 1.27.0

Available acceleration providers:

- PyTorch: CPU only
- OpenVINO: CPU and Intel GPU
- ONNX Runtime: CPUExecutionProvider only

## Results

| Backend | Device | Median end-to-end latency | Median reported inference | Approximate throughput | Top-1 accuracy |
|---|---|---:|---:|---:|---:|
| PyTorch | CPU | 10.411 ms/image | 8.197 ms/image | 96 images/s | 398/400 (99.5%) |
| OpenVINO FP16 | CPU | 6.339 ms/image | 3.837 ms/image | 158 images/s | 398/400 (99.5%) |
| OpenVINO FP16 | Intel GPU | **5.798 ms/image** | **3.252 ms/image** | **172 images/s** | 398/400 (99.5%) |
| ONNX Runtime | CPU | 27.455 ms/image | 3.136 ms/image | 36 images/s | 398/400 (99.5%) |

OpenVINO CPU reduced median end-to-end latency by approximately 39% compared with PyTorch CPU. OpenVINO Intel GPU reduced it by approximately 44%. The Intel GPU result was approximately 9% faster than OpenVINO CPU.

ONNX Runtime reported low model inference time, but its end-to-end Ultralytics execution was slow and inconsistent. Its three measured end-to-end runs were 174.081, 21.414, and 27.455 ms/image. It is not recommended through the currently tested Ultralytics path.

## Accuracy comparison

Every successful backend produced the same 398 correct predictions and the same two errors:

| True class | Image | Predicted class | PyTorch confidence | OpenVINO confidence |
|---|---|---|---:|---:|
| Flipped | `flipped/25_E.bmp` | Normal | 0.597470 | 0.598633 |
| Normal | `normal/98_A_37e5233dc8.bmp` | Flipped | 0.690910 | 0.691895 |

The small confidence differences did not change either classification. OpenVINO therefore preserved the measured test accuracy and flipped recall:

- Overall accuracy: 99.5% (398/400)
- Flipped recall: 97.67% (42/43)
- Normal recall: 99.72% (356/357)

## INT8 experiment

An OpenVINO INT8 export was attempted using `images/seal_dataset_v2` for calibration. Ultralytics automatically installed NNCF 3.2.0, but calibration failed before producing an INT8 model:

```text
TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union
```

The failure occurred inside NNCF statistics collection. There is no valid INT8 accuracy or performance result. INT8 should not be deployed until compatible Ultralytics, OpenVINO, and NNCF versions are identified and pinned, followed by a complete accuracy validation.

## Recommendation

Use **OpenVINO CPU** as the default production backend. It provides a substantial improvement over PyTorch while avoiding contention with the integrated GPU and dependence on GPU driver behavior.

Use **OpenVINO Intel GPU** when the additional approximately 0.54 ms reduction is operationally useful and the production machine has a compatible Intel GPU and driver. Validate it under realistic concurrent display and inspection workloads before deployment.

Do not use the tested ONNX Runtime path for this application. Do not use INT8 until the calibration compatibility problem is resolved and its accuracy is verified.

## Generated artifacts

- OpenVINO model: `runs/classify/yolo26-seal/weights/best_openvino_model/`
- ONNX model: `runs/classify/yolo26-seal/weights/best.onnx`
- Benchmark script: `benchmark_yolo_backends.py`

## Reproduction commands

PyTorch CPU:

```powershell
uv run python benchmark_yolo_backends.py `
  "runs\classify\yolo26-seal\weights\best.pt" `
  "images\seal_dataset_v2\test" `
  --runs 3 --warmup 20
```

OpenVINO CPU:

```powershell
uv run python benchmark_yolo_backends.py `
  "runs\classify\yolo26-seal\weights\best_openvino_model" `
  "images\seal_dataset_v2\test" `
  --runs 3 --warmup 20
```

OpenVINO Intel GPU:

```powershell
uv run python benchmark_yolo_backends.py `
  "runs\classify\yolo26-seal\weights\best_openvino_model" `
  "images\seal_dataset_v2\test" `
  --runs 3 --warmup 20 --device intel:gpu
```

ONNX Runtime CPU:

```powershell
uv run python benchmark_yolo_backends.py `
  "runs\classify\yolo26-seal\weights\best.onnx" `
  "images\seal_dataset_v2\test" `
  --runs 3 --warmup 20
```

## Interpretation limits

These measurements apply to this computer, software environment, model, and test dataset. Production latency will also depend on ROI extraction, image acquisition, application integration, concurrent workloads, and hardware power settings. The benchmark should be repeated on the final production computer using the complete inspection pipeline.
