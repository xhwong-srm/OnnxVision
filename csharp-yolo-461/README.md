# C# YOLO26 classification experiment (.NET Framework 4.6.1)

This console application runs the exported YOLO26 classification ONNX model in-process with ONNX Runtime. Image ownership and ROI access use Euresys Open eVision 22.12. The runner supports grayscale `EImageBW8`/`EROIBW8`, color `EImageC24`/`EROIC24`, and the original float-input model.

The project references the installed production assembly at `C:\VisionRef64\Open_eVision_NetApi_22_12.dll` and must run on a machine with the matching Open eVision 22.12 x64 runtime.

Build and run from the repository root:

```powershell
dotnet build CSharpYolo461\CSharpYolo461.csproj -c Release
& CSharpYolo461\bin\Release\net461\win7-x64\CSharpYolo461.exe `
  runs\classify\yolo26-seal-260721\weights\best.onnx `
  images\seal_dataset_v2\test
```

With a fixed production ROI, append `x y width height`:

```powershell
& CSharpYolo461\bin\Release\net461\win7-x64\CSharpYolo461.exe `
  runs\classify\yolo26-seal-260721\weights\best.onnx `
  images\seal_dataset_v2\test `
  100 80 640 640
```

When the production pipeline already owns an image, call `Classifier.Predict(EImageBW8)` or `Classifier.Predict(EImageC24)` directly. The file-based test overload selects the matching Euresys image type from the ONNX input metadata. Both paths attach a reusable ROI, apply the configured placement, copy rows through `GetImagePtr(0, y)`, and detach the ROI before the source image can be disposed.

Generate and validate both embedded-preprocessing wrappers:

```powershell
uv run python python-scripts\experiment_embedded_preprocessing.py `
  runs\classify\yolo26-seal-260721\weights\best.onnx `
  images\seal_dataset_v2\test `
  --bw8-output artifacts\best-embedded-preprocess-bw8.onnx `
  --c24-output artifacts\best-embedded-preprocess-c24.onnx
```

Run either model with the same executable:

```powershell
& CSharpYolo461\bin\Release\net461\win7-x64\CSharpYolo461.exe `
  artifacts\best-embedded-preprocess-bw8.onnx `
  images\seal_dataset_v2\test

& CSharpYolo461\bin\Release\net461\win7-x64\CSharpYolo461.exe `
  artifacts\best-embedded-preprocess-c24.onnx `
  images\seal_dataset_v2\test
```

The BW8 model accepts `uint8 [1,1,H,W]` NCHW grayscale. The C24 model accepts raw Euresys/Windows-compatible `uint8 [1,H,W,3]` NHWC BGR and performs BGR-to-RGB conversion inside ONNX. Both graphs perform resize, float conversion, normalization, and then execute the same classifier core.

The executable must run as x64 because the ONNX Runtime native package is architecture-specific.

## Measured result on this machine

- Runtime: ONNX Runtime 1.17.3, CPU execution provider
- Dataset: `images/seal_dataset_v2/test` (400 images)
- Model: `runs/classify/yolo26-seal-260721/weights/best.onnx`
- Accuracy: 397/400 (99.25%)
- Flipped recall: 42/43 (97.67%)
- Normal recall: 355/357 (99.44%)
- Warmed end-to-end latency: 5.529-6.977 ms/image across three runs (143-181 images/s)
- Preprocessing: 2.224-2.598 ms/image
- ONNX inference: 3.286-4.352 ms/image
- Mismatches: `flipped/22_E.bmp`, `normal/36_D.bmp`, `normal/38_C.bmp`

## Euresys full-image test result

- Accuracy: 398/400 (99.50%)
- Flipped recall: 43/43 (100.00%)
- Normal recall: 355/357 (99.44%)
- Warmed end-to-end latency: 18.236 ms/image (54.8 images/s)
- Preprocessing (Euresys load, ROI access, and resize): 14.858 ms/image
- ONNX inference: 3.360 ms/image
- Mismatches: `normal/36_D.bmp`, `normal/38_C.bmp`

## Embedded ONNX preprocessing results

- Models: `artifacts/best-embedded-preprocess-bw8.onnx` and `artifacts/best-embedded-preprocess-c24.onnx`
- BW8 input: dynamic-size `uint8 [1,1,height,width]` NCHW grayscale
- C24 input: dynamic-size `uint8 [1,height,width,3]` NHWC BGR
- Python validation: both wrappers agreed with the reference on 400/400 predictions and with each other on 400/400 predictions
- Accuracy: 398/400 (99.50%) for both wrappers
- Flipped recall: 43/43 (100.00%)
- Normal recall: 355/357 (99.44%)
- BW8 C# end-to-end: 4.162 ms/image (240.3 images/s), with 0.467 ms/image outside ONNX
- C24 C# end-to-end: 4.311 ms/image (232.0 images/s), with 0.496 ms/image outside ONNX
- Mismatches: `normal/36_D.bmp`, `normal/38_C.bmp`

The Euresys path reuses its ROI, tensor buffer, and ONNX input wrapper. Embedded models copy raw Euresys rows without resizing or channel conversion in C#. In production, passing an existing Euresys image also removes file loading from the preprocessing measurement.

## Deployment options considered

1. **Direct ONNX Runtime in C# (implemented):** simplest in-process route for .NET Framework 4.6.1 and preserves test accuracy.
2. **OpenVINO through native interop:** potentially faster at model execution on Intel hardware, but it requires owning a C API/P/Invoke wrapper and native deployment. The optimized C# ONNX pipeline already matches the measured Python OpenVINO end-to-end latency.
3. **External Python/OpenVINO worker:** already proven faster by the Python benchmark, but adds process lifecycle, IPC, Python environment, and failure-recovery concerns.

For an existing .NET Framework application, start with option 1. Keep one `InferenceSession` alive for the process lifetime; constructing it per image is expensive.
