# C# YOLO26 classification experiment (.NET Framework 4.6.1)

This console application runs the exported YOLO26 classification ONNX model in-process with ONNX Runtime. It reproduces the training pipeline's full-frame 224 x 224 resize, RGB conversion, and pixel scaling to `[0, 1]`.

Build and run from the repository root:

```powershell
dotnet build CSharpYolo461\CSharpYolo461.csproj -c Release
& CSharpYolo461\bin\Release\net461\win7-x64\CSharpYolo461.exe `
  runs\classify\yolo26-seal-260721\weights\best.onnx `
  images\seal_dataset_v2\test
```

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

The optimized path reuses its resize surface, tensor buffer, and ONNX input wrapper. It fills the CHW tensor directly from locked bitmap memory, avoiding per-image tensor and intermediate byte-array allocations.

## Deployment options considered

1. **Direct ONNX Runtime in C# (implemented):** simplest in-process route for .NET Framework 4.6.1 and preserves test accuracy.
2. **OpenVINO through native interop:** potentially faster at model execution on Intel hardware, but it requires owning a C API/P/Invoke wrapper and native deployment. The optimized C# ONNX pipeline already matches the measured Python OpenVINO end-to-end latency.
3. **External Python/OpenVINO worker:** already proven faster by the Python benchmark, but adds process lifecycle, IPC, Python environment, and failure-recovery concerns.

For an existing .NET Framework application, start with option 1. Keep one `InferenceSession` alive for the process lifetime; constructing it per image is expensive.
