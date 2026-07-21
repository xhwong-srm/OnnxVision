# C# YOLO26 classification experiment (.NET Framework 4.6.1)

This console application runs the exported YOLO26 classification ONNX model in-process with ONNX Runtime. It reproduces the training pipeline's full-frame 224 x 224 resize, RGB conversion, and pixel scaling to `[0, 1]`.

Build and run from the repository root:

```powershell
dotnet build CSharpYolo461\CSharpYolo461.csproj -c Release
& CSharpYolo461\bin\Release\net461\win7-x64\CSharpYolo461.exe `
  runs\classify\yolo26-seal\weights\best.onnx `
  images\seal_dataset_v2\test
```

The executable must run as x64 because the ONNX Runtime native package is architecture-specific.

## Measured result on this machine

- Runtime: ONNX Runtime 1.17.3, CPU execution provider
- Dataset: `images/seal_dataset_v2/test` (400 images)
- Accuracy: 398/400 (99.50%)
- Flipped recall: 43/43 (100.00%)
- Normal recall: 355/357 (99.44%)
- Warmed end-to-end latency: 18.949-22.740 ms/image across two runs (44.0-52.8 images/s)
- Mismatches: `normal/36_D.bmp`, `normal/38_C.bmp`

The same ONNX model with the original TorchVision full-frame preprocessing produces the same two mismatches. Confidence varies slightly because `System.Drawing` and TorchVision use different resize implementations.

## Deployment options considered

1. **Direct ONNX Runtime in C# (implemented):** simplest in-process route for .NET Framework 4.6.1 and preserves test accuracy.
2. **OpenVINO through native interop:** potentially faster on this Intel machine, but it requires owning a C API/P/Invoke wrapper and native deployment. There is no benefit to taking on that integration until 22.7 ms/image is insufficient.
3. **External Python/OpenVINO worker:** already proven faster by the Python benchmark, but adds process lifecycle, IPC, Python environment, and failure-recovery concerns.

For an existing .NET Framework application, start with option 1. Keep one `InferenceSession` alive for the process lifetime; constructing it per image is expensive.
