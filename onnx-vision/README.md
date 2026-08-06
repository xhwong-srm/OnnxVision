# OnnxVision

Reusable ONNX classification and object-detection inference for Euresys Open eVision images, plus a command-line interface and benchmark harness.

The shared library accepts `EImageBW8`, `EImageC24`, and Euresys ROI objects directly. Image loading is kept outside the shared model call when benchmarking so image I/O and inference can be measured separately.

## Project structure

```text
onnx-vision/
├── OnnxVisionCLI.csproj             # SDK-style CLI executable
├── Program.cs                       # CLI entrypoint and classification/detection orchestration
├── Cli/                              # CLI options and ROI argument parsing
├── Datasets/                         # ImageNet/COCO loaders and input models
├── Imaging/                          # Euresys image loading and disposal
├── Reporting/                        # Text/JSON reports and classification helpers
├── Validation/                       # Classification and COCO-style detection metrics
└── OnnxVision/
    ├── OnnxVision.csproj            # Legacy project for Vision Studio workflows
    ├── OnnxVision.Sdk.csproj        # SDK-style shared-library project
    ├── Adapters/Euresys/            # EImageBW8/EImageC24/ROI adapters
    ├── Classification/              # Classification model and result types
    ├── Detection/                   # Object-detection model and result types
    ├── Imaging/                     # Image-buffer contracts
    └── Runtime/                     # Provider selection and deployment helpers
```

Use only one shared-project variant per consumer:

- Existing `lead-2000` Vision Studio projects should reference `OnnxVision.csproj`.
- New SDK-style projects and `OnnxVisionCLI` should reference `OnnxVision.Sdk.csproj`.
- Do not reference both variants from the same application.

Both variants intentionally produce the `OnnxVision` assembly and contain the same source/API surface. Their outputs are separated to prevent build collisions.

## Requirements

- Windows x64.
- .NET Framework 4.6.1 targeting pack.
- Visual Studio 2019/MSBuild for the legacy Vision Studio configuration, or a compatible .NET SDK for SDK-style builds.
- Euresys Open eVision 22.12. The projects currently expect:
  `C:\VisionRef64\Open_eVision_NetApi_22_12.dll`
- NuGet restore access to `Intel.ML.OnnxRuntime.OpenVino` version `1.24.1`.

The Intel OpenVINO package supplies the managed ONNX Runtime dependency transitively to the SDK-style build. The legacy project includes an assembly hint to the same package version because old-style MSBuild does not expose that transitive compile reference reliably. The projects copy the package's `win-x64` native OpenVINO/ONNX Runtime DLLs to their output directories.

## Build

From the repository root, restore and build the CLI:

```powershell
dotnet restore .\onnx-vision\OnnxVisionCLI.csproj
dotnet build .\onnx-vision\OnnxVisionCLI.csproj -c Release --no-restore --nologo
```

Build the SDK-style shared library directly:

```powershell
dotnet build .\onnx-vision\OnnxVision\OnnxVision.Sdk.csproj -c Release --no-restore --nologo
```

Build the legacy project with the Vision Studio configuration from a Visual Studio Developer PowerShell:

```powershell
MSBuild.exe .\onnx-vision\OnnxVision\OnnxVision.csproj /t:Restore /p:Configuration=Debug_22_12 /p:Platform=AnyCPU /m /nologo
MSBuild.exe .\onnx-vision\OnnxVision\OnnxVision.csproj /p:Configuration=Debug_22_12 /p:Platform=AnyCPU /m /nologo
```

The legacy project declares `win-x64` so current NuGet/MSBuild versions select the OpenVINO native assets consistently.

## CLI

The executable is produced at:

```text
onnx-vision\bin\Release\net461\win7-x64\OnnxVisionCLI.exe
```

Show usage:

```powershell
.\onnx-vision\bin\Release\net461\win7-x64\OnnxVisionCLI.exe --help
```

The CLI identifies classification versus object detection from the required ONNX metadata contract. Classification models use `onnx-vision-classification`; object-detection models use `onnx-vision-object-detection`. Consumers accept valid `2.x.y` versions, validate every known serialized field and tensor, and permit additive unknown metadata; incompatible semantics require a new major version. Each artifact is a single embedded-preprocessing variant: BW8 uses `uint8[B,1,H,W]` NCHW and C24 uses `uint8[B,H,W,3]` raw-BGR NHWC. Batch may be dynamic or fixed. Classification outputs are categorical probabilities in `float32[B,C]`, with finite `[0,1]` rows summing to 1. Detection outputs are normalized ordered `xyxy` `boxes[B,Q,4]`, `[0,1]` `scores[B,Q]`, and `class_ids[B,Q]`; score-zero rows are padding and ignored before box/class validation. Provider-owned exports use confidence `0` and IoU `0.7`; class-aware consumer NMS at IoU `0.7` is applied only when `nms_required=true`. The CLI preloads images, runs warmup calls, repeats the measured pass as requested, and reports session construction, image loading, warmup calls, shared model calls, measured wall time, end-to-end time, and logical-image throughput. For fixed-batch models it requires the model batch internally, pads only the final batch by duplicating its last image, and discards padded results. `repeats` defaults to `1`.

Both tasks accept a single image, a directory of images, or a labeled dataset. A
single image and a flat image directory are inference-only inputs. Add `-dataset`
to force dataset interpretation when the layout is ambiguous. `-set` selects a
dataset split (`train`, `val`, or `test`); when it is omitted, `val` is selected
when present, followed by `train` and then `test`.

Classification datasets use the usual ImageNet folder layout:

```text
classification-dataset/
├── train/
│   ├── class-a/image.bmp
│   └── class-b/image.bmp
├── val/
│   ├── class-a/image.bmp
│   └── class-b/image.bmp
└── test/
    └── class-a/image.bmp
```

```powershell
OnnxVisionCLI.exe model.onnx image.bmp cpu
OnnxVisionCLI.exe model.onnx image-directory cpu --json
OnnxVisionCLI.exe model.onnx classification-dataset cpu -set val -validate
OnnxVisionCLI.exe model.onnx classification-dataset cpu -set test -validate --json
OnnxVisionCLI.exe model.onnx classification-dataset cpu 1 10 20 224 224 -set val -validate
```

Classification validation reports top-1 accuracy, macro precision/recall/F1,
and per-class support and scores.

Detection datasets use COCO annotations. The loader supports standard
`annotations/instances_<set>.json` files, including `instances_train2017.json`
and `instances_val2017.json`, as well as split-local
`<set>/_annotations.coco.json` files.

```powershell
OnnxVisionCLI.exe model.onnx image.bmp 0.5 1 cpu
OnnxVisionCLI.exe model.onnx image-directory 0.5 10 cpu --json
OnnxVisionCLI.exe model.onnx coco-dataset 0.5 cpu -set val -validate
OnnxVisionCLI.exe model.onnx coco-dataset 0.5 cpu -set test -validate --json
```

Detection validation reports IoU-0.50 precision/recall/F1, mAP50, mAP50-95,
and per-class AP and matching counts. Validation is rejected for a single
image or an unlabeled image directory.

Supported providers are `cpu`, `openvino-cpu`, and `openvino-gpu`. The report includes both the requested and actual provider because provider initialization may fall back to CPU.

The former `benchmark-detect` command is no longer required. It remains accepted as a compatibility alias and routes through the same detection path with a default of three repeats.

## Shared API example

Use the Euresys extensions when the application already owns an Euresys image:

```csharp
using Euresys.Open_eVision_22_12;
using OnnxVision.Classification;
using OnnxVision.Euresys;
using OnnxVision.Runtime;

using (var model = new OnnxClassificationModel(
    modelPath,
    OnnxExecutionProvider.Cpu))
using (var image = new EImageBW8())
{
    image.Load(imagePath);
    OnnxClassification result = model.Classify(image);
}
```

The same adapter pattern is available for `EImageC24`, `EROIBW8`, `EROIC24`, and object detection. Region overloads accept a `System.Drawing.Rectangle` without requiring an intermediate image file.

Use `ClassifyBatch` or `DetectBatch` with images that share the same dimensions
and pixel format. Dynamic models accept any positive batch; fixed models require
exactly `FixedBatchSize`. Single-image methods reject fixed batches greater than
one. The returned results preserve input order; detection postprocessing and NMS
are applied independently per image.

## Runtime deployment

The application must run as a 64-bit process and have these files beside the final executable:

- `OnnxVision.dll`
- `Microsoft.ML.OnnxRuntime.dll`
- `onnxruntime.dll` and provider DLLs
- OpenVINO and TBB native DLLs
- The Microsoft Visual C++ x64 runtime

`OnnxRuntimeEnvironment.ValidateDeployment` can validate the final application directory before inference starts. A successful library build alone does not prove that the final Vision Studio application output contains every native runtime DLL.
