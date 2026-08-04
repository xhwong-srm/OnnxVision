# OnnxVision

Reusable ONNX classification and object-detection inference for Euresys Open eVision images, plus a command-line interface and benchmark harness.

The shared library accepts `EImageBW8`, `EImageC24`, and Euresys ROI objects directly. Image loading is kept outside the shared model call when benchmarking so image I/O and inference can be measured separately.

## Project structure

```text
onnx-vision/
├── OnnxVisionCLI.csproj             # SDK-style CLI executable
├── Program.cs                       # CLI, JSON reports, and benchmark harness
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

The Intel OpenVINO package supplies the managed ONNX Runtime dependency transitively. There is no explicit `Microsoft.ML.OnnxRuntime` project reference. The projects copy the package's `win-x64` native OpenVINO/ONNX Runtime DLLs to their output directories.

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

Classification expects a directory grouped by class name:

```text
test-directory/
├── class-a/
│   └── image.bmp
└── class-b/
    └── image.bmp
```

```powershell
OnnxVisionCLI.exe model.onnx test-directory cpu
OnnxVisionCLI.exe model.onnx test-directory openvino-cpu --json
```

Detection and benchmarking:

```powershell
OnnxVisionCLI.exe detect model.onnx image-or-directory 0.5 cpu
OnnxVisionCLI.exe benchmark-detect model.onnx image-directory 0.5 10 cpu --json
```

Supported providers are `cpu`, `openvino-cpu`, and `openvino-gpu`. The report includes both the requested and actual provider because provider initialization may fall back to CPU.

The benchmark preloads Euresys images, measures the shared model call separately, and also reports measured-loop and total end-to-end timings.

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

## Runtime deployment

The application must run as a 64-bit process and have these files beside the final executable:

- `OnnxVision.dll`
- `Microsoft.ML.OnnxRuntime.dll`
- `onnxruntime.dll` and provider DLLs
- OpenVINO and TBB native DLLs
- The Microsoft Visual C++ x64 runtime

`OnnxRuntimeEnvironment.ValidateDeployment` can validate the final application directory before inference starts. A successful library build alone does not prove that the final Vision Studio application output contains every native runtime DLL.
