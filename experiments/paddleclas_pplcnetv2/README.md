# PaddleClas PP-LCNetV2 seal experiment

This experiment fine-tunes PaddleClas release `2.6` `PPLCNetV2_base` on
`seal_dataset_v6.2`. The source dataset is not copied into this repository;
the manifest generator records paths relative to the dataset root so the same
dataset can be mounted on a training machine.

The class mapping is stable and intentional:

| ID | Class |
| --- | --- |
| 0 | `flipped` |
| 1 | `normal` |

The `train` and `val` splits are used during training. The untouched `test`
split is evaluated only after training and is the primary reported result.

## Remote setup

The commands below assume the repository is checked out at
`C:\Users\srm\xhwong\OnnxVision` on the remote Windows machine and PaddleClas
is cloned into the ignored `PaddleClas` directory beside this README.

```powershell
$repo = 'C:\Users\srm\xhwong\OnnxVision'
$pc = Join-Path $repo 'experiments\paddleclas_pplcnetv2\PaddleClas'
$py = Join-Path $repo '.venv-paddleclas\Scripts\python.exe'

& $py (Join-Path $repo 'experiments\paddleclas_pplcnetv2\prepare_dataset.py') `
  --source (Join-Path $repo 'datasets\seal_dataset_v6.2') `
  --manifest-dir (Join-Path $repo 'experiments\paddleclas_pplcnetv2\artifacts\seal_dataset_v6.2')

Set-Location $pc
& $py tools\train.py `
  -c ..\configs\PPLCNetV2_base_seal_v6.2.yaml
```

PaddleClas release 2.6's documented Windows GPU wheel is CUDA 12.0. The
remote RTX A1000 driver is newer, so install the framework in a dedicated
Python 3.11 environment and verify `paddle.device.get_device()` before
training:

```powershell
uv venv .venv-paddleclas --python 3.11
uv pip install --python .venv-paddleclas\Scripts\python.exe `
  'paddlepaddle-gpu==2.6.1.post120' `
  -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
git clone --depth 1 --branch release/2.6 `
  https://github.com/PaddlePaddle/PaddleClas `
  experiments\paddleclas_pplcnetv2\PaddleClas
& .\.venv-paddleclas\Scripts\python.exe -c `
  "import paddle; print(paddle.__version__); print(paddle.device.get_device()); paddle.utils.run_check()"
```

PaddlePaddle 2.6.1 expects cuDNN 8.9 DLLs. Keep those DLLs scoped to the
dedicated environment when running commands:

```powershell
$cuda = Join-Path $repo '.venv-paddleclas\Lib\site-packages\nvidia'
$env:PATH = (Join-Path $cuda 'cudnn\bin') + ';' +
  (Join-Path $cuda 'cublas\bin') + ';' +
  (Join-Path $cuda 'cuda_nvrtc\bin') + ';' + $env:PATH
```

If the prebuilt GPU wheel is unavailable for the selected Python/platform,
stop at the environment check rather than silently running this experiment on
CPU. PaddleClas's Windows documentation notes that its native Windows GPU
path is single-GPU.

## Evaluation

The training configuration evaluates `val` after each epoch and saves the
best checkpoint under the configured output directory. Evaluate the final
holdout with the same preprocessing and the test manifest:

```powershell
& $py tools\eval.py `
  -c ..\configs\PPLCNetV2_base_seal_v6.2.yaml `
  -o Global.pretrained_model=output\PPLCNetV2_base_seal_v6.2\best_model `
  -o DataLoader.Eval.dataset.cls_label_path=..\artifacts\seal_dataset_v6.2\test_list.txt
```

The training log and the test evaluation log are the evidence for the final
Top-1 result. Do not report the best validation score as test performance.

## ONNX export and C# validation

PaddleClas first exports the checkpoint to a float RGB-NCHW inference model.
Paddle2ONNX 2.1.0 rejects this Paddle 2.6.1 model, so the remote export used
an isolated Python 3.10 environment with Paddle2ONNX 1.0.6. That converter
supports only opset 16; `export_contract.py` upgrades the core to opset 18
locally before adding the shared embedded-preprocessing contract:

```powershell
uv run --extra timm python experiments/paddleclas_pplcnetv2/export_contract.py `
  --core experiments/paddleclas_pplcnetv2/artifacts/pplcnetv2_seal_float_opset16.onnx `
  --output experiments/paddleclas_pplcnetv2/artifacts/pplcnetv2_seal_float.onnx
```

The emitted artifacts are ignored experiment outputs:

- `pplcnetv2_seal_float-bw8.onnx`: `uint8[B,1,H,W]`, BW8, NCHW.
- `pplcnetv2_seal_float-c24.onnx`: `uint8[B,H,W,3]`, raw BGR C24, NHWC.

Both contain resize-to-224 stretch preprocessing, ImageNet normalization,
and softmax, and advertise `onnx-vision-classification` contract `2.0.0`.
The current PaddleClas eval recipe uses resize-short 256 plus center crop;
the ONNX experiment's stretch mode is therefore a deployment preprocessing
variant and its metrics should be read separately from PaddleClas's native
metrics.

Build and run the C# CLI against the requested local dataset:

```powershell
dotnet restore onnx-vision\OnnxVisionCLI.csproj
dotnet build onnx-vision\OnnxVisionCLI.csproj -c Release --no-restore
& .\onnx-vision\bin\Release\net461\win7-x64\OnnxVisionCLI.exe `
  .\experiments\paddleclas_pplcnetv2\artifacts\pplcnetv2_seal_float-bw8.onnx `
  'C:\Users\xhwong\Desktop\Images\0603 seal\datasets\v6\seal_dataset_v6.2' `
  cpu -set test -validate --json
```

Use batch 1 for this dataset: the source crops have varying dimensions, and
the CLI requires equal raw-image dimensions within a dynamic batch.

For the batch-1 optimization experiment, simplify the fixed-shape float core
with `onnxsim` using `[1,3,224,224]`, then wrap it with `--batch-size 1`:

```powershell
uv run --cache-dir .tmp-uv-cache --with onnxsim --extra timm python -c `
  "import onnx; from onnxsim import simplify; model,ok=simplify('experiments/paddleclas_pplcnetv2/artifacts/pplcnetv2_seal_float-core-opset18.onnx', overwrite_input_shapes={'x':[1,3,224,224]}, check_n=3); assert ok; onnx.save(model, 'experiments/paddleclas_pplcnetv2/artifacts/pplcnetv2_seal_float-core-fixed1-simplified.onnx')"
uv run --extra timm python experiments/paddleclas_pplcnetv2/export_contract.py `
  --core experiments/paddleclas_pplcnetv2/artifacts/pplcnetv2_seal_float-core-fixed1-simplified.onnx `
  --output experiments/paddleclas_pplcnetv2/artifacts/pplcnetv2_seal_float-fixed1.onnx `
  --batch-size 1
```

The actual experiment also simplified the wrapped graph while preserving
dynamic source height/width, using one representative raw image shape only
for simplifier validation.

The final production decision is documented in
[`docs/experiments/pplcnetv2-seal-v6.2.md`](../../docs/experiments/pplcnetv2-seal-v6.2.md):
keep PP-LCNetV2 as an experiment and do not add it to `vision_workflows` as a
supported model.
