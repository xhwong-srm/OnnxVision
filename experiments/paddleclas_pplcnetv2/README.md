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
