# Third-party notices

## Directly used by the scripts

| Package | Locked version | License | Project / license source |
|---|---:|---|---|
| timm | 1.0.28 | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| torch | 2.13.0 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchvision | 0.28.0 | BSD-3-Clause | https://github.com/pytorch/vision |
| numpy | 2.5.1 | BSD-3-Clause and bundled component licenses | https://github.com/numpy/numpy |
| onnx | 1.22.0 | Apache-2.0 | https://github.com/onnx/onnx |
| onnxruntime | 1.27.0 | MIT | https://github.com/microsoft/onnxruntime |
| Pillow | 12.3.0 | MIT-CMU | https://github.com/python-pillow/Pillow |

`onnxslim` is an optional import in `export_timm_classification.py`. If it is
installed and used, its current package license is MIT:
https://github.com/inisis/OnnxSlim

## Runtime dependencies pulled by `timm` and the export stack

| Package | Locked version | License |
|---|---:|---|
| huggingface-hub | 1.24.0 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| tqdm | 4.69.0 | MPL-2.0 or MIT |
| protobuf | 7.35.1 | BSD-3-Clause |
| ml-dtypes | 0.5.4 | Apache-2.0 |
| flatbuffers | 25.12.19 | Apache-2.0 |
| packaging | 26.2 | Apache-2.0 or BSD-2-Clause |
| filelock | 3.30.2 | MIT |
| fsspec | 2026.6.0 | BSD-3-Clause |
| Jinja2 | 3.1.6 | BSD-3-Clause |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| networkx | 3.6.1 | BSD-3-Clause |
| typing-extensions | 4.15.0 | PSF-2.0 |
| click | 8.4.2 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| httpcore | 1.0.9 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| idna | 3.18 | BSD-3-Clause |
| colorama | 0.4.6 | BSD-3-Clause |
| hf-xet | 1.5.2 | Apache-2.0 |

The exact transitive set can vary by platform. This table reflects the locked
environment used for this workspace; regenerate it when changing Python,
platform, or dependency versions.

## License texts and upstream notices

Package-specific upstream license texts are included as the sibling
`*-LICENSE.txt` files. ONNX Runtime's package-specific third-party notice is
included as `onnxruntime-ThirdPartyNotices.txt`.

The fsspec and httpcore repositories do not expose a stable license filename
at the URLs used during collection; their license identifiers and project
links remain recorded above and should be checked against the exact wheel
metadata during release packaging.
