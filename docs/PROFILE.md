# Detection Model Compute Profile

This document summarizes the approximate inference compute cost of the local
detection checkpoints. Measurements were produced with `calflops` using a
PyTorch batch-1 forward pass.

The estimates are theoretical graph-level operation counts. They do not
represent measured latency and exclude image decoding, host/device copies,
runtime-specific preprocessing, post-processing/NMS, and training
backward/optimizer memory.

## Summary

| Model | Native input | Parameters | Native GMACs | Native GFLOPs | GMACs at 640x640 | GFLOPs at 640x640 |
|---|---:|---:|---:|---:|---:|---:|
| timm v2 custom | 384x384 | 2.148M | 1.24 | 2.505 | 3.45 | 6.952 |
| timm RetinaNet | 384x384 | 9.264M | 17.05 | 34.147 | 47.35 | 94.853 |
| RTMDet-t | 640x640 | 4.873M | 13.55 | 27.171 | 13.55 | 27.171 |
| PicoDet-s | 320x320 | 0.962M | 0.34 | 0.681 | 1.34 | 2.724 |
| YOLOv9-n | 640x640 | 2.018M | 3.84 | 7.755 | 3.84 | 7.755 |
| YOLO26-n | 640x640 | 2.505M | 2.81 | 7.065 | 2.81 | 7.065 |
| RF-DETR-nano | 384x384 | 30.151M | 14.49 | 29.012 | 38.34 | 76.796 |

## Checkpoints

The measurements use these local checkpoints:

- timm v2 custom: `python-scripts/timm/runs/mobilenetv4_small_custom_v1/best.pt`
- timm RetinaNet: `python-scripts/timm/runs/mobilenetv4_small_retinanet_v1/best.pt`
- RTMDet-t: `python-scripts/libreyolo/runs/rtmdet-t-v1/best.pt`
- PicoDet-s: `python-scripts/libreyolo/runs/picodet-s-v1/best.pt`
- YOLOv9-n: `python-scripts/libreyolo/runs/yolov9-n-v1/best.pt`
- YOLO26-n: `python-scripts/yolo/runs/yolo26-n-v1/best.pt`
- RF-DETR-nano: `python-scripts/rf-detr/runs/rfdetr-nano-v1/checkpoint_best_total.pth`

The timm v2 checkpoint is the custom model with FPN 128 channels, 8 queries,
and 2 decoder layers. It is different from the smaller auto-query smoke
checkpoint previously profiled.

## GMACs versus GFLOPs

- **MAC** means multiply-accumulate: one multiplication followed by one
  addition.
- **GMACs** means billions of MAC operations.
- **GFLOPs** means billions of floating-point operations.

Under the convention used by this profiling run, one MAC is counted as two
FLOPs:

```text
1 GMAC ~= 2 GFLOPs
```

Therefore, GMACs and GFLOPs describe essentially the same theoretical
compute density using different counting conventions. Some papers and tools
count a fused multiply-add as one FLOP, so values from different tools may
differ by approximately a factor of two.

## Interpretation at 640x640

For a common 640x640 input, the approximate compute ranking is:

1. PicoDet-s: 2.724 GFLOPs
2. timm v2 custom: 6.952 GFLOPs
3. YOLO26-n: 7.065 GFLOPs
4. YOLOv9-n: 7.755 GFLOPs
5. RTMDet-t: 27.171 GFLOPs
6. RF-DETR-nano: 76.796 GFLOPs
7. timm RetinaNet: 94.853 GFLOPs

The native-input comparison is also useful because PicoDet is configured for
320x320, RF-DETR and the timm models use 384x384, and the YOLO/LibreYOLO
models use 640x640. Increasing a convolutional detector's image dimensions
usually increases compute roughly with the number of input pixels.

Parameter count describes model weight size, while GMACs/GFLOPs describe
arithmetic work. Neither alone predicts latency: memory bandwidth, operator
implementation, accelerator, precision, post-processing, and batch size also
matter.

## Re-running the profile

The reusable profiler is:

```powershell
uv run python python-scripts/profile_detection_models.py
```

For a common-size comparison:

```powershell
uv run python python-scripts/profile_detection_models.py --resolution 640
```

The profiler reports both parameters and the numeric FLOPs/MACs values in
JSON when passed an output path, for example:

```powershell
uv run python python-scripts/profile_detection_models.py `
  --model "timm-v2::python-scripts/timm/runs/mobilenetv4_small_custom_v1/best.pt" `
  --output tmp/detection_compute_profile_timm_v2_custom.json
```
