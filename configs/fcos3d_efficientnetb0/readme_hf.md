# FCOS3D (EfficientNet-b0)

FCOS3D extends FCOS anchor-free fully-convolutional detection to 3D: EfficientNet-b0 + BiFPN extract multi-scale features; FCOS3DHead predicts 3D detection boxes (center offset, depth, size, orientation, class) at each feature point; FCOS3DTarget performs 3D target assignment; FCOS3DPostProcess decodes final 3D detection boxes.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| FCOS3D | Single front-view image `(B,3,512,896)` | EfficientNet-b0 | BiFPN | Front-view 3D detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.312 | 0.3044 | 0.3099 | 0.3098 |
|  | mono mAP | 0.2101 | 0.2042 | 0.2067 | 0.207 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 1.89 | 650.14 | 18.40 |
| J6P | 1.51 | 3294.91 | 19.30 |
| J6B | 4.91 | 244.24 | 14.90 |

---

## Model Overview

### Core Design

FCOS3D extends FCOS anchor-free fully-convolutional detection to 3D: EfficientNet-b0 + BiFPN extract multi-scale features; FCOS3DHead predicts 3D detection boxes (center offset, depth, size, orientation, class) at each feature point; FCOS3DTarget performs 3D target assignment; FCOS3DPostProcess decodes final 3D detection boxes.

- **Task type**: Monocular 3D object detection (Monocular 3D Object Detection).
- **backbone**: EfficientNet-b0 (`efficientnet`, `model_type=b0`, `include_top=False`, `activation=relu`, `use_se_block=False`).
- **neck**: BiFPN (`BiFPN`, bidirectional feature pyramid, `stack=3`, `out_channels=64`, `num_outs=5`).
- **Detection head**: `FCOS3DHead` (fully-convolutional anchor-free 3D detection head).
- **Loss**: `FCOS3DLoss` (FocalLoss + SmoothL1Loss + CrossEntropyLoss).
- **Model input**: Single front-view image (only CAM_FRONT from 6 cameras), `(B,3,512,896)` (Resize3D `img_scale=(896,512)` + Pad `(512,896)`).
- **Model output**: Front-view 3D detection boxes (class + center + size + orientation), per-feature-point prediction `group_reg_dims=(2,1,3,1,2)`, `num_classes=10`, decoded via `FCOS3DPostProcess` + NMS (`max_per_img=100`).

### Official Repo and Paper

Official repo: https://github.com/open-mmlab/mmdetection3d
Paper: https://arxiv.org/abs/2104.10956

Note: backbone is EfficientNet-b0; official implementation uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14096
