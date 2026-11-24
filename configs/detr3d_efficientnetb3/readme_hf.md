# Detr3D (EfficientNet-b3)

Detr3D brings the DETR paradigm to 3D detection: EfficientNet-b3 + BiFPN extract multi-scale image features; view transformation projects 2D features to 3D space; Detr3dTransformer iteratively samples multi-view features with learnable queries and predicts 3D detection boxes; Detr3dTarget performs Hungarian matching during training.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| Detr3D | 6-camera multi-view images `(B,6,3,512,1408)` | EfficientNet-b3 | BiFPN | 3D detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3357 | 0.3299 | 0.338 | 0.337 |
|  | mAP | 0.2694 | 0.2618 | 0.2688 | 0.2683 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 21.79 | 46.43 | 98.10 |
| J6P | 15.24 | 253.11 | 94.80 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

Detr3D brings the DETR paradigm to 3D detection: EfficientNet-b3 + BiFPN extract multi-scale image features; view transformation projects 2D features to 3D space; Detr3dTransformer iteratively samples multi-view features with learnable queries and predicts 3D detection boxes; Detr3dTarget performs Hungarian matching during training.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False`, `activation=relu`, `use_se_block=False`).
- **neck**: BiFPN (`BiFPN`, bidirectional feature pyramid, `stack=3`, `out_channels=256`, `num_outs=5`).
- **Detection head**: `Detr3dHead` + `Detr3dTransformer` + `Detr3dDecoder` (DETR-style 3D decoder).
- **Loss**: FocalLoss (cls) + L1Loss (bbox), via Detr3dTarget Hungarian matching.
- **Model input**: 6-camera multi-view images, `(B,6,3,512,1408)` (original `orig_shape=(3,900,1600)` → resize `(3,792,1408)` → crop `data_shape=(3,512,1408)`, `num_views=6`).
- **Model output**: 3D detection boxes (class + center + size + orientation), `num_query=900`, `num_classes=10`, decoded via `Detr3dPostProcess` (`max_num=300`).

### Official Repo and Paper

Official repo: https://github.com/WangYueFt/detr3d
Paper: https://arxiv.org/abs/2110.06922

Note: backbone is EfficientNet-b3; official repo uses a different backbone.
