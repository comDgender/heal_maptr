# PETR (EfficientNet-b3)

PETR introduces 3D position into Transformer decoding: EfficientNet-b3 extracts multi-view image features, which are associated with 3D spatial positions via 3D positional encoding (SinePositionalEncoding3D); PETRTransformer queries interact directly with 3D position-encoded features to predict 3D detection boxes, without explicit BEV feature construction.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| PETR | 6-camera multi-view images `(B,6,3,512,1408)` | EfficientNet-b3 | — | Detection boxes `(B,900,21)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3881 | 0.3679 | 0.38 | 0.38 |
|  | mAP | 0.3031 | 0.2807 | 0.2942 | 0.2942 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 33.00 | 30.52 | 101.40 |
| J6P | 21.58 | 186.30 | 105.40 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

PETR introduces 3D position into Transformer decoding: EfficientNet-b3 extracts multi-view image features, which are associated with 3D spatial positions via 3D positional encoding (SinePositionalEncoding3D); PETRTransformer queries interact directly with 3D position-encoded features to predict 3D detection boxes, without explicit BEV feature construction.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False` removes the classification head).
- **neck**: — (no standalone neck; backbone features feed directly into `PETRHead`).
- **Detection head**: `PETRHead` + `PETRTransformer` + `PETRDecoder` (3D position-aware Transformer decoder, `num_query=900`, `num_views=6`, `embed_dims=256`).
- **Loss function**: FocalLoss (cls) + L1Loss (reg), with Detr3dTarget Hungarian matching.
- **Model input**: 6-camera multi-view images, `data_shape=(3,512,1408)` (original `(3,900,1600)` resized to `(3,792,1408)` then cropped to `data_shape`), 6 views stacked as `(B,6,3,512,1408)`.
- **Model output**: Detection boxes `(B,900,21)` — 900 queries × (10 class scores + 11 regression: center xyz + size wlh + orientation sin/cos + velocity vxvy), `num_classes=10`, `reg_out_channels=10`.

### Official Repo and Paper

Official repo: https://github.com/megvii-research/PETR
Paper: https://arxiv.org/abs/2203.05625

Note: The backbone is EfficientNet-b3; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14091
