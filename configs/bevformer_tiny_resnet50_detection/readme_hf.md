# BEVFormer (ResNet-50)

BEVFormer extracts BEV features from multi-camera sequences via learnable spatiotemporal attention (Temporal Self-Attention + Spatial Cross-Attention): ResNet-50 + FPN extract multi-scale image features, BevFormerViewTransformer projects to BEV, BEVFormerEncoder fuses temporal and spatial information, and BEVFormerDetDecoder decodes 3D bounding boxes.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BevFormer | 6-camera multi-view image sequence `(B,6,3,480,800)` | ResNet-50 | FPN | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3739 | 0.3607 | 0.3734 | 0.3669 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 21.76 | 46.52 | 112.80 |
| J6P | 14.15 | 274.97 | 113.80 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

BEVFormer extracts BEV features from multi-camera sequences via learnable spatiotemporal attention (Temporal Self-Attention + Spatial Cross-Attention): ResNet-50 + FPN extract multi-scale image features, BevFormerViewTransformer projects to BEV, BEVFormerEncoder fuses temporal and spatial information, and BEVFormerDetDecoder decodes 3D bounding boxes.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: ResNet-50 (`ResNet50`, `include_top=False`, pretrained `num_classes=1000`).
- **neck**: FPN (`FPN`, multi-scale feature pyramid, `out_strides=[32]`, `out_channels=[256]`).
- **Detection head**: `BEVFormerDetDecoder` (DETR-style decoder + Hungarian matching).
- **Loss function**: `BevFormerCriterion` (FocalLoss + L1Loss, matched via BevFormerHungarianAssigner3D).
- **Model input**: 6-camera multi-view image sequence, `(B,6,3,480,800)` (original `orig_shape=(3,900,1600)` → resize `(3,450,800)` → pad to `(3,480,800)`, `num_views=6`, training `queue_length=3`, evaluation `queue_length=1`).
- **Model output**: 3D bounding boxes on BEV features (class + center + size + orientation), `num_query=900`, `num_classes=10`, decoded via `BevFormerProcess` (`max_num=300`, `score_threshold=0.3`).

### Official Repo and Paper

Official repo: https://github.com/fundamentalvision/BevFormer
Paper: https://arxiv.org/abs/2203.17270

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14101
