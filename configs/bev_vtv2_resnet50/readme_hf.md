# BEV (ResNet-50)

This model follows the LSS (Lift-Splat-Shoot) view transformation approach: 6 camera images are processed by ResNet-50 + FPN for multi-scale features, projected to the BEV plane via LSSTransformer, encoded by BevEncoder, and finally predicted on the BEV grid by CenterPointHead for 3D bounding boxes.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BEV-VTv2 | 6-camera multi-view images `(B,6,3,256,704)` | ResNet-50 | FPN | BEV grid 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3603 | 0.3546 | 0.3564 | 0.3539 |
|  | mAP | 0.2833 | 0.2799 | 0.2805 | 0.2797 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 16.38 | 62.06 | 120.90 |
| J6P | 11.43 | 347.06 | 122.60 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

This model follows the LSS (Lift-Splat-Shoot) view transformation approach: 6 camera images are processed by ResNet-50 + FPN for multi-scale features, projected to the BEV plane via LSSTransformer, encoded by BevEncoder, and finally predicted on the BEV grid by CenterPointHead for 3D bounding boxes.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: ResNet-50 (`ResNet50`, `include_top=False`, pretrained `num_classes=1000`).
- **neck**: FPN (`FPN`, multi-scale feature pyramid, output strides 16/32, `fix_out_channel=256`).
- **Detection head**: `CenterPointHead` (BEV grid center-point detection head).
- **Loss function**: `CenterPointLoss` (GaussianFocalLoss + L1Loss).
- **Model input**: 6-camera multi-view images, `(B,6,3,256,704)` (original `orig_shape=(3,900,1600)` → resize `(3,396,704)` → crop `data_shape=(3,256,704)`, `num_views=6`).
- **Model output**: 3D bounding boxes on BEV grid (class + center + size + orientation), `CenterPointHead` outputs heatmap + reg/height/dim/rot/vel per task group, `num_classes=10`, decoded via `CenterPointPostProcess` + NMS.

### Official Repo and Paper

Official repo: https://github.com/HuangJunJie2017/BEVDet
Paper: https://arxiv.org/pdf/2112.11790

Note: Backbone is ResNet-50; detection head is CenterPoint.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14085
