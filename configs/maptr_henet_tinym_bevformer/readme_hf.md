# MapTR+HENet (BevFormer)

MapTR uses HENet as the camera backbone to extract multi-view features, transforms them to BEV features via BevFormer's single-frame ViewTransformer and BEV Encoder, then feeds BEV features to the MapTR decoder with fixed-point polyline queries (`fixed_ptsnum_per_pred_line=20`) to predict vectorized map elements (divider/ped_crossing/boundary). This task has `use_lidar_gt=False`; map GT is generated online.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| MapTR | 6-camera multi-view images `(B,6,3,480,800)` | HENet-tiny | FPN | vectorized map `(B,L,P,2)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | chamfer mAP (MAP) | 0.6626 | 0.6588 | — | 0.6315 |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with 8 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 9.26 | 111.22 | 88.00 |
| J6P | 6.20 | 662.25 | 82.90 |
| J6B | 25.69 | 39.70 | 79.70 |

---

## Model Overview

### Core Design

MapTR uses HENet as the camera backbone to extract multi-view features, transforms them to BEV features via BevFormer's single-frame ViewTransformer and BEV Encoder, then feeds BEV features to the MapTR decoder with fixed-point polyline queries (`fixed_ptsnum_per_pred_line=20`) to predict vectorized map elements (divider/ped_crossing/boundary). This task has `use_lidar_gt=False`; map GT is generated online.

- **Task type**: Online Vectorized Map Construction.
- **backbone**: HENet-tiny (pretrained), extracts multi-view camera features.
- **neck**: FPN.
- **view transformation**: `SingleFrameBevFormerViewTransformer` + `SingleFrameBEVFormerEncoder` (`queue_length=1`/`test_queue_length=1`).
- **map elements**: `map_classes=[divider, ped_crossing, boundary]`, `fixed_ptsnum_per_gt_line=20`.
- **BEV range**: `use_lidar_gt=False` else branch, `point_cloud_range=[-30.0,-15.0,-10.0,30.0,15.0,10.0]`, `bev_h_=50`, `bev_w_=100` (bev 50×100).
- **Model input**: 6-camera single-frame images `(B,6,3,480,800)` .
- **Model output**: 3 classes of vectorized map elements (divider/ped_crossing/boundary), 20 points per line.

### Official Repo and Paper

Official repo: https://github.com/hustvl/MapTR
Paper: https://arxiv.org/abs/2208.14437

Note: The camera backbone HENet is developed in HEAL; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14100
