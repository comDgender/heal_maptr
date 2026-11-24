# FlashOcc + HENet + LSS

FlashOcc uses LSS (Lift-Splat-Shoot) view transformation: HENet extracts multi-view camera features; `LSSTransformer` predicts depth distribution and lifts 2D features to 3D voxel space (`depth=45`, `num_points=10`, `bev_size=(40,40,0.625)`); fused via `BevEncoder` (`BiFPN`); `FlashOccDetDecoder`/`BEVOCCHead2D` outputs 18-class 3D occupancy predictions.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| FlashOcc | 6-camera multi-view images `(B,6,3,512,960)` | HENet | FPN + LSSTransformer + BevEncoder | Occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Occ mIoU (MeanIOU) | 0.3664 | 0.3688 | — | 0.369 |

> Results are based on `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 7.61 | 136.01 | 73.40 |
| J6P | 5.64 | 730.55 | 77.60 |
| J6B | 23.63 | 43.17 | 81.60 |

---

## Model Overview

### Core Design

FlashOcc uses LSS (Lift-Splat-Shoot) view transformation: HENet extracts multi-view camera features; `LSSTransformer` predicts depth distribution and lifts 2D features to 3D voxel space (`depth=45`, `num_points=10`, `bev_size=(40,40,0.625)`); fused via `BevEncoder` (`BiFPN`); `FlashOccDetDecoder`/`BEVOCCHead2D` outputs 18-class 3D occupancy predictions.

- **Task type**: BEV 3D occupancy prediction (BEV Occupancy Prediction).
- **backbone**: HENet (`type=HENet`, `depth=45`, `num_points=10`), extracts multi-view camera features.
- **neck**: `FPN` + `LSSTransformer` (Lift-Splat-Shoot view transformation, `bev_size=(40,40,0.625)`, `grid_size=(128,128)`) + `BevEncoder` (`BiFPN`).
- **Occupancy head**: `FlashOccDetDecoder` (`BEVOCCHead2D`, `num_classes=18`, `ignore_index=17`).
- **Loss**: `CrossEntropyLoss` (occ seg).
- **Model input**: 6-view camera images `(B,6,3,512,960)` (`data_shape=(3,512,960)`).
- **Model output**: 18-class 3D occupancy grid `(B,C,H,W)` (`occ3d_seg_class` 18 classes, includes `others`/`ignore_index=17`).

### Official Repo and Paper

Official repo: https://github.com/Yzichen/FlashOCC
Paper: https://arxiv.org/abs/2311.12058

Note: camera backbone HENet is HEAL-developed; official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14099
