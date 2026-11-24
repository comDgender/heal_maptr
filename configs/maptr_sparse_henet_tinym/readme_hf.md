# SparseMapTR+HENet

SparseMapTR uses HENet as the camera backbone to extract multi-view features, converts them to BEV features, then feeds them to SparseMapHead (6-layer sparse query stack) for vectorized map element prediction. Unlike MapTR, SparseMapTR uses a sparse query mechanism (`InstanceBankOE` + `SparsePoint3DEncoder`), refining only a small set of candidate queries iteratively to reduce compute. This task has `use_lidar_gt=True`.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseMapTR | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny | FPN | vectorized map `(B,L,P,2)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | chamfer mAP (MAP) | 0.5924 | 0.5882 | — | 0.5892 |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS for J6M/J6P is measured with 8 threads on a single core; J6B uses 4 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 11.42 | 89.32 | 67.80 |
| J6P | 9.40 | 257.29 | 98.10 |
| J6B | 131.69 | 23.36 | 120.20 |

---

## Model Overview

### Core Design

SparseMapTR uses HENet as the camera backbone to extract multi-view features, converts them to BEV features, then feeds them to SparseMapHead (6-layer sparse query stack) for vectorized map element prediction. Unlike MapTR, SparseMapTR uses a sparse query mechanism (`InstanceBankOE` + `SparsePoint3DEncoder`), refining only a small set of candidate queries iteratively to reduce compute. This task has `use_lidar_gt=True`.

- **Task type**: Sparse Vectorized Map Construction.
- **backbone**: HENet-tiny (pretrained), extracts multi-view camera features.
- **neck**: FPN (`out_strides=[4,8,16,32]`, outputs 256-dim multi-scale features).
- **decoder**: `SparseMapHead` (6-layer sparse query stack, `InstanceBankOE` + `SparsePoint3DEncoder`).
- **map elements**: `map_classes=[divider, ped_crossing, boundary]`, `fixed_ptsnum_per_gt_line=20`.
- **BEV range**: `use_lidar_gt=True` branch, `point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]`, `bev_h_=100`, `bev_w_=50` (bev 100×50).
- **Model input**: 6-camera images `(B,6,3,256,704)` + lidar point cloud `(B,N,D)` (`use_lidar_gt=True`).
- **Model output**: 3 classes of vectorized map elements (divider/ped_crossing/boundary), 20 points per line.

### Official Repo and Paper

Official repo: https://github.com/hustvl/MapTR
Paper: https://arxiv.org/abs/2208.14437

Note: The camera backbone HENet is developed in HEAL; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14100
