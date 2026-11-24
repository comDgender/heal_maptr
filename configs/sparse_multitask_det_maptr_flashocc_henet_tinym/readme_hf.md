# SparseMultitaskOE+MapTR+FlashOcc+HENet Multitask

SparseMultitaskOE uses HENet as backbone to extract multi-view features, converts to BEV features, then SparseBEVOEHead drives three task heads: detection head (SparseBEVOE detection, 10-class 3D boxes), map head (MapTR-style vectorized map, 3 element classes), occupancy head (FlashOcc-style semantic occupancy, 18 classes). This task uses three-stage Float training: stage1 jointly trains det+map+occ (backbone), stage2 freezes backbone and trains map+occ heads only (`lr=1e-5`), merge stage combines stage1 (det) and stage2 (map/occ) weights into a complete model. `use_lidar_gt=True`, `defer_vectormap=True` (map generated online).

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseMultitaskOE | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny | MMFPN | det detection boxes `(B,N,cls+reg)`; map vectorized map `(B,L,P,2)`; occ occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5434 | 0.5325 | — | 0.5281 |
|  | mAP | — | — | — | — |
|  | chamfer mAP (MAP) | 0.592 | 0.5838 | — | 0.5833 |
|  | Occ mIoU | 0.3197 | 0.3277 | — | 0.3273 |

> Data measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS for J6M/J6P is single-core eight-thread; J6B is single-core dual-thread; Latency is single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 27.50 | 36.71 | 181.90 |
| J6P | 17.76 | 188.07 | 210.80 |
| J6B | 164.46 | 9.53 | 301.60 |

---

## Model Overview

### Core Design

SparseMultitaskOE uses HENet as backbone to extract multi-view features, converts to BEV features, then SparseBEVOEHead drives three task heads: detection head (SparseBEVOE detection, 10-class 3D boxes), map head (MapTR-style vectorized map, 3 element classes), occupancy head (FlashOcc-style semantic occupancy, 18 classes). This task uses three-stage Float training: stage1 jointly trains det+map+occ (backbone), stage2 freezes backbone and trains map+occ heads only (`lr=1e-5`), merge stage combines stage1 (det) and stage2 (map/occ) weights into a complete model. `use_lidar_gt=True`, `defer_vectormap=True` (map generated online).

- **Task type**: Multitask fusion (3D object detection + vectorized map construction + occupancy grid prediction).
- **backbone**: HENet-tiny (pretrained).
- **neck**: MMFPN.
- **Detection head**: `SparseBEVOEHead` (`SparseBEVOEEncoder` + `SparseBEVOERefinementModule`, 10-class 3D detection boxes, `num_classes=10`).
- **Map head**: `SparseMapPerceptionDecoder` (`SparseMapHead`, 3-class vectorized map elements, `map_classes=[divider,ped_crossing,boundary]`).
- **Occupancy head**: Semantic occupancy prediction, 18 classes (`num_classes_occ=18`).
- **BEV range**: `point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]` (det/occ), `map_point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]` (map), `occ_bev_size=(40,40,0.625)`.
- **Model input**: 6-camera multi-view images (B,6,3,256,704) + lidar point cloud (B,N,D).
- **Model output**: det 3D detection boxes + map vectorized map elements + occ occupancy grid semantics.

**Deployment notes**: HBIR export enables `enable_vpu=True`; compilation uses `input_source=compile_cfg["input_source"]`.

### Official Repo and Paper

Official repo: det SparseBEV/map MapTR/occ FlashOcc
https://github.com/MCG-NJU/SparseBEV, https://github.com/hustvl/MapTR, https://github.com/Yzichen/FlashOCC
Paper: https://arxiv.org/abs/2308.09244, https://arxiv.org/abs/2208.14437, https://arxiv.org/abs/2311.12058

Note: Camera backbone HENet is HEAL in-house; upstream papers for det/map/occ are SparseBEV/MapTR/FlashOcc respectively.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14095
