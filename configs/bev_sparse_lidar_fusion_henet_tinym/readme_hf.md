# SparseBEV + Lidar Fusion + HENet-tiny

SparseBevFusion extends SparseBEV with a lidar branch: HENet-tiny extracts camera features, `CenterPointDetector` (`PillarFeatureNet` + `PointPillarScatter`) processes lidar point clouds, `DeformableFeatureAggregationLiF` (with `InstanceFuseModule`) fuses camera-lidar features in BEV space, and `SparseBEVHead` performs sparse query detection.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseBevFusion | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny (camera) + PointPillarScatter (lidar) | MMFPN + DenseDepthNet + DFA-LiF | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.6704 | 0.651 | 0.6647 | 0.6628 |
|  | mAP | 0.6086 | 0.5853 | 0.6076 | 0.5961 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 21.62 | 56.80 | 149.80 |
| J6P | 16.58 | 309.65 | 161.50 |
| J6B | 68.80 | 24.34 | 136.40 |

---

## Model Overview

### Core Design

SparseBevFusion extends SparseBEV with a lidar branch: HENet-tiny extracts camera features, `CenterPointDetector` (`PillarFeatureNet` + `PointPillarScatter`) processes lidar point clouds, `DeformableFeatureAggregationLiF` (with `InstanceFuseModule`) fuses camera-lidar features in BEV space, and `SparseBEVHead` performs sparse query detection.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection, camera + lidar fusion).
- **backbone**: HENet-tiny (`type=HENet`, `in_channels=3`, `embed_dims=[64,128,192,384]`, multi-view camera feature extraction); lidar branch `CenterPointDetector` with `PillarFeatureNet` (`num_input_features=5`) + `PointPillarScatter` + HENet (`in_channels=64`) for pillar features (`voxel_size=[0.2,0.2,8]`).
- **neck**: `MMFPN` (camera branch, `in_strides=[2,4,8,16,32]`→`out_strides=[4,8,16,32]`) + `DenseDepthNet` (dense depth auxiliary) + `DeformableFeatureAggregationLiF` (with `InstanceFuseModule`, BEV camera-lidar feature fusion).
- **Detection head**: `SparseBEVHead` (`MemoryBank` + `SparseBEVEncoder`, sparse query + `DeformableFeatureAggregationLiF` fusion, `num_classes=10`, `num_decoder=6`, `num_anchors=384`).
- **Loss function**: `FocalLoss` (cls) + `L1Loss` (reg) + `CrossEntropyLoss` (cns) + `GaussianFocalLoss` (yns).
- **Model input**: 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` (`load_dim=5`, `use_dim=[0,1,2,3,4]`, `num_lidar_sweeps=9`, `voxel_size=[0.2,0.2,8]`, `max_voxels=(30000,40000)`).
- **Model output**: 10-class 3D bounding boxes (`num_classes=10`), `(B,N,cls+reg)`.

### Official Repo and Paper

Official repo: https://github.com/yichen928/SparseFusion
Paper: https://arxiv.org/abs/2304.14340

Note: The camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.
