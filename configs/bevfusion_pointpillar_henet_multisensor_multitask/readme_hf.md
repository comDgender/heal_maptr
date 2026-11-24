# BevFusion+PointPillar+HENet Multisensor Multitask

The BevFusion multisensor multitask model extracts BEV features via dual branches: the camera branch uses HENet to extract multi-view features, then BevFormer ViewTransformer to BEV; the lidar branch uses PointPillars (`PillarFeatureNet` + `PointPillarScatter` + `SECONDNeck`) to voxelize point clouds and extract BEV features. Fused BEV features feed CenterPoint detection head (3D object detection) and occupancy head (semantic occupancy prediction) for joint det+occ training.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BevFusion | 6-camera multi-view images `(B,6,3,512,960)` + lidar point cloud `(B,N,5)` | PointPillar (lidar) + HENet (camera) | FPN (camera) + SECONDNeck (lidar) | det bounding boxes `(B,N,cls+reg)`; occ occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.6421 | 0.6301 | — | 0.6294 |
|  | mAP | 0.5825 | 0.5724 | — | 0.5726 |
|  | Occ mIoU | 0.5187 | 0.52 | — | 0.5206 |

> Results measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 23.68 | 49.91 | 186.40 |
| J6P | 16.75 | 287.11 | 195.00 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

The BevFusion multisensor multitask model extracts BEV features via dual branches: the camera branch uses HENet to extract multi-view features, then BevFormer ViewTransformer to BEV; the lidar branch uses PointPillars (`PillarFeatureNet` + `PointPillarScatter` + `SECONDNeck`) to voxelize point clouds and extract BEV features. Fused BEV features feed CenterPoint detection head (3D object detection) and occupancy head (semantic occupancy prediction) for joint det+occ training.

- **Task type**: Multisensor multitask fusion (3D object detection + occupancy grid prediction).
- **backbone**: Camera HENet (multi-view feature extraction) + lidar PointPillars (`PillarFeatureNet` + `PointPillarScatter`).
- **neck**: Camera FPN + lidar SECONDNeck.
- **Detection head**: CenterPoint detection head, outputting 10-class 3D bounding boxes + velocity (`num_classes=10`).
- **Occupancy head**: Semantic occupancy prediction, 18 classes (`num_classes_occ=18`).
- **BEV range**: `bev_size=(51.2, 51.2, 0.8)`, `bev_size_occ=(40, 40)`; `point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`.
- **Model input**: 6-camera multi-view images (B,6,3,512,960) + lidar point cloud (B,N,D).
- **Model output**: det 3D bounding boxes + occ occupancy grid semantics.

**Deployment note**: HBIR export enables `enable_vpu=True`; compilation uses `input_source="ddr, ddr, pyramid, ddr, ddr, ddr, ddr"` (DDR preferred; lidar features read from DDR).

### Official Repo and Paper

Official repo: https://github.com/mit-han-lab/bevfusion
Paper: https://arxiv.org/abs/2205.13542

Note: Camera branch is BevFormer, lidar branch is PointPillars/CenterPoint; camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14092
