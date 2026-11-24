# PointPillars (Car)

PointPillars voxelizes point clouds into pillars; `PillarFeatureNet` learns pillar features, which are scattered into pseudo-images via `PointPillarScatter`, then multi-scale feature extraction and 3D box regression are performed by a SECOND-style FPN (`SECONDNeck`) and `PointPillarsHead`; training uses joint supervision with FocalLoss + SmoothL1Loss + direction classification.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| PointPillars | Single-frame LiDAR point cloud `(N,4)` | PointPillarScatter | SECONDNeck | Car 3D detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | 3D AP (Car) | 0.7731 | 0.7569 | 0.7709 | 0.7707 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 22.51 | 213.05 | 55.80 |
| J6P | 20.47 | 345.73 | 55.80 |
| J6B | 1353.52 | 2.75 | 63.40 |

---

## Model Overview

### Core Design

PointPillars voxelizes point clouds into pillars; `PillarFeatureNet` learns pillar features, which are scattered into pseudo-images via `PointPillarScatter`, then multi-scale feature extraction and 3D box regression are performed by a SECOND-style FPN (`SECONDNeck`) and `PointPillarsHead`; training uses joint supervision with FocalLoss + SmoothL1Loss + direction classification.

- **Task type**: 3D object detection (LiDAR point cloud 3D Object Detection).
- **backbone**: `PointPillarScatter` (scatters 64-dim pillar features learned by `PillarFeatureNet` into `H×W×64` pseudo-image feature maps by coordinates, `use_horizon_pillar_scatter=True`).
- **neck**: `SECONDNeck` (SECOND-style FPN: downsample layer channels `[64, 128, 256]`, strides `[2, 2, 2]`; upsample layer channels `[128, 128, 128]`, strides `[1, 2, 4]`; outputs three-scale features concatenated to 384 channels).
- **Point cloud preprocessing**: `PointPillarsPreProcess` (voxelization: `pc_range=[0, -39.68, -3, 69.12, 39.68, 1]`, `voxel_size=[0.16, 0.16, 4]`, max 100 points per voxel, max 12000 voxels).
- **Feature extraction**: `PillarFeatureNet` (`num_filters=(64)`, 4-dim input, MLP + max-pool to 64-dim pillar features).
- **Detection head**: `PointPillarsHead` (`in_channels=384`, `use_direction_classifier=True`, outputs classification + box regression + direction classification).
- **Anchor generation**: `Anchor3DGeneratorStride`: Car anchor size `1.6×3.9×1.56`, stride `[0.32, 0.32, 0.0]`, offset `[0.16, -39.52, -1.78]`, rotation angles `[0, 1.57]`; match threshold `0.6`, mismatch threshold `0.45`.
- **Post-processing**: `PointPillarsPostProcess`: NMS (`nms_iou_threshold=0.5`, `score_threshold=0.4`, `nms_pre_max_size=1000`, `nms_post_max_size=300`, `max_per_img=100`).
- **Loss**: `FocalLoss` (classification, `alpha=0.25, gamma=2.0, weight=1.0`) + `SmoothL1Loss` (box regression, `beta=1/9, weight=2.0`) + `CrossEntropyLoss` (direction, `weight=0.2`).
- **Model input**: Single-frame LiDAR point cloud, shape `(N, 4)`, N = number of points (deployment input padded to `150000` points), 4 dims = `[x, y, z, intensity]`.
- **Model output**: Car 3D boxes per frame (`x, y, z, w, l, h, θ`) + score + direction.
- **Classes**: `["Car"]`.

**Deployment notes**: The PointPillars deployment graph includes full voxelization + feature extraction + backbone + head + post-processing. HBIR export enables `enable_vpu=True`; compilation uses `input_source=["ddr"]` (point cloud read from DDR, not pyramid image input), unlike image-based tasks. This task has no subgraph export; only one `deploy.py`.

### Official Repo and Paper

Official repo: https://github.com/nutonomy/second.pytorch
Paper: https://arxiv.org/abs/1812.05784v1

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14086
