# CenterPoint (PointPillars)

CenterPoint (PointPillars variant) voxelizes lidar point clouds into pillars, learns pillar features via `PillarFeatureNet` and scatters them into a 2D pseudo-image via `PointPillarScatter`, extracts multi-scale features via `SECONDNeck`, and `CenterPointHead` regresses box center, size, orientation, and velocity in an anchor-free manner to output 3D bounding boxes. Training uses CBGS (Class-Balanced Grouping and Sampling) data augmentation.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| CenterPoint | lidar point cloud `(B,N,5)` | PointPillarScatter | SECONDNeck | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5865 | 0.5703 | 0.5853 | 0.5846 |
|  | mAP | 0.474 | 0.4487 | 0.4699 | 0.4693 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 9.20 | 183.98 | 51.10 |
| J6P | 7.60 | 831.31 | 47.20 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

CenterPoint (PointPillars variant) voxelizes lidar point clouds into pillars, learns pillar features via `PillarFeatureNet` and scatters them into a 2D pseudo-image via `PointPillarScatter`, extracts multi-scale features via `SECONDNeck`, and `CenterPointHead` regresses box center, size, orientation, and velocity in an anchor-free manner to output 3D bounding boxes. Training uses CBGS (Class-Balanced Grouping and Sampling) data augmentation.

- **Task type**: 3D object detection (3D Object Detection, lidar).
- **backbone**: `PillarFeatureNet` (`num_input_features=5`, `num_filters=(64)`, learns pillar features) + `PointPillarScatter` (`num_input_features=64`, `use_horizon_pillar_scatter=True`, scatters pillar features into 2D pseudo-image).
- **neck**: `SECONDNeck` (`in_feature_channel=64`, `down_layer_nums=[3,5,5]`, `down_layer_channels=[64,128,256]`, `up_layer_channels=[128,128,128]`, multi-scale feature extraction).
- **Detection head**: `CenterPointHead` (anchor-free, `common_heads=dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2))`, `with_velocity=True`).
- **Loss function**: `CenterPointLoss` (`GaussianFocalLoss` cls + `L1Loss` reg).
- **Model input**: lidar point cloud `(B,N,5)` (`point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`, `voxel_size=[0.2,0.2,8]`, `max_num_points=20`, `max_voxels=(30000,40000)`, `load_dim=5`, `use_dim=[0,1,2,3,4]`, `num_sweeps=9`).
- **Model output**: 10-class 3D bounding boxes (`class_names` = `[car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone]`), `(B,N,cls+reg)`.

### Official Repo and Paper

Official repo: https://github.com/tianweiy/CenterPoint
Paper: https://arxiv.org/abs/2006.11275

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14088
