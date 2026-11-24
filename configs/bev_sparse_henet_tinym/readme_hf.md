# SparseBEV + HENet-tiny

SparseBEV performs 3D detection in BEV space via sparse queries: HENet-tiny extracts multi-view camera features, `MMFPN` fuses multi-scale features, `DenseDepthNet` uses lidar point clouds for dense depth supervision during training; `SparseBEVHead` refines sparse queries layer by layer via `MemoryBank` + `SparseBEVEncoder` (with `AsymmetricFFN`, `DeformableFeatureAggregation`, `SparseBEVRefinementModule`) and outputs detection boxes.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseBEV | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny | MMFPN + DenseDepthNet | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5414 | 0.5319 | — | 0.5307 |
|  | mAP | 0.4338 | 0.4265 | — | 0.4235 |

> Results measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 10.43 | 98.46 | 72.80 |
| J6P | 7.38 | 527.73 | 85.90 |
| J6B | 23.16 | 44.28 | 71.00 |

---

## Model Overview

### Core Design

SparseBEV performs 3D detection in BEV space via sparse queries: HENet-tiny extracts multi-view camera features, `MMFPN` fuses multi-scale features, `DenseDepthNet` uses lidar point clouds for dense depth supervision during training; `SparseBEVHead` refines sparse queries layer by layer via `MemoryBank` + `SparseBEVEncoder` (with `AsymmetricFFN`, `DeformableFeatureAggregation`, `SparseBEVRefinementModule`) and outputs detection boxes.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: HENet-tiny (`type=HENet`, `in_channels=3`, 4 stages `[4,3,8,6]` blocks, `embed_dims=[64,128,192,384]`, `GroupDWCB`/`AltDWCB`/`DWCB` blocks, `S2DDown` downsampling; multi-view camera feature extraction).
- **neck**: `MMFPN` (`in_strides=[2,4,8,16,32]`→`out_strides=[4,8,16,32]`, multi-scale fusion) + `DenseDepthNet` (dense depth supervision from lidar point clouds during training, `num_depth_layers=3`).
- **Detection head**: `SparseBEVHead` (`MemoryBank` + `SparseBEVEncoder` + `SparseBEVEncoder`, with `AsymmetricFFN`, `DeformableFeatureAggregation`, `SparseBEVRefinementModule`, layer-wise sparse query refinement, `num_classes=10`, `num_decoder=6`, `num_anchors=384`).
- **Loss function**: `FocalLoss` (cls) + `L1Loss` (reg) + `CrossEntropyLoss` (cns) + `GaussianFocalLoss` (yns).
- **Model input**: 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` (`load_dim=5`, `use_dim=[0,1,2,4]`, `num_lidar_sweeps=0`, dense depth GT auxiliary).
- **Model output**: 10-class 3D bounding boxes (`num_classes=10`), `(B,N,cls+reg)`.

### Official Repo and Paper

Official repo: https://github.com/MCG-NJU/SparseBEV
Paper: https://arxiv.org/abs/2308.09244

Note: The camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14074
