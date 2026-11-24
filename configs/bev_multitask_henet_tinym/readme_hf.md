# BEV Multitask + HENet-tiny

A camera-only BEV multitask model for 3D detection, occupancy, stop-line detection, and online lane-level mapping.

BEV Multitask + HENet-tiny lifts six surround cameras into a shared bird’s-eye feature and runs five heads together—dynamic/static objects, occupancy, stop lines, and vectorized map—so scene cues stay consistent. This release is calibrated for Horizon J6M / J6P deployment without lidar at inference.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BEVFusionMultitask | 6-camera multi-view images `(B,6,3,512,960)` | HENet-tiny | FPN + SingleBevFormerViewTransformer | dynamic/static 3D boxes `(B,N,cls+reg)`; occ voxel semantics; discrete-map 2D boxes; map polylines |

### Accuracy Metrics

| March | Metric | float | calibration | hbm |
| --- | --- | --- | --- | --- |
| J6M | Dynamic NDS | 0.4553 | 0.4480 | 0.4535 |
|  | Static NDS | 0.4873 | 0.4839 | 0.4842 |
|  | Occ mIoU | 0.3422 | 0.3401 | 0.3399 |
|  | stop_line NDS | 0.4802 | 0.4762 | 0.4759 |
|  | MapTR mAP | 0.5486 | 0.5405 | 0.5376 |

> Results measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage.
>
> HEAL version: heal 0.14.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 77.82 | 13.21 | 386.90 |
| J6P | 47.65 | 76.17 | 373.50 |

---

## Model Overview

### Core Design

Six surround cameras are encoded by HENet-tiny, then lifted into a compact BEV canvas (about 100 m around the ego) with a single-frame BevFormer view transformer. That shared BEV feature fans out to five lightweight heads: SparseBEV-style queries for dynamic and static 3D boxes, a voxel occupancy head, a stop-line detector, and a MapTR-style polyline head for dividers, crossings, and boundaries — so detection, free space, and map cues stay consistent without running five separate networks.

- **Task type**: BEV multitask perception (dynamic det + static det + occupancy + discrete map + online map).
- **backbone**: HENet-tiny (`type=HENet`, `block_nums=[4,3,8,6]`, `embed_dims=[64,128,192,384]`, `GroupDWCB`/`AltDWCB`/`DWCB` blocks, `S2DDown` downsampling), initialized from the ImageNet-pretrained `henet_tinym` float checkpoint.
- **neck**: `FPN` (`in_strides=[32]` → `out_strides=[32]`, `_dim_=256`) + `BevFuseModule` (`input_c=384+256` → `fuse_c=256`; lidar branch disabled, `lidar_input=False`).
- **Dynamic det head**: `DenseBEVSparseHead` (8 classes, `num_anchor=384`, `num_decoder=6`, `MemoryBank` with `num_temp_instances=128` for tracking).
- **Static det head**: `DenseBEVSparseHead` (2 classes: `traffic_cone`, `barrier`; `num_anchor=200`).
- **Occupancy head**: `BevformerOccDetDecoder` + `BEVOCCHead2D` (`Dz=16`, 18 classes).
- **Discrete map head**: `BEVFormerDet2DDecoder` (1 class `stop_line`, `num_query=100`).
- **Online map head**: `MaptrDetDecoder` + `MapInstanceDetectorHead` (3 classes: `divider`, `ped_crossing`, `boundary`; 20 points per polyline).
- **Loss functions**: det — `FocalLoss` (cls) + `L1Loss` (reg) + `CrossEntropyLoss` (cns) + `GaussianFocalLoss` (yns); occ — `CrossEntropyLoss`; maptr — `MapTRCriterion`; discrete map — `BevFormer2DCriterion`.
- **Model input**: 6-camera multi-view images `(B,6,3,512,960)` (YUV444 + `Normalize(128,128)`; camera-only, no lidar at inference).
- **Model output**: dynamic/static 3D boxes + occ voxel semantics + stop-line boxes + map polylines.

### Official Repo and Paper

SparseBEV-style det head repo: https://github.com/MCG-NJU/SparseBEV
Paper: https://arxiv.org/abs/2308.09244
MapTR repo: https://github.com/hustvl/MapTR
Paper: https://arxiv.org/abs/2208.14437

Note: The camera backbone HENet and the multitask assembly are HEAL in-house implementations; the official repos use different backbones and single-task setups.
