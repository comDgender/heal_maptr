# CenterPoint (PointPillars)

`centerpoint_pointpillar` is a CenterPoint 3D detection model based on PointPillars, using `CenterPointDetector` (with CBGS data augmentation), lidar lmdb only, covering the full pipeline from float training → quantization calibration → QAT → HBIR export → HBM compilation → inference.

Model config directory: `configs/centerpoint_pointpillar/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| CenterPoint | lidar point cloud `(B,N,5)` | PointPillarScatter | SECONDNeck | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb`). Metrics: NDS / mAP.

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

Compilation artifacts:
- HBIR: `./tmp_models/centerpoint_pointpillar/qat.bc`
- HBM: `./tmp_models/centerpoint_pointpillar/compile/model.hbm`

---

## Model Overview

### Core Design

CenterPoint (PointPillars variant) voxelizes lidar point clouds into pillars, learns pillar features via `PillarFeatureNet` and scatters them into a 2D pseudo-image via `PointPillarScatter`, extracts multi-scale features via `SECONDNeck`, and `CenterPointHead` regresses box center, size, orientation, and velocity in an anchor-free manner to output 3D bounding boxes. Training uses CBGS (Class-Balanced Grouping and Sampling) data augmentation.

- **Task type**: 3D object detection (3D Object Detection, lidar).
- **backbone**: `PillarFeatureNet` (`num_input_features=5`, `num_filters=(64,)`, learns pillar features) + `PointPillarScatter` (`num_input_features=64`, `use_horizon_pillar_scatter=True`, scatters pillar features into 2D pseudo-image).
- **neck**: `SECONDNeck` (`in_feature_channel=64`, `down_layer_nums=[3,5,5]`, `down_layer_channels=[64,128,256]`, `up_layer_channels=[128,128,128]`, multi-scale feature extraction).
- **Detection head**: `CenterPointHead` (anchor-free, `common_heads=dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2))`, `with_velocity=True`).
- **Loss function**: `CenterPointLoss` (`GaussianFocalLoss` cls + `L1Loss` reg).
- **Model input**: lidar point cloud `(B,N,5)` (`point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`, `voxel_size=[0.2,0.2,8]`, `max_num_points=20`, `max_voxels=(30000,40000)`, `load_dim=5`, `use_dim=[0,1,2,3,4]`, `num_sweeps=9`).
- **Model output**: 10-class 3D bounding boxes (nuScenes 10 classes, `class_names` = `[car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone]`), `(B,N,cls+reg)`.

### Official Repo and Paper

Official repo: https://github.com/tianweiy/CenterPoint
Paper: https://arxiv.org/abs/2006.11275

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14088
---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes packed lidar lmdb only (`NuscenesLidarDataset`, with CBGS augmentation during training). Data paths read by the model (all joined from `data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval"` in `common.py`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | Training lidar lmdb (`NuscenesLidarDataset` + `CBGSDataset`) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | Evaluation lidar lmdb (`NuscenesLidarDataset`) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config consumption).

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (registration required). Download the full **v1.0-trainval** dataset; after extraction includes `samples/` (annotated keyframes, 6 cameras), `sweeps/` (intermediate frames), `maps/` (map expansion), `v1.0-trainval/` (metadata json: annotation, scene, sample, ego_pose, etc.). This task (lidar detection) only uses `sweeps/` (lidar point clouds) and `v1.0-trainval/` metadata, but the packer requires the full nuScenes directory structure.

Extract the above into `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# Place official extracted content in this directory
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
```

#### Directory Layout After Extraction

```
./tmp_data/orig_data/nuscenes/
├── samples/              # Keyframe images (6 camera subdirs)
├── sweeps/               # Intermediate frames (includes LiDAR LIDAR_TOP)
├── maps/                 # Map expansion (expansion + png)
└── v1.0-trainval/        # Metadata json (annotation, scene, sample, ego_pose, etc.)
```

#### Packing Commands

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and pack directly from the official extracted directory (do not run create_data_*.py).

nuScenes has four packers:
- **base** `nuscenes_packer.py` (`NuscenesPacker`) packs base camera lmdb (`{split}_lmdb`) for camera detection tasks.
- **lidar** `nuscenes_lidar_packer.py` (`NuscenesLidarPacker`) packs lidar point cloud lmdb (`{split}_lidar_lmdb`) for lidar detection tasks.
- **map** `nuscenes_map_packer.py` (`NuscenesMapGtPacker`) packs vectorized map lmdb (`{split}_map_lmdb`) for online map tasks (PV seg mask not packed, generated online during training).
- **occ** `nuscenes_occ_packer.py` (`NuscenesOccPacker`) packs occupancy prediction lmdb (`{split}_occ_lmdb`) for occ tasks (must run after base with matching version/split; requires Occ3D labels).

Dependency order: **base first**; occ must follow base with matching version/split; lidar/map only require matching version/split.

**This task uses `{split}_lidar_lmdb` only; base/map/occ packing is optional** (only the lidar packer is required). Below are the lidar packer commands needed for this task; other packer commands can be found in other config readmes if needed.

`NuscenesLidarPacker` (lidar) packs lidar point cloud lmdb (`{split}_lidar_lmdb`):

```bash
# Pack lidar train split
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# Pack lidar val split
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, corresponding to config `data_rootdir`.
- `--pack-type lmdb`: Pack format is lmdb.
- `--split-name`: `train` or `val` for training or validation set.
- `--version v1.0-trainval`: nuScenes dataset version (packer default is `v1.0-mini`; must override to `v1.0-trainval`).
- `--num-workers`: Parallel packing workers (default 20).
- `--max-sweeps`: Maximum sweep count for lidar packer (default 10).
- `--load-dim`: Raw point cloud dimension for lidar packer (default 5).

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` layout (this task uses lidar lmdb only under `v1.0-trainval/`; base/map/occ lmdb are absent if not packed):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # Training base lmdb(not used by this task)
│   ├── val_lmdb/            # Evaluation base lmdb(not used by this task)
│   ├── train_lidar_lmdb/    # train lidar lmdb         ← used by this task
│   ├── val_lidar_lmdb/      # val lidar lmdb           ← used by this task
│   ├── train_map_lmdb/      # train map lmdb (not used by this task)
│   ├── val_map_lmdb/        # val map lmdb (not used by this task)
│   ├── train_occ_lmdb/      # train occ lmdb (not used by this task)
│   └── val_occ_lmdb/        # val occ lmdb (not used by this task)
├── meta/                    # Precomputed map expansion meta directory
└── osm/                     # OSM vector map (for map tasks)
```

Lidar lmdb uses `nuscenes_lidar_viz.py` (packing validation, reuses `lidar_det_visualize` from `heal/visualize/lidar_det.py`, renders BEV point cloud + 3D GT boxes, no model required):

```bash
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar
```

Visualization PNGs are saved to `--save-dir` (default `WORKSPACE/nuscenes_lidar_viz`; must be a local writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as save target). Manually verify point clouds and GT boxes are aligned.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | QAT training epochs | Float optimizer | QAT optimizer | LR schedule | Loss function |
|---|---|---|---|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 4 | 20 | 100 | 10 | AdamW (`lr=2e-4`, `weight_decay=1e-2`) | SGD (`lr=2e-4`, `momentum=0.9`, `weight_decay=0.0`) | `CosLrUpdater` (Float) | `CenterPointLoss` (`GaussianFocalLoss` cls + `L1Loss` reg) |

This task's float training starts from scratch with no cross-task pretraining dependency. Calibration loads this task's float output; QAT loads this task's Calibration output, in a serial chain.

```bash
# Float training
python3 tools/train.py \
  --config configs/centerpoint_pointpillar \
  --stage float
# Output:./tmp_models/centerpoint_pointpillar/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/centerpoint_pointpillar \
  --stage calibration
# Output:./tmp_models/centerpoint_pointpillar/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/centerpoint_pointpillar \
  --stage qat
# Output:./tmp_models/centerpoint_pointpillar/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/centerpoint_pointpillar/deploy.py
# Output:./tmp_models/centerpoint_pointpillar/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/centerpoint_pointpillar/deploy.py
# Output:./tmp_models/centerpoint_pointpillar/compile/model.hbm
```

> Export and compilation use `enable_vpu` passed via `enable_vpu=True` in `common.py` (imported by deploy.py); compilation input `input_source=["ddr","ddr"]` (lidar point clouds read from DDR, not pyramid image input), unlike image-based tasks.

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat—just re-run export and compilation on the quantized model. Modify the `march` field in `configs/centerpoint_pointpillar/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then re-run `export_hbir.py` and `compile_perf.py`. Compilation artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage hbm
```

Evaluation metrics are NDS and mAP; evaluation set is nuScenes val (`val_lidar_lmdb`, `val_nuscenes_metric`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPUs with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference samples from the evaluation set (saved to `./demo/centerpoint_pointpillar/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/centerpoint_pointpillar/)
python3 tools/gen_demo.py \
  --config configs/centerpoint_pointpillar

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/centerpoint_pointpillar \
  --stage float
```

Visualization results are saved to `./demo/centerpoint_pointpillar/`.
