# BEVFormer (ResNet-50)

`bevformer_tiny_resnet50_detection` is a BEVFormer tiny nuScenes BEV 3D object detection model with ResNet-50 backbone + FPN neck, covering the full pipeline from float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/bevformer_tiny_resnet50_detection/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BevFormer | 6-camera multi-view image sequence `(B,6,3,480,800)` | ResNet-50 | FPN | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`). Metric: NDS (nuScenes detection composite metric, monitor metric, higher is better).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3739 | 0.3607 | 0.3734 | 0.3669 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 21.76 | 46.52 | 112.80 |
| J6P | 14.15 | 274.97 | 113.80 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifacts:
- HBIR: `./tmp_models/bevformer_tiny_resnet50_detection/qat.bc`
- HBM: `./tmp_models/bevformer_tiny_resnet50_detection/compile/model.hbm`

---

## Model Overview

### Core Design

BEVFormer extracts BEV features from multi-camera sequences via learnable spatiotemporal attention (Temporal Self-Attention + Spatial Cross-Attention): ResNet-50 + FPN extract multi-scale image features, BevFormerViewTransformer projects to BEV, BEVFormerEncoder fuses temporal and spatial information, and BEVFormerDetDecoder decodes 3D bounding boxes.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: ResNet-50 (`ResNet50`, `include_top=False`, ImageNet pretrained `num_classes=1000`).
- **neck**: FPN (`FPN`, multi-scale feature pyramid, `out_strides=[32]`, `out_channels=[256]`).
- **Detection head**: `BEVFormerDetDecoder` (DETR-style decoder + Hungarian matching).
- **Loss function**: `BevFormerCriterion` (FocalLoss + L1Loss, matched via BevFormerHungarianAssigner3D).
- **Model input**: 6-camera multi-view image sequence, `(B,6,3,480,800)` (original `orig_shape=(3,900,1600)` → resize `(3,450,800)` → pad to `(3,480,800)`, `num_views=6`, training `queue_length=3`, evaluation `queue_length=1`, `NuscenesBevSequenceDataset`).
- **Model output**: 3D bounding boxes on BEV features (class + center + size + orientation), `num_query=900`, `num_classes=10`, decoded via `BevFormerProcess` (`max_num=300`, `score_threshold=0.3`).

### Official Repo and Paper

Official repo: https://github.com/fundamentalvision/BevFormer
Paper: https://arxiv.org/abs/2203.17270

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14101

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes v1.0-trainval packed lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training lmdb (base) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation lmdb (base, nuScenes val) |

> Note: This task generates map GT online via `VectorizedLocalMap` (`map_path=meta_rootdir`); it does not read map lmdb, so map lmdb packing is not required.

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (registration required, v1.0-trainval). After extraction, includes `samples/` (annotated keyframes, 6 cameras), `sweeps/` (intermediate frames), `maps/` (map expansion), `v1.0-trainval/` (metadata json: annotation, scene, sample, ego_pose, etc.). Place raw data in `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir`):

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
├── samples/          # Keyframes, 6 cameras (CAM_FRONT/FRONT_LEFT/...)
├── sweeps/           # Intermediate frames, 6 cameras
├── maps/             # Map expansion data
└── v1.0-trainval/    # Metadata json (annotation/scene/sample/ego_pose, etc.)
```

#### Packing Commands

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and pack directly from the official extracted directory (do not run legacy scripts such as create_data_*.py).

Packing order: base first; occ after base requires matching version/split; lidar/map only require matching version/split. This task uses base lmdb only (`{split}_lmdb`); lidar/map/occ packing is optional.

`NuscenesPacker` (`tools/datasets/nuscenes_packer.py`) reads the official extracted directory and produces `{split}_lmdb`. Base packer train + val commands for this task:

```bash
# Pack train split (base)
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# Pack val split (base)
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

The other 3 packers and when they are needed (optional for this task):
- **lidar** `NuscenesLidarPacker` (`tools/datasets/nuscenes_lidar_packer.py`): produces `{split}_lidar_lmdb`; required only for lidar tasks; key params `--max-sweeps 10` `--load-dim` `--use_dim`.
- **map** `NuscenesMapGtPacker` (`tools/datasets/nuscenes_map_packer.py`): produces `{split}_map_lmdb`; required only for map tasks; `--map-path` points to meta directory; PV seg mask is generated online during training, not packed.
- **occ** `NuscenesOccPacker` (`tools/datasets/nuscenes_occ_packer.py`): produces `{split}_occ_lmdb`; required only for occ tasks; `occ3d/gts/` from Occ3D-nuScenes official labels requires user download.

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, produces `{split}_lmdb`, matching config `data_rootdir`.
- `--pack-type lmdb`: Pack format is lmdb.
- `--split-name`: `train` or `val` for training or validation set.
- `--version v1.0-trainval`: Data version (packer default is `v1.0-mini`; must override to `v1.0-trainval`).
- `--num-workers`: Parallel packing workers (default 20).

`orig_data/nuscenes` is `--src-data-dir` (raw data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (corresponds to config `data_rootdir`).

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/v1.0-trainval/` layout:

```
./tmp_data/heal_data/nuscenes/v1.0-trainval/
├── train_lmdb/          # train lmdb (base) ← used by this task
├── val_lmdb/            # val lmdb (base) ← used by this task
├── train_lidar_lmdb/    # not used by this task
├── val_lidar_lmdb/      # not used by this task
├── train_map_lmdb/      # not used by this task
├── val_map_lmdb/        # not used by this task
├── train_occ_lmdb/      # not used by this task
├── val_occ_lmdb/        # not used by this task
└── meta/                # Metadata directory (read when generating map GT online)
```

Use visualization scripts to spot-check packing results:

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | Float optimizer | LR schedule | Loss function |
|---|---|---|---|---|---|---|
| 8 | 2 | 24 | 100 | AdamW (lr=4e-4, backbone lr_mult=0.1, weight_decay=0.01) | CosineAnnealingLrUpdater (warmup_lr_ratio=1/3)<br>QAT stage uses StepDecayLrUpdater | BevFormerCriterion |

This task has no cross-task pretraining dependency; float training starts from scratch. Calibration loads this task's float output `float-checkpoint-best.pth.tar`. QAT loads Calibration output `calibration-checkpoint-best.pth.tar`.

```bash
# Float training
python3 tools/train.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage float
# Output:./tmp_models/bevformer_tiny_resnet50_detection/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage calibration
# Output:./tmp_models/bevformer_tiny_resnet50_detection/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage qat
# Output:./tmp_models/bevformer_tiny_resnet50_detection/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/bevformer_tiny_resnet50_detection/deploy.py
# Output:./tmp_models/bevformer_tiny_resnet50_detection/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/bevformer_tiny_resnet50_detection/deploy.py
# Output:./tmp_models/bevformer_tiny_resnet50_detection/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat—just re-run export and compilation on the quantized model. Modify the `march` field in `configs/bevformer_tiny_resnet50_detection/common.py` to the target chip:
> ```python
> march = March.NASH_M   # J6M
> march = March.NASH_P   # J6P
> march = March.NASH_B   # J6B
> ```
> Then re-run `export_hbir.py` and `compile_perf.py` (the two commands above). Compilation artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage hbm
```

Evaluation metric is NDS; evaluation set is nuScenes val. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPUs with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the evaluation set (saved to `./demo/bevformer_tiny_resnet50_detection/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/bevformer_tiny_resnet50_detection/)
python3 tools/gen_demo.py \
  --config configs/bevformer_tiny_resnet50_detection

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage float
```

Visualization results are saved to `./demo/bevformer_tiny_resnet50_detection/`.
