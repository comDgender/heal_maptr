# BEV (ResNet-50)

`bev_vtv2_resnet50` is a nuScenes BEV 3D object detection model with ResNet-50 backbone + FPN neck, covering the full pipeline from float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/bev_vtv2_resnet50/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BEV-VTv2 | 6-camera multi-view images `(B,6,3,256,704)` | ResNet-50 | FPN | BEV grid 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`). Metrics: NDS / mAP (nuScenes detection composite metrics, higher is better).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3603 | 0.3546 | 0.3564 | 0.3539 |
|  | mAP | 0.2833 | 0.2799 | 0.2805 | 0.2797 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 16.38 | 62.06 | 120.90 |
| J6P | 11.43 | 347.06 | 122.60 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifacts:
- HBIR: `./tmp_models/bev_vtv2_resnet50/qat.bc`
- HBM: `./tmp_models/bev_vtv2_resnet50/compile/model.hbm`

---

## Model Overview

### Core Design

This model follows the LSS (Lift-Splat-Shoot) view transformation approach: 6 camera images are processed by ResNet-50 + FPN for multi-scale features, projected to the BEV plane via LSSTransformer, encoded by BevEncoder, and finally predicted on the BEV grid by CenterPointHead for 3D bounding boxes.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: ResNet-50 (`ResNet50`, `include_top=False`, ImageNet pretrained `num_classes=1000`).
- **neck**: FPN (`FPN`, multi-scale feature pyramid, output strides 16/32, `fix_out_channel=256`).
- **Detection head**: `CenterPointHead` (BEV grid center-point detection head).
- **Loss function**: `CenterPointLoss` (GaussianFocalLoss + L1Loss).
- **Model input**: 6-camera multi-view images, `(B,6,3,256,704)` (original `orig_shape=(3,900,1600)` → resize `(3,396,704)` → crop `data_shape=(3,256,704)`, `num_views=6`, `NuscenesBevDataset`).
- **Model output**: 3D bounding boxes on BEV grid (class + center + size + orientation), `CenterPointHead` outputs heatmap + reg/height/dim/rot/vel per task group, `num_classes=10`, decoded via `CenterPointPostProcess` + NMS.

### Official Repo and Paper

Official repo: https://github.com/HuangJunJie2017/BEVDet
Paper: https://arxiv.org/pdf/2112.11790

Note: Backbone is ResNet-50; detection head is CenterPoint.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14085

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
└── meta/                # Metadata directory
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
| 8 | 2 | 40 | 10 | AdamW (lr=2e-4, weight_decay=0.01) | CosineAnnealingLrUpdater (warmup_lr_ratio=1/3) | CenterPointLoss |

This task has no cross-task pretraining dependency; float training starts from scratch. Calibration loads this task's float output `float-checkpoint-best.pth.tar`. QAT loads Calibration output `calibration-checkpoint-best.pth.tar`.

```bash
# Float training
python3 tools/train.py \
  --config configs/bev_vtv2_resnet50 \
  --stage float
# Output:./tmp_models/bev_vtv2_resnet50/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/bev_vtv2_resnet50 \
  --stage calibration
# Output:./tmp_models/bev_vtv2_resnet50/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/bev_vtv2_resnet50 \
  --stage qat
# Output:./tmp_models/bev_vtv2_resnet50/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/bev_vtv2_resnet50/deploy.py
# Output:./tmp_models/bev_vtv2_resnet50/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/bev_vtv2_resnet50/deploy.py
# Output:./tmp_models/bev_vtv2_resnet50/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat—just re-run export and compilation on the quantized model. Modify the `march` field in `configs/bev_vtv2_resnet50/common.py` to the target chip:
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
  --config configs/bev_vtv2_resnet50 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/bev_vtv2_resnet50 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/bev_vtv2_resnet50 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/bev_vtv2_resnet50 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/bev_vtv2_resnet50 \
  --stage hbm
```

Evaluation metrics are NDS and mAP; evaluation set is nuScenes val. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPUs with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the evaluation set (saved to `./demo/bev_vtv2_resnet50/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/bev_vtv2_resnet50/)
python3 tools/gen_demo.py \
  --config configs/bev_vtv2_resnet50

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/bev_vtv2_resnet50 \
  --stage float
```

Visualization results are saved to `./demo/bev_vtv2_resnet50/`.
