# PETR (EfficientNet-b3)

`petr_efficientnetb3` is a nuScenes BEV 3D object detection model based on the Detr3d framework with an EfficientNet-b3 backbone and PETR detection head, covering the full pipeline from float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/petr_efficientnetb3/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| PETR | 6-camera multi-view images `(B,6,3,512,1408)` | EfficientNet-b3 | — | Detection boxes `(B,900,21)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`). Metrics: NDS / mAP (nuScenes detection composite metrics; higher is better).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3881 | 0.3679 | 0.38 | 0.38 |
|  | mAP | 0.3031 | 0.2807 | 0.2942 | 0.2942 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 33.00 | 30.52 | 101.40 |
| J6P | 21.58 | 186.30 | 105.40 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compiled artifact paths:
- HBIR: `./tmp_models/petr_efficientnetb3/qat.bc`
- HBM: `./tmp_models/petr_efficientnetb3/compile/model.hbm`

---

## Model Overview

### Core Design

PETR introduces 3D position into Transformer decoding: EfficientNet-b3 extracts multi-view image features, which are associated with 3D spatial positions via 3D positional encoding (SinePositionalEncoding3D); PETRTransformer queries interact directly with 3D position-encoded features to predict 3D detection boxes, without explicit BEV feature construction.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False` removes the classification head).
- **neck**: — (no standalone neck; backbone features feed directly into `PETRHead`).
- **Detection head**: `PETRHead` + `PETRTransformer` + `PETRDecoder` (3D position-aware Transformer decoder, `num_query=900`, `num_views=6`, `embed_dims=256`).
- **Loss function**: FocalLoss (cls) + L1Loss (reg), with Detr3dTarget Hungarian matching.
- **Model input**: 6-camera multi-view images, `data_shape=(3,512,1408)` (original `(3,900,1600)` resized to `(3,792,1408)` then cropped to `data_shape`), 6 views stacked as `(B,6,3,512,1408)`.
- **Model output**: Detection boxes `(B,900,21)` — 900 queries × (10 class scores + 11 regression: center xyz + size wlh + orientation sin/cos + velocity vxvy), `num_classes=10`, `reg_out_channels=10`.

### Official Repo and Paper

Official repo: https://github.com/megvii-research/PETR
Paper: https://arxiv.org/abs/2203.05625

Note: The backbone is EfficientNet-b3; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14091

---

## Usage Guide

### Environment Setup

Complete HEAL environment installation first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes v1.0-trainval packed lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training lmdb (base) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation lmdb (base, nuScenes val) |

#### Official Download

nuScenes dataset official page: <https://www.nuscenes.org/download> (registration required; v1.0-trainval). After extraction, includes `samples/` (annotated keyframes, 6 cameras), `sweeps/` (intermediate frames), `maps/` (map expansion), `v1.0-trainval/` (metadata json: annotation, scene, sample, ego_pose, etc.). Place raw data under `./tmp_data/orig_data/nuscenes/` (this directory is `--src-data-dir`):

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

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly (do not run legacy scripts such as create_data_*.py).

Packing order constraint: base first; occ must match version/split after base; lidar/map only require version/split match. This task only uses base lmdb (`{split}_lmdb`); lidar/map/occ packing is optional.

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

Purpose and when needed for the other 3 packers (optional for this task):
- **lidar** `NuscenesLidarPacker` (`tools/datasets/nuscenes_lidar_packer.py`): produces `{split}_lidar_lmdb`; only needed for lidar tasks; key params `--max-sweeps 10` `--load-dim` `--use_dim`.
- **map** `NuscenesMapGtPacker` (`tools/datasets/nuscenes_map_packer.py`): produces `{split}_map_lmdb`; only needed for map tasks; `--map-path` points to meta directory; PV seg mask is generated online at training time, not packed.
- **occ** `NuscenesOccPacker` (`tools/datasets/nuscenes_occ_packer.py`): produces `{split}_occ_lmdb`; only needed for occ tasks; `occ3d/gts/` from Occ3D-nuScenes official labels must be downloaded and placed by the user.

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: Pack output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, produces `{split}_lmdb`, matching config `data_rootdir`.
- `--pack-type lmdb`: Pack format is always lmdb.
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--version v1.0-trainval`: Data version (packer default is `v1.0-mini`; must explicitly override to `v1.0-trainval`).
- `--num-workers`: Parallel packing worker count (default 20).

`orig_data/nuscenes` is `--src-data-dir` (raw data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (matches config `data_rootdir`).

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

Use the visualization script to spot-check packing results:

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch Size per GPU | Float Training Epochs | Calibration Steps | Float Optimizer | LR Schedule | Loss Function |
|---|---|---|---|---|---|---|
| 8 | 2 | 48 | 5（`40 // len(device_ids)`) | AdamW (lr=2e-4, backbone lr=2e-4, weight_decay=0.05) | Float: CosineAnnealingLrUpdater (warmup_lr_ratio=1/3)<br>QAT: StepDecayLrUpdater | FocalLoss + L1Loss |

This task has no cross-task pretraining dependency; Float training starts from scratch. Calibration loads `float-checkpoint-best.pth.tar` from this task's Float training. QAT loads `calibration-checkpoint-best.pth.tar` from Calibration.

```bash
# Float training
python3 tools/train.py \
  --config configs/petr_efficientnetb3 \
  --stage float
# Output:./tmp_models/petr_efficientnetb3/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/petr_efficientnetb3 \
  --stage calibration
# Output:./tmp_models/petr_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/petr_efficientnetb3 \
  --stage qat
# Output:./tmp_models/petr_efficientnetb3/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/petr_efficientnetb3/deploy.py
# Output:./tmp_models/petr_efficientnetb3/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/petr_efficientnetb3/deploy.py
# Output:./tmp_models/petr_efficientnetb3/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat — only re-export and re-compile on the quantized model. Edit the `march` field in `configs/petr_efficientnetb3/common.py` to the target chip:
> ```python
> march = March.NASH_M   # J6M
> march = March.NASH_P   # J6P
> march = March.NASH_B   # J6B
> ```
> Then rerun `export_hbir.py` and `compile_perf.py` (the two commands above). Compiled artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage hbm
```

Evaluation metrics are NDS and mAP on nuScenes val. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPU with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the evaluation set (saved to `./demo/petr_efficientnetb3/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/petr_efficientnetb3/)
python3 tools/gen_demo.py \
  --config configs/petr_efficientnetb3

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/petr_efficientnetb3 \
  --stage float
```

Visualization results are saved to `./demo/petr_efficientnetb3/`.
