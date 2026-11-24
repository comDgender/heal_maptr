# Detr3D (EfficientNet-b3)

`detr3d_efficientnetb3` is a Detr3D nuScenes BEV 3D object detection model with EfficientNet-b3 backbone and BiFPN neck, covering the full pipeline: float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/detr3d_efficientnetb3/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| Detr3D | 6-camera multi-view images `(B,6,3,512,1408)` | EfficientNet-b3 | BiFPN | 3D detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`). Metrics: NDS / mAP (nuScenes detection composite metrics; higher is better).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3357 | 0.3299 | 0.338 | 0.337 |
|  | mAP | 0.2694 | 0.2618 | 0.2688 | 0.2683 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 21.79 | 46.43 | 98.10 |
| J6P | 15.24 | 253.11 | 94.80 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifact paths:
- HBIR: `./tmp_models/detr3d_efficientnetb3/qat.bc`
- HBM: `./tmp_models/detr3d_efficientnetb3/compile/model.hbm`

---

## Model Overview

### Core Design

Detr3D brings the DETR paradigm to 3D detection: EfficientNet-b3 + BiFPN extract multi-scale image features; view transformation projects 2D features to 3D space; Detr3dTransformer iteratively samples multi-view features with learnable queries and predicts 3D detection boxes; Detr3dTarget performs Hungarian matching during training.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False`, `activation=relu`, `use_se_block=False`).
- **neck**: BiFPN (`BiFPN`, bidirectional feature pyramid, `stack=3`, `out_channels=256`, `num_outs=5`).
- **Detection head**: `Detr3dHead` + `Detr3dTransformer` + `Detr3dDecoder` (DETR-style 3D decoder).
- **Loss**: FocalLoss (cls) + L1Loss (bbox), via Detr3dTarget Hungarian matching.
- **Model input**: 6-camera multi-view images, `(B,6,3,512,1408)` (original `orig_shape=(3,900,1600)` → resize `(3,792,1408)` → crop `data_shape=(3,512,1408)`, `num_views=6`, `NuscenesBevDataset`).
- **Model output**: 3D detection boxes (class + center + size + orientation), `num_query=900`, `num_classes=10`, decoded via `Detr3dPostProcess` (`max_num=300`).

### Official Repo and Paper

Official repo: https://github.com/WangYueFt/detr3d
Paper: https://arxiv.org/abs/2110.06922

Note: backbone is EfficientNet-b3; official repo uses a different backbone.

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes v1.0-trainval packed as lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training lmdb (base) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation lmdb (base, nuScenes val) |

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (registration required; v1.0-trainval). After extraction includes `samples/` (annotated keyframes, 6 cameras), `sweeps/` (intermediate frames), `maps/` (map expansion), `v1.0-trainval/` (metadata json: annotation, scene, sample, ego_pose, etc.). Place raw data in `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir`):

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

nuScenes requires no additional preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly (do not run legacy scripts like create_data_*.py).

Packing order constraint: base first; occ after base must match version/split; lidar/map only require version/split match. This task only uses base lmdb (`{split}_lmdb`); lidar/map/occ packing is optional.

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
- **lidar** `NuscenesLidarPacker` (`tools/datasets/nuscenes_lidar_packer.py`): produces `{split}_lidar_lmdb`; only lidar tasks need packing; key params `--max-sweeps 10` `--load-dim` `--use_dim`.
- **map** `NuscenesMapGtPacker` (`tools/datasets/nuscenes_map_packer.py`): produces `{split}_map_lmdb`; only map tasks need packing; `--map-path` points to meta directory; PV seg mask is generated online during training, not packed.
- **occ** `NuscenesOccPacker` (`tools/datasets/nuscenes_occ_packer.py`): produces `{split}_occ_lmdb`; only occ tasks need packing; `occ3d/gts/` from Occ3D-nuScenes official labels must be downloaded and placed separately.

Parameter description:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (with `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: Packed output `./tmp_data/heal_data/nuscenes/v1.0-trainval`, produces `{split}_lmdb`, matching config `data_rootdir`.
- `--pack-type lmdb`: Packing format is lmdb.
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--version v1.0-trainval`: Data version (packer default is `v1.0-mini`; must explicitly override to `v1.0-trainval`).
- `--num-workers`: Parallel packing worker count (default 20).

`orig_data/nuscenes` is `--src-data-dir` (raw data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (matches config `data_rootdir`).

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/v1.0-trainval/` has the following structure:

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

Use the visualization script to sample and verify packing results:

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | Float optimizer | LR schedule | Loss |
|---|---|---|---|---|---|---|
| 8 | 1 | 36 | 10 | AdamW (lr=3e-4, backbone lr=3e-5, weight_decay=0.01) | CosineAnnealingLrUpdater (warmup_lr_ratio=1/3)<br>StepDecayLrUpdater in QAT stage | FocalLoss + L1Loss |

No cross-task pretrained dependency; Float training starts from scratch. Calibration loads `float-checkpoint-best.pth.tar` from this task's Float training. QAT loads `calibration-checkpoint-best.pth.tar` from Calibration.

```bash
# Float training
python3 tools/train.py \
  --config configs/detr3d_efficientnetb3 \
  --stage float
# Output:./tmp_models/detr3d_efficientnetb3/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/detr3d_efficientnetb3 \
  --stage calibration
# Output:./tmp_models/detr3d_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/detr3d_efficientnetb3 \
  --stage qat
# Output:./tmp_models/detr3d_efficientnetb3/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/detr3d_efficientnetb3/deploy.py
# Output:./tmp_models/detr3d_efficientnetb3/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/detr3d_efficientnetb3/deploy.py
# Output:./tmp_models/detr3d_efficientnetb3/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to re-run training/calibration/qat — only re-run export and compilation on the quantized model. Modify the `march` field in `configs/detr3d_efficientnetb3/common.py` to the target chip:
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
  --config configs/detr3d_efficientnetb3 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/detr3d_efficientnetb3 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/detr3d_efficientnetb3 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/detr3d_efficientnetb3 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/detr3d_efficientnetb3 \
  --stage hbm
```

Evaluation metrics are NDS and mAP; evaluation set is nuScenes val. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPU via `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the evaluation set (saved to `./demo/detr3d_efficientnetb3/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/detr3d_efficientnetb3/)
python3 tools/gen_demo.py \
  --config configs/detr3d_efficientnetb3

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/detr3d_efficientnetb3 \
  --stage float
```

Visualization results are saved to `./demo/detr3d_efficientnetb3/`.
