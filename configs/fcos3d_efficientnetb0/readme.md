# FCOS3D (EfficientNet-b0)

`fcos3d_efficientnetb0` is an FCOS3D nuScenes monocular 3D object detection model with EfficientNet-b0 backbone and BiFPN neck, covering the full pipeline: float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/fcos3d_efficientnetb0/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| FCOS3D | Single front-view image `(B,3,512,896)` | EfficientNet-b0 | BiFPN | Front-view 3D detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`). Metric: mono mAP (monocular 3D detection mAP, `NuscenesMonoMetric`; higher is better).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.312 | 0.3044 | 0.3099 | 0.3098 |
|  | mono mAP | 0.2101 | 0.2042 | 0.2067 | 0.207 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 1.89 | 650.14 | 18.40 |
| J6P | 1.51 | 3294.91 | 19.30 |
| J6B | 4.91 | 244.24 | 14.90 |

Compilation artifact paths:
- HBIR: `./tmp_models/fcos3d_efficientnetb0/qat.bc`
- HBM: `./tmp_models/fcos3d_efficientnetb0/compile/model.hbm`

---

## Model Overview

### Core Design

FCOS3D extends FCOS anchor-free fully-convolutional detection to 3D: EfficientNet-b0 + BiFPN extract multi-scale features; FCOS3DHead predicts 3D detection boxes (center offset, depth, size, orientation, class) at each feature point; FCOS3DTarget performs 3D target assignment; FCOS3DPostProcess decodes final 3D detection boxes.

- **Task type**: Monocular 3D object detection (Monocular 3D Object Detection).
- **backbone**: EfficientNet-b0 (`efficientnet`, `model_type=b0`, `include_top=False`, `activation=relu`, `use_se_block=False`).
- **neck**: BiFPN (`BiFPN`, bidirectional feature pyramid, `stack=3`, `out_channels=64`, `num_outs=5`).
- **Detection head**: `FCOS3DHead` (fully-convolutional anchor-free 3D detection head).
- **Loss**: `FCOS3DLoss` (FocalLoss + SmoothL1Loss + CrossEntropyLoss).
- **Model input**: Single front-view image (only CAM_FRONT from 6 cameras), `(B,3,512,896)` (Resize3D `img_scale=(896,512)` + Pad `(512,896)`, `NuscenesMonoDataset` monocular loading).
- **Model output**: Front-view 3D detection boxes (class + center + size + orientation), per-feature-point prediction `group_reg_dims=(2,1,3,1,2)`, `num_classes=10`, decoded via `FCOS3DPostProcess` + NMS (`max_per_img=100`).

### Official Repo and Paper

Official repo: https://github.com/open-mmlab/mmdetection3d
Paper: https://arxiv.org/abs/2104.10956

Note: backbone is EfficientNet-b0; official implementation uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14096

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

> Note: This task is monocular 3D detection; `NuscenesMonoDataset` uses only a single front-view image (CAM_FRONT). `data_rootdir` is defined in `data.py` without trailing slash (`./tmp_data/heal_data/nuscenes/v1.0-trainval`); path concatenation uses `data_rootdir + "/train_lmdb"`.

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
| 4 | 8 | 48 | 100 | AdamW (lr=2e-4, weight_decay=0.01) | CyclicLrUpdater<br>Fixed lr in QAT stage | FCOS3DLoss |

No cross-task pretrained dependency; Float training starts from scratch. Calibration loads `float-checkpoint-best.pth.tar` from this task's Float training. QAT loads `calibration-checkpoint-best.pth.tar` from Calibration.

```bash
# Float training
python3 tools/train.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage float
# Output:./tmp_models/fcos3d_efficientnetb0/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage calibration
# Output:./tmp_models/fcos3d_efficientnetb0/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage qat
# Output:./tmp_models/fcos3d_efficientnetb0/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/fcos3d_efficientnetb0/deploy.py
# Output:./tmp_models/fcos3d_efficientnetb0/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/fcos3d_efficientnetb0/deploy.py
# Output:./tmp_models/fcos3d_efficientnetb0/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to re-run training/calibration/qat — only re-run export and compilation on the quantized model. Modify the `march` field in `configs/fcos3d_efficientnetb0/common.py` to the target chip:
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
  --config configs/fcos3d_efficientnetb0 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage hbm
```

Evaluation metric is mono mAP; evaluation set is nuScenes val. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPU via `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the evaluation set (saved to `./demo/fcos3d_efficientnetb0/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/fcos3d_efficientnetb0/)
python3 tools/gen_demo.py \
  --config configs/fcos3d_efficientnetb0

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/fcos3d_efficientnetb0 \
  --stage float
```

Visualization results are saved to `./demo/fcos3d_efficientnetb0/`.
