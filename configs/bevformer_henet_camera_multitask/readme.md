# BevFormer+HENet Multitask

`bevformer_henet_camera_multitask` is a camera-side multitask model based on BevFormer + HENet, jointly performing 3D object detection (det) and occupancy grid prediction (occ). This task provides a **float pretrained model** with float training and evaluation only; its outputs initialize the camera branch for float training in [`bevfusion_pointpillar_henet_multisensor_multitask`](../bevfusion_pointpillar_henet_multisensor_multitask/readme.md). Quantization and deployment are not supported for this task itself.

Model config directory: `configs/bevformer_henet_camera_multitask/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BevFusion | 6-camera multi-view image sequence `(B,6,3,512,960)` | HENet | FPN | det bounding boxes `(B,N,cls+reg)`; occ occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_occ_lmdb`). Metrics: NDS/mAP (det, `NuscenesMetric`) + Occ mIoU (`MeanIOU`, `ignore_index=17`).

| March | Metric | float |
| --- | --- | --- |
| J6M | NDS | 0.3573 |
|  | mAP | 0.3017 |
|  | Occ mIoU | 0.3146 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

---

## Model Overview

### Core Design

The BevFormer+HENet multitask model uses HENet as the camera backbone to extract multi-view features, then BevFormer ViewTransformer and temporal BEV Encoder to transform them into bird's-eye-view features. BEV features are fed to both a detection head (3D object detection) and an occupancy head (FlashOcc-style semantic occupancy prediction), enabling joint det+occ training.

- **Task type**: Multitask fusion (3D object detection + occupancy grid prediction).
- **backbone**: HENet (camera feature extraction), extracting multi-view features.
- **neck**: FPN.
- **Detection head**: BevFormer detection head, outputting 10-class 3D bounding boxes + velocity (`num_classes=10`).
- **Occupancy head**: Semantic occupancy prediction, 18 classes (`num_classes_occ=18`).
- **BEV range**: `bev_size=(51.2, 51.2, 0.8)`, `bev_size_occ=(40, 40)`.
- **Model input**: 6-camera image sequence `(B,6,3,512,960)` (`queue_length=1`, `MergeDataset(Bev+Occ)` merges base lmdb and occ lmdb).
- **Model output**: det 3D bounding boxes + occ occupancy grid semantics `(B,C,H,W)`.

### Official Repo and Paper

Official repo: https://github.com/fundamentalvision/BevFormer
Paper: https://arxiv.org/abs/2203.17270

Note: The camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes packed base lmdb + occ lmdb (`MergeDataset(Bev+Occ)`). Data paths read by the model (all joined from `data_rootdir` in `common.py`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training base lmdb (det data) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation base lmdb (det data) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | Training occ lmdb (occupancy labels) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | Evaluation occ lmdb (occupancy labels) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config consumption).

nuScenes requires no extra preprocessing; all four packers (base/lidar/map/occ) include nuscenes-devkit parsing and pack directly from the official extracted directory. This task uses **base lmdb + occ lmdb** (not lidar/map lmdb).

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (v1.0-trainval, registration required). Download the full trainval package, including:

- `samples/`: annotated keyframes (6 cameras + lidar)
- `sweeps/`: intermediate frames
- `maps/`: map expansion
- `v1.0-trainval/`: metadata (annotation, scene, sample, ego_pose json, etc.)

Extract the above into `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# Extract samples/ sweeps/ maps/ v1.0-trainval/ into this directory
```

This task includes occ labels; additionally download Occ3D-nuScenes labels (<https://github.com/Tsinghua-MARS-Lab/Occ3D>) and place `occ3d/gts/<scene>/<token>/labels.npz` under `./tmp_data/orig_data/nuscenes/occ3d/gts/`:

```bash
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# Place Occ3D-nuScenes labels.npz by scene/token into this directory
```

#### Directory Layout After Extraction

```
./tmp_data/orig_data/nuscenes/
├── samples/                 # Keyframes (6 cameras + LiDAR)
├── sweeps/                  # Intermediate frames
├── maps/                    # Map expansion
├── v1.0-trainval/           # Metadata json
└── occ3d/
    └── gts/                 # Occ3D-nuScenes occupancy labels (labels.npz)
```

#### Packing Commands

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and pack directly from the official extracted directory. Dependency order: base first; occ must follow base with matching version/split; lidar/map only require matching version/split. This task uses **base lmdb + occ lmdb**; run base and occ packers in sequence (occ after base).

**base packer** (`nuscenes_packer.py`, `NuscenesPacker`) → `{split}_lmdb`:

```bash
# Pack train base lmdb
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20 \
  --max-sweeps 10

# Pack val base lmdb
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20 \
  --max-sweeps 10
```

**occ packer** (`nuscenes_occ_packer.py`, `NuscenesOccPacker`) → `{split}_occ_lmdb` (must follow base with matching version/split):

```bash
# Pack train occ lmdb
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20

# Pack val occ lmdb
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, matching config `data_rootdir`.
- `--pack-type lmdb`: Pack format is lmdb.
- `--version v1.0-trainval`: nuScenes version (packer default is v1.0-mini; must override to v1.0-trainval).
- `--split-name`: `train` or `val` for training or validation set.
- `--max-sweeps 10` (base packer): Maximum sweep count.
- `--occ-gt-dir occ3d/gts` (occ packer): Occ3D-nuScenes label directory (default); requires user download.
- `--num-workers`: Parallel packing workers (default 20).

> **Four packers overview** (this task uses base + occ only):
> - **base** (`nuscenes_packer.py` / `NuscenesPacker`) → `{split}_lmdb`: camera + annotation base data, **used by this task**.
> - **lidar** (`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`) → `{split}_lidar_lmdb`: lidar point clouds; not used by this task.
> - **map** (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) → `{split}_map_lmdb`: vectorized map GT; not used by this task.
> - **occ** (`nuscenes_occ_packer.py` / `NuscenesOccPacker`) → `{split}_occ_lmdb`: Occ3D occupancy labels, **used by this task**.

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` layout (items used by this task are marked):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base train lmdb      ← used by this task
│   ├── val_lmdb/            # base val lmdb        ← used by this task
│   ├── train_lidar_lmdb/    # lidar train lmdb      (not used by this task)
│   ├── val_lidar_lmdb/      # lidar val lmdb        (not used by this task)
│   ├── train_map_lmdb/      # map train lmdb        (not used by this task)
│   ├── val_map_lmdb/        # map val lmdb          (not used by this task)
│   ├── train_occ_lmdb/      # occ train lmdb      ← used by this task
│   └── val_occ_lmdb/        # occ val lmdb        ← used by this task
├── meta/                    # Precomputed map expansion (not used by this task)
└── osm/                     # OpenStreetMap vector map (not used by this task)
```

Use visualization scripts to spot-check packing results:

```bash
# Validate base lmdb
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb \
  --viz-num 50

# Validate occ lmdb
python3 tools/datasets/nuscenes_occ_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb \
  --viz-num 50
```

### Training Workflow

Key training hyperparameters:

| # Training GPUs | Batch size per GPU | Float training epochs | Float optimizer | LR schedule |
|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 1 | 24 | AdamW (`lr=4e-4`, backbone `lr_mult=0.1`) | CosineAnnealingLrUpdater (warmup) |

#### Pretrained Weight Dependency

This task's float training loads [`henet_tinym`](../henet_tinym/readme.md) (ImageNet classification) float weights via `LoadCheckpoint` to initialize `camera_net.backbone`:

- **Weight source task**: `henet_tinym` (ImageNet classification, HENet-tiny/m backbone).
- **Weight path**: `./tmp_pretrained_models/henet_tinym/float-checkpoint-best.pth.tar` (produced by that task's float training).
- **Loading method**: `float_trainer` in `train.py` loads via `LoadCheckpoint`, using `update_state_dict_by_add_prefix(prefix="camera_net.")` to map ImageNet backbone weights into this task's `camera_net.` namespace; `allow_miss=True`, `ignore_extra=True` tolerate structural differences.

Users must **train `henet_tinym` first** (or obtain its float ckpt) and place it at the path above before running this task's float training.

```bash
# Float training
python3 tools/train.py \
  --config configs/bevformer_henet_camera_multitask \
  --stage float
# Output:./tmp_models/bevformer_henet_camera_multitask/float-checkpoint-last.pth.tar
```

#### Downstream Usage

This task's float output `./tmp_models/bevformer_henet_camera_multitask/float-checkpoint-last.pth.tar` initializes the camera branch for float training in [`bevfusion_pointpillar_henet_multisensor_multitask`](../bevfusion_pointpillar_henet_multisensor_multitask/readme.md):

- That task's `float_trainer` in `train.py` loads this task's float ckpt via `LoadCheckpoint` (no prefix, since camera branch structure matches); `allow_miss=True`, `ignore_extra=True` tolerate randomly initialized multisensor fusion modules (e.g. `BevFuseModule`).
- That task's `camera_ckpt_dir` in `common.py` points to the directory for this task's ckpt; users must complete this task's float training and place `float-checkpoint-last.pth.tar` in its `camera_ckpt_dir` before running bevfusion.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/bevformer_henet_camera_multitask \
  --stage float
```

Evaluation metrics are NDS/mAP (det) and Occ mIoU (occ, `MeanIOU`, `ignore_index=17`); evaluation set is nuScenes val (`val_lmdb` + `val_occ_lmdb`). Evaluation commands follow config `device_ids` by default; override evaluation GPUs with `--device-ids` if needed.
