# FlashOcc + HENet + LSS

`flashocc_henet_lss_occ3d` is a BEV occupancy prediction model based on HENet + LSS (Lift-Splat-Shoot), using MergeDataset to fuse camera base lmdb and occ lmdb (Occ3D-nuScenes labels), covering the full pipeline: float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/flashocc_henet_lss_occ3d/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| FlashOcc | 6-camera multi-view images `(B,6,3,512,960)` | HENet | FPN + LSSTransformer + BevEncoder | Occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_occ_lmdb`). Metric: Occ mIoU (MeanIOU, `ignore_index=17`, `num_classes=18`).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Occ mIoU (MeanIOU) | 0.3664 | 0.3688 | — | 0.369 |

> Results are based on `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 7.61 | 136.01 | 73.40 |
| J6P | 5.64 | 730.55 | 77.60 |
| J6B | 23.63 | 43.17 | 81.60 |

Compilation artifact paths:
- HBIR: `./tmp_models/flashocc_henet_lss_occ3d/qat.bc`
- HBM: `./tmp_models/flashocc_henet_lss_occ3d/compile/model.hbm`

---

## Model Overview

### Core Design

FlashOcc uses LSS (Lift-Splat-Shoot) view transformation: HENet extracts multi-view camera features; `LSSTransformer` predicts depth distribution and lifts 2D features to 3D voxel space (`depth=45`, `num_points=10`, `bev_size=(40,40,0.625)`); fused via `BevEncoder` (`BiFPN`); `FlashOccDetDecoder`/`BEVOCCHead2D` outputs 18-class 3D occupancy predictions.

- **Task type**: BEV 3D occupancy prediction (BEV Occupancy Prediction).
- **backbone**: HENet (`type=HENet`, `depth=45`, `num_points=10`), extracts multi-view camera features.
- **neck**: `FPN` + `LSSTransformer` (Lift-Splat-Shoot view transformation, `bev_size=(40,40,0.625)`, `grid_size=(128,128)`) + `BevEncoder` (`BiFPN`).
- **Occupancy head**: `FlashOccDetDecoder` (`BEVOCCHead2D`, `num_classes=18`, `ignore_index=17`).
- **Loss**: `CrossEntropyLoss` (occ seg).
- **Model input**: 6-view camera images `(B,6,3,512,960)` (`data_shape=(3,512,960)`, `MergeDataset(Bev+Occ)` fuses base lmdb and occ lmdb).
- **Model output**: 18-class 3D occupancy grid `(B,C,H,W)` (Occ3D, `occ3d_seg_class` 18 classes, includes `others`/`ignore_index=17`).

### Official Repo and Paper

Official repo: https://github.com/Yzichen/FlashOCC
Paper: https://arxiv.org/abs/2311.12058

Note: camera backbone HENet is HEAL-developed; official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14099

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes packed base lmdb and occ lmdb (fused via `MergeDataset`, aligned by token). Data paths read by the model (all concatenated from `common.py` `data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training base lmdb (`NuscenesBevDataset`) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation base lmdb (`NuscenesBevDataset`) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | Training occ lmdb (`NuscenesOccDataset`, Occ3D labels) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | Evaluation occ lmdb (`NuscenesOccDataset`, Occ3D labels) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config).

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (registration required). Download the full **v1.0-trainval** dataset; after extraction includes `samples/` (annotated keyframes, 6 cameras), `sweeps/` (intermediate frames), `maps/` (map expansion), `v1.0-trainval/` (metadata json: annotation, scene, sample, ego_pose, etc.).

**This task additionally requires Occ3D-nuScenes occupancy labels**: Occ3D-nuScenes official repo <https://github.com/Tsinghua-MARS-Lab/Occ3D>, download `occ3d/gts/<scene>/<token>/labels.npz` occupancy labels and place under `occ3d/gts/` in the nuScenes root (i.e. `./tmp_data/orig_data/nuscenes/occ3d/gts/`); occ packer reads via `--occ-gt-dir occ3d/gts`.

Extract the above to the raw data directory `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# Place official extracted content in this directory
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
# Also place Occ3D occupancy labels
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# Place Occ3D-nuScenes occ3d/gts/<scene>/<token>/labels.npz into this directory
# ./tmp_data/orig_data/nuscenes/occ3d/gts/<scene>/<token>/labels.npz
```

#### Directory Layout After Extraction

```
./tmp_data/orig_data/nuscenes/
├── samples/              # Keyframe images (6 camera subdirs)
├── sweeps/               # Intermediate frames (6 camera subdirs)
├── maps/                 # Map expansion (expansion + png)
├── v1.0-trainval/        # Metadata json (annotation, scene, sample, ego_pose, etc.)
└── occ3d/
    └── gts/              # Occ3D-nuScenes occupancy labels
        └── <scene>/
            └── <token>/
                └── labels.npz   # Occupancy semantic labels (18 classes, ignore_index=17)
```

#### Packing Commands

nuScenes requires no additional preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly (do not run create_data_*.py).

nuScenes has four packers with dependency order: **base first**; occ must run after base with matching version/split; lidar/map only require version/split match. This task uses **base lmdb + occ lmdb** — run base packer first, then occ packer (occ must run after base with matching version/split). Unused lidar packer (`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`) packs lidar lmdb for lidar point cloud tasks; map packer (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) packs map lmdb for online map vectorization tasks — neither is used in this task.

`NuscenesPacker` (base) packs base camera lmdb (`{split}_lmdb`):

```bash
# Pack base train split (must run before occ)
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# Pack base val split (must run before occ)
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

`NuscenesOccPacker` (occ) packs occupancy prediction lmdb (`{split}_occ_lmdb`), **must run after base with matching version/split**, reads Occ3D-nuScenes occupancy labels via `--occ-gt-dir occ3d/gts`:

```bash
# Pack occ train split (after base train; same version/split)
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20

# Pack occ val split (after base val; same version/split)
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20
```

Parameter description:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (with `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`, `occ3d/gts/`).
- `--target-data-dir`: Packed output `./tmp_data/heal_data/nuscenes/v1.0-trainval`, matches config `data_rootdir`.
- `--pack-type lmdb`: Packing format is lmdb.
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--version v1.0-trainval`: nuScenes dataset version (packer default is `v1.0-mini`; must explicitly override to `v1.0-trainval`); occ must match base.
- `--occ-gt-dir occ3d/gts`: Occ3D-nuScenes occupancy label directory read by occ packer (relative to `--src-data-dir`, i.e. `./tmp_data/orig_data/nuscenes/occ3d/gts/`), from <https://github.com/Tsinghua-MARS-Lab/Occ3D>.
- `--num-workers`: Parallel packing worker count (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` has the following structure (this task uses base + occ lmdb under `v1.0-trainval/`):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # train base lmdb          ← used by this task
│   ├── val_lmdb/            # val base lmdb            ← used by this task
│   ├── train_lidar_lmdb/    # Training lidar lmdb(not used by this task)
│   ├── val_lidar_lmdb/      # Evaluation lidar lmdb(not used by this task)
│   ├── train_map_lmdb/      # train map lmdb (not used by this task)
│   ├── val_map_lmdb/        # val map lmdb (not used by this task)
│   ├── train_occ_lmdb/      # train occ lmdb           ← used by this task
│   └── val_occ_lmdb/        # val occ lmdb             ← used by this task
├── meta/                    # Precomputed map expansion meta directory
└── osm/                     # OSM vector map (for map tasks)
```

Use visualization scripts to sample and verify packing results. Base lmdb uses `nuscenes_viz.py`; occ lmdb uses `nuscenes_occ_viz.py`:

```bash
# Visualize base lmdb
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50

# Visualize occ lmdb
python3 tools/datasets/nuscenes_occ_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | Float optimizer | LR schedule | Loss |
|---|---|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 4 (eval `val_batch_size_per_gpu=1`) | 150 | 100 | AdamW (`lr=1e-4`, `weight_decay=1e-2`) | `CosLrUpdater` (`warmup_lr_ratio=1/3`, `stop_lr=2e-4*1e-3`) | `CrossEntropyLoss` (occ seg, `ignore_index=17`) |

Float training for this task starts from scratch; no cross-task pretrained dependency. Calibration loads `float-checkpoint-best.pth.tar` from this task's Float training. This task has no QAT stage.

```bash
# Float training
python3 tools/train.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage float
# Output:./tmp_models/flashocc_henet_lss_occ3d/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage calibration
# Output:./tmp_models/flashocc_henet_lss_occ3d/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/flashocc_henet_lss_occ3d/deploy.py
# Output:./tmp_models/flashocc_henet_lss_occ3d/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/flashocc_henet_lss_occ3d/deploy.py
# Output:./tmp_models/flashocc_henet_lss_occ3d/compile/model.hbm
```

> Export uses `enable_vpu=False`; compilation input `input_source=["pyramid","ddr","ddr"]` (camera images via pyramid, occ auxiliary via ddr).

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration)**. When switching J6M/J6P/J6B, no need to re-run training/calibration — only re-run export and compilation on the quantized model. Modify the `march` field in `configs/flashocc_henet_lss_occ3d/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then re-run `export_hbir.py` and `compile_perf.py`. Compilation artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage hbm
```

Evaluation metric is Occ mIoU (`MeanIOU`, `ignore_index=17`, `num_classes=18`); evaluation set is nuScenes val (`val_lmdb` + `val_occ_lmdb`, `val_metrics`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPU via `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference samples from the evaluation set (saved to `./demo/flashocc_henet_lss_occ3d/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/flashocc_henet_lss_occ3d/)
python3 tools/gen_demo.py \
  --config configs/flashocc_henet_lss_occ3d

# 2. Inference and visualization (float shown; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage float
```

Visualization results are saved to `./demo/flashocc_henet_lss_occ3d/`.
