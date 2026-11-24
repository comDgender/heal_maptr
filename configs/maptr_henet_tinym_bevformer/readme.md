# MapTR+HENet (BevFormer)

`maptr_henet_tinym_bevformer` is an online vectorized map construction model based on MapTR + HENet with BevFormer view transformation. This task has `use_lidar_gt=False`; map GT is generated online by `VectorizedLocalMap` (using base lmdb + meta directory + osm directory; no map lmdb packing required).

Model config directory: `configs/maptr_henet_tinym_bevformer/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| MapTR | 6-camera multi-view images `(B,6,3,480,800)` | HENet-tiny | FPN | vectorized map `(B,L,P,2)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`). Metric: chamfer mAP (`NuscenesMapMetric`, `metric="chamfer"`, monitor_metric_key=`"MAP"`).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | chamfer mAP (MAP) | 0.6626 | 0.6588 | — | 0.6315 |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with 8 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 9.26 | 111.22 | 88.00 |
| J6P | 6.20 | 662.25 | 82.90 |
| J6B | 25.69 | 39.70 | 79.70 |

Compiled artifact paths:
- HBIR: `./tmp_models/maptr_henet_tinym_bevformer/qat.bc`
- HBM: `./tmp_models/maptr_henet_tinym_bevformer/compile/model.hbm`

---

## Model Overview

### Core Design

MapTR uses HENet as the camera backbone to extract multi-view features, transforms them to BEV features via BevFormer's single-frame ViewTransformer and BEV Encoder, then feeds BEV features to the MapTR decoder with fixed-point polyline queries (`fixed_ptsnum_per_pred_line=20`) to predict vectorized map elements (divider/ped_crossing/boundary). This task has `use_lidar_gt=False`; map GT is not read from lmdb but generated online by `VectorizedLocalMap` (`map_path=meta_rootdir`, `sd_map_path=osm`).

- **Task type**: Online Vectorized Map Construction.
- **backbone**: HENet-tiny (ImageNet pretrained), extracts multi-view camera features.
- **neck**: FPN.
- **view transformation**: `SingleFrameBevFormerViewTransformer` + `SingleFrameBEVFormerEncoder` (`queue_length=1`/`test_queue_length=1`).
- **map elements**: `map_classes=[divider, ped_crossing, boundary]`, `fixed_ptsnum_per_gt_line=20`.
- **BEV range**: `use_lidar_gt=False` else branch, `point_cloud_range=[-30.0,-15.0,-10.0,30.0,15.0,10.0]`, `bev_h_=50`, `bev_w_=100` (bev 50×100).
- **Model input**: 6-camera single-frame images `(B,6,3,480,800)` (`NuscenesMapDataset`, `use_lidar_gt=False`, map GT generated online by `VectorizedLocalMap`, no map lmdb).
- **Model output**: 3 classes of vectorized map elements (divider/ped_crossing/boundary), 20 points per line.

### Official Repo and Paper

Official repo: https://github.com/hustvl/MapTR
Paper: https://arxiv.org/abs/2208.14437

Note: The camera backbone HENet is developed in HEAL; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14100

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see the top-level [`README.md`](../../README.md)). Run the following commands from the **HEAL repository root**:

### Data Preparation

This task has `use_lidar_gt=False` and uses base lmdb + meta directory + osm directory; no map lmdb packing is required (map GT is generated online by `VectorizedLocalMap`, `map_path=meta_rootdir`, `sd_map_path=osm`). Data paths read by the model (all joined from `data_rootdir` in `common.py`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | training base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | evaluation base lmdb |
| `./tmp_data/heal_data/nuscenes/meta` | map extension precomputed directory (required for online map GT generation) |
| `./tmp_data/heal_data/nuscenes/osm` | OpenStreetMap vector map (required for online map GT generation) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb, read by config).

nuScenes requires no extra preprocessing; all four packers (base/lidar/map/occ) include nuscenes-devkit parsing and read the official extracted directory directly. This task uses **base lmdb + meta directory + osm directory**; map lmdb packing is not required.

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (v1.0-trainval, registration required). Download the full trainval package, including:

- `samples/`: annotated keyframes (6 cameras + lidar)
- `sweeps/`: intermediate frames
- `maps/`: map expansion
- `v1.0-trainval/`: metadata (annotation, scene, sample, ego_pose json, etc.)

Extract the contents to the raw data directory `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# Extract samples/ sweeps/ maps/ v1.0-trainval/ into this directory
```

This task requires the meta directory (map extension precomputation) and osm directory (OpenStreetMap vector map) for `VectorizedLocalMap` online map GT generation. The meta/osm directories are produced during the map packer precomputation stage (see map packer notes under Packing Commands).

#### Directory Layout After Extraction

```
./tmp_data/orig_data/nuscenes/
├── samples/                 # annotated keyframes (6 cameras + lidar)
├── sweeps/                  # intermediate frames
├── maps/                    # map expansion
└── v1.0-trainval/           # metadata json
```

#### Packing Commands

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly. Dependency order: base first; occ must follow base with matching version/split; lidar/map only need matching version/split. This task uses base lmdb only; map lmdb packing is not required (map GT is generated online; only meta and osm directories are needed).

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

Parameter notes:
- `--src-data-dir`: raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, matches config `data_rootdir`.
- `--pack-type lmdb`: pack format is lmdb.
- `--version v1.0-trainval`: nuScenes version (packer default is v1.0-mini; must override to v1.0-trainval).
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--max-sweeps 10`: maximum sweep count.
- `--num-workers`: parallel packing worker count (default 20).

> **map packer notes** (this task does not use map lmdb; packing is optional):
>
> `nuscenes_map_packer.py` (`NuscenesMapGtPacker`) → `{split}_map_lmdb`, for offline pre-packing vectorized map GT. Parameters must match this task's `common.py`: `--map-path ./tmp_data/heal_data/nuscenes/meta` (meta directory), `--pc-range -30 -15 -10 30 15 10` (`use_lidar_gt=False` else branch), `--bev-size 50 100`, `--fixed-ptsnum-per-line 20`, `--map-classes divider ped_crossing boundary`. PV seg mask is not packed; generated online during training.
>
> **This task has `use_lidar_gt=False`; map GT is generated online by `VectorizedLocalMap` and does not use map lmdb**; but the meta directory (produced during map packer precomputation, `--map-path` points to it) and osm directory (OpenStreetMap vector map) are required.

> **Four packer overview** (this task uses base only):
> - **base** (`nuscenes_packer.py` / `NuscenesPacker`) → `{split}_lmdb`: camera + annotation base data, **used by this task**.
> - **lidar** (`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`) → `{split}_lidar_lmdb`: lidar point cloud, not used by this task.
> - **map** (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) → `{split}_map_lmdb`: vectorized map GT, not used by this task (map GT generated online).
> - **occ** (`nuscenes_occ_packer.py` / `NuscenesOccPacker`) → `{split}_occ_lmdb`: Occ3D occupancy labels, not used by this task.

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` has the following layout (items used by this task are marked):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base training lmdb      ← used by this task
│   ├── val_lmdb/            # base evaluation lmdb    ← used by this task
│   ├── train_lidar_lmdb/    # lidar training lmdb      (not used by this task)
│   ├── val_lidar_lmdb/      # lidar evaluation lmdb    (not used by this task)
│   ├── train_map_lmdb/      # map training lmdb        (not used by this task, but meta/ and osm/ required)
│   ├── val_map_lmdb/        # map evaluation lmdb      (not used by this task, but meta/ and osm/ required)
│   ├── train_occ_lmdb/      # occ training lmdb        (not used by this task)
│   └── val_occ_lmdb/        # occ evaluation lmdb        (not used by this task)
├── meta/                    # map extension precomputed directory  ← used by this task (online map GT generation)
└── osm/                     # OpenStreetMap vector map ← used by this task (online map GT generation)
```

Use visualization scripts to spot-check packing correctness:

```bash
# base lmdb validation
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb \
  --viz-num 50

# lidar lmdb validation (optional, only if lidar lmdb was packed)
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar

# map lmdb validation (optional, only if map lmdb was packed; auto-reads pc_range from __meta__)
python3 tools/datasets/nuscenes_map_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/map
```

Visualization PNGs are saved to each script's `--save-dir` (must be a locally writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as the save target). Manually verify alignment. This task generates map GT online and does not pack lidar/map lmdb by default, so lidar/map validation is optional.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Per-GPU Batch Size | Float Training Epochs | Calibration Steps | Float Optimizer |
|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 2 | 30 | (per config) | AdamW (`lr=2.5e-4`) |

This task has no QAT stage. Calibration loads Float training output `float-checkpoint-best.pth.tar` from this task.

```bash
# Float training
python3 tools/train.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage float
# Output: ./tmp_models/maptr_henet_tinym_bevformer/float-checkpoint-best.pth.tar

# Quantization calibration (Calibration)
python3 tools/train.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage calibration
# Output: ./tmp_models/maptr_henet_tinym_bevformer/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/maptr_henet_tinym_bevformer/deploy.py
# Output: ./tmp_models/maptr_henet_tinym_bevformer/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/maptr_henet_tinym_bevformer/deploy.py
# Output: ./tmp_models/maptr_henet_tinym_bevformer/compile/model.hbm
```

> **Switching chips (march) tips**: Switching chips only affects HBIR export and HBM compilation; it does **not** affect prior float training and quantization (calibration/qat). When switching J6M/J6P/J6B, you do not need to rerun training/calibration/qat—re-run export and compilation on the quantized model. Set the `march` field in `configs/maptr_henet_tinym_bevformer/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then re-run `export_hbir.py` and `compile_perf.py`. Compiled artifacts differ per chip; performance is not directly portable across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation first)
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation, plus board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage hbm
```

Evaluation metric is chamfer mAP (MAP) on nuScenes val (`val_lmdb`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPUs with `--device-ids`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference data from the eval set (saved to `./demo/maptr_henet_tinym_bevformer/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference samples (reads eval samples from gen_demo_cfg in infer.py, saved to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/maptr_henet_tinym_bevformer

# 2. Inference and visualization (float example; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage float
```

Visualization output saved to `./demo/maptr_henet_tinym_bevformer/`.
