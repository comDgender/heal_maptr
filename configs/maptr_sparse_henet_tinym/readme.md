# SparseMapTR+HENet

`maptr_sparse_henet_tinym` is a sparse vectorized map construction model based on SparseMapTR (SparseMapHead) + HENet. This task has `use_lidar_gt=True` and reads map lmdb (using base+lidar+map lmdb + meta+osm).

Model config directory: `configs/maptr_sparse_henet_tinym/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseMapTR | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny | FPN | vectorized map `(B,L,P,2)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb` + `val_map_lmdb`). Metric: chamfer mAP (`NuscenesMapMetric`, `metric="chamfer"`, monitor_metric_key=`"MAP"`).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | chamfer mAP (MAP) | 0.5924 | 0.5882 | — | 0.5892 |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS for J6M/J6P is measured with 8 threads on a single core; J6B uses 4 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 11.42 | 89.32 | 67.80 |
| J6P | 9.40 | 257.29 | 98.10 |
| J6B | 131.69 | 23.36 | 120.20 |

Compiled artifact paths:
- HBIR: `./tmp_models/maptr_sparse_henet_tinym/qat.bc`
- HBM: `./tmp_models/maptr_sparse_henet_tinym/compile/model.hbm`

---

## Model Overview

### Core Design

SparseMapTR uses HENet as the camera backbone to extract multi-view features, converts them to BEV features, then feeds them to SparseMapHead (6-layer sparse query stack) for vectorized map element prediction. Unlike MapTR, SparseMapTR uses a sparse query mechanism (`InstanceBankOE` + `SparsePoint3DEncoder`), refining only a small set of candidate queries iteratively to reduce compute. This task has `use_lidar_gt=True`, reads map lmdb as map GT, and uses lidar lmdb for depth supervision.

- **Task type**: Sparse Vectorized Map Construction.
- **backbone**: HENet-tiny (ImageNet pretrained), extracts multi-view camera features.
- **neck**: FPN (`out_strides=[4,8,16,32]`, outputs 256-dim multi-scale features).
- **decoder**: `SparseMapHead` (6-layer sparse query stack, `InstanceBankOE` + `SparsePoint3DEncoder`).
- **map elements**: `map_classes=[divider, ped_crossing, boundary]`, `fixed_ptsnum_per_gt_line=20`.
- **BEV range**: `use_lidar_gt=True` branch, `point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]`, `bev_h_=100`, `bev_w_=50` (bev 100×50).
- **data**: `MergeDataset(SparseMap(base+map_gt) + Lidar)`, uses base+lidar+map lmdb + meta+osm.
- **Model input**: 6-camera images `(B,6,3,256,704)` + lidar point cloud `(B,N,D)` (`use_lidar_gt=True`, reads map lmdb as map GT and lidar lmdb for depth supervision).
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

This task has `use_lidar_gt=True` and reads map lmdb, using base+lidar+map lmdb + meta+osm. Data paths read by the model (all joined from `data_rootdir` in `common.py`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | training base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | evaluation base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | training lidar lmdb (depth supervision) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | evaluation lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_map_lmdb` | training map lmdb (map GT) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb` | evaluation map lmdb (map GT) |
| `./tmp_data/heal_data/nuscenes/meta` | map extension precomputed directory |
| `./tmp_data/heal_data/nuscenes/osm` | OpenStreetMap vector map |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb, read by config).

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly. This task uses **base + lidar + map lmdb** (+ meta/osm directories).

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

This task requires the meta directory (map extension precomputation) and osm directory (OpenStreetMap vector map) for map packer and online map generation.

#### Directory Layout After Extraction

```
./tmp_data/orig_data/nuscenes/
├── samples/                 # annotated keyframes (6 cameras + lidar)
├── sweeps/                  # intermediate frames
├── maps/                    # map expansion
└── v1.0-trainval/           # metadata json
```

#### Packing Commands

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly. Dependency order: base first; occ must follow base with matching version/split; lidar/map only need matching version/split. This task uses **base + lidar + map lmdb**; run base, lidar, and map packers in sequence.

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

**lidar packer** (`nuscenes_lidar_packer.py`, `NuscenesLidarPacker`) → `{split}_lidar_lmdb`:

```bash
# Pack train lidar lmdb
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --max-sweeps 10 \
  --load-dim 5 \
  --use-dim 0 1 2 4

# Pack val lidar lmdb
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --max-sweeps 10 \
  --load-dim 5 \
  --use-dim 0 1 2 4
```

**map packer** (`nuscenes_map_packer.py`, `NuscenesMapGtPacker`) → `{split}_map_lmdb` (`--pc-range` must match the `use_lidar_gt=True` branch):

```bash
# Pack train map lmdb
python3 tools/datasets/nuscenes_map_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --map-path ./tmp_data/heal_data/nuscenes/meta \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --pc-range -15 -30 -10 15 30 10 \
  --bev-size 100 50 \
  --fixed-ptsnum-per-line 20 \
  --map-classes divider ped_crossing boundary \
  --padding-value -10000

# Pack val map lmdb
python3 tools/datasets/nuscenes_map_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --map-path ./tmp_data/heal_data/nuscenes/meta \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --pc-range -15 -30 -10 15 30 10 \
  --bev-size 100 50 \
  --fixed-ptsnum-per-line 20 \
  --map-classes divider ped_crossing boundary \
  --padding-value -10000
```

Parameter notes:
- `--src-data-dir`: raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, matches config `data_rootdir`.
- `--pack-type lmdb`: pack format is lmdb.
- `--version v1.0-trainval`: nuScenes version (packer default is v1.0-mini; must override to v1.0-trainval).
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--max-sweeps 10` (base/lidar packer): maximum sweep count.
- `--load-dim 5` / `--use-dim 0 1 2 4` (lidar packer): point cloud load/use dimensions, matching `common.py` `lidar_load_dim=5`, `lidar_use_dim=[0,1,2,4]`.
- `--map-path ./tmp_data/heal_data/nuscenes/meta` (map packer): map extension precomputed meta directory.
- `--pc-range -15 -30 -10 15 30 10` (map packer): matches `use_lidar_gt=True` branch `point_cloud_range`.
- `--bev-size 100 50` (map packer): matches `bev_h_=100`, `bev_w_=50`.
- `--fixed-ptsnum-per-line 20` (map packer): fixed points per line, matches `fixed_ptsnum_per_gt_line=20`.
- `--map-classes divider ped_crossing boundary` (map packer): map element classes, matches `map_classes`.
- PV seg mask is not packed; generated online during training.

> **Four packer overview** (this task uses base + lidar + map):
> - **base** (`nuscenes_packer.py` / `NuscenesPacker`) → `{split}_lmdb`: camera + annotation base data, **used by this task**.
> - **lidar** (`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`) → `{split}_lidar_lmdb`: lidar point cloud, **used by this task**.
> - **map** (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) → `{split}_map_lmdb`: vectorized map GT, **used by this task**.
> - **occ** (`nuscenes_occ_packer.py` / `NuscenesOccPacker`) → `{split}_occ_lmdb`: Occ3D occupancy labels, not used by this task.

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` has the following layout (items used by this task are marked):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base training lmdb      ← used by this task
│   ├── val_lmdb/            # base evaluation lmdb    ← used by this task
│   ├── train_lidar_lmdb/    # lidar training lmdb    ← used by this task
│   ├── val_lidar_lmdb/      # lidar evaluation lmdb    ← used by this task
│   ├── train_map_lmdb/      # map training lmdb      ← used by this task
│   ├── val_map_lmdb/        # map evaluation lmdb      ← used by this task
│   ├── train_occ_lmdb/      # occ training lmdb        (not used by this task)
│   └── val_occ_lmdb/        # occ evaluation lmdb        (not used by this task)
├── meta/                    # map extension precomputed directory  ← used by this task
└── osm/                     # OpenStreetMap vector map ← used by this task
```

Use visualization scripts to spot-check packing correctness:

```bash
# base lmdb validation
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb \
  --viz-num 50

# lidar lmdb validation (reuses heal/visualize/lidar_det.py, BEV point cloud + 3D GT boxes, no model required)
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar

# map lmdb validation (reuses heal/visualize/nuscenes_map.py, BEV vectorized map GT, no model required; auto-reads pc_range from __meta__)
python3 tools/datasets/nuscenes_map_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/map
```

Visualization PNGs are saved to each script's `--save-dir` (must be a locally writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as the save target). Manually verify point cloud, vectorized map, and packing alignment.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Per-GPU Batch Size | Float Training Epochs | Calibration Steps | Float Optimizer |
|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 4 | 100 | (per config) | AdamW (backbone `lr=1e-3`, base `lr=3e-4`) |

This task has no QAT stage (`train.py` has no qat_trainer). Calibration loads Float training output `float-checkpoint-best.pth.tar` from this task.

```bash
# Float training
python3 tools/train.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage float
# Output: ./tmp_models/maptr_sparse_henet_tinym/float-checkpoint-best.pth.tar

# Quantization calibration (Calibration)
python3 tools/train.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage calibration
# Output: ./tmp_models/maptr_sparse_henet_tinym/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/maptr_sparse_henet_tinym/deploy.py
# Output: ./tmp_models/maptr_sparse_henet_tinym/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/maptr_sparse_henet_tinym/deploy.py
# Output: ./tmp_models/maptr_sparse_henet_tinym/compile/model.hbm
```

> **Switching chips (march) tips**: Switching chips only affects HBIR export and HBM compilation; it does **not** affect prior float training and quantization (calibration/qat). When switching J6M/J6P/J6B, you do not need to rerun training/calibration/qat—re-run export and compilation on the quantized model. Set the `march` field in `configs/maptr_sparse_henet_tinym/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then re-run `export_hbir.py` and `compile_perf.py`. Compiled artifacts differ per chip; performance is not directly portable across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation first)
python3 tools/predict.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation, plus board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage hbm
```

Evaluation metric is chamfer mAP (MAP) on nuScenes val (`val_lmdb` + `val_lidar_lmdb` + `val_map_lmdb`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPUs with `--device-ids`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference data from the eval set (saved to `./demo/maptr_sparse_henet_tinym/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference samples (reads eval samples from gen_demo_cfg in infer.py, saved to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/maptr_sparse_henet_tinym

# 2. Inference and visualization (float example; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/maptr_sparse_henet_tinym \
  --stage float
```

Visualization output saved to `./demo/maptr_sparse_henet_tinym/`.
