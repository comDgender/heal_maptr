# SparseMultitaskOE+MapTR+FlashOcc+HENet Multitask

`sparse_multitask_det_maptr_flashocc_henet_tinym` is a multitask model based on SparseMultitaskOE (SparseBEVOEHead), jointly performing 3D object detection (det) + vectorized map construction (map) + occupancy grid prediction (occ). This task uses three-stage Float training (stage1 backbone → stage2 add occ/map heads → merge), `use_lidar_gt=True`, using base+lidar+map+occ lmdb + meta+osm.

Model config directory: `configs/sparse_multitask_det_maptr_flashocc_henet_tinym/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseMultitaskOE | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny | MMFPN | det detection boxes `(B,N,cls+reg)`; map vectorized map `(B,L,P,2)`; occ occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb` + `val_map_lmdb` + `val_occ_lmdb`). Metrics: NDS/mAP (det, `NuscenesMetric`) + chamfer mAP (map, `NuscenesMapMetric`, `metric="chamfer"`) + Occ mIoU (`MeanIOU`, `ignore_index=17`).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5434 | 0.5325 | — | 0.5281 |
|  | mAP | — | — | — | — |
|  | chamfer mAP (MAP) | 0.592 | 0.5838 | — | 0.5833 |
|  | Occ mIoU | 0.3197 | 0.3277 | — | 0.3273 |

> Data measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS for J6M/J6P is single-core eight-thread; J6B is single-core dual-thread; Latency is single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 27.50 | 36.71 | 181.90 |
| J6P | 17.76 | 188.07 | 210.80 |
| J6B | 164.46 | 9.53 | 301.60 |

Compiled artifact paths:
- HBIR: `./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/qat.bc`
- HBM: `./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/compile/model.hbm`

---

## Model Overview

### Core Design

SparseMultitaskOE uses HENet as backbone to extract multi-view features, converts to BEV features, then SparseBEVOEHead drives three task heads: detection head (SparseBEVOE detection, 10-class 3D boxes), map head (MapTR-style vectorized map, 3 element classes), occupancy head (FlashOcc-style semantic occupancy, 18 classes). This task uses three-stage Float training: stage1 jointly trains det+map+occ (backbone), stage2 freezes backbone and trains map+occ heads only (`lr=1e-5`), merge stage combines stage1 (det) and stage2 (map/occ) weights into a complete model. `use_lidar_gt=True`, `defer_vectormap=True` (map generated online + map lmdb used together).

- **Task type**: Multitask fusion (3D object detection + vectorized map construction + occupancy grid prediction).
- **backbone**: HENet-tiny (ImageNet pretrained).
- **neck**: MMFPN.
- **Detection head**: `SparseBEVOEHead` (`SparseBEVOEEncoder` + `SparseBEVOERefinementModule`, 10-class 3D detection boxes, `num_classes=10`).
- **Map head**: `SparseMapPerceptionDecoder` (`SparseMapHead`, 3-class vectorized map elements, `map_classes=[divider,ped_crossing,boundary]`).
- **Occupancy head**: Semantic occupancy prediction, 18 classes (`num_classes_occ=18`).
- **BEV range**: `point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]` (det/occ), `map_point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]` (map), `occ_bev_size=(40,40,0.625)`.
- **Data**: `MergeDataset(SparseMap(base+map) + Lidar + Occ)`, using base+lidar+map+occ lmdb + meta+osm.
- **Model input**: 6-camera multi-view images (B,6,3,256,704) + lidar point cloud (B,N,D).
- **Model output**: det 3D detection boxes + map vectorized map elements + occ occupancy grid semantics.

**Deployment notes**: HBIR export enables `enable_vpu=True`; compilation uses `input_source=compile_cfg["input_source"]`.

### Official Repo and Paper

Official repo: det SparseBEV/map MapTR/occ FlashOcc
https://github.com/MCG-NJU/SparseBEV, https://github.com/hustvl/MapTR, https://github.com/Yzichen/FlashOCC
Paper: https://arxiv.org/abs/2308.09244, https://arxiv.org/abs/2208.14437, https://arxiv.org/abs/2311.12058

Note: Camera backbone HENet is HEAL in-house; upstream papers for det/map/occ are SparseBEV/MapTR/FlashOcc respectively.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14095

---

## Usage Guide

### Environment Setup

Complete HEAL environment installation first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses `use_lidar_gt=True`, requiring base+lidar+map+occ lmdb + meta+osm. Data paths read by the model (all joined from `common.py` `data_rootdir`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | Training lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | Evaluation lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_map_lmdb` | Training map lmdb (map GT) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb` | Evaluation map lmdb (map GT) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | Training occ lmdb (occupancy labels) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | Evaluation occ lmdb (occupancy labels) |
| `./tmp_data/heal_data/nuscenes/meta` | Map expansion precomputed directory |
| `./tmp_data/heal_data/nuscenes/osm` | OpenStreetMap vector map |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config).

nuScenes requires no extra preprocessing; all four packers (base/lidar/map/occ) include nuscenes-devkit parsing and read the official extracted directory directly. This task actually uses **base + lidar + map + occ lmdb** (+ meta/osm directories).

#### Official Download

nuScenes dataset official page: <https://www.nuscenes.org/download> (v1.0-trainval, registration required). Download the complete trainval package, including:

- `samples/`: Annotated keyframes (6 cameras + LiDAR)
- `sweeps/`: Intermediate frames
- `maps/`: Map expansion
- `v1.0-trainval/`: Metadata (annotation, scene, sample, ego_pose, etc. json)

Extract downloaded content to raw data directory `./tmp_data/orig_data/nuscenes/` (this directory is `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# Extract samples/ sweeps/ maps/ v1.0-trainval/ into this directory
```

This task includes occ labels; additionally download Occ3D-nuScenes labels (<https://github.com/Tsinghua-MARS-Lab/Occ3D>), place `occ3d/gts/<scene>/<token>/labels.npz` under `./tmp_data/orig_data/nuscenes/occ3d/gts/`:

```bash
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# Place Occ3D-nuScenes labels.npz by scene/token into this directory
```

This task requires meta directory (map expansion precomputed) and osm directory (OpenStreetMap vector map) for map packer and online map generation.

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

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and read the official extracted directory directly. Dependency order: base first; occ must be after base with matching version/split; lidar/map only require matching version/split. This task uses **base + lidar + map + occ lmdb**; run base, lidar, map, occ packers in sequence (occ after base).

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
  --use-dim 0 1 2 3 4

# Pack val lidar lmdb
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --max-sweeps 10 \
  --load-dim 5 \
  --use-dim 0 1 2 3 4
```

**map packer** (`nuscenes_map_packer.py`, `NuscenesMapGtPacker`) → `{split}_map_lmdb` (`--pc-range` must match `map_point_cloud_range`):

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

**occ packer** (`nuscenes_occ_packer.py`, `NuscenesOccPacker`) → `{split}_occ_lmdb` (must be after base, version/split consistent):

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
- `--target-data-dir`: Pack output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, matching config `data_rootdir`.
- `--pack-type lmdb`: Pack format is always lmdb.
- `--version v1.0-trainval`: nuScenes version (packer default v1.0-mini; must explicitly override to v1.0-trainval).
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--max-sweeps 10` (base/lidar packer): Maximum sweep count.
- `--load-dim 5` / `--use-dim 0 1 2 3 4` (lidar packer): Point cloud load/use dimensions, matching `common.py` `lidar_load_dim=5`, `lidar_use_dim=[0,1,2,3,4]`.
- `--map-path ./tmp_data/heal_data/nuscenes/meta` (map packer): Map expansion precomputed meta directory.
- `--pc-range -15 -30 -10 15 30 10` (map packer): Matches `map_point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]`.
- `--bev-size 100 50` (map packer): BEV size.
- `--fixed-ptsnum-per-line 20` (map packer): Fixed points per line, matching `fixed_ptsnum_per_gt_line=20`.
- `--map-classes divider ped_crossing boundary` (map packer): Map element classes, matching `map_classes`.
- `--occ-gt-dir occ3d/gts` (occ packer): Occ3D-nuScenes label directory (default); user must download and place separately.
- PV seg mask is not packed; generated online at training time (`defer_vectormap=True`).

> **Four packer summary** (this task uses base + lidar + map + occ):
> - **base** (`nuscenes_packer.py` / `NuscenesPacker`) → `{split}_lmdb`: Camera + annotation base data, **used by this task**.
> - **lidar** (`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`) → `{split}_lidar_lmdb`: LiDAR point clouds, **used by this task**.
> - **map** (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) → `{split}_map_lmdb`: Vectorized map GT, **used by this task**.
> - **occ** (`nuscenes_occ_packer.py` / `NuscenesOccPacker`) → `{split}_occ_lmdb`: Occ3D occupancy labels, **used by this task**.

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` layout (marked for what this task actually uses):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base train lmdb      ← used by this task
│   ├── val_lmdb/            # base val lmdb        ← used by this task
│   ├── train_lidar_lmdb/    # lidar train lmdb    ← used by this task
│   ├── val_lidar_lmdb/      # lidar val lmdb      ← used by this task
│   ├── train_map_lmdb/      # map train lmdb      ← used by this task
│   ├── val_map_lmdb/        # map val lmdb        ← used by this task
│   ├── train_occ_lmdb/      # occ train lmdb      ← used by this task
│   └── val_occ_lmdb/        # occ val lmdb        ← used by this task
├── meta/                    # Precomputed map expansion meta  ← used by this task
└── osm/                     # OpenStreetMap vector map ← used by this task
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

# Validate lidar lmdb (uses heal/visualize/lidar_det.py; BEV point cloud + 3D GT boxes; no model needed)
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar

# Validate map lmdb (uses heal/visualize/nuscenes_map.py; BEV vector map GT; no model; auto-reads pc_range from __meta__)
python3 tools/datasets/nuscenes_map_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/map
```

Visualization PNGs saved to each script's `--save-dir` (must be a local writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as save target); manually verify point clouds, vector maps, and occupancy labels align with packing.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch Size per GPU | Float Stage1 Steps | Float Stage2 Steps | Float Merge | Calibration Steps | Float Stage1 Optimizer | Float Stage2 Optimizer |
|---|---|---|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 4 | `num_steps_per_epoch * 200` (`stop_by="step"`) | `num_steps * 0.7` (`stop_by="step"`) | No training (`num_epochs=0`, weight merge only) | 100 | AdamW (backbone `lr=3e-4`, base `lr=3e-4`) | AdamW (`lr=1e-5`) |

This task uses three-stage Float training:
- **stage1** (`float_stage1`): Jointly train det+map+occ, full backbone update, loads pretrained backbone weights.
- **stage2** (`float_stage2`): Freeze backbone, train map/occ heads only (det absent), `lr=1e-5`, loads stage1 EMA weights.
- **merge** (`float_merge`): No training steps; merges stage1 (all weights) and stage2 (map/occ heads + neck weights) into complete model, saves as `float_merged-checkpoint-last.pth.tar`.

This task has no QAT stage. Calibration loads the complete model from merge output.

```bash
# Float stage1 training (joint det+map+occ; full backbone update)
HAT_TRAINING_STEP=float_stage1 python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage1
# Output:./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/float_stage1-checkpoint-best.pth.tar

# Float stage2 training (frozen backbone; train map/occ heads only)
HAT_TRAINING_STEP=float_stage2 python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage2
# Output:./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/float_stage2-checkpoint-best.pth.tar

# Float merge (merge stage1 + stage2 weights; no training)
HAT_TRAINING_STEP=float_merge python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_merge
# Output:./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/float_merged-checkpoint-last.pth.tar

# Quantization calibration (loads merge output)
python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage calibration
# Output:./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym/deploy.py
# Output:./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym/deploy.py
# Output:./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/compile/model.hbm
```

> Export uses `enable_vpu=True`; compilation uses `input_source=compile_cfg["input_source"]`.

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat — only re-export and re-compile on the quantized model. Edit the `march` field in `configs/sparse_multitask_det_maptr_flashocc_henet_tinym/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then rerun `export_hbir.py` and `compile_perf.py`. Compiled artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float stage1 evaluation (full stage1 model)
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage1

# Float stage2 evaluation (stage2 map/occ heads)
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage2

# Float merge evaluation (merged full model; recommended for final accuracy comparison)
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_merge

# Calibration evaluation
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage calibration

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage hbm
```

Evaluation metrics are NDS/mAP (det) + chamfer mAP (map, MAP) + Occ mIoU (occ, `MeanIOU`, `ignore_index=17`) on nuScenes val (`val_lmdb` + `val_lidar_lmdb` + `val_map_lmdb` + `val_occ_lmdb`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPU with `--device-ids` if needed.

#### Quantization Sensitivity Analysis (Optional)

This task defines `quant_analysis_solver` for per-layer quantization impact analysis to help diagnose accuracy drops:

```bash
python3 tools/quant_analysis.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym/quant_analysis.py
```

Results output to `./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/analysis/`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference data from the evaluation set (saved to `./demo/sparse_multitask_det_maptr_flashocc_henet_tinym/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym

# 2. Inference and visualization (float_merge shown; replace --stage with calibration/hbm)
python3 tools/inference.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_merge
```

Visualization results are saved to `./demo/sparse_multitask_det_maptr_flashocc_henet_tinym/`.
