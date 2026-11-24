# SparseBEV + HENet-tiny

`bev_sparse_henet_tinym` is a BEV 3D detection model with HENet-tiny backbone + SparseBEVHead, using MergeDataset to fuse camera base lmdb and lidar lmdb, covering the full pipeline from float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/bev_sparse_henet_tinym/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| SparseBEV | 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` | HENet-tiny | MMFPN + DenseDepthNet | 3D bounding boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb`). Metrics: NDS / mAP.

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5414 | 0.5319 | — | 0.5307 |
|  | mAP | 0.4338 | 0.4265 | — | 0.4235 |

> Results measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 10.43 | 98.46 | 72.80 |
| J6P | 7.38 | 527.73 | 85.90 |
| J6B | 23.16 | 44.28 | 71.00 |

Compilation artifacts:
- HBIR: `./tmp_models/bev_sparse_henet_tinym/qat.bc`
- HBM: `./tmp_models/bev_sparse_henet_tinym/compile/model.hbm`

---

## Model Overview

### Core Design

SparseBEV performs 3D detection in BEV space via sparse queries: HENet-tiny extracts multi-view camera features, `MMFPN` fuses multi-scale features, `DenseDepthNet` uses lidar point clouds for dense depth supervision during training; `SparseBEVHead` refines sparse queries layer by layer via `MemoryBank` + `SparseBEVEncoder` (with `AsymmetricFFN`, `DeformableFeatureAggregation`, `SparseBEVRefinementModule`) and outputs detection boxes.

- **Task type**: BEV 3D object detection (BEV 3D Object Detection).
- **backbone**: HENet-tiny (`type=HENet`, `in_channels=3`, 4 stages `[4,3,8,6]` blocks, `embed_dims=[64,128,192,384]`, `GroupDWCB`/`AltDWCB`/`DWCB` blocks, `S2DDown` downsampling; multi-view camera feature extraction).
- **neck**: `MMFPN` (`in_strides=[2,4,8,16,32]`→`out_strides=[4,8,16,32]`, multi-scale fusion) + `DenseDepthNet` (dense depth supervision from lidar point clouds during training, `num_depth_layers=3`).
- **Detection head**: `SparseBEVHead` (`MemoryBank` + `SparseBEVEncoder` + `SparseBEVEncoder`, with `AsymmetricFFN`, `DeformableFeatureAggregation`, `SparseBEVRefinementModule`, layer-wise sparse query refinement, `num_classes=10`, `num_decoder=6`, `num_anchors=384`).
- **Loss function**: `FocalLoss` (cls) + `L1Loss` (reg) + `CrossEntropyLoss` (cns) + `GaussianFocalLoss` (yns).
- **Model input**: 6-camera multi-view images `(B,6,3,256,704)` + lidar point cloud `(B,N,5)` (`load_dim=5`, `use_dim=[0,1,2,4]`, `num_lidar_sweeps=0`, dense depth GT auxiliary).
- **Model output**: 10-class 3D bounding boxes (nuScenes 10 classes, `num_classes=10`), `(B,N,cls+reg)`.

### Official Repo and Paper

Official repo: https://github.com/MCG-NJU/SparseBEV 
Paper: https://arxiv.org/abs/2308.09244

Note: The camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14074

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes packed base lmdb and lidar lmdb (fused via `MergeDataset`). Data paths read by the model (all joined from `data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"` in `common.py`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training base lmdb (`NuscenesBevDataset`) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation base lmdb (`NuscenesBevDataset`) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | Training lidar lmdb (`NuscenesLidarDataset`, dense depth GT) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | Evaluation lidar lmdb (`NuscenesLidarDataset`) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config consumption).

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (registration required). Download the full **v1.0-trainval** dataset; after extraction includes `samples/` (annotated keyframes, 6 cameras), `sweeps/` (intermediate frames), `maps/` (map expansion), `v1.0-trainval/` (metadata json: annotation, scene, sample, ego_pose, etc.).

Extract the above into `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir` for packing):

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
├── samples/              # Keyframe images (6 camera subdirs)
├── sweeps/               # Intermediate frames (6 camera subdirs)
├── maps/                 # Map expansion (expansion + png)
└── v1.0-trainval/        # Metadata json (annotation, scene, sample, ego_pose, etc.)
```

#### Packing Commands

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and pack directly from the official extracted directory (do not run create_data_*.py).

nuScenes has four packers; dependency order: **base first**; occ must follow base with matching version/split; lidar/map only require matching version/split. This task uses **base lmdb + lidar lmdb**; run base and lidar packers each for train/val. Unused map packer (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) packs map lmdb for online map vectorization tasks; occ packer (`nuscenes_occ_packer.py` / `NuscenesOccPacker`) packs occ lmdb for occupancy prediction tasks (requires Occ3D labels); neither is used by this task.

`NuscenesPacker` (base) packs base camera lmdb (`{split}_lmdb`):

```bash
# Pack base train split
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# Pack base val split
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

`NuscenesLidarPacker` (lidar) packs lidar point cloud lmdb (`{split}_lidar_lmdb`):

```bash
# Pack lidar train split
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# Pack lidar val split
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, corresponding to config `data_rootdir`.
- `--pack-type lmdb`: Pack format is lmdb.
- `--split-name`: `train` or `val` for training or validation set.
- `--version v1.0-trainval`: nuScenes dataset version (packer default is `v1.0-mini`; must override to `v1.0-trainval`).
- `--num-workers`: Parallel packing workers (default 20).
- `--max-sweeps`: Maximum sweep count for base/lidar packer (default 10).
- `--load-dim`: Raw point cloud dimension for lidar packer (default 5).

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` layout (this task uses base + lidar lmdb under `v1.0-trainval/`):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # train base lmdb          ← used by this task
│   ├── val_lmdb/            # val base lmdb            ← used by this task
│   ├── train_lidar_lmdb/    # train lidar lmdb         ← used by this task
│   ├── val_lidar_lmdb/      # val lidar lmdb           ← used by this task
│   ├── train_map_lmdb/      # train map lmdb (not used by this task)
│   ├── val_map_lmdb/        # val map lmdb (not used by this task)
│   ├── train_occ_lmdb/      # train occ lmdb (not used by this task)
│   └── val_occ_lmdb/        # val occ lmdb (not used by this task)
├── meta/                    # Precomputed map expansion meta directory
└── osm/                     # OSM vector map (for map tasks)
```

Use visualization scripts to spot-check packing results. Base lmdb uses `nuscenes_viz.py`:

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

Lidar lmdb uses `nuscenes_lidar_viz.py` (packing validation, reuses `lidar_det_visualize` from `heal/visualize/lidar_det.py`, renders BEV point cloud + 3D GT boxes, no model required):

```bash
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar
```

Visualization PNGs are saved to `--save-dir` (default `WORKSPACE/nuscenes_lidar_viz`; must be a local writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as save target). Manually verify point clouds and GT boxes are aligned.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | Float optimizer | LR schedule | Loss function |
|---|---|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 4 | 100 | 12 | AdamW (`lr=3e-4`, `weight_decay=1e-3`)<br>backbone separate `lr=3e-4` | `CosLrUpdater` (`warmup_lr_ratio=1/3`, `min_lr_ratio=1e-3`) | `FocalLoss` + `L1Loss` + `CrossEntropyLoss` + `GaussianFocalLoss` |

This task's float training starts from scratch with no cross-task pretraining dependency. Calibration loads this task's float output `float-checkpoint-best.pth.tar`. This task has no QAT stage.

```bash
# Float training
python3 tools/train.py \
  --config configs/bev_sparse_henet_tinym \
  --stage float
# Output:./tmp_models/bev_sparse_henet_tinym/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/bev_sparse_henet_tinym \
  --stage calibration
# Output:./tmp_models/bev_sparse_henet_tinym/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/bev_sparse_henet_tinym/deploy.py
# Output:./tmp_models/bev_sparse_henet_tinym/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/bev_sparse_henet_tinym/deploy.py
# Output:./tmp_models/bev_sparse_henet_tinym/compile/model.hbm
```

> Compilation uses `input_source="pyramid, ddr, ddr, ddr"` (camera images via pyramid, lidar auxiliary via ddr); `enable_vpu` is defined via qconfig. This task's deploy.py has no explicit `enable_vpu` line (via qconfig).

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration)**. When switching J6M/J6P/J6B, no need to rerun training/calibration—just re-run export and compilation on the quantized model. Modify the `march` field in `configs/bev_sparse_henet_tinym/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then re-run `export_hbir.py` and `compile_perf.py`. Compilation artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage hbm
```

Evaluation metrics are NDS and mAP; evaluation set is nuScenes val (`val_lmdb` + `val_lidar_lmdb`, `val_nuscenes_metric`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPUs with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference samples from the evaluation set (saved to `./demo/bev_sparse_henet_tinym/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/bev_sparse_henet_tinym/)
python3 tools/gen_demo.py \
  --config configs/bev_sparse_henet_tinym

# 2. Inference and visualization (float shown; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/bev_sparse_henet_tinym \
  --stage float
```

Visualization results are saved to `./demo/bev_sparse_henet_tinym/`.
