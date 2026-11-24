# BevFusion+PointPillar+HENet Multisensor Multitask

`bevfusion_pointpillar_henet_multisensor_multitask` is a BevFusion-based (CenterPointDetector + multisensor fusion) multitask model, jointly performing 3D object detection (det) and occupancy grid prediction (occ). The camera branch uses HENet to extract BEV features; the lidar branch uses PointPillars to extract BEV features; both are fused and fed to detection and occupancy heads respectively.

Model config directory: `configs/bevfusion_pointpillar_henet_multisensor_multitask/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BevFusion | 6-camera multi-view images `(B,6,3,512,960)` + lidar point cloud `(B,N,5)` | PointPillar (lidar) + HENet (camera) | FPN (camera) + SECONDNeck (lidar) | det bounding boxes `(B,N,cls+reg)`; occ occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb` + `val_occ_lmdb`). Metrics: NDS/mAP (det, `NuscenesMetric`) + Occ mIoU (`MeanIOU`, `ignore_index=17`).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.6421 | 0.6301 | — | 0.6294 |
|  | mAP | 0.5825 | 0.5724 | — | 0.5726 |
|  | Occ mIoU | 0.5187 | 0.52 | — | 0.5206 |

> Results measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 23.68 | 49.91 | 186.40 |
| J6P | 16.75 | 287.11 | 195.00 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifacts:
- HBIR: `./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/qat.bc`
- HBM: `./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/compile/model.hbm`

---

## Model Overview

### Core Design

The BevFusion multisensor multitask model extracts BEV features via dual branches: the camera branch uses HENet to extract multi-view features, then BevFormer ViewTransformer to BEV; the lidar branch uses PointPillars (`PillarFeatureNet` + `PointPillarScatter` + `SECONDNeck`) to voxelize point clouds and extract BEV features. Fused BEV features feed CenterPoint detection head (3D object detection) and occupancy head (semantic occupancy prediction) for joint det+occ training.

- **Task type**: Multisensor multitask fusion (3D object detection + occupancy grid prediction).
- **backbone**: Camera HENet (multi-view feature extraction) + lidar PointPillars (`PillarFeatureNet` + `PointPillarScatter`).
- **neck**: Camera FPN + lidar SECONDNeck.
- **Detection head**: CenterPoint detection head, outputting 10-class 3D bounding boxes + velocity (`num_classes=10`).
- **Occupancy head**: Semantic occupancy prediction, 18 classes (`num_classes_occ=18`).
- **BEV range**: `bev_size=(51.2, 51.2, 0.8)`, `bev_size_occ=(40, 40)`; `point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`.
- **Model input**: 6-camera multi-view images (B,6,3,512,960) + lidar point cloud (B,N,D).
- **Model output**: det 3D bounding boxes + occ occupancy grid semantics.

**Deployment note**: HBIR export enables `enable_vpu=True`; compilation uses `input_source="ddr, ddr, pyramid, ddr, ddr, ddr, ddr"` (DDR preferred; lidar features read from DDR).

### Official Repo and Paper

Official repo: https://github.com/mit-han-lab/bevfusion
Paper: https://arxiv.org/abs/2205.13542

Note: Camera branch is BevFormer, lidar branch is PointPillars/CenterPoint; camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14092

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes packed base lmdb + lidar lmdb + occ lmdb (`MergeDataset(Bev+Lidar+Occ)`). Data paths read by the model (all joined from `data_rootdir` in `common.py`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training base lmdb (camera + annotations) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | Training lidar lmdb (lidar point clouds) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | Evaluation lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | Training occ lmdb (occupancy labels) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | Evaluation occ lmdb (occupancy labels) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config consumption).

nuScenes requires no extra preprocessing; all four packers (base/lidar/map/occ) include nuscenes-devkit parsing and pack directly from the official extracted directory. This task uses **base + lidar + occ lmdb** (not map lmdb).

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

nuScenes requires no extra preprocessing; all four packers include nuscenes-devkit parsing and pack directly from the official extracted directory. Dependency order: base first; occ must follow base with matching version/split; lidar/map only require matching version/split. This task uses **base + lidar + occ lmdb**; run base, lidar, and occ packers in sequence (occ after base).

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
- `--max-sweeps 10` (base/lidar packer): Maximum sweep count.
- `--load-dim 5` / `--use-dim 0 1 2 3 4` (lidar packer): Point cloud load/use dimensions, matching `lidar_load_dim=5`, `lidar_use_dim=[0,1,2,3,4]` in `common.py`.
- `--occ-gt-dir occ3d/gts` (occ packer): Occ3D-nuScenes label directory (default); requires user download.
- `--num-workers`: Parallel packing workers (default 20).

> **Four packers overview** (this task uses base + lidar + occ):
> - **base** (`nuscenes_packer.py` / `NuscenesPacker`) → `{split}_lmdb`: camera + annotation base data, **used by this task**.
> - **lidar** (`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`) → `{split}_lidar_lmdb`: lidar point clouds, **used by this task**.
> - **map** (`nuscenes_map_packer.py` / `NuscenesMapGtPacker`) → `{split}_map_lmdb`: vectorized map GT; not used by this task.
> - **occ** (`nuscenes_occ_packer.py` / `NuscenesOccPacker`) → `{split}_occ_lmdb`: Occ3D occupancy labels, **used by this task**.

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` layout (items used by this task are marked):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base train lmdb      ← used by this task
│   ├── val_lmdb/            # base val lmdb        ← used by this task
│   ├── train_lidar_lmdb/    # lidar train lmdb    ← used by this task
│   ├── val_lidar_lmdb/      # lidar val lmdb      ← used by this task
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

# Validate lidar lmdb (uses heal/visualize/lidar_det.py; BEV point cloud + 3D GT boxes; no model needed)
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar
```

Visualization PNGs are saved to `--save-dir` (default `WORKSPACE/nuscenes_lidar_viz`; must be a local writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as save target). Manually verify point clouds and GT boxes are aligned.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | Float optimizer |
|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 4 | 24 | (see config) | AdamW (`lr=4e-4`) |

This task has no QAT stage. Calibration loads this task's float output `float-checkpoint-best.pth.tar`.

```bash
# Float training
python3 tools/train.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage float
# Output:./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage calibration
# Output:./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask/deploy.py
# Output:./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask/deploy.py
# Output:./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/compile/model.hbm
```

> Export uses `enable_vpu=True`; compilation uses `input_source="ddr, ddr, pyramid, ddr, ddr, ddr, ddr"` (DDR preferred; lidar features read from DDR).

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat—just re-run export and compilation on the quantized model. Modify the `march` field in `configs/bevfusion_pointpillar_henet_multisensor_multitask/common.py` to the target chip (`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B), then re-run `export_hbir.py` and `compile_perf.py`. Compilation artifacts differ per chip; performance cannot be directly transferred across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage hbm
```

Evaluation metrics are NDS/mAP (det) and Occ mIoU (occ, `MeanIOU`, `ignore_index=17`); evaluation set is nuScenes val (`val_lmdb` + `val_lidar_lmdb` + `val_occ_lmdb`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPUs with `--device-ids` if needed.

#### Quantization Sensitivity Analysis (Optional)

This task defines `quant_analysis_solver` for per-layer quantization impact analysis to help diagnose accuracy drops:

```bash
python3 tools/quant_analysis.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask/quant_analysis.py
```

Results are written to `./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/analysis/`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference data from the evaluation set (saved to `./demo/bevfusion_pointpillar_henet_multisensor_multitask/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask

# 2. Inference and visualization (float shown; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage float
```

Visualization results are saved to `./demo/bevfusion_pointpillar_henet_multisensor_multitask/`.
