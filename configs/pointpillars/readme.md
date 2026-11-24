# PointPillars (Car)

This task is a LiDAR-based 3D object detection task using the PointPillars model to detect the Car class on the KITTI 3D object detection dataset. This document describes the full pipeline from raw KITTI data through float training → quantization calibration → QAT → HBIR export → HBM compilation → inference visualization.

Model config directory: `configs/pointpillars/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| PointPillars | Single-frame LiDAR point cloud `(N,4)` | PointPillarScatter | SECONDNeck | Car 3D detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: KITTI val (`./tmp_data/heal_data/kitti3d/val_lmdb`). Metric: KITTI 3D AP (Car).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Kitti3D AP (Car) | 0.7731 | 0.7569 | 0.7709 | 0.7707 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 22.51 | 213.05 | 55.80 |
| J6P | 20.47 | 345.73 | 55.80 |
| J6B | 1353.52 | 2.75 | 63.40 |

Compiled artifact paths:
- HBIR: `./tmp_models/pointpillars/qat.bc`
- HBM: `./tmp_models/pointpillars/compile/model.hbm`

---

## Model Overview

### Core Design

PointPillars voxelizes point clouds into pillars; `PillarFeatureNet` learns pillar features, which are scattered into pseudo-images via `PointPillarScatter`, then multi-scale feature extraction and 3D box regression are performed by a SECOND-style FPN (`SECONDNeck`) and `PointPillarsHead`; training uses joint supervision with FocalLoss + SmoothL1Loss + direction classification.

- **Task type**: 3D object detection (LiDAR point cloud 3D Object Detection).
- **backbone**: `PointPillarScatter` (scatters 64-dim pillar features learned by `PillarFeatureNet` into `H×W×64` pseudo-image feature maps by coordinates, `use_horizon_pillar_scatter=True`).
- **neck**: `SECONDNeck` (SECOND-style FPN: downsample layer channels `[64, 128, 256]`, strides `[2, 2, 2]`; upsample layer channels `[128, 128, 128]`, strides `[1, 2, 4]`; outputs three-scale features concatenated to 384 channels).
- **Point cloud preprocessing**: `PointPillarsPreProcess` (voxelization: `pc_range=[0, -39.68, -3, 69.12, 39.68, 1]`, `voxel_size=[0.16, 0.16, 4]`, max 100 points per voxel, max 12000 voxels).
- **Feature extraction**: `PillarFeatureNet` (`num_filters=(64,)`, 4-dim input, MLP + max-pool to 64-dim pillar features).
- **Detection head**: `PointPillarsHead` (`in_channels=384`, `use_direction_classifier=True`, outputs classification + box regression + direction classification).
- **Anchor generation**: `Anchor3DGeneratorStride`: Car anchor size `1.6×3.9×1.56`, stride `[0.32, 0.32, 0.0]`, offset `[0.16, -39.52, -1.78]`, rotation angles `[0, 1.57]`; match threshold `0.6`, mismatch threshold `0.45`.
- **Post-processing**: `PointPillarsPostProcess`: NMS (`nms_iou_threshold=0.5`, `score_threshold=0.4`, `nms_pre_max_size=1000`, `nms_post_max_size=300`, `max_per_img=100`).
- **Loss**: `FocalLoss` (classification, `alpha=0.25, gamma=2.0, weight=1.0`) + `SmoothL1Loss` (box regression, `beta=1/9, weight=2.0`) + `CrossEntropyLoss` (direction, `weight=0.2`).
- **Model input**: Single-frame LiDAR point cloud, shape `(N, 4)`, N = number of points (deployment input padded to `150000` points), 4 dims = `[x, y, z, intensity]`.
- **Model output**: Car 3D boxes per frame (`x, y, z, w, l, h, θ`) + score + direction.
- **Classes**: `["Car"]`.

**Deployment notes**: The PointPillars deployment graph includes full voxelization + feature extraction + backbone + head + post-processing. HBIR export enables `enable_vpu=True`; compilation uses `input_source=["ddr"]` (point cloud read from DDR, not pyramid image input), unlike image-based tasks. This task has no subgraph export; only one `deploy.py`.

### Official Repo and Paper

Official repo: https://github.com/nutonomy/second.pytorch
Paper: https://arxiv.org/abs/1812.05784v1

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14086

---

## Usage Guide

### Environment Setup

Complete HEAL environment installation first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses KITTI 3D packed lmdb and several intermediate artifacts (info pkl / velodyne_reduced / gt_database / dbinfos). Data paths read by the model (all joined from `common.py` `base_data_dir = "./tmp_data/heal_data/kitti3d/"`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/kitti3d/train_lmdb` | Training lmdb (`data.py` `data_loader`) |
| `./tmp_data/heal_data/kitti3d/val_lmdb` | Evaluation lmdb (`data.py` `val_data_loader`) |
| `./tmp_data/heal_data/kitti3d/kitti3d_dbinfos_train.pkl` | GT sampling info read by `DataBaseSampler` at training time (`data.py`) |
| `./tmp_data/heal_data/kitti3d/kitti3d_gt_database/` | GT point cloud `.bin` files read by `DataBaseSampler` at training time (`data.py`) |
| `./tmp_data/heal_data/kitti3d/kitti3d_infos_{train,val,test,trainval}.pkl` | Dataset metadata (read during packing) |

> **Directory convention**: `orig_data/kitti3d` is `--src-data-dir` (raw extracted data); `heal_data/kitti3d` is `--target-data-dir` (packed lmdb + intermediate artifacts for config). KITTI intermediate artifacts (reduced point clouds / info pkl / gt_database) are tightly coupled to raw data (`velodyne_reduced` shares tree with `velodyne`; info pkl records relative paths), so `--prepare-data` runs `create_*` on the src directory to generate intermediates, then copies them to the target directory so the target is fully self-contained.

#### Official Download

KITTI 3D object detection dataset official page: <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>. Download 4 files:

- `data_object_image_2.zip` (training + test images)
- `data_object_label_2.zip` (training labels)
- `data_object_calib.zip` (calibration files)
- `data_object_velodyne.zip` (point clouds)

KITTI does not provide train/val/test splits officially; obtain ImageSets split files from the community (from `second.pytorch`):

```bash
mkdir -p ./tmp_data/orig_data/kitti3d/ImageSets
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/test.txt
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/train.txt
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/val.txt
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/trainval.txt
```

#### Directory Layout After Extraction

Extract the 4 zips to raw data directory `./tmp_data/orig_data/kitti3d/` (this directory is `--src-data-dir`), and place the ImageSets downloaded above:

```
./tmp_data/orig_data/kitti3d/
├── ImageSets/{train,val,test,trainval}.txt   # Community split files
├── training/
│   ├── calib/      # training/calib from data_object_calib.zip
│   ├── image_2/    # training/image_2 from data_object_image_2.zip
│   ├── label_2/    # training/label_2 from data_object_label_2.zip
│   └── velodyne/   # training/velodyne from data_object_velodyne.zip
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/   # Official test has no label_2; labels not public
```

Example extraction commands:

```bash
mkdir -p ./tmp_data/orig_data/kitti3d
unzip data_object_image_2.zip -d ./tmp_data/orig_data/kitti3d
unzip data_object_label_2.zip -d ./tmp_data/orig_data/kitti3d
unzip data_object_calib.zip -d ./tmp_data/orig_data/kitti3d
unzip data_object_velodyne.zip -d ./tmp_data/orig_data/kitti3d
```

#### Packing Commands

KITTI 3D data processing has three steps, all using `tools/datasets/kitti3d_packer.py` (`Kitti3DDetectionPacker`).

**Step 1: Initial data preparation (generate intermediate artifacts; split-independent; run once)**

`--prepare-data` runs on the src directory: `create_kitti_info_file` (generates `kitti3d_infos_{train,val,test,trainval}.pkl`) → `create_reduced_point_cloud` (generates `training/velodyne_reduced/`, `testing/velodyne_reduced/`) → `create_groundtruth_database` (generates `kitti3d_gt_database/` + `kitti3d_dbinfos_train.pkl`), and **copies** info pkl / gt_database / dbinfos to the target directory. This step is time-consuming (iterates all samples), split-independent, and **only needs to run once**.

```bash
python3 tools/datasets/kitti3d_packer.py \
  --prepare-data \
  --src-data-dir ./tmp_data/orig_data/kitti3d \
  --target-data-dir ./tmp_data/heal_data/kitti3d
```

**Step 2: Pack train lmdb**

During packing, the packer reads info pkl + velodyne_reduced + raw calib/image under src, and packs into lmdb output to target.

```bash
python3 tools/datasets/kitti3d_packer.py \
  --src-data-dir ./tmp_data/orig_data/kitti3d \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/kitti3d \
  --split-name train \
  --num-workers 20
```

**Step 3: Pack val lmdb**

```bash
python3 tools/datasets/kitti3d_packer.py \
  --src-data-dir ./tmp_data/orig_data/kitti3d \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/kitti3d \
  --split-name val \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/kitti3d` (contains `ImageSets/`, `training/`, `testing/`).
- `--target-data-dir`: Pack output directory `./tmp_data/heal_data/kitti3d`, produces `train_lmdb` and `val_lmdb`, plus info pkl / gt_database / dbinfos copied by prepare-data; matches `common.py` `base_data_dir`.
- `--prepare-data`: `store_true` flag; when set, only runs intermediate artifact generation and copy, does not pack lmdb.
- `--num-workers`: Parallel packing worker count (default 20).

#### Target Layout

After `--prepare-data` + two packing runs, `./tmp_data/heal_data/kitti3d/` layout:

```
./tmp_data/heal_data/kitti3d/
├── kitti3d_infos_train.pkl         # Copied by prepare-data
├── kitti3d_infos_val.pkl
├── kitti3d_infos_test.pkl
├── kitti3d_infos_trainval.pkl
├── kitti3d_dbinfos_train.pkl       # Read by DataBaseSampler during training
├── kitti3d_gt_database/            # .bin files read by DataBaseSampler
├── train_lmdb/                     # Packing output
└── val_lmdb/
```

Use the visualization script to spot-check packing results (point cloud and GT 3D boxes aligned):

```bash
python3 tools/datasets/kitti3d_viz.py \
  --data-path ./tmp_data/heal_data/kitti3d/train_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch Size per GPU | Float Training Epochs | Calibration Steps | QAT Training Epochs | Float Optimizer | QAT Optimizer | LR Schedule | Float SyncBN |
|---|---|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`) | 4 | 300 | 50 | 50 | AdamW (`lr=5e-4`, `betas=(0.95, 0.99)`, `weight_decay=0.01`) | SGD (`lr=2e-4`, `momentum=0.9`, `weight_decay=0`) | `CyclicLrUpdater`（`target_ratio=(10, 1e-4)`, `step_ratio_up=0.4`) | Enabled |

> This task starts Float training from scratch with **no LoadCheckpoint** and no external pretrained weights. Calibration loads this task's Float artifact; QAT loads this task's Calibration artifact (serial chain, no cross-task dependency).

```bash
# Float training
python3 tools/train.py \
  --config configs/pointpillars \
  --stage float
# Output:./tmp_models/pointpillars/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/pointpillars \
  --stage calibration
# Output:./tmp_models/pointpillars/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/pointpillars \
  --stage qat
# Output:./tmp_models/pointpillars/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/pointpillars/deploy.py
# Output:./tmp_models/pointpillars/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/pointpillars/deploy.py
# Output:./tmp_models/pointpillars/compile/model.hbm
```

> Export uses `enable_vpu=True`; compilation input `input_source=["ddr"]` (point cloud read from DDR), unlike pyramid input for image tasks.

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat — only re-export and re-compile on the existing QAT quantized model. Edit the `march` field in `configs/pointpillars/common.py` to the target chip:
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
  --config configs/pointpillars \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage hbm
```

Evaluation set is KITTI val (`./tmp_data/heal_data/kitti3d/val_lmdb`); metric is `Kitti3DMetricDet` (`compute_aos=True`, `current_classes=["Car"]`, `difficultys=[0,1,2]`); checkpoint selection uses `mAP_3D_moderate` as the monitoring metric. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPU with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference point cloud frames from the evaluation set (saved to `./demo/pointpillars/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference frames (read eval samples from gen_demo_cfg in infer.py, save to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/pointpillars

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/pointpillars \
  --stage float
```

Visualization results are saved to `./tmp_viz_results/pointpillars_hbir/`. `infer.py` also includes `bpu_val_preprocess` logic (generates `reduced_velodyne`/`gt`/`lst` and other evaluation helper files for on-board BPU evaluation alignment).
