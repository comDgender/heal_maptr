# BEV Multitask + HENet-tiny

A camera-only BEV multitask model for 3D detection, occupancy, stop-line detection, and online lane-level mapping.

BEV Multitask + HENet-tiny lifts six surround cameras into a shared bird’s-eye feature and runs five heads together—dynamic/static objects, occupancy, stop lines, and vectorized map—so scene cues stay consistent. This release is calibrated for Horizon J6M / J6P deployment without lidar at inference.

Model config directory: `configs/bev_multitask_henet_tinym/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BEVFusionMultitask | 6-camera multi-view images `(B,6,3,512,960)` | HENet-tiny | FPN + SingleBevFormerViewTransformer | dynamic/static 3D boxes `(B,N,cls+reg)`; occ voxel semantics; discrete-map 2D boxes; map polylines |

### Accuracy Metrics

Evaluation set: nuScenes val (`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_occ_lmdb`). Metrics: NDS/mAP (`NuscenesMetric`, dynamic/static det), discrete-map NDS (`BEV2DNDSMetric`, stop line), Occ mIoU (`MeanIOU`, `ignore_index=17`), MapTR chamfer mAP (`NuscenesMapMetric`).

| March | Metric | float | calibration | hbm |
| --- | --- | --- | --- | --- |
| J6M | Dynamic NDS | 0.4553 | 0.4480 | 0.4535 |
|  | Static NDS | 0.4873 | 0.4839 | 0.4842 |
|  | Occ mIoU | 0.3422 | 0.3401 | 0.3399 |
|  | stop_line NDS | 0.4802 | 0.4762 | 0.4759 |
|  | MapTR mAP | 0.5486 | 0.5405 | 0.5376 |

> Results measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage.
>
> HEAL version: heal 0.14.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance benchmark**: FPS is measured with single-core 8 threads; latency is single-core single-thread; memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 77.82 | 13.21 | 386.90 |
| J6P | 47.65 | 76.17 | 373.50 |

> Latency is dominated by BPU (bpu-only: 77.72 ms J6M / 47.58 ms J6P; ucp < 0.1 ms). FPS is wall-clock throughput from the 8-thread run (not `8 / latency`). CSV: `./perf_results/bev_multitask_henet_tinym/perf.csv` (`tools/perf.py`).

Compilation artifacts (per-march under `./tmp_models/bev_multitask_henet_tinym/<march>/`):
- HBIR: `<march>/qat.bc` (+ `quantized_infer.bc`)
- HBM: `<march>/compile/model.hbm`

```bash
# Measure one chip on its default board (uploads the hbm over ssh)
PYTHONPATH=. python3 tools/perf.py \
  --hbm tmp_models/bev_multitask_henet_tinym/nash_m/compile/model.hbm \
  --chip j6m -c configs/bev_multitask_henet_tinym
```

---

## Model Overview

### Core Design

Six surround cameras are encoded by HENet-tiny, then lifted into a compact BEV canvas (about 100 m around the ego) with a single-frame BevFormer view transformer. That shared BEV feature fans out to five lightweight heads: SparseBEV-style queries for dynamic and static 3D boxes, a voxel occupancy head, a stop-line detector, and a MapTR-style polyline head for dividers, crossings, and boundaries — so detection, free space, and map cues stay consistent without running five separate networks.

- **Task type**: BEV multitask perception (dynamic det + static det + occupancy + discrete map + online map).
- **backbone**: HENet-tiny (`type=HENet`, `block_nums=[4,3,8,6]`, `embed_dims=[64,128,192,384]`, `GroupDWCB`/`AltDWCB`/`DWCB` blocks, `S2DDown` downsampling), loaded from the ImageNet-pretrained `henet_tinym` float checkpoint.
- **neck**: `FPN` (`in_strides=[32]` → `out_strides=[32]`, `_dim_=256`) + `BevFuseModule` (`input_c=384+256` → `fuse_c=256`; lidar branch disabled, `lidar_input=False`).
- **Dynamic det head**: `DenseBEVSparseHead` (`SparseBEVHead`-style, 8 classes, `num_anchor=384`, `num_decoder=6`, `MemoryBank` with `num_temp_instances=128` for tracking, DN denoising groups 5 + temporal 3).
- **Static det head**: `DenseBEVSparseHead` (2 classes: `traffic_cone`, `barrier`; `num_anchor=200`, `num_temp_instances=80`).
- **Occupancy head**: `BevformerOccDetDecoder` + `BEVOCCHead2D` (`Dz=16`, 18 classes, ROI crop `(14,14,114,114)`).
- **Discrete map head**: `BEVFormerDet2DDecoder` (1 class `stop_line`, `num_query=100`, 6-layer `Detection2DTransformerDecoder`).
- **Online map head**: `MaptrDetDecoder` + `MapInstanceDetectorHead` (3 classes: `divider`, `ped_crossing`, `boundary`; 6-layer `MapInstanceDecoder`, `num_vec_one2one=50` / `num_vec_one2many=300`, 20 points per polyline).
- **Loss functions**: det — `FocalLoss` (cls) + `L1Loss` (reg) + `CrossEntropyLoss` (cns) + `GaussianFocalLoss` (yns); occ — `CrossEntropyLoss`; maptr — `MapTRCriterion` (`FocalLoss` + `PtsL1Loss` + `PtsDirCosLoss` + aux seg); discrete map — `BevFormer2DCriterion` (`FocalLoss` + `L1Loss`). Task loss weights: det 4 / static 2 / occ 96 / discrete map 2 / maptr 1.1.
- **Model input**: 6-camera multi-view images `(B,6,3,512,960)` (YUV444 + `Normalize(128,128)`; camera-only, no lidar point cloud at inference).
- **Model output**: dynamic/static 3D boxes + occ voxel semantics + stop-line boxes + map polylines.

### Official Repo and Paper

SparseBEV-style det head repo: https://github.com/MCG-NJU/SparseBEV (paper: https://arxiv.org/abs/2308.09244)
MapTR repo: https://github.com/hustvl/MapTR (paper: https://arxiv.org/abs/2208.14437)

Note: The camera backbone HENet and the multitask assembly are HEAL in-house implementations; the official repos use different backbones and single-task setups.

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses nuScenes packed base lmdb + occ lmdb (fused via `MergeDataset(NuscenesMultitaskDataset + NuscenesOccDataset)`), plus the precomputed map expansion `meta/` directory and a k-means anchor file. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | Training base lmdb (dynamic/static det + maptr + discrete-map GT) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | Evaluation base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | Training occ lmdb (Occ3D occupancy labels) |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | Evaluation occ lmdb |
| `./tmp_data/heal_data/nuscenes/meta` | Precomputed map expansion (MapTR GT rasterization, `meta_rootdir`) |
| `./tmp_data/nuscenes/nuscenes_kmeans900.npy` | K-means anchor file (`anchor_file`) |

> **Directory convention**: `orig_data/nuscenes` is `--src-data-dir` (raw extracted data); `heal_data/nuscenes/v1.0-trainval` is `--target-data-dir` (packed lmdb for config consumption).

#### Official Download

nuScenes official page: <https://www.nuscenes.org/download> (registration required). Download the full **v1.0-trainval** dataset; after extraction includes `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/` (metadata json).

Extract into `./tmp_data/orig_data/nuscenes/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
```

This task includes occ labels; additionally download Occ3D-nuScenes labels (<https://github.com/Tsinghua-MARS-Lab/Occ3D>) and place `labels.npz` by scene/token under `./tmp_data/orig_data/nuscenes/occ3d/gts/`:

```bash
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# Place Occ3D-nuScenes labels.npz by scene/token into this directory
```

#### Directory Layout After Extraction

```
./tmp_data/orig_data/nuscenes/
├── samples/              # Keyframe images (6 camera subdirs)
├── sweeps/               # Intermediate frames
├── maps/                 # Map expansion (expansion + png)
├── v1.0-trainval/        # Metadata json
└── occ3d/
    └── gts/              # Occ3D-nuScenes occupancy labels (labels.npz)
```

#### Packing Commands

nuScenes requires no extra preprocessing; packers include nuscenes-devkit parsing and pack directly from the official extracted directory. Dependency order: **base first**; occ must follow base with matching version/split. This task uses **base lmdb + occ lmdb**; run base and occ packers each for train/val. Unused lidar/map packers are not needed by this task.

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

`NuscenesOccPacker` (occ) packs occupancy lmdb (`{split}_occ_lmdb`); must follow base with matching version/split:

```bash
# Pack occ train split
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20

# Pack occ val split
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
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/nuscenes` (contains `samples/`, `sweeps/`, `maps/`, `v1.0-trainval/`, `occ3d/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/nuscenes/v1.0-trainval`, corresponding to config `data_rootdir`.
- `--pack-type lmdb`: Pack format is lmdb.
- `--split-name`: `train` or `val`.
- `--version v1.0-trainval`: nuScenes dataset version (packer default is `v1.0-mini`; must override).
- `--occ-gt-dir occ3d/gts` (occ packer): Occ3D-nuScenes label directory (default); requires user download.
- `--num-workers`: Parallel packing workers (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/nuscenes/` layout (items used by this task are marked):

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # train base lmdb       ← used by this task
│   ├── val_lmdb/            # val base lmdb         ← used by this task
│   ├── train_lidar_lmdb/    # train lidar lmdb      (not used by this task)
│   ├── val_lidar_lmdb/      # val lidar lmdb        (not used by this task)
│   ├── train_map_lmdb/      # train map lmdb        (not used by this task)
│   ├── val_map_lmdb/        # val map lmdb          (not used by this task)
│   ├── train_occ_lmdb/      # train occ lmdb       ← used by this task
│   └── val_occ_lmdb/        # val occ lmdb         ← used by this task
├── meta/                    # Precomputed map expansion meta dir  ← used by this task
└── osm/                     # OSM vector map        (not used by this task)
```

Use visualization scripts to spot-check packing results:

```bash
# Validate base lmdb
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50

# Validate occ lmdb
python3 tools/datasets/nuscenes_occ_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters (float runs 8 GPUs × 2 machines, `num_machines=2` in `common.py`):

| # Training GPUs | Batch size per GPU | Stage | Epochs | LR | Optimizer | LR schedule |
|---|---|---|---|---|---|---|
| 8×2 (`device_ids=[0..7]`, `num_machines=2`) | 4 | stage1 (dynamic+static+occ+discrete map) | 200 | 1e-4 | AdamW (`weight_decay=0.1`, backbone `lr_mult=0.1`, det/occ/discrete_map heads `lr_mult=2`) | `CosineAnnealingLrUpdater` (warmup 500 steps, ratio 1/3) |
| 8×2 | 4 | stage2 (maptr only, frozen backbone/neck/view_transformer/fuse) | 100 | 2.5e-4 | AdamW | same |
| 8×2 | 4 | stage3 (full tasks joint) | 200 | 2e-5 | AdamW (maptr `lr_mult=3`, other heads `lr_mult=0.1`) | same |

#### Pretrained Weight Dependency

Stage1 float training loads the [`henet_tinym`](../henet_tinym/readme.md) ImageNet classification float checkpoint to initialize `camera_net.backbone` (via `LoadCheckpoint` + `update_state_dict_by_add_prefix(prefix="camera_net.")`):

- **Weight path**: `./tmp_pretrained_models/henet_tinym/float-checkpoint-best.pth.tar`.

Users must **train `henet_tinym` first** (or obtain its float ckpt) and place it at the path above before running stage1.

#### 3-Stage Float Training

Float training runs in three stages sequentially; stage checkpoints are chained (`float_stage1-checkpoint-last.pth.tar` → stage2 / stage3 inputs):

```bash
# Stage 1: dynamic + static + occ + discrete map
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym/train.py \
  --stage float_stage1
# Output: ./tmp_models/bev_multitask_henet_tinym/float_stage1-checkpoint-last.pth.tar

# Stage 2: maptr only (shared parts frozen)
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym/train.py \
  --stage float_stage2
# Output: ./tmp_models/bev_multitask_henet_tinym/float_stage2-checkpoint-last.pth.tar

# Stage 3: full tasks joint fine-tuning
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym/train.py \
  --stage float_stage3
# Output: ./tmp_models/bev_multitask_henet_tinym/float_stage3-checkpoint-best.pth.tar
```

Quantization calibration loads stage3 float output `float_stage3-checkpoint-best.pth.tar`. This task has no QAT stage; HBIR export reuses the calibration checkpoint with a fake-quant graph (`Float2QAT state="val"`).

```bash
# Quantization calibration
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym \
  --stage calibration
# Output: ./tmp_models/bev_multitask_henet_tinym/calibration-checkpoint-last.pth.tar
```

### Export and Compilation

```bash
# Export HBIR + compile HBM (re-run per march; add nash-b for J6B)
for march in nash-m nash-p; do
  HAT_TRAINING_STEP=calibration PYTHONPATH=. python3 tools/export_hbir.py \
    --config configs/bev_multitask_henet_tinym/deploy.py \
    --march $march
  PYTHONPATH=. python3 tools/compile_perf.py \
    --config configs/bev_multitask_henet_tinym/deploy.py \
    --march $march
done
# Outputs per march:
#   ./tmp_models/bev_multitask_henet_tinym/<march>/qat.bc
#   ./tmp_models/bev_multitask_henet_tinym/<march>/quantized_infer.bc
#   ./tmp_models/bev_multitask_henet_tinym/<march>/compile/model.hbm
```

> Export note: run `export_hbir.py` **without** `--debug` — under the debug branch the exporter saves only `.mlir` and skips the `.bc` that `compile_perf.py` reads.

> Compilation uses `input_source=["pyramid", "ddr", ...]` (camera `img` via pyramid/NV12 on VPU; projection matrices, BEV grids and MemoryBank caches via ddr); `enable_vpu=True`. Deploy inputs are tensor-only (`img` + `projection_mat` + grids + dynamic/static MemoryBank caches); the MemoryBank state is carried across frames by `BevMultitaskIrInfer`, not by the exported graph itself.

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and calibration**. Use `--march nash-b / nash-m / nash-p` on `export_hbir.py` and `compile_perf.py` (maps to J6B / J6M / J6P); per-march artifacts land in `./tmp_models/bev_multitask_henet_tinym/<march>/` (`nash_b` / `nash_m` / `nash_p` with `-` → `_`). Setting `march` in `configs/bev_multitask_henet_tinym/common.py` modifies the default chip for non-explicit runs.

### Accuracy Evaluation

```bash
# Float evaluation (stage3 full tasks)
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage calibration

# HBIR evaluation (requires export first)
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage hbir

# HBM evaluation (requires compilation, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage hbm
```

Per-stage float predictors evaluate each stage checkpoint with its matching metric subset (`float_stage1_predictor` / `float_stage2_predictor`; `float_predictor` is an alias of stage3):

```bash
# Evaluate stage1 ckpt (dynamic+static+occ+discrete map metrics)
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym/predict.py \
  --stage float_stage1
```

Evaluation metrics: NDS/mAP for dynamic det (8 classes) and static det (2 classes) via `NuscenesMetric`; stop-line discrete map via `BEV2DNDSMetric`; occ mIoU via `MeanIOU` (18 classes, `ignore_index=17`); online map via `NuscenesMapMetric` (chamfer). HBM evaluation feeds raw RGB uint8 images (NV12 conversion on-device); float/calibration/HBIR use YUV444 + `Normalize(128,128)`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference samples from the evaluation set (saved to `./demo/bev_multitask_henet_tinym/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference samples
python3 tools/gen_demo.py \
  --config configs/bev_multitask_henet_tinym

# 2. Inference and visualization (float shown; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/bev_multitask_henet_tinym \
  --stage float
```

Visualization results (camera + BEV det boxes, occ, map) are saved to `./demo/bev_multitask_henet_tinym/`.
