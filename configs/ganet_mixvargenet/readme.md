# GaNet (MixVarGENet)

`ganet_mixvargenet` is a GaNet lane detection model with MixVarGENet backbone, performing end-to-end lane detection via keypoint heatmaps and per-point offset regression. This document explains how to complete the full pipeline from the original CULane dataset: float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/ganet_mixvargenet/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| GaNet | Single frame `1x3x320x800` | MixVarGENet | `GaNetNeck` | Lane point sequences `(B,L,P,2)` |

### Accuracy Metrics

Evaluation set: CULane test split (`./tmp_data/heal_data/CULane/test_lmdb`). Metric: CulaneF1Score (F1).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | CulaneF1Score (F1) | 0.7937 | 0.791 | — | 0.7908 |

> Results are based on `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is —).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.68 | 2262.93 | 6.50 |
| J6P | 0.61 | 8656.18 | 6.70 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifact paths:
- HBIR: `./tmp_models/ganet_mixvargenet/qat.bc`
- HBM: `./tmp_models/ganet_mixvargenet/compile/model.hbm`

---

## Model Overview

### Core Design

GaNet models lanes as "root keypoints + along-line offsets": the network predicts an 8× downsampled heatmap to locate lane starting points and regresses per-point offsets to reconstruct the full lane; the decoder filters keypoints by `kpt_thr` threshold, clusters and merges points on the same lane via `cluster_thr`, and finally maps back to original image coordinates.

- **Task type**: Lane detection (Lane Detection).
- **backbone**: MixVarGENet (`include_top=False`, `output_list=[2, 3, 4]` three-scale features, corresponding to stride 8/16/32).
- **neck**: `GaNetNeck`: FPN (three-scale fusion to `hid_dim=32`) + Attention module (`attn_ratio=4`, position encoding `pos_shape=(1,10,25)`).
- **Detection head**: `GaNetHead` (outputs `hid_dim=32` dimensional features).
- **Target generation**: `GaNetTarget` (`hm_down_scale=8`, keypoint radius `radius=2`).
- **Post-processing**: `GaNetDecoder` (`root_thr=1`, `kpt_thr=0.4`, `cluster_thr=5`, `downscale=8`).
- **Loss**: `GaNetLoss`: `LaneFastFocalLoss` (keypoint classification, weight=1.0) + `L1Loss` (per-point offset regression weight=0.5 + integer offset regression weight=1.0).
- **Model input**: Single frame, size `320 × 800` (`FixedCrop` takes original region `(0,270,1640,320)` then `Resize` to 320×800; original resolution 1640×590).
- **Model output**: Ordered point sequences per lane (original image coordinate system), obtained via `GaNetDecoder` from heatmap keypoint clustering + offset decoding.

**Deployment note**: The deployment graph (`deploy_model`) removes `targets`/`post_process`/`losses`, keeping only backbone + neck + head; output name is `pred` heatmap features; `GaNetDecoder` is re-attached during inference (float/hbir/hbm infer) to complete lane decoding.

### Official Repo and Paper

Official repo: https://github.com/Wolfwjs/GANet
Paper: https://arxiv.org/abs/2204.07335

Note: backbone MixVarGENet is HEAL-developed.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14098

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses CULane packed as lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/CULane/train_lmdb` | Training set lmdb (matches `common.py` `train_data_path`) |
| `./tmp_data/heal_data/CULane/test_lmdb` | Evaluation set lmdb (matches `common.py` `val_data_path`; CULane test split for evaluation) |

> Directory convention: `./tmp_data/orig_data/CULane/` is the **raw data extraction directory** (`--src-data-dir` for packing); `./tmp_data/heal_data/CULane/` is the **packed output directory** (`--target-data-dir` for config). Note **CULane is uppercase**.

#### Official Download

CULane official page: <https://xingyizhou.xyz/rgnn/> (or CULane official release page). Download these files:

- Images (three driver packages):
  - `driver_23-30.zip`
  - `driver_161-193.zip`
  - `driver_1823.zip`
- Labels: `annotations_new.tar.gz` (contains `list/` directory and `.lines.txt` annotation files under each driver)

Extract all files to the raw data directory `./tmp_data/orig_data/CULane/` (used as `--src-data-dir` for packing). **Extraction order matters**: extract three driver zips first (images), then **last** extract `annotations_new.tar.gz` (it writes `train.txt`/`test.txt`/`val.txt` to `list/` and `test_split/*.txt`, and writes corresponding `.lines.txt` annotation sub-files to each `driver_*/` directory).

```bash
mkdir -p ./tmp_data/orig_data/CULane

# 1. Extract three image archives first (driver_23-30 / driver_161-193 / driver_1823)
unzip driver_23-30.zip    -d ./tmp_data/orig_data/CULane
unzip driver_161-193.zip  -d ./tmp_data/orig_data/CULane
unzip driver_1823.zip     -d ./tmp_data/orig_data/CULane

# 2. Extract annotation archive last (generates list/ and lines files under driver_*/)
tar -xzf annotations_new.tar.gz -C ./tmp_data/orig_data/CULane
```

#### Directory Layout After Extraction

After extraction, `./tmp_data/orig_data/CULane/` has the following structure:

```
./tmp_data/orig_data/CULane/
├── driver_23-30/                       # Images, organized by driver
│   └── 00020/                          #   Subdirs are video clips
│       └── 0002940.jpg ...             #   Frame images (1640 × 590)
├── driver_161-193/                     # Images
│   └── ...
├── driver_1823/                        # Images
│   └── ...
└── list/                               # Extracted from annotations_new.tar.gz
    ├── train.txt                       # Training sample list (one relative path per line, e.g. driver_23-30/00020/0002940.jpg)
    ├── test.txt                        # Test sample list (for evaluation)
    ├── val.txt                         # Validation sample list
    └── test_split/                     # Test set split by scenario
        ├── test0_normal.txt
        ├── test1_crowd.txt
        ├── test2_hlight.txt
        ├── test3_shadow.txt
        ├── test4_noline.txt
        ├── test5_arrow.txt
        ├── test6_curve.txt
        ├── test7_cross.txt
        └── test8_night.txt
```

Each `driver_*/<segment>/` directory also contains `.lines.txt` annotation files with the same name as images (written by `annotations_new.tar.gz`); each line is one lane with coordinates in `x1 y1 x2 y2 ...` format. The packing script indexes samples via `list/train.txt` and `list/test.txt`, reading corresponding images and `.lines.txt` by relative path.

#### Packing Commands

Use `tools/datasets/culane_packer.py` (`CuLanePacker`) to pack official raw CULane into lmdb. `--src-data-dir` points to raw data `./tmp_data/orig_data/CULane` (with `driver_*/` images and `list/*.txt`); `--target-data-dir` points to packed output `./tmp_data/heal_data/CULane` (for config). The packer reads `list/train.txt` (train) or `list/test.txt` (test) per `--split-name`, outputting to `{target}/{split}_lmdb`. This task's config `val_data_path` uses `test_lmdb`, so the evaluation split is test.

```bash
# Pack train split (reads list/train.txt, outputs train_lmdb)
python3 tools/datasets/culane_packer.py \
  --src-data-dir ./tmp_data/orig_data/CULane \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/CULane \
  --split-name train \
  --num-workers 20

# Pack test split (reads list/test.txt, outputs test_lmdb for evaluation)
python3 tools/datasets/culane_packer.py \
  --src-data-dir ./tmp_data/orig_data/CULane \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/CULane \
  --split-name test \
  --num-workers 20
```

Parameter description:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/CULane` (with `driver_*/` images and `list/*.txt`).
- `--target-data-dir`: Packed output `./tmp_data/heal_data/CULane`, produces `train_lmdb` and `test_lmdb`, exactly matching `common.py` `train_data_path`/`val_data_path`.
- `--split-name`: `train` reads `list/train.txt`, `test` reads `list/test.txt`, producing `train_lmdb`/`test_lmdb` respectively.
- `--num-workers`: Parallel packing worker count (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/CULane/` has the following structure:

```
./tmp_data/heal_data/CULane/
├── train_lmdb/        # train lmdb
└── test_lmdb/         # val lmdb (config val_data_path points here)
```

Use the visualization script to sample and verify packing results:

```bash
python3 tools/datasets/culane_viz.py \
  --data-path ./tmp_data/heal_data/CULane/train_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | Float optimizer | LR schedule | Sync BN |
|---|---|---|---|---|---|---|
| 8（`device_ids = [0,1,2,3,4,5,6,7]`) | 64 | 240 | 10 | Adam (lr=0.01, weight_decay=4e-5) | CosLrUpdater (warmup 1 epoch) | sync_bn=True |

> This task has no QAT stage (`train.py` only defines `float_trainer` and `calibration_trainer`); quantization path is float → calibration → HBIR → HBM. Float training starts from scratch; no cross-task pretrained dependency.

```bash
# Float training
python3 tools/train.py \
  --config configs/ganet_mixvargenet \
  --stage float
# Output:./tmp_models/ganet_mixvargenet/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/ganet_mixvargenet \
  --stage calibration
# Output:./tmp_models/ganet_mixvargenet/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/ganet_mixvargenet/deploy.py
# Output:
#   ./tmp_models/ganet_mixvargenet/qat.bc (deployment HBIR for compile_perf.py)
#   ./tmp_models/ganet_mixvargenet/quantized_infer.bc (inference HBIR for hbir_predictor evaluation)

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/ganet_mixvargenet/deploy.py
# Output:./tmp_models/ganet_mixvargenet/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization calibration (calibration)**. When switching J6M/J6P/J6B, no need to re-run training/calibration — only re-run export and compilation on the existing calibration quantized model. Modify the `march` field in `configs/ganet_mixvargenet/common.py` to the target chip:
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
  --config configs/ganet_mixvargenet \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/ganet_mixvargenet \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/ganet_mixvargenet \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/ganet_mixvargenet \
  --stage hbm
```

Evaluation metric is CulaneF1Score (lane detection F1); evaluation set is CULane test split (`./tmp_data/heal_data/CULane/test_lmdb`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPU via `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference frames from the evaluation set (saved to `./demo/ganet_mixvargenet/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference frames (read eval samples from gen_demo_cfg in infer.py, save to ./demo/ganet_mixvargenet/)
python3 tools/gen_demo.py \
  --config configs/ganet_mixvargenet

# 2. Inference and visualization (float shown; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/ganet_mixvargenet \
  --stage float
```

Visualization results are saved to `./demo/ganet_mixvargenet/<stage>/` (subdirectories per stage).
