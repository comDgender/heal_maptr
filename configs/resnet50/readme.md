# ResNet-50

`resnet50` is an ImageNet (ILSVRC2012, 1000-class) image classification model based on the ResNet-50 backbone, covering the full pipeline from float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/resnet50/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| ResNet-50 | `1x3x224x224` | ResNet-50 | — | Classification logits `(B,1000)` |

### Accuracy Metrics

Evaluation set: ImageNet val (`./tmp_data/heal_data/imagenet/val_lmdb`, 50000 images). Metrics: Accuracy / TopKAccuracy(top_k=5).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Accuracy | 0.774 | 0.7721 | — | 0.7691 |
|  | TopKAccuracy(5) | — | — | — | — |

> Data measured with `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.89 | 1493.90 | 28.80 |
| J6P | 0.62 | 5974.73 | 29.00 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compiled artifact paths:
- HBIR: `./tmp_models/resnet50/qat.bc`
- HBM: `./tmp_models/resnet50/compile/model.hbm`

---

## Model Overview

### Core Design

ResNet-50 consists of 4 stages of residual blocks; each stage downsamples via stride=2, then global average pooling followed by a fully connected layer outputs class probabilities.

- **Task type**: Image classification (Image Classification).
- **backbone**: ResNet-50 (`ResNet50`, `num_classes=1000`), 4 stages of residual blocks, each stage downsamples via stride=2.
- **neck**: — (ResNet-50 has built-in fully connected classification head, no standalone neck).
- **Classification head**: ResNet-50 built-in fully connected classification head, directly outputs 1000-class logits.
- **Loss function**: `CEWithLabelSmooth` (cross-entropy with label smoothing).
- **Model input**: Single RGB image, resolution `224 × 224` (`1x3x224x224`).
- **Model output**: 1000-class prediction logits; argmax gives predicted class.

### Official Repo and Paper

Official repo: https://github.com/pytorch/vision (torchvision ResNet implementation)
Paper: https://arxiv.org/abs/1512.03385

---

## Usage Guide

### Environment Setup

Complete HEAL environment installation first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses ImageNet (ILSVRC2012) packed lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/imagenet/train_lmdb` | Training lmdb |
| `./tmp_data/heal_data/imagenet/val_lmdb` | Evaluation lmdb (ImageNet val, 50000 images) |

#### Official Download

ImageNet ILSVRC2012 dataset official page: <https://image-net.org/challenges/LSVRC/2012/> (registration required). Prepare three raw archives:

- `ILSVRC2012_img_train.tar`: Training set, contains 1000 class sub-archives `n*.tar` (~1300 images per class)
- `ILSVRC2012_img_val.tar`: Validation set, 50000 flat images
- `ILSVRC2012_devkit_t12.tar.gz`: Devkit, contains val ground truth and `meta.bin` metadata

Place downloaded archives in raw data directory `./tmp_data/orig_data/imagenet/` (this directory is `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/imagenet
# Place the three original archives in this directory
# ./tmp_data/orig_data/imagenet/ILSVRC2012_img_train.tar
# ./tmp_data/orig_data/imagenet/ILSVRC2012_img_val.tar
# ./tmp_data/orig_data/imagenet/ILSVRC2012_devkit_t12.tar.gz
```

#### Directory Layout After Extraction

Recommended: place the three raw archives directly under `./tmp_data/orig_data/imagenet/` (no manual extraction needed); the packer internally uses torchvision's `ImageNet` dataset interface: `parse_archives()` extracts `train/<class_id>/*.JPEG` from `ILSVRC2012_img_train.tar` (`<class_id>` is ImageNet WordNet class ID, e.g. n01440764), extracts val images from `ILSVRC2012_img_val.tar` and auto-sorts into `val/<class_id>/` using devkit ground truth. Devkit is required (generates `meta.bin`; val auto-sorting depends on it).

```
./tmp_data/orig_data/imagenet/
├── ILSVRC2012_img_train.tar         # 1000-class packed tar (keep as single tar; packer auto-extracts)
├── ILSVRC2012_img_val.tar           # 50000 flat val images
└── ILSVRC2012_devkit_t12.tar.gz     # devkit (generates meta.bin; required for val auto-categorization)
```

**Optional preprocessing**: If you have manually extracted `ILSVRC2012_img_train.tar` into individual `n*.tar` files (instead of keeping a single large tar), use `tools/datasets/extract_imagenet_train.sh` to extract `n*.tar` into `train/<class_id>/` layout:

```bash
bash tools/datasets/extract_imagenet_train.sh ./tmp_data/orig_data/imagenet
```

If a single `ILSVRC2012_img_train.tar` is kept under root, the packer's built-in torchvision handles it automatically; this script is not needed.

#### Packing Commands

Use `tools/datasets/imagenet_packer.py` (`ImageNetPacker`) to pack official raw ImageNet into classification lmdb. `--src-data-dir` points to raw data directory `./tmp_data/orig_data/imagenet` (contains three tar/devkit files); `--target-data-dir` points to pack output directory `./tmp_data/heal_data/imagenet` (for config). `orig_data/imagenet` is `--src-data-dir` (raw data); `heal_data/imagenet` is `--target-data-dir` (packed lmdb, matches config `data_path`).

Pack train and val once each:

```bash
# Pack train split
python3 tools/datasets/imagenet_packer.py \
  --src-data-dir ./tmp_data/orig_data/imagenet \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/imagenet \
  --split-name train \
  --num-workers 20

# Pack val split
python3 tools/datasets/imagenet_packer.py \
  --src-data-dir ./tmp_data/orig_data/imagenet \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/imagenet \
  --split-name val \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/imagenet` (contains `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`, `ILSVRC2012_devkit_t12.tar.gz`).
- `--target-data-dir`: Pack output directory `./tmp_data/heal_data/imagenet`, produces `train_lmdb` and `val_lmdb`, matching config `data_path + "train_lmdb/"` / `data_path + "val_lmdb/"`.
- `--pack-type lmdb`: Pack format is always lmdb.
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--num-workers`: Parallel packing worker count (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/imagenet/` layout:

```
./tmp_data/heal_data/imagenet/
├── train_lmdb/        # train lmdb
└── val_lmdb/          # val lmdb (ImageNet val, 50000 images)
```

Use the visualization script to spot-check packing results:

```bash
python3 tools/datasets/imagenet_viz.py \
  --data-path ./tmp_data/heal_data/imagenet/train_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch Size per GPU | Float Training Epochs | Calibration Steps | Float Optimizer | LR Schedule | Loss Function |
|---|---|---|---|---|---|---|
| 8 | 64 | 240 | 10 | SGD (lr=0.4, momentum=0.9, weight_decay=4e-5) | CosLrUpdater (cosine annealing, warmup 5 epochs) | CEWithLabelSmooth |

This task has no cross-task pretraining dependency; Float training starts from scratch. Calibration loads `float-checkpoint-best.pth.tar` from this task's Float training. This task has no QAT stage.

```bash
# Float training
python3 tools/train.py \
  --config configs/resnet50 \
  --stage float
# Output:./tmp_models/resnet50/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/resnet50 \
  --stage calibration
# Output:./tmp_models/resnet50/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/resnet50/deploy.py
# Output:./tmp_models/resnet50/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/resnet50/deploy.py
# Output:./tmp_models/resnet50/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration)**. When switching J6M/J6P/J6B, no need to rerun training/calibration — only re-export and re-compile on the quantized model. Edit the `march` field in `configs/resnet50/common.py` to the target chip:
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
  --config configs/resnet50 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/resnet50 \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/resnet50 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/resnet50 \
  --stage hbm
```

Evaluation metrics are Accuracy and TopKAccuracy(top_k=5) on ImageNet val (50000 images). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPU with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the evaluation set (saved to `./demo/resnet50/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference samples (read eval samples from gen_demo_cfg in infer.py, save to ./demo/resnet50/)
python3 tools/gen_demo.py \
  --config configs/resnet50

# 2. Inference and visualization (float shown; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/resnet50 \
  --stage float
```

Visualization results are saved to `./demo/resnet50/`.
