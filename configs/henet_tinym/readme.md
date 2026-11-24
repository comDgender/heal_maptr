# HENet-tinyM

`henet_tinym` is an ImageNet (ILSVRC2012, 1000-class) image classification model based on the HENet backbone (tiny-M configuration), covering the full pipeline of float training → quantization calibration → HBIR export → HBM compilation → inference.

Model config directory: `configs/henet_tinym/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| HENet-tinym | `1x3x224x224` | HENet | — | classification logits `(B,1000)` |

### Accuracy Metrics

Evaluation set: ImageNet val (`./tmp_data/heal_data/imagenet/val_lmdb`, 50000 images). Metrics: Accuracy / TopKAccuracy(top_k=5).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Accuracy | 0.784 | 0.7795 | — | 0.7801 |
|  | TopKAccuracy(5) | — | — | — | — |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with 8 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.52 | 3429.49 | 12.90 |
| J6P | 0.43 | 11190.73 | 13.10 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compiled artifact paths:
- HBIR: `./tmp_models/henet_tinym/qat.bc`
- HBM: `./tmp_models/henet_tinym/compile/model.hbm`

---

## Model Overview

### Core Design

HENet uses depthwise conv to build lightweight blocks with LayerScale for stable training; the tiny-M configuration has widths 64/128/192/384 and depths 4/3/8/6, with GroupDWCB in the first stage and parameter count in the tiny range.

- **Task type**: Image Classification.
- **backbone**: HENet (tiny-M configuration, `depth/block_nums=[4,3,8,6]`, `width/embed_dims=[64,128,192,384]`, `block_cls=["GroupDWCB","GroupDWCB","AltDWCB","DWCB"]`, GELU activation, LayerScale, S2D downsampling, `num_classes=1000`), uses depthwise conv to build lightweight blocks with LayerScale for stable training; first stage uses GroupDWCB.
- **neck**: — (HENet has a built-in fully connected classification head; no separate neck).
- **classification head**: HENet built-in fully connected classification head (`include_top=True`, `feature_mix_channel=1024`).
- **Loss function**: `SoftTargetCrossEntropy` (soft-label cross-entropy with mixup).
- **Model input**: single RGB image at resolution `224 × 224` (`1x3x224x224`).
- **Model output**: 1000-class prediction logits; argmax gives the predicted class.

### Official Repo and Paper

HybridEfficient Network is an efficient backbone designed by Horizon Robotics for the Journey series chips.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14094
---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see the top-level [`README.md`](../../README.md)). Run the following commands from the **HEAL repository root**:

### Data Preparation

This task uses packed ImageNet (ILSVRC2012) lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/imagenet/train_lmdb` | training lmdb |
| `./tmp_data/heal_data/imagenet/val_lmdb` | evaluation lmdb (ImageNet val, 50000 images) |

#### Official Download

ImageNet ILSVRC2012 official page: <https://image-net.org/challenges/LSVRC/2012/> (registration required for download). Prepare three original archives:

- `ILSVRC2012_img_train.tar`: training set, containing 1000 per-class sub-archives `n*.tar` (~1300 images per class)
- `ILSVRC2012_img_val.tar`: validation set, 50000 flat images
- `ILSVRC2012_devkit_t12.tar.gz`: devkit with val ground truth and `meta.bin` metadata

After download, place the archives in the raw data directory `./tmp_data/orig_data/imagenet/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/imagenet
# Place the three original archives in this directory
# ./tmp_data/orig_data/imagenet/ILSVRC2012_img_train.tar
# ./tmp_data/orig_data/imagenet/ILSVRC2012_img_val.tar
# ./tmp_data/orig_data/imagenet/ILSVRC2012_devkit_t12.tar.gz
```

#### Directory Layout After Extraction

Recommended: place the three original archives under `./tmp_data/orig_data/imagenet/` without manual extraction. The packing script handles extraction automatically via torchvision's `ImageNet` dataset API: `parse_archives()` extracts `train/<class_id>/*.JPEG` from `ILSVRC2012_img_train.tar` (`<class_id>` is the ImageNet WordNet class ID, e.g. n01440764), extracts val images from `ILSVRC2012_img_val.tar`, and auto-sorts them into `val/<class_id>/` using devkit ground truth. devkit is required (generates `meta.bin`; val auto-sorting depends on it).

```
./tmp_data/orig_data/imagenet/
├── ILSVRC2012_img_train.tar         # single tar of 1000 classes (keep as one tar; packer auto-extracts)
├── ILSVRC2012_img_val.tar           # 50000 flat val images
└── ILSVRC2012_devkit_t12.tar.gz     # devkit (required for meta.bin and val auto-sorting)
```

**Optional preprocessing**: If you have manually extracted `ILSVRC2012_img_train.tar` into multiple `n*.tar` files (instead of keeping one large tar), use `tools/datasets/extract_imagenet_train.sh` to extract `n*.tar` into the `train/<class_id>/` layout:

```bash
bash tools/datasets/extract_imagenet_train.sh ./tmp_data/orig_data/imagenet
```

If a single `ILSVRC2012_img_train.tar` remains at root, the packer's built-in torchvision handles it; this script is not needed.

#### Packing Commands

Use `tools/datasets/imagenet_packer.py` (`ImageNetPacker`) to pack official raw ImageNet into classification lmdb. `--src-data-dir` points to the raw data directory `./tmp_data/orig_data/imagenet` (three tar/devkit files); `--target-data-dir` points to the packed output directory `./tmp_data/heal_data/imagenet` (read by config). `orig_data/imagenet` is `--src-data-dir` (raw data); `heal_data/imagenet` is `--target-data-dir` (packed lmdb, corresponding to config `data_path`).

Pack train and val separately:

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
- `--src-data-dir`: raw data root `./tmp_data/orig_data/imagenet` (contains `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`, `ILSVRC2012_devkit_t12.tar.gz`).
- `--target-data-dir`: packed output directory `./tmp_data/heal_data/imagenet`, produces `train_lmdb` and `val_lmdb`, matching config paths `data_path + "train_lmdb/"` / `data_path + "val_lmdb/"`.
- `--pack-type lmdb`: pack format is lmdb.
- `--split-name`: `train` or `val`, for training and validation sets respectively.
- `--num-workers`: parallel packing worker count (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/imagenet/` has the following layout:

```
./tmp_data/heal_data/imagenet/
├── train_lmdb/        # training lmdb
└── val_lmdb/          # evaluation lmdb (ImageNet val, 50000 images)
```

Use the visualization script to spot-check packing correctness:

```bash
python3 tools/datasets/imagenet_viz.py \
  --data-path ./tmp_data/heal_data/imagenet/train_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Per-GPU Batch Size | Float Training Epochs | Calibration Steps | Float Optimizer | LR Schedule | Loss |
|---|---|---|---|---|---|---|
| 8 | 128 | 300 | 100 | AdamW (lr=2e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.05) | CosLrUpdater (cosine annealing, warmup 5 epoch) | SoftTargetCrossEntropy (mixup enabled, mixup_alpha=0.8, cutmix_alpha=1.0) |

> Data augmentation: TimmTransforms (rand-augment `rand-m9-mstd0.5-inc1`, RandomErasing re_prob=0.25, bicubic interpolation).

No cross-task pretraining; Float training starts from scratch. Calibration loads Float training output `float-checkpoint-best.pth.tar` from this task. This task has no QAT stage.

```bash
# Float training
python3 tools/train.py \
  --config configs/henet_tinym \
  --stage float
# Output: ./tmp_models/henet_tinym/float-checkpoint-best.pth.tar

# Quantization calibration (Calibration)
python3 tools/train.py \
  --config configs/henet_tinym \
  --stage calibration
# Output: ./tmp_models/henet_tinym/calibration-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/henet_tinym/deploy.py
# Output: ./tmp_models/henet_tinym/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/henet_tinym/deploy.py
# Output: ./tmp_models/henet_tinym/compile/model.hbm
```

> **Switching chips (march) tips**: Switching chips only affects HBIR export and HBM compilation; it does **not** affect prior float training and quantization (calibration). When switching J6M/J6P/J6B, you do not need to rerun training/calibration—re-run export and compilation on the quantized model. Set the `march` field in `configs/henet_tinym/common.py` to the target chip:
> ```python
> march = March.NASH_M   # J6M
> march = March.NASH_P   # J6P
> march = March.NASH_B   # J6B
> ```
> Then re-run `export_hbir.py` and `compile_perf.py` (the two commands above). Compiled artifacts differ per chip; performance is not directly portable across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/henet_tinym \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/henet_tinym \
  --stage calibration

# HBIR evaluation (requires export from Export and Compilation first)
python3 tools/predict.py \
  --config configs/henet_tinym \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation, plus board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/henet_tinym \
  --stage hbm
```

Evaluation metrics are Accuracy and TopKAccuracy(top_k=5) on ImageNet val (50000 images). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPUs with `--device-ids`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference images from the eval set (saved to `./demo/henet_tinym/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference samples (reads eval samples from gen_demo_cfg in infer.py, saved to ./demo/henet_tinym/)
python3 tools/gen_demo.py \
  --config configs/henet_tinym

# 2. Inference and visualization (float example; replace --stage with calibration/hbir/hbm)
python3 tools/inference.py \
  --config configs/henet_tinym \
  --stage float
```

Visualization output saved to `./demo/henet_tinym/`.
