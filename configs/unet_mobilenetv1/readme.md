# UNet (MobileNetV1)

This document describes the `unet_mobilenetv1` model and how to complete the full pipeline from raw Cityscapes data through float training → quantization calibration → QAT → HBIR export → HBM compilation → inference.

Model config directory: `configs/unet_mobilenetv1/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| UNet | Single image `1x3x1024x2048` | MobileNetV1 | `DwUnet` | Segmentation mask `(B,19,H,W)` |

### Accuracy Metrics

Evaluation set: Cityscapes val (`./tmp_data/heal_data/cityscapes/val_lmdb`). Metric: MeanIOU (mIoU, 19 classes).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | MeanIOU | 0.6774 | 0.646 | 0.6707 | 0.6721 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 1.13 | 1115.70 | 15.80 |
| J6P | 0.86 | 6100.48 | 15.80 |
| J6B | 2.65 | 368.45 | 13.60 |

Compiled artifact paths:
- HBIR: `./tmp_models/unet_mobilenetv1/qat.bc`
- HBM: `./tmp_models/unet_mobilenetv1/compile/model.hbm`

---

## Model Overview

### Core Design

UNet encoder-decoder structure: MobileNetV1 extracts multi-scale features (stride 4/8/16/32/64); `DwUnet` upsamples level-by-level with depthwise separable convolutions and fuses feature maps; `SegHead` outputs segmentation logits at 5 scales; `SoftmaxFocalLoss` provides joint supervision; at deployment, `SegHead` exports only the highest-resolution (stride 4) scale argmax result.

- **Task type**: Semantic segmentation (Semantic Segmentation).
- **backbone**: MobileNetV1 (`alpha=0.25`, `include_top=False` removes classification head, lightweight depthwise separable convolution backbone).
- **neck**: `DwUnet` (`base_channels=8`, `output_scales=(4,8,16,32,64)`, depthwise separable convolution U-Net decoder).
- **Segmentation head**: `SegHead` (`num_classes=19`, outputs at stride 4/8/16/32/64 (5 scales); training outputs logits, deployment outputs argmax).
- **Loss function**: `SoftmaxFocalLoss` (scale-weighted, mitigates class imbalance).
- **Normalization**: BGR→YUV444 then `Normalize(mean=128, std=128)`, then split into multi-scale pyramid input via `scale_factors`.
- **Model input**: `1×3×1024×2048` (RGB, Cityscapes native resolution).
- **Model output**: Per-pixel 19-class semantic segmentation map `(B,19,H,W)`.

### Official Repo and Paper

Official repo: https://github.com/pytorch/vision (MobileNetV1 PyTorch implementation)
Paper: https://arxiv.org/abs/1704.04861

Note: UNet paper at https://arxiv.org/abs/1505.04597.

---

## Usage Guide

### Environment Setup

Complete HEAL environment installation first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses Cityscapes packed lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/cityscapes/train_lmdb` | Training lmdb |
| `./tmp_data/heal_data/cityscapes/val_lmdb` | Evaluation lmdb |

#### Official Download

Cityscapes dataset official page: <https://www.cityscapes-dataset.com/downloads/> (registration required). Obtain these two archives:

- `gtFine_trainvaltest.zip` (fine-grained annotations, includes train/val/test splits)
- `leftImg8bit_trainvaltest.zip` (raw left camera images, includes train/val/test splits)

Extract both zips to raw data directory `./tmp_data/orig_data/cityscapes/` (this directory is `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/cityscapes
unzip gtFine_trainvaltest.zip -d ./tmp_data/orig_data/cityscapes
unzip leftImg8bit_trainvaltest.zip -d ./tmp_data/orig_data/cityscapes
# Extract to ./tmp_data/orig_data/cityscapes/{gtFine,leftImg8bit}/...
```

Cityscapes requires no extra preprocessing scripts; pack directly after extraction.

#### Directory Layout After Extraction

After extraction, `./tmp_data/orig_data/cityscapes/` layout:

```
./tmp_data/orig_data/cityscapes/
├── gtFine/                                   # Fine-grained annotations
│   ├── train/                                # 18 cities
│   │   └── <city>/                           #   e.g. aachen, bochum, bremen, ...
│   │       ├── <name>_gtFine_color.png        #   Color visualization labels
│   │       ├── <name>_gtFine_instanceIds.png  #   Instance ID labels
│   │       └── <name>_gtFine_labelIds.png      #   Semantic label IDs (packer uses this file)
│   ├── val/                                  # 3 cities: frankfurt, lindau, munster
│   │   └── <city>/<name>_gtFine_{color,instanceIds,labelIds}.png
│   └── test/                                 # 6 cities (test has no public labelIds; not used by this task)
│       └── <city>/<name>_gtFine_{color,instanceIds,labelIds}.png
└── leftImg8bit/                              # Original left camera images
    ├── train/
    │   └── <city>/<name>_leftImg8bit.png      # 8 bit PNG, 1024×2048
    ├── val/
    │   └── <city>/<name>_leftImg8bit.png
    └── test/
        └── <city>/<name>_leftImg8bit.png
```

**Note**: The packer reads `gtFine/<split>/<city>/*_gtFine_labelIds.png` (`mode="fine"`, `target_type="semantic"`) and corresponding `leftImg8bit/<split>/<city>/*_leftImg8bit.png` via `torchvision.datasets.Cityscapes`; `color.png` and `instanceIds.png` are not used in packing.

#### Packing Commands

Use `tools/datasets/cityscapes_packer.py` (`CityscapesPacker`) to pack official raw Cityscapes into lmdb. `--src-data-dir` points to raw data directory `./tmp_data/orig_data/cityscapes` (contains `gtFine/`, `leftImg8bit/`); `--target-data-dir` points to pack output directory `./tmp_data/heal_data/cityscapes` (for config). `--split-name` corresponds to `torchvision.datasets.Cityscapes` `split` parameter, `train` or `val`. Pack train and val splits once each.

```bash
# Pack train split
python3 tools/datasets/cityscapes_packer.py \
  --src-data-dir ./tmp_data/orig_data/cityscapes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/cityscapes \
  --split-name train \
  --num-workers 20

# Pack val split
python3 tools/datasets/cityscapes_packer.py \
  --src-data-dir ./tmp_data/orig_data/cityscapes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/cityscapes \
  --split-name val \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/cityscapes` (contains `gtFine/`, `leftImg8bit/`).
- `--pack-type`: Pack format, always `lmdb`. Output name is `{split_name}_{pack_type}`, i.e. `train_lmdb`, `val_lmdb`.
- `--target-data-dir`: Pack output directory `./tmp_data/heal_data/cityscapes`, produces `train_lmdb` and `val_lmdb`, exactly matching paths joined from `common.py` `data_path="./tmp_data/heal_data/cityscapes/"`.
- `--split-name`: Data split, `train` or `val` (corresponds to `torchvision.datasets.Cityscapes` `split` parameter).
- `--num-workers`: Parallel packing worker count (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/cityscapes/` layout:

```
./tmp_data/heal_data/cityscapes/
├── train_lmdb/     # train lmdb
└── val_lmdb/       # val lmdb
```

Use the visualization script to spot-check packing results (input images aligned with semantic labels):

```bash
python3 tools/datasets/cityscapes_viz.py \
  --data-path ./tmp_data/heal_data/cityscapes/train_lmdb \
  --viz-num 50 \
  --save-path ./tmp_data/heal_data/cityscapes/viz
```

Visualization PNGs saved to `./tmp_data/heal_data/cityscapes/viz/`; manually verify images align with `gt_seg`.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch Size per GPU | Float Training Epochs | Calibration Steps | QAT Training Epochs | Float Optimizer | QAT Optimizer | SyncBN | # Classes |
|---|---|---|---|---|---|---|---|---|
| 8 | 4 | 600 | 10 | 30 | SGD (lr=0.01, momentum=0.9, weight_decay=5e-3, CosLr warmup 2 epoch) | SGD (lr=8e-5, momentum=0.9, weight_decay=5e-3, StepDecay@epoch20 ×0.1) | Enabled in Float and QAT stages | 19 |

> This task starts Float training from scratch with no cross-task pretrained weight dependency.

```bash
# Float training
python3 tools/train.py \
  --config configs/unet_mobilenetv1 \
  --stage float
# Output:./tmp_models/unet_mobilenetv1/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/unet_mobilenetv1 \
  --stage calibration
# Output:./tmp_models/unet_mobilenetv1/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/unet_mobilenetv1 \
  --stage qat
# Output:./tmp_models/unet_mobilenetv1/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/unet_mobilenetv1/deploy.py
# Output:./tmp_models/unet_mobilenetv1/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/unet_mobilenetv1/deploy.py
# Output:./tmp_models/unet_mobilenetv1/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat — only re-export and re-compile on the existing QAT quantized model. Edit the `march` field in `configs/unet_mobilenetv1/common.py` to the target chip:
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
  --config configs/unet_mobilenetv1 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage hbm
```

Evaluation metric is MeanIOU (mean Intersection over Union, mIoU) on Cityscapes val (`./tmp_data/heal_data/cityscapes/val_lmdb`). Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPU with `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference image frames from the evaluation set (saved to `./demo/unet_mobilenetv1/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference frames (read eval samples from gen_demo_cfg in infer.py, save to ./demo/unet_mobilenetv1/)
python3 tools/gen_demo.py \
  --config configs/unet_mobilenetv1

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/unet_mobilenetv1 \
  --stage float
```

Visualization results are saved to `./demo/unet_mobilenetv1/`.
