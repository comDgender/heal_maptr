# Deformable DETR (ResNet-50)

This document introduces the `deform_detr_resnet50` model and explains how to complete the full pipeline from the original COCO 2017 dataset: float training → quantization calibration → QAT → HBIR export → HBM compilation → inference.

Model config directory: `configs/deform_detr_resnet50/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| DeformableDETR | Single image `1x3x800x1332` | ResNet-50 | `ChannelMapperNeck` | Detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: COCO val2017 (`./tmp_data/heal_data/mscoco/val_lmdb` + `instances_val2017.json`). Metric: COCO mAP.

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | COCO mAP | 0.4384 | 0.412 | 0.4526 | 0.4529 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 144.26 | 6.94 | 647.40 |
| J6P | 78.36 | 29.15 | 656.30 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifact paths:
- HBIR: `./tmp_models/deform_detr_resnet50/qat.bc`
- HBM: `./tmp_models/deform_detr_resnet50/compile/model.hbm`

---

## Model Overview

### Core Design

Deformable DETR replaces DETR's global self-attention with multi-scale deformable attention: each query only samples a small number of points near reference points, enabling faster convergence and lower compute. Four feature levels plus 900 queries provide multi-scale candidates, then the decoder progressively refines boxes layer by layer.

- **Task type**: 2D object detection (2D Object Detection).
- **backbone**: ResNet-50 (`ResNet50`, `include_top=False` removes classification head).
- **neck**: `ChannelMapperNeck` (`in_channels=[512,1024,2048]`, `out_channel=256`, 1×1 conv, `extra_convs=1`).
- **Position encoding**: `PositionEmbeddingSine` (`num_pos_feats=128`, normalized).
- **Transformer**: `DeformableDetrTransformer` (encoder 6 layers + decoder 6 layers, `embed_dim=256`, `num_heads=8`, `feedforward_dim=1024`, `num_feature_levels=4`, `num_queries=900`).
- **Post-processing**: `DeformDetrPostProcess` (evaluation selects `select_box_nums_for_evaluation=300` boxes).
- **Loss**: `DeformableCriterion`: classification focal loss + L1 bbox + GIoU, `HungarianMatcher` bipartite matching, `aux_loss=True`.
- **Key flags**: `with_box_refine=False`, `as_two_stage=False`.
- **Model input**: Single image, size `800 × 1332`.
- **Model output**: 80-class detection boxes + confidence scores.

### Official Repo and Paper

Official repo: https://github.com/fundamentalvision/Deformable-DETR
Paper: https://arxiv.org/abs/2010.04159

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses MSCOCO 2017 packed as lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/mscoco/train_lmdb` | Training set lmdb |
| `./tmp_data/heal_data/mscoco/val_lmdb` | Evaluation set lmdb |
| `./tmp_data/heal_data/mscoco/instances_val2017.json` | COCO annotations for evaluation (`COCODetectionMetric` reads this) |

> **Path note**: `orig_data/mscoco` is `--src-data-dir` (raw extracted data); `heal_data/mscoco` is `--target-data-dir` (packed lmdb for config). Note that `instances_val2017.json` for evaluation must be placed at `heal_data/mscoco/` root (not in `annotations/` subdirectory). The packer only produces lmdb, not json — manual copy required (see packing commands).

#### Official Download

COCO 2017 official page: <https://cocodataset.org/#download>. Download these three archives:

| Archive | Size | Content |
|---|---|---|
| `train2017.zip` | ~18 GB | Training images (118K) |
| `val2017.zip` | ~0.8 GB | Validation images (5K) |
| `annotations_trainval2017.zip` | ~241 MB | Annotations (`instances_train2017.json`, `instances_val2017.json`, etc.) |

Extract all three archives to the raw data directory `./tmp_data/orig_data/mscoco/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/mscoco
unzip train2017.zip -d ./tmp_data/orig_data/mscoco
unzip val2017.zip -d ./tmp_data/orig_data/mscoco
unzip annotations_trainval2017.zip -d ./tmp_data/orig_data/mscoco
```

#### Directory Layout After Extraction

After extraction, `./tmp_data/orig_data/mscoco/` has the following structure:

```
./tmp_data/orig_data/mscoco/
├── train2017/                     # Training images
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ... (~118K images)
├── val2017/                       # Validation images
│   ├── 000000000139.jpg
│   └── ... (~5K images)
└── annotations/                   # Annotations
    ├── instances_train2017.json   # Training detection annotations
    ├── instances_val2017.json     # Validation detection annotations (for evaluation)
    └── ... (captions/person_keypoints and other annotations; not used by this task)
```

#### Packing Commands

Use `tools/datasets/mscoco_packer.py` (`CocoDetectionPacker`) to pack official raw COCO 2017 into lmdb. `--src-data-dir` points to raw data directory `./tmp_data/orig_data/mscoco` (with `train2017/`, `val2017/`, `annotations/`); `--target-data-dir` points to packed output `./tmp_data/heal_data/mscoco` (for config). The packer reads images from `src_data_dir/{split}2017/` and annotations from `src_data_dir/annotations/instances_{split}2017.json`.

```bash
# Pack train split
python3 tools/datasets/mscoco_packer.py \
  --src-data-dir ./tmp_data/orig_data/mscoco \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mscoco \
  --split-name train \
  --num-workers 20

# Pack val split
python3 tools/datasets/mscoco_packer.py \
  --src-data-dir ./tmp_data/orig_data/mscoco \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mscoco \
  --split-name val \
  --num-workers 20
```

The packer only produces `train_lmdb` and `val_lmdb`, **not** `instances_val2017.json`. Copy the evaluation annotation file manually from the raw data directory to the packed output root (config `ann_file` points to `heal_data/mscoco/instances_val2017.json`):

```bash
cp ./tmp_data/orig_data/mscoco/annotations/instances_val2017.json ./tmp_data/heal_data/mscoco/
```

Parameter description:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/mscoco` (with `train2017/`, `val2017/`, `annotations/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/mscoco`, produces `train_lmdb` and `val_lmdb`, matching `common.py` paths `data_path + "train_lmdb/"` and `data_path + "val_lmdb/"`.
- `--split-name`: `train` or `val`, corresponding to COCO 2017 train/val split.
- `--num-workers`: Parallel packing worker count (default 20).

#### Target Layout

After packing and copying annotations, `./tmp_data/heal_data/mscoco/` has the following structure:

```
./tmp_data/heal_data/mscoco/
├── train_lmdb/                 # train lmdb
├── val_lmdb/                   # val lmdb
└── instances_val2017.json      # COCO annotations for evaluation (manually copied from orig_data/mscoco/annotations/)
```

Use the visualization script to sample and verify packing results:

```bash
python3 tools/datasets/mscoco_viz.py \
  --data-path ./tmp_data/heal_data/mscoco/train_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| Training GPUs | Batch size per GPU | Float training epochs | Calibration steps | QAT training epochs | Float optimizer | QAT optimizer |
|---|---|---|---|---|---|---|
| 8 | 2 | 50 | 10 | 4 | AdamW (base lr=1e-4, backbone lr=1e-5, weight_decay=1e-4, norm weight_decay=0), StepDecayLr @ epoch 40, grad clip max_norm=0.1 | AdamW (lr=1e-5, weight_decay=0), StepDecayLr @ epoch 3, grad clip max_norm=0.1 |

**Pretrained weights**: Float training requires loading float backbone weights from ResNet-50 (ImageNet classification) to transfer ResNet-50. Complete Float training for `configs/resnet50` first, and place its `float-checkpoint-best.pth.tar` at `./tmp_pretrained_models/resnet50/float-checkpoint-best.pth.tar`. Loading allows partial parameter mismatch (`allow_miss=True, ignore_extra=True`); only backbone weights are transferred.

```bash
# Float training
python3 tools/train.py \
  --config configs/deform_detr_resnet50 \
  --stage float
# Output:./tmp_models/deform_detr_resnet50/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/deform_detr_resnet50 \
  --stage calibration
# Output:./tmp_models/deform_detr_resnet50/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/deform_detr_resnet50 \
  --stage qat
# Output:./tmp_models/deform_detr_resnet50/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/deform_detr_resnet50/deploy.py
# Output:./tmp_models/deform_detr_resnet50/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/deform_detr_resnet50/deploy.py
# Output:./tmp_models/deform_detr_resnet50/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to re-run training/calibration/qat — only re-run export and compilation on the existing QAT quantized model. Modify the `march` field in `configs/deform_detr_resnet50/common.py` to the target chip:
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
  --config configs/deform_detr_resnet50 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage hbm
```

Evaluation metric is COCO mAP (`COCODetectionMetric`). Annotation file is at `./tmp_data/heal_data/mscoco/instances_val2017.json`; evaluation set is COCO val2017. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPU via `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference frames from the evaluation set (saved to `./demo/deform_detr_resnet50/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference frames (read eval samples from gen_demo_cfg in infer.py, save to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/deform_detr_resnet50

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/deform_detr_resnet50 \
  --stage float
```

Visualization results are saved to `./demo/deform_detr_resnet50/`.
