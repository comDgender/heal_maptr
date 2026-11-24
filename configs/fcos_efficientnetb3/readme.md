# FCOS (EfficientNet-b3)

This document introduces the `fcos_efficientnetb3` model and explains how to complete the full pipeline from the original COCO 2017 dataset: float training → quantization calibration → QAT → HBIR export → HBM compilation → inference.

Model config directory: `configs/fcos_efficientnetb3/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| FCOS | Single image `1x3x896x896` | EfficientNet-b3 | `BiFPN` | Detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

Evaluation set: COCO val2017 (`./tmp_data/heal_data/mscoco/val_lmdb` + `instances_val2017.json`). Metric: COCO mAP.

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | COCO mAP | 0.4802 | 0.4634 | 0.4773 | 0.4772 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 3.89 | 276.72 | 30.80 |
| J6P | 3.00 | 1439.73 | 34.80 |
| J6B | - | - | - |

J6B performance is not available for this model.

Compilation artifact paths:
- HBIR: `./tmp_models/fcos_efficientnetb3/qat.bc`
- HBM: `./tmp_models/fcos_efficientnetb3/compile/model.hbm`

---

## Model Overview

### Core Design

FCOS eliminates anchors and predicts classification scores, centerness, and distances to box edges per pixel; `BiFPN` performs bidirectional feature fusion across 5 scales to improve small-object detection; training jointly supervises classification, centerness, and regression losses.

- **Task type**: 2D object detection (2D Object Detection).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False` removes classification head, `use_se_block=False`, ReLU activation).
- **neck**: `BiFPN` (`out_channels=120`, `stack=6`, input stride `[2,4,8,16,32]` → output stride `[8,16,32,64,128]`, `bifpn_sum` fusion).
- **Detection head**: `FCOSHead` (`feat_channels=120`, `stacked_convs=4`, 5 feature layers stride `[8,16,32,64,128]`, `dequant_output=True`).
- **Target assignment**: `DynamicFcosTarget` (`topK=10`, classification `FocalLoss`, regression `GIoULoss`).
- **Post-processing**: `FCOSDecoder` (`score_thr=0.05`, `nms_pre=1000`, NMS `iou_threshold=0.6`, `max_per_img=100`, uses centerness).
- **Loss**: Classification `FocalLoss` (`alpha=0.25, gamma=2.0`) + centerness `CrossEntropyLoss` (sigmoid) + regression `GIoULoss`.
- **Model input**: Single image, size `896 × 896`.
- **Model output**: 80-class detection boxes + confidence scores.

### Official Repo and Paper

Official repo: https://github.com/tianzhi0549/FCOS
Paper: https://arxiv.org/abs/1904.01355

Note: backbone is EfficientNet-b3; official repo uses a different backbone.

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

The packer only produces `train_lmdb` and `val_lmdb`, **not** `instances_val2017.json`. Copy the evaluation annotation file manually from the raw data directory to the packed output root (config `ann_file_val` points to `heal_data/mscoco/instances_val2017.json`):

```bash
cp ./tmp_data/orig_data/mscoco/annotations/instances_val2017.json ./tmp_data/heal_data/mscoco/
```

Parameter description:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/mscoco` (with `train2017/`, `val2017/`, `annotations/`).
- `--target-data-dir`: Packed output directory `./tmp_data/heal_data/mscoco`, produces `train_lmdb` and `val_lmdb`, matching `data.py` paths `data_path + "train_lmdb/"` and `data_path + "val_lmdb/"`.
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
| 8 | 12 | 600 | 100 | 30 | SGD (lr=0.2, momentum=0.937, nesterov, weight_decay=4e-5), CosLr warmup 2 epoch, EMA | SGD (lr=1e-4, momentum=0.9, weight_decay=4e-5), CosLr warmup 0.1 epoch |

> Float training for this task starts from scratch; no pretrained weights required.

```bash
# Float training
python3 tools/train.py \
  --config configs/fcos_efficientnetb3 \
  --stage float
# Output:./tmp_models/fcos_efficientnetb3/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/fcos_efficientnetb3 \
  --stage calibration
# Output:./tmp_models/fcos_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/fcos_efficientnetb3 \
  --stage qat
# Output:./tmp_models/fcos_efficientnetb3/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/fcos_efficientnetb3/deploy.py
# Output:./tmp_models/fcos_efficientnetb3/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/fcos_efficientnetb3/deploy.py
# Output:./tmp_models/fcos_efficientnetb3/compile/model.hbm
```

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to re-run training/calibration/qat — only re-run export and compilation on the existing QAT quantized model. Modify the `march` field in `configs/fcos_efficientnetb3/common.py` to the target chip:
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
  --config configs/fcos_efficientnetb3 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage hbm
```

Evaluation metric is COCO mAP (`COCODetectionMetric`). Annotation file is at `./tmp_data/heal_data/mscoco/instances_val2017.json`; evaluation set is COCO val2017. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPU via `--device-ids` if needed.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference frames from the evaluation set (saved to `./demo/fcos_efficientnetb3/`), then `inference.py` reads that directory for inference and visualization.

```bash
# 1. Generate inference frames (read eval samples from gen_demo_cfg in infer.py, save to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/fcos_efficientnetb3

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/fcos_efficientnetb3 \
  --stage float
```

Visualization results are saved to `./demo/fcos_efficientnetb3/`.
