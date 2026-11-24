# MOTR (EfficientNet-b3)

This document describes the `motr_efficientnetb3` model and how to run the full pipeline from raw MOT17 data through float training → quantization calibration → QAT → HBIR export → HBM compilation → inference.

Model config directory: `configs/motr_efficientnetb3/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| MOTR | multi-frame image sequence `1x3x800x1422` | EfficientNet-b3 | — | tracking boxes + track IDs `(B,Q,cls+reg+id)` |

### Accuracy Metrics

Evaluation set: MOT17 test split (`./tmp_data/heal_data/mot17/test_lmdb` + `test_gt`). Metric: MOTA (multi-object tracking accuracy; higher is better).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | MOTA | 0.5837 | 0.5704 | 0.5799 | 0.5767 |

> Data tested with `march = March.NASH_M` (J6M).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with 8 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | Metric | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|---|
| J6M | main graph | 8.01 | 128.55 | 64.80 |
|  | qim | 0.37 | 5120.87 | 6.80 |
| J6P | main graph | 5.85 | 680.63 | 74.60 |
|  | qim | 0.36 | 10348.73 | 7.00 |
| J6B | main graph | - | - | - |
|  | qim | - | - | - |

J6B performance is not available for this model.

Compiled artifact paths:
- Main graph: `./tmp_models/motr_efficientnetb3/qat.bc` (HBIR), `./tmp_models/motr_efficientnetb3/compile/model.hbm` (HBM)
- QIM subgraph: `./tmp_models/motr_efficientnetb3/qim/qat.bc` (HBIR), `./tmp_models/motr_efficientnetb3/qim/compile/model.hbm` (HBM)

---

## Model Overview

### Core Design

MOTR models tracking as per-frame query propagation: detection queries are passed across frames via `QueryInteractionModule`; new targets get new queries and disappeared targets are cleared; training jointly supervises detection and tracking losses.

- **Task type**: Multi-Object Tracking (MOT).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False` removes classification head, `use_se_block=False`, ReLU activation).
- **neck**: — (no separate neck; backbone features feed directly into `MotrHead`).
- **detection head**: `MotrHead` + `MotrDeformableTransformer` (`d_model=256`, `num_queries=256`, `dim_feedforward=1024`, Deformable Attention, `in_channels=[384]`).
- **tracking module**: `QueryInteractionModule` (cross-frame query interaction for end-to-end tracking association).
- **post-processing**: `MotrPostProcess`.
- **Model input**: multi-frame image sequence, single-frame size `800 × 1422`, sequence sampling interval 10 frames; main graph `img:1x3x800x1422`, QIM subgraph input is query/feat (no images).
- **Model output**: per-frame detection boxes + cross-frame track IDs (256 queries, 1 class).

**Deployment note**: MOTR inference is split into a main graph (backbone + head + post_process) and a QIM subgraph (`QueryInteractionModule`, DDR input, no images), exported and compiled separately via `deploy.py` and `deploy_qim.py`; they work together at inference time (see Usage Guide > Export and Compilation).

### Official Repo and Paper

Official repo: https://github.com/megvii-research/MOTR
Paper: https://arxiv.org/pdf/2105.03247

Note: backbone is EfficientNet-b3; the official repo uses a different backbone.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14093

---

## Usage Guide

### Environment Setup

Complete HEAL environment setup first (see the top-level [`README.md`](../../README.md)). Run the following commands from the **HEAL repository root**:

### Data Preparation

This task uses packed MOT17 lmdb. Data paths read by the model:

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/mot17/train_lmdb` | training sequence lmdb |
| `./tmp_data/heal_data/mot17/test_lmdb` | evaluation sequence lmdb |
| `./tmp_data/heal_data/mot17/test_gt` | evaluation text gt (`motmetrics` reads text format only, not lmdb) |

#### Official Download

MOT17 official page: <https://motchallenge.net/data/MOT17/> (registration required; download `MOT17.zip`, includes train + test, ~5.46 GiB).

Extract `MOT17.zip` to the raw data directory `./tmp_data/orig_data/mot17/` (used as `--src-data-dir` for packing):

```bash
mkdir -p ./tmp_data/orig_data/mot17
unzip MOT17.zip -d ./tmp_data/orig_data/mot17
# Extracts to ./tmp_data/orig_data/mot17/MOT17/{train,test}/...
```

#### Directory Layout After Extraction

After extraction, `./tmp_data/orig_data/mot17/MOT17/` has the following layout:

```
./tmp_data/orig_data/mot17/
└── MOT17/
    ├── train/                              # 7 video sequences × 3 detectors = 21 directories
    │   ├── MOT17-02-DPM/                   #   each sequence has DPM/FRCNN/SDP detector variants
    │   ├── MOT17-02-FRCNN/
    │   ├── MOT17-02-SDP/                   # ← packing uses *SDP variants only
    │   │   ├── det/det.txt                 #   detector results (not used by this task)
    │   │   ├── gt/gt.txt                   #   labels: 9-column CSV = fid,tid,x,y,w,h,mark,label,sss
    │   │   ├── img1/000001.jpg ... 000NNN.jpg  # sequence frames
    │   │   └── seqinfo.ini                 #   seqLength / imWidth / imHeight / imExt
    │   ├── MOT17-04-{DPM,FRCNN,SDP}/
    │   └── ... (MOT17-05/09/10/11/13, 3 variants each)
    └── test/                               # 7 sequences × 3 variants, but official test has no public gt
        ├── MOT17-01-{DPM,FRCNN,SDP}/        #   only {det,img1,seqinfo.ini}, no gt/
        └── ... (MOT17-03/06/07/08/12/14)
```

**Note**: Official `test/` has no public gt and cannot be used for evaluation directly. This task splits `train/*SDP` sequences with gt in half: first half for training, second half for validation (frame IDs renumbered to 1..M). The packing script performs this split in memory; no extra preprocessing is required.

#### Packing Commands

Use `tools/datasets/mot17_packer.py` (`Mot17Packer`) to pack official raw MOT17 into sequence lmdb in one step. `--src-data-dir` points to the raw data directory `./tmp_data/orig_data/mot17/MOT17` (contains `train/`, `test/`); `--target-data-dir` points to the packed output directory `./tmp_data/heal_data/mot17` (read by config). The packing script reads `train/*SDP` directly, performs split-half and label generation in memory, and produces no intermediate files.

```bash
# Pack train split (first half of each train/*SDP sequence)
python3 tools/datasets/mot17_packer.py \
  --src-data-dir ./tmp_data/orig_data/mot17/MOT17 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mot17 \
  --split-name train \
  --num-workers 20

# Pack test split (second half of each train/*SDP sequence, frame IDs renumbered to 1..M)
python3 tools/datasets/mot17_packer.py \
  --src-data-dir ./tmp_data/orig_data/mot17/MOT17 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mot17 \
  --split-name test \
  --num-workers 20
```

Parameter notes:
- `--src-data-dir`: raw data root `./tmp_data/orig_data/mot17/MOT17` (contains `train/`, `test/`). Packing uses `train/*SDP` only.
- `--target-data-dir`: packed output directory `./tmp_data/heal_data/mot17`, produces `train_lmdb` and `test_lmdb`, matching `common.py` `train_lmdb`/`val_lmdb` paths.
- With `--split-name test`, the script also writes each sequence's second-half `gt.txt` (frame IDs renumbered to 1..M) to `./tmp_data/heal_data/mot17/test_gt/<seq>/gt/gt.txt` for evaluation. This directory is auto-generated; no manual organization required.
- `--num-workers`: parallel packing worker count (default 20).

#### Target Layout

After packing, `./tmp_data/heal_data/mot17/` has the following layout:

```
./tmp_data/heal_data/mot17/
├── train_lmdb/        # training sequence lmdb
├── test_lmdb/         # evaluation sequence lmdb
└── test_gt/           # evaluation text gt (auto-generated by script)
    └── <seq>/gt/gt.txt   # frame IDs renumbered to 1..M
```

Use the visualization script to spot-check packing correctness:

```bash
python3 tools/datasets/mot17_viz.py \
  --data-path ./tmp_data/heal_data/mot17/train_lmdb \
  --viz-num 50
```

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Per-GPU Batch Size | Float Training Epochs | Calibration Steps | QAT Training Epochs | Float Optimizer | QAT Optimizer |
|---|---|---|---|---|---|---|
| 8 | 1 | 250 | 500 | 20 | AdamW (backbone/sampling_offsets lr=2e-5, base lr=2e-4, wd=1e-4) | AdamW (lr=1e-5, wd=1e-4) |

**Pretrained weights**: Float training requires FCOS (MSCOCO) float backbone weights to transfer EfficientNet-b3. Complete Float training for `configs/fcos_efficientnetb3` first and place its `float-checkpoint-best.pth.tar` at:

```
./tmp_pretrained_models/fcos_efficientnetb3/float-checkpoint-best.pth.tar
```

```bash
# Float training
python3 tools/train.py \
  --config configs/motr_efficientnetb3 \
  --stage float
# Output: ./tmp_models/motr_efficientnetb3/float-checkpoint-best.pth.tar

# Quantization calibration (Calibration)
python3 tools/train.py \
  --config configs/motr_efficientnetb3 \
  --stage calibration
# Output: ./tmp_models/motr_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/motr_efficientnetb3 \
  --stage qat
# Output: ./tmp_models/motr_efficientnetb3/qat-checkpoint-best.pth.tar
```

### Export and Compilation

MOTR has main graph and QIM subgraph export configs; run HBIR export and HBM compilation for each:

```bash
# Export HBIR
# Main graph (backbone + head + post_process)
python3 tools/export_hbir.py \
  --config configs/motr_efficientnetb3/deploy.py
# QIM subgraph (QueryInteractionModule, DDR input, no images)
python3 tools/export_hbir.py \
  --config configs/motr_efficientnetb3/deploy_qim.py
# Output: main graph ./tmp_models/motr_efficientnetb3/qat.bc; QIM subgraph ./tmp_models/motr_efficientnetb3/qim/qat.bc

# Compile HBM
# Main graph
python3 tools/compile_perf.py \
  --config configs/motr_efficientnetb3/deploy.py
# QIM subgraph
python3 tools/compile_perf.py \
  --config configs/motr_efficientnetb3/deploy_qim.py
# Output: main graph ./tmp_models/motr_efficientnetb3/compile/model.hbm; QIM subgraph ./tmp_models/motr_efficientnetb3/qim/compile/model.hbm
```

> **Switching chips (march) tips**: Switching chips only affects HBIR export and HBM compilation; it does **not** affect prior float training and quantization (calibration/qat). When switching J6M/J6P/J6B, you do not need to rerun training/calibration/qat—re-run export and compilation on the existing QAT quantized model. Set the `march` field in `configs/motr_efficientnetb3/common.py` to the target chip:
> ```python
> march = March.NASH_M   # J6M
> march = March.NASH_P   # J6P
> march = March.NASH_B   # J6B
> ```
> Then re-run `export_hbir.py` and `compile_perf.py` for **both main graph and QIM subgraph** (the four commands above). Compiled artifacts differ per chip; performance is not directly portable across chips.

### Accuracy Evaluation

```bash
# Float evaluation
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation first)
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation, plus board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage hbm
```

Evaluation metric is MOTA; gt is at `./tmp_data/heal_data/mot17/test_gt`. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`). Override evaluation GPUs with `--device-ids`.

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference frames from the eval set (saved to `./demo/motr_efficientnetb3/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference frames (reads eval samples from gen_demo_cfg in infer.py, saved to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/motr_efficientnetb3

# 2. Inference and visualization (float example; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/motr_efficientnetb3 \
  --stage float
```

Visualization output saved to `./demo/motr_efficientnetb3/`.
