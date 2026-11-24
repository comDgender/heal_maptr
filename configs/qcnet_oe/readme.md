# QCNetOE (Trajectory Prediction)

This task is a multimodal trajectory prediction task on the Argoverse 2 motion forecasting dataset, using the QCNetOE model to predict future trajectories for agents in a scene. This document describes the full pipeline from raw Argoverse 2 data through float training → quantization calibration → QAT → HBIR export → HBM compilation → inference.

Model config directory: `configs/qcnet_oe/`

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| QCNetOE | Streaming scene representation tensor group `(B,A,pl,pt,HT)` | QCNetOEMapEncoder + QCNetOEAgentEncoderStream | — | Candidate future trajectories `(B,A,6,12,2)` + trajectory probabilities |

### Accuracy Metrics

Evaluation set: Argoverse 2 val (`./tmp_data/heal_data/argoverse2/val_lmdb`). Metric: HitRate (trajectory prediction hit rate; higher is better; this task focuses on HitRate for deployment; Brier/minADE/minFDE/MR metrics are also reported by the evaluation script).

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | HitRate | 0.8003 | 0.6817 | 0.7984 | 0.7981 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 3.63 | 300.69 | 32.10 |
| J6P | 2.67 | 1593.50 | 36.50 |
| J6B | 10.07 | 177.52 | 29.60 |

Compiled artifact paths:
- HBIR: `./tmp_models/qcnet_oe/qat.bc`
- HBM: `./tmp_models/qcnet_oe/compile/model.hbm`

---

## Model Overview

### Core Design

QCNetOE encodes each agent's relative relationships with surrounding map and other agents in a query-centric manner, streaming encoder hidden states agent-by-agent, then the decoder outputs multimodal candidate trajectories and probabilities; removes `torch_geometric`/`torch_cluster` dependencies and eliminates most index/gather/scatter ops for quantization-friendly deployment.

- **Task type**: Trajectory prediction (Motion Forecasting, multimodal trajectory prediction).
- **backbone**: `QCNetOEMapEncoder` + `QCNetOEAgentEncoderStream` (streaming inference, `stream_infer=True`; `hidden_dim=128`, `num_heads=8`, `head_dim=16`, `num_freq_bands=32`, `num_map_layers=1`, `num_agent_layers=1`, `time_span=2`, `dropout=0.1`; agent-map interaction `num_pl2a=32`, agent-agent interaction `num_a2a=36`).
- **neck**: — (QCNetOE has no standalone neck; encoder output feeds directly into decoder).
- **Decoder**: `QCNetOEDecoder` (`num_dec_layers=1`), outputs `num_modes=6` candidate trajectories and their probabilities.
- **Preprocessing**: `QCNetOEPreprocess` (`stream=True`, constructs agent/map relative representations and spatiotemporal relative position encoding).
- **Post-processing**: `QCNetOEPostprocess` (output dimension `output_dim=2`, i.e. predicted trajectory (x, y)).
- **Loss**: `QCNetOELoss`.
- **Model input**: Streaming scene representation tensor group (`B=1, A=30` agents, `pl=80` map polygons, `pt=50` polygon points, `HT=10` history steps), including `agent`/`map_polygon`/`map_point`/`decoder` etc. as OrderedDict inputs; 5s history (`num_historical_steps=10`) + 6s prediction (`num_future_steps=12`).
- **Model output**: 6 candidate future trajectories per agent to predict (`num_modes=6`, 12 steps, 2-dim coordinates `output_dim=2`) + probability per trajectory.

**Deployment notes**: Model supports cold-start/hot-start streaming inference (`quant_infer_cold_start` controls). Both `cali_model` and `deploy_model` enable `stream_infer=True`; `save_memory` configured separately for training/deployment (training `save_memory=True` saves GPU memory, deployment `save_memory=False`). HBIR export enables `enable_vpu=True`; compilation uses `input_source=["ddr"]` (trajectory prediction input read from DDR, not pyramid image input), unlike image-based tasks.

### Official Repo and Paper

Official repo: https://github.com/ZikangZhou/QCNet
Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14089

---

## Usage Guide

### Environment Setup

Complete HEAL environment installation first (see top-level [`README.md`](../../README.md)). All commands below are run from the **HEAL repository root**:

### Data Preparation

This task uses Argoverse 2 packed lmdb. Data paths read by the model (all joined from `common.py` `data_rootdir = "./tmp_data/heal_data/argoverse2/"`):

| Path | Purpose |
|---|---|
| `./tmp_data/heal_data/argoverse2/train_lmdb` | Training lmdb (`data.py` `data_loader` / `qat_data_loader`) |
| `./tmp_data/heal_data/argoverse2/val_lmdb` | Evaluation lmdb (`data.py` `val_data_loader` / `int_infer_data_loader`) |

> **Directory convention**: `orig_data/argoverse2` is `--src-data-dir` (raw extracted data); `heal_data/argoverse2` is `--target-data-dir` (packed lmdb for config). Argoverse 2 packer reads official extracted parquet + map json directly; **no extra preprocessing scripts needed**; one packing step suffices.

#### Official Download

Argoverse 2 Motion Forecasting dataset official page: <https://www.argoverse.org/av2.html#forecasting-link>. Download the motion forecasting subset (includes `train/` and `val/` splits, each split contains multiple `log` directories).

> Official download links and post-extraction directory layout follow the official page; if the official distribution method changes, extract to the layout below per official instructions.

#### Directory Layout After Extraction

Extract downloaded Argoverse 2 motion forecasting data to raw data directory `./tmp_data/orig_data/argoverse2/` (this directory is `--src-data-dir`). Each `log` directory contains one scene parquet (trajectories/annotations) and map json:

```
./tmp_data/orig_data/argoverse2/
├── train/
│   └── <log_id>/                         # One directory per scene
│       ├── scenario_<log_id>.parquet      # Scene trajectories and annotations
│       └── log_map_archive_<log_id>.json  # Vectorized map for this scene
└── val/
    └── <log_id>/
        ├── scenario_<log_id>.parquet
        └── log_map_archive_<log_id>.json
```

> `train/` and `val/` each contain tens of thousands of `<log_id>` subdirectories; `<log_id>` is a unique scene identifier. The packer iterates all subdirectories under `--src-data-dir/{split}/`, reads parquet + map json per scene, and extracts agent/map features.

#### Packing Commands

Use `tools/datasets/argoverse2_packer.py` (`Argoverse2Packer`) to pack official raw scenes into lmdb. Argoverse 2 requires no preprocessing scripts; call the packer directly. Pack train and val splits once each:

```bash
# Pack train lmdb
python3 tools/datasets/argoverse2_packer.py \
  --src-data-dir ./tmp_data/orig_data/argoverse2 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/argoverse2 \
  --split-name train \
  --num-workers 10

# Pack val lmdb
python3 tools/datasets/argoverse2_packer.py \
  --src-data-dir ./tmp_data/orig_data/argoverse2 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/argoverse2 \
  --split-name val \
  --num-workers 10
```

Parameter notes:
- `--src-data-dir`: Raw data root `./tmp_data/orig_data/argoverse2` (contains `train/` and `val/` subdirectories, each with multiple `<log_id>` scene directories).
- `--target-data-dir`: Pack output directory `./tmp_data/heal_data/argoverse2`; packer produces `{split}_lmdb`, matching `common.py` `data_rootdir`.
- `--pack-type`: Pack type, always `lmdb`.
- `--split-name`: Split to pack (`train` or `val`); packer outputs to `{target}/{split}_lmdb`.
- `--num-workers`: Parallel packing worker count (default 10).

#### Target Layout

After two packing runs, `./tmp_data/heal_data/argoverse2/` layout:

```
./tmp_data/heal_data/argoverse2/
├── train_lmdb/      # train lmdb
└── val_lmdb/        # val lmdb
```

After packing, spot-check lmdb with visualization. Argoverse 2 uses `tools/datasets/argoverse2_viz.py` (pack validation, reuses `heal/visualize/argoverse2.py` `Argoverse2Viz`, draws packed map + history/future GT trajectories, no model needed):

```bash
python3 tools/datasets/argoverse2_viz.py \
  --data-path ./tmp_data/heal_data/argoverse2/val_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/argoverse2
```

Visualization PNGs saved to `--save-dir` (must be a local writable directory; `tmp_data/heal_data` is a read-only bucket and cannot be used as save target); one subdirectory per sample `sample_{i}/traj_pred.png`; manually verify history/future trajectories align with packed map. Defaults `--num-historical-steps=10`, `--num-future-steps=12`, `--input-dim=2`, `--split=val`, matching this task's config; override via corresponding flags if needed.

> Inference visualization (predicted trajectories overlaid on map and history trajectories) is done by `QCNetViz` built into `infer.py` (wraps `heal/visualize/argoverse2.py` `Argoverse2Viz`); see "Accuracy Evaluation" section.

### Training and Quantization

Key training hyperparameters:

| # Training GPUs | Batch Size per GPU | Float Training Epochs | Calibration Steps | QAT Training Epochs | Float Optimizer | QAT Optimizer | LR Schedule | Float SyncBN |
|---|---|---|---|---|---|---|---|---|
| 8 (`device_ids=[0,1,2,3,4,5,6,7]`) | 16 (`batch_size_per_gpu = 4 * 4`) | 75 | 100 | 15 | AdamW (`lr=5e-4`, `weight_decay=1e-4`; zero weight decay for bias/norm/Embedding) | AdamW (`lr=8*5e-6`, `weight_decay=1e-4`; zero weight decay for bias/norm/Embedding) | Float: `CosLrUpdater` (`warmup_len=1`, `warmup_by="epoch"`)<br>QAT: `StepDecayLrUpdater` (`lr_decay_id=[4]`) | Enabled (`sync_bn=True`) |

> This task starts Float training from scratch with **no LoadCheckpoint** and no external pretrained weights. Calibration loads this task's Float artifact; QAT loads this task's Calibration artifact (serial chain, no cross-task dependency).

```bash
# Float training
python3 tools/train.py \
  --config configs/qcnet_oe \
  --stage float
# Output:./tmp_models/qcnet_oe/float-checkpoint-best.pth.tar

# Quantization calibration
python3 tools/train.py \
  --config configs/qcnet_oe \
  --stage calibration
# Output:./tmp_models/qcnet_oe/calibration-checkpoint-best.pth.tar

# QAT training
python3 tools/train.py \
  --config configs/qcnet_oe \
  --stage qat
# Output:./tmp_models/qcnet_oe/qat-checkpoint-best.pth.tar
```

### Export and Compilation

```bash
# Export HBIR
python3 tools/export_hbir.py \
  --config configs/qcnet_oe/deploy.py
# Output:./tmp_models/qcnet_oe/qat.bc

# Compile HBM
python3 tools/compile_perf.py \
  --config configs/qcnet_oe/deploy.py
# Output:./tmp_models/qcnet_oe/compile/model.hbm
```

> Export uses `enable_vpu=True`; compilation input `input_source=["ddr"]` (trajectory prediction input read from DDR), unlike pyramid input for image tasks.

> **Switching chip (march) tips**: Switching chips only affects HBIR export and HBM compilation; **it does not affect prior float training and quantization (calibration/qat)**. When switching J6M/J6P/J6B, no need to rerun training/calibration/qat — only re-export and re-compile on the existing QAT quantized model. Edit the `march` field in `configs/qcnet_oe/common.py` to the target chip:
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
  --config configs/qcnet_oe \
  --stage float

# Calibration evaluation
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage calibration

# QAT evaluation
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage qat

# HBIR evaluation (requires export from Export and Compilation section first)
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage hbir

# HBM evaluation (requires compilation from Export and Compilation section, and board/hbm_infer runtime)
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage hbm
```

Evaluation set is Argoverse 2 val (`./tmp_data/heal_data/argoverse2/val_lmdb`); deployment metric focuses on `HitRate` (hit rate; higher is better); checkpoint selection uses `HitRate` as monitoring metric (`mode="max"`); evaluation script also outputs auxiliary metrics `BrierMetric`/`MinADE`/`MinFDE`/`MissRate`, but deployment metrics table only reports `HitRate`. Evaluation commands follow config `device_ids` by default; calibration training and some hbir/hbm evaluation stages run single-process per config (see `data.py`); override evaluation GPU with `--device-ids` if needed.

> HBM stage (`hbm_predictor`) wraps on-board `.hbm` model via `QCNetOEHbmInfer`; requires board and `hbm_infer` runtime (`predict.py` `HbmModule` `host` points to board address).

**Inference and visualization**: Before inference, run `gen_demo.py` to generate inference scene data from the evaluation set (saved to `./demo/qcnet_oe/`), then run `inference.py` on that directory.

```bash
# 1. Generate inference scenarios (read eval samples from gen_demo_cfg in infer.py, save to ./demo/<task>/)
python3 tools/gen_demo.py \
  --config configs/qcnet_oe

# 2. Inference and visualization (float shown; replace --stage with calibration/qat/hbir/hbm)
python3 tools/inference.py \
  --config configs/qcnet_oe \
  --stage float
```

Visualization is done by `QCNetViz` built into `infer.py` (wraps `Argoverse2Viz`), overlaying predicted trajectories on history trajectories and map; results saved to `./tmp_viz_results/qcnet_oe_float/` (each stage has corresponding `_calibration`/`_qat`/`_hbir`/`_hbm` subdirectory).

#### Quantization Sensitivity Analysis (Optional)

This task's `predict.py` defines `quant_analysis_solver` for per-layer quantization impact analysis to help diagnose accuracy drops:

```bash
python3 tools/quant_analysis.py \
  --config configs/qcnet_oe/predict.py
```

Results output to `./tmp_models/qcnet_oe/analysis/`.
