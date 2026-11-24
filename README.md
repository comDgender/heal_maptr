# HEAL

HEAL (Horizon Embedded AI Library) is a deep learning training and deployment framework for Horizon chips. Built on top of the [`hat_engine`](hat_engine/) training engine, it provides task-specific models, datasets, evaluation metrics, and visualization utilities, covering the full deployment pipeline from float training and quantization calibration through QAT, HBIR export, and HBM compilation.

Current version: **0.14.0** (see [`heal/version.py`](heal/version.py))

## Features

- **Multi-task support**: image classification, semantic segmentation, 2D/3D object detection, lane detection, tracking, BEV perception, occupancy prediction, trajectory forecasting, and more
- **Stage-based training and inference**: switch between float / calibration / qat / hbir / hbm stages via `--stage`
- **Config-driven workflow**: each task is organized as Python configs under `configs/<task_name>/`, with `--opts` for CLI overrides
- **Horizon quantization and deployment**: integrates `horizon_plugin_pytorch` with HBIR export and HBM compilation
- **Testable**: unit tests plus end-to-end smoke tests covering major task pipelines

## Repository Layout

```
HEAL/
├── heal/              # Task models, datasets, transforms, metrics, visualization
├── hat_engine/        # Training engine, registry, deployment utilities
├── configs/           # Per-task train / predict / infer / deploy configs
├── tools/             # CLI entry points (train, predict, inference, etc.)
├── tests/             # unit_tests and smoke tests
├── requirements/      # Split dependencies (build / develop / optional)
└── .pre-commit-config.yaml
```

### Main `heal` Packages

| Module | Description |
|--------|-------------|
| [`heal/models`](heal/models/) | Backbones (HENet, EfficientNet, MixVarGENet, etc.), task heads, and full structures (Classifier, FCOS, PointPillars, BEV, Segmentor, QCNet, FlashOcc, etc.) |
| [`heal/data`](heal/data/) | Datasets (ImageNet, COCO, Cityscapes, KITTI, nuScenes, Argoverse2, etc.), transforms, and collates |
| [`heal/metrics`](heal/metrics/) | Evaluation metrics for COCO detection, KITTI 3D, nuScenes, Argoverse2, and more |
| [`heal/visualize`](heal/visualize/) | Visualization for classification, detection, segmentation, point clouds, BEV, maps, etc. |
| [`heal/utils`](heal/utils/) | Config loading, example-input refresh, and other helpers |

Importing `heal` automatically registers default configs with `hat_engine`:

```python
import heal  # triggers register_default_config
```

## Requirements

- Python 3.10 (recommended)
- CUDA GPU (for training and inference)
- PyTorch 2.8 + CUDA 12.8
- `horizon_plugin_pytorch` 3.3.10 (Horizon quantization plugin)
- `hbdk4-compiler` 4.11.11 (HBIR compilation)

## Installation

Add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

Install Python dependencies from the Horizon PyPI mirrors:

```bash
PIP_EXT="-i https://pypi.hobot.cc/simple \
  --extra-index-url=https://pypi.hobot.cc/hobot-local/simple \
  --trusted-host pypi.hobot.cc"

pip3 install -r requirements.txt $PIP_EXT
```

Also install the matching PyTorch and Horizon packages (CUDA 12.8 + Torch 2.8):

```bash
TORCH_INDEX="-i https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple"
HORIZON_F="-f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/cu128/torch280 \
  --trusted-host art-internal.hobot.cc"

pip3 install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  $TORCH_INDEX $PIP_EXT
pip3 install hbdk4-compiler==4.9.8 $PIP_EXT
pip3 install --pre horizon-plugin-pytorch $HORIZON_F $PIP_EXT
```

Optional lint / test extras are in [`requirements/develop.txt`](requirements/develop.txt). After installing them:

```bash
pre-commit install
pre-commit run --all-files
```

### Pretrained models and data

Pretrained checkpoints and related assets are published on Hugging Face under [OpenExplorer](https://huggingface.co/OpenExplorer). For example:

- [`OpenExplorer/bev_multitask_henet_tinym`](https://huggingface.co/OpenExplorer/bev_multitask_henet_tinym)

Download into the local layout expected by configs (`./tmp_models/<task_name>/`):

```bash
huggingface-cli download OpenExplorer/bev_multitask_henet_tinym \
  --local-dir ./tmp_models/bev_multitask_henet_tinym
```

Configs read packed datasets from `./tmp_data/heal_data/...`. Prepare datasets following each task's `readme.md`, or link your packed data:

```bash
ln -snf <path_to_pack_data> ./tmp_data
ln -snf <path_to_orig_data> ./tmp_orig_data
```

`tmp_orig_data` is only needed for map visualization overlays.

## Built-in Task Configs

Each directory under `configs/` is one task. Typical files:

| File | Purpose |
|------|---------|
| `common.py` | Shared variables: task name, checkpoint path, devices, march, etc. |
| `model.py` | Model architecture |
| `data.py` | Data loading and batch processors |
| `qconfig.py` | Quantization config and model convert pipelines |
| `train.py` | `{stage}_trainer` (e.g. `float_trainer`, `calibration_trainer`) |
| `predict.py` | `{stage}_predictor` |
| `infer.py` | `{stage}_infer` and visualization (`float_infer`, `hbir_infer`, …) |
| `deploy.py` | `hbir_exporter`, `hbir_compiler` |
| `callback.py` | Training callbacks |

| Config directory | Task | Dataset |
|------------------|------|---------|
| `henet_tinye` / `henet_tinym` | Image classification | ImageNet |
| `resnet50` | Image classification | ImageNet |
| `efficientnet` | Image classification | ImageNet |
| `mixvargenet` | Image classification | ImageNet |
| `unet_mobilenetv1` | Semantic segmentation | Cityscapes |
| `fcos_efficientnetb3` | 2D detection | MS COCO |
| `deform_detr_resnet50` | 2D detection | MS COCO |
| `ganet_mixvargenet` | Lane detection | CULane |
| `motr_efficientnetb3` | Multi-object tracking | MOT17 |
| `pointpillars` | 3D LiDAR detection | KITTI |
| `centerpoint_pointpillar` | 3D LiDAR detection | nuScenes |
| `fcos3d_efficientnetb0` | Camera 3D detection | nuScenes |
| `petr_efficientnetb3` | Camera 3D detection | nuScenes |
| `detr3d_efficientnetb3` | Camera 3D detection | nuScenes |
| `bev_vtv2_resnet50` | BEV perception | nuScenes |
| `bev_sparse_henet_tinym` | Sparse BEV detection | nuScenes |
| `bev_sparse_lidar_fusion_henet_tinym` | Camera–LiDAR fusion BEV | nuScenes |
| `bev_multitask_henet_tinym` | Camera BEV multitask | nuScenes |
| `bevformer_tiny_resnet50_detection` | BEVFormer detection | nuScenes |
| `bevformer_henet_camera_multitask` | BEVFormer multi-task | nuScenes |
| `bevfusion_pointpillar_henet_multisensor_multitask` | Multi-sensor multi-task | nuScenes |
| `maptr_henet_tinym_bevformer` | Online HD map | nuScenes |
| `maptr_sparse_henet_tinym` | Sparse MapTR | nuScenes |
| `flashocc_henet_lss_occ3d` | Occupancy | nuScenes |
| `sparse_multitask_det_maptr_flashocc_henet_tinym` | Det + map + occ | nuScenes |
| `qcnet_oe` | Trajectory forecasting | Argoverse2 |

## Quick Start

The examples below use `henet_tinye`. Pass a config directory to `-c`; tools resolve the default file for each stage (`train.py`, `predict.py`, `infer.py`, `deploy.py`).

### 1. Float training

```bash
python3 tools/train.py \
  --config configs/henet_tinye \
  --stage float \
  --device-ids 0
```

### 2. Quantization calibration

```bash
python3 tools/train.py \
  --config configs/henet_tinye \
  --stage calibration \
  --device-ids 0
```

### 3. Prediction and evaluation

```bash
python3 tools/predict.py \
  --config configs/henet_tinye \
  --stage float \
  --device-ids 0
```

### 4. Demo sample, inference, and visualization

Generate a demo sample, then run float inference and save one visualization image:

```bash
python3 tools/gen_demo.py \
  --config configs/henet_tinye

python3 tools/inference.py \
  --config configs/henet_tinye \
  --stage float \
  --device-ids 0 \
  --max-samples 1 \
  --save-path ./viz_out
```

`--save-path` overrides the visualizer output directory. `--max-samples` limits how many demo samples are run. Requires `gen_demo_cfg` in `infer.py` and a float checkpoint under `tmp_models/<task>/`.

### 5. Export HBIR and compile HBM

```bash
python3 tools/export_hbir.py \
  --config configs/henet_tinye/deploy.py

python3 tools/compile_perf.py \
  --config configs/henet_tinye/deploy.py
```

### 6. Quantization sensitivity analysis

```bash
python3 tools/quant_analysis.py \
  -c configs/qcnet_oe
```

Requires `quant_analysis_solver` to be defined in the task `quant_analysis.py`.

## CLI Tools

| Tool | Description |
|------|-------------|
| [`tools/train.py`](tools/train.py) | Stage-based training (float / calibration / qat) |
| [`tools/predict.py`](tools/predict.py) | Stage-based prediction and evaluation on the validation set |
| [`tools/inference.py`](tools/inference.py) | Inference with visualization (`--stage`, `--save-path`, `--max-samples`) |
| [`tools/gen_demo.py`](tools/gen_demo.py) | Export demo inputs from a dataset |
| [`tools/export_hbir.py`](tools/export_hbir.py) | Export HBIR from a QAT / calibration model |
| [`tools/compile_perf.py`](tools/compile_perf.py) | Compile HBIR to HBM |
| [`tools/quant_analysis.py`](tools/quant_analysis.py) | Per-layer quantization sensitivity analysis |
| [`tools/merge_checkpoints.py`](tools/merge_checkpoints.py) | Merge checkpoints |
| [`tools/datasets/`](tools/datasets/) | Dataset pack / visualization scripts |

### Common arguments

- `--config` / `-c`: config file or config directory
- `--stage` / `-s`: stage name, maps to `{stage}_trainer` / `{stage}_predictor` / `{stage}_infer` in the config
- `--device-ids` / `-ids`: GPU IDs, e.g. `0`
- `--opts`: override config fields, e.g. `--opts float_trainer.num_epochs 10`
- `--pipeline-test`: sets `HAT_PIPELINE_TEST=1` to shorten smoke tests
- `--launcher torch`: multi-node distributed launch (via `hat_engine`)

## Typical Deployment Pipeline

```
Float training → Calibration → QAT (optional) → Predict / validate
    → Export HBIR → Compile HBM → HBIR / HBM inference
```

The `HAT_TRAINING_STEP` environment variable selects the current quantization stage in configs (default: `float`).

## Testing

### Smoke tests

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
pytest tests/smoke/henet_tinye -s -x
```

Full suite (needs GPUs and packed datasets):

```bash
pytest tests/smoke -s
```

### Unit tests

```bash
pytest tests/unit_tests -v
```

## Development Notes

- **Config convention**: each stage defines `{stage}_trainer`, `{stage}_predictor`, or `{stage}_infer` in the corresponding config file
- **Registry**: models, datasets, metrics, etc. are registered via `hat_engine.registry` and become available after `import heal`

## Changelog

### 0.0.1 (2026-06-12)

- Initial public release

## License

Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.
