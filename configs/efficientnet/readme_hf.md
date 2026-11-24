# EfficientNet-B0

EfficientNet-B0 consists of MBConv (mobile inverted bottleneck) convolution blocks, scaled jointly in depth/width/resolution via compound scaling; this config uses ReLU activation and disables SE Block (better suited for BPU quantization deployment).

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| EfficientNet-B0 | `1x3x224x224` | EfficientNet-B0 | — | Classification logits `(B,1000)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Accuracy | 0.7491 | 0.7433 | — | 0.7436 |
|  | TopKAccuracy(5) | — | — | — | — |

> Results are based on `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is `—`).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.40 | 4753.34 | 8.90 |
| J6P | 0.37 | 10490.35 | 9.00 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

EfficientNet-B0 consists of MBConv (mobile inverted bottleneck) convolution blocks, scaled jointly in depth/width/resolution via compound scaling; this config uses ReLU activation and disables SE Block (better suited for BPU quantization deployment).

- **Task type**: Image classification (Image Classification).
- **backbone**: EfficientNet-B0 (`efficientnet`, `model_type="b0"`, `activation="relu"`, `use_se_block=False`, `num_classes=1000`), composed of MBConv blocks scaled via compound scaling in depth/width/resolution; this config uses ReLU activation and disables SE Block (better suited for BPU quantization deployment).
- **neck**: — (EfficientNet-B0 has built-in fully-connected classification head; no separate neck).
- **Classification head**: EfficientNet-B0 built-in fully-connected classification head, directly outputs 1000-class logits.
- **Loss**: `CEWithLabelSmooth` (cross-entropy with label smoothing).
- **Model input**: Single RGB image, resolution `224 × 224` (`1x3x224x224`).
- **Model output**: 1000-class prediction logits; argmax gives predicted class.

### Official Repo and Paper

Official repo: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
Paper: https://arxiv.org/abs/1905.11946
