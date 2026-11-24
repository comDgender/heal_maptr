# ResNet-50

ResNet-50 consists of 4 stages of residual blocks; each stage downsamples via stride=2, then global average pooling followed by a fully connected layer outputs class probabilities.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| ResNet-50 | `1x3x224x224` | ResNet-50 | — | Classification logits `(B,1000)` |

### Accuracy Metrics

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
