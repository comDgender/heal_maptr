# HENet-tinyE

HENet uses depthwise conv to build lightweight blocks with LayerScale for stable training; the tiny-E configuration has widths 48/96/192/384 and depths 3/3/8/6, with parameter count in the tiny range.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| HENet-tinye | `1x3x224x224` | HENet | — | classification logits `(B,1000)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Accuracy | 0.7762 | 0.7693 | — | 0.7719 |
|  | TopKAccuracy(5) | 0.9372 | — | — | — |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with 8 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.50 | 3658.88 | 12.70 |
| J6P | 0.42 | 11396.93 | 12.90 |
| J6B | 0.89 | 1969.62 | 10.10 |

---

## Model Overview

### Core Design

HENet uses depthwise conv to build lightweight blocks with LayerScale for stable training; the tiny-E configuration has widths 48/96/192/384 and depths 3/3/8/6, with parameter count in the tiny range.

- **Task type**: Image Classification.
- **backbone**: HENet (tiny-E configuration, `depth/block_nums=[3,3,8,6]`, `width/embed_dims=[48,96,192,384]`, `block_cls=["DWCB","GroupDWCB","AltDWCB","DWCB"]`, GELU activation, LayerScale, S2D downsampling, `num_classes=1000`), uses depthwise conv to build lightweight blocks with LayerScale for stable training.
- **neck**: — (HENet has a built-in fully connected classification head; no separate neck).
- **classification head**: HENet built-in fully connected classification head (`include_top=True`, `feature_mix_channel=1024`).
- **Loss function**: `SoftTargetCrossEntropy` (soft-label cross-entropy with mixup).
- **Model input**: single RGB image at resolution `224 × 224` (`1x3x224x224`).
- **Model output**: 1000-class prediction logits; argmax gives the predicted class.

### Official Repo and Paper

HybridEfficient Network is an efficient backbone designed by Horizon Robotics for the Journey series chips.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14094
