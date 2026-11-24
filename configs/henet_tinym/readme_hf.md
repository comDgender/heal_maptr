# HENet-tinyM

HENet uses depthwise conv to build lightweight blocks with LayerScale for stable training; the tiny-M configuration has widths 64/128/192/384 and depths 4/3/8/6, with GroupDWCB in the first stage and parameter count in the tiny range.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| HENet-tinym | `1x3x224x224` | HENet | — | classification logits `(B,1000)` |

### Accuracy Metrics

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
