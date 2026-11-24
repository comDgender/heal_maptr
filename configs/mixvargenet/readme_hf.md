# MixVarGENet

MixVarGENet builds a lightweight backbone with mixed-variant blocks (mixvarge); the first two stages use f2/f4 base blocks, the last two stages use f2_gb16 blocks with groups, downsampling stage by stage to stride 32.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| MixVarGENet | `1x3x224x224` | MixVarGENet | 5 stage mixvarge blocks | classification logits `(B,1000)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Accuracy | 0.716 | 0.7116 | — | 0.7116 |

> Data tested with `march = March.NASH_M` (J6M); this task has no QAT stage (`—` in the qat column).
>
> HEAL versions: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with 8 threads on a single core; Latency is measured with single core, single thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.36 | 5210.16 | 6.30 |
| J6P | 0.35 | 10820.24 | 6.40 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

MixVarGENet builds a lightweight backbone with mixed-variant blocks (mixvarge); the first two stages use f2/f4 base blocks, the last two stages use f2_gb16 blocks with groups, downsampling stage by stage to stride 32.

- **Task type**: Image Classification.
- **backbone**: MixVarGENet (`type=MixVarGENet`, 5 stages: stride 2/4/8/16/32, `num_classes=1000`), builds a lightweight backbone with mixed-variant blocks (mixvarge), downsampling stage by stage to stride 32.
- **neck**: 5 stages of mixvarge blocks: `mixvarge_f2`, `mixvarge_f4`, `mixvarge_f2_gb16` (last two stages use group blocks, gb16), channels 32→32→64→96→160.
- **classification head**: MixVarGENet built-in fully connected classification head, directly outputs 1000-class logits.
- **Loss function**: `CEWithLabelSmooth` (cross-entropy with label smoothing).
- **Model input**: single RGB image at resolution `224 × 224` (`1x3x224x224`).
- **Model output**: 1000-class prediction logits; argmax gives the predicted class.

### Official Repo and Paper

MixVarGENet is an efficient network architecture proposed by Horizon Robotics, optimized for the Journey series chips.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14084
