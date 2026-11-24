# MOTR (EfficientNet-b3)

MOTR models tracking as per-frame query propagation: detection queries are passed across frames via `QueryInteractionModule`; new targets get new queries and disappeared targets are cleared; training jointly supervises detection and tracking losses.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| MOTR | multi-frame image sequence `1x3x800x1422` | EfficientNet-b3 | — | tracking boxes + track IDs `(B,Q,cls+reg+id)` |

### Accuracy Metrics

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
