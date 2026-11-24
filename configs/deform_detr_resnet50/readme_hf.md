# Deformable DETR (ResNet-50)

Deformable DETR replaces DETR's global self-attention with multi-scale deformable attention: each query only samples a small number of points near reference points, enabling faster convergence and lower compute. Four feature levels plus 900 queries provide multi-scale candidates, then the decoder progressively refines boxes layer by layer.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| DeformableDETR | Single image `1x3x800x1332` | ResNet-50 | `ChannelMapperNeck` | Detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | mAP | 0.4384 | 0.412 | 0.4526 | 0.4529 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 144.26 | 6.94 | 647.40 |
| J6P | 78.36 | 29.15 | 656.30 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

Deformable DETR replaces DETR's global self-attention with multi-scale deformable attention: each query only samples a small number of points near reference points, enabling faster convergence and lower compute. Four feature levels plus 900 queries provide multi-scale candidates, then the decoder progressively refines boxes layer by layer.

- **Task type**: 2D object detection (2D Object Detection).
- **backbone**: ResNet-50 (`ResNet50`, `include_top=False` removes classification head).
- **neck**: `ChannelMapperNeck` (`in_channels=[512,1024,2048]`, `out_channel=256`, 1×1 conv, `extra_convs=1`).
- **Position encoding**: `PositionEmbeddingSine` (`num_pos_feats=128`, normalized).
- **Transformer**: `DeformableDetrTransformer` (encoder 6 layers + decoder 6 layers, `embed_dim=256`, `num_heads=8`, `feedforward_dim=1024`, `num_feature_levels=4`, `num_queries=900`).
- **Post-processing**: `DeformDetrPostProcess` (evaluation selects `select_box_nums_for_evaluation=300` boxes).
- **Loss**: `DeformableCriterion`: classification focal loss + L1 bbox + GIoU, `HungarianMatcher` bipartite matching, `aux_loss=True`.
- **Key flags**: `with_box_refine=False`, `as_two_stage=False`.
- **Model input**: Single image, size `800 × 1332`.
- **Model output**: 80-class detection boxes + confidence scores.

### Official Repo and Paper

Official repo: https://github.com/fundamentalvision/Deformable-DETR
Paper: https://arxiv.org/abs/2010.04159
