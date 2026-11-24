# UNet (MobileNetV1)

UNet encoder-decoder structure: MobileNetV1 extracts multi-scale features (stride 4/8/16/32/64); `DwUnet` upsamples level-by-level with depthwise separable convolutions and fuses feature maps; `SegHead` outputs segmentation logits at 5 scales; `SoftmaxFocalLoss` provides joint supervision; at deployment, `SegHead` exports only the highest-resolution (stride 4) scale argmax result.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| UNet | Single image `1x3x1024x2048` | MobileNetV1 | `DwUnet` | Segmentation mask `(B,19,H,W)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | MeanIOU | 0.6774 | 0.646 | 0.6707 | 0.6721 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 1.13 | 1115.70 | 15.80 |
| J6P | 0.86 | 6100.48 | 15.80 |
| J6B | 2.65 | 368.45 | 13.60 |

---

## Model Overview

### Core Design

UNet encoder-decoder structure: MobileNetV1 extracts multi-scale features (stride 4/8/16/32/64); `DwUnet` upsamples level-by-level with depthwise separable convolutions and fuses feature maps; `SegHead` outputs segmentation logits at 5 scales; `SoftmaxFocalLoss` provides joint supervision; at deployment, `SegHead` exports only the highest-resolution (stride 4) scale argmax result.

- **Task type**: Semantic segmentation (Semantic Segmentation).
- **backbone**: MobileNetV1 (`alpha=0.25`, `include_top=False` removes classification head, lightweight depthwise separable convolution backbone).
- **neck**: `DwUnet` (`base_channels=8`, `output_scales=(4,8,16,32,64)`, depthwise separable convolution U-Net decoder).
- **Segmentation head**: `SegHead` (`num_classes=19`, outputs at stride 4/8/16/32/64 (5 scales); training outputs logits, deployment outputs argmax).
- **Loss function**: `SoftmaxFocalLoss` (scale-weighted, mitigates class imbalance).
- **Normalization**: BGR→YUV444 then `Normalize(mean=128, std=128)`, then split into multi-scale pyramid input via `scale_factors`.
- **Model input**: `1×3×1024×2048` (RGB, native resolution).
- **Model output**: Per-pixel 19-class semantic segmentation map `(B,19,H,W)`.

### Official Repo and Paper

Official repo: https://github.com/pytorch/vision (MobileNetV1 PyTorch implementation)
Paper: https://arxiv.org/abs/1704.04861

Note: UNet paper at https://arxiv.org/abs/1505.04597.
