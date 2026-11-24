# FCOS (EfficientNet-b3)

FCOS eliminates anchors and predicts classification scores, centerness, and distances to box edges per pixel; `BiFPN` performs bidirectional feature fusion across 5 scales to improve small-object detection; training jointly supervises classification, centerness, and regression losses.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| FCOS | Single image `1x3x896x896` | EfficientNet-b3 | `BiFPN` | Detection boxes `(B,N,cls+reg)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | mAP | 0.4802 | 0.4634 | 0.4773 | 0.4772 |

> Results are based on `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 3.89 | 276.72 | 30.80 |
| J6P | 3.00 | 1439.73 | 34.80 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

FCOS eliminates anchors and predicts classification scores, centerness, and distances to box edges per pixel; `BiFPN` performs bidirectional feature fusion across 5 scales to improve small-object detection; training jointly supervises classification, centerness, and regression losses.

- **Task type**: 2D object detection (2D Object Detection).
- **backbone**: EfficientNet-b3 (`efficientnet`, `model_type=b3`, `include_top=False` removes classification head, `use_se_block=False`, ReLU activation).
- **neck**: `BiFPN` (`out_channels=120`, `stack=6`, input stride `[2,4,8,16,32]` → output stride `[8,16,32,64,128]`, `bifpn_sum` fusion).
- **Detection head**: `FCOSHead` (`feat_channels=120`, `stacked_convs=4`, 5 feature layers stride `[8,16,32,64,128]`, `dequant_output=True`).
- **Target assignment**: `DynamicFcosTarget` (`topK=10`, classification `FocalLoss`, regression `GIoULoss`).
- **Post-processing**: `FCOSDecoder` (`score_thr=0.05`, `nms_pre=1000`, NMS `iou_threshold=0.6`, `max_per_img=100`, uses centerness).
- **Loss**: Classification `FocalLoss` (`alpha=0.25, gamma=2.0`) + centerness `CrossEntropyLoss` (sigmoid) + regression `GIoULoss`.
- **Model input**: Single image, size `896 × 896`.
- **Model output**: 80-class detection boxes + confidence scores.

### Official Repo and Paper

Official repo: https://github.com/tianzhi0549/FCOS
Paper: https://arxiv.org/abs/1904.01355

Note: backbone is EfficientNet-b3; official repo uses a different backbone.
