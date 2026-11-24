# GaNet (MixVarGENet)

GaNet models lanes as "root keypoints + along-line offsets": the network predicts an 8× downsampled heatmap to locate lane starting points and regresses per-point offsets to reconstruct the full lane; the decoder filters keypoints by `kpt_thr` threshold, clusters and merges points on the same lane via `cluster_thr`, and finally maps back to original image coordinates.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| GaNet | Single frame `1x3x320x800` | MixVarGENet | `GaNetNeck` | Lane point sequences `(B,L,P,2)` |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | F1Score (F1) | 0.7937 | 0.791 | — | 0.7908 |

> Results are based on `march = March.NASH_M` (J6M) configuration; this task has no QAT stage (qat column is —).
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance measurement**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.68 | 2262.93 | 6.50 |
| J6P | 0.61 | 8656.18 | 6.70 |
| J6B | - | - | - |

J6B performance is not available for this model.

---

## Model Overview

### Core Design

GaNet models lanes as "root keypoints + along-line offsets": the network predicts an 8× downsampled heatmap to locate lane starting points and regresses per-point offsets to reconstruct the full lane; the decoder filters keypoints by `kpt_thr` threshold, clusters and merges points on the same lane via `cluster_thr`, and finally maps back to original image coordinates.

- **Task type**: Lane detection (Lane Detection).
- **backbone**: MixVarGENet (`include_top=False`, `output_list=[2, 3, 4]` three-scale features, corresponding to stride 8/16/32).
- **neck**: `GaNetNeck`: FPN (three-scale fusion to `hid_dim=32`) + Attention module (`attn_ratio=4`, position encoding `pos_shape=(1,10,25)`).
- **Detection head**: `GaNetHead` (outputs `hid_dim=32` dimensional features).
- **Target generation**: `GaNetTarget` (`hm_down_scale=8`, keypoint radius `radius=2`).
- **Post-processing**: `GaNetDecoder` (`root_thr=1`, `kpt_thr=0.4`, `cluster_thr=5`, `downscale=8`).
- **Loss**: `GaNetLoss`: `LaneFastFocalLoss` (keypoint classification, weight=1.0) + `L1Loss` (per-point offset regression weight=0.5 + integer offset regression weight=1.0).
- **Model input**: Single frame, size `320 × 800` (`FixedCrop` takes original region `(0,270,1640,320)` then `Resize` to 320×800; original resolution 1640×590).
- **Model output**: Ordered point sequences per lane (original image coordinate system), obtained via `GaNetDecoder` from heatmap keypoint clustering + offset decoding.

**Deployment note**: The deployment graph (`deploy_model`) removes `targets`/`post_process`/`losses`, keeping only backbone + neck + head; output name is `pred` heatmap features; `GaNetDecoder` is re-attached during inference (float/hbir/hbm infer) to complete lane decoding.

### Official Repo and Paper

Official repo: https://github.com/Wolfwjs/GANet
Paper: https://arxiv.org/abs/2204.07335

Note: backbone MixVarGENet is HEAL-developed.

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14098
