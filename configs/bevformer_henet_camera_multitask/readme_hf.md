# BevFormer+HENet Multitask

The BevFormer+HENet multitask model uses HENet as the camera backbone to extract multi-view features, then BevFormer ViewTransformer and temporal BEV Encoder to transform them into bird's-eye-view features. BEV features are fed to both a detection head (3D object detection) and an occupancy head (FlashOcc-style semantic occupancy prediction), enabling joint det+occ training.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| BevFusion | 6-camera multi-view image sequence `(B,6,3,512,960)` | HENet | FPN | det bounding boxes `(B,N,cls+reg)`; occ occupancy grid `(B,C,H,W)` |

### Accuracy Metrics

| March | Metric | float |
| --- | --- | --- |
| J6M | NDS | 0.3573 |
|  | mAP | 0.3017 |
|  | Occ mIoU | 0.3146 |

> Results measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

---

## Model Overview

### Core Design

The BevFormer+HENet multitask model uses HENet as the camera backbone to extract multi-view features, then BevFormer ViewTransformer and temporal BEV Encoder to transform them into bird's-eye-view features. BEV features are fed to both a detection head (3D object detection) and an occupancy head (FlashOcc-style semantic occupancy prediction), enabling joint det+occ training.

- **Task type**: Multitask fusion (3D object detection + occupancy grid prediction).
- **backbone**: HENet (camera feature extraction), extracting multi-view features.
- **neck**: FPN.
- **Detection head**: BevFormer detection head, outputting 10-class 3D bounding boxes + velocity (`num_classes=10`).
- **Occupancy head**: Semantic occupancy prediction, 18 classes (`num_classes_occ=18`).
- **BEV range**: `bev_size=(51.2, 51.2, 0.8)`, `bev_size_occ=(40, 40)`.
- **Model input**: 6-camera image sequence `(B,6,3,512,960)` (`queue_length=1`).
- **Model output**: det 3D bounding boxes + occ occupancy grid semantics `(B,C,H,W)`.

### Official Repo and Paper

Official repo: https://github.com/fundamentalvision/BevFormer
Paper: https://arxiv.org/abs/2203.17270

Note: The camera backbone HENet is a HEAL in-house implementation; the official repo uses a different backbone.
