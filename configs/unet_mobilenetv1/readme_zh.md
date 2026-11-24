# UNet (MobileNetV1)

本文档介绍 `unet_mobilenetv1` 模型，并说明如何基于原始 Cityscapes 数据集，完成「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/unet_mobilenetv1/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| UNet | 单张图像 `1x3x1024x2048` | MobileNetV1 | `DwUnet` | 分割掩码 `(B,19,H,W)` |

### 精度指标

评测集：Cityscapes val（`./tmp_data/heal_data/cityscapes/val_lmdb`）。指标：MeanIOU（mIoU，19 类）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | MeanIOU | 0.6774 | 0.646 | 0.6707 | 0.6721 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 1.13 | 1115.70 | 15.80 |
| J6P | 0.86 | 6100.48 | 15.80 |
| J6B | 2.65 | 368.45 | 13.60 |

编译产物路径：
- HBIR：`./tmp_models/unet_mobilenetv1/qat.bc`
- HBM：`./tmp_models/unet_mobilenetv1/compile/model.hbm`

---

## 模型介绍

### 核心设计

UNet 编码-解码结构：MobileNetV1 提取多尺度特征（stride 4/8/16/32/64），`DwUnet` 用深度可分离卷积逐级上采样并融合特征图，`SegHead` 在 5 个尺度输出分割 logits，`SoftmaxFocalLoss` 联合监督；部署时 `SegHead` 仅导出最高分辨率（stride 4）尺度的 argmax 结果。

- **任务类型**：语义分割（Semantic Segmentation）。
- **backbone**：MobileNetV1（`alpha=0.25`，`include_top=False` 去除分类头，深度可分离卷积轻量骨干）。
- **neck**：`DwUnet`（`base_channels=8`，`output_scales=(4,8,16,32,64)`，深度可分离卷积 U-Net 解码器）。
- **分割头**：`SegHead`（`num_classes=19`，在 stride 4/8/16/32/64 共 5 个尺度输出；训练时输出 logits，部署时 argmax 输出）。
- **损失函数**：`SoftmaxFocalLoss`（按尺度加权，缓解类别不平衡）。
- **归一化**：BGR→YUV444 后做 `Normalize(mean=128, std=128)`，再按 `scale_factors` 拆分多尺度金字塔输入。
- **模型输入**：`1×3×1024×2048`（RGB，Cityscapes 原始分辨率）。
- **模型输出**：每像素 19 类语义分割图 `(B,19,H,W)`。

### 官方 repo 与论文链接

官方repo: https://github.com/pytorch/vision （MobileNetV1 PyTorch实现）
论文: https://arxiv.org/abs/1704.04861

注：UNet 论文见 https://arxiv.org/abs/1505.04597。

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 Cityscapes 打包后的 lmdb。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/cityscapes/train_lmdb` | 训练集 lmdb |
| `./tmp_data/heal_data/cityscapes/val_lmdb` | 评测集 lmdb |

#### 官方下载

Cityscapes 数据集官方页面：<https://www.cityscapes-dataset.com/downloads/>（需注册下载）。需获取以下两个压缩包：

- `gtFine_trainvaltest.zip`（细粒度标注，含 train/val/test 三 split）
- `leftImg8bit_trainvaltest.zip`（原始左目图像，含 train/val/test 三 split）

下载后将两个 zip 解压到原始数据目录 `./tmp_data/orig_data/cityscapes/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/cityscapes
unzip gtFine_trainvaltest.zip -d ./tmp_data/orig_data/cityscapes
unzip leftImg8bit_trainvaltest.zip -d ./tmp_data/orig_data/cityscapes
# 解压得到 ./tmp_data/orig_data/cityscapes/{gtFine,leftImg8bit}/...
```

Cityscapes 无需额外预处理脚本，解压后即可直接打包。

#### 解压后目录结构

解压后 `./tmp_data/orig_data/cityscapes/` 结构如下：

```
./tmp_data/orig_data/cityscapes/
├── gtFine/                                   # 细粒度标注
│   ├── train/                                # 18 个城市
│   │   └── <city>/                           #   如 aachen, bochum, bremen, ...
│   │       ├── <name>_gtFine_color.png        #   彩色可视化标签
│   │       ├── <name>_gtFine_instanceIds.png  #   实例 ID 标签
│   │       └── <name>_gtFine_labelIds.png      #   语义标签 ID（packer 使用此文件）
│   ├── val/                                  # 3 个城市：frankfurt, lindau, munster
│   │   └── <city>/<name>_gtFine_{color,instanceIds,labelIds}.png
│   └── test/                                 # 6 个城市（test 无公开标注的 labelIds，本任务不使用）
│       └── <city>/<name>_gtFine_{color,instanceIds,labelIds}.png
└── leftImg8bit/                              # 原始左目图像
    ├── train/
    │   └── <city>/<name>_leftImg8bit.png      # 8 bit PNG，1024×2048
    ├── val/
    │   └── <city>/<name>_leftImg8bit.png
    └── test/
        └── <city>/<name>_leftImg8bit.png
```

**说明**：打包脚本通过 `torchvision.datasets.Cityscapes` 读取 `gtFine/<split>/<city>/*_gtFine_labelIds.png`（`mode="fine"`, `target_type="semantic"`）与对应的 `leftImg8bit/<split>/<city>/*_leftImg8bit.png`，`color.png` 与 `instanceIds.png` 不参与打包。

#### 打包命令

用 `tools/datasets/cityscapes_packer.py`（`CityscapesPacker`）把官方原始 Cityscapes 打包成 lmdb。`--src-data-dir` 指向原始数据目录 `./tmp_data/orig_data/cityscapes`（含 `gtFine/`、`leftImg8bit/`），`--target-data-dir` 指向打包产物目录 `./tmp_data/heal_data/cityscapes`（供 config 读取）。`--split-name` 对应 `torchvision.datasets.Cityscapes` 的 `split` 参数，取 `train` 或 `val`。train 与 val 两个 split 各打包一次。

```bash
# 打 train split
python3 tools/datasets/cityscapes_packer.py \
  --src-data-dir ./tmp_data/orig_data/cityscapes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/cityscapes \
  --split-name train \
  --num-workers 20

# 打 val split
python3 tools/datasets/cityscapes_packer.py \
  --src-data-dir ./tmp_data/orig_data/cityscapes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/cityscapes \
  --split-name val \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/cityscapes`（含 `gtFine/`、`leftImg8bit/`）。
- `--pack-type`：打包格式，统一用 `lmdb`。产物名为 `{split_name}_{pack_type}`，即 `train_lmdb`、`val_lmdb`。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/cityscapes`，产出 `train_lmdb` 与 `val_lmdb`，与 `common.py` 的 `data_path="./tmp_data/heal_data/cityscapes/"` 拼接后的路径逐字对应。
- `--split-name`：数据 split，`train` 或 `val`（对应 `torchvision.datasets.Cityscapes` 的 `split` 参数）。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/cityscapes/` 结构如下：

```
./tmp_data/heal_data/cityscapes/
├── train_lmdb/     # 训练集 lmdb
└── val_lmdb/       # 评测集 lmdb
```

可用可视化脚本抽样检查打包结果是否正确（输入图像与语义标签对齐）：

```bash
python3 tools/datasets/cityscapes_viz.py \
  --data-path ./tmp_data/heal_data/cityscapes/train_lmdb \
  --viz-num 50 \
  --save-path ./tmp_data/heal_data/cityscapes/viz
```

可视化 PNG 保存到 `./tmp_data/heal_data/cityscapes/viz/`，可人工核对图像与 `gt_seg` 是否对齐。

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | QAT 训练 epoch 数 | Float 优化器 | QAT 优化器 | SyncBN | 类别数 |
|---|---|---|---|---|---|---|---|---|
| 8 | 4 | 600 | 10 | 30 | SGD（lr=0.01，momentum=0.9，weight_decay=5e-3，CosLr warmup 2 epoch） | SGD（lr=8e-5，momentum=0.9，weight_decay=5e-3，StepDecay@epoch20 ×0.1） | Float 与 QAT 阶段均开启 | 19 |

> 本任务 Float 训练从零开始，无跨任务预训练权重依赖。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/unet_mobilenetv1 \
  --stage float
# 产物：./tmp_models/unet_mobilenetv1/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/unet_mobilenetv1 \
  --stage calibration
# 产物：./tmp_models/unet_mobilenetv1/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/unet_mobilenetv1 \
  --stage qat
# 产物：./tmp_models/unet_mobilenetv1/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/unet_mobilenetv1/deploy.py
# 产物：./tmp_models/unet_mobilenetv1/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/unet_mobilenetv1/deploy.py
# 产物：./tmp_models/unet_mobilenetv1/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在已有 QAT 量化模型基础上重新执行导出与编译。修改 `configs/unet_mobilenetv1/common.py` 的 `march` 字段为目标芯片：
> ```python
> march = March.NASH_M   # J6M
> march = March.NASH_P   # J6P
> march = March.NASH_B   # J6B
> ```
> 然后重新执行 `export_hbir.py` 与 `compile_perf.py`（即上方两条命令）。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/unet_mobilenetv1 \
  --stage hbm
```

评测指标为 MeanIOU（平均交并比，mIoU），评测集为 Cityscapes val（`./tmp_data/heal_data/cityscapes/val_lmdb`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像帧（保存到 `./demo/unet_mobilenetv1/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理帧（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/unet_mobilenetv1/）
python3 tools/gen_demo.py \
  --config configs/unet_mobilenetv1

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/unet_mobilenetv1 \
  --stage float
```

可视化结果保存到 `./demo/unet_mobilenetv1/`。
