# MixVarGENet

`mixvargenet` 是基于 MixVarGENet 骨干（5 个 stage）的 ImageNet（ILSVRC2012，1000 类）图像分类模型，覆盖「浮点训练 → 量化校准 → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/mixvargenet/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| MixVarGENet | `1x3x224x224` | MixVarGENet | 5 stage mixvarge blocks | 分类 logits `(B,1000)` |

### 精度指标

评测集：ImageNet val（`./tmp_data/heal_data/imagenet/val_lmdb`，50000 张）。指标：Accuracy。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Accuracy | 0.716 | 0.7116 | — | 0.7116 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 `—`）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.36 | 5210.16 | 6.30 |
| J6P | 0.35 | 10820.24 | 6.40 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/mixvargenet/qat.bc`
- HBM：`./tmp_models/mixvargenet/compile/model.hbm`

---

## 模型介绍

### 核心设计

MixVarGENet 用混合变体 block（mixvarge）构造轻量骨干，前两 stage 用 f2/f4 基础 block，后两 stage 用带 group 的 f2_gb16 block，逐 stage 下采样到 stride 32。

- **任务类型**：图像分类（Image Classification）。
- **backbone**：MixVarGENet（`type=MixVarGENet`，5 个 stage：stride 2/4/8/16/32，`num_classes=1000`），用混合变体 block（mixvarge）构造轻量骨干，逐 stage 下采样到 stride 32。
- **neck**：5 个 stage 的 mixvarge block：`mixvarge_f2`、`mixvarge_f4`、`mixvarge_f2_gb16`（后两 stage 用 group block，gb16），通道 32→32→64→96→160。
- **分类头**：MixVarGENet 内置全连接分类头，直接输出 1000 类 logits。
- **损失函数**：`CEWithLabelSmooth`（带标签平滑的交叉熵）。
- **模型输入**：单张 RGB 图像，分辨率 `224 × 224`（`1x3x224x224`）。
- **模型输出**：1000 类预测 logits，取 argmax 为预测类别。

### 官方 repo 与论文链接

MixVarGENet为地平线提出的高效适配征程系列芯片的网络结构。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14084
---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 ImageNet（ILSVRC2012）打包后的 lmdb。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/imagenet/train_lmdb` | 训练 lmdb |
| `./tmp_data/heal_data/imagenet/val_lmdb` | 评测 lmdb（ImageNet val，50000 张） |

#### 官方下载

ImageNet ILSVRC2012 数据集官方页面：<https://image-net.org/challenges/LSVRC/2012/>（需注册下载）。需准备三个原始档案：

- `ILSVRC2012_img_train.tar`：训练集，含 1000 个类别子档案 `n*.tar`（每类约 1300 张）
- `ILSVRC2012_img_val.tar`：验证集，50000 张扁平图片
- `ILSVRC2012_devkit_t12.tar.gz`：开发包，含 val 真值与 `meta.bin` 元数据

下载后将上述档案放入原始数据目录 `./tmp_data/orig_data/imagenet/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/imagenet
# 将三个原始档案放入该目录
# ./tmp_data/orig_data/imagenet/ILSVRC2012_img_train.tar
# ./tmp_data/orig_data/imagenet/ILSVRC2012_img_val.tar
# ./tmp_data/orig_data/imagenet/ILSVRC2012_devkit_t12.tar.gz
```

#### 解压后目录结构

推荐直接把三个原始档案放在 `./tmp_data/orig_data/imagenet/` 下（无需手动解压），打包脚本内部通过 torchvision 的 `ImageNet` 数据集接口自动处理：`parse_archives()` 会从 `ILSVRC2012_img_train.tar` 解出 `train/<类别编号>/*.JPEG`（`<类别编号>` 为 ImageNet 的 WordNet 类别编号，例如 n01440764），从 `ILSVRC2012_img_val.tar` 解出 val 图片并利用 devkit 的真值自动归类到 `val/<类别编号>/`。devkit 必需（用于生成 `meta.bin`，val 自动归类依赖它）。

```
./tmp_data/orig_data/imagenet/
├── ILSVRC2012_img_train.tar         # 1000 类打包大 tar（推荐保持单个大 tar，packer 自动解压）
├── ILSVRC2012_img_val.tar           # 50000 张扁平 val 图片
└── ILSVRC2012_devkit_t12.tar.gz     # devkit（生成 meta.bin，val 自动归类必需）
```

**可选前置**：若已手动把 `ILSVRC2012_img_train.tar` 解成一堆 `n*.tar`（而非保持单个大 tar），可用 `tools/datasets/extract_imagenet_train.sh` 把 `n*.tar` 解成 `train/<类别编号>/` 布局：

```bash
bash tools/datasets/extract_imagenet_train.sh ./tmp_data/orig_data/imagenet
```

若 root 下保持单个 `ILSVRC2012_img_train.tar`，则 packer 自带的 torchvision 会自动处理，无需此脚本。

#### 打包命令

用 `tools/datasets/imagenet_packer.py`（`ImageNetPacker`）把官方原始 ImageNet 打包成分类 lmdb。`--src-data-dir` 指向原始数据目录 `./tmp_data/orig_data/imagenet`（含三个 tar/devkit），`--target-data-dir` 指向打包产物目录 `./tmp_data/heal_data/imagenet`（供 config 读取）。`orig_data/imagenet` 为 `--src-data-dir`（原始数据），`heal_data/imagenet` 为 `--target-data-dir`（打包后 lmdb，对应 config 的 `data_path`）。

train 与 val 各打包一次：

```bash
# 打 train split
python3 tools/datasets/imagenet_packer.py \
  --src-data-dir ./tmp_data/orig_data/imagenet \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/imagenet \
  --split-name train \
  --num-workers 20

# 打 val split
python3 tools/datasets/imagenet_packer.py \
  --src-data-dir ./tmp_data/orig_data/imagenet \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/imagenet \
  --split-name val \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/imagenet`（含 `ILSVRC2012_img_train.tar`、`ILSVRC2012_img_val.tar`、`ILSVRC2012_devkit_t12.tar.gz`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/imagenet`，产出 `train_lmdb` 与 `val_lmdb`，与 config 的 `data_path + "train_lmdb/"` / `data_path + "val_lmdb/"` 路径对应。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--split-name`：`train` 或 `val`，分别打训练集与验证集。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/imagenet/` 结构如下：

```
./tmp_data/heal_data/imagenet/
├── train_lmdb/        # 训练 lmdb
└── val_lmdb/          # 评测 lmdb（ImageNet val，50000 张）
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
python3 tools/datasets/imagenet_viz.py \
  --data-path ./tmp_data/heal_data/imagenet/train_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch | Calibration 步数 | Float 优化器 | 学习率策略 | 损失函数 |
|---|---|---|---|---|---|---|
| 8 | 128 | 240 | 10 | SGD（lr=0.4，momentum=0.9，weight_decay=4e-5） | CosLrUpdater（cosine 退火，warmup 5 epoch，warmup_begin_lr=1e-6，stop_lr=1e-5） | CEWithLabelSmooth |

本任务无跨任务预训练依赖，Float 训练从零开始。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。本任务无 QAT 阶段。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/mixvargenet \
  --stage float
# 产物：./tmp_models/mixvargenet/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/mixvargenet \
  --stage calibration
# 产物：./tmp_models/mixvargenet/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/mixvargenet/deploy.py
# 产物：./tmp_models/mixvargenet/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/mixvargenet/deploy.py
# 产物：./tmp_models/mixvargenet/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration，只需在量化模型基础上重新执行导出与编译。修改 `configs/mixvargenet/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/mixvargenet \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/mixvargenet \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/mixvargenet \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/mixvargenet \
  --stage hbm
```

评测指标为 Accuracy 与 TopKAccuracy(top_k=5)，评测集为 ImageNet val（50000 张）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像（保存到 `./demo/mixvargenet/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/mixvargenet/）
python3 tools/gen_demo.py \
  --config configs/mixvargenet

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/mixvargenet \
  --stage float
```

可视化结果保存到 `./demo/mixvargenet/`。
