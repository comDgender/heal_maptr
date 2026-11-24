# Deformable DETR (ResNet-50)

本文档介绍 `deform_detr_resnet50` 模型，并说明如何基于原始 COCO 2017 数据集，完成「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/deform_detr_resnet50/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| DeformableDETR | 单张图像 `1x3x800x1332` | ResNet-50 | `ChannelMapperNeck` | 检测框 `(B,N,cls+reg)` |

### 精度指标

评测集：COCO val2017（`./tmp_data/heal_data/mscoco/val_lmdb` + `instances_val2017.json`）。指标：COCO mAP。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | COCO mAP | 0.4384 | 0.412 | 0.4526 | 0.4529 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 144.26 | 6.94 | 647.40 |
| J6P | 78.36 | 29.15 | 656.30 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/deform_detr_resnet50/qat.bc`
- HBM：`./tmp_models/deform_detr_resnet50/compile/model.hbm`

---

## 模型介绍

### 核心设计

Deformable DETR 把 DETR 的全局自注意力替换为多尺度可变形注意力（Deformable Attention），每个 query 只参考参考点附近的少量采样点，收敛快、计算量低；4 个特征层级 + 900 个 query 提供多尺度候选，再经 decoder 逐层精修框。

- **任务类型**：2D 目标检测（2D Object Detection）。
- **backbone**：ResNet-50（`ResNet50`，`include_top=False` 去除分类头）。
- **neck**：`ChannelMapperNeck`（`in_channels=[512,1024,2048]`，`out_channel=256`，1×1 conv，`extra_convs=1`）。
- **位置编码**：`PositionEmbeddingSine`（`num_pos_feats=128`，归一化）。
- **Transformer**：`DeformableDetrTransformer`（encoder 6 层 + decoder 6 层，`embed_dim=256`，`num_heads=8`，`feedforward_dim=1024`，`num_feature_levels=4`，`num_queries=900`）。
- **后处理**：`DeformDetrPostProcess`（评测选 `select_box_nums_for_evaluation=300` 框）。
- **损失**：`DeformableCriterion`：分类 focal loss + L1 bbox + GIoU，`HungarianMatcher` 二分匹配，`aux_loss=True`。
- **关键开关**：`with_box_refine=False`，`as_two_stage=False`。
- **模型输入**：单张图像，尺寸 `800 × 1332`。
- **模型输出**：80 类检测框 + 置信度。

### 官方 repo 与论文链接

官方repo: https://github.com/fundamentalvision/Deformable-DETR
论文: https://arxiv.org/abs/2010.04159

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 MSCOCO 2017 打包后的 lmdb。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/mscoco/train_lmdb` | 训练集 lmdb |
| `./tmp_data/heal_data/mscoco/val_lmdb` | 评测集 lmdb |
| `./tmp_data/heal_data/mscoco/instances_val2017.json` | 评测用 COCO 标注（`COCODetectionMetric` 读取） |

> **路径说明**：`orig_data/mscoco` 为 `--src-data-dir`（原始解压数据），`heal_data/mscoco` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。注意评测用的 `instances_val2017.json` 需放在 `heal_data/mscoco/` 根目录（非 `annotations/` 子目录），packer 只产 lmdb 不产 json，需手动拷贝（见打包命令）。

#### 官方下载

COCO 2017 数据集官方页面：<https://cocodataset.org/#download>。需下载以下三个档案：

| 档案 | 大小 | 内容 |
|---|---|---|
| `train2017.zip` | ~18 GB | 训练集图片（118K 张） |
| `val2017.zip` | ~0.8 GB | 验证集图片（5K 张） |
| `annotations_trainval2017.zip` | ~241 MB | 标注（`instances_train2017.json`、`instances_val2017.json` 等） |

下载后将三个档案解压到原始数据目录 `./tmp_data/orig_data/mscoco/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/mscoco
unzip train2017.zip -d ./tmp_data/orig_data/mscoco
unzip val2017.zip -d ./tmp_data/orig_data/mscoco
unzip annotations_trainval2017.zip -d ./tmp_data/orig_data/mscoco
```

#### 解压后目录结构

解压后 `./tmp_data/orig_data/mscoco/` 结构如下：

```
./tmp_data/orig_data/mscoco/
├── train2017/                     # 训练集图片
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ... (共约 118K 张)
├── val2017/                       # 验证集图片
│   ├── 000000000139.jpg
│   └── ... (共约 5K 张)
└── annotations/                   # 标注
    ├── instances_train2017.json   # 训练集检测标注
    ├── instances_val2017.json     # 验证集检测标注（评测用）
    └── ... (captions/person_keypoints 等其他标注，本任务不使用)
```

#### 打包命令

用 `tools/datasets/mscoco_packer.py`（`CocoDetectionPacker`）把官方原始 COCO 2017 打包成 lmdb。`--src-data-dir` 指向原始数据目录 `./tmp_data/orig_data/mscoco`（含 `train2017/`、`val2017/`、`annotations/`），`--target-data-dir` 指向打包产物目录 `./tmp_data/heal_data/mscoco`（供 config 读取）。packer 读取 `src_data_dir/{split}2017/` 图片与 `src_data_dir/annotations/instances_{split}2017.json` 标注。

```bash
# 打 train split
python3 tools/datasets/mscoco_packer.py \
  --src-data-dir ./tmp_data/orig_data/mscoco \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mscoco \
  --split-name train \
  --num-workers 20

# 打 val split
python3 tools/datasets/mscoco_packer.py \
  --src-data-dir ./tmp_data/orig_data/mscoco \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mscoco \
  --split-name val \
  --num-workers 20
```

packer 只产出 `train_lmdb` 与 `val_lmdb`，**不产出** `instances_val2017.json`。评测用的标注文件需手动从原始数据目录拷贝到打包产物目录根（config 的 `ann_file` 指向 `heal_data/mscoco/instances_val2017.json`）：

```bash
cp ./tmp_data/orig_data/mscoco/annotations/instances_val2017.json ./tmp_data/heal_data/mscoco/
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/mscoco`（含 `train2017/`、`val2017/`、`annotations/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/mscoco`，产出 `train_lmdb` 与 `val_lmdb`，与 `common.py` 的 `data_path + "train_lmdb/"`、`data_path + "val_lmdb/"` 路径对应。
- `--split-name`：`train` 或 `val`，对应 COCO 2017 的 train/val 划分。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包并拷贝标注后，`./tmp_data/heal_data/mscoco/` 结构如下：

```
./tmp_data/heal_data/mscoco/
├── train_lmdb/                 # 训练集 lmdb
├── val_lmdb/                   # 评测集 lmdb
└── instances_val2017.json      # 评测用 COCO 标注（手动拷贝自 orig_data/mscoco/annotations/）
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
python3 tools/datasets/mscoco_viz.py \
  --data-path ./tmp_data/heal_data/mscoco/train_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | QAT 训练 epoch 数 | Float 优化器 | QAT 优化器 |
|---|---|---|---|---|---|---|
| 8 | 2 | 50 | 10 | 4 | AdamW（base lr=1e-4，backbone lr=1e-5，weight_decay=1e-4，norm weight_decay=0），StepDecayLr @ epoch 40，梯度裁剪 max_norm=0.1 | AdamW（lr=1e-5，weight_decay=0），StepDecayLr @ epoch 3，梯度裁剪 max_norm=0.1 |

**预训练权重**：Float 训练需加载 ResNet-50（ImageNet 分类）的浮点骨干权重以迁移 ResNet-50。请先完成 `configs/resnet50` 的 Float 训练，并将其 `float-checkpoint-best.pth.tar` 放到 `./tmp_pretrained_models/resnet50/float-checkpoint-best.pth.tar`。加载时允许部分参数缺失（`allow_miss=True, ignore_extra=True`），仅迁移骨干权重。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/deform_detr_resnet50 \
  --stage float
# 产物：./tmp_models/deform_detr_resnet50/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/deform_detr_resnet50 \
  --stage calibration
# 产物：./tmp_models/deform_detr_resnet50/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/deform_detr_resnet50 \
  --stage qat
# 产物：./tmp_models/deform_detr_resnet50/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/deform_detr_resnet50/deploy.py
# 产物：./tmp_models/deform_detr_resnet50/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/deform_detr_resnet50/deploy.py
# 产物：./tmp_models/deform_detr_resnet50/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在已有 QAT 量化模型基础上重新执行导出与编译。修改 `configs/deform_detr_resnet50/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/deform_detr_resnet50 \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/deform_detr_resnet50 \
  --stage hbm
```

评测指标为 COCO mAP（`COCODetectionMetric`），标注文件位于 `./tmp_data/heal_data/mscoco/instances_val2017.json`，评测集为 COCO val2017。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像帧（保存到 `./demo/deform_detr_resnet50/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理帧（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/deform_detr_resnet50

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/deform_detr_resnet50 \
  --stage float
```

可视化结果保存到 `./demo/deform_detr_resnet50/`。
