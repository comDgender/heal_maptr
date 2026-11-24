# FCOS (EfficientNet-b3)

本文档介绍 `fcos_efficientnetb3` 模型，并说明如何基于原始 COCO 2017 数据集，完成「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/fcos_efficientnetb3/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| FCOS | 单张图像 `1x3x896x896` | EfficientNet-b3 | `BiFPN` | 检测框 `(B,N,cls+reg)` |

### 精度指标

评测集：COCO val2017（`./tmp_data/heal_data/mscoco/val_lmdb` + `instances_val2017.json`）。指标：COCO mAP。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | COCO mAP | 0.4802 | 0.4634 | 0.4773 | 0.4772 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 3.89 | 276.72 | 30.80 |
| J6P | 3.00 | 1439.73 | 34.80 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/fcos_efficientnetb3/qat.bc`
- HBM：`./tmp_models/fcos_efficientnetb3/compile/model.hbm`

---

## 模型介绍

### 核心设计

FCOS 摒弃 anchor，逐像素预测分类分数、中心度与到框四边的距离；`BiFPN` 在 5 个尺度间做双向特征融合，提升小目标检测；训练时联合监督分类、中心度与回归损失。

- **任务类型**：2D 目标检测（2D Object Detection）。
- **backbone**：EfficientNet-b3（`efficientnet`，`model_type=b3`，`include_top=False` 去除分类头，`use_se_block=False`，ReLU 激活）。
- **neck**：`BiFPN`（`out_channels=120`，`stack=6`，输入 stride `[2,4,8,16,32]` → 输出 stride `[8,16,32,64,128]`，`bifpn_sum` 融合）。
- **检测头**：`FCOSHead`（`feat_channels=120`，`stacked_convs=4`，5 个特征层 stride `[8,16,32,64,128]`，`dequant_output=True`）。
- **目标分配**：`DynamicFcosTarget`（`topK=10`，分类 `FocalLoss`、回归 `GIoULoss`）。
- **后处理**：`FCOSDecoder`（`score_thr=0.05`，`nms_pre=1000`，NMS `iou_threshold=0.6`，`max_per_img=100`，使用 centerness）。
- **损失函数**：分类 `FocalLoss`（`alpha=0.25, gamma=2.0`）+ 中心度 `CrossEntropyLoss`（sigmoid）+ 回归 `GIoULoss`。
- **模型输入**：单张图像，尺寸 `896 × 896`。
- **模型输出**：80 类检测框 + 置信度。

### 官方 repo 与论文链接

官方repo: https://github.com/tianzhi0549/FCOS
论文: https://arxiv.org/abs/1904.01355

注：backbone 为 EfficientNet-b3，官方 repo backbone 不同。

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

packer 只产出 `train_lmdb` 与 `val_lmdb`，**不产出** `instances_val2017.json`。评测用的标注文件需手动从原始数据目录拷贝到打包产物目录根（config 的 `ann_file_val` 指向 `heal_data/mscoco/instances_val2017.json`）：

```bash
cp ./tmp_data/orig_data/mscoco/annotations/instances_val2017.json ./tmp_data/heal_data/mscoco/
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/mscoco`（含 `train2017/`、`val2017/`、`annotations/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/mscoco`，产出 `train_lmdb` 与 `val_lmdb`，与 `data.py` 的 `data_path + "train_lmdb/"`、`data_path + "val_lmdb/"` 路径对应。
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
| 8 | 12 | 600 | 100 | 30 | SGD（lr=0.2，momentum=0.937，nesterov，weight_decay=4e-5），CosLr warmup 2 epoch，EMA | SGD（lr=1e-4，momentum=0.9，weight_decay=4e-5），CosLr warmup 0.1 epoch |

> 本任务 Float 训练从零开始，无需加载预训练权重。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/fcos_efficientnetb3 \
  --stage float
# 产物：./tmp_models/fcos_efficientnetb3/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/fcos_efficientnetb3 \
  --stage calibration
# 产物：./tmp_models/fcos_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/fcos_efficientnetb3 \
  --stage qat
# 产物：./tmp_models/fcos_efficientnetb3/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/fcos_efficientnetb3/deploy.py
# 产物：./tmp_models/fcos_efficientnetb3/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/fcos_efficientnetb3/deploy.py
# 产物：./tmp_models/fcos_efficientnetb3/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在已有 QAT 量化模型基础上重新执行导出与编译。修改 `configs/fcos_efficientnetb3/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/fcos_efficientnetb3 \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/fcos_efficientnetb3 \
  --stage hbm
```

评测指标为 COCO mAP（`COCODetectionMetric`），标注文件位于 `./tmp_data/heal_data/mscoco/instances_val2017.json`，评测集为 COCO val2017。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像帧（保存到 `./demo/fcos_efficientnetb3/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理帧（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/fcos_efficientnetb3

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/fcos_efficientnetb3 \
  --stage float
```

可视化结果保存到 `./demo/fcos_efficientnetb3/`。
