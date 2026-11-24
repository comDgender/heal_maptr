# BEVFormer (ResNet-50)

`bevformer_tiny_resnet50_detection` 是基于 ResNet-50 骨干 + FPN 颈的 BEVFormer tiny 版 nuScenes BEV 3D 目标检测模型，覆盖「浮点训练 → 量化校准 → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/bevformer_tiny_resnet50_detection/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| BevFormer | 6 相机多视图图像序列 `(B,6,3,480,800)` | ResNet-50 | FPN | 3D 检测框 `(B,N,cls+reg)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`）。指标：NDS（nuScenes 检测综合指标，monitor 指标，越高越好）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3739 | 0.3607 | 0.3734 | 0.3669 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 21.76 | 46.52 | 112.80 |
| J6P | 14.15 | 274.97 | 113.80 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/bevformer_tiny_resnet50_detection/qat.bc`
- HBM：`./tmp_models/bevformer_tiny_resnet50_detection/compile/model.hbm`

---

## 模型介绍

### 核心设计

BEVFormer 通过可学习的时空注意力（Temporal Self-Attention + Spatial Cross-Attention）从多相机序列中提取 BEV 特征：ResNet-50 + FPN 提取多尺度图像特征，经 BevFormerViewTransformer 投影到 BEV，BEVFormerEncoder 融合时序与空间信息，最终由 BEVFormerDetDecoder 解码 3D 检测框。

- **任务类型**：BEV 3D 目标检测（BEV 3D Object Detection）。
- **backbone**：ResNet-50（`ResNet50`，`include_top=False`，ImageNet 预训练 `num_classes=1000`）。
- **neck**：FPN（`FPN`，多尺度特征金字塔，`out_strides=[32]`，`out_channels=[256]`）。
- **检测头**：`BEVFormerDetDecoder`（DETR 风格解码器 + 匈牙利匹配）。
- **损失函数**：`BevFormerCriterion`（FocalLoss + L1Loss，经 BevFormerHungarianAssigner3D 匹配）。
- **模型输入**：6 相机多视图图像序列，`(B,6,3,480,800)`（原始 `orig_shape=(3,900,1600)` → resize `(3,450,800)` → pad 至 `(3,480,800)`，`num_views=6`，训练 `queue_length=3`、评测 `queue_length=1`，`NuscenesBevSequenceDataset`）。
- **模型输出**：BEV 特征上的 3D 检测框（类别 + 中心 + 尺寸 + 朝向），`num_query=900`，`num_classes=10`，经 `BevFormerProcess` 解码（`max_num=300`，`score_threshold=0.3`）。

### 官方 repo 与论文链接

官方repo: https://github.com/fundamentalvision/BevFormer
论文: https://arxiv.org/abs/2203.17270

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14101

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 nuScenes v1.0-trainval 打包后的 lmdb。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 lmdb（base） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 lmdb（base，nuScenes val） |

> 说明：本任务 map GT 由 `VectorizedLocalMap` 在线生成（`map_path=meta_rootdir`），不读 map lmdb，因此无需打包 map lmdb。

#### 官方下载

nuScenes 数据集官方页面：<https://www.nuscenes.org/download>（需注册下载，v1.0-trainval）。解压后含 `samples/`（标注关键帧，6 相机）、`sweeps/`（中间帧）、`maps/`（地图扩张）、`v1.0-trainval/`（元数据 json：annotation、scene、sample、ego_pose 等）。原始数据放入 `./tmp_data/orig_data/nuscenes/`（该目录作为 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# 将官方解压结果放入该目录
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
```

#### 解压后目录结构

```
./tmp_data/orig_data/nuscenes/
├── samples/          # 标注关键帧，6 相机（CAM_FRONT/FRONT_LEFT/...）
├── sweeps/           # 中间帧，6 相机
├── maps/             # 地图扩张数据
└── v1.0-trainval/    # 元数据 json（annotation/scene/sample/ego_pose 等）
```

#### 打包命令

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析逻辑，直接读取官方解压目录即可打包（禁止运行 create_data_*.py 等旧脚本）。

打包顺序约束：base 先；occ 在 base 后须 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际仅使用 base lmdb（`{split}_lmdb`），lidar/map/occ 可不打包。

`NuscenesPacker`（`tools/datasets/nuscenes_packer.py`）读取官方解压目录，产出 `{split}_lmdb`。本任务 base packer 的 train + val 两条命令如下：

```bash
# 打 train split（base）
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# 打 val split（base）
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

其余 3 个 packer 的用途与何时需要（本任务可不打包）：
- **lidar** `NuscenesLidarPacker`（`tools/datasets/nuscenes_lidar_packer.py`）：产出 `{split}_lidar_lmdb`，仅 lidar 类任务需打包，参数要点 `--max-sweeps 10` `--load-dim` `--use_dim`。
- **map** `NuscenesMapGtPacker`（`tools/datasets/nuscenes_map_packer.py`）：产出 `{split}_map_lmdb`，仅 map 类任务需打包，`--map-path` 指向 meta 目录，PV seg mask 不打包训练时在线生成。
- **occ** `NuscenesOccPacker`（`tools/datasets/nuscenes_occ_packer.py`）：产出 `{split}_occ_lmdb`，仅 occ 类任务需打包，`occ3d/gts/` 来自 Occ3D-nuScenes 官方标签需用户额外下载放置。

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/nuscenes`（含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/nuscenes/v1.0-trainval`，产出 `{split}_lmdb`，与 config 的 `data_rootdir` 一致。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--split-name`：`train` 或 `val`，分别打训练集与验证集。
- `--version v1.0-trainval`：数据版本（packer 默认 `v1.0-mini`，须显式覆盖为 `v1.0-trainval`）。
- `--num-workers`：并行打包进程数（默认 20）。

`orig_data/nuscenes` 为 `--src-data-dir`（原始数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（对应 config 的 `data_rootdir`）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/v1.0-trainval/` 结构如下：

```
./tmp_data/heal_data/nuscenes/v1.0-trainval/
├── train_lmdb/          # 训练 lmdb（base）← 本任务使用
├── val_lmdb/            # 评测 lmdb（base）← 本任务使用
├── train_lidar_lmdb/    # 本任务不使用
├── val_lidar_lmdb/      # 本任务不使用
├── train_map_lmdb/      # 本任务不使用
├── val_map_lmdb/        # 本任务不使用
├── train_occ_lmdb/      # 本任务不使用
├── val_occ_lmdb/        # 本任务不使用
└── meta/                # 元数据目录（map GT 在线生成时读取）
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | Float 优化器 | 学习率策略 | 损失函数 |
|---|---|---|---|---|---|---|
| 8 | 2 | 24 | 100 | AdamW（lr=4e-4，backbone lr_mult=0.1，weight_decay=0.01） | CosineAnnealingLrUpdater（warmup_lr_ratio=1/3）<br>QAT 阶段用 StepDecayLrUpdater | BevFormerCriterion |

本任务无跨任务预训练依赖，Float 训练从零开始。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。QAT 加载 Calibration 产出的 `calibration-checkpoint-best.pth.tar`。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage float
# 产物：./tmp_models/bevformer_tiny_resnet50_detection/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage calibration
# 产物：./tmp_models/bevformer_tiny_resnet50_detection/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage qat
# 产物：./tmp_models/bevformer_tiny_resnet50_detection/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/bevformer_tiny_resnet50_detection/deploy.py
# 产物：./tmp_models/bevformer_tiny_resnet50_detection/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/bevformer_tiny_resnet50_detection/deploy.py
# 产物：./tmp_models/bevformer_tiny_resnet50_detection/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在量化模型基础上重新执行导出与编译。修改 `configs/bevformer_tiny_resnet50_detection/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/bevformer_tiny_resnet50_detection \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage hbm
```

评测指标为 NDS，评测集为 nuScenes val。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像（保存到 `./demo/bevformer_tiny_resnet50_detection/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/bevformer_tiny_resnet50_detection/）
python3 tools/gen_demo.py \
  --config configs/bevformer_tiny_resnet50_detection

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/bevformer_tiny_resnet50_detection \
  --stage float
```

可视化结果保存到 `./demo/bevformer_tiny_resnet50_detection/`。
