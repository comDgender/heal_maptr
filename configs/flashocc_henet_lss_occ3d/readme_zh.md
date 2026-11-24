# FlashOcc + HENet + LSS

`flashocc_henet_lss_occ3d` 是基于 HENet + LSS（Lift-Splat-Shoot）的 BEV 占用预测模型，采用 MergeDataset 融合相机 base lmdb 与 occ lmdb（Occ3D-nuScenes 标签），覆盖「浮点训练 → 量化校准 → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/flashocc_henet_lss_occ3d/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| FlashOcc | 6 相机多视图图像 `(B,6,3,512,960)` | HENet | FPN + LSSTransformer + BevEncoder | 占用栅格 `(B,C,H,W)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_occ_lmdb`）。指标：Occ mIoU（MeanIOU，`ignore_index=17`，`num_classes=18`）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Occ mIoU (MeanIOU) | 0.3664 | 0.3688 | — | 0.369 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 `—`）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 7.61 | 136.01 | 73.40 |
| J6P | 5.64 | 730.55 | 77.60 |
| J6B | 23.63 | 43.17 | 81.60 |

编译产物路径：
- HBIR：`./tmp_models/flashocc_henet_lss_occ3d/qat.bc`
- HBM：`./tmp_models/flashocc_henet_lss_occ3d/compile/model.hbm`

---

## 模型介绍

### 核心设计

FlashOcc 采用 LSS（Lift-Splat-Shoot）视角变换：HENet 提取多视图相机特征，`LSSTransformer` 预测深度分布并将 2D 特征 lift 到 3D 体素空间（`depth=45`，`num_points=10`，`bev_size=(40,40,0.625)`），经 `BevEncoder`（`BiFPN`）融合，`FlashOccDetDecoder`/`BEVOCCHead2D` 输出 18 类 3D 占用预测。

- **任务类型**：BEV 3D 占用预测（BEV Occupancy Prediction）。
- **backbone**：HENet（`type=HENet`，`depth=45`，`num_points=10`），提取多视图相机特征。
- **neck**：`FPN` + `LSSTransformer`（Lift-Splat-Shoot 视角变换，`bev_size=(40,40,0.625)`，`grid_size=(128,128)`）+ `BevEncoder`（`BiFPN`）。
- **占用头**：`FlashOccDetDecoder`（`BEVOCCHead2D`，`num_classes=18`，`ignore_index=17`）。
- **损失函数**：`CrossEntropyLoss`（occ seg）。
- **模型输入**：6 视图相机图像 `(B,6,3,512,960)`（`data_shape=(3,512,960)`，`MergeDataset(Bev+Occ)` 融合 base lmdb 与 occ lmdb）。
- **模型输出**：18 类 3D 占用栅格 `(B,C,H,W)`（Occ3D，`occ3d_seg_class` 18 类，含 `others`/`ignore_index=17`）。

### 官方 repo 与论文链接

官方repo: https://github.com/Yzichen/FlashOCC
论文: https://arxiv.org/abs/2311.12058

注：相机骨干 HENet 为 HEAL 自研，官方 repo backbone 不同。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14099

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 nuScenes 打包后的 base lmdb 与 occ lmdb（经 `MergeDataset` 融合，按 token 对齐）。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb（`NuscenesBevDataset`） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb（`NuscenesBevDataset`） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | 训练 occ lmdb（`NuscenesOccDataset`，Occ3D 标签） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | 评测 occ lmdb（`NuscenesOccDataset`，Occ3D 标签） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

#### 官方下载

nuScenes 数据集官方页面：<https://www.nuscenes.org/download>（需注册下载）。需下载 **v1.0-trainval** 全量数据集，解压后含 `samples/`（标注关键帧，6 相机）、`sweeps/`（中间帧）、`maps/`（地图扩张）、`v1.0-trainval/`（元数据 json：annotation、scene、sample、ego_pose 等）。

**本任务额外需要 Occ3D-nuScenes 占用标签**：Occ3D-nuScenes 官方仓库 <https://github.com/Tsinghua-MARS-Lab/Occ3D>，需下载 `occ3d/gts/<scene>/<token>/labels.npz` 占用标签，放置到 nuScenes 根下的 `occ3d/gts/` 目录（即 `./tmp_data/orig_data/nuscenes/occ3d/gts/`），occ packer 通过 `--occ-gt-dir occ3d/gts` 读取。

下载后将上述内容解压到原始数据目录 `./tmp_data/orig_data/nuscenes/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# 将官方解压内容放入该目录
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
# 额外放置 Occ3D 占用标签
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# 将 Occ3D-nuScenes 的 occ3d/gts/<scene>/<token>/labels.npz 放入该目录
# ./tmp_data/orig_data/nuscenes/occ3d/gts/<scene>/<token>/labels.npz
```

#### 解压后目录结构

```
./tmp_data/orig_data/nuscenes/
├── samples/              # 标注关键帧图像（6 相机子目录）
├── sweeps/               # 中间帧图像（6 相机子目录）
├── maps/                 # 地图扩张（expansion + png）
├── v1.0-trainval/        # 元数据 json（annotation、scene、sample、ego_pose 等）
└── occ3d/
    └── gts/              # Occ3D-nuScenes 占用标签
        └── <scene>/
            └── <token>/
                └── labels.npz   # 占用语义标签（18 类，ignore_index=17）
```

#### 打包命令

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包（禁止运行 create_data_*.py）。

nuScenes 共有四个 packer，依赖顺序：**base 先**；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际使用 **base lmdb + occ lmdb**，需先执行 base packer，再执行 occ packer（occ 必须在 base 之后跑，且 version/split 与 base 一致）。未使用的 lidar packer（`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`）用于 lidar 点云任务打包 lidar lmdb，map packer（`nuscenes_map_packer.py` / `NuscenesMapGtPacker`）用于在线地图矢量化任务打包 map lmdb，本任务均不使用。

`NuscenesPacker`（base）打 base 相机 lmdb（`{split}_lmdb`）：

```bash
# 打 base train split（须先于 occ）
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# 打 base val split（须先于 occ）
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

`NuscenesOccPacker`（occ）打占用预测 lmdb（`{split}_occ_lmdb`），**必须在 base 之后跑且 version/split 与 base 一致**，通过 `--occ-gt-dir occ3d/gts` 读取 Occ3D-nuScenes 占用标签：

```bash
# 打 occ train split（须在 base train 之后，version/split 一致）
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20

# 打 occ val split（须在 base val 之后，version/split 一致）
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/nuscenes`（含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`、`occ3d/gts/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/nuscenes/v1.0-trainval`，与 config 的 `data_rootdir` 对应。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--split-name`：`train` 或 `val`，分别打训练集与验证集。
- `--version v1.0-trainval`：nuScenes 数据集版本（packer 默认 `v1.0-mini`，须显式覆盖为 `v1.0-trainval`）；occ 须与 base 一致。
- `--occ-gt-dir occ3d/gts`：occ packer 读取的 Occ3D-nuScenes 占用标签目录（相对 `--src-data-dir`，即 `./tmp_data/orig_data/nuscenes/occ3d/gts/`），来自 <https://github.com/Tsinghua-MARS-Lab/Occ3D>。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（本任务实际使用 `v1.0-trainval/` 下的 base + occ lmdb）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # 训练 base lmdb          ← 本任务使用
│   ├── val_lmdb/            # 评测 base lmdb          ← 本任务使用
│   ├── train_lidar_lmdb/    # 训练 lidar lmdb（本任务不使用）
│   ├── val_lidar_lmdb/      # 评测 lidar lmdb（本任务不使用）
│   ├── train_map_lmdb/      # 训练 map lmdb（本任务不使用）
│   ├── val_map_lmdb/        # 评测 map lmdb（本任务不使用）
│   ├── train_occ_lmdb/      # 训练 occ lmdb           ← 本任务使用
│   └── val_occ_lmdb/        # 评测 occ lmdb           ← 本任务使用
├── meta/                    # 预计算地图扩展 meta 目录
└── osm/                     # OSM 矢量地图（map 类任务用）
```

可用可视化脚本抽样检查打包结果是否正确。base lmdb 用 `nuscenes_viz.py`，occ lmdb 用 `nuscenes_occ_viz.py`：

```bash
# base lmdb 可视化
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50

# occ lmdb 可视化
python3 tools/datasets/nuscenes_occ_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | Float 优化器 | 学习率策略 | 损失函数 |
|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 4（评测 `val_batch_size_per_gpu=1`） | 150 | 100 | AdamW（`lr=1e-4`，`weight_decay=1e-2`） | `CosLrUpdater`（`warmup_lr_ratio=1/3`，`stop_lr=2e-4*1e-3`） | `CrossEntropyLoss`（occ seg，`ignore_index=17`） |

本任务 Float 训练从零开始，无跨任务预训练依赖。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。本任务无 QAT 阶段。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage float
# 产物：./tmp_models/flashocc_henet_lss_occ3d/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage calibration
# 产物：./tmp_models/flashocc_henet_lss_occ3d/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/flashocc_henet_lss_occ3d/deploy.py
# 产物：./tmp_models/flashocc_henet_lss_occ3d/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/flashocc_henet_lss_occ3d/deploy.py
# 产物：./tmp_models/flashocc_henet_lss_occ3d/compile/model.hbm
```

> 导出时 `enable_vpu=False`，编译输入 `input_source=["pyramid","ddr","ddr"]`（相机图像走 pyramid，occ 辅助走 ddr）。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration，只需在量化模型基础上重新执行导出与编译。修改 `configs/flashocc_henet_lss_occ3d/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage hbm
```

评测指标为 Occ mIoU（`MeanIOU`，`ignore_index=17`，`num_classes=18`），评测集为 nuScenes val（`val_lmdb` + `val_occ_lmdb`，`val_metrics`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的样本（保存到 `./demo/flashocc_henet_lss_occ3d/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/flashocc_henet_lss_occ3d/）
python3 tools/gen_demo.py \
  --config configs/flashocc_henet_lss_occ3d

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/flashocc_henet_lss_occ3d \
  --stage float
```

可视化结果保存到 `./demo/flashocc_henet_lss_occ3d/`。
