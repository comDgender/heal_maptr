# SparseBEV + HENet-tiny

`bev_sparse_henet_tinym` 是基于 HENet-tiny 骨干 + SparseBEVHead 检测头的 BEV 3D 检测模型，采用 MergeDataset 融合相机 base lmdb 与 lidar lmdb，覆盖「浮点训练 → 量化校准 → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/bev_sparse_henet_tinym/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| SparseBEV | 6 相机多视图图像 `(B,6,3,256,704)` + lidar 点云 `(B,N,5)` | HENet-tiny | MMFPN + DenseDepthNet | 3D 检测框 `(B,N,cls+reg)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb`）。指标：NDS / mAP。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5414 | 0.5319 | — | 0.5307 |
|  | mAP | 0.4338 | 0.4265 | — | 0.4235 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 `—`）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 10.43 | 98.46 | 72.80 |
| J6P | 7.38 | 527.73 | 85.90 |
| J6B | 23.16 | 44.28 | 71.00 |

编译产物路径：
- HBIR：`./tmp_models/bev_sparse_henet_tinym/qat.bc`
- HBM：`./tmp_models/bev_sparse_henet_tinym/compile/model.hbm`

---

## 模型介绍

### 核心设计

SparseBEV 以稀疏 query 方式在 BEV 空间进行 3D 检测：HENet-tiny 提取多视图相机特征，经 `MMFPN` 融合多尺度特征，`DenseDepthNet` 利用 lidar 点云生成密集深度监督辅助训练；`SparseBEVHead` 以 `MemoryBank` + `SparseBEVEncoder`（含 `AsymmetricFFN`、`DeformableFeatureAggregation`、`SparseBEVRefinementModule`）逐层 refine 稀疏 query 并输出检测框。

- **任务类型**：BEV 3D 目标检测（BEV 3D Object Detection）。
- **backbone**：HENet-tiny（`type=HENet`，`in_channels=3`，4 stage `[4,3,8,6]` block，`embed_dims=[64,128,192,384]`，`GroupDWCB`/`AltDWCB`/`DWCB` block，`S2DDown` 下采样；相机分支多视图特征提取）。
- **neck**：`MMFPN`（`in_strides=[2,4,8,16,32]`→`out_strides=[4,8,16,32]`，多尺度特征融合）+ `DenseDepthNet`（利用 lidar 点云生成密集深度监督辅助训练，`num_depth_layers=3`）。
- **检测头**：`SparseBEVHead`（`MemoryBank` + `SparseBEVEncoder` + `SparseBEVEncoder`，含 `AsymmetricFFN`、`DeformableFeatureAggregation`、`SparseBEVRefinementModule`，逐层 refine 稀疏 query，`num_classes=10`，`num_decoder=6`，`num_anchors=384`）。
- **损失函数**：`FocalLoss`（cls）+ `L1Loss`（reg）+ `CrossEntropyLoss`（cns）+ `GaussianFocalLoss`（yns）。
- **模型输入**：6 相机多视图图像 `(B,6,3,256,704)` + lidar 点云 `(B,N,5)`（`load_dim=5`，`use_dim=[0,1,2,4]`，`num_lidar_sweeps=0`，dense depth GT 辅助）。
- **模型输出**：10 类 3D 检测框（nuScenes 10 类，`num_classes=10`），`(B,N,cls+reg)`。

### 官方 repo 与论文链接

官方repo: https://github.com/MCG-NJU/SparseBEV 
论文: https://arxiv.org/abs/2308.09244

注：相机骨干 HENet 为 HEAL 自研，官方 repo backbone 不同。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14074

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 nuScenes 打包后的 base lmdb 与 lidar lmdb（经 `MergeDataset` 融合）。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb（`NuscenesBevDataset`） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb（`NuscenesBevDataset`） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | 训练 lidar lmdb（`NuscenesLidarDataset`，dense depth GT） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | 评测 lidar lmdb（`NuscenesLidarDataset`） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

#### 官方下载

nuScenes 数据集官方页面：<https://www.nuscenes.org/download>（需注册下载）。需下载 **v1.0-trainval** 全量数据集，解压后含 `samples/`（标注关键帧，6 相机）、`sweeps/`（中间帧）、`maps/`（地图扩张）、`v1.0-trainval/`（元数据 json：annotation、scene、sample、ego_pose 等）。

下载后将上述内容解压到原始数据目录 `./tmp_data/orig_data/nuscenes/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# 将官方解压内容放入该目录
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
```

#### 解压后目录结构

```
./tmp_data/orig_data/nuscenes/
├── samples/              # 标注关键帧图像（6 相机子目录）
├── sweeps/               # 中间帧图像（6 相机子目录）
├── maps/                 # 地图扩张（expansion + png）
└── v1.0-trainval/        # 元数据 json（annotation、scene、sample、ego_pose 等）
```

#### 打包命令

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包（禁止运行 create_data_*.py）。

nuScenes 共有四个 packer，依赖顺序：**base 先**；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际使用 **base lmdb + lidar lmdb**，需执行 base packer 与 lidar packer 各打 train/val 一次。未使用的 map packer（`nuscenes_map_packer.py` / `NuscenesMapGtPacker`）用于在线地图矢量化任务打包 map lmdb，occ packer（`nuscenes_occ_packer.py` / `NuscenesOccPacker`）用于占用预测任务打包 occ lmdb（需 Occ3D 标签），本任务均不使用。

`NuscenesPacker`（base）打 base 相机 lmdb（`{split}_lmdb`）：

```bash
# 打 base train split
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# 打 base val split
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

`NuscenesLidarPacker`（lidar）打 lidar 点云 lmdb（`{split}_lidar_lmdb`）：

```bash
# 打 lidar train split
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20

# 打 lidar val split
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/nuscenes`（含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/nuscenes/v1.0-trainval`，与 config 的 `data_rootdir` 对应。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--split-name`：`train` 或 `val`，分别打训练集与验证集。
- `--version v1.0-trainval`：nuScenes 数据集版本（packer 默认 `v1.0-mini`，须显式覆盖为 `v1.0-trainval`）。
- `--num-workers`：并行打包进程数（默认 20）。
- `--max-sweeps`：base/lidar packer 最大 sweep 数（默认 10）。
- `--load-dim`：lidar packer 原始点云维度（默认 5）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（本任务实际使用 `v1.0-trainval/` 下的 base + lidar lmdb）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # 训练 base lmdb          ← 本任务使用
│   ├── val_lmdb/            # 评测 base lmdb          ← 本任务使用
│   ├── train_lidar_lmdb/    # 训练 lidar lmdb         ← 本任务使用
│   ├── val_lidar_lmdb/      # 评测 lidar lmdb         ← 本任务使用
│   ├── train_map_lmdb/      # 训练 map lmdb（本任务不使用）
│   ├── val_map_lmdb/        # 评测 map lmdb（本任务不使用）
│   ├── train_occ_lmdb/      # 训练 occ lmdb（本任务不使用）
│   └── val_occ_lmdb/        # 评测 occ lmdb（本任务不使用）
├── meta/                    # 预计算地图扩展 meta 目录
└── osm/                     # OSM 矢量地图（map 类任务用）
```

可用可视化脚本抽样检查打包结果是否正确。base lmdb 用 `nuscenes_viz.py`：

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

lidar lmdb 用 `nuscenes_lidar_viz.py`（打包校验，复用 `heal/visualize/lidar_det.py` 的 `lidar_det_visualize`，画 BEV 点云 + 3D GT 框，无需模型）：

```bash
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar
```

可视化 PNG 保存到 `--save-dir`（默认 `WORKSPACE/nuscenes_lidar_viz`，须为本地可写目录；`tmp_data/heal_data` 是只读 bucket，不能作为保存目标），可人工核对点云与 GT 框是否对齐。

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | Float 优化器 | 学习率策略 | 损失函数 |
|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 4 | 100 | 12 | AdamW（`lr=3e-4`，`weight_decay=1e-3`）<br>backbone 单独 `lr=3e-4` | `CosLrUpdater`（`warmup_lr_ratio=1/3`，`min_lr_ratio=1e-3`） | `FocalLoss` + `L1Loss` + `CrossEntropyLoss` + `GaussianFocalLoss` |

本任务 Float 训练从零开始，无跨任务预训练依赖。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。本任务无 QAT 阶段。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/bev_sparse_henet_tinym \
  --stage float
# 产物：./tmp_models/bev_sparse_henet_tinym/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/bev_sparse_henet_tinym \
  --stage calibration
# 产物：./tmp_models/bev_sparse_henet_tinym/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/bev_sparse_henet_tinym/deploy.py
# 产物：./tmp_models/bev_sparse_henet_tinym/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/bev_sparse_henet_tinym/deploy.py
# 产物：./tmp_models/bev_sparse_henet_tinym/compile/model.hbm
```

> 编译时 `input_source="pyramid, ddr, ddr, ddr"`（相机图像走 pyramid，lidar 辅助走 ddr），`enable_vpu` 经 qconfig 定义。本任务 deploy.py 无 `enable_vpu` 显式行（经 qconfig）。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration，只需在量化模型基础上重新执行导出与编译。修改 `configs/bev_sparse_henet_tinym/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/bev_sparse_henet_tinym \
  --stage hbm
```

评测指标为 NDS 与 mAP，评测集为 nuScenes val（`val_lmdb` + `val_lidar_lmdb`，`val_nuscenes_metric`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的样本（保存到 `./demo/bev_sparse_henet_tinym/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/bev_sparse_henet_tinym/）
python3 tools/gen_demo.py \
  --config configs/bev_sparse_henet_tinym

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/bev_sparse_henet_tinym \
  --stage float
```

可视化结果保存到 `./demo/bev_sparse_henet_tinym/`。
