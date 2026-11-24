# CenterPoint (PointPillars)

`centerpoint_pointpillar` 是基于 PointPillars 的 CenterPoint 3D 检测模型，采用 `CenterPointDetector`（含 CBGS 数据增强），仅使用 lidar lmdb，覆盖「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/centerpoint_pointpillar/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| CenterPoint | lidar 点云 `(B,N,5)` | PointPillarScatter | SECONDNeck | 3D 检测框 `(B,N,cls+reg)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb`）。指标：NDS / mAP。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5865 | 0.5703 | 0.5853 | 0.5846 |
|  | mAP | 0.474 | 0.4487 | 0.4699 | 0.4693 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 9.20 | 183.98 | 51.10 |
| J6P | 7.60 | 831.31 | 47.20 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/centerpoint_pointpillar/qat.bc`
- HBM：`./tmp_models/centerpoint_pointpillar/compile/model.hbm`

---

## 模型介绍

### 核心设计

CenterPoint（PointPillars 变体）将 lidar 点云体素化为柱体，经 `PillarFeatureNet` 学习柱特征并 `PointPillarScatter` 散布为 2D 伪图像，由 `SECONDNeck` 提取多尺度特征，`CenterPointHead` 以 anchor-free 方式回归框中心、尺寸、朝向、速度，输出 3D 检测框。训练时启用 CBGS（Class-Balanced Grouping and Sampling）数据增强。

- **任务类型**：3D 目标检测（3D Object Detection，lidar）。
- **backbone**：`PillarFeatureNet`（`num_input_features=5`，`num_filters=(64,)`，学习柱特征）+ `PointPillarScatter`（`num_input_features=64`，`use_horizon_pillar_scatter=True`，将柱特征散布为 2D 伪图像）。
- **neck**：`SECONDNeck`（`in_feature_channel=64`，`down_layer_nums=[3,5,5]`，`down_layer_channels=[64,128,256]`，`up_layer_channels=[128,128,128]`，提取多尺度特征）。
- **检测头**：`CenterPointHead`（anchor-free，`common_heads=dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2))`，`with_velocity=True`）。
- **损失函数**：`CenterPointLoss`（`GaussianFocalLoss` cls + `L1Loss` reg）。
- **模型输入**：lidar 点云 `(B,N,5)`（`point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`，`voxel_size=[0.2,0.2,8]`，`max_num_points=20`，`max_voxels=(30000,40000)`，`load_dim=5`，`use_dim=[0,1,2,3,4]`，`num_sweeps=9`）。
- **模型输出**：10 类 3D 检测框（nuScenes 10 类，`class_names` = `[car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone]`），`(B,N,cls+reg)`。

### 官方 repo 与论文链接

官方repo: https://github.com/tianweiy/CenterPoint
论文: https://arxiv.org/abs/2006.11275

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14088
---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务仅使用 nuScenes 打包后的 lidar lmdb（`NuscenesLidarDataset`，训练时经 CBGS 增强）。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval"` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | 训练 lidar lmdb（`NuscenesLidarDataset` + `CBGSDataset`） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | 评测 lidar lmdb（`NuscenesLidarDataset`） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

#### 官方下载

nuScenes 数据集官方页面：<https://www.nuscenes.org/download>（需注册下载）。需下载 **v1.0-trainval** 全量数据集，解压后含 `samples/`（标注关键帧，6 相机）、`sweeps/`（中间帧）、`maps/`（地图扩张）、`v1.0-trainval/`（元数据 json：annotation、scene、sample、ego_pose 等）。本任务（lidar 检测）实际只使用 `sweeps/`（lidar 点云）与 `v1.0-trainval/` 元数据，但 packer 需要完整 nuScenes 目录结构。

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
├── sweeps/               # 中间帧（含 lidar 点云 LIDAR_TOP）
├── maps/                 # 地图扩张（expansion + png）
└── v1.0-trainval/        # 元数据 json（annotation、scene、sample、ego_pose 等）
```

#### 打包命令

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包（禁止运行 create_data_*.py）。

nuScenes 共有四个 packer：
- **base** `nuscenes_packer.py`（`NuscenesPacker`）打 base 相机 lmdb（`{split}_lmdb`），用于相机检测任务。
- **lidar** `nuscenes_lidar_packer.py`（`NuscenesLidarPacker`）打 lidar 点云 lmdb（`{split}_lidar_lmdb`），用于 lidar 检测任务。
- **map** `nuscenes_map_packer.py`（`NuscenesMapGtPacker`）打地图矢量化 lmdb（`{split}_map_lmdb`），用于在线地图任务（PV seg mask 不打包，训练时在线生成）。
- **occ** `nuscenes_occ_packer.py`（`NuscenesOccPacker`）打占用预测 lmdb（`{split}_occ_lmdb`），用于 occ 任务（须在 base 之后跑，version/split 一致；需 Occ3D 标签）。

依赖顺序：**base 先**；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。

**本任务实际仅使用 `{split}_lidar_lmdb`，base/map/occ 可不打包**（仅需执行 lidar packer 即可）。以下给出本任务实际需要的 lidar packer 完整命令；其余 packer 命令若其他任务需要可参考本目录其他 readme。

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
- `--max-sweeps`：lidar packer 最大 sweep 数（默认 10）。
- `--load-dim`：lidar packer 原始点云维度（默认 5）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（本任务实际仅使用 `v1.0-trainval/` 下的 lidar lmdb，base/map/occ lmdb 未打包即不存在）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # 训练 base lmdb（本任务不使用）
│   ├── val_lmdb/            # 评测 base lmdb（本任务不使用）
│   ├── train_lidar_lmdb/    # 训练 lidar lmdb         ← 本任务使用
│   ├── val_lidar_lmdb/      # 评测 lidar lmdb         ← 本任务使用
│   ├── train_map_lmdb/      # 训练 map lmdb（本任务不使用）
│   ├── val_map_lmdb/        # 评测 map lmdb（本任务不使用）
│   ├── train_occ_lmdb/      # 训练 occ lmdb（本任务不使用）
│   └── val_occ_lmdb/        # 评测 occ lmdb（本任务不使用）
├── meta/                    # 预计算地图扩展 meta 目录
└── osm/                     # OSM 矢量地图（map 类任务用）
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

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | QAT 训练 epoch 数 | Float 优化器 | QAT 优化器 | 学习率策略 | 损失函数 |
|---|---|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 4 | 20 | 100 | 10 | AdamW（`lr=2e-4`，`weight_decay=1e-2`） | SGD（`lr=2e-4`，`momentum=0.9`，`weight_decay=0.0`） | `CosLrUpdater`（Float） | `CenterPointLoss`（`GaussianFocalLoss` cls + `L1Loss` reg） |

本任务 Float 训练从零开始，无跨任务预训练依赖。Calibration 加载本任务 Float 产物，QAT 加载本任务 Calibration 产物，串行链。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/centerpoint_pointpillar \
  --stage float
# 产物：./tmp_models/centerpoint_pointpillar/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/centerpoint_pointpillar \
  --stage calibration
# 产物：./tmp_models/centerpoint_pointpillar/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/centerpoint_pointpillar \
  --stage qat
# 产物：./tmp_models/centerpoint_pointpillar/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/centerpoint_pointpillar/deploy.py
# 产物：./tmp_models/centerpoint_pointpillar/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/centerpoint_pointpillar/deploy.py
# 产物：./tmp_models/centerpoint_pointpillar/compile/model.hbm
```

> 导出与编译时 `enable_vpu` 经 `common.py` 的 `enable_vpu=True` 变量传入（deploy.py 从 common 导入），编译输入 `input_source=["ddr","ddr"]`（lidar 点云从 DDR 读入，非 pyramid 图像输入），与图像类任务不同。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在量化模型基础上重新执行导出与编译。修改 `configs/centerpoint_pointpillar/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/centerpoint_pointpillar \
  --stage hbm
```

评测指标为 NDS 与 mAP，评测集为 nuScenes val（`val_lidar_lmdb`，`val_nuscenes_metric`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的样本（保存到 `./demo/centerpoint_pointpillar/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/centerpoint_pointpillar/）
python3 tools/gen_demo.py \
  --config configs/centerpoint_pointpillar

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/centerpoint_pointpillar \
  --stage float
```

可视化结果保存到 `./demo/centerpoint_pointpillar/`。
