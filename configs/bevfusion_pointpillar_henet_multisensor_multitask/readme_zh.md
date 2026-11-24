# BevFusion+PointPillar+HENet 多传感器多任务

`bevfusion_pointpillar_henet_multisensor_multitask` 是基于 BevFusion（CenterPointDetector + 多传感器融合）的多任务模型，联合完成 3D 目标检测（det）与占据网格预测（occ）。相机分支用 HENet 提取 BEV 特征，激光雷达分支用 PointPillars 提取 BEV 特征，两者融合后分别送入检测头与占据头。

模型配置目录：`configs/bevfusion_pointpillar_henet_multisensor_multitask/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| BevFusion | 6 相机多视图图像 `(B,6,3,512,960)` + lidar 点云 `(B,N,5)` | PointPillar（lidar）+ HENet（相机） | FPN（相机）+ SECONDNeck（lidar） | det 检测框 `(B,N,cls+reg)`；occ 占用栅格 `(B,C,H,W)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb` + `val_occ_lmdb`）。指标：NDS/mAP（det，`NuscenesMetric`）+ Occ mIoU（`MeanIOU`，`ignore_index=17`）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.6421 | 0.6301 | — | 0.6294 |
|  | mAP | 0.5825 | 0.5724 | — | 0.5726 |
|  | Occ mIoU | 0.5187 | 0.52 | — | 0.5206 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 `—`）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 23.68 | 49.91 | 186.40 |
| J6P | 16.75 | 287.11 | 195.00 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/qat.bc`
- HBM：`./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/compile/model.hbm`

---

## 模型介绍

### 核心设计

BevFusion 多传感器多任务模型以双分支提取 BEV 特征：相机分支用 HENet 提取多视角特征后经 BevFormer 的 ViewTransformer 转换到 BEV；激光雷达分支用 PointPillars（`PillarFeatureNet` + `PointPillarScatter` + `SECONDNeck`）体素化点云后提取 BEV 特征。两路 BEV 特征融合后分别送入 CenterPoint 检测头（3D 目标检测）与占据头（语义占据预测），实现 det+occ 联合训练。

- **任务类型**：多传感器多任务融合（3D 目标检测 + 占据网格预测）。
- **backbone**：相机 HENet（多视角特征提取）+ 激光雷达 PointPillars（`PillarFeatureNet` + `PointPillarScatter`）。
- **neck**：相机 FPN + 激光雷达 SECONDNeck。
- **检测头**：CenterPoint 检测头，输出 10 类 3D 检测框 + 速度（`num_classes=10`）。
- **占据头**：语义占据预测，18 类（`num_classes_occ=18`）。
- **BEV 范围**：`bev_size=(51.2, 51.2, 0.8)`，`bev_size_occ=(40, 40)`；`point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`。
- **模型输入**：6 相机多视图图像 (B,6,3,512,960) + lidar 点云 (B,N,D)。
- **模型输出**：det 的 3D 检测框 + occ 的占据网格语义。

**部署说明**：HBIR 导出开启 `enable_vpu=True`，编译时 `input_source="ddr, ddr, pyramid, ddr, ddr, ddr, ddr"`（ddr 优先，激光雷达特征从 DDR 读入）。

### 官方 repo 与论文链接

官方repo: https://github.com/mit-han-lab/bevfusion
论文: https://arxiv.org/abs/2205.13542

注：相机分支为 BevFormer、激光雷达分支为 PointPillars/CenterPoint；相机骨干 HENet 为 HEAL 自研，官方 repo backbone 不同。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14092

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 nuScenes 打包后的 base lmdb + lidar lmdb + occ lmdb（`MergeDataset(Bev+Lidar+Occ)`）。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb（相机 + 标注） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | 训练 lidar lmdb（激光雷达点云） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | 评测 lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | 训练 occ lmdb（占据标签） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | 评测 occ lmdb（占据标签） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

nuScenes 无需额外预处理，四个 packer（base/lidar/map/occ）自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。本任务实际使用 **base + lidar + occ lmdb**（不使用 map lmdb）。

#### 官方下载

nuScenes 数据集官方页面：<https://www.nuscenes.org/download>（v1.0-trainval，需注册）。需下载完整 trainval 包，含以下内容：

- `samples/`：标注关键帧（6 相机 + 激光雷达）
- `sweeps/`：中间帧
- `maps/`：地图扩张
- `v1.0-trainval/`：元数据（annotation、scene、sample、ego_pose 等 json）

下载后将上述内容解压到原始数据目录 `./tmp_data/orig_data/nuscenes/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# 将 samples/ sweeps/ maps/ v1.0-trainval/ 解压到该目录
```

本任务含 occ 标签，需额外下载 Occ3D-nuScenes 标签（<https://github.com/Tsinghua-MARS-Lab/Occ3D>），将 `occ3d/gts/<scene>/<token>/labels.npz` 放置到 `./tmp_data/orig_data/nuscenes/occ3d/gts/`：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# 将 Occ3D-nuScenes 的 labels.npz 按场景/token 放入该目录
```

#### 解压后目录结构

```
./tmp_data/orig_data/nuscenes/
├── samples/                 # 标注关键帧（6 相机 + 激光雷达）
├── sweeps/                  # 中间帧
├── maps/                    # 地图扩张
├── v1.0-trainval/           # 元数据 json
└── occ3d/
    └── gts/                 # Occ3D-nuScenes 占据标签（labels.npz）
```

#### 打包命令

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。依赖顺序：base 先；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际使用 **base + lidar + occ lmdb**，需依次执行 base、lidar、occ 三个 packer（occ 在 base 之后）。

**base packer**（`nuscenes_packer.py`，`NuscenesPacker`）→ `{split}_lmdb`：

```bash
# 打 train base lmdb
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --num-workers 20 \
  --max-sweeps 10

# 打 val base lmdb
python3 tools/datasets/nuscenes_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --num-workers 20 \
  --max-sweeps 10
```

**lidar packer**（`nuscenes_lidar_packer.py`，`NuscenesLidarPacker`）→ `{split}_lidar_lmdb`：

```bash
# 打 train lidar lmdb
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --max-sweeps 10 \
  --load-dim 5 \
  --use-dim 0 1 2 3 4

# 打 val lidar lmdb
python3 tools/datasets/nuscenes_lidar_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --max-sweeps 10 \
  --load-dim 5 \
  --use-dim 0 1 2 3 4
```

**occ packer**（`nuscenes_occ_packer.py`，`NuscenesOccPacker`）→ `{split}_occ_lmdb`（须在 base 之后，version/split 一致）：

```bash
# 打 train occ lmdb
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20

# 打 val occ lmdb
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
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/nuscenes`（含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/nuscenes/v1.0-trainval`，与 config 的 `data_rootdir` 一致。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--version v1.0-trainval`：nuScenes 版本（packer 默认 v1.0-mini，须显式覆盖为 v1.0-trainval）。
- `--split-name`：`train` 或 `val`，分别打训练集与验证集。
- `--max-sweeps 10`（base/lidar packer）：最大 sweep 数。
- `--load-dim 5` / `--use-dim 0 1 2 3 4`（lidar packer）：点云加载/使用维度，与 `common.py` 的 `lidar_load_dim=5`、`lidar_use_dim=[0,1,2,3,4]` 一致。
- `--occ-gt-dir occ3d/gts`（occ packer）：Occ3D-nuScenes 标签目录（默认值），需用户额外下载放置。
- `--num-workers`：并行打包进程数（默认 20）。

> **四个 packer 说明**（本任务使用 base + lidar + occ）：
> - **base**（`nuscenes_packer.py` / `NuscenesPacker`）→ `{split}_lmdb`：相机 + 标注基础数据，**本任务使用**。
> - **lidar**（`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`）→ `{split}_lidar_lmdb`：激光雷达点云，**本任务使用**。
> - **map**（`nuscenes_map_packer.py` / `NuscenesMapGtPacker`）→ `{split}_map_lmdb`：矢量化地图 GT，本任务不使用。
> - **occ**（`nuscenes_occ_packer.py` / `NuscenesOccPacker`）→ `{split}_occ_lmdb`：Occ3D 占据标签，**本任务使用**。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（标注本任务实际用哪些）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base 训练 lmdb      ← 本任务使用
│   ├── val_lmdb/            # base 评测 lmdb      ← 本任务使用
│   ├── train_lidar_lmdb/    # lidar 训练 lmdb    ← 本任务使用
│   ├── val_lidar_lmdb/      # lidar 评测 lmdb    ← 本任务使用
│   ├── train_map_lmdb/      # map 训练 lmdb        （本任务不使用）
│   ├── val_map_lmdb/        # map 评测 lmdb        （本任务不使用）
│   ├── train_occ_lmdb/      # occ 训练 lmdb      ← 本任务使用
│   └── val_occ_lmdb/        # occ 评测 lmdb      ← 本任务使用
├── meta/                    # 地图扩展预计算目录    （本任务不使用）
└── osm/                     # OpenStreetMap 矢量地图（本任务不使用）
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
# base lmdb 校验
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb \
  --viz-num 50

# occ lmdb 校验
python3 tools/datasets/nuscenes_occ_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb \
  --viz-num 50

# lidar lmdb 校验（复用 heal/visualize/lidar_det.py，画 BEV 点云 + 3D GT 框，无需模型）
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar
```

可视化 PNG 保存到 `--save-dir`（默认 `WORKSPACE/nuscenes_lidar_viz`，须为本地可写目录；`tmp_data/heal_data` 是只读 bucket，不能作为保存目标），可人工核对点云与 GT 框是否对齐。

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | Float 优化器 |
|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 4 | 24 | （以 config 为准） | AdamW（`lr=4e-4`） |

本任务无 QAT 阶段。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage float
# 产物：./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage calibration
# 产物：./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask/deploy.py
# 产物：./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask/deploy.py
# 产物：./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/compile/model.hbm
```

> 导出时 `enable_vpu=True`，编译时 `input_source="ddr, ddr, pyramid, ddr, ddr, ddr, ddr"`（ddr 优先，激光雷达特征从 DDR 读入）。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在量化模型基础上重新执行导出与编译。修改 `configs/bevfusion_pointpillar_henet_multisensor_multitask/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage hbm
```

评测指标为 NDS/mAP（det）与 Occ mIoU（occ，`MeanIOU`，`ignore_index=17`），评测集为 nuScenes val（`val_lmdb` + `val_lidar_lmdb` + `val_occ_lmdb`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

#### 量化敏感性分析（可选）

本任务定义了 `quant_analysis_solver`，可逐层分析量化对精度的影响，辅助排查量化掉点来源：

```bash
python3 tools/quant_analysis.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask/quant_analysis.py
```

结果输出到 `./tmp_models/bevfusion_pointpillar_henet_multisensor_multitask/analysis/`。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的数据（保存到 `./demo/bevfusion_pointpillar_henet_multisensor_multitask/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/bevfusion_pointpillar_henet_multisensor_multitask \
  --stage float
```

可视化结果保存到 `./demo/bevfusion_pointpillar_henet_multisensor_multitask/`。
