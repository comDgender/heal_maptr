# MapTR+HENet (BevFormer)

`maptr_henet_tinym_bevformer` 是基于 MapTR + HENet 的在线矢量化地图构建模型，采用 BevFormer 视角变换。本任务 `use_lidar_gt=False`，map GT 由 `VectorizedLocalMap` 在线生成（使用 base lmdb + meta 目录 + osm 目录，无需打包 map lmdb）。

模型配置目录：`configs/maptr_henet_tinym_bevformer/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| MapTR | 6 相机多视图图像 `(B,6,3,480,800)` | HENet-tiny | FPN | 矢量化地图 `(B,L,P,2)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`）。指标：chamfer mAP（`NuscenesMapMetric`，`metric="chamfer"`，monitor_metric_key=`"MAP"`）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | chamfer mAP (MAP) | 0.6626 | 0.6588 | — | 0.6315 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 `—`）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 9.26 | 111.22 | 88.00 |
| J6P | 6.20 | 662.25 | 82.90 |
| J6B | 25.69 | 39.70 | 79.70 |

编译产物路径：
- HBIR：`./tmp_models/maptr_henet_tinym_bevformer/qat.bc`
- HBM：`./tmp_models/maptr_henet_tinym_bevformer/compile/model.hbm`

---

## 模型介绍

### 核心设计

MapTR 以 HENet 为相机骨干提取多视角特征，经 BevFormer 的单帧 ViewTransformer 与 BEV Encoder 转换到鸟瞰图特征；BEV 特征送入 MapTR 解码器，以固定点数（`fixed_ptsnum_per_pred_line=20`）的折线 query 预测矢量化地图元素（divider/ped_crossing/boundary）。本任务 `use_lidar_gt=False`，map GT 不从 lmdb 读取，而是由 `VectorizedLocalMap` 在线生成（`map_path=meta_rootdir`，`sd_map_path=osm`）。

- **任务类型**：在线矢量化地图构建（Online Vectorized Map Construction）。
- **backbone**：HENet-tiny（ImageNet 预训练），提取多视角相机特征。
- **neck**：FPN。
- **视角变换**：`SingleFrameBevFormerViewTransformer` + `SingleFrameBEVFormerEncoder`（`queue_length=1`/`test_queue_length=1`）。
- **地图元素**：`map_classes=[divider, ped_crossing, boundary]`，`fixed_ptsnum_per_gt_line=20`。
- **BEV 范围**：`use_lidar_gt=False` 走 else 分支，`point_cloud_range=[-30.0,-15.0,-10.0,30.0,15.0,10.0]`，`bev_h_=50`，`bev_w_=100`（bev 50×100）。
- **模型输入**：6 路相机图像单帧 `(B,6,3,480,800)`（`NuscenesMapDataset`，`use_lidar_gt=False`，map GT 由 `VectorizedLocalMap` 在线生成，不使用 map lmdb）。
- **模型输出**：3 类矢量化地图元素（divider/ped_crossing/boundary），每条线 20 个点。

### 官方 repo 与论文链接

官方repo: https://github.com/hustvl/MapTR
论文: https://arxiv.org/abs/2208.14437

注：相机骨干 HENet 为 HEAL 自研，官方 repo backbone 不同。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14100

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务 `use_lidar_gt=False`，实际使用 base lmdb + meta 目录 + osm 目录，无需打包 map lmdb（map GT 由 `VectorizedLocalMap` 在线生成，`map_path=meta_rootdir`、`sd_map_path=osm`）。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb |
| `./tmp_data/heal_data/nuscenes/meta` | 地图扩展预计算目录（map GT 在线生成依赖） |
| `./tmp_data/heal_data/nuscenes/osm` | OpenStreetMap 矢量地图（map GT 在线生成依赖） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

nuScenes 无需额外预处理，四个 packer（base/lidar/map/occ）自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。本任务实际使用 **base lmdb + meta 目录 + osm 目录**，无需打包 map lmdb。

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

本任务需 meta 目录（地图扩展预计算）与 osm 目录（OpenStreetMap 矢量地图）供 `VectorizedLocalMap` 在线生成 map GT。meta/osm 目录由 map packer 预计算阶段生成（见下方「打包命令」中的 map packer 说明）。

#### 解压后目录结构

```
./tmp_data/orig_data/nuscenes/
├── samples/                 # 标注关键帧（6 相机 + 激光雷达）
├── sweeps/                  # 中间帧
├── maps/                    # 地图扩张
└── v1.0-trainval/           # 元数据 json
```

#### 打包命令

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。依赖顺序：base 先；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际仅使用 base lmdb，无需打包 map lmdb（map GT 在线生成，仅需 meta 目录与 osm 目录）。

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

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/nuscenes`（含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/nuscenes/v1.0-trainval`，与 config 的 `data_rootdir` 一致。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--version v1.0-trainval`：nuScenes 版本（packer 默认 v1.0-mini，须显式覆盖为 v1.0-trainval）。
- `--split-name`：`train` 或 `val`，分别打训练集与验证集。
- `--max-sweeps 10`：最大 sweep 数。
- `--num-workers`：并行打包进程数（默认 20）。

> **map packer 说明**（本任务不使用 map lmdb，可不打包）：
>
> `nuscenes_map_packer.py`（`NuscenesMapGtPacker`）→ `{split}_map_lmdb`，用于离线预打包矢量化地图 GT。参数须与本任务 `common.py` 一致：`--map-path ./tmp_data/heal_data/nuscenes/meta`（meta 目录）、`--pc-range -30 -15 -10 30 15 10`（`use_lidar_gt=False` 的 else 分支）、`--bev-size 50 100`、`--fixed-ptsnum-per-line 20`、`--map-classes divider ped_crossing boundary`。PV seg mask 不打包，训练时在线生成。
>
> **本任务 `use_lidar_gt=False`，map GT 由 `VectorizedLocalMap` 在线生成，实际不使用 map lmdb**；但需 meta 目录（map packer 预计算阶段产出，`--map-path` 指向该目录）与 osm 目录（OpenStreetMap 矢量地图）。

> **四个 packer 说明**（本任务仅使用 base）：
> - **base**（`nuscenes_packer.py` / `NuscenesPacker`）→ `{split}_lmdb`：相机 + 标注基础数据，**本任务使用**。
> - **lidar**（`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`）→ `{split}_lidar_lmdb`：激光雷达点云，本任务不使用。
> - **map**（`nuscenes_map_packer.py` / `NuscenesMapGtPacker`）→ `{split}_map_lmdb`：矢量化地图 GT，本任务不使用（map GT 在线生成）。
> - **occ**（`nuscenes_occ_packer.py` / `NuscenesOccPacker`）→ `{split}_occ_lmdb`：Occ3D 占据标签，本任务不使用。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（标注本任务实际用哪些）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base 训练 lmdb      ← 本任务使用
│   ├── val_lmdb/            # base 评测 lmdb      ← 本任务使用
│   ├── train_lidar_lmdb/    # lidar 训练 lmdb      （本任务不使用）
│   ├── val_lidar_lmdb/      # lidar 评测 lmdb      （本任务不使用）
│   ├── train_map_lmdb/      # map 训练 lmdb        （本任务不使用，但需 meta/ 与 osm/）
│   ├── val_map_lmdb/        # map 评测 lmdb        （本任务不使用，但需 meta/ 与 osm/）
│   ├── train_occ_lmdb/      # occ 训练 lmdb        （本任务不使用）
│   └── val_occ_lmdb/        # occ 评测 lmdb        （本任务不使用）
├── meta/                    # 地图扩展预计算目录  ← 本任务使用（map GT 在线生成依赖）
└── osm/                     # OpenStreetMap 矢量地图 ← 本任务使用（map GT 在线生成依赖）
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
# base lmdb 校验
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb \
  --viz-num 50

# lidar lmdb 校验（可选，仅当打包了 lidar lmdb 时）
python3 tools/datasets/nuscenes_lidar_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/lidar

# map lmdb 校验（可选，仅当打包了 map lmdb 时；自动从 __meta__ 读取 pc_range）
python3 tools/datasets/nuscenes_map_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/map
```

可视化 PNG 保存到各脚本的 `--save-dir`（须为本地可写目录；`tmp_data/heal_data` 是只读 bucket，不能作为保存目标），可人工核对结果是否对齐。本任务地图 GT 在线生成、默认不打包 lidar/map lmdb，故上述 lidar/map 校验为可选项。

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | Float 优化器 |
|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 2 | 30 | （以 config 为准） | AdamW（`lr=2.5e-4`） |

本任务无 QAT 阶段。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage float
# 产物：./tmp_models/maptr_henet_tinym_bevformer/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage calibration
# 产物：./tmp_models/maptr_henet_tinym_bevformer/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/maptr_henet_tinym_bevformer/deploy.py
# 产物：./tmp_models/maptr_henet_tinym_bevformer/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/maptr_henet_tinym_bevformer/deploy.py
# 产物：./tmp_models/maptr_henet_tinym_bevformer/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在量化模型基础上重新执行导出与编译。修改 `configs/maptr_henet_tinym_bevformer/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage hbm
```

评测指标为 chamfer mAP（MAP），评测集为 nuScenes val（`val_lmdb`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的数据（保存到 `./demo/maptr_henet_tinym_bevformer/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/maptr_henet_tinym_bevformer

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/maptr_henet_tinym_bevformer \
  --stage float
```

可视化结果保存到 `./demo/maptr_henet_tinym_bevformer/`。
