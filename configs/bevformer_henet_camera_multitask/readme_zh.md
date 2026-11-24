# BevFormer+HENet 多任务

`bevformer_henet_camera_multitask` 是基于 BevFormer + HENet 的相机端多任务模型，联合完成 3D 目标检测（det）与占据网格预测（occ）。本任务为**浮点预训练模型**，仅提供 float 训练与评测，其产物用于 [`bevfusion_pointpillar_henet_multisensor_multitask`](../bevfusion_pointpillar_henet_multisensor_multitask/readme.md) 的 float 训练相机分支初始化，本身不做量化与部署。

模型配置目录：`configs/bevformer_henet_camera_multitask/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| BevFusion | 6 相机多视图图像序列 `(B,6,3,512,960)` | HENet | FPN | det 检测框 `(B,N,cls+reg)`；occ 占用栅格 `(B,C,H,W)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_occ_lmdb`）。指标：NDS/mAP（det，`NuscenesMetric`）+ Occ mIoU（`MeanIOU`，`ignore_index=17`）。

| March | 评测指标 | float |
| --- | --- | --- |
| J6M | NDS | 0.3573 |
|  | mAP | 0.3017 |
|  | Occ mIoU | 0.3146 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

---

## 模型介绍

### 核心设计

BevFormer+HENet 多任务模型以 HENet 为相机骨干提取多视角特征，经 BevFormer 的 ViewTransformer 与时序 BEV Encoder 转换到鸟瞰图特征；BEV 特征同时送入检测头（3D 目标检测）与占据头（FlashOcc 风格的语义占据预测），实现 det+occ 联合训练。

- **任务类型**：多任务融合（3D 目标检测 + 占据网格预测）。
- **backbone**：HENet（相机特征提取），提取多视角特征。
- **neck**：FPN。
- **检测头**：BevFormer 检测头，输出 10 类 3D 检测框 + 速度（`num_classes=10`）。
- **占据头**：语义占据预测，18 类（`num_classes_occ=18`）。
- **BEV 范围**：`bev_size=(51.2, 51.2, 0.8)`，`bev_size_occ=(40, 40)`。
- **模型输入**：6 路相机图像序列 `(B,6,3,512,960)`（`queue_length=1`，`MergeDataset(Bev+Occ)` 融合 base lmdb 与 occ lmdb）。
- **模型输出**：det 的 3D 检测框 + occ 的占据网格语义 `(B,C,H,W)`。

### 官方 repo 与论文链接

官方repo: https://github.com/fundamentalvision/BevFormer
论文: https://arxiv.org/abs/2203.17270

注：相机骨干 HENet 为 HEAL 自研实现，官方 repo backbone 不同。

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 nuScenes 打包后的 base lmdb + occ lmdb（`MergeDataset(Bev+Occ)`）。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb（det 数据） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb（det 数据） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | 训练 occ lmdb（占据标签） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | 评测 occ lmdb（占据标签） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

nuScenes 无需额外预处理，四个 packer（base/lidar/map/occ）自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。本任务实际使用 **base lmdb + occ lmdb**（不使用 lidar/map lmdb）。

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

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。依赖顺序：base 先；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际使用 **base lmdb + occ lmdb**，需依次执行 base 与 occ 两个 packer（occ 在 base 之后）。

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
- `--max-sweeps 10`（base packer）：最大 sweep 数。
- `--occ-gt-dir occ3d/gts`（occ packer）：Occ3D-nuScenes 标签目录（默认值），需用户额外下载放置。
- `--num-workers`：并行打包进程数（默认 20）。

> **四个 packer 说明**（本任务仅使用 base + occ）：
> - **base**（`nuscenes_packer.py` / `NuscenesPacker`）→ `{split}_lmdb`：相机 + 标注基础数据，**本任务使用**。
> - **lidar**（`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`）→ `{split}_lidar_lmdb`：激光雷达点云，本任务不使用。
> - **map**（`nuscenes_map_packer.py` / `NuscenesMapGtPacker`）→ `{split}_map_lmdb`：矢量化地图 GT，本任务不使用。
> - **occ**（`nuscenes_occ_packer.py` / `NuscenesOccPacker`）→ `{split}_occ_lmdb`：Occ3D 占据标签，**本任务使用**。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（标注本任务实际用哪些）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # base 训练 lmdb      ← 本任务使用
│   ├── val_lmdb/            # base 评测 lmdb      ← 本任务使用
│   ├── train_lidar_lmdb/    # lidar 训练 lmdb      （本任务不使用）
│   ├── val_lidar_lmdb/      # lidar 评测 lmdb      （本任务不使用）
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
```

### 训练流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Float 优化器 | 学习率策略 |
|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 1 | 24 | AdamW（`lr=4e-4`，backbone `lr_mult=0.1`） | CosineAnnealingLrUpdater（warmup） |

#### 预训练权重依赖

本任务 float 训练通过 `LoadCheckpoint` 加载 [`henet_tinym`](../henet_tinym/readme.md)（ImageNet 分类）的 float 权重作为 `camera_net.backbone` 初始化：

- **权重来源任务**：`henet_tinym`（ImageNet 分类，HENet-tiny/m 骨干）。
- **权重路径**：`./tmp_pretrained_models/henet_tinym/float-checkpoint-best.pth.tar`（由该任务 float 训练产出）。
- **载入方式**：`train.py` 的 `float_trainer` 经 `LoadCheckpoint` 加载，并用 `update_state_dict_by_add_prefix(prefix="camera_net.")` 把 ImageNet 骨干权重映射到本任务 `camera_net.` 命名空间下；`allow_miss=True`、`ignore_extra=True` 容忍结构差异。

客户需**先训练 `henet_tinym`**（或自行获取其 float ckpt）并放置到上述路径，再执行本任务 float 训练。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/bevformer_henet_camera_multitask \
  --stage float
# 产物：./tmp_models/bevformer_henet_camera_multitask/float-checkpoint-last.pth.tar
```

#### 下游用途

本任务 float 训练产物 `./tmp_models/bevformer_henet_camera_multitask/float-checkpoint-last.pth.tar` 用于 [`bevfusion_pointpillar_henet_multisensor_multitask`](../bevfusion_pointpillar_henet_multisensor_multitask/readme.md) 的 float 训练相机分支初始化：

- 该任务 `train.py` 的 `float_trainer` 经 `LoadCheckpoint` 加载本任务 float ckpt（无前缀，因相机分支结构与本任务一致），`allow_miss=True`、`ignore_extra=True` 容忍多传感器融合新增模块（如 `BevFuseModule`）的随机初始化。
- 该任务 `common.py` 的 `camera_ckpt_dir` 指向本任务 ckpt 的放置目录，客户做 bevfusion 前需先完成本任务 float 训练并把 `float-checkpoint-last.pth.tar` 放到其 `camera_ckpt_dir`。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/bevformer_henet_camera_multitask \
  --stage float
```

评测指标为 NDS/mAP（det）与 Occ mIoU（occ，`MeanIOU`，`ignore_index=17`），评测集为 nuScenes val（`val_lmdb` + `val_occ_lmdb`）。评测命令默认遵循 config 的 `device_ids`，如需指定评测 GPU 可通过 `--device-ids` 覆盖。
