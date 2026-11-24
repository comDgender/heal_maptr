# SparseMultitaskOE+MapTR+FlashOcc+HENet 多任务

`sparse_multitask_det_maptr_flashocc_henet_tinym` 是基于 SparseMultitaskOE（SparseBEVOEHead）的多任务模型，联合完成 3D 目标检测（det）+ 矢量化地图构建（map）+ 占据网格预测（occ）。本任务采用三段式 Float 训练（stage1 主干 → stage2 加入 occ/map 头 → merge 合并），`use_lidar_gt=True`，使用 base+lidar+map+occ lmdb + meta+osm。

模型配置目录：`configs/sparse_multitask_det_maptr_flashocc_henet_tinym/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| SparseMultitaskOE | 6 相机多视图图像 `(B,6,3,256,704)` + lidar 点云 `(B,N,5)` | HENet-tiny | MMFPN | det 检测框 `(B,N,cls+reg)`；map 矢量化地图 `(B,L,P,2)`；occ 占用栅格 `(B,C,H,W)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_lidar_lmdb` + `val_map_lmdb` + `val_occ_lmdb`）。指标：NDS/mAP（det，`NuscenesMetric`）+ chamfer mAP（map，`NuscenesMapMetric`，`metric="chamfer"`）+ Occ mIoU（`MeanIOU`，`ignore_index=17`）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.5434 | 0.5325 | — | 0.5281 |
|  | mAP | — | — | — | — |
|  | chamfer mAP (MAP) | 0.592 | 0.5838 | — | 0.5833 |
|  | Occ mIoU | 0.3197 | 0.3277 | — | 0.3273 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 `—`）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS J6M/J6P 为单核八线程、J6B 为单核双线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 27.50 | 36.71 | 181.90 |
| J6P | 17.76 | 188.07 | 210.80 |
| J6B | 164.46 | 9.53 | 301.60 |

编译产物路径：
- HBIR：`./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/qat.bc`
- HBM：`./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/compile/model.hbm`

---

## 模型介绍

### 核心设计

SparseMultitaskOE 以 HENet 为 backbone 提取多视角特征，转换为 BEV 特征后由 SparseBEVOEHead 统一驱动三个任务头：检测头（SparseBEVOE 检测，10 类 3D 框）、地图头（MapTR 风格矢量化地图，3 类元素）、占据头（FlashOcc 风格语义占据，18 类）。本任务采用三段式 Float 训练：stage1 联合训练 det+map+occ（主干），stage2 冻结主干单独训练 map+occ 头（`lr=1e-5`），merge 阶段将 stage1（det）与 stage2（map/occ）权重合并为完整模型。`use_lidar_gt=True`，`defer_vectormap=True`（map 在线生成 + map lmdb 并用）。

- **任务类型**：多任务融合（3D 目标检测 + 矢量化地图构建 + 占据网格预测）。
- **backbone**：HENet-tiny（ImageNet 预训练）。
- **neck**：MMFPN。
- **检测头**：`SparseBEVOEHead`（`SparseBEVOEEncoder` + `SparseBEVOERefinementModule`，10 类 3D 检测框，`num_classes=10`）。
- **地图头**：`SparseMapPerceptionDecoder`（`SparseMapHead`，3 类矢量化地图元素，`map_classes=[divider,ped_crossing,boundary]`）。
- **占据头**：语义占据预测，18 类（`num_classes_occ=18`）。
- **BEV 范围**：`point_cloud_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`（det/occ），`map_point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]`（map），`occ_bev_size=(40,40,0.625)`。
- **数据**：`MergeDataset(SparseMap(base+map) + Lidar + Occ)`，使用 base+lidar+map+occ lmdb + meta+osm。
- **模型输入**：6 相机多视图图像 (B,6,3,256,704) + lidar 点云 (B,N,D)。
- **模型输出**：det 的 3D 检测框 + map 的矢量化地图元素 + occ 的占据网格语义。

**部署说明**：HBIR 导出开启 `enable_vpu=True`，编译时 `input_source=compile_cfg["input_source"]`。

### 官方 repo 与论文链接

官方repo: det SparseBEV/map MapTR/occ FlashOcc
https://github.com/MCG-NJU/SparseBEV, https://github.com/hustvl/MapTR, https://github.com/Yzichen/FlashOCC
论文: https://arxiv.org/abs/2308.09244, https://arxiv.org/abs/2208.14437, https://arxiv.org/abs/2311.12058

注：相机骨干 HENet 为 HEAL 自研；det/map/occ 各自上游论文分别为 SparseBEV/MapTR/FlashOcc。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14095

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务 `use_lidar_gt=True`，使用 base+lidar+map+occ lmdb + meta+osm。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lidar_lmdb` | 训练 lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb` | 评测 lidar lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_map_lmdb` | 训练 map lmdb（地图 GT） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb` | 评测 map lmdb（地图 GT） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | 训练 occ lmdb（占据标签） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | 评测 occ lmdb（占据标签） |
| `./tmp_data/heal_data/nuscenes/meta` | 地图扩展预计算目录 |
| `./tmp_data/heal_data/nuscenes/osm` | OpenStreetMap 矢量地图 |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

nuScenes 无需额外预处理，四个 packer（base/lidar/map/occ）自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。本任务实际使用 **base + lidar + map + occ lmdb**（+ meta/osm 目录）。

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

本任务需 meta 目录（地图扩展预计算）与 osm 目录（OpenStreetMap 矢量地图）供 map packer 与在线地图生成使用。

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

nuScenes 无需额外预处理，四个 packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。依赖顺序：base 先；occ 必须在 base 之后且 version/split 一致；lidar/map 仅需 version/split 一致。本任务实际使用 **base + lidar + map + occ lmdb**，需依次执行 base、lidar、map、occ 四个 packer（occ 在 base 之后）。

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

**map packer**（`nuscenes_map_packer.py`，`NuscenesMapGtPacker`）→ `{split}_map_lmdb`（`--pc-range` 须与 `map_point_cloud_range` 一致）：

```bash
# 打 train map lmdb
python3 tools/datasets/nuscenes_map_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --map-path ./tmp_data/heal_data/nuscenes/meta \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --pc-range -15 -30 -10 15 30 10 \
  --bev-size 100 50 \
  --fixed-ptsnum-per-line 20 \
  --map-classes divider ped_crossing boundary \
  --padding-value -10000

# 打 val map lmdb
python3 tools/datasets/nuscenes_map_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --map-path ./tmp_data/heal_data/nuscenes/meta \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name val \
  --version v1.0-trainval \
  --pc-range -15 -30 -10 15 30 10 \
  --bev-size 100 50 \
  --fixed-ptsnum-per-line 20 \
  --map-classes divider ped_crossing boundary \
  --padding-value -10000
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
- `--map-path ./tmp_data/heal_data/nuscenes/meta`（map packer）：地图扩展预计算 meta 目录。
- `--pc-range -15 -30 -10 15 30 10`（map packer）：与 `map_point_cloud_range=[-15.0,-30.0,-10.0,15.0,30.0,10.0]` 一致。
- `--bev-size 100 50`（map packer）：bev 尺寸。
- `--fixed-ptsnum-per-line 20`（map packer）：每条线固定点数，与 `fixed_ptsnum_per_gt_line=20` 一致。
- `--map-classes divider ped_crossing boundary`（map packer）：地图元素类别，与 `map_classes` 一致。
- `--occ-gt-dir occ3d/gts`（occ packer）：Occ3D-nuScenes 标签目录（默认值），需用户额外下载放置。
- PV seg mask 不打包，训练时在线生成（`defer_vectormap=True`）。

> **四个 packer 说明**（本任务使用 base + lidar + map + occ）：
> - **base**（`nuscenes_packer.py` / `NuscenesPacker`）→ `{split}_lmdb`：相机 + 标注基础数据，**本任务使用**。
> - **lidar**（`nuscenes_lidar_packer.py` / `NuscenesLidarPacker`）→ `{split}_lidar_lmdb`：激光雷达点云，**本任务使用**。
> - **map**（`nuscenes_map_packer.py` / `NuscenesMapGtPacker`）→ `{split}_map_lmdb`：矢量化地图 GT，**本任务使用**。
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
│   ├── train_map_lmdb/      # map 训练 lmdb      ← 本任务使用
│   ├── val_map_lmdb/        # map 评测 lmdb      ← 本任务使用
│   ├── train_occ_lmdb/      # occ 训练 lmdb      ← 本任务使用
│   └── val_occ_lmdb/        # occ 评测 lmdb      ← 本任务使用
├── meta/                    # 地图扩展预计算目录  ← 本任务使用
└── osm/                     # OpenStreetMap 矢量地图 ← 本任务使用
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

# map lmdb 校验（复用 heal/visualize/nuscenes_map.py，画 BEV 矢量地图 GT，无需模型；自动从 __meta__ 读取 pc_range）
python3 tools/datasets/nuscenes_map_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/nuscenes/map
```

可视化 PNG 保存到各脚本的 `--save-dir`（须为本地可写目录；`tmp_data/heal_data` 是只读 bucket，不能作为保存目标），可人工核对点云、矢量地图、占据标签与打包是否对齐。

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float stage1 训练步数 | Float stage2 训练步数 | Float merge | Calibration 步数 | Float stage1 优化器 | Float stage2 优化器 |
|---|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 4 | `num_steps_per_epoch * 200`（`stop_by="step"`） | `num_steps * 0.7`（`stop_by="step"`） | 无训练（`num_epochs=0`，仅合并权重） | 100 | AdamW（backbone `lr=3e-4`，base `lr=3e-4`） | AdamW（`lr=1e-5`） |

本任务采用三段式 Float 训练：
- **stage1**（`float_stage1`）：联合训练 det+map+occ，主干全量更新，加载预训练骨干权重。
- **stage2**（`float_stage2`）：冻结主干，单独训练 map/occ 头（det 缺席），`lr=1e-5`，加载 stage1 的 EMA 权重。
- **merge**（`float_merge`）：无训练步骤，将 stage1（全部权重）与 stage2（map/occ 头+neck 权重）合并为完整模型，保存为 `float_merged-checkpoint-last.pth.tar`。

本任务无 QAT 阶段。Calibration 加载 merge 产出的完整模型。

```bash
# Float stage1 训练（det+map+occ 联合，主干全量更新）
HAT_TRAINING_STEP=float_stage1 python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage1
# 产物：./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/float_stage1-checkpoint-best.pth.tar

# Float stage2 训练（冻结主干，单独训练 map/occ 头）
HAT_TRAINING_STEP=float_stage2 python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage2
# 产物：./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/float_stage2-checkpoint-best.pth.tar

# Float merge（合并 stage1 + stage2 权重，无训练）
HAT_TRAINING_STEP=float_merge python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_merge
# 产物：./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/float_merged-checkpoint-last.pth.tar

# 量化校准（Calibration，加载 merge 产物）
python3 tools/train.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage calibration
# 产物：./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym/deploy.py
# 产物：./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym/deploy.py
# 产物：./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/compile/model.hbm
```

> 导出时 `enable_vpu=True`，编译时 `input_source=compile_cfg["input_source"]`。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在量化模型基础上重新执行导出与编译。修改 `configs/sparse_multitask_det_maptr_flashocc_henet_tinym/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float stage1 评测（stage1 完整模型）
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage1

# Float stage2 评测（stage2 map/occ 头）
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_stage2

# Float merge 评测（合并后完整模型，推荐用于最终精度对比）
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_merge

# Calibration 评测
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage calibration

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage hbm
```

评测指标为 NDS/mAP（det）+ chamfer mAP（map，MAP）+ Occ mIoU（occ，`MeanIOU`，`ignore_index=17`），评测集为 nuScenes val（`val_lmdb` + `val_lidar_lmdb` + `val_map_lmdb` + `val_occ_lmdb`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

#### 量化敏感性分析（可选）

本任务定义了 `quant_analysis_solver`，可逐层分析量化对精度的影响，辅助排查量化掉点来源：

```bash
python3 tools/quant_analysis.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym/quant_analysis.py
```

结果输出到 `./tmp_models/sparse_multitask_det_maptr_flashocc_henet_tinym/analysis/`。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的数据（保存到 `./demo/sparse_multitask_det_maptr_flashocc_henet_tinym/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym

# 2. 推理与可视化（以 float_merge 为例；可用 calibration/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/sparse_multitask_det_maptr_flashocc_henet_tinym \
  --stage float_merge
```

可视化结果保存到 `./demo/sparse_multitask_det_maptr_flashocc_henet_tinym/`。
