# BEV Multitask + HENet-tiny

纯相机 BEV 多任务模型，覆盖 3D 检测、占用预测、停止线检测与在线车道级建图。

BEV Multitask + HENet-tiny 将六路环视抬升为共享鸟瞰特征，五个头协同输出动静目标、占用、停止线与矢量地图，保证场景描述一致。本版本面向 Horizon J6M / J6P 部署，推理不依赖激光雷达。

模型配置目录：`configs/bev_multitask_henet_tinym/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| BEVFusionMultitask | 6 相机多视图图像 `(B,6,3,512,960)` | HENet-tiny | FPN + SingleBevFormerViewTransformer | 动态/静态 3D 检测框 `(B,N,cls+reg)`；occ 体素语义；离散地图 2D 框；地图 polyline |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` + `val_occ_lmdb`）。指标：NDS/mAP（`NuscenesMetric`，动态/静态 det）、离散地图 NDS（`BEV2DNDSMetric`，stop line）、Occ mIoU（`MeanIOU`，`ignore_index=17`）、MapTR chamfer mAP（`NuscenesMapMetric`）。

| March | 评测指标 | float | calibration | hbm |
| --- | --- | --- | --- | --- |
| J6M | 动态 NDS | 0.4553 | 0.4480 | 0.4535 |
|  | 静态 NDS | 0.4873 | 0.4839 | 0.4842 |
|  | Occ mIoU | 0.3422 | 0.3401 | 0.3399 |
|  | stop_line NDS | 0.4802 | 0.4762 | 0.4759 |
|  | MapTR mAP | 0.5486 | 0.5405 | 0.5376 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段。
>
> HEAL 版本：heal 0.14.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 77.82 | 13.21 | 386.90 |
| J6P | 47.65 | 76.17 | 373.50 |

编译产物路径：
- HBIR：`./tmp_models/bev_multitask_henet_tinym/qat.bc`
- HBM：`./tmp_models/bev_multitask_henet_tinym/compile/model.hbm`

---

## 模型介绍

### 核心设计

六路环视先由 HENet-tiny 编码，再经单帧 BevFormer 抬到约百米范围的紧凑 BEV 画布。同一份 BEV 特征分给五个轻量头：SparseBEV 风格的动静 3D 框、体素占用、停止线检测，以及 MapTR 风格的分隔带/人行横道/边界折线——检测、可行驶空间与地图线索保持一致，无需再跑五套独立网络。

- **任务类型**：BEV 多任务感知（动态 det + 静态 det + occupancy + 离散地图 + 在线地图）。
- **backbone**：HENet-tiny（`type=HENet`，`block_nums=[4,3,8,6]`，`embed_dims=[64,128,192,384]`，`GroupDWCB`/`AltDWCB`/`DWCB` block，`S2DDown` 下采样），加载 ImageNet 预训练的 `henet_tinym` float 权重初始化。
- **neck**：`FPN`（`in_strides=[32]` → `out_strides=[32]`，`_dim_=256`）+ `BevFuseModule`（`input_c=384+256` → `fuse_c=256`；lidar 分支未启用，`lidar_input=False`）。
- **动态 det 头**：`DenseBEVSparseHead`（SparseBEVHead 风格，8 类，`num_anchor=384`，`num_decoder=6`，`MemoryBank` 带 `num_temp_instances=128` 用于跟踪，DN denoising groups 5 + temporal 3）。
- **静态 det 头**：`DenseBEVSparseHead`（2 类：`traffic_cone`、`barrier`；`num_anchor=200`，`num_temp_instances=80`）。
- **Occupancy 头**：`BevformerOccDetDecoder` + `BEVOCCHead2D`（`Dz=16`，18 类，ROI 裁剪 `(14,14,114,114)`）。
- **离散地图头**：`BEVFormerDet2DDecoder`（1 类 `stop_line`，`num_query=100`，6 层 `Detection2DTransformerDecoder`）。
- **在线地图头**：`MaptrDetDecoder` + `MapInstanceDetectorHead`（3 类：`divider`、`ped_crossing`、`boundary`；6 层 `MapInstanceDecoder`，`num_vec_one2one=50` / `num_vec_one2many=300`，每条 polyline 20 个点）。
- **损失函数**：det —— `FocalLoss`（cls）+ `L1Loss`（reg）+ `CrossEntropyLoss`（cns）+ `GaussianFocalLoss`（yns）；occ —— `CrossEntropyLoss`；maptr —— `MapTRCriterion`（`FocalLoss` + `PtsL1Loss` + `PtsDirCosLoss` + aux seg）；离散地图 —— `BevFormer2DCriterion`（`FocalLoss` + `L1Loss`）。任务 loss 权重：det 4 / static 2 / occ 96 / discrete map 2 / maptr 1.1。
- **模型输入**：6 相机多视图图像 `(B,6,3,512,960)`（YUV444 + `Normalize(128,128)`；纯相机，推理时无 lidar 点云）。
- **模型输出**：动态/静态 3D 检测框 + occ 体素语义 + stop line 2D 框 + 地图 polyline。

### 官方 repo 与论文链接

SparseBEV 风格 det 头官方 repo：https://github.com/MCG-NJU/SparseBEV （论文：https://arxiv.org/abs/2308.09244 ）
MapTR 官方 repo：https://github.com/hustvl/MapTR （论文：https://arxiv.org/abs/2208.14437 ）

注：相机骨干 HENet 与多任务组装为 HEAL 自研，官方 repo backbone 与单任务配置不同。

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 nuScenes 打包后的 base lmdb 与 occ lmdb（经 `MergeDataset(NuscenesMultitaskDataset + NuscenesOccDataset)` 融合），另需预计算的地图扩展 `meta/` 目录与 k-means anchor 文件。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_lmdb` | 训练 base lmdb（动态/静态 det + maptr + 离散地图 GT） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb` | 评测 base lmdb |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/train_occ_lmdb` | 训练 occ lmdb（Occ3D 占用标签） |
| `./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb` | 评测 occ lmdb |
| `./tmp_data/heal_data/nuscenes/meta` | 预计算地图扩展（MapTR GT 栅格化，`meta_rootdir`） |
| `./tmp_data/nuscenes/nuscenes_kmeans900.npy` | K-means anchor 文件（`anchor_file`） |

> **目录约定**：`orig_data/nuscenes` 为 `--src-data-dir`（原始解压数据），`heal_data/nuscenes/v1.0-trainval` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。

#### 官方下载

nuScenes 数据集官方页面：<https://www.nuscenes.org/download>（需注册下载）。需下载 **v1.0-trainval** 全量数据集，解压后含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`（元数据 json）。

下载后解压到原始数据目录 `./tmp_data/orig_data/nuscenes/`（作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes
# ./tmp_data/orig_data/nuscenes/samples/
# ./tmp_data/orig_data/nuscenes/sweeps/
# ./tmp_data/orig_data/nuscenes/maps/
# ./tmp_data/orig_data/nuscenes/v1.0-trainval/
```

本任务含 occ 标签；另需下载 Occ3D-nuScenes 标签（<https://github.com/Tsinghua-MARS-Lab/Occ3D>），将 `labels.npz` 按 scene/token 放入 `./tmp_data/orig_data/nuscenes/occ3d/gts/`：

```bash
mkdir -p ./tmp_data/orig_data/nuscenes/occ3d/gts
# 将 Occ3D-nuScenes 的 labels.npz 按 scene/token 放入该目录
```

#### 解压后目录结构

```
./tmp_data/orig_data/nuscenes/
├── samples/              # 标注关键帧图像（6 相机子目录）
├── sweeps/               # 中间帧图像
├── maps/                 # 地图扩张（expansion + png）
├── v1.0-trainval/        # 元数据 json
└── occ3d/
    └── gts/              # Occ3D-nuScenes 占用标签（labels.npz）
```

#### 打包命令

nuScenes 无需额外预处理，packer 自带 nuscenes-devkit 解析，直接读取官方解压目录即可打包。依赖顺序：**base 先**；occ 必须在 base 之后且 version/split 一致。本任务使用 **base lmdb + occ lmdb**，需执行 base packer 与 occ packer 各打 train/val 一次。未使用的 lidar/map packer 本任务不需要。

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

`NuscenesOccPacker`（occ）打占用 lmdb（`{split}_occ_lmdb`）；必须在 base 之后且 version/split 一致：

```bash
# 打 occ train split
python3 tools/datasets/nuscenes_occ_packer.py \
  --src-data-dir ./tmp_data/orig_data/nuscenes \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/nuscenes/v1.0-trainval \
  --split-name train \
  --version v1.0-trainval \
  --occ-gt-dir occ3d/gts \
  --num-workers 20

# 打 occ val split
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
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/nuscenes`（含 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/`、`occ3d/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/nuscenes/v1.0-trainval`，与 config 的 `data_rootdir` 对应。
- `--pack-type lmdb`：打包格式统一用 lmdb。
- `--split-name`：`train` 或 `val`。
- `--version v1.0-trainval`：nuScenes 数据集版本（packer 默认 `v1.0-mini`，须显式覆盖）。
- `--occ-gt-dir occ3d/gts`（occ packer）：Occ3D-nuScenes 标签目录（默认值；需用户自行下载）。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/nuscenes/` 结构如下（本任务实际使用的已标注）：

```
./tmp_data/heal_data/nuscenes/
├── v1.0-trainval/
│   ├── train_lmdb/          # 训练 base lmdb          ← 本任务使用
│   ├── val_lmdb/            # 评测 base lmdb          ← 本任务使用
│   ├── train_lidar_lmdb/    # 训练 lidar lmdb（本任务不使用）
│   ├── val_lidar_lmdb/      # 评测 lidar lmdb（本任务不使用）
│   ├── train_map_lmdb/      # 训练 map lmdb（本任务不使用）
│   ├── val_map_lmdb/        # 评测 map lmdb（本任务不使用）
│   ├── train_occ_lmdb/      # 训练 occ lmdb          ← 本任务使用
│   └── val_occ_lmdb/        # 评测 occ lmdb          ← 本任务使用
├── meta/                    # 预计算地图扩展 meta 目录  ← 本任务使用
└── osm/                     # OSM 矢量地图（本任务不使用）
```

可用可视化脚本抽样检查打包结果：

```bash
# 校验 base lmdb
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50

# 校验 occ lmdb
python3 tools/datasets/nuscenes_occ_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_occ_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参（float 为 8 卡 × 2 机训练，`common.py` 中 `num_machines=2`）：

| 训练 GPU 数 | 每卡 batch size | 阶段 | epoch 数 | 学习率 | 优化器 | 学习率策略 |
|---|---|---|---|---|---|---|
| 8×2（`device_ids=[0..7]`，`num_machines=2`） | 4 | stage1（dynamic+static+occ+离散地图） | 200 | 1e-4 | AdamW（`weight_decay=0.1`，backbone `lr_mult=0.1`，det/occ/discrete_map 头 `lr_mult=2`） | `CosineAnnealingLrUpdater`（warmup 500 步，ratio 1/3） |
| 8×2 | 4 | stage2（仅 maptr，冻结 backbone/neck/view_transformer/fuse） | 100 | 2.5e-4 | AdamW | 同上 |
| 8×2 | 4 | stage3（全任务联合微调） | 200 | 2e-5 | AdamW（maptr `lr_mult=3`，其余头 `lr_mult=0.1`） | 同上 |

#### 预训练权重依赖

stage1 float 训练加载 [`henet_tinym`](../henet_tinym/readme.md) 的 ImageNet 分类 float 权重初始化 `camera_net.backbone`（经 `LoadCheckpoint` + `update_state_dict_by_add_prefix(prefix="camera_net.")`）：

- **权重路径**：`./tmp_pretrained_models/henet_tinym/float-checkpoint-best.pth.tar`。

用户须**先训练 `henet_tinym`**（或拿到其 float ckpt）并放置到上述路径，再运行本任务 stage1。

#### 三阶段 Float 训练

Float 训练按顺序分三阶段进行；阶段间 checkpoint 串联（`float_stage1-checkpoint-last.pth.tar` → stage2 / stage3 输入）：

```bash
# Stage 1：动态 + 静态 + occ + 离散地图
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym/train.py \
  --stage float_stage1
# 产物：./tmp_models/bev_multitask_henet_tinym/float_stage1-checkpoint-last.pth.tar

# Stage 2：仅 maptr（共享部分冻结）
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym/train.py \
  --stage float_stage2
# 产物：./tmp_models/bev_multitask_henet_tinym/float_stage2-checkpoint-last.pth.tar

# Stage 3：全任务联合微调
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym/train.py \
  --stage float_stage3
# 产物：./tmp_models/bev_multitask_henet_tinym/float_stage3-checkpoint-best.pth.tar
```

量化校准加载 stage3 float 产物 `float_stage3-checkpoint-best.pth.tar`。本任务无 QAT 阶段；HBIR 导出在浮点图基础上直接套 fake-quant（`Float2QAT state="val"`）并复用 calibration checkpoint。

```bash
# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/bev_multitask_henet_tinym \
  --stage calibration
# 产物：./tmp_models/bev_multitask_henet_tinym/calibration-checkpoint-last.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/bev_multitask_henet_tinym/deploy.py
# 产物：./tmp_models/bev_multitask_henet_tinym/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/bev_multitask_henet_tinym/deploy.py
# 产物：./tmp_models/bev_multitask_henet_tinym/compile/model.hbm
```

> 编译时 `input_source=["pyramid", "ddr", ...]`（相机 `img` 走 pyramid/NV12（VPU）；投影矩阵、BEV 网格与 MemoryBank 缓存走 ddr）；`enable_vpu=True`。部署输入为纯 tensor（`img` + `projection_mat` + BEV 网格 + 动/静态 MemoryBank 缓存）；MemoryBank 状态时序传递由 `BevMultitaskIrInfer` 负责，不编译进导出图。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration）**。修改 `configs/bev_multitask_henet_tinym/common.py` 的 `march` 字段为目标芯片（`March.NASH_M`=J6M / `March.NASH_P`=J6P / `March.NASH_B`=J6B），然后重新执行 `export_hbir.py` 与 `compile_perf.py`。

### 精度验证

```bash
# Float 评测（stage3 全任务）
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym \
  --stage hbm
```

也可分阶段评测各 float 阶段 checkpoint（`float_stage1_predictor` / `float_stage2_predictor`，各阶段指标子集与训练阶段对应；`float_predictor` 为 stage3 别名）：

```bash
# 评测 stage1 ckpt（动态+静态+occ+离散地图指标）
python3 tools/predict.py \
  --config configs/bev_multitask_henet_tinym/predict.py \
  --stage float_stage1
```

评测指标：动态 det（8 类）与静态 det（2 类）为 NDS/mAP（`NuscenesMetric`）；stop line 离散地图为 `BEV2DNDSMetric`；occ 为 mIoU（`MeanIOU`，18 类，`ignore_index=17`）；在线地图为 `NuscenesMapMetric`（chamfer）。HBM 评测输入为原始 RGB uint8 图像（NV12 转换在设备端完成）；float/calibration/HBIR 使用 YUV444 + `Normalize(128,128)`。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的样本（保存到 `./demo/bev_multitask_henet_tinym/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本
python3 tools/gen_demo.py \
  --config configs/bev_multitask_henet_tinym

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/bev_multitask_henet_tinym \
  --stage float
```

可视化结果（相机画面 + BEV 检测框、occ、地图）保存到 `./demo/bev_multitask_henet_tinym/`。
