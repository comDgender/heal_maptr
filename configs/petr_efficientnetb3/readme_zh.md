# PETR (EfficientNet-b3)

`petr_efficientnetb3` 是基于 EfficientNet-b3 骨干、PETR 检测头的 Detr3d 框架 nuScenes BEV 3D 目标检测模型，覆盖「浮点训练 → 量化校准 → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/petr_efficientnetb3/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| PETR | 6 相机多视图图像 `(B,6,3,512,1408)` | EfficientNet-b3 | — | 检测框 `(B,900,21)` |

### 精度指标

评测集：nuScenes val（`./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb`）。指标：NDS / mAP（nuScenes 检测综合指标，越高越好）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | NDS | 0.3881 | 0.3679 | 0.38 | 0.38 |
|  | mAP | 0.3031 | 0.2807 | 0.2942 | 0.2942 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 33.00 | 30.52 | 101.40 |
| J6P | 21.58 | 186.30 | 105.40 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/petr_efficientnetb3/qat.bc`
- HBM：`./tmp_models/petr_efficientnetb3/compile/model.hbm`

---

## 模型介绍

### 核心设计

PETR 把 3D 位置引入 Transformer 解码：EfficientNet-b3 提取多视图图像特征，经 3D 位置编码（SinePositionalEncoding3D）将图像特征与 3D 空间位置关联，PETRTransformer 的 query 直接与 3D 位置编码特征交互预测 3D 检测框，无需显式 BEV 特征构造。

- **任务类型**：BEV 3D 目标检测（BEV 3D Object Detection）。
- **backbone**：EfficientNet-b3（`efficientnet`，`model_type=b3`，`include_top=False` 去除分类头）。
- **neck**：—（无独立 neck，backbone 特征直入 `PETRHead`）。
- **检测头**：`PETRHead` + `PETRTransformer` + `PETRDecoder`（3D 位置感知 Transformer 解码器，`num_query=900`、`num_views=6`、`embed_dims=256`）。
- **损失函数**：FocalLoss（cls）+ L1Loss（reg），经 Detr3dTarget 匈牙利匹配。
- **模型输入**：6 相机多视图图像，`data_shape=(3,512,1408)`（原图 `(3,900,1600)` 经 resize 到 `(3,792,1408)` 再 crop 到 `data_shape`），6 视图堆叠为 `(B,6,3,512,1408)`。
- **模型输出**：检测框 `(B,900,21)`——900 query × (10 类别分数 + 11 回归：中心 xyz + 尺寸 wlh + 朝向 sin/cos + 速度 vxvy)，`num_classes=10`、`reg_out_channels=10`。

### 官方 repo 与论文链接

官方repo: https://github.com/megvii-research/PETR
论文: https://arxiv.org/abs/2203.05625

注：backbone 为 EfficientNet-b3，官方 repo backbone 不同。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14091

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
└── meta/                # 元数据目录
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
python3 tools/datasets/nuscenes_viz.py \
  --data-path ./tmp_data/heal_data/nuscenes/v1.0-trainval/val_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch | Calibration 步数 | Float 优化器 | 学习率策略 | 损失函数 |
|---|---|---|---|---|---|---|
| 8 | 2 | 48 | 5（`40 // len(device_ids)`） | AdamW（lr=2e-4，backbone lr=2e-4，weight_decay=0.05） | Float: CosineAnnealingLrUpdater（warmup_lr_ratio=1/3）<br>QAT: StepDecayLrUpdater | FocalLoss + L1Loss |

本任务无跨任务预训练依赖，Float 训练从零开始。Calibration 加载本任务 Float 训练产出的 `float-checkpoint-best.pth.tar`。QAT 加载 Calibration 产出的 `calibration-checkpoint-best.pth.tar`。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/petr_efficientnetb3 \
  --stage float
# 产物：./tmp_models/petr_efficientnetb3/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/petr_efficientnetb3 \
  --stage calibration
# 产物：./tmp_models/petr_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/petr_efficientnetb3 \
  --stage qat
# 产物：./tmp_models/petr_efficientnetb3/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/petr_efficientnetb3/deploy.py
# 产物：./tmp_models/petr_efficientnetb3/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/petr_efficientnetb3/deploy.py
# 产物：./tmp_models/petr_efficientnetb3/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在量化模型基础上重新执行导出与编译。修改 `configs/petr_efficientnetb3/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/petr_efficientnetb3 \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/petr_efficientnetb3 \
  --stage hbm
```

评测指标为 NDS 与 mAP，评测集为 nuScenes val。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像（保存到 `./demo/petr_efficientnetb3/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理样本（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/petr_efficientnetb3/）
python3 tools/gen_demo.py \
  --config configs/petr_efficientnetb3

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/petr_efficientnetb3 \
  --stage float
```

可视化结果保存到 `./demo/petr_efficientnetb3/`。
