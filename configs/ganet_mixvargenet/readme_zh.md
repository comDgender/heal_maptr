# GaNet（MixVarGENet）

`ganet_mixvargenet` 是基于 GaNet 的车道线检测模型，以 MixVarGENet 为骨干，通过关键点热力图与逐点偏移回归实现端到端车道线检测。本文说明如何基于原始 CULane 数据集，完成「浮点训练 → 量化校准 → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/ganet_mixvargenet/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| GaNet | 单帧图像 `1x3x320x800` | MixVarGENet | `GaNetNeck` | 车道线点序列 `(B,L,P,2)` |

### 精度指标

评测集：CULane test split（`./tmp_data/heal_data/CULane/test_lmdb`）。指标：CulaneF1Score (F1)。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | CulaneF1Score (F1) | 0.7937 | 0.791 | — | 0.7908 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试；本任务无 QAT 阶段（qat 列为 —）。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 0.68 | 2262.93 | 6.50 |
| J6P | 0.61 | 8656.18 | 6.70 |
| J6B | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- HBIR：`./tmp_models/ganet_mixvargenet/qat.bc`
- HBM：`./tmp_models/ganet_mixvargenet/compile/model.hbm`

---

## 模型介绍

### 核心设计

GaNet 将车道线建模为「根关键点 + 沿线偏移」：网络预测下采样 8 倍的热力图定位车道线起点，并回归逐点偏移以重建整条车道线；解码端按 `kpt_thr` 阈值筛选关键点、按 `cluster_thr` 聚类合并同一条线，最终映射回原图坐标。

- **任务类型**：车道线检测（Lane Detection）。
- **backbone**：MixVarGENet（`include_top=False`，`output_list=[2, 3, 4]` 三个尺度特征，对应 stride 8/16/32）。
- **neck**：`GaNetNeck`：FPN（三尺度融合到 `hid_dim=32`）+ Attention 模块（`attn_ratio=4`，位置编码 `pos_shape=(1,10,25)`）。
- **检测头**：`GaNetHead`（输出 `hid_dim=32` 维特征）。
- **目标生成**：`GaNetTarget`（`hm_down_scale=8`，关键点半径 `radius=2`）。
- **后处理**：`GaNetDecoder`（`root_thr=1`，`kpt_thr=0.4`，`cluster_thr=5`，`downscale=8`）。
- **损失**：`GaNetLoss`：`LaneFastFocalLoss`（关键点分类，weight=1.0）+ `L1Loss`（逐点偏移回归 weight=0.5 + 整数偏移回归 weight=1.0）。
- **模型输入**：单帧图像，尺寸 `320 × 800`（由 `FixedCrop` 取原图 `(0,270,1640,320)` 区域后 `Resize` 到 320×800，原图分辨率 1640×590）。
- **模型输出**：每条车道线的有序点序列（原图坐标系），经 `GaNetDecoder` 从热力图关键点聚类 + 偏移解码得到。

**部署说明**：部署图（`deploy_model`）去除 `targets`/`post_process`/`losses`，只保留 backbone + neck + head，输出名为 `pred` 的热力图特征；`GaNetDecoder` 在推理阶段（float/hbir/hbm infer）重新挂回，完成车道线解码。

### 官方 repo 与论文链接

官方repo: https://github.com/Wolfwjs/GANet
论文: https://arxiv.org/abs/2204.07335

注：backbone MixVarGENet 为 HEAL 自研。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14098

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 CULane 打包后的 lmdb。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/CULane/train_lmdb` | 训练集 lmdb（对应 `common.py` 的 `train_data_path`） |
| `./tmp_data/heal_data/CULane/test_lmdb` | 评测集 lmdb（对应 `common.py` 的 `val_data_path`，评测用 CULane test split） |

> 目录约定：`./tmp_data/orig_data/CULane/` 为**原始数据解压目录**（打包命令的 `--src-data-dir`）；`./tmp_data/heal_data/CULane/` 为**打包产物目录**（打包命令的 `--target-data-dir`，供 config 读取）。注意 **CULane 大写**。

#### 官方下载

CULane 数据集官方页面：<https://xingyizhou.xyz/rgnn/>（或 CULane 官方发布页）。需下载以下文件：

- 图片（三个 driver 包）：
  - `driver_23-30.zip`
  - `driver_161-193.zip`
  - `driver_1823.zip`
- 标签：`annotations_new.tar.gz`（含 `list/` 目录及各 driver 下的 `.lines.txt` 标注文件）

下载后将所有文件解压到原始数据目录 `./tmp_data/orig_data/CULane/`（该目录作为打包的 `--src-data-dir`）。**解压顺序很重要**：先解压三个 driver zip（图片），**最后**解压 `annotations_new.tar.gz`（它会向 `list/` 写入 `train.txt`/`test.txt`/`val.txt` 及 `test_split/*.txt`，并向各 `driver_*/` 目录写入对应的 `.lines.txt` 标注子文件）。

```bash
mkdir -p ./tmp_data/orig_data/CULane

# 1. 先解压三个图片包（driver_23-30 / driver_161-193 / driver_1823）
unzip driver_23-30.zip    -d ./tmp_data/orig_data/CULane
unzip driver_161-193.zip  -d ./tmp_data/orig_data/CULane
unzip driver_1823.zip     -d ./tmp_data/orig_data/CULane

# 2. 最后解压标注包（生成 list/ 及 driver_*/ 下的 lines 文件）
tar -xzf annotations_new.tar.gz -C ./tmp_data/orig_data/CULane
```

#### 解压后目录结构

解压后 `./tmp_data/orig_data/CULane/` 结构如下：

```
./tmp_data/orig_data/CULane/
├── driver_23-30/                       # 图片，按 driver 分目录
│   └── 00020/                          #   子目录为视频片段
│       └── 0002940.jpg ...             #   帧图片（1640 × 590）
├── driver_161-193/                     # 图片
│   └── ...
├── driver_1823/                        # 图片
│   └── ...
└── list/                               # 由 annotations_new.tar.gz 解出
    ├── train.txt                       # 训练样本列表（每行一帧相对路径，如 driver_23-30/00020/0002940.jpg）
    ├── test.txt                        # 测试样本列表（评测用）
    ├── val.txt                         # 验证样本列表
    └── test_split/                     # 测试集按场景拆分
        ├── test0_normal.txt
        ├── test1_crowd.txt
        ├── test2_hlight.txt
        ├── test3_shadow.txt
        ├── test4_noline.txt
        ├── test5_arrow.txt
        ├── test6_curve.txt
        ├── test7_cross.txt
        └── test8_night.txt
```

各 `driver_*/<segment>/` 目录下还有与图片同名的 `.lines.txt` 标注文件（由 `annotations_new.tar.gz` 写入），每行一条车道线，坐标按 `x1 y1 x2 y2 ...` 排列。打包脚本通过 `list/train.txt` 与 `list/test.txt` 索引样本，按相对路径读取对应图片与 `.lines.txt`。

#### 打包命令

用 `tools/datasets/culane_packer.py`（`CuLanePacker`）把官方原始 CULane 打包成 lmdb。`--src-data-dir` 指向原始数据目录 `./tmp_data/orig_data/CULane`（含 `driver_*/` 图片与 `list/*.txt`），`--target-data-dir` 指向打包产物目录 `./tmp_data/heal_data/CULane`（供 config 读取）。packer 按 `--split-name` 读取 `list/train.txt`（train）或 `list/test.txt`（test），输出到 `{target}/{split}_lmdb`。本任务 config 的 `val_data_path` 使用 `test_lmdb`，因此评测 split 为 test。

```bash
# 打 train split（读 list/train.txt，产出 train_lmdb）
python3 tools/datasets/culane_packer.py \
  --src-data-dir ./tmp_data/orig_data/CULane \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/CULane \
  --split-name train \
  --num-workers 20

# 打 test split（读 list/test.txt，产出 test_lmdb，供评测用）
python3 tools/datasets/culane_packer.py \
  --src-data-dir ./tmp_data/orig_data/CULane \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/CULane \
  --split-name test \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/CULane`（含 `driver_*/` 图片与 `list/*.txt`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/CULane`，产出 `train_lmdb` 与 `test_lmdb`，与 `common.py` 的 `train_data_path`/`val_data_path` 路径逐字对应。
- `--split-name`：`train` 读 `list/train.txt`，`test` 读 `list/test.txt`，分别产出 `train_lmdb`/`test_lmdb`。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/CULane/` 结构如下：

```
./tmp_data/heal_data/CULane/
├── train_lmdb/        # 训练集 lmdb
└── test_lmdb/         # 评测集 lmdb（config 的 val_data_path 指向此处）
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
python3 tools/datasets/culane_viz.py \
  --data-path ./tmp_data/heal_data/CULane/train_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | Float 优化器 | 学习率调度 | 同步 BN |
|---|---|---|---|---|---|---|
| 8（`device_ids = [0,1,2,3,4,5,6,7]`） | 64 | 240 | 10 | Adam（lr=0.01，weight_decay=4e-5） | CosLrUpdater（warmup 1 epoch） | sync_bn=True |

> 本任务无 QAT 阶段（`train.py` 只定义 `float_trainer` 与 `calibration_trainer`），量化路径为 float → calibration → HBIR → HBM。Float 训练从零开始，无跨任务预训练依赖。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/ganet_mixvargenet \
  --stage float
# 产物：./tmp_models/ganet_mixvargenet/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/ganet_mixvargenet \
  --stage calibration
# 产物：./tmp_models/ganet_mixvargenet/calibration-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/ganet_mixvargenet/deploy.py
# 产物：
#   ./tmp_models/ganet_mixvargenet/qat.bc（部署用 HBIR，供 compile_perf.py 编译）
#   ./tmp_models/ganet_mixvargenet/quantized_infer.bc（推理用 HBIR，供 hbir_predictor 评测）

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/ganet_mixvargenet/deploy.py
# 产物：./tmp_models/ganet_mixvargenet/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化校准（calibration）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration，只需在已有 calibration 量化模型基础上重新执行导出与编译。修改 `configs/ganet_mixvargenet/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/ganet_mixvargenet \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/ganet_mixvargenet \
  --stage calibration

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/ganet_mixvargenet \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/ganet_mixvargenet \
  --stage hbm
```

评测指标为 CulaneF1Score（车道线检测 F1），评测集为 CULane test split（`./tmp_data/heal_data/CULane/test_lmdb`）。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像帧（保存到 `./demo/ganet_mixvargenet/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理帧（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/ganet_mixvargenet/）
python3 tools/gen_demo.py \
  --config configs/ganet_mixvargenet

# 2. 推理与可视化（以 float 为例；可用 calibration/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/ganet_mixvargenet \
  --stage float
```

可视化结果保存到 `./demo/ganet_mixvargenet/<stage>/`（按阶段分子目录）。
