# MOTR (EfficientNet-b3)

本文档介绍 `motr_efficientnetb3` 模型，并说明如何基于原始 MOT17 数据集，完成「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/motr_efficientnetb3/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| MOTR | 多帧图像序列 `1x3x800x1422` | EfficientNet-b3 | — | 跟踪框 + 轨迹 ID `(B,Q,cls+reg+id)` |

### 精度指标

评测集：MOT17 test split（`./tmp_data/heal_data/mot17/test_lmdb` + `test_gt`）。指标：MOTA（多目标跟踪精度，越高越好）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | MOTA | 0.5837 | 0.5704 | 0.5799 | 0.5767 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | 评测指标 | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|---|
| J6M | 主体 | 8.01 | 128.55 | 64.80 |
|  | qim | 0.37 | 5120.87 | 6.80 |
| J6P | 主体 | 5.85 | 680.63 | 74.60 |
|  | qim | 0.36 | 10348.73 | 7.00 |
| J6B | 主体 | - | - | - |
|  | qim | - | - | - |

本模型暂不提供J6B性能

编译产物路径：
- 主图：`./tmp_models/motr_efficientnetb3/qat.bc`（HBIR）、`./tmp_models/motr_efficientnetb3/compile/model.hbm`（HBM）
- QIM 子图：`./tmp_models/motr_efficientnetb3/qim/qat.bc`（HBIR）、`./tmp_models/motr_efficientnetb3/qim/compile/model.hbm`（HBM）

---

## 模型介绍

### 核心设计

MOTR 把跟踪建模为 query 的逐帧传播：检测 query 经 `QueryInteractionModule` 在帧间传递，新生目标用新 query、消失目标置空；训练时联合监督检测与跟踪损失。

- **任务类型**：多目标跟踪（Multi-Object Tracking, MOT）。
- **backbone**：EfficientNet-b3（`efficientnet`，`model_type=b3`，`include_top=False` 去除分类头，`use_se_block=False`，ReLU 激活）。
- **neck**：—（无独立 neck，backbone 特征直入 `MotrHead`）。
- **检测头**：`MotrHead` + `MotrDeformableTransformer`（`d_model=256`，`num_queries=256`，`dim_feedforward=1024`，Deformable Attention，`in_channels=[384]`）。
- **跟踪模块**：`QueryInteractionModule`（跨帧 query 交互，实现端到端跟踪关联）。
- **后处理**：`MotrPostProcess`。
- **模型输入**：多帧图像序列，单帧尺寸 `800 × 1422`，序列采样间隔 10 帧；主图 `img:1x3x800x1422`，QIM 子图输入为 query/feat（无图像）。
- **模型输出**：每帧检测框 + 跨帧轨迹 ID（256 query，1 类）。

**部署说明**：MOTR 推理图分为主图（backbone + head + post_process）与 QIM 子图（`QueryInteractionModule`，DDR 输入，无图像）两部分，分别经 `deploy.py` 与 `deploy_qim.py` 导出与编译，推理时协同工作（详见「使用指南 > 导出编译」）。

### 官方 repo 与论文链接

官方repo: https://github.com/megvii-research/MOTR
论文: https://arxiv.org/pdf/2105.03247

注：backbone 为 EfficientNet-b3，官方 repo backbone 不同。

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14093

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 MOT17 打包后的 lmdb。模型读取的数据路径如下：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/mot17/train_lmdb` | 训练序列 lmdb |
| `./tmp_data/heal_data/mot17/test_lmdb` | 评测序列 lmdb |
| `./tmp_data/heal_data/mot17/test_gt` | 评测用文本 gt（`motmetrics` 只读文本格式，不读 lmdb） |

#### 官方下载

MOT17 数据集官方页面：<https://motchallenge.net/data/MOT17/>（需注册下载，获取 `MOT17.zip`，含 train + test 两部分，约 5.46 GiB）。

下载后将 `MOT17.zip` 解压到原始数据目录 `./tmp_data/orig_data/mot17/`（该目录作为打包的 `--src-data-dir`）：

```bash
mkdir -p ./tmp_data/orig_data/mot17
unzip MOT17.zip -d ./tmp_data/orig_data/mot17
# 解压得到 ./tmp_data/orig_data/mot17/MOT17/{train,test}/...
```

#### 解压后目录结构

解压后 `./tmp_data/orig_data/mot17/MOT17/` 结构如下：

```
./tmp_data/orig_data/mot17/
└── MOT17/
    ├── train/                              # 7 个视频序列 × 3 种检测器 = 21 个目录
    │   ├── MOT17-02-DPM/                   #   每序列有 DPM/FRCNN/SDP 三种检测器变体
    │   ├── MOT17-02-FRCNN/
    │   ├── MOT17-02-SDP/                   # ← 打包只用 *SDP 变体
    │   │   ├── det/det.txt                 #   检测器结果（本任务不使用）
    │   │   ├── gt/gt.txt                   #   标签：9 列 CSV = fid,tid,x,y,w,h,mark,label,sss
    │   │   ├── img1/000001.jpg ... 000NNN.jpg  # 序列帧
    │   │   └── seqinfo.ini                 #   seqLength / imWidth / imHeight / imExt
    │   ├── MOT17-04-{DPM,FRCNN,SDP}/
    │   └── ... (MOT17-05/09/10/11/13 各 3 变体)
    └── test/                               # 7 序列 × 3 变体，但官方 test 无公开 gt
        ├── MOT17-01-{DPM,FRCNN,SDP}/        #   仅 {det,img1,seqinfo.ini}，无 gt/
        └── ... (MOT17-03/06/07/08/12/14)
```

**说明**：官方 `test/` 无公开 gt，无法直接用于评测。因此本任务把带 gt 的 `train/*SDP` 序列拆成两半：前半帧作为训练集，后半帧作为验证集（帧号重排为 1..M）。这一拆分由打包脚本在内存完成，无需额外预处理。

#### 打包命令

用 `tools/datasets/mot17_packer.py`（`Mot17Packer`）把官方原始 MOT17 一步打包成序列 lmdb。`--src-data-dir` 指向原始数据目录 `./tmp_data/orig_data/mot17/MOT17`（含 `train/`、`test/`），`--target-data-dir` 指向打包产物目录 `./tmp_data/heal_data/mot17`（供 config 读取）。打包脚本直接读取 `train/*SDP`，在内存中完成拆半与标签生成，不产生中间文件。

```bash
# 打 train split（每个 train/*SDP 序列的前半帧）
python3 tools/datasets/mot17_packer.py \
  --src-data-dir ./tmp_data/orig_data/mot17/MOT17 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mot17 \
  --split-name train \
  --num-workers 20

# 打 test split（每个 train/*SDP 序列的后半帧，帧号重排为 1..M）
python3 tools/datasets/mot17_packer.py \
  --src-data-dir ./tmp_data/orig_data/mot17/MOT17 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/mot17 \
  --split-name test \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/mot17/MOT17`（含 `train/`、`test/`）。打包只使用 `train/*SDP`。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/mot17`，产出 `train_lmdb` 与 `test_lmdb`，与 `common.py` 的 `train_lmdb`/`val_lmdb` 路径对应。
- `--split-name test` 时，脚本会额外把每个序列后半帧的 `gt.txt`（帧号重排为 1..M）写到 `./tmp_data/heal_data/mot17/test_gt/<seq>/gt/gt.txt`，供评测使用。该目录由脚本自动生成，无需手动组织。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

打包完成后，`./tmp_data/heal_data/mot17/` 结构如下：

```
./tmp_data/heal_data/mot17/
├── train_lmdb/        # 训练序列 lmdb
├── test_lmdb/         # 评测序列 lmdb
└── test_gt/           # 评测文本 gt（脚本自动生成）
    └── <seq>/gt/gt.txt   # 帧号已重排为 1..M
```

可用可视化脚本抽样检查打包结果是否正确：

```bash
python3 tools/datasets/mot17_viz.py \
  --data-path ./tmp_data/heal_data/mot17/train_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | QAT 训练 epoch 数 | Float 优化器 | QAT 优化器 |
|---|---|---|---|---|---|---|
| 8 | 1 | 250 | 500 | 20 | AdamW（backbone/sampling_offsets lr=2e-5，base lr=2e-4，wd=1e-4） | AdamW（lr=1e-5，wd=1e-4） |

**预训练权重**：Float 训练需加载 FCOS（MSCOCO）的浮点骨干权重以迁移 EfficientNet-b3。请先完成 `configs/fcos_efficientnetb3` 的 Float 训练，并将其 `float-checkpoint-best.pth.tar` 放到如下路径：

```
./tmp_pretrained_models/fcos_efficientnetb3/float-checkpoint-best.pth.tar
```

```bash
# Float 训练
python3 tools/train.py \
  --config configs/motr_efficientnetb3 \
  --stage float
# 产物：./tmp_models/motr_efficientnetb3/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/motr_efficientnetb3 \
  --stage calibration
# 产物：./tmp_models/motr_efficientnetb3/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/motr_efficientnetb3 \
  --stage qat
# 产物：./tmp_models/motr_efficientnetb3/qat-checkpoint-best.pth.tar
```

### 导出编译

MOTR 有主图与 QIM 子图两个导出配置，须分别执行导出 HBIR 与编译 HBM：

```bash
# 导出 HBIR
# 主图（backbone + head + post_process）
python3 tools/export_hbir.py \
  --config configs/motr_efficientnetb3/deploy.py
# QIM 子图（QueryInteractionModule，DDR 输入，无图像）
python3 tools/export_hbir.py \
  --config configs/motr_efficientnetb3/deploy_qim.py
# 产物：主图 ./tmp_models/motr_efficientnetb3/qat.bc；QIM 子图 ./tmp_models/motr_efficientnetb3/qim/qat.bc

# 编译 HBM
# 主图
python3 tools/compile_perf.py \
  --config configs/motr_efficientnetb3/deploy.py
# QIM 子图
python3 tools/compile_perf.py \
  --config configs/motr_efficientnetb3/deploy_qim.py
# 产物：主图 ./tmp_models/motr_efficientnetb3/compile/model.hbm；QIM 子图 ./tmp_models/motr_efficientnetb3/qim/compile/model.hbm
```

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在已有 QAT 量化模型基础上重新执行导出与编译。修改 `configs/motr_efficientnetb3/common.py` 的 `march` 字段为目标芯片：
> ```python
> march = March.NASH_M   # J6M
> march = March.NASH_P   # J6P
> march = March.NASH_B   # J6B
> ```
> 然后对**主图与 QIM 子图都**重新执行 `export_hbir.py` 与 `compile_perf.py`（即上方四条命令）。不同芯片的编译产物不同，性能不可跨芯片直接迁移。

### 精度验证

```bash
# Float 评测
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/motr_efficientnetb3 \
  --stage hbm
```

评测指标为 MOTA，gt 位于 `./tmp_data/heal_data/mot17/test_gt`。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的图像帧（保存到 `./demo/motr_efficientnetb3/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理帧（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/motr_efficientnetb3

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/motr_efficientnetb3 \
  --stage float
```

可视化结果保存到 `./demo/motr_efficientnetb3/`。
