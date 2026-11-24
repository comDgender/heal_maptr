# QCNetOE (轨迹预测)

本任务为基于 Argoverse 2 运动预测数据集的多模态轨迹预测任务，采用 QCNetOE 模型对场景中各智能体的未来轨迹进行预测。本文说明如何基于原始 Argoverse 2 数据集，完成「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理」的完整流程。

模型配置目录：`configs/qcnet_oe/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| QCNetOE | 流式场景表征张量组 `(B,A,pl,pt,HT)` | QCNetOEMapEncoder + QCNetOEAgentEncoderStream | — | 候选未来轨迹 `(B,A,6,12,2)` + 轨迹概率 |

### 精度指标

评测集：Argoverse 2 val（`./tmp_data/heal_data/argoverse2/val_lmdb`）。指标：HitRate（轨迹预测命中率，越高越好；本任务以 HitRate 为部署关注指标，其余 Brier/minADE/minFDE/MR 指标由评测脚本同时给出）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | HitRate | 0.8003 | 0.6817 | 0.7984 | 0.7981 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 3.63 | 300.69 | 32.10 |
| J6P | 2.67 | 1593.50 | 36.50 |
| J6B | 10.07 | 177.52 | 29.60 |

编译产物路径：
- HBIR：`./tmp_models/qcnet_oe/qat.bc`
- HBM：`./tmp_models/qcnet_oe/compile/model.hbm`

---

## 模型介绍

### 核心设计

QCNetOE 以 query-centric 方式对场景中每个 agent 编码其与周围地图、其他 agent 的相对关系，流式（streaming）地逐 agent 输出 encoder 隐变量，再由 decoder 解出多模态候选轨迹与概率；移除 `torch_geometric`/`torch_cluster` 依赖、消除大部分 index/gather/scatter 算子，使其对量化部署友好。

- **任务类型**：轨迹预测（Motion Forecasting，多模态轨迹预测）。
- **backbone**：`QCNetOEMapEncoder` + `QCNetOEAgentEncoderStream`（流式 streaming 推理，`stream_infer=True`；`hidden_dim=128`，`num_heads=8`，`head_dim=16`，`num_freq_bands=32`，`num_map_layers=1`，`num_agent_layers=1`，`time_span=2`，`dropout=0.1`；agent-map 交互 `num_pl2a=32`、agent-agent 交互 `num_a2a=36`）。
- **neck**：—（QCNetOE 无独立 neck，encoder 直接输出 decoder 输入）。
- **解码器**：`QCNetOEDecoder`（`num_dec_layers=1`），输出 `num_modes=6` 条候选轨迹及其概率。
- **预处理**：`QCNetOEPreprocess`（`stream=True`，构造 agent/map 的相对表征与时间-空间相对位置编码）。
- **后处理**：`QCNetOEPostprocess`（输出维度 `output_dim=2`，即预测轨迹的 (x, y)）。
- **损失**：`QCNetOELoss`。
- **模型输入**：流式场景表征张量组（`B=1, A=30` agent 数，`pl=80` 地图 polygon 数，`pt=50` polygon 点数，`HT=10` 历史步数），含 `agent`/`map_polygon`/`map_point`/`decoder` 等 OrderedDict 输入；历史 5s（`num_historical_steps=10`）+ 预测 6s（`num_future_steps=12`）。
- **模型输出**：每个待预测 agent 的 6 条候选未来轨迹（`num_modes=6`，12 步，2 维坐标 `output_dim=2`）+ 每条轨迹的概率。

**部署说明**：模型支持冷启动/热启动流式推理（`quant_infer_cold_start` 控制）。`cali_model` 与 `deploy_model` 均开启 `stream_infer=True`、`save_memory` 按训练/部署分别配置（训练 `save_memory=True` 节省显存，部署 `save_memory=False`）。HBIR 导出开启 `enable_vpu=True`，编译时 `input_source=["ddr"]`（轨迹预测输入从 DDR 读入，非 pyramid 图像输入），与图像类任务不同。

### 官方 repo 与论文链接

官方repo: https://github.com/ZikangZhou/QCNet
论文: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14089

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 Argoverse 2 打包后的 lmdb。模型读取的数据路径如下（均由 `common.py` 的 `data_rootdir = "./tmp_data/heal_data/argoverse2/"` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/argoverse2/train_lmdb` | 训练集 lmdb（`data.py` 的 `data_loader` / `qat_data_loader`） |
| `./tmp_data/heal_data/argoverse2/val_lmdb` | 评测集 lmdb（`data.py` 的 `val_data_loader` / `int_infer_data_loader`） |

> **目录约定**：`orig_data/argoverse2` 为 `--src-data-dir`（原始解压数据），`heal_data/argoverse2` 为 `--target-data-dir`（打包后 lmdb，供 config 读取）。Argoverse 2 packer 直接读取官方解压后的 parquet + 地图 json，**无需任何额外预处理脚本**，一次打包即可。

#### 官方下载

Argoverse 2 Motion Forecasting 数据集官方页面：<https://www.argoverse.org/av2.html#forecasting-link>。需下载 motion forecasting 子集（含 `train/`、`val/` 两个 split，每个 split 为若干 `log` 目录）。

> 官方下载链接与解压后目录结构以官方页面实际提供为准；若官方分发方式有更新，按官方说明解压至下方结构即可。

#### 解压后目录结构

将下载的 Argoverse 2 motion forecasting 数据解压到原始数据目录 `./tmp_data/orig_data/argoverse2/`（该目录作为 `--src-data-dir`）。每个 `log` 目录内含一个场景的 parquet（轨迹/标注）与地图 json：

```
./tmp_data/orig_data/argoverse2/
├── train/
│   └── <log_id>/                         # 每个场景一个目录
│       ├── scenario_<log_id>.parquet      # 场景轨迹与标注
│       └── log_map_archive_<log_id>.json  # 该场景的矢量化地图
└── val/
    └── <log_id>/
        ├── scenario_<log_id>.parquet
        └── log_map_archive_<log_id>.json
```

> `train/` 与 `val/` 下各有数万个 `<log_id>` 子目录；`<log_id>` 为不重复的场景标识。packer 遍历 `--src-data-dir/{split}/` 下所有子目录，逐场景读取 parquet + 地图 json 并提取 agent/map 特征。

#### 打包命令

用 `tools/datasets/argoverse2_packer.py`（`Argoverse2Packer`）把官方原始场景打成 lmdb。Argoverse 2 无需任何预处理脚本，直接调用 packer 即可。对 train 与 val 两个 split 各打一次：

```bash
# 打 train lmdb
python3 tools/datasets/argoverse2_packer.py \
  --src-data-dir ./tmp_data/orig_data/argoverse2 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/argoverse2 \
  --split-name train \
  --num-workers 10

# 打 val lmdb
python3 tools/datasets/argoverse2_packer.py \
  --src-data-dir ./tmp_data/orig_data/argoverse2 \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/argoverse2 \
  --split-name val \
  --num-workers 10
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/argoverse2`（含 `train/`、`val/` 两个子目录，每个子目录下为若干 `<log_id>` 场景目录）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/argoverse2`，packer 在其下产出 `{split}_lmdb`，与 `common.py` 的 `data_rootdir` 对应。
- `--pack-type`：打包类型，统一用 `lmdb`。
- `--split-name`：要打包的 split（`train` 或 `val`），packer 输出到 `{target}/{split}_lmdb`。
- `--num-workers`：并行打包进程数（默认 10）。

#### 目标结构

两次打包后，`./tmp_data/heal_data/argoverse2/` 结构如下：

```
./tmp_data/heal_data/argoverse2/
├── train_lmdb/      # 训练集 lmdb
└── val_lmdb/        # 评测集 lmdb
```

打包后建议抽样可视化校验 lmdb 是否正确。Argoverse 2 用 `tools/datasets/argoverse2_viz.py`（打包校验，复用 `heal/visualize/argoverse2.py` 的 `Argoverse2Viz`，画打包地图 + 历史/未来 GT 轨迹，无需模型）：

```bash
python3 tools/datasets/argoverse2_viz.py \
  --data-path ./tmp_data/heal_data/argoverse2/val_lmdb \
  --viz-num 10 \
  --save-dir ./viz_out/argoverse2
```

可视化 PNG 保存到 `--save-dir`（须为本地可写目录；`tmp_data/heal_data` 是只读 bucket，不能作为保存目标），每样本一个子目录 `sample_{i}/traj_pred.png`，可人工核对历史/未来轨迹与打包地图是否对齐。默认 `--num-historical-steps=10`、`--num-future-steps=12`、`--input-dim=2`、`--split=val`，与本任务 config 一致；如需改可经对应 flag 覆盖。

> 推理可视化（预测轨迹叠加到地图与历史轨迹上）由 `infer.py` 内置的 `QCNetViz`（封装 `heal/visualize/argoverse2.py` 的 `Argoverse2Viz`）完成，见「精度验证」段。

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | QAT 训练 epoch 数 | Float 优化器 | QAT 优化器 | 学习率调度 | Float SyncBN |
|---|---|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 16（`batch_size_per_gpu = 4 * 4`） | 75 | 100 | 15 | AdamW（`lr=5e-4`，`weight_decay=1e-4`；bias/norm/Embedding 权重衰减置 0） | AdamW（`lr=8*5e-6`，`weight_decay=1e-4`；bias/norm/Embedding 权重衰减置 0） | Float：`CosLrUpdater`（`warmup_len=1`，`warmup_by="epoch"`）<br>QAT：`StepDecayLrUpdater`（`lr_decay_id=[4]`） | 开启（`sync_bn=True`） |

> 本任务 Float 训练从零开始，**无 LoadCheckpoint**，不依赖外部预训练权重。Calibration 阶段加载本任务 Float 产物，QAT 阶段加载本任务 Calibration 产物，串行链，无跨任务依赖。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/qcnet_oe \
  --stage float
# 产物：./tmp_models/qcnet_oe/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/qcnet_oe \
  --stage calibration
# 产物：./tmp_models/qcnet_oe/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/qcnet_oe \
  --stage qat
# 产物：./tmp_models/qcnet_oe/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/qcnet_oe/deploy.py
# 产物：./tmp_models/qcnet_oe/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/qcnet_oe/deploy.py
# 产物：./tmp_models/qcnet_oe/compile/model.hbm
```

> 导出时 `enable_vpu=True`，编译输入 `input_source=["ddr"]`（轨迹预测输入从 DDR 读入），与图像类任务的 pyramid 输入不同。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在已有 QAT 量化模型基础上重新执行导出与编译。修改 `configs/qcnet_oe/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/qcnet_oe \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/qcnet_oe \
  --stage hbm
```

评测集为 Argoverse 2 val（`./tmp_data/heal_data/argoverse2/val_lmdb`），部署指标关注 `HitRate`（命中率，越高越好），checkpoint 以 `HitRate` 为监控指标选优（`mode="max"`）；评测脚本同时输出 `BrierMetric`/`MinADE`/`MinFDE`/`MissRate` 等辅助指标，但部署指标表只报告 `HitRate`。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

> HBM 阶段（`hbm_predictor`）通过 `QCNetOEHbmInfer` 封装 `.hbm` 板上模型，需板卡与 `hbm_infer` 运行时（`predict.py` 中 `HbmModule` 的 `host` 指向板卡地址）。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的场景数据（保存到 `./demo/qcnet_oe/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理场景（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/qcnet_oe

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/qcnet_oe \
  --stage float
```

可视化由 `infer.py` 内置的 `QCNetViz`（封装 `Argoverse2Viz`）完成，将预测轨迹叠加到历史轨迹与地图上，结果保存到 `./tmp_viz_results/qcnet_oe_float/`（各阶段对应 `_calibration`/`_qat`/`_hbir`/`_hbm` 子目录）。

#### 量化敏感性分析（可选）

本任务 `predict.py` 定义了 `quant_analysis_solver`，可逐层分析量化对精度的影响，辅助排查量化掉点来源：

```bash
python3 tools/quant_analysis.py \
  --config configs/qcnet_oe/predict.py
```

结果输出到 `./tmp_models/qcnet_oe/analysis/`。
