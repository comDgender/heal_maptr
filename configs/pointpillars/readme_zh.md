# PointPillars (Car)

本任务为基于点云的 3D 目标检测任务，采用 PointPillars 模型在 KITTI 3D object detection 数据集上检测 Car 类。文档说明如何从原始 KITTI 数据出发，完成「浮点训练 → 量化校准 → QAT → HBIR 导出 → HBM 编译 → 推理可视化」的完整流程。

模型配置目录：`configs/pointpillars/`

---

## 部署指标

### 模型参数

| 模型 | 模型输入 | backbone | neck | 模型输出 |
|---|---|---|---|---|
| PointPillars | 单帧 LiDAR 点云 `(N,4)` | PointPillarScatter | SECONDNeck | Car 3D 检测框 `(B,N,cls+reg)` |

### 精度指标

评测集：KITTI val（`./tmp_data/heal_data/kitti3d/val_lmdb`）。指标：KITTI 3D AP（Car）。

| March | 评测指标 | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | Kitti3D AP (Car) | 0.7731 | 0.7569 | 0.7709 | 0.7707 |

> 数据基于 `march = March.NASH_M`（J6M）配置测试。
>
> HEAL 版本：heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5（板端 UCP 3.16.2）。

### 性能指标

> **性能测试口径**：FPS 均为单核八线程，Latency 为单核单线程测试，Memory 为峰值 DDR 占用。
> 数据来源：`tools/test_perf_all.py` 工具（默认采样间隔 `-d 200ms`，perf-time 1）。

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 22.51 | 213.05 | 55.80 |
| J6P | 20.47 | 345.73 | 55.80 |
| J6B | 1353.52 | 2.75 | 63.40 |

编译产物路径：
- HBIR：`./tmp_models/pointpillars/qat.bc`
- HBM：`./tmp_models/pointpillars/compile/model.hbm`

---

## 模型介绍

### 核心设计

PointPillars 把点云体素化为柱体（pillar），用 `PillarFeatureNet` 学到柱特征后经 `PointPillarScatter` 展开为伪图像，再由 SECOND 风格 FPN（`SECONDNeck`）与 `PointPillarsHead` 做多尺度特征提取与 3D 框回归；训练以 FocalLoss + SmoothL1Loss + 方向分类联合监督。

- **任务类型**：3D 目标检测（LiDAR 点云 3D Object Detection）。
- **backbone**：`PointPillarScatter`（将 `PillarFeatureNet` 学到的 64 维柱特征按坐标 scatter 到 `H×W×64` 伪图像特征图，`use_horizon_pillar_scatter=True`）。
- **neck**：`SECONDNeck`（SECOND 风格的 FPN：下采样层通道 `[64, 128, 256]`、步长 `[2, 2, 2]`，上采样层通道 `[128, 128, 128]`、步长 `[1, 2, 4]`，输出三尺度特征拼接为 384 通道）。
- **点云预处理**：`PointPillarsPreProcess`（体素化：`pc_range=[0, -39.68, -3, 69.12, 39.68, 1]`，`voxel_size=[0.16, 0.16, 4]`，每 voxel 最多 100 个点，最多 12000 个 voxel）。
- **特征提取**：`PillarFeatureNet`（`num_filters=(64,)`，4 维输入，MLP + max-pool 得到 64 维柱特征）。
- **检测头**：`PointPillarsHead`（`in_channels=384`，`use_direction_classifier=True`，输出分类 + 框回归 + 方向分类）。
- **Anchor 生成**：`Anchor3DGeneratorStride`：Car 类 anchor 尺寸 `1.6×3.9×1.56`，stride `[0.32, 0.32, 0.0]`，offset `[0.16, -39.52, -1.78]`，旋转角 `[0, 1.57]`；匹配阈值 `0.6`，失配阈值 `0.45`。
- **后处理**：`PointPillarsPostProcess`：NMS（`nms_iou_threshold=0.5`，`score_threshold=0.4`，`nms_pre_max_size=1000`，`nms_post_max_size=300`，`max_per_img=100`）。
- **损失**：`FocalLoss`（分类，`alpha=0.25, gamma=2.0, weight=1.0`）+ `SmoothL1Loss`（框回归，`beta=1/9, weight=2.0`）+ `CrossEntropyLoss`（方向，`weight=0.2`）。
- **模型输入**：单帧 LiDAR 点云，形状 `(N, 4)`，N 为点数（部署输入按 `150000` 点 padding），4 维 = `[x, y, z, intensity]`。
- **模型输出**：每帧检测到的 Car 3D 框（`x, y, z, w, l, h, θ`）+ 分数 + 方向。
- **类别**：`["Car"]`。

**部署说明**：PointPillars 部署图含完整的 voxel 化 + 特征提取 + backbone + head + 后处理。HBIR 导出开启 `enable_vpu=True`，编译时 `input_source=["ddr"]`（点云从 DDR 读入，非 pyramid 图像输入），与图像类任务不同。本任务无子图导出，仅一个 `deploy.py`。

### 官方 repo 与论文链接

官方repo: https://github.com/nutonomy/second.pytorch
论文: https://arxiv.org/abs/1812.05784v1

### 参考算法

如需更多 J6 芯片部署细节，可以参考 https://developer.horizon.auto/blog/14086

---

## 使用指南

### 环境准备

请先完成 HEAL 环境安装（详见顶层 [`README.md`](../../README.md)）。以下命令均在 **HEAL 仓库根目录**执行：

### 数据处理

本任务使用 KITTI 3D 打包后的 lmdb 与若干中间产物（info pkl / velodyne_reduced / gt_database / dbinfos）。模型读取的数据路径如下（均由 `common.py` 的 `base_data_dir = "./tmp_data/heal_data/kitti3d/"` 拼接）：

| 路径 | 用途 |
|---|---|
| `./tmp_data/heal_data/kitti3d/train_lmdb` | 训练集 lmdb（`data.py` 的 `data_loader`） |
| `./tmp_data/heal_data/kitti3d/val_lmdb` | 评测集 lmdb（`data.py` 的 `val_data_loader`） |
| `./tmp_data/heal_data/kitti3d/kitti3d_dbinfos_train.pkl` | `DataBaseSampler` 训练时读取的 GT 采样信息（`data.py`） |
| `./tmp_data/heal_data/kitti3d/kitti3d_gt_database/` | `DataBaseSampler` 训练时读取的 GT 点云 `.bin`（`data.py`） |
| `./tmp_data/heal_data/kitti3d/kitti3d_infos_{train,val,test,trainval}.pkl` | 数据集元信息（打包时读取） |

> **目录约定**：`orig_data/kitti3d` 为 `--src-data-dir`（原始解压数据），`heal_data/kitti3d` 为 `--target-data-dir`（打包 lmdb + 中间产物，供 config 读取）。KITTI 的中间产物（reduced 点云 / info pkl / gt_database）与原始数据强耦合（`velodyne_reduced` 与 `velodyne` 同树、info pkl 记录相对路径），故 `--prepare-data` 对 src 目录执行 `create_*` 生成中间产物后，再把它们拷贝到 target 目录，保证 target 完整自含。

#### 官方下载

KITTI 3D object detection 数据集官方页面：<http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>。需下载 4 个文件：

- `data_object_image_2.zip`（训练 + 测试图像）
- `data_object_label_2.zip`（训练标签）
- `data_object_calib.zip`（标定文件）
- `data_object_velodyne.zip`（点云）

KITTI 官方不直接提供 train/val/test 划分，需从社区获取 ImageSets 划分文件（来源 `second.pytorch`）：

```bash
mkdir -p ./tmp_data/orig_data/kitti3d/ImageSets
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/test.txt
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/train.txt
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/val.txt
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate -O ./tmp_data/orig_data/kitti3d/ImageSets/trainval.txt
```

#### 解压后目录结构

将 4 个 zip 解压到原始数据目录 `./tmp_data/orig_data/kitti3d/`（该目录作为 `--src-data-dir`），并放入上一步下载的 ImageSets：

```
./tmp_data/orig_data/kitti3d/
├── ImageSets/{train,val,test,trainval}.txt   # 社区划分文件
├── training/
│   ├── calib/      # data_object_calib.zip 的 training/calib
│   ├── image_2/    # data_object_image_2.zip 的 training/image_2
│   ├── label_2/    # data_object_label_2.zip 的 training/label_2
│   └── velodyne/   # data_object_velodyne.zip 的 training/velodyne
└── testing/
    ├── calib/
    ├── image_2/
    └── velodyne/   # 官方 test 无 label_2，标签不公开
```

解压命令示例：

```bash
mkdir -p ./tmp_data/orig_data/kitti3d
unzip data_object_image_2.zip -d ./tmp_data/orig_data/kitti3d
unzip data_object_label_2.zip -d ./tmp_data/orig_data/kitti3d
unzip data_object_calib.zip -d ./tmp_data/orig_data/kitti3d
unzip data_object_velodyne.zip -d ./tmp_data/orig_data/kitti3d
```

#### 打包命令

KITTI 3D 数据处理分三步，均用 `tools/datasets/kitti3d_packer.py`（`Kitti3DDetectionPacker`）。

**第 1 步：首次准备数据（生成中间产物，与 split 无关，只跑一次）**

`--prepare-data` 对 src 目录依次执行 `create_kitti_info_file`（生成 `kitti3d_infos_{train,val,test,trainval}.pkl`）→ `create_reduced_point_cloud`（生成 `training/velodyne_reduced/`、`testing/velodyne_reduced/`）→ `create_groundtruth_database`（生成 `kitti3d_gt_database/` + `kitti3d_dbinfos_train.pkl`），并把 info pkl / gt_database / dbinfos **拷贝**到 target 目录。此步耗时较长（遍历全部样本），与 split 无关，**只需执行一次**。

```bash
python3 tools/datasets/kitti3d_packer.py \
  --prepare-data \
  --src-data-dir ./tmp_data/orig_data/kitti3d \
  --target-data-dir ./tmp_data/heal_data/kitti3d
```

**第 2 步：打 train lmdb**

打包时 packer 读 src 下的 info pkl + velodyne_reduced + 原始 calib/image，打包成 lmdb 输出到 target。

```bash
python3 tools/datasets/kitti3d_packer.py \
  --src-data-dir ./tmp_data/orig_data/kitti3d \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/kitti3d \
  --split-name train \
  --num-workers 20
```

**第 3 步：打 val lmdb**

```bash
python3 tools/datasets/kitti3d_packer.py \
  --src-data-dir ./tmp_data/orig_data/kitti3d \
  --pack-type lmdb \
  --target-data-dir ./tmp_data/heal_data/kitti3d \
  --split-name val \
  --num-workers 20
```

参数说明：
- `--src-data-dir`：原始数据根目录 `./tmp_data/orig_data/kitti3d`（含 `ImageSets/`、`training/`、`testing/`）。
- `--target-data-dir`：打包产物目录 `./tmp_data/heal_data/kitti3d`，产出 `train_lmdb` 与 `val_lmdb`，以及 prepare-data 拷贝来的 info pkl / gt_database / dbinfos，与 `common.py` 的 `base_data_dir` 对应。
- `--prepare-data`：`store_true` 开关，置位时只执行中间产物生成与拷贝，不打 lmdb。
- `--num-workers`：并行打包进程数（默认 20）。

#### 目标结构

`--prepare-data` + 两次打包后，`./tmp_data/heal_data/kitti3d/` 结构如下：

```
./tmp_data/heal_data/kitti3d/
├── kitti3d_infos_train.pkl         # prepare-data 拷贝来
├── kitti3d_infos_val.pkl
├── kitti3d_infos_test.pkl
├── kitti3d_infos_trainval.pkl
├── kitti3d_dbinfos_train.pkl       # DataBaseSampler 训练时读
├── kitti3d_gt_database/            # DataBaseSampler 读的 .bin
├── train_lmdb/                     # 打包产物
└── val_lmdb/
```

可用可视化脚本抽样检查打包结果是否正确（可视化点云与 GT 3D 框对齐）：

```bash
python3 tools/datasets/kitti3d_viz.py \
  --data-path ./tmp_data/heal_data/kitti3d/train_lmdb \
  --viz-num 50
```

### 训练/量化流程

关键训练超参：

| 训练 GPU 数 | 每卡 batch size | Float 训练 epoch 数 | Calibration 步数 | QAT 训练 epoch 数 | Float 优化器 | QAT 优化器 | 学习率调度 | Float SyncBN |
|---|---|---|---|---|---|---|---|---|
| 8（`device_ids=[0,1,2,3,4,5,6,7]`） | 4 | 300 | 50 | 50 | AdamW（`lr=5e-4`，`betas=(0.95, 0.99)`，`weight_decay=0.01`） | SGD（`lr=2e-4`，`momentum=0.9`，`weight_decay=0`） | `CyclicLrUpdater`（`target_ratio=(10, 1e-4)`，`step_ratio_up=0.4`） | 开启 |

> 本任务 Float 训练从零开始，**无 LoadCheckpoint**，不依赖外部预训练权重。Calibration 阶段加载本任务 Float 产物，QAT 阶段加载本任务 Calibration 产物（串行链，无跨任务依赖）。

```bash
# Float 训练
python3 tools/train.py \
  --config configs/pointpillars \
  --stage float
# 产物：./tmp_models/pointpillars/float-checkpoint-best.pth.tar

# 量化校准（Calibration）
python3 tools/train.py \
  --config configs/pointpillars \
  --stage calibration
# 产物：./tmp_models/pointpillars/calibration-checkpoint-best.pth.tar

# QAT 训练
python3 tools/train.py \
  --config configs/pointpillars \
  --stage qat
# 产物：./tmp_models/pointpillars/qat-checkpoint-best.pth.tar
```

### 导出编译

```bash
# 导出 HBIR
python3 tools/export_hbir.py \
  --config configs/pointpillars/deploy.py
# 产物：./tmp_models/pointpillars/qat.bc

# 编译 HBM
python3 tools/compile_perf.py \
  --config configs/pointpillars/deploy.py
# 产物：./tmp_models/pointpillars/compile/model.hbm
```

> 导出时 `enable_vpu=True`，编译输入 `input_source=["ddr"]`（点云从 DDR 读入），与图像类任务的 pyramid 输入不同。

> **切换芯片（march）tips**：切换芯片只影响 HBIR 导出与 HBM 编译两个步骤，**不影响此前的浮点训练与量化（calibration/qat）**，因此切换 J6M/J6P/J6B 时无需重跑 training/calibration/qat，只需在已有 QAT 量化模型基础上重新执行导出与编译。修改 `configs/pointpillars/common.py` 的 `march` 字段为目标芯片：
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
  --config configs/pointpillars \
  --stage float

# Calibration 评测
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage calibration

# QAT 评测
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage qat

# HBIR 评测（需先完成「导出编译」段导出）
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage hbir

# HBM 评测（需先完成「导出编译」段编译，且需板卡/hbm_infer 运行时）
python3 tools/predict.py \
  --config configs/pointpillars \
  --stage hbm
```

评测集为 KITTI val（`./tmp_data/heal_data/kitti3d/val_lmdb`），指标为 `Kitti3DMetricDet`（`compute_aos=True`，`current_classes=["Car"]`，`difficultys=[0,1,2]`），checkpoint 以 `mAP_3D_moderate` 为监控指标选优。评测命令默认遵循 config 的 `device_ids`；其中 calibration 训练及部分 hbir/hbm 评测阶段按 config 实际定义为单进程运行（以 `data.py` 为准），如需指定评测 GPU 可通过 `--device-ids` 覆盖。

**推理与可视化**：推理前需先用 `gen_demo.py` 从评测集生成推理用的点云帧（保存到 `./demo/pointpillars/`），再由 `inference.py` 读取该目录完成推理与可视化。

```bash
# 1. 生成推理帧（从 infer.py 的 gen_demo_cfg 读取评测集样本，存为 ./demo/<task>/）
python3 tools/gen_demo.py \
  --config configs/pointpillars

# 2. 推理与可视化（以 float 为例；可用 calibration/qat/hbir/hbm 替换 --stage）
python3 tools/inference.py \
  --config configs/pointpillars \
  --stage float
```

可视化结果保存到 `./tmp_viz_results/pointpillars_hbir/`。`infer.py` 还含 `bpu_val_preprocess` 逻辑（生成 `reduced_velodyne`/`gt`/`lst` 等评测辅助文件，用于板上 BPU 评测对齐）。
