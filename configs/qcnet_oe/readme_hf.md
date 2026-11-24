# QCNetOE (Trajectory Prediction)

QCNetOE encodes each agent's relative relationships with surrounding map and other agents in a query-centric manner, streaming encoder hidden states agent-by-agent, then the decoder outputs multimodal candidate trajectories and probabilities; removes `torch_geometric`/`torch_cluster` dependencies and eliminates most index/gather/scatter ops for quantization-friendly deployment.

---

## Deployment Metrics

### Model Parameters

| Model | Model Input | Backbone | Neck | Model Output |
|---|---|---|---|---|
| QCNetOE | Streaming scene representation tensor group `(B,A,pl,pt,HT)` | QCNetOEMapEncoder + QCNetOEAgentEncoderStream | — | Candidate future trajectories `(B,A,6,12,2)` + trajectory probabilities |

### Accuracy Metrics

| March | Metric | float | calibration | qat | hbm |
| --- | --- | --- | --- | --- | --- |
| J6M | HitRate | 0.8003 | 0.6817 | 0.7984 | 0.7981 |

> Data measured with `march = March.NASH_M` (J6M) configuration.
>
> HEAL version: heal 0.6.0 / hbdk4-compiler 4.13.1 / horizon_plugin_pytorch 3.4.5 (board UCP 3.16.2).

### Performance Metrics

> **Performance test methodology**: FPS is measured with single-core eight-thread; Latency is measured with single-core single-thread; Memory is peak DDR usage.
> Data source: `tools/test_perf_all.py` (default sampling interval `-d 200ms`, perf-time 1).

| March | latency (ms) | fps | Memory Usage (MB) |
|---|---|---|---|
| J6M | 3.63 | 300.69 | 32.10 |
| J6P | 2.67 | 1593.50 | 36.50 |
| J6B | 10.07 | 177.52 | 29.60 |

---

## Model Overview

### Core Design

QCNetOE encodes each agent's relative relationships with surrounding map and other agents in a query-centric manner, streaming encoder hidden states agent-by-agent, then the decoder outputs multimodal candidate trajectories and probabilities; removes `torch_geometric`/`torch_cluster` dependencies and eliminates most index/gather/scatter ops for quantization-friendly deployment.

- **Task type**: Trajectory prediction (Motion Forecasting, multimodal trajectory prediction).
- **backbone**: `QCNetOEMapEncoder` + `QCNetOEAgentEncoderStream` (streaming inference, `stream_infer=True`; `hidden_dim=128`, `num_heads=8`, `head_dim=16`, `num_freq_bands=32`, `num_map_layers=1`, `num_agent_layers=1`, `time_span=2`, `dropout=0.1`; agent-map interaction `num_pl2a=32`, agent-agent interaction `num_a2a=36`).
- **neck**: — (QCNetOE has no standalone neck; encoder output feeds directly into decoder).
- **Decoder**: `QCNetOEDecoder` (`num_dec_layers=1`), outputs `num_modes=6` candidate trajectories and their probabilities.
- **Preprocessing**: `QCNetOEPreprocess` (`stream=True`, constructs agent/map relative representations and spatiotemporal relative position encoding).
- **Post-processing**: `QCNetOEPostprocess` (output dimension `output_dim=2`, i.e. predicted trajectory (x, y)).
- **Loss**: `QCNetOELoss`.
- **Model input**: Streaming scene representation tensor group (`B=1, A=30` agents, `pl=80` map polygons, `pt=50` polygon points, `HT=10` history steps), including `agent`/`map_polygon`/`map_point`/`decoder` etc. as OrderedDict inputs; 5s history (`num_historical_steps=10`) + 6s prediction (`num_future_steps=12`).
- **Model output**: 6 candidate future trajectories per agent to predict (`num_modes=6`, 12 steps, 2-dim coordinates `output_dim=2`) + probability per trajectory.

**Deployment notes**: Model supports cold-start/hot-start streaming inference (`quant_infer_cold_start` controls). Both `cali_model` and `deploy_model` enable `stream_infer=True`; `save_memory` configured separately for training/deployment (training `save_memory=True` saves GPU memory, deployment `save_memory=False`). HBIR export enables `enable_vpu=True`; compilation uses `input_source=["ddr"]` (trajectory prediction input read from DDR, not pyramid image input), unlike image-based tasks.

### Official Repo and Paper

Official repo: https://github.com/ZikangZhou/QCNet
Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf

### Reference

For more J6 chip deployment details, see https://developer.horizon.auto/blog/14089
