import copy
import os
from collections import OrderedDict

import torch
from common import (
    HT,
    A,
    B,
    ckpt_dir,
    dropout,
    head_dim,
    hidden_dim,
    input_dim,
    num_a2a,
    num_agent_layers,
    num_dec_layers,
    num_freq_bands,
    num_future_steps,
    num_heads,
    num_historical_steps,
    num_map_layers,
    num_modes,
    num_pl2a,
    num_recurrent_steps,
    num_t2m_steps,
    output_dim,
    pl,
    pt,
    quant_infer_cold_start,
    reuse_agent_rembs,
    split_rec_modules,
    time_span,
)

model = dict(
    type="QCNetOE",
    encoder=dict(
        type="QCNetOEEncoder",
        map_encoder=dict(
            type="QCNetOEMapEncoder",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        ),
        agent_encoder=dict(
            type="QCNetOEAgentEncoderStream",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_t2m_steps=num_t2m_steps,
            num_pl2a=num_pl2a,
            num_a2a=num_a2a,
            dropout=dropout,
            save_memory=True,
            stream_infer=True,
            reuse_agent_rembs=reuse_agent_rembs,
            deploy=False,
            quant_infer_cold_start=quant_infer_cold_start,
        ),
    ),
    decoder=dict(
        type="QCNetOEDecoder",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
        num_modes=num_modes,
        num_recurrent_steps=num_recurrent_steps,
        num_t2m_steps=num_t2m_steps,
        num_freq_bands=num_freq_bands,
        num_layers=num_dec_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=dropout,
        split_rec_modules=split_rec_modules,
        reuse_agent_rembs=reuse_agent_rembs,
        deploy=False,
        quant_infer_cold_start=quant_infer_cold_start,
    ),
    loss=dict(
        type="QCNetOELoss",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
    ),
    preprocess=dict(
        type="QCNetOEPreprocess",
        input_dim=input_dim,
        num_historical_steps=num_historical_steps,
        time_span=time_span,
        num_t2m_steps=num_t2m_steps,
        num_agent_layers=num_agent_layers,
        num_pl2a=num_pl2a,
        num_a2a=num_a2a,
        stream=True,
        save_memory=True,
        deploy=False,
        quant_infer_cold_start=quant_infer_cold_start,
    ),
    postprocess=dict(
        type="QCNetOEPostprocess",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
    ),
    quant_infer_cold_start=quant_infer_cold_start,
)


cali_model = copy.deepcopy(model)
cali_model["encoder"]["agent_encoder"]["save_memory"] = False
cali_model["encoder"]["agent_encoder"]["stream_infer"] = True
cali_model["preprocess"]["save_memory"] = False
cali_model["preprocess"]["stream"] = True

deploy_model = dict(
    type="QCNetOE",
    encoder=dict(
        type="QCNetOEEncoder",
        map_encoder=dict(
            type="QCNetOEMapEncoder",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        ),
        agent_encoder=dict(
            type="QCNetOEAgentEncoderStream",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_t2m_steps=num_t2m_steps,
            dropout=dropout,
            save_memory=False,
            stream_infer=True,
            reuse_agent_rembs=reuse_agent_rembs,
            deploy=True,
            quant_infer_cold_start=quant_infer_cold_start,
        ),
    ),
    decoder=dict(
        type="QCNetOEDecoder",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
        num_modes=num_modes,
        num_recurrent_steps=num_recurrent_steps,
        num_t2m_steps=num_t2m_steps,
        num_freq_bands=num_freq_bands,
        num_layers=num_dec_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=dropout,
        split_rec_modules=split_rec_modules,
        reuse_agent_rembs=reuse_agent_rembs,
        deploy=True,
        quant_infer_cold_start=quant_infer_cold_start,
    ),
    quant_infer_cold_start=quant_infer_cold_start,
)


deploy_inputs = OrderedDict()
# Initialize agent as an ordered dictionary
deploy_inputs["agent"] = OrderedDict(
    {
        "valid_mask": (torch.rand(1, A, HT) > 0.2).bool(),  # 0
        "valid_mask_a2a": (torch.rand(1, HT, A, A) > 0.2).bool(),  # 1
        "agent_type": (torch.rand(1, A, 1) * 3).long(),  # 2
        "x_a_cur": [torch.rand(B, 1, A, 1) for _ in range(4)],  # 3 4 5 6
        "r_pl2a_cur": [torch.rand(B, 1, A, pl) for _ in range(3)],  # 7 8 9
        "r_t_cur": [
            torch.rand(B, 1, A, num_t2m_steps) for _ in range(4)
        ],  # 10 11 12 13 stream and reuse
        "r_a2a_cur": [torch.rand(B, 1, A, A) for _ in range(3)],  # 14 15 16
        "mask_a_cur": (torch.rand(B, A) > 0.2).bool(),  # [B, A]   17
        "mask_a2a_cur": (torch.rand(B, A, A) > 0.2).bool(),  # [B, A, A] 18
        "mask_t_key": (torch.rand(B, A, 2) > 0.2).bool(),  # [B, A, 2]   19
        "x_a_mid_emb": [
            torch.rand(B, A, time_span, hidden_dim)
            for _ in range(num_agent_layers)
        ],  # 20
    }
)

deploy_inputs["agent"]["x_a_his"] = torch.zeros(
    B, A, num_t2m_steps - 1, hidden_dim
)  # 21

# Initialize map_polygon as an ordered dictionary
deploy_inputs["map_polygon"] = OrderedDict(
    {
        "pl_type": (torch.rand(1, pl) * 3).long(),  # 22
        "is_intersection": (torch.rand(1, pl)).long(),  # 23
        "r_pl2pl": [torch.rand(B, 1, pl, pl) for _ in range(3)],  # 24 25 26
        "r_pt2pl": [torch.rand(B, 1, pl, pt) for _ in range(3)],  # 27 28 29
        "mask_pl2pl": (
            torch.ones([B, pl, pl]) - torch.eye(pl).unsqueeze(0)
        ).bool(),  # 30
    }
)
# Initialize map_point as an ordered dictionary
deploy_inputs["map_point"] = OrderedDict(
    {
        "magnitude": torch.rand(1, pl, pt),  # 31
        "pt_type": (torch.rand(1, pl, pt) * 3).long(),  # 32
        "side": (torch.rand(1, pl, pt) * 3).long(),  # 33
        "mask": (torch.rand(1, pl, pt) > 0.2).bool(),  # 34
    }
)
# Initialize decoder as an ordered dictionary
deploy_inputs["decoder"] = OrderedDict(
    {
        "mask_a2m": (torch.rand(1, A, A) > 0.2).bool(),  # 35
        "mask_dst": (torch.rand(1, A, 1) > 0.2).bool(),  # 36
    }
)
# Conditionally add additional keys if reuse_agent_rembs is False
if not reuse_agent_rembs:
    deploy_inputs["decoder"].update(
        {
            "r_t2m": [torch.rand(1, 1, A, num_t2m_steps) for _ in range(4)],
            "r_pl2m": [torch.rand(1, 1, A, pl) for _ in range(3)],
            "r_a2m": [torch.rand(1, 1, A, A) for _ in range(3)],
        }
    )

deploy_inputs["type_pl2pl"] = (torch.rand([B, pl, pl]) * 5).long()  # 37


example_tmp_inputs = copy.deepcopy(deploy_inputs)
example_tmp_inputs["map_point"]["orientation"] = torch.rand(1, pl, pt)
example_tmp_inputs["map_point"]["position"] = torch.rand(B, pl, pt, 2)
example_tmp_inputs["map_point"]["height"] = torch.rand(1, pl, pt)

example_tmp_inputs["agent"]["num_nodes"] = list(range(16))
example_tmp_inputs["agent"]["position"] = torch.rand(B, A, 22, 2)
example_tmp_inputs["agent"]["velocity"] = torch.rand(B, A, 22, 2)
example_tmp_inputs["agent"]["category"] = torch.rand(B, A, 1)
example_tmp_inputs["agent"]["predict_mask"] = (
    torch.rand(B, A, 22) > 0.2
).bool()
example_tmp_inputs["agent"]["heading"] = torch.rand(B, A, 22)

example_tmp_inputs["map_polygon"]["height"] = torch.rand(B, pl)
example_tmp_inputs["map_polygon"]["orientation"] = torch.rand(B, pl)
example_tmp_inputs["map_polygon"]["position"] = torch.rand(B, pl, 2)
example_tmp_inputs["map_polygon"]["valid_mask"] = (
    torch.rand(B, pl) > 0.2
).bool()


# --------------------------- int inference model ---------------------------
hbir_infer_model = dict(
    type="QCNetOEIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    preprocess=dict(
        type="QCNetOEPreprocess",
        input_dim=input_dim,
        num_historical_steps=num_historical_steps,
        time_span=time_span,
        num_t2m_steps=num_t2m_steps,
        num_agent_layers=num_agent_layers,
        agent_num=A,
        pl_num=pl,
        pt_num=pt,
        stream=True,
        deploy=True,
        quant_infer_cold_start=True,
    ),
    postprocess=dict(
        type="QCNetOEPostprocess",
        output_dim=output_dim,
        num_historical_steps=num_historical_steps,
    ),
    quant_infer_cold_start=True,
    example_data=deploy_inputs,
)
