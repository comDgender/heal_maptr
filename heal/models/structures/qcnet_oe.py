# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.global_var import get_value
from torch.quantization import DeQuantStub
from torch.utils._pytree import tree_flatten

try:
    from hbdk4.compiler import load
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
    )
except ImportError:
    load = None
    get_hbir_input_flattener = None
    get_hbir_output_unflattener = None

__all__ = ["QCNetOE", "QCNetOEIrInfer", "QCNetOEHbmInfer"]


def _init_streaming_state(B, A, D, time_span, num_t2m_steps, device):
    """Initialize the agent-encoder streaming state queues for cold-start.

    Returns:
        Tuple (x_a_mid_emb, x_a_his_enc) of deques pre-filled with zeros,
        sized for cold-start streaming inference.
    """
    x_a_mid_emb = deque(maxlen=time_span)
    x_a_his_enc = deque(maxlen=num_t2m_steps - 1)
    for _ in range(num_t2m_steps - 1):
        x_a_his_enc.append(torch.zeros((B, A, 1, D), device=device))
    for _ in range(time_span):
        x_a_mid_emb.append(torch.zeros((B, A, 1, D), device=device))
    return x_a_mid_emb, x_a_his_enc


@OBJECT_REGISTRY.register
class QCNetOE(nn.Module):
    """
    Implements the toolchain version trajectory prediction Model QCNet.

    Refer to paper `Query-Centric Trajectory Prediction
    <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf>`,  # noqa
    and `official GitHub repo <https://github.com/ZikangZhou/QCNet>`.

    Compared to the official QCNet, this QCNetOE removes the dependencies on
    `torch_geometric` and `torch_cluster` libraries. Most of the index, gather,
    and scatter operations in the model have been eliminated, making deployment
    more friendly.

    Args:
        encoder: Encoder module for the model.
        decoder: Decoder module for the model.
        preprocess: Pre-processing module for the model.
        postprocess: Post-processing module for the model.
        loss: Loss module for training the model.
        quant_infer_cold_start: Indicates whether to use cold start for streaming inference.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        preprocess: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        quant_infer_cold_start: bool = False,
    ) -> None:
        super(QCNetOE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.quant_infer_cold_start = quant_infer_cold_start

        self.preprocess = preprocess
        self.postprocess = postprocess

        self.dequant = DeQuantStub()

    def forward(self, data: dict):
        deploy = self.encoder.agent_encoder.deploy

        if self.quant_infer_cold_start and not deploy:
            #  predict and cold_start
            B, A = data["agent"]["position"].shape[:2]
            D = self.encoder.agent_encoder.hidden_dim
            num_t2m_steps = self.decoder.num_t2m_steps
            time_span = self.encoder.agent_encoder.time_span
            num_historical_steps = self.decoder.num_historical_steps
            cur_step = (
                num_historical_steps - num_t2m_steps - time_span
            )  # init 2
            cur_device = data["agent"]["valid_mask"].device
            x_a_mid_emb, x_a_his_enc = _init_streaming_state(
                B, A, D, time_span, num_t2m_steps, cur_device
            )
            map_enc = None

            for i in range(num_t2m_steps + time_span):
                cur_step += 1
                cur_inputs = self.preprocess(
                    data, cur_step, his_model_input=False
                )
                cur_inputs["cur_step"] = cur_step
                cur_inputs["agent"]["x_a_mid_emb"] = []
                cur_inputs["agent"]["x_a_mid_emb"].append(
                    torch.cat(list(x_a_mid_emb), dim=2)
                )  # [B, A, 2, D]

                cur_inputs["agent"]["x_a_his"] = torch.cat(
                    list(x_a_his_enc), dim=2
                )  # [B, A, 5, D]

                cur_inputs["map_enc"] = map_enc

                scene_enc = self.encoder(
                    cur_inputs
                )  # scene_enc: x_pt, x_pl, x_a_cur_emb, x_a_cur_enc
                if i == 0:
                    map_enc = {
                        "x_pl": scene_enc["x_pl"],  # [B, pl, D]
                        "x_pt": scene_enc["x_pt"],  # [B, ol, pt, D]
                    }

                x_a_mid_emb.append(scene_enc["x_a_cur_emb"])  # [B, A, 1, D]
                x_a_his_enc.append(
                    self.dequant(scene_enc["x_a_cur"])  # [B, A, 1, D]
                )  # [B, A, 1, D]

                if i == num_t2m_steps + time_span - 1:
                    pred = self.decoder(cur_inputs, scene_enc)
                    if get_value("preparing_qat"):
                        return pred
                    if self.training and self.loss is not None:
                        loss_dict = self.loss(pred, data)
                        return loss_dict
                    else:
                        if self.postprocess:
                            pred = self.postprocess(pred, data)
                        return pred
        elif self.quant_infer_cold_start and deploy:
            # export hbir and cold start
            scene_enc = self.encoder(data)
            output = {
                "x_a_cur": self.dequant(scene_enc["x_a_cur"]),  # [B, A, 1, D]
                "x_a_cur_emb": self.dequant(
                    scene_enc["x_a_cur_emb"]
                ),  # [B, A, 1, D]]
            }
            pred = self.decoder(data, scene_enc)
            pred.update(output)
            return pred
        else:
            if self.preprocess is not None:
                data = self.preprocess(data, cur_step=None)
            scene_enc = self.encoder(data)

            if self.decoder is not None:
                pred = self.decoder(data, scene_enc)
            else:
                return {
                    "x_a": self.dequant(scene_enc["x_a"]),
                    "x_a_mid_emb": [
                        self.dequant(h) for h in scene_enc["x_a_mid_emb"]
                    ],
                }
            if self.training and self.loss is not None:
                loss_dict = self.loss(pred, data)
                return loss_dict
            else:
                if self.postprocess:
                    pred = self.postprocess(pred, data)
                return pred

    def set_qconfig(self):
        self.encoder.set_qconfig()
        if self.decoder is not None:
            self.decoder.set_qconfig()


@OBJECT_REGISTRY.register
class QCNetOEIrInfer(nn.Module):
    """
    The basic structure of QCNetOEIrInfer.

    Args:
        ir_model: The ir model of QCNet.
        preprocess: Preprocess module.
        postprocess: Postprocess module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        preprocess: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        quant_infer_cold_start: bool = False,
        example_data: Optional[dict] = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.quant_infer_cold_start = quant_infer_cold_start
        self.example_data = example_data

    def forward(self, data: dict):
        if self.quant_infer_cold_start:
            num_historical_steps = self.example_data["agent"][
                "valid_mask"
            ].shape[
                2
            ]  # history steps
            num_t2m_steps = (
                self.example_data["agent"]["x_a_his"].shape[2] + 1
            )  # steps for decoder
            B, A, time_span, D = self.example_data["agent"]["x_a_mid_emb"][
                0
            ].shape[:]
            cur_step = (
                num_historical_steps - num_t2m_steps - time_span
            )  # current steps of stream infer (init 2)

            # Store the agent encoder mid outputs (Embedding) \
            # for the current time step
            x_a_mid_emb_list = deque(maxlen=time_span)
            cur_device = data["agent"]["valid_mask"].device
            for _ in range(time_span):
                x_a_mid_emb_list.append(
                    torch.zeros((B, A, 1, D), device=cur_device)
                )
            x_a_his_enc = (
                []
            )  # Store the agent encoder outputs for the current time step

            x_a_mid_emb, x_a_his_enc = _init_streaming_state(
                B, A, D, time_span, num_t2m_steps, cur_device
            )

            for i in range(num_t2m_steps + time_span):
                cur_step += 1
                cur_inputs = self.preprocess(
                    data, cur_step, his_model_input=False
                )
                cur_inputs["cur_step"] = cur_step

                cur_inputs["agent"]["x_a_mid_emb"] = []
                cur_inputs["agent"]["x_a_mid_emb"].append(
                    torch.cat(list(x_a_mid_emb), dim=2)
                )  # [B, A, 2, D]

                cur_inputs["agent"]["x_a_his"] = torch.cat(
                    list(x_a_his_enc), dim=2
                )  # [B, A, 5, D]

                cur_inputs_flatten = get_hbir_input(
                    cur_inputs
                )  # dist {'_input_0', }
                pred = self.ir_model(cur_inputs_flatten)
                # get agent enc and agent emb
                x_a_mid_emb.append(pred["x_a_cur_emb"])  # [B, A, 1, D]
                x_a_his_enc.append(pred["x_a_cur"])  # [B, A, 1, D]

                if i == num_t2m_steps + time_span - 1:
                    output = self.postprocess(pred, data)
                    return output
        else:
            print("Hbir Infer only supports cold start.")


@OBJECT_REGISTRY.register
class QCNetOEHbmInfer(nn.Module):
    """
    The basic structure of QCNetOEHbirInfer.

    This class serves as an interface for the HBM inference validation tool,
        pending integration with the main branch.

    Args:
        ir_model: The QCNet hbm model.
        preprocess: Preprocess module.
        postprocess: Postprocess module.
        quant_infer_cold_start: Indicates whether to use cold start
            for streaming inference.
        example_data: The dict of deploy inputs
    """

    def __init__(
        self,
        ir_model: nn.Module,
        preprocess: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        quant_infer_cold_start: bool = True,
        example_data: Optional[dict] = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.quant_infer_cold_start = quant_infer_cold_start
        self.example_data = example_data

    def forward(self, data: dict):
        if self.quant_infer_cold_start:
            num_historical_steps = self.example_data["agent"][
                "valid_mask"
            ].shape[
                2
            ]  # history steps
            num_t2m_steps = (
                self.example_data["agent"]["x_a_his"].shape[2] + 1
            )  # steps for decoder
            B, A, time_span, D = self.example_data["agent"]["x_a_mid_emb"][
                0
            ].shape[:]
            cur_step = (
                num_historical_steps - num_t2m_steps - time_span
            )  # current steps of stream infer (init 2)

            # Store the agent encoder mid outputs (Embedding) \
            # for the current time step
            x_a_mid_emb_list = deque(maxlen=time_span)
            cur_device = data["agent"]["valid_mask"].device
            for _ in range(time_span):
                x_a_mid_emb_list.append(
                    torch.zeros((B, A, 1, D), device=cur_device)
                )
            x_a_his_enc = (
                []
            )  # Store the agent encoder outputs for the current time step

            x_a_mid_emb, x_a_his_enc = _init_streaming_state(
                B, A, D, time_span, num_t2m_steps, cur_device
            )

            for i in range(num_t2m_steps + time_span):
                cur_step += 1
                cur_inputs = self.preprocess(
                    data, cur_step, his_model_input=False
                )
                cur_inputs["cur_step"] = cur_step

                cur_inputs["agent"]["x_a_mid_emb"] = []
                cur_inputs["agent"]["x_a_mid_emb"].append(
                    torch.cat(list(x_a_mid_emb), dim=2)
                )  # [B, A, 2, D]

                cur_inputs["agent"]["x_a_his"] = torch.cat(
                    list(x_a_his_enc), dim=2
                )  # [B, A, 5, D]

                cur_inputs_flatten = get_hbir_input(
                    cur_inputs
                )  # dist {'_input_0', }
                pred = self.ir_model(cur_inputs_flatten)
                # pred: ['loc_refine_pos', 'pi', 'x_a_cur', 'x_a_cur_emb']
                pred = {
                    "loc_refine_pos": pred["_output_0"].to(cur_device),
                    "pi": pred["_output_1"].to(cur_device),
                    "x_a_cur": pred["_output_2"].to(cur_device),
                    "x_a_cur_emb": pred["_output_3"].to(cur_device),
                }
                # pred = self.output_unflattener(pred)
                pred = output_to_tensor(pred, cur_device)

                # get agent enc and agent emb
                x_a_mid_emb.append(pred["x_a_cur_emb"])  # [B, A, 1, D]
                x_a_his_enc.append(pred["x_a_cur"])  # [B, A, 1, D]

                if i == num_t2m_steps + time_span - 1:
                    output = self.postprocess(pred, data)

                    # tmp_device = "cuda:0"
                    # for key in pred.keys():
                    #     if isinstance(pred[key], torch.Tensor):
                    #         if pred[key].device != tmp_device:
                    #             pred[key] = pred[key].to(tmp_device)

                    return output
        else:
            print("Hbm Infer only supports cold start.")


def get_hbir_input(example_inputs):
    flat_inputs, _ = tree_flatten(example_inputs)
    name_info = {}
    for i in range(len(flat_inputs)):
        value = flat_inputs[i]
        if isinstance(value, torch.Tensor) and value.dtype == torch.int64:
            value = value.to(torch.int32)
        name_info[f"_input_{i}"] = value

    return name_info


def output_to_tensor(hbir_output, cur_device=None):
    data_device = cur_device if cur_device is not None else "cpu"
    if isinstance(hbir_output, list):
        for k in range(len(hbir_output)):
            hbir_output[k] = output_to_tensor(hbir_output[k], data_device)
    elif isinstance(hbir_output, tuple):
        hbir_output = list(output_to_tensor(list(hbir_output)), data_device)
    elif isinstance(hbir_output, dict):
        for k in hbir_output:
            hbir_output[k] = output_to_tensor(hbir_output[k], data_device)
    else:
        hbir_output = hbir_output.to(data_device)

    return hbir_output
