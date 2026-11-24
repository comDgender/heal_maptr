# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = ["Detr3d", "Detr3dIrInfer"]


@OBJECT_REGISTRY.register
class Detr3d(nn.Module):
    """The basic structure of detr3d.

    Args:
        backbone: backbone module.
        neck: neck module.
        head: head module with transformer architecture.
        target: detr3d target generator.
        post_process: post process module.
        loss_cls: loss module for classification.
        loss_reg: loss module for regression.
        compile_model: Whether in compile model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        target: nn.Module = None,
        post_process: nn.Module = None,
        loss_cls: nn.Module = None,
        loss_reg: nn.Module = None,
        compile_model: bool = False,
    ):
        super(Detr3d, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target = target
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.compile_model = compile_model
        self.post_process = post_process

    def extract_feat(self, img: Tensor) -> Tensor:
        """Directly extract features from the backbone + neck.

        Args:
            img: The input image to be encoded.

        Returns:
            The encoded features of the input image.
        """
        x = self.backbone(img)

        if self.neck is not None:
            x = self.neck(x)
        return x

    def _get_val(self, loss: Any) -> Tensor:
        """Retrieve the value of the loss.

        Args:
            loss: The loss value.

        Returns:
            loss: The loss value tensor.
        """
        if isinstance(loss, Dict):
            for _, v in loss.items():
                return v
        return loss

    def export_reference_points(
        self, data: Dict, feat_wh: Tuple[int, int]
    ) -> Dict:
        """Export the reference points.

        Args:
            data: The data used for exporting the reference points.
            feat_wh: The size of the feature map.
        Returns:
            The exported reference points.
        """
        return self.head.export_reference_points(data, feat_wh)

    def forward(self, data: Dict) -> Dict:
        """Perform the forward pass of the model.

        Args:
            data: A dictionary containing the input data.

        Returns:
            A dictionary containing the output of the forward pass.
        """
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        (cls_list, reg_list), ref_p = self.head(
            feats, data, self.compile_model
        )
        return self._post_process(data, cls_list, reg_list, ref_p)

    @fx_wrap()
    def _post_process(
        self,
        data: Dict,
        cls_list: List[Tensor],
        reg_list: List[Tensor],
        reference_points: Tensor,
    ) -> Dict:
        """Perform the post-processing of the model's output.

        Args:
            data: The input data.
            cls_list: A list of predicted class probabilities.
            reg_list: A list of predicted bounding boxes.
            reference_points: The reference points used for prediction.

        Returns:
            If in training mode, returns a dictionary containing
            the losses for each predicted class and regression;
            otherwise, returns the predicted class probabilities
            and bounding boxes, or the results of the post-processing
            if specified.
        """
        if self.training:
            labels = data["ego_bboxes_labels"]
            loss_dict = {}
            for i in range(len(cls_list)):
                cls_target, reg_target = self.target(
                    labels, cls_list[i], reg_list[i], reference_points
                )

                cls_loss = self._get_val(self.loss_cls(**cls_target))
                reg_loss = self._get_val(self.loss_reg(**reg_target))
                loss_dict.update({f"cls{i}": cls_loss})
                loss_dict.update({f"reg{i}": reg_loss})

            return loss_dict
        else:
            cls_pred = cls_list[-1]
            reg_pred = reg_list[-1]
            if self.post_process is None:
                return cls_pred, reg_pred
            results = self.post_process(cls_pred, reg_pred, reference_points)
            return results

    def fuse_model(self) -> None:
        """Fuse the model."""
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()


@OBJECT_REGISTRY.register
class Detr3dIrInfer(nn.Module):
    """
    The basic structure of Detr3dIrInfer.

    Args:
        deploy_model: The deploy model to generate refpoints.
        model_convert_pipeline: Define the process of model convert.
        vt_input_hw: Feature map shape.
        ir_model: The ir model.
        post_process: Postprocess module.
    """

    def __init__(
        self,
        deploy_model: nn.Module,
        model_convert_pipeline: List[callable],
        vt_input_hw: List[int],
        ir_model: nn.Module,
        post_process: nn.Module = None,
        use_nv12: bool = False,
        keep_last_img: bool = True,
    ):
        super(Detr3dIrInfer, self).__init__()

        self.deploy_model = model_convert_pipeline(deploy_model)
        self.deploy_model.eval()
        self.vt_input_hw = vt_input_hw
        self.post_process = post_process
        self.ir_model = ir_model
        self.use_nv12 = use_nv12
        self.keep_last_img = keep_last_img

    def forward(self, data):
        meta = self.deploy_model.export_reference_points(
            data, self.vt_input_hw
        )

        ref_p = meta["reference_points"]
        meta.pop("reference_points")
        data.update(meta)

        if self.use_nv12:
            img_6v = data.pop("img")
            if self.keep_last_img is True:
                img_5v = {
                    "img": img_6v[:5],
                }
                nv12_img = self.nv12_runtime(img_5v)
                img_last = {"img_5": img_6v[-1:, :, :, :]}
                nv12_img.update(img_last)
            else:
                img_6v = {
                    "img": img_6v,
                }
                nv12_img = self.nv12_runtime(img_6v)
            data.update(nv12_img)

        outputs = self.ir_model(data)
        if isinstance(outputs, dict):
            outputs = [outputs["_output_0"], outputs["_output_1"]]
        results = self.post_process(outputs[0], outputs[1], ref_p)
        return results

    def nv12_runtime(self, data):
        def img2nv12(input_image, device):
            image = input_image.astype(np.uint8)
            image = image.squeeze(0)
            image = np.transpose(image, (1, 2, 0))
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
                (height * width * 3 // 2,)
            )
            y = yuv420p[: height * width]
            uv_planar = yuv420p[height * width :].reshape(
                (2, height * width // 4)
            )
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2,)
            )
            return torch.tensor(
                y.reshape(1, height, width, 1),
                dtype=torch.uint8,
                device=device,
            ), torch.tensor(
                uv_packed.reshape(1, height // 2, width // 2, 2),
                dtype=torch.uint8,
                device=device,
            )

        dict_data = {}
        for key in data.keys():
            if data[key].shape[0] == 1:
                dict_data[f"{key}_y"], dict_data[f"{key}_uv"] = img2nv12(
                    data[key].cpu().numpy(), device=data[key].device
                )
            else:
                for i in range(data[key].shape[0]):
                    (
                        dict_data[f"{key}_{i}_y"],
                        dict_data[f"{key}_{i}_uv"],
                    ) = img2nv12(
                        data[key][i : i + 1, :, :, :].cpu().numpy(),
                        device=data[key].device,
                    )
        return dict_data

    def nv12_runtime_from_yuv444(self, data):
        def inverse_normalize(tensor, mean=128, std=128):
            return tensor * std + mean

        def yuv444_to_yuv420(yuv444):
            if isinstance(yuv444, np.ndarray):
                yuv444 = torch.from_numpy(yuv444)
            if yuv444.ndim == 3:
                yuv444 = yuv444.unsqueeze(0)
            N, C, H, W = yuv444.shape
            assert C == 3

            # ensure uint8
            if yuv444.dtype != torch.uint8:
                yuv444 = yuv444.round().clamp(0, 255).to(torch.uint8)

            # Y component
            Y = yuv444[:, 0, :, :].unsqueeze(-1)  # [N, H, W, 1]

            # UV component, average downsampling
            U = yuv444[:, 1, :, :]
            V = yuv444[:, 2, :, :]
            U_420 = (
                U.reshape(N, H // 2, 2, W // 2, 2)
                .float()
                .mean(dim=(2, 4))
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            V_420 = (
                V.reshape(N, H // 2, 2, W // 2, 2)
                .float()
                .mean(dim=(2, 4))
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            # NV12: interleaved
            UV = torch.stack([U_420, V_420], dim=-1)  # [N, H//2, W//2, 2]
            return Y, UV

        dict_data = {}
        for key in data.keys():
            data[key] = inverse_normalize(data[key])
            if data[key].shape[0] == 1:
                (
                    dict_data[f"{key}_y"],
                    dict_data[f"{key}_uv"],
                ) = yuv444_to_yuv420(data[key].cpu().numpy())
            else:
                for i in range(data[key].shape[0]):
                    (
                        dict_data[f"{key}_{i}_y"],
                        dict_data[f"{key}_{i}_uv"],
                    ) = yuv444_to_yuv420(
                        data[key][i : i + 1, :, :, :].cpu().numpy()
                    )
        return dict_data
