# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Tuple

import cv2
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from hat_engine.utils.qconfig_manager import get_default_qat_qconfig
from torch import nn

__all__ = ["FCOS3D", "FCOS3DIrInfer"]


@OBJECT_REGISTRY.register
class FCOS3D(nn.Module):
    """The basic structure of fcos3d.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target module.
        post_process: post_process module.
        loss: loss module.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        post_process: nn.Module = None,
        loss: nn.Module = None,
    ):
        super(FCOS3D, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.targets = targets
        self.post_process = post_process
        self.loss = loss

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @fx_wrap()
    def _post_process(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_pred,
        attr_pred,
        centerness,
        data,
    ):
        if self.training and self.targets is not None:
            (
                labels_3d,
                bbox_targets_3d,
                centerness_targets,
                attr_targets,
            ) = self.targets(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                gt_bboxes_list=data["gt_bboxes"],
                gt_labels_list=data["gt_labels"],
                gt_bboxes_3d_list=data["gt_bboxes_3d"],
                gt_labels_3d_list=data["gt_labels_3d"],
                centers2d_list=data["centers2d"],
                depths_list=data["depths"],
                attr_labels_list=data["attr_labels"],
            )
            loss = self.loss(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                dir_cls_preds=dir_cls_pred,
                attr_preds=attr_pred,
                centernesses=centerness,
                labels_3d=labels_3d,
                bbox_targets_3d=bbox_targets_3d,
                centerness_targets=centerness_targets,
                attr_targets=attr_targets,
            )
            return loss
        else:
            bbox_outputs = self.post_process(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                dir_cls_preds=dir_cls_pred,
                attr_preds=attr_pred,
                centernesses=centerness,
                img_metas=data,
                cfg=None,
                rescale=True,
            )
            return bbox_outputs

    def forward(self, data):
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        if (self.training and self.targets is not None) or (
            self.post_process is not None
        ):
            (
                cls_scores,
                bbox_preds,
                dir_cls_pred,
                attr_pred,
                centerness,
            ) = self.head(feats)
            return self._post_process(
                cls_scores,
                bbox_preds,
                dir_cls_pred,
                attr_pred,
                centerness,
                data,
            )
        else:
            return self.head(feats)

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            module.fuse_model()

    def set_qconfig(self):
        self.qconfig = get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class FCOS3DIrInfer(nn.Module):
    """The basic structure of FCOS3DIrInfer.

    Args:
        ir_model: The ir model.
        post_process: Postprocess module.
        strides: A list of strides.
        nv12_input: Whether transform the input image to nv12.
        hbm_infer: Whether the model is hbm.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        post_process: nn.Module,
        strides: Tuple[int],
        nv12_input: bool = False,
        hbm_infer: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process
        self.strides = strides
        self.nv12_input = nv12_input
        self.hbm_infer = hbm_infer

    def _process_outputs(self, outputs):
        cls_scores = outputs[0]
        bbox_preds = []
        for i in range(len(outputs[0])):
            bbox_pred = []
            for j in list(range(1, 6)):
                bbox_pred.append(outputs[j][i])

            bbox_pred = self._decode(bbox_pred, self.strides[i])
            bbox_preds.append(bbox_pred)

        dir_cls_pred = outputs[6]
        attr_pred = outputs[7]
        centerness = outputs[8]
        return cls_scores, bbox_preds, dir_cls_pred, attr_pred, centerness

    def forward(self, data):
        if self.nv12_input and self.hbm_infer:
            ir_input = {"img": data["img"]}
            ir_input = _nv12_runtime(ir_input)
            outputs = self.ir_model(ir_input)
            results = []
            for i in range(9):
                cur_results = []
                for j in range(5):
                    idx = 5 * i + j
                    tmp_output = outputs[f"_output_{idx}"]
                    cur_results.append(tmp_output)
                results.append(cur_results)
            outputs = results
        else:
            outputs = self.ir_model(data)

        (
            cls_scores,
            bbox_preds,
            dir_cls_pred,
            attr_pred,
            centerness,
        ) = self._process_outputs(outputs)

        bbox_outputs = self.post_process(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            dir_cls_preds=dir_cls_pred,
            attr_preds=attr_pred,
            centernesses=centerness,
            img_metas=data,
            cfg=None,
            rescale=True,
        )
        return bbox_outputs

    def _decode(self, bbox_pred, stride):
        bbox_pred = torch.cat(bbox_pred, dim=1)
        clone_bbox = bbox_pred.clone()
        bbox_pred[:, :2] = clone_bbox[:, :2].float()
        bbox_pred[:, 2] = clone_bbox[:, 2].float()
        bbox_pred[:, 3:6] = clone_bbox[:, 3:6].float()

        bbox_pred[:, 2] = bbox_pred[:, 2].exp()
        bbox_pred[:, 3:6] = bbox_pred[:, 3:6].exp()
        bbox_pred[:, :2] *= stride
        return bbox_pred


def _nv12_runtime(data):
    def img2nv12(input_image, device):
        image = input_image.astype(np.uint8)
        image = image.squeeze(0)
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        height, width = image.shape[0], image.shape[1]
        yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
            (height * width * 3 // 2,)
        )
        y = yuv420p[: height * width]
        uv_planar = yuv420p[height * width :].reshape((2, height * width // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((height * width // 2,))
        return torch.tensor(
            y.reshape(1, height, width, 1), dtype=torch.uint8, device=device
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
