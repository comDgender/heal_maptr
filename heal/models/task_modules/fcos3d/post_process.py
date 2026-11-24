# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict, Tuple

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.config import Config
from hat_engine.utils.model_helpers import fx_wrap
from torch import nn

from heal.core.box3d_utils import bbox3d2result, points_img2cam, xywhr2xyxyr
from heal.core.cam_box3d import CameraInstance3DBoxes
from heal.core.nms.box3d_nms import box2d3d_multiclass_nms
from heal.models.task_modules.fcos.target import get_points


@OBJECT_REGISTRY.register
class FCOS3DPostProcess(nn.Module):
    """Post-process for FOCS3D.

    Args:
        num_classes: Number of categories excluding the background category.
        use_direction_classifier: Whether to add a direction classifier.
        strides: Downsample factor of each feature map.
        group_reg_dims: The dimension of each regression target group.
        pred_attrs: Whether to predict attributes.
        num_attrs: The number of attributes to be predicted.
        attr_background_label: background label.
        bbox_coder: bbox coder class.
        bbox_code_size: Dimensions of predicted bounding boxes.
        dir_offset: Parameter used in direction classification.
        test_cfg: Testing config of anchor head.
        pred_bbox2d: Whether to predict 2D boxes.
    """

    def __init__(
        self,
        num_classes: int,
        use_direction_classifier: bool,
        strides: Tuple[int],
        group_reg_dims: Tuple[int],
        pred_attrs: bool,
        num_attrs: int,
        attr_background_label: int,
        bbox_coder: Dict,
        bbox_code_size: int,
        dir_offset: float,
        test_cfg: Dict,
        pred_bbox2d: bool = False,
    ):
        super(FCOS3DPostProcess, self).__init__()
        self.cls_out_channels = num_classes
        self.use_direction_classifier = use_direction_classifier
        self.strides = strides
        self.group_reg_dims = group_reg_dims
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.attr_background_label = attr_background_label
        self.bbox_coder = bbox_coder
        self.bbox_code_size = bbox_code_size
        self.dir_offset = dir_offset
        self.test_cfg = test_cfg
        self.pred_bbox2d = pred_bbox2d

    @fx_wrap()
    def forward(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        attr_preds,
        centernesses,
        img_metas,
        cfg=None,
        rescale=None,
    ):
        n_levels = len(cls_scores)
        assert (
            n_levels
            == len(bbox_preds)
            == len(dir_cls_preds)
            == len(centernesses)
            == len(attr_preds)
        )

        feat_hw = [lvl.shape[-2:] for lvl in cls_scores]
        mlvl_points = get_points(
            feat_hw,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )

        batch_size = cls_scores[0].size(0)
        results = []
        for bi in range(batch_size):
            cls_lvls = [lvl[bi].detach() for lvl in cls_scores]
            box_lvls = [lvl[bi].detach() for lvl in bbox_preds]
            ctr_lvls = [lvl[bi].detach() for lvl in centernesses]

            if self.use_direction_classifier:
                dir_lvls = [lvl[bi].detach() for lvl in dir_cls_preds]
            else:
                dir_lvls = []
                for cls_lvl in cls_lvls:
                    zeros = cls_lvl.new_zeros((2,) + cls_lvl.shape[1:])
                    dir_lvls.append(zeros.detach())

            if self.pred_attrs:
                attr_lvls = [lvl[bi].detach() for lvl in attr_preds]
            else:
                attr_lvls = []
                for cls_lvl in cls_lvls:
                    fill = cls_lvl.new_full(
                        (self.num_attrs,) + cls_lvl.shape[1:],
                        self.attr_background_label,
                    )
                    attr_lvls.append(fill.detach())

            meta = {
                "box_type_3d": CameraInstance3DBoxes,
                "cam2img": img_metas["cam2img"][bi],
            }
            results.append(
                self._get_bboxes_single(
                    cls_lvls,
                    box_lvls,
                    dir_lvls,
                    attr_lvls,
                    ctr_lvls,
                    mlvl_points,
                    meta,
                    cfg,
                    rescale,
                )
            )
        return results

    def _get_bboxes_single(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        attr_preds,
        centernesses,
        mlvl_points,
        input_meta,
        cfg,
        rescale=False,
    ):
        view = np.array(input_meta["cam2img"])
        cfg = Config(cfg_dict=self.test_cfg if cfg is None else cfg)
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)

        centers2d_lvls = []
        boxes_lvls = []
        scores_lvls = []
        dir_lvls = []
        attr_lvls = []
        ctr_lvls = []

        reg_dim = sum(self.group_reg_dims)
        nms_pre = cfg.get("nms_pre", -1)

        for cls_map, box_map, dir_map, attr_map, ctr_map, pts in zip(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            attr_preds,
            centernesses,
            mlvl_points,
        ):
            assert cls_map.shape[-2:] == box_map.shape[-2:]

            # CHW -> (H*W, C)
            scores = (
                cls_map.permute(1, 2, 0)
                .reshape(-1, self.cls_out_channels)
                .sigmoid()
            )
            dir_logits = dir_map.permute(1, 2, 0).reshape(-1, 2)
            dir_cls = dir_logits.argmax(dim=-1)
            attr_logits = attr_map.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_cls = attr_logits.argmax(dim=-1)
            ctr = ctr_map.permute(1, 2, 0).reshape(-1).sigmoid()

            boxes = box_map.permute(1, 2, 0).reshape(-1, reg_dim)
            boxes = boxes[:, : self.bbox_code_size]

            # Keep the highest score*centerness locations before NMS.
            if nms_pre > 0 and scores.size(0) > nms_pre:
                joint = (scores * ctr.unsqueeze(1)).amax(dim=1)
                keep = joint.topk(nms_pre).indices
                pts = pts[keep]
                boxes = boxes[keep]
                scores = scores[keep]
                dir_logits = dir_logits[keep]
                ctr = ctr[keep]
                dir_cls = dir_cls[keep]
                attr_cls = attr_cls[keep]

            # Decode 2D center offsets, then lift (u, v, depth) into camera 3D.
            boxes = boxes.clone()
            boxes[:, :2] = pts - boxes[:, :2]
            center2d = boxes[:, :3].clone()
            boxes[:, :3] = points_img2cam(boxes[:, :3], view)

            centers2d_lvls.append(center2d)
            boxes_lvls.append(boxes)
            scores_lvls.append(scores)
            dir_lvls.append(dir_cls)
            attr_lvls.append(attr_cls)
            ctr_lvls.append(ctr)

        all_centers2d = torch.cat(centers2d_lvls, dim=0)
        all_boxes = torch.cat(boxes_lvls, dim=0)
        all_dir = torch.cat(dir_lvls, dim=0)

        cam2img = all_centers2d.new_zeros(4, 4)
        cam2img[: view.shape[0], : view.shape[1]] = all_centers2d.new_tensor(
            view
        )
        all_boxes = self.bbox_coder.decode_yaw(
            all_boxes,
            all_centers2d,
            all_dir,
            self.dir_offset,
            cam2img,
        )

        boxes_for_nms = xywhr2xyxyr(
            input_meta["box_type_3d"](
                all_boxes,
                box_dim=self.bbox_code_size,
                origin=(0.5, 0.5, 0.5),
            ).bev
        )

        all_scores = torch.cat(scores_lvls, dim=0)
        # Trailing background column required by multiclass NMS.
        all_scores = torch.cat(
            [all_scores, all_scores.new_zeros(all_scores.size(0), 1)],
            dim=1,
        )
        all_attr = torch.cat(attr_lvls, dim=0)
        all_ctr = torch.cat(ctr_lvls, dim=0)
        nms_scores = all_scores * all_ctr.unsqueeze(1)
        results = box2d3d_multiclass_nms(
            mlvl_bboxes3d=all_boxes,
            mlvl_bboxes3d_for_nms=boxes_for_nms,
            mlvl_scores3d=nms_scores,
            score_thr=cfg.score_thr,
            max_num=cfg.max_per_img,
            nms_thr=cfg.nms_thr,
            mlvl_dir_scores=all_dir,
            mlvl_attr_scores=all_attr,
            do_nms_bev=True,
        )
        labels, _, _, scores, bboxes, _, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)
        bboxes = input_meta["box_type_3d"](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)
        )
        bbox_img = bbox3d2result(bboxes, scores, labels, attrs)
        box_center = bboxes.gravity_center
        box_dims = bboxes.dims
        box_yaw = bboxes.yaw
        box_dims[:, [0, 1, 2]] = box_dims[:, [2, 0, 1]]
        box_yaw = -box_yaw
        velocity = bboxes.tensor[..., 7:9]
        if self.pred_attrs:
            ret = torch.cat(
                [
                    box_center,
                    box_dims,
                    box_yaw.view(-1, 1),
                    velocity,
                    scores.view(-1, 1),
                    labels.view(-1, 1),
                    attrs.view(-1, 1),
                ],
                dim=-1,
            )
        else:
            ret = torch.cat(
                [
                    box_center,
                    box_dims,
                    box_yaw.view(-1, 1),
                    velocity,
                    scores.view(-1, 1),
                    labels.view(-1, 1),
                ],
                dim=-1,
            )
        return {
            "ret": ret,
            "bboxes": bbox_img["boxes_3d"],
            "scores": bbox_img["scores_3d"],
            "labels": bbox_img["labels_3d"],
            "attrs": bbox_img["attrs_3d"],
        }
