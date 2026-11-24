# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict, Tuple

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from torch import nn

from .bbox_coder import limit_period


@OBJECT_REGISTRY.register
class FCOS3DLoss(nn.Module):
    """Loss for FCOS3D.

    Args:
        num_classes: Number of categories excluding the background category.
        pred_attrs: Whether to predict attributes.
        num_attrs: The number of attributes to be predicted.
        group_reg_dims: The dimension of each regression target group.
        pred_velo: Whether to predict velocity.
        use_direction_classifier: Whether to add a direction classifier.
        dir_offset: Parameter used in direction classification.
        dir_limit_offset: Parameter used in direction classification.
        diff_rad_by_sin: Whether to change the difference into sin difference.
        loss_cls: Config of classification loss.
        loss_bbox: Config of localization loss.
        loss_dir: Config of direction classifier loss.
        loss_attr: Config of attribute classifier loss.
        loss_centerness: Config of centerness loss.
        train_cfg: Training config of anchor head.
    """

    def __init__(
        self,
        num_classes: int,
        pred_attrs: False,
        num_attrs: int,
        group_reg_dims: Tuple[int],
        pred_velo: bool,
        use_direction_classifier: bool,
        dir_offset: float,
        dir_limit_offset: float,
        diff_rad_by_sin: bool,
        loss_cls: Dict,
        loss_bbox: Dict,
        loss_dir: Dict,
        loss_attr: Dict,
        loss_centerness: Dict,
        train_cfg: Dict,
    ):
        super(FCOS3DLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.group_reg_dims = group_reg_dims
        self.pred_velo = pred_velo
        self.use_direction_classifier = use_direction_classifier
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.diff_rad_by_sin = diff_rad_by_sin
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_dir = loss_dir
        self.loss_attr = loss_attr
        self.loss_centerness = loss_centerness
        self.train_cfg = train_cfg

    @staticmethod
    def get_direction_target(
        reg_targets,
        dir_offset=0,
        dir_limit_offset=0.0,
        num_bins=2,
        one_hot=True,
    ):
        rot_gt = reg_targets[..., 6]
        offset_rot = limit_period(
            rot_gt - dir_offset, dir_limit_offset, 2 * np.pi
        )
        dir_cls_targets = torch.floor(
            offset_rot / (2 * np.pi / num_bins)
        ).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=reg_targets.dtype,
                device=dir_cls_targets.device,
            )
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7]
        )
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(
            boxes2[..., 6:7]
        )
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], dim=-1
        )
        return boxes1, boxes2

    @staticmethod
    def _flatten_pred(pred_list, channels: int):
        """NCHW level maps -> concatenated (N*H*W, C) tensor."""
        flat = [
            lvl.permute(0, 2, 3, 1).reshape(-1, channels) for lvl in pred_list
        ]
        return torch.cat(flat, dim=0)

    @staticmethod
    def _flatten_map(map_list):
        """NCHW single-channel maps -> concatenated (N*H*W,) tensor."""
        flat = [lvl.permute(0, 2, 3, 1).reshape(-1) for lvl in map_list]
        return torch.cat(flat, dim=0)

    def forward(
        self,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        attr_preds,
        centernesses,
        labels_3d,
        bbox_targets_3d,
        centerness_targets,
        attr_targets,
    ):
        batch_size = cls_scores[0].size(0)
        reg_dim = sum(self.group_reg_dims)

        flat_cls = self._flatten_pred(cls_scores, self.cls_out_channels)
        flat_bbox = self._flatten_pred(bbox_preds, reg_dim)
        flat_dir = self._flatten_pred(dir_cls_preds, 2)
        flat_ctr = self._flatten_map(centernesses)

        flat_labels = torch.cat(labels_3d, dim=0)
        flat_bbox_tgt = torch.cat(bbox_targets_3d, dim=0)
        flat_ctr_tgt = torch.cat(centerness_targets, dim=0)

        # Foreground: label in [0, num_classes).
        is_pos = (flat_labels >= 0) & (flat_labels < self.num_classes)
        pos_inds = torch.nonzero(is_pos, as_tuple=False).reshape(-1)
        n_pos = int(pos_inds.numel())

        loss_cls = self.loss_cls(
            flat_cls,
            flat_labels.to(torch.int64),
            avg_factor=n_pos + batch_size,
        )

        pos_bbox = flat_bbox[pos_inds]
        pos_dir = flat_dir[pos_inds]
        pos_ctr = flat_ctr[pos_inds]

        pos_attr = None
        if self.pred_attrs:
            flat_attr = self._flatten_pred(attr_preds, self.num_attrs)
            flat_attr_tgt = torch.cat(attr_targets, dim=0)
            pos_attr = flat_attr[pos_inds]

        loss_velo = loss_dir = loss_attr = None

        if n_pos > 0:
            pos_bbox_tgt = flat_bbox_tgt[pos_inds]
            pos_ctr_tgt = flat_ctr_tgt[pos_inds]
            bbox_w = pos_ctr_tgt.new_ones(n_pos, reg_dim)
            ones = pos_ctr_tgt.new_ones(pos_ctr_tgt.shape)
            avg = ones.sum()

            code_weight = self.train_cfg.get("code_weight", None)
            if code_weight:
                assert len(code_weight) == reg_dim
                bbox_w = bbox_w * bbox_w.new_tensor(code_weight)

            if self.use_direction_classifier:
                pos_dir_tgt = self.get_direction_target(
                    pos_bbox_tgt,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False,
                )

            if self.diff_rad_by_sin:
                pos_bbox, pos_bbox_tgt = self.add_sin_difference(
                    pos_bbox, pos_bbox_tgt
                )

            def _bbox_term(pred_sl, tgt_sl, w_sl):
                return self.loss_bbox(
                    pred_sl, tgt_sl, weight=w_sl, avg_factor=avg
                )

            loss_offset = _bbox_term(
                pos_bbox[:, :2], pos_bbox_tgt[:, :2], bbox_w[:, :2]
            )
            loss_depth = _bbox_term(
                pos_bbox[:, 2], pos_bbox_tgt[:, 2], bbox_w[:, 2]
            )
            loss_size = _bbox_term(
                pos_bbox[:, 3:6], pos_bbox_tgt[:, 3:6], bbox_w[:, 3:6]
            )
            loss_rotsin = _bbox_term(
                pos_bbox[:, 6], pos_bbox_tgt[:, 6], bbox_w[:, 6]
            )
            if self.pred_velo:
                loss_velo = _bbox_term(
                    pos_bbox[:, 7:9], pos_bbox_tgt[:, 7:9], bbox_w[:, 7:9]
                )

            loss_centerness = self.loss_centerness(pos_ctr, pos_ctr_tgt)

            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir, pos_dir_tgt, ones, avg_factor=avg
                )

            if self.pred_attrs:
                pos_attr_tgt = flat_attr_tgt[pos_inds]
                loss_attr = self.loss_attr(
                    pos_attr,
                    pos_attr_tgt,
                    pos_ctr_tgt,
                    avg_factor=pos_ctr_tgt.sum(),
                )
        else:
            # Keep graph connected when a batch has no positives.
            loss_offset = pos_bbox[:, :2].sum()
            loss_depth = pos_bbox[:, 2].sum()
            loss_size = pos_bbox[:, 3:6].sum()
            loss_rotsin = pos_bbox[:, 6].sum()
            if self.pred_velo:
                loss_velo = pos_bbox[:, 7:9].sum()
            loss_centerness = pos_ctr.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir.sum()
            if self.pred_attrs:
                loss_attr = pos_attr.sum()

        loss_dict = {
            "loss_cls": loss_cls,
            "loss_offset": loss_offset,
            "loss_depth": loss_depth,
            "loss_size": loss_size,
            "loss_rotsin": loss_rotsin,
            "loss_centerness": loss_centerness,
        }
        if loss_velo is not None:
            loss_dict["loss_velo"] = loss_velo
        if loss_dir is not None:
            loss_dict["loss_dir"] = loss_dir
        if loss_attr is not None:
            loss_dict["loss_attr"] = loss_attr
        return loss_dict
