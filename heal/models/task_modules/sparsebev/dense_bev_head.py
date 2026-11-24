# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch.amp import autocast
from torch.quantization import DeQuantStub

from .head import SparseBEVHead

__all__ = ["DenseBEVSparseHead"]


@OBJECT_REGISTRY.register
class DenseBEVSparseHead(SparseBEVHead):
    """
    Module for handling object detection and 3D object processing.

    Args:
        instance_bank : Instance bank module for managing instance features.
        anchor_encoder : Anchor encoder module for encoding anchor features.
        ffn : Feedforward network module for processing encoded features.
        deformable_model : Deformable model module for
                            modeling deformable aspects.
        refine_layer : Refinement layer module for refining output predictions.
        bev_size : (w, h, resolusion)
        num_decoder : Number of decoders. Defaults to 6.
        num_single_frame_decoder : Number of single frame decoders.
                                   Defaults to -1.
        embed_dims : Dimension of embeddings. Defaults to 256.
        num_heads : Number of attention heads. Defaults to 8.
        num_views : Number of views. Defaults to 6.
        meta_process_func: The function to process meta data.
        level_index : List of level indices. Defaults to None.
        loss_cls : Classification loss module. Defaults to None.
        loss_reg : Regression loss module. Defaults to None.
        loss_cns : CNS loss module. Defaults to None.
        loss_yns : YNS loss module. Defaults to None.
        decoder : Decoder function. Defaults to None.
        target : Target function. Defaults to None.
        gt_key : Key for ground truth data.
                 Defaults to "lidar_bboxes_labels".
        reg_weights : List of regression weights. Defaults to None.
        operation_order: List of operation orders. Defaults to None.
        cls_threshold_to_reg: Classification threshold for regression.
                              Defaults to -1.0.
        enable_dn : Flag to enable dn. Defaults to False.
        enable_temp_dn : Flag to enable temporal dn. Defaults to False.
        cls_allow_reverse : Allow reverse classification flag.
                            Defaults to None.
        projection_mat_key : Key for projection matrix.
                             Defaults to None.
        bbox_dim: dim of gt_bbox. Defaults to 10.
        with_track: Whether to use tracking mode. Defaults to False.
        is_compile: Whether for compile.
    """

    def __init__(
        self,
        instance_bank: nn.Module,
        anchor_encoder: nn.Module,
        ffn: nn.Module,
        deformable_model: nn.Module,
        refine_layer: nn.Module,
        bev_size: Optional[tuple] = None,
        point_cloud_range: Optional[tuple] = None,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_views: int = 6,
        meta_process_func: Callable = None,
        level_index: List = None,
        loss_cls: Optional[nn.Module] = None,
        loss_reg: Optional[nn.Module] = None,
        loss_cns: Optional[nn.Module] = None,
        loss_yns: Optional[nn.Module] = None,
        decoder: Optional[Callable] = None,
        target: Optional[Callable] = None,
        gt_key: str = "lidar_bboxes_labels",
        reg_weights: Optional[List[float]] = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1.0,
        enable_dn: bool = False,
        enable_temp_dn: bool = False,
        cls_allow_reverse: int = None,
        projection_mat_key: Optional[str] = None,
        bbox_dim: int = 10,
        with_track: bool = False,
        decode_with_track_id: bool = False,
        use_adapter: bool = False,
        in_dim: int = 256,
        is_compile: bool = False,
        is_train: bool = False,
        filter_empty_gt: bool = False,
        return_bev_embed: bool = False,
    ):
        super(DenseBEVSparseHead, self).__init__(
            instance_bank=instance_bank,
            anchor_encoder=anchor_encoder,
            ffn=ffn,
            deformable_model=deformable_model,
            refine_layer=refine_layer,
            num_decoder=num_decoder,
            num_single_frame_decoder=num_single_frame_decoder,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_views=num_views,
            level_index=level_index,
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_cns=loss_cns,
            loss_yns=loss_yns,
            decoder=decoder,
            target=target,
            gt_key=gt_key,
            reg_weights=reg_weights,
            operation_order=operation_order,
            cls_threshold_to_reg=cls_threshold_to_reg,
            enable_dn=enable_dn,
            enable_temp_dn=enable_temp_dn,
            cls_allow_reverse=cls_allow_reverse,
            projection_mat_key=projection_mat_key,
        )
        self.bev_size = bev_size
        self.point_cloud_range = point_cloud_range
        self.bbox_dim = bbox_dim
        self.meta_process_func = meta_process_func
        self.with_track = with_track
        self.decode_with_track_id = decode_with_track_id
        self.use_adapter = use_adapter
        self.is_compile = is_compile
        self.is_train = is_train
        self.filter_empty_gt = filter_empty_gt
        self.return_bev_embed = return_bev_embed
        # self.fc_before_list = []
        # self.fc_after_list = []
        # for _ in range(len(2 * self.num_decoder)):
        #     self.fc_before_list.append(copy.deepcopy(self.fc_before))
        #     self.fc_after_list.append(copy.deepcopy(self.fc_after))
        if self.use_adapter:
            self.skip_add = nn.quantized.FloatFunctional()
            self.adapter = nn.Sequential(
                ConvModule2d(
                    in_dim,
                    in_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(in_dim),
                    act_layer=nn.ReLU(inplace=False),
                ),
                ConvModule2d(
                    in_dim,
                    in_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(in_dim),
                    act_layer=None,
                    # act_layer=nn.ReLU(inplace=False),
                ),
            )
        if self.return_bev_embed:
            self.bev_dequant = DeQuantStub()

    @staticmethod
    def gen_projection_mat_custom(
        projection_mat: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        """Modify the given homography matrix.

        Args:
            projection_mat: projection_mat.
            scales: normalized parameter.

        Returns:
            homopgrahpy: Modified homography matrix.
        """
        views = []
        for scale in scales:
            view = torch.eye(4)
            view[0, 0] = 1 / scale[0]
            view[1, 1] = 1 / scale[1]
            views.append(torch.tensor(view).to(device=projection_mat.device))
        views = torch.stack(views)
        projection_mat = torch.matmul(views.double(), projection_mat.double())
        return projection_mat

    def gen_projection_matrix(self, projection_mat, point_cloud_range, bs):
        # (x-min_x)/(max_x-min_x)*2-1
        # x/((max_x - min_x) / 2) - min_x / ((max_x - min_x) / 2) - 1
        min_x, max_x = point_cloud_range[0], point_cloud_range[3]
        min_y, max_y = point_cloud_range[1], point_cloud_range[4]
        views = []
        for _ in range(bs):
            view = torch.eye(4)
            view[0, 0] = 1 / ((max_x - min_x) / 2)
            view[0, 3] = -min_x / ((max_x - min_x) / 2) - 1
            view[1, 1] = 1 / ((max_y - min_y) / 2)
            view[1, 3] = -min_y / ((max_y - min_y) / 2) - 1
            views.append(torch.tensor(view).to(device=projection_mat.device))
        views = torch.stack(views)
        projection_mat = torch.matmul(views.double(), projection_mat.double())
        return projection_mat

    def parse_labels(self, data):
        device = data["img"].device
        gt_cls = []
        gt_reg = []
        for _, labels in enumerate(data[self.gt_key]):
            if len(labels) == 0:
                cls = []
                reg = torch.zeros(
                    (1, self.bbox_dim - 1), device=device
                ).float()
            else:
                cls = labels[:, -1]
                reg = torch.tensor(labels[:, :-1], device=device).float()
            gt_cls.append(torch.tensor(cls, device=device).long())
            gt_reg.append(reg)
        return gt_cls, gt_reg

    @autocast("cuda", enabled=False)
    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: Dict[str, Any],
        lidar_feature=None,
    ) -> torch.Tensor:
        """
        Forward pass of the SparseBEVOEHead module.

        Args:
            feature_maps: Input feature maps.If a single torch.Tensor,
                          it's converted to a List[torch.Tensor].
            metas : Metadata dictionary containing additional information.
        """

        if self.meta_process_func is not None:
            metas = self.meta_process_func(metas, self.is_compile)

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]

        if len(feature_maps) == 1:
            if self.use_adapter:
                feature_maps = [
                    self.skip_add.add(
                        feature_maps[0], self.adapter(feature_maps[0])
                    )
                ]
        if self.return_bev_embed:
            assert len(feature_maps) == 1
            dynamic_bev_embed = self.bev_dequant(feature_maps[0])

        if self.level_index is not None:
            feature_maps_head = []
            for index in self.level_index:
                if self.is_compile is False:
                    feature_maps_head.append(feature_maps[index].float())
                else:
                    feature_maps_head.append(feature_maps[index])
            feature_maps = feature_maps_head
        if feature_maps is not None:
            batch_size = feature_maps[0].shape[0] // self.num_views
        else:
            batch_size = lidar_feature.shape[0]

        dn_metas = None
        if (
            self.target is not None
            and self.target.dn_metas is not None
            and self.target.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.target.dn_metas = None

        if self.is_compile is False:
            dn_metas = self.target.dn_metas

        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas, self.is_compile
        )

        if not self.with_track:
            temp_instance_feature = None
            temp_anchor = None
        dn_metas = None
        attn_mask = None
        temp_dn_reg_target = None
        # prepare for denosing training
        if self.enable_dn:
            if (self.training or self.is_train) and hasattr(
                self.target, "get_dn_anchors"
            ):
                if "instance_ids" in metas:
                    gt_instance_id = []
                    for instance_id in metas["instance_ids"]:
                        instance_id = np.array(instance_id, dtype=np.int64)
                        gt_instance_id.append(
                            torch.from_numpy(instance_id).cuda()
                        )
                else:
                    gt_instance_id = None
                dn_metas = self.target.get_dn_anchors(
                    *self.parse_labels(metas), gt_instance_id
                )
            if dn_metas is not None:
                (
                    dn_anchor,
                    dn_reg_target,
                    dn_cls_target,
                    dn_attn_mask,
                    valid_mask,
                    dn_id_target,
                ) = dn_metas
                num_dn_anchor = dn_anchor.shape[1]
                if dn_anchor.shape[-1] != anchor.shape[-1]:
                    remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]

                    dn_anchor = torch.cat(
                        [
                            dn_anchor,
                            dn_anchor.new_zeros(
                                batch_size, num_dn_anchor, remain_state_dims
                            ),
                        ],
                        dim=-1,
                    )
                anchor = torch.cat([anchor, dn_anchor], dim=1)
                instance_feature = torch.cat(
                    [
                        instance_feature,
                        instance_feature.new_zeros(
                            batch_size,
                            num_dn_anchor,
                            instance_feature.shape[-1],
                        ),
                    ],
                    dim=1,
                )
                num_instance = instance_feature.shape[1]
                num_free_instance = num_instance - num_dn_anchor
                attn_mask = anchor.new_ones(
                    (num_instance, num_instance), dtype=torch.bool
                )
                attn_mask[:num_free_instance, :num_free_instance] = False
                attn_mask[num_free_instance:, num_free_instance:] = (
                    dn_attn_mask
                )

        # generate anchor embed
        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:

            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        bs, _, _ = anchor.shape
        if self.is_compile is True:
            projection_mat = metas["projection_mat"]
        else:
            if self.projection_mat_key == "lidar2ego":
                # lidar2ego
                projection_mat = metas[self.projection_mat_key]
            else:
                # ego
                projection_mat = torch.eye(4)
            if len(projection_mat.shape) == 2:
                projection_mat = projection_mat[None, :, :]
                projection_mat = projection_mat.tile(
                    self.num_views * bs, 1, 1
                ).to(device=anchor.device)
            else:
                projection_mat = projection_mat.to(device=anchor.device)
            if self.point_cloud_range is None:
                scales = [(self.bev_size[0], self.bev_size[1])] * (
                    self.num_views * bs
                )
                projection_mat = DenseBEVSparseHead.gen_projection_mat_custom(
                    projection_mat, scales
                )
            else:
                # (x-(-51.2))/(102.4-(-51.2))*2-1
                projection_mat = self.gen_projection_matrix(
                    projection_mat, self.point_cloud_range, bs
                )
        projection_mat = self.mat_quant_stub(
            projection_mat.view(bs, -1, 4, 4).float()
        )

        # forward layers
        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_interaction":
                value = temp_instance_feature
                if temp_instance_feature is not None:
                    value = self.fc_before(value)
                instance_feature = self.layers[i](
                    instance_feature,
                    temp_instance_feature,
                    value,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=(
                        attn_mask if temp_instance_feature is None else None
                    ),
                )
                instance_feature = self.fc_after(instance_feature)
            elif op == "interaction":
                value = instance_feature
                value = self.fc_before(value)
                instance_feature = self.layers[i](
                    instance_feature,
                    instance_feature,
                    value=value,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
                instance_feature = self.fc_after(instance_feature)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    projection_mat,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    return_cls=(
                        (self.training or self.is_train)
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                    last_layer=(i == len(self.operation_order) - 1),
                )
                prediction.append(self.dequant(anchor))
                if i == len(self.operation_order) - 1:
                    classification.append(self.dequant(cls))
                    quality.append(self.dequant(qt))
                else:
                    classification.append(cls)
                    quality.append(qt)
                if (
                    len(prediction) == self.num_single_frame_decoder
                    and self.num_single_frame_decoder > 0
                ):
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature,
                        anchor,
                        cls,
                        temp_instance_feature,
                        temp_anchor,
                    )
                    if (
                        dn_metas is not None
                        and self.target.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.target.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)

                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                    and self.instance_bank.update_temp()
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.get_num_temp()
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}
        cls = self.dequant(cls)
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]
            self.target.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        if self.is_compile is True:
            output.update(
                {
                    "classification": classification[-1],
                    "prediction": prediction[-1],
                    "quality": quality[-1],
                    "feature": instance_feature,
                    # "cached_anchor": self.dequant(cached_anchor),
                    # "cached_feature": self.dequant(cached_feature),
                    # "cached_confidence": self.dequant(cached_confidence),
                }
            )
        else:
            output.update(
                {
                    "classification": classification,
                    "prediction": prediction,
                    "quality": quality,
                }
            )
            if os.getenv("HAT_TRAINING_STEP", "float") == "quant_analysis":
                output["classification"] = output["classification"][-1]
                output["prediction"] = output["prediction"][-1]
                output["quality"] = output["quality"][-1]
                output["feature"] = instance_feature
                return output
            if self.with_track:
                self.instance_bank.cache(
                    instance_feature,
                    anchor,
                    cls,
                    metas,
                    update_confidence=True,
                )
        if Tracer.is_tracing() is True:
            self.instance_bank.reset()

        if not self.is_compile:
            if (
                self.training or self.is_train
            ) and Tracer.is_tracing() is False:
                output = self.loss(output, metas)
            else:
                if self.decode_with_track_id:
                    instance_id = self.instance_bank.get_instance_id(
                        cls, anchor, self.decoder.score_threshold
                    )
                    output["instance_id"] = instance_id
                output = self.post_process(output)

        # Attach dynamic_bev_embed after loss/post_process so it is preserved
        # in the output dict (loss/post_process drop unrecognized keys)
        if self.return_bev_embed:
            for i in range(len(output)):
                output[i]["dynamic_bev_embed"] = dynamic_bev_embed[i].cpu()
        return output

    @autocast("cuda", enabled=False)
    def loss(
        self,
        model_outs: Dict[str, torch.Tensor],
        data: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for the SparseBEVOEHead module.

        Args:
            model_outs: Dictionary containing model outputs,
                including 'classification', 'prediction', and 'quality'.
            data : Dictionary containing ground truth data.

        Returns:
            Dictionary containing computed loss values.

        """

        if self.bbox_dim == 10:
            center_dim = 3
            yaw_ind = 6
        elif self.bbox_dim == 8:
            center_dim = 2
            yaw_ind = 4
        else:
            raise NotImplementedError(
                f"Not support `bbox_dim={self.bbox_dim}`"
            )
        gt_cls, gt_reg = self.parse_labels(data)

        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}

        # filter empty gt images
        if self.filter_empty_gt:
            valid_img_mask = torch.tensor(
                [len(x) > 0 for x in gt_cls],
                device=cls_scores[0].device,
                dtype=torch.bool,
            )

        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            if self.filter_empty_gt:
                cls = cls[valid_img_mask]
                reg = reg[valid_img_mask]
                qt = qt[valid_img_mask] if qt is not None else None
                cur_gt_cls = [
                    gt_cls[i] for i in range(len(gt_cls)) if valid_img_mask[i]
                ]
                cur_gt_reg = [
                    gt_reg[i] for i in range(len(gt_reg)) if valid_img_mask[i]
                ]
            else:
                cur_gt_cls = gt_cls
                cur_gt_reg = gt_reg

            # ===== 如果当前 rank 没有任何 GT =====
            if len(cur_gt_cls) == 0:
                zero = cls.sum() * 0.0
                output[f"loss_cls_{decoder_idx}"] = zero
                output[f"loss_reg_{decoder_idx}"] = zero
                if qt is not None:
                    output[f"loss_cns_{decoder_idx}"] = zero
                    output[f"loss_yns_{decoder_idx}"] = zero
                # DDP-safe: 给通信留位置，空 GT rank num_pos=0, has_pos=0
                if torch.distributed.is_initialized() and cls.is_cuda:
                    num_pos_tensor = cls.new_tensor(0.0)
                    has_pos_tensor = cls.new_tensor(0.0)
                    torch.distributed.all_reduce(
                        num_pos_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        has_pos_tensor, op=torch.distributed.ReduceOp.SUM
                    )
                continue

            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, cls_weights, reg_weights = self.target(
                cls,
                reg,
                cur_gt_cls,
                cur_gt_reg,
                None,
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask = torch.logical_and(mask, cls_weights != 0)

            mask_flat = mask.reshape(-1)
            num_pos_local = mask_flat.sum().to(dtype=reg.dtype)
            has_pos_local = torch.tensor(
                1.0 if num_pos_local.item() > 0 else 0.0,
                device=cls.device,
                dtype=reg.dtype,
            )

            if torch.distributed.is_initialized() and mask.is_cuda:
                num_pos_tensor = num_pos_local.clone()
                has_pos_tensor = has_pos_local.clone()
                torch.distributed.all_reduce(
                    num_pos_tensor, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    has_pos_tensor, op=torch.distributed.ReduceOp.SUM
                )
                num_pos = max(num_pos_tensor / has_pos_tensor, 1.0)
            else:
                num_pos = max(num_pos_local, 1.0)
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weights = cls_weights.flatten(end_dim=1)
            cls_loss = self.loss_cls(
                cls, cls_target, avg_factor=num_pos, weight=cls_weights
            )
            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            cls = cls[mask]

            if self.cls_allow_reverse is not None and cls_target.shape[0] != 0:
                if_reverse = (
                    torch.nn.functional.cosine_similarity(
                        reg_target[..., yaw_ind : yaw_ind + 2],
                        reg[..., yaw_ind : yaw_ind + 2],
                        dim=-1,
                    )
                    < 0
                )
                if_reverse = (
                    torch.isin(
                        cls_target,
                        cls_target.new_tensor(self.cls_allow_reverse),
                    )
                    & if_reverse
                )
                reg_target[..., yaw_ind : yaw_ind + 2] = torch.where(
                    if_reverse[..., None],
                    -reg_target[..., yaw_ind : yaw_ind + 2],
                    reg_target[..., yaw_ind : yaw_ind + 2],
                )
            reg_loss = self.loss_reg(
                reg, reg_target, weight=reg_weights, avg_factor=num_pos
            )

            cls_loss = (
                sum(cls_loss.values())
                if isinstance(cls_loss, dict)
                else cls_loss
            )
            reg_loss = (
                sum(reg_loss.values())
                if isinstance(reg_loss, dict)
                else reg_loss
            )
            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]
                cns = qt[..., 0]
                yns = qt[..., 1].sigmoid()
                cns_target = torch.norm(
                    reg_target[..., :center_dim] - reg[..., :center_dim],
                    p=2,
                    dim=-1,
                )
                cns_target = torch.exp(-cns_target)
                cns_loss = self.loss_cns(cns, cns_target, avg_factor=num_pos)
                output[f"loss_cns_{decoder_idx}"] = cns_loss
                yns_target = (
                    torch.nn.functional.cosine_similarity(
                        reg_target[..., yaw_ind : yaw_ind + 2],
                        reg[..., yaw_ind : yaw_ind + 2],
                        dim=-1,
                    )
                    > 0
                )
                yns_target = yns_target.float()
                yns_loss = self.loss_yns(yns, yns_target)
                output[f"loss_yns_{decoder_idx}"] = yns_loss

        if "dn_prediction" not in model_outs:
            return output

        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)

        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
            )
            cls_loss = (
                sum(cls_loss.values())
                if isinstance(cls_loss, dict)
                else cls_loss
            )
            reg_loss = (
                sum(reg_loss.values())
                if isinstance(reg_loss, dict)
                else reg_loss
            )
            output.update(
                {
                    f"dn_loss_cls_{decoder_idx}": cls_loss,
                    f"dn_loss_reg_{decoder_idx}": reg_loss,
                }
            )
        return output

    def post_process(
        self, model_outs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if self.decode_with_track_id:
            return self.decoder(
                cls_scores=model_outs["classification"],
                box_preds=model_outs["prediction"],
                instance_id=model_outs.get("instance_id"),
                qulity=model_outs.get("quality"),
                output_idx=-1,
            )

        else:
            return self.decoder(
                cls_scores=model_outs["classification"],
                box_preds=model_outs["prediction"],
                qulity=model_outs.get("quality"),
            )
