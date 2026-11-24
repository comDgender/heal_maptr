# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Sparse multitask structure: det (SparseBEV) + map (SparseMapTR) + occ (FlashOcc)."""  # noqa: E501

import copy
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.global_var import get_value
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch import nn

from heal.models.task_modules.flashocc.decoder import FlashOccHead
from heal.models.task_modules.sparse_maptr.decoder import (
    SparseMapPerceptionDecoder,
)
from heal.models.task_modules.sparsebev.head import SparseBEVHead
from heal.models.task_modules.sparsebev.instance_bank import InstanceBank
from .sparsebev import SparseBEV, SparseBEVIrInfer

__all__ = [
    "SparseMultitask",
    "SparseMultitaskIrInfer",
]


@OBJECT_REGISTRY.register
class SparseMultitask(SparseBEV):
    """Shared-backbone multitask model with optional per-head necks."""

    def __init__(
        self,
        backbone: nn.Module,
        head: Union[nn.Module, Dict[str, nn.Module]],
        neck: Optional[nn.Module] = None,
        depth_branch: Optional[nn.Module] = None,
        compiler_model: bool = False,
        loss_weights: Optional[dict] = None,
        split_neck: bool = False,
    ):
        super(SparseMultitask, self).__init__(
            backbone=backbone,
            head=head,
            neck=neck,
            depth_branch=depth_branch,
            compiler_model=compiler_model,
        )
        self.split_neck = split_neck
        if split_neck:
            self.necks = nn.ModuleDict(
                {
                    head_key: copy.deepcopy(neck)
                    for head_key in self.head.keys()
                }
            )
        if isinstance(self.head, dict):
            self.head = nn.ModuleDict(self.head)
            if loss_weights is None:
                loss_weights = {k: 1.0 for k in self.head.keys()}
            self.loss_weights = loss_weights

    def _post_process(
        self,
        feature_maps: List,
        model_outs: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.compiler_model is False:
            if (
                self.training is True
                and Tracer.is_tracing() is False
                and not get_value("preparing_qat")
            ):
                if self.depth_branch is not None and not self.split_neck:
                    depths = self.depth_branch(feature_maps, data)
                    model_outs["loss_dense_depth"] = self.depth_branch.loss(
                        depths, data
                    )
        return model_outs

    def forward_single_head(
        self,
        head_name,
        head,
        feature_maps,
        data,
        compiler_model,
    ):
        if self.split_neck:
            feature_maps = list(self.necks[head_name](feature_maps))

        if isinstance(head, SparseBEVHead):
            model_outs = head(
                feature_maps=feature_maps,
                metas=data,
                compiler_model=compiler_model,
            )
        elif isinstance(head, FlashOccHead):
            model_outs = head(
                feature_maps=feature_maps,
                metas=data,
                compiler_model=compiler_model,
            )
        elif isinstance(head, SparseMapPerceptionDecoder):
            model_outs = head(
                mlvl_feats=feature_maps,
                data=data,
            )
        else:
            raise ValueError(
                "Head should be SparseBEVHead, SparseMapPerceptionDecoder, "
                "or FlashOccHead."
            )

        if self.training:
            model_outs = {
                k: v * self.loss_weights[head_name]
                for k, v in model_outs.items()
            }

        if self.split_neck:
            if (
                self.training is True
                and Tracer.is_tracing() is False
                and not get_value("preparing_qat")
            ):
                depths = self.depth_branch(feature_maps, data)
                model_outs["loss_dense_depth"] = self.depth_branch.loss(
                    depths, data
                )
        return model_outs

    def extract_feat(self, img: torch.Tensor) -> List:
        feature_maps = self.backbone(img)
        if not self.split_neck:
            feature_maps = list(self.neck(feature_maps))
        return feature_maps

    def forward(self, data: Dict[str, Any]) -> Union[Dict, List]:
        feature_maps = self.extract_feat(data["img"])
        model_outs = OrderedDict()
        if isinstance(self.head, nn.ModuleDict):
            for head_name, head in self.head.items():
                head_outputs = self.forward_single_head(
                    head_name,
                    head,
                    feature_maps,
                    data,
                    self.compiler_model,
                )
                if isinstance(head_outputs, dict):
                    model_outs.update(
                        {
                            f"{head_name}.{k}": v
                            for k, v in head_outputs.items()
                        }
                    )
                else:
                    model_outs[head_name] = head_outputs
        else:
            model_outs = self.forward_single_head(
                "head",
                self.head,
                feature_maps,
                data,
                self.compiler_model,
            )

        out = self._post_process(feature_maps, model_outs, data)
        if os.getenv("HAT_TRAINING_STEP", "float") == "quant_analysis":
            if get_value("preparing_qat"):
                return out
            analysis_out = {}
            analysis_out["det.classification"] = out["det.classification"][-1]
            analysis_out["det.prediction"] = out["det.prediction"][-1]
            analysis_out["det.quality"] = out["det.quality"][-1]
            analysis_out["om.all_cls_scores"] = out["om.all_cls_scores"]
            analysis_out["om.all_bbox_preds"] = out["om.all_bbox_preds"]
            analysis_out["om.all_pts_preds"] = out["om.all_pts_preds"]
            return analysis_out
        return out


@OBJECT_REGISTRY.register
class SparseMultitaskIrInfer(SparseBEVIrInfer):
    """HBIR/BPU runtime for sparse multitask (det + map + occ)."""

    def __init__(
        self,
        ir_model: nn.Module,
        projection_mat_key: str,
        global_mat_key: str,
        first_frame_input: Dict[str, Any],
        max_time_interval: int = 2,
        det_decoder: nn.Module = None,
        use_memory_bank: bool = True,
        confidence_decay: float = 0.6,
        num_temp_instances: int = 128,
        num_memory_instances: int = 384,
        vt_input_hw: List[int] = None,
        deploy_model: nn.Module = None,
        model_convert_pipeline: List[callable] = None,
        use_nv12: bool = False,
    ):
        super(SparseMultitaskIrInfer, self).__init__(
            ir_model=ir_model,
            projection_mat_key=projection_mat_key,
            global_mat_key=global_mat_key,
            first_frame_input=first_frame_input,
            max_time_interval=max_time_interval,
            decoder=det_decoder,
            use_memory_bank=use_memory_bank,
            confidence_decay=confidence_decay,
            num_temp_instances=num_temp_instances,
            num_memory_instances=num_memory_instances,
            use_nv12=use_nv12,
        )
        if model_convert_pipeline is not None:
            self.ir_model = model_convert_pipeline(ir_model)
        self.deploy_model = deploy_model
        self.vt_input_hw = vt_input_hw

    def forward(self, metas: Dict[str, Any]) -> Dict[str, Any]:
        input_data = {"img": metas["img"]}
        if self.use_nv12 is True:
            # On-board BPU (HBM) runtime expects NV12 planar inputs (img_y /
            # img_uv) instead of packed RGB; the BPU pyramid input_source
            # converts on-chip. Mirrors SparseBEVIrInfer.forward /
            # SparseMapIrInfer.
            input_data.update(self.nv12_runtime(input_data["img"]))
        projection_mat = SparseBEVHead.gen_projection_mat(
            self.projection_mat_key, metas
        ).float()
        input_data["projection_mat"] = projection_mat

        device = metas["img"].device
        if self.metas is None:
            for k, v in self.first_frame_input.items():
                setattr(self, k, v.to(device))
        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(device=self.cached_anchor.device)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = InstanceBank.gen_temp2cur(
                self.global_mat_key, self.metas, metas
            ).to(device)
            self.cached_anchor = InstanceBank.anchor_projection(
                self.cached_anchor,
                T_temp2cur,
                time_interval,
            )
        input_data["cached_anchor"] = self.cached_anchor
        input_data["cached_feature"] = self.cached_feature
        if self.use_memory_bank is False:
            input_data["cached_confidence"] = self.cached_confidence
            input_data["mask"] = self.mask

        ref_p = self.deploy_model.head[
            "occ"
        ].view_transformer.export_reference_points(metas, self.vt_input_hw)
        input_data.update(ref_p)

        outs = self.ir_model(input_data)
        # occ_pred is the raw compiled head logits
        # (B, Dx, Dy, Dz, num_classes). HbirModule may wrap a single named
        # output as a 1-element list — unwrap before softmax. Keep the batch
        # dim and wrap the argmax result in a 1-element list so this matches
        # the float/eval path's occ contract: model_outs["occ"] = [occ_pre]
        # where occ_pre is a (B, Dx, Dy, Dz) class-index tensor. The val
        # callback does ``model_outs["occ"][0].squeeze()`` and
        # ``semantics_pred[mask]`` with a (Dx,Dy,Dz) boolean mask; a bare
        # (Dx,Dy,Dz) tensor (no batch, no list) made
        # ``model_outs["occ"][0]`` drop the leading axis -> 2D tensor and
        # crash with "too many indices for tensor of dimension 2". See
        # update_val_metric in the sparse_multitask callback.
        occ_pred = outs["occ"]
        if isinstance(occ_pred, (list, tuple)):
            occ_pred = occ_pred[0]
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)

        cache_data = {
            "classification": outs["det.classification"],
            "prediction": outs["det.prediction"],
            "feature": outs["det.feature"],
        }
        self.cache(metas, cache_data)
        det_results = self.decoder(
            outs["det.classification"],
            outs["det.prediction"],
            outs.get("det.quality"),
        )
        om_results = self.deploy_model.head["om"].get_outputs(
            outputs_classes=[outs["om.classification"]],
            reference_out=[outs["om.prediction"]],
        )
        om_results = self.deploy_model.head["om"].post_process(
            om_results, metas
        )
        return {
            "det": det_results,
            "occ": [occ_res],
            "om": om_results,
        }


# HAT config compatibility aliases.
OBJECT_REGISTRY.alias("SparseMultitaskOE")(SparseMultitask)
OBJECT_REGISTRY.alias("SparseMultitaskOEIrInfer")(SparseMultitaskIrInfer)
