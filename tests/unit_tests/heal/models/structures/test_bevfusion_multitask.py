# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import pytest

_N = 400
_OCC_LOGITS = (1, 200, 200, 16, 18)  # (B, Dx, Dy, Dz, num_classes)
_DYN_MEM = 384
_STA_MEM = 200
_DIM = 256


def test_bev_multitask_importable():
    from heal.models.structures.bevfusion_multitask import (
        BEVFusionMultitask,
        BevMultitaskIrInfer,
    )

    assert BEVFusionMultitask is not None
    assert BevMultitaskIrInfer is not None


def _make_modules():
    import torch
    import torch.nn as nn

    class IrModel(nn.Module):
        def forward(self, inputs):
            return {
                "dynamic_objects_det.classification": torch.randn(1, _N, 8),
                "dynamic_objects_det.prediction": torch.randn(1, _N, 11),
                "dynamic_objects_det.quality": torch.randn(1, _N, 1),
                "dynamic_objects_det.feature": torch.randn(1, _N, _DIM),
                "static_objects_det.classification": torch.randn(1, _N, 2),
                "static_objects_det.prediction": torch.randn(1, _N, 11),
                "static_objects_det.quality": torch.randn(1, _N, 1),
                "static_objects_det.feature": torch.randn(1, _N, _DIM),
                "occ": [torch.randn(*_OCC_LOGITS)],
                "discrete_map": [
                    torch.randn(1, 128, _DIM),
                    torch.randn(1, 100, 1),
                    torch.randn(1, 100, 6),
                    torch.randn(1, 100, 6),
                ],
                "maptr": [
                    torch.randn(1, 50, 3),
                    torch.randn(1, 50, 20, 2),
                ],
            }

    class FakeMaptrDecoder(nn.Module):
        def get_outputs(self, outputs_classes, reference_out):
            return {
                "all_cls_scores": outputs_classes[0],
                "all_bbox_preds": reference_out[0],
            }

    class FakeDiscreteMap(nn.Module):
        def get_outputs(self, cls, ref, pred, bev_embed):
            return [{"discrete": "decoded"}]

    class FakeDetDecoder(nn.Module):
        def forward(self, classification, prediction, quality):
            return [{"det": "decoded"}]

    class FakeDynDet(nn.Module):
        bev_size = (51.2, 51.2, 0.8)
        num_views = 6

    class FakeCameraNet(nn.Module):
        def export_reference_points(self, data):
            return (
                torch.zeros(6, 8, 8, 2),
                torch.zeros(1, 16, 8, 2),
                torch.zeros(6, 4, 4, 2),
                torch.zeros(1, 64, 1),
            )

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.camera_net = FakeCameraNet()
            maptr = nn.Module()
            maptr.bev_decoders = nn.ModuleList([FakeMaptrDecoder()])
            discrete_map = FakeDiscreteMap()
            dyn = FakeDynDet()
            self.task_decoders = nn.ModuleDict(
                {
                    "maptr": maptr,
                    "discrete_map": discrete_map,
                    "dynamic_objects_det": dyn,
                }
            )

    return IrModel, DeployModel, FakeDetDecoder


def _make_ir_infer(*, use_nv12=..., use_memory_bank=True):
    import torch

    from heal.models.structures.bevfusion_multitask import BevMultitaskIrInfer

    IrModel, DeployModel, FakeDetDecoder = _make_modules()
    first_frame_input = {
        "dynamic_objects_cached_anchor": torch.zeros((1, _DYN_MEM, 11)),
        "dynamic_objects_cached_feature": torch.zeros((1, _DYN_MEM, _DIM)),
        "dynamic_objects_cached_confidence": torch.zeros((1, _DYN_MEM)),
        "static_objects_cached_anchor": torch.zeros((1, _STA_MEM, 11)),
        "static_objects_cached_feature": torch.zeros((1, _STA_MEM, _DIM)),
        "static_objects_cached_confidence": torch.zeros((1, _STA_MEM)),
        "mask": torch.zeros((1,)).bool(),
    }
    kwargs = {
        "ir_model": IrModel(),
        "global_mat_key": "lidar2global",
        "first_frame_input": first_frame_input,
        "deploy_model": DeployModel(),
        "dynamic_objects_det_decoder": FakeDetDecoder(),
        "static_objects_det_decoder": FakeDetDecoder(),
        "maptr_post_process": lambda outputs, metas: [{"om": "decoded"}],
        "discrete_map_post_process": lambda outputs: outputs,
        "use_memory_bank": use_memory_bank,
        "dynamic_objects_num_temp_instances": 128,
        "dynamic_objects_num_memory_instances": _DYN_MEM,
        "static_objects_num_temp_instances": 80,
        "static_objects_num_memory_instances": _STA_MEM,
    }
    if use_nv12 is not ...:
        kwargs["use_nv12"] = use_nv12
    infer = BevMultitaskIrInfer(**kwargs).eval()
    metas = {
        "img": torch.randn(6, 3, 32, 48),
        "lidar2img": torch.eye(4).unsqueeze(0).repeat(6, 1, 1),
        "timestamp": torch.tensor(0.0),
    }
    return infer, metas


def test_ir_infer_forward_returns_task_keys():
    pytest.importorskip("torch")
    import torch

    infer, metas = _make_ir_infer(use_nv12=False)
    with torch.no_grad():
        outputs = infer(metas)

    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == {
        "dynamic_objects_det",
        "static_objects_det",
        "occ",
        "maptr",
        "discrete_map",
    }


def test_ir_infer_occ_output_contract():
    """occ is a 1-element list; [0][b] is 3D (Dx,Dy,Dz) for callback mask."""
    pytest.importorskip("torch")
    import torch

    infer, metas = _make_ir_infer(use_nv12=False)
    with torch.no_grad():
        outputs = infer(metas)

    occ = outputs["occ"]
    assert isinstance(occ, list)
    assert len(occ) == 1
    assert isinstance(occ[0], torch.Tensor)
    assert occ[0].dim() == 4
    assert tuple(occ[0][0].shape) == (200, 200, 16)


def test_ir_infer_occ_satisfies_callback_mask():
    """End-to-end: callback-style indexing must not IndexError."""
    pytest.importorskip("torch")
    import torch

    infer, metas = _make_ir_infer(use_nv12=False)
    with torch.no_grad():
        model_outs = infer(metas)

    gt_occ = {
        "voxel_semantics": torch.randint(
            0, 18, (1, 200, 200, 16), dtype=torch.uint8
        ),
        "mask_camera": torch.ones(1, 200, 200, 16, dtype=torch.bool),
    }
    occ_out = model_outs["occ"]
    semantics_pred_all = (
        occ_out[0] if isinstance(occ_out, (list, tuple)) else occ_out
    )
    if semantics_pred_all.dim() == 5:
        semantics_pred_all = semantics_pred_all.argmax(-1)

    for b_idx in range(semantics_pred_all.shape[0]):
        gt_semantics = gt_occ["voxel_semantics"][b_idx].squeeze()
        semantics_pred = semantics_pred_all[b_idx].squeeze()
        mask = gt_occ["mask_camera"][b_idx].squeeze().bool()
        assert gt_semantics.shape == semantics_pred.shape == mask.shape
        assert gt_semantics.dim() == 3
        label = gt_semantics[mask].reshape(-1)
        preds = semantics_pred[mask].reshape(-1)
        assert label.shape == preds.shape


def test_ir_infer_accepts_use_nv12():
    """use_nv12 is accepted; default remains True for HBM-oriented ctor."""
    pytest.importorskip("torch")

    infer_default, _ = _make_ir_infer()
    assert infer_default.use_nv12 is True

    infer_hbir, _ = _make_ir_infer(use_nv12=False)
    assert infer_hbir.use_nv12 is False

    infer_hbm, _ = _make_ir_infer(use_nv12=True)
    assert infer_hbm.use_nv12 is True
