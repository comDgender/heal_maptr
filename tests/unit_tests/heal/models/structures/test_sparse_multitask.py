"""Tests for heal.models.structures.sparse_multitask.

These lock in three regression contracts for ``SparseMultitaskIrInfer``
(``heal/models/structures/sparse_multitask.py``), each of which previously
crashed ``tools/inference.py`` / ``tools/predict.py`` on the IR/HBM path:

* Bug A — ``SparseMultitaskIrInfer.__init__`` must accept ``use_nv12`` and
  forward it to the parent ``SparseBEVIrInfer`` (hbm stage), and ``forward``
  must run the NV12 branch when ``use_nv12=True``.
* Bug B — ``SparseMultitaskIrInfer.forward`` calls
  ``deploy_model.head["om"].post_process(...)``; that attribute must be a
  callable (the config must NOT pop it off ``deploy_model``).
* Bug C — ``forward`` must return ``occ`` as a 1-element list wrapping a
  ``(B, Dx, Dy, Dz)`` class-index tensor, matching the float/eval path's
  ``[occ_pre]`` contract that the val callback's
  ``model_outs["occ"][0].squeeze()`` + 3D bool mask relies on. A bare
  ``(Dx,Dy,Dz)`` tensor (no batch, no list) made ``model_outs["occ"][0]``
  drop the leading axis -> 2D -> ``IndexError: too many indices``.
"""

import pytest

# Anchor/query count large enough that the default ``num_temp_instances``
# (128) and ``num_memory_instances`` (384) ``torch.topk`` in
# InstanceBank.cache_instance does not go out of range.
_N = 400
_OCC_LOGITS = (1, 200, 200, 16, 18)  # (B, Dx, Dy, Dz, num_classes)


def test_sparse_multitask_importable():
    from heal.models.structures.sparse_multitask import (
        SparseMultitask,
        SparseMultitaskIrInfer,
    )

    assert SparseMultitask is not None
    assert SparseMultitaskIrInfer is not None


def _make_modules():
    """Build the fakes needed to construct a SparseMultitaskIrInfer.

    Returns ``(IrModel, FakeOmHead, FakeOccHead, FakeDetDecoder)`` classes.
    Mirrors the IR-output contract SparseMultitaskIrInfer.forward consumes:
    outs["occ"], outs["det.classification"/"prediction"/"feature"/"quality"],
    outs["om.classification"/"prediction"].
    """
    import torch
    import torch.nn as nn

    class IrModel(nn.Module):
        def forward(self, inputs):
            # HbirModule commonly wraps a single named output as a 1-element
            # list; SparseMultitaskIrInfer must unwrap before softmax.
            return {
                "occ": [torch.randn(*_OCC_LOGITS)],
                "det.classification": torch.randn(1, _N, 4),
                "det.prediction": torch.randn(1, _N, 11),
                "det.feature": torch.randn(1, _N, 256),
                "det.quality": torch.randn(1, _N, 1),
                "om.classification": torch.randn(1, _N, 3),
                "om.prediction": torch.randn(1, _N, 50, 2),
            }

    class FakeOmHead(nn.Module):
        """get_outputs + a post_process that records it was called (Bug B)."""

        def __init__(self):
            super().__init__()
            self.post_process_called = False

        def get_outputs(self, outputs_classes, reference_out):
            return {
                "all_cls_scores": outputs_classes[0],
                "all_bbox_preds": reference_out[0],
                "all_pts_preds": reference_out[0],
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
                "enc_pts_preds": None,
            }

        def post_process(self, outputs, data):
            self.post_process_called = True
            return [{"om": "decoded"}]

    class FakeOccHead(nn.Module):
        class _VT:
            def export_reference_points(self, metas, vt_input_hw):
                return {
                    "points0": torch.randn(1, 128, 128, 2),
                    "points1": torch.randn(1, 128, 128, 2),
                }

        view_transformer = _VT()

    class FakeDetDecoder(nn.Module):
        def forward(self, classification, prediction, quality):
            return [{"det": "decoded"}]

    return IrModel, FakeOmHead, FakeOccHead, FakeDetDecoder


def _make_ir_infer(
    *,
    use_memory_bank=False,
    use_nv12=...,
    model_convert_pipeline=None,
):
    """Construct a SparseMultitaskIrInfer with fake sub-modules + first metas.

    Returns ``(infer, om_head, metas)`` so tests can inspect the om head.
    ``use_nv12=...`` (sentinel) means "do not pass the kwarg" (exercises the
    default).
    """
    import torch
    import torch.nn as nn

    from heal.models.structures.sparse_multitask import SparseMultitaskIrInfer

    IrModel, FakeOmHead, FakeOccHead, FakeDetDecoder = _make_modules()
    om_head = FakeOmHead()

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.ModuleDict({"om": om_head, "occ": FakeOccHead()})

    first_frame_input = {
        "cached_anchor": torch.zeros((1, _N, 11)),
        "cached_feature": torch.zeros((1, _N, 256)),
        "cached_confidence": torch.zeros((1, _N)),
        "mask": torch.zeros((1,)).bool(),
    }

    kwargs = {
        "ir_model": IrModel(),
        "projection_mat_key": "lidar2img",
        "global_mat_key": "lidar2global",
        "first_frame_input": first_frame_input,
        "det_decoder": FakeDetDecoder(),
        "use_memory_bank": use_memory_bank,
        "vt_input_hw": [256, 704],
        "deploy_model": DeployModel(),
        "model_convert_pipeline": model_convert_pipeline
        or (lambda model: model),
    }
    if use_nv12 is not ...:
        kwargs["use_nv12"] = use_nv12
    infer = SparseMultitaskIrInfer(**kwargs).eval()

    metas = {
        "img": torch.randn(6, 3, 32, 48),
        "lidar2img": torch.eye(4).unsqueeze(0).repeat(6, 1, 1),
        "timestamp": torch.tensor(0.0),
    }
    return infer, om_head, metas


def test_ir_infer_forward_returns_det_occ_om():
    pytest.importorskip("torch")
    import torch

    infer, _om_head, metas = _make_ir_infer()

    with torch.no_grad():
        outputs = infer(metas)

    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == {"det", "occ", "om"}
    # om is whatever deploy_model.head["om"].post_process returned.
    assert outputs["om"] == [{"om": "decoded"}]
    assert isinstance(outputs["det"], list)


def test_ir_infer_occ_output_contract():
    """Bug C: occ is a 1-element list; [0].squeeze() is 3D (Dx,Dy,Dz)."""
    pytest.importorskip("torch")
    import torch

    infer, _om_head, metas = _make_ir_infer()

    with torch.no_grad():
        outputs = infer(metas)

    occ = outputs["occ"]
    assert isinstance(occ, list)
    assert len(occ) == 1
    assert isinstance(occ[0], torch.Tensor)
    # (1, Dx, Dy, Dz) class-index tensor after argmax(-1) over the 5D logits.
    assert occ[0].dim() == 4
    squeezed = occ[0].squeeze()
    assert squeezed.dim() == 3
    assert tuple(squeezed.shape) == (200, 200, 16)


def test_ir_infer_occ_satisfies_callback_mask():
    """Bug C end-to-end: the occ output shape the val callback needs.

    Replicates ``update_val_metric``'s occ branch
    (configs/sparse_multitask_.../callback.py) against the IR-infer output:
    ``model_outs["occ"][0].squeeze()`` must yield a 3D (Dx,Dy,Dz) tensor that a
    3D bool ``mask_camera`` can index without ``IndexError``.
    """
    pytest.importorskip("torch")
    import torch

    infer, _om_head, metas = _make_ir_infer()

    with torch.no_grad():
        model_outs = infer(metas)

    batch = {
        "gt_occ_info": {
            "voxel_semantics": torch.randint(
                0, 18, (1, 200, 200, 16), dtype=torch.uint8
            ),
            "mask_camera": torch.ones(1, 200, 200, 16, dtype=torch.bool),
        }
    }

    gt_semantics = batch["gt_occ_info"]["voxel_semantics"][0].squeeze()
    semantics_pred = model_outs["occ"][0].squeeze()
    mask = batch["gt_occ_info"]["mask_camera"][0].squeeze()

    assert gt_semantics.shape == semantics_pred.shape == mask.shape
    assert gt_semantics.dim() == 3

    masked_gt = gt_semantics[mask]
    masked_pred = semantics_pred[mask]
    label = masked_gt.reshape(-1)
    preds = masked_pred.reshape(-1)

    assert label.dim() == 1 and preds.dim() == 1
    assert label.shape == preds.shape


def test_ir_infer_accepts_use_nv12_and_routes():
    """Bug A: use_nv12 accepted, forwarded to parent, routes in forward."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")  # nv12_runtime imports cv2
    import numpy as np
    import torch

    # With use_nv12=True, no unexpected-kwarg error (Bug A regression).
    infer, _om_head, _metas = _make_ir_infer(use_nv12=True)
    assert infer.use_nv12 is True

    # nv12_runtime expects a uint8 RGB NCHW image (the hbm pre_processor feeds
    # raw RGB uint8). Single-batch returns {"img_y", "img_uv"}.
    metas = {
        "img": torch.from_numpy(
            (np.random.rand(1, 3, 32, 48) * 255).astype("uint8")
        ),
        "lidar2img": torch.eye(4).unsqueeze(0).repeat(1, 1, 1),
        "timestamp": torch.tensor(0.0),
    }
    with torch.no_grad():
        outputs = infer(metas)
    assert set(outputs.keys()) == {"det", "occ", "om"}

    # Default (omitted) keeps use_nv12 False -> hbir path not broken.
    infer_default, _o2, _m2 = _make_ir_infer()
    assert infer_default.use_nv12 is False


def test_ir_infer_post_process_contract():
    """Bug B: forward calls deploy_model.head["om"].post_process (must be
    present — config must NOT pop it)."""
    pytest.importorskip("torch")
    import torch

    infer, om_head, metas = _make_ir_infer()

    with torch.no_grad():
        infer(metas)

    assert om_head.post_process_called is True
