"""Tests for heal.models.task_modules.bevformer.view_transformer."""

import pytest


def _make_view_transformer():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch.nn as nn

    from heal.models.task_modules.bevformer.view_transformer import (
        BevFormerViewTransformer,
    )

    class MockEncoder(nn.Module):
        def forward(self, *args, **kwargs):
            return args[0]

    class MockPosEncoding(nn.Module):
        def forward(self, mask):
            import torch

            b, h, w = mask.shape
            return torch.zeros(b, 16, h, w, dtype=mask.dtype)

    return BevFormerViewTransformer(
        pc_range=[-10.0, -10.0, -1.0, 10.0, 10.0, 1.0],
        num_points_in_pillar=2,
        bev_h=4,
        bev_w=4,
        embed_dims=16,
        queue_length=1,
        max_camoverlap_num=2,
        virtual_bev_h=4,
        virtual_bev_w=4,
        encoder=MockEncoder(),
        positional_encoding=MockPosEncoding(),
        numcam=2,
        in_indices=(0,),
        single_bev=True,
    )


def test_gen_reference_points_2d_and_3d():
    pytest.importorskip("torch")
    vt = _make_view_transformer().eval()

    ref_3d = vt.gen_reference_points(4, 4, 2, 2, dim="3d", bs=1)
    assert ref_3d.shape == (1, 2, 16, 3)

    ref_2d = vt.gen_reference_points(4, 4, dim="2d", bs=1)
    assert ref_2d.shape == (1, 16, 1, 2)


def test_export_reference_points():
    pytest.importorskip("torch")
    import torch

    vt = _make_view_transformer().eval()
    ref_2d, ref_3d = vt.export_reference_points(
        bs=2, device=torch.device("cpu")
    )
    assert ref_2d.shape[0] == 2
    assert ref_3d.shape[0] == 2
