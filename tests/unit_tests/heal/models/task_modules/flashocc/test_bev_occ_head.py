"""Tests for heal.models.task_modules.flashocc.bev_occ_head."""

import pytest


def test_bev_occ_head_importable():
    from heal.models.task_modules.flashocc.bev_occ_head import BEVOCCHead2D

    assert BEVOCCHead2D is not None


def test_bev_occ_head_forward_with_upsample():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.task_modules.flashocc.bev_occ_head import BEVOCCHead2D

    head = BEVOCCHead2D(
        in_dim=48,
        out_dim=128,
        Dz=16,
        num_classes=18,
        use_predicter=True,
        use_upsample=True,
    ).eval()

    # small spatial size; use_upsample=True interpolates to 200×200
    img_feats = torch.randn(1, 48, 50, 50)
    with torch.no_grad():
        occ_pred = head(img_feats)

    # (B, Dx, Dy, Dz, num_classes)
    assert occ_pred.shape == (1, 200, 200, 16, 18)


def test_bev_occ_head_forward_without_upsample():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.task_modules.flashocc.bev_occ_head import BEVOCCHead2D

    head = BEVOCCHead2D(
        in_dim=32,
        out_dim=64,
        Dz=8,
        num_classes=18,
        use_predicter=True,
        use_upsample=False,
    ).eval()

    img_feats = torch.randn(1, 32, 10, 10)
    with torch.no_grad():
        occ_pred = head(img_feats)

    assert occ_pred.shape == (1, 10, 10, 8, 18)
