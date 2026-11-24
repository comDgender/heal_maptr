"""Tests for heal.models.task_modules.flashocc.decoder."""

import pytest


def test_flashocc_det_decoder_importable():
    from heal.models.task_modules.flashocc.decoder import (
        FlashOccDecoderInfer,
        FlashOccDetDecoder,
    )

    assert FlashOccDetDecoder is not None
    assert FlashOccDecoderInfer is not None


def test_flashocc_det_decoder_forward_eval():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.task_modules.flashocc.bev_occ_head import BEVOCCHead2D
    from heal.models.task_modules.flashocc.decoder import FlashOccDetDecoder

    head = BEVOCCHead2D(
        in_dim=48,
        out_dim=128,
        Dz=16,
        num_classes=18,
        use_predicter=True,
        use_upsample=True,
    )
    decoder = FlashOccDetDecoder(
        occ_head=head,
        num_classes=18,
        use_mask=True,
    ).eval()

    # BEV feature: (B, C, H, W); use_upsample upsamples to 200×200 internally
    bev_feat = [torch.randn(1, 48, 50, 50)]
    with torch.no_grad():
        preds, results = decoder(bev_feat, data={})

    assert "occ_pre" in results
    # argmax over 18 classes → (B, 200, 200, 16)
    assert results["occ_pre"].shape == (1, 200, 200, 16)
    assert results["occ_pre"].dtype == torch.int64


def test_flashocc_det_decoder_forward_train_loss():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.losses.cross_entropy_loss import CrossEntropyLoss
    from heal.models.task_modules.flashocc.bev_occ_head import BEVOCCHead2D
    from heal.models.task_modules.flashocc.decoder import FlashOccDetDecoder

    head = BEVOCCHead2D(
        in_dim=48,
        out_dim=128,
        Dz=16,
        num_classes=18,
        use_predicter=True,
        use_upsample=True,
    )
    decoder = FlashOccDetDecoder(
        occ_head=head,
        loss_occ=CrossEntropyLoss(
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
        ),
        num_classes=18,
        use_mask=True,
    ).train()

    # In train mode forward() routes through _post_process → _loss.
    bev_feat = [torch.randn(1, 48, 50, 50)]
    data = {
        "gt_occ_info": {
            # gt voxel labels and camera mask match the (B, 200, 200, 16)
            # spatial layout produced by the upsampling occ head.
            "voxel_semantics": torch.randint(0, 18, (1, 200, 200, 16)),
            "mask_camera": torch.ones(1, 200, 200, 16),
        }
    }
    preds, losses = decoder(bev_feat, data)

    assert isinstance(preds, list)
    assert "loss_occ" in losses
    # Scalar loss that is differentiable w.r.t. the head parameters.
    assert losses["loss_occ"].ndim == 0
    assert losses["loss_occ"].requires_grad


def test_flashocc_decoder_infer_forward():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.task_modules.flashocc.decoder import FlashOccDecoderInfer

    infer = FlashOccDecoderInfer(name="occ_det")

    # simulate raw logits from HBM/HBIR: (1, 200, 200, 16, 18)
    raw_logits = [torch.randn(1, 200, 200, 16, 18)]
    with torch.no_grad():
        _, results = infer(raw_logits, data={})

    assert "occ_pre" in results
    assert results["occ_pre"].shape == (1, 200, 200, 16)
