"""Tests for heal.models.task_modules.ganet.decoder."""

import pytest


def test_ganet_decoder_forward_empty():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.ganet.decoder import GaNetDecoder

    decoder = GaNetDecoder(
        root_thr=1.0,
        kpt_thr=0.4,
        cluster_thr=4.0,
        downscale=8,
        min_points=2,
    )
    heat = torch.zeros(1, 1, 8, 16)
    offset = torch.zeros(1, 2, 8, 16)
    error = torch.zeros(1, 2, 8, 16)
    meta_data = {
        "scale_factor": torch.tensor([[1.0, 1.0]]),
        "crop_offset": ((0.0, 0.0),),
    }
    results = decoder(heat, offset, error, meta_data)
    assert isinstance(results, list)
    assert len(results) == 1


def test_ganet_decoder_forward_with_keypoint():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.ganet.decoder import GaNetDecoder

    decoder = GaNetDecoder(
        root_thr=1.0,
        kpt_thr=0.4,
        cluster_thr=4.0,
        downscale=8,
        min_points=2,
    )
    heat = torch.zeros(1, 1, 8, 16)
    heat[0, 0, 4, 8] = 10.0
    offset = torch.zeros(1, 2, 8, 16)
    offset[0, 1, 4, 8] = -5.0
    error = torch.zeros(1, 2, 8, 16)
    meta_data = {
        "scale_factor": torch.tensor([[1.0, 1.0]]),
        "crop_offset": ((0.0, 0.0),),
    }
    results = decoder(heat, offset, error, meta_data)
    assert isinstance(results, list)
    assert len(results) == 1
