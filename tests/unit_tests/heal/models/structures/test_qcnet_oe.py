"""Tests for heal.models.structures.qcnet_oe."""

import pytest


def test_qcnet_oe_importable():
    from heal.models.structures.qcnet_oe import (
        QCNetOE,
        QCNetOEHbmInfer,
        QCNetOEIrInfer,
    )

    assert QCNetOE is not None
    assert QCNetOEIrInfer is not None
    assert QCNetOEHbmInfer is not None


def _build_small_qcnet_cfg():
    input_dim = 2
    hidden_dim = 8
    ori_historical_sec = 5
    ori_future_sec = 6
    ori_sample_fre = 10
    sample_fre = 2
    num_historical_steps = ori_historical_sec * sample_fre
    num_future_steps = ori_future_sec * sample_fre
    num_freq_bands = 4
    num_heads = 2
    head_dim = 4
    output_dim = 2
    time_span = 2
    dropout = 0.1
    num_map_layers = 1
    num_agent_layers = 1
    num_modes = 2
    num_recurrent_steps = 1
    num_t2m_steps = 30 // ori_sample_fre * sample_fre
    num_dec_layers = 1
    reuse_agent_rembs = True
    split_rec_modules = True

    model = dict(  # noqa C408
        type="QCNetOE",
        encoder=dict(  # noqa C408
            type="QCNetOEEncoder",
            map_encoder=dict(  # noqa C408
                type="QCNetOEMapEncoder",
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_freq_bands=num_freq_bands,
                num_layers=num_map_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            ),
            agent_encoder=dict(  # noqa C408
                type="QCNetOEAgentEncoderStream",
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                time_span=time_span,
                num_freq_bands=num_freq_bands,
                num_layers=num_agent_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                num_t2m_steps=num_t2m_steps,
                dropout=dropout,
                save_memory=True,
                stream_infer=True,
                reuse_agent_rembs=reuse_agent_rembs,
                deploy=False,
            ),
        ),
        decoder=dict(  # noqa C408
            type="QCNetOEDecoder",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            split_rec_modules=split_rec_modules,
            reuse_agent_rembs=reuse_agent_rembs,
            deploy=False,
        ),
        loss=dict(  # noqa C408
            type="QCNetOELoss",
            output_dim=output_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
        ),
        preprocess=dict(  # noqa C408
            type="QCNetOEPreprocess",
            input_dim=input_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            num_t2m_steps=num_t2m_steps,
            num_agent_layers=num_agent_layers,
            num_pl2a=32,
            num_a2a=36,
            stream=True,
            save_memory=True,
            deploy=False,
        ),
        postprocess=dict(  # noqa C408
            type="QCNetOEPostprocess",
            output_dim=output_dim,
            num_historical_steps=num_historical_steps,
        ),
    )
    meta = dict(  # noqa C408
        num_historical_steps=num_historical_steps,
        num_future_steps=num_future_steps,
        num_modes=num_modes,
        output_dim=output_dim,
        ori_historical_sec=ori_historical_sec,
        sample_fre=sample_fre,
    )
    return model, meta


def _build_dummy_data(meta):
    import torch

    B = 1
    A = 3
    pl_N = 8
    pt_N = 5
    T = meta["num_historical_steps"] + meta["num_future_steps"]
    HT = meta["ori_historical_sec"] * meta["sample_fre"]

    data = {}

    mp = {}
    mp["position"] = torch.rand([B, pl_N, pt_N, 2])
    mp["orientation"] = torch.rand([B, pl_N, pt_N])
    mp["magnitude"] = torch.rand([B, pl_N, pt_N])
    mp["height"] = torch.rand([B, pl_N, pt_N])
    mp["pt_type"] = (torch.rand([B, pl_N, pt_N]) * 3).long()
    mp["side"] = (torch.rand([B, pl_N, pt_N]) * 3).long()
    mp["mask"] = torch.rand([B, pl_N, pt_N]) > 0.2
    data["map_point"] = mp

    mpl = {}
    mpl["position"] = torch.rand([B, pl_N, 2])
    mpl["orientation"] = torch.rand([B, pl_N])
    mpl["height"] = torch.rand([B, pl_N])
    mpl["pl_type"] = (torch.rand([B, pl_N]) * 3).long()
    mpl["is_intersection"] = (torch.rand([B, pl_N]) * 3).long()
    mpl["valid_mask"] = torch.rand([B, pl_N]) > 0.2
    data["map_polygon"] = mpl

    agent = {}
    agent["num_nodes"] = [A]
    agent["valid_mask"] = torch.rand([B, A, T]) > 0.2
    agent["predict_mask"] = torch.rand([B, A, T]) > 0.2
    agent["agent_type"] = (torch.rand([B, A, 1]) * 3).long()
    agent["position"] = torch.rand([B, A, T, 2])
    agent["heading"] = torch.rand([B, A, T])
    agent["velocity"] = torch.rand([B, A, T, 2])
    agent["category"] = (torch.rand([B, A, 1]) * 3).long()
    mask_ta = agent["valid_mask"].transpose(1, 2)[:, :HT]
    agent["valid_mask_a2a"] = mask_ta[:, :, :, None] & mask_ta[:, :, None, :]
    data["agent"] = agent

    decoder = {}
    decoder["mask_dst"] = (torch.rand(1, A, 1) > 0.2).bool()
    decoder["mask_a2m"] = (torch.rand(1, A, A) > 0.2).bool()
    data["decoder"] = decoder

    data["type_pl2pl"] = (torch.rand([B, pl_N, pl_N]) * 5).long()
    return data


def test_qcnet_oe_float_forward():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    from hat_engine.registry import build_from_registry

    import heal  # noqa: F401

    model_cfg, meta = _build_small_qcnet_cfg()
    data = _build_dummy_data(meta)

    qcnet = build_from_registry(model_cfg)
    qcnet.eval()
    results = qcnet(data)

    assert "pred" in results and "prob" in results
    assert results["pred"].shape[1] == meta["num_modes"]
    assert results["pred"].shape[2] == meta["num_future_steps"]
    assert results["pred"].shape[3] == meta["output_dim"]
    assert results["prob"].shape[1] == meta["num_modes"]
    assert results["pred"].shape[0] == results["prob"].shape[0]
