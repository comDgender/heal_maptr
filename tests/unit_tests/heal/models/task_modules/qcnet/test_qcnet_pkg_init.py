"""Tests for heal.models.task_modules.qcnet modules."""


def test_qcnet_task_modules_importable():
    from heal.models.task_modules.qcnet.agent_st_module import (
        QCNetOEAgentEncoderStream,
    )
    from heal.models.task_modules.qcnet.loss import QCNetOELoss
    from heal.models.task_modules.qcnet.map_module import QCNetOEMapEncoder
    from heal.models.task_modules.qcnet.postprocess import QCNetOEPostprocess
    from heal.models.task_modules.qcnet.preprocess import QCNetOEPreprocess
    from heal.models.task_modules.qcnet.qc_decoder import QCNetOEDecoder
    from heal.models.task_modules.qcnet.qcnet_encoder import QCNetOEEncoder

    assert QCNetOEEncoder is not None
    assert QCNetOEMapEncoder is not None
    assert QCNetOEAgentEncoderStream is not None
    assert QCNetOEDecoder is not None
    assert QCNetOELoss is not None
    assert QCNetOEPreprocess is not None
    assert QCNetOEPostprocess is not None


def test_qcnet_utils_importable():
    from heal.models.task_modules.qcnet.fourier_embedding import (
        FourierConvEmbedding,
        FourierEmbedding,
    )
    from heal.models.task_modules.qcnet.rattention import RAttentionLayer
    from heal.models.task_modules.qcnet.utils import weight_init, wrap_angle

    assert FourierEmbedding is not None
    assert FourierConvEmbedding is not None
    assert RAttentionLayer is not None
    assert callable(weight_init)
    assert callable(wrap_angle)
