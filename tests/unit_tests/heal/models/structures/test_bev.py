"""Tests for heal.models.structures.bev."""

import pytest


def _make_bev_modules():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class Backbone(nn.Module):
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 8, x.shape[-2] // 4, x.shape[-1] // 4)
            ]

    class Neck(nn.Module):
        def forward(self, feats):
            return feats

    class ViewTransformer(nn.Module):
        def forward(self, feat, data, compile_model):
            return torch.randn(feat.shape[0], 16, 8, 8), {}

        def export_reference_points(self, data, feat_wh):
            return {"ref": torch.zeros(1)}

    class BevEncoder(nn.Module):
        def forward(self, bev_feat, data):
            return bev_feat

    class BevDecoder(nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def forward(self, bev_feat, data):
            # Bev.forward always passes bev_feat as a Sequence (list) of
            # feature tensors; index the first one for the batch dim.
            feat = (
                bev_feat[0]
                if isinstance(bev_feat, (list, tuple))
                else bev_feat
            )
            pred = [torch.randn(feat.shape[0], 4, 8, 8)]
            return pred, {self.tag: pred[0]}

    return Backbone, Neck, ViewTransformer, BevEncoder, BevDecoder


def test_bev_forward_eval():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.bev import Bev

    BB, Neck, VT, Enc, Dec = _make_bev_modules()
    model = Bev(
        BB(),
        Neck(),
        VT(),
        Enc(),
        [Dec("det")],
        bev_upscale=1,
    ).eval()
    data = {"img": torch.randn(1, 3, 32, 32)}
    preds, results = model(data)
    assert preds is not None
    assert "det" in results


def test_bev_forward_compile_model():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.bev import Bev

    BB, Neck, VT, Enc, Dec = _make_bev_modules()
    model = Bev(
        BB(),
        Neck(),
        VT(),
        Enc(),
        [Dec("det")],
        bev_upscale=1,
        compile_model=True,
    ).eval()
    preds = model({"img": torch.randn(1, 3, 32, 32)})
    assert isinstance(preds, list)


def test_bev_img_encode_and_export_reference_points():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.bev import Bev

    BB, Neck, VT, Enc, Dec = _make_bev_modules()
    model = Bev(BB(), Neck(), VT(), Enc(), [Dec("det")], bev_upscale=1).eval()
    img = torch.randn(1, 3, 32, 32)
    feats = model.img_encode(img)
    assert isinstance(feats, list)
    ref = model.export_reference_points({"img": img}, (8, 8))
    assert "ref" in ref


def test_bev_ir_infer_forward():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.bev import BevIrInfer

    class DeployModel(nn.Module):
        def export_reference_points(self, data, feat_wh):
            return {"ref": torch.zeros(1)}

    class IrModel(nn.Module):
        def forward(self, inputs):
            return [torch.randn(1, 4)]

    class DecoderInfer(nn.Module):
        def input_num(self):
            return 1

        def forward(self, decoder_input, data):
            return None, {"pred": decoder_input[0]}

    infer = BevIrInfer(
        IrModel(),
        DeployModel(),
        lambda m: m,
        vt_input_hw=[8, 8],
        bev_decoder_infers=[DecoderInfer()],
    ).eval()
    _, results = infer({"img": torch.randn(1, 3, 32, 32)})
    assert "pred" in results


def test_bev_upscale_and_multiple_decoders():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch
    import torch.nn as nn

    from heal.models.structures.bev import Bev

    BB, Neck, VT, Enc, Dec = _make_bev_modules()

    class TensorEncoder(nn.Module):
        def forward(self, bev_feat, data):
            return bev_feat

    model = Bev(
        BB(),
        Neck(),
        VT(),
        TensorEncoder(),
        [Dec("det"), Dec("seg")],
        bev_upscale=2,
    ).eval()
    preds, results = model({"img": torch.randn(1, 3, 32, 32)})
    assert preds is not None
    assert "det" in results and "seg" in results


def test_bev_fuse_model_and_set_qconfig():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch.nn as nn

    from heal.models.structures.bev import Bev

    BB, Neck, VT, Enc, Dec = _make_bev_modules()

    class FuseChild(nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = False

        def fuse_model(self):
            self.fused = True

        def set_qconfig(self):
            self.qconfig_set = True

    bb = FuseChild()
    model = Bev(bb, Neck(), VT(), Enc(), [Dec("det")], bev_upscale=1)
    model.fuse_model()
    assert bb.fused
    model.set_qconfig()
    assert hasattr(bb, "qconfig_set")


def test_bev_ir_infer_dict_output_and_multi_decoder():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.bev import BevIrInfer

    class DeployModel(nn.Module):
        def export_reference_points(self, data, feat_wh):
            return {"ref": torch.zeros(1)}

    class IrModel(nn.Module):
        def forward(self, inputs):
            return {"a": torch.randn(1, 2), "b": torch.randn(1, 3)}

    class DecoderInfer(nn.Module):
        def __init__(self, n, tag):
            super().__init__()
            self.n = n
            self.tag = tag

        def input_num(self):
            return self.n

        def forward(self, decoder_input, data):
            return None, {self.tag: decoder_input[0]}

    infer = BevIrInfer(
        IrModel(),
        DeployModel(),
        lambda m: m,
        vt_input_hw=[8, 8],
        bev_decoder_infers=[DecoderInfer(1, "det"), DecoderInfer(1, "seg")],
    ).eval()
    _, results = infer({"img": torch.randn(1, 3, 32, 32)})
    assert "det" in results and "seg" in results


def test_bev_ir_infer_nv12_single_and_multi_batch():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch
    import torch.nn as nn

    from heal.models.structures.bev import BevIrInfer

    class DeployModel(nn.Module):
        def export_reference_points(self, data, feat_wh):
            return {}

    class IrModel(nn.Module):
        def forward(self, inputs):
            return [torch.randn(1, 1)]

    class DecoderInfer(nn.Module):
        def input_num(self):
            return 1

        def forward(self, decoder_input, data):
            return None, {"pred": decoder_input[0]}

    infer = BevIrInfer(
        IrModel(),
        DeployModel(),
        lambda m: m,
        vt_input_hw=[8, 8],
        bev_decoder_infers=[DecoderInfer()],
        use_nv12=True,
    ).eval()
    img = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.uint8)
    planes = infer.nv12_runtime(img)
    assert "img_y" in planes and "img_uv" in planes

    img2 = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    planes2 = infer.nv12_runtime(img2)
    assert "img_0_y" in planes2 and "img_1_uv" in planes2

    _, results = infer({"img": img})
    assert "pred" in results
