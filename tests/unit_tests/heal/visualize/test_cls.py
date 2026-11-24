import torch

from heal.visualize.cls import ClsViz


def test_cls_viz_returns_argmax_and_updates_sample_idx():
    viz = ClsViz()
    preds = torch.tensor([[0.1, 0.9, 0.2]])
    inputs = {"img": torch.zeros(1, 8, 8, 3), "classes": {1: "class-1"}}

    pred_idx = viz(preds, inputs)

    assert pred_idx == 1
    assert viz.sample_idx == 1


def test_cls_viz_handles_spatial_logits():
    viz = ClsViz()
    preds = torch.randn(1, 1000, 1, 1)
    inputs = {"img": torch.zeros(1, 8, 8, 3)}

    pred_idx = viz(preds, inputs)

    assert 0 <= pred_idx < 1000


def test_cls_viz_saves_image_when_save_path_set(tmp_path):
    viz = ClsViz(save_path=str(tmp_path))
    preds = torch.tensor([[0.4, 0.6]])
    inputs = {"img": torch.zeros(1, 8, 8, 3)}

    pred_idx = viz(preds, inputs)

    assert pred_idx == 1
    assert (tmp_path / "cls_pred_0.png").is_file()
