from heal.data.transforms.classification import TimmMixup, TimmTransforms


def test_timm_transforms_call_updates_img_in_place():
    transform = TimmTransforms.__new__(TimmTransforms)
    transform.transform = lambda img: f"tx::{img}"
    data = {"img": "raw-image"}

    out = transform(data)

    assert out is data
    assert out["img"] == "tx::raw-image"


def test_timm_mixup_even_batch_updates_img_and_labels():
    mixup = TimmMixup.__new__(TimmMixup)
    mixup.mixup = lambda x, y: (["x0", "x1"], ["y0", "y1"])
    data = {"img": ["a", "b"], "labels": [1, 2]}

    out = mixup(data)

    assert out["img"] == ["x0", "x1"]
    assert out["labels"] == ["y0", "y1"]


def test_timm_mixup_odd_batch_keeps_original_and_logs_warning(caplog):
    mixup = TimmMixup.__new__(TimmMixup)
    mixup.mixup = lambda x, y: (["changed"], ["changed"])
    data = {"img": ["only-one"], "labels": [7]}

    with caplog.at_level("WARNING"):
        out = mixup(data)

    assert out["img"] == ["only-one"]
    assert out["labels"] == [7]
    assert "Batch size should be even when using Mixup" in caplog.text
