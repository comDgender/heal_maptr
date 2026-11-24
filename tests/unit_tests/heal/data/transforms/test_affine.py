"""Tests for heal.data.transforms.affine."""

from heal.data.transforms.affine import AffineMat2DGenerator


def test_resize_matrix_without_pad():
    m = AffineMat2DGenerator.resize(
        (100, 200), (50, 100), pad_to_keep_ratio=False
    )
    assert m.shape == (3, 3)
    assert abs(m[0, 0] - 0.5) < 1e-6
    assert abs(m[1, 1] - 0.5) < 1e-6
