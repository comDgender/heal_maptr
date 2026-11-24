"""Tests for heal.data.collates.qc_collate."""


def test_qc_collate_importable():
    from heal.data.collates.qc_collate import collate_qc_argoverse2

    assert callable(collate_qc_argoverse2)
