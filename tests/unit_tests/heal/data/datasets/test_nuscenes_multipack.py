# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import numpy as np
import pytest

from heal.data.datasets.merge_dataset import MergeDataset


class _ListDataset:
    """Minimal in-memory dataset returning a (copied) dict per index."""

    def __init__(self, items, meta=None):
        self.items = items
        self.meta = meta

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return dict(self.items[idx])

    def get_meta(self):
        return self.meta


def _base_items(tokens):
    # Base dataset nests sample_token under ``meta`` (like NuscenesBevDataset).
    return [
        {"img": f"img-{i}", "meta": {"sample_token": tok}}
        for i, tok in enumerate(tokens)
    ]


def _occ_items(tokens):
    # Occ dataset exposes sample_token at the top level.
    return [
        {"sample_token": tok, "gt_occ_info": {"voxel": i}}
        for i, tok in enumerate(tokens)
    ]


def test_merge_joins_children_with_meta_fallback():
    tokens = ["tokA", "tokB", "tokC"]
    ds = MergeDataset(
        datasets=[
            _ListDataset(_base_items(tokens)),
            _ListDataset(_occ_items(tokens)),
        ],
        join_key="sample_token",
    )
    assert len(ds) == 3
    merged = ds[1]
    # base keys + occ keys both present
    assert merged["img"] == "img-1"
    assert merged["meta"]["sample_token"] == "tokB"
    assert merged["gt_occ_info"]["voxel"] == 1


def test_merge_runs_transforms_once():
    tokens = ["t0", "t1"]
    seen = {}

    def _transform(data):
        seen["called"] = seen.get("called", 0) + 1
        data["extra"] = "done"
        return data

    ds = MergeDataset(
        datasets=[
            _ListDataset(_base_items(tokens)),
            _ListDataset(_occ_items(tokens)),
        ],
        transforms=_transform,
    )
    out = ds[0]
    assert out["extra"] == "done"
    assert seen["called"] == 1


def test_merge_detects_join_key_mismatch():
    tokens = ["tokA", "tokB", "tokC"]
    occ = _occ_items(tokens)
    occ[2]["sample_token"] = "WRONG"
    ds = MergeDataset(
        datasets=[_ListDataset(_base_items(tokens)), _ListDataset(occ)],
        join_key="sample_token",
    )
    ds[0]  # aligned index is fine
    with pytest.raises(RuntimeError, match="sample_token.*mismatch"):
        ds[2]


def test_merge_delegates_get_meta_to_first_child_with_meta():
    tokens = ["tokA", "tokB"]
    scene_info = {"scene-0": 2}
    ds = MergeDataset(
        datasets=[
            _ListDataset(_base_items(tokens), meta={"scene_info": scene_info}),
            _ListDataset(_occ_items(tokens)),
        ],
        join_key="sample_token",
    )
    assert ds.get_meta() == {"scene_info": scene_info}


def test_merge_asserts_equal_length():
    with pytest.raises(AssertionError, match="equal length"):
        MergeDataset(
            datasets=[
                _ListDataset(_base_items(["a", "b"])),
                _ListDataset(_occ_items(["a"])),
            ]
        )


# --- NuscenesOccDataset: reads a real occ LMDB (occ + camera geometry) ---


def _write_occ_lmdb(path, tokens):
    import msgpack
    from hat_engine.utils.pack_type import PackTypeMapper

    from heal.data.datasets.nuscenes.nuscenes_occ import OCC_SHAPE

    voxel = np.zeros(OCC_SHAPE, dtype=np.uint8)
    pack = PackTypeMapper["lmdb"](str(path), writable=True)
    for idx, tok in enumerate(tokens):
        sample = {
            "sample_token": tok,
            "scene_token": "scene-0",
            "voxel_semantics": (voxel + idx).tobytes(),
            "mask_lidar": voxel.tobytes(),
            "mask_camera": voxel.tobytes(),
            "ego2img": [np.eye(4).tolist() for _ in range(6)],
            "cam2ego": [np.eye(4).tolist() for _ in range(6)],
        }
        pack.write(idx, msgpack.packb(sample))
    pack.write("__len__", "{}".format(len(tokens)).encode("ascii"))
    pack.close()


def test_occ_dataset_decodes_occ_and_geometry(tmp_path):
    pytest.importorskip("lmdb")
    from heal.data.datasets.nuscenes.nuscenes_occ import NuscenesOccDataset

    occ_dir = tmp_path / "occ_lmdb"
    _write_occ_lmdb(occ_dir, ["tokA", "tokB", "tokC"])

    ds = NuscenesOccDataset(data_path=str(occ_dir))
    assert len(ds) == 3

    sample = ds[1]
    assert sample["sample_token"] == "tokB"
    assert sample["scene_token"] == "scene-0"
    occ = sample["gt_occ_info"]
    assert occ["voxel_semantics"].shape == (200, 200, 16)
    assert occ["voxel_semantics"][0, 0, 0] == 1
    assert "mask_lidar" in occ and "mask_camera" in occ
    # camera geometry: 6 cameras, 4x4 matrices
    assert len(sample["ego2img"]) == 6
    assert sample["ego2img"][0].shape == (4, 4)
    assert len(sample["cam2ego"]) == 6


def test_occ_dataset_merges_with_a_base_dataset(tmp_path):
    pytest.importorskip("lmdb")
    from heal.data.datasets.nuscenes.nuscenes_occ import NuscenesOccDataset

    tokens = ["tokA", "tokB"]
    occ_dir = tmp_path / "occ_lmdb"
    _write_occ_lmdb(occ_dir, tokens)

    ds = MergeDataset(
        datasets=[
            _ListDataset(_base_items(tokens)),
            NuscenesOccDataset(data_path=str(occ_dir)),
        ],
        join_key="sample_token",
    )
    merged = ds[0]
    assert merged["img"] == "img-0"
    assert merged["gt_occ_info"]["voxel_semantics"][0, 0, 0] == 0
    assert len(merged["ego2img"]) == 6


def test_occ_geometry_helpers_match_old_sampler_math():
    pytest.importorskip("nuscenes")
    from heal.data.datasets.nuscenes.nuscenes_occ import (
        _cam2ego_matrix,
        _ego2img_matrix,
    )

    intrinsic = [[1260.0, 0.0, 800.0], [0.0, 1260.0, 450.0], [0.0, 0.0, 1.0]]
    s2e_t = [1.0, 0.0, 1.5]
    s2e_r = [0.5, -0.5, 0.5, -0.5]  # a valid unit quaternion

    cam2ego = _cam2ego_matrix(s2e_t, s2e_r)
    assert cam2ego.shape == (4, 4)
    np.testing.assert_allclose(cam2ego[:3, 3], s2e_t)
    assert cam2ego[3, 3] == 1.0

    ego2img = _ego2img_matrix(intrinsic, s2e_t, s2e_r)
    assert ego2img.shape == (4, 4)


# --- NuscenesMapGtDataset: precomputed map vector GT ---


def _write_map_gt_lmdb(path, tokens):
    import msgpack
    from hat_engine.utils.pack_type import PackTypeMapper

    pack = PackTypeMapper["lmdb"](str(path), writable=True)
    for idx, tok in enumerate(tokens):
        sample = {
            "sample_token": tok,
            "scene_token": "scene-0",
            "gt_labels_map": [0, 1],
            "gt_instance_coords": [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ],
        }
        pack.write(idx, msgpack.packb(sample))
    meta = {
        "scene_info": {"scene-0": len(tokens)},
        "scene_len": 1,
        "data_len": len(tokens),
        "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        "map_classes": ["divider", "ped_crossing", "boundary"],
        "fixed_ptsnum_per_line": 20,
        "bev_size": [100, 50],
        "padding_value": -10000,
        "use_lidar_gt": True,
    }
    pack.write("__len__", "{}".format(len(tokens)).encode("ascii"))
    pack.write("__meta__", msgpack.packb(meta))
    pack.close()


def test_map_gt_dataset_decodes_instances(tmp_path):
    pytest.importorskip("lmdb")
    from heal.data.datasets.nuscenes.nuscenes_map_gt import (
        NuscenesMapGtDataset,
    )

    map_dir = tmp_path / "map_lmdb"
    _write_map_gt_lmdb(map_dir, ["tokA", "tokB"])

    ds = NuscenesMapGtDataset(
        data_path=str(map_dir),
        pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        fixed_ptsnum_per_line=20,
    )
    assert len(ds) == 2
    sample = ds[0]
    assert sample["sample_token"] == "tokA"
    assert sample["gt_labels_map"] == [0, 1]
    assert len(sample["gt_instances"].instance_list) == 2
    assert sample["gt_instances"].fixed_num_sampled_points.shape[0] == 2


def test_map_gt_build_lidar_instance_lines():
    from heal.data.datasets.nuscenes.nuscenes_map_gt import (
        build_lidar_instance_lines,
    )

    instances = build_lidar_instance_lines(
        [[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]],
        patch_size=(60.0, 30.0),
        fixed_num=20,
    )
    assert instances.fixed_num_sampled_points.shape == (1, 20, 2)
