# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import glob
import logging
import os
from typing import List, Optional

import cv2  # noqa: F401  (used in to_rgb branch)
import msgpack
import numpy as np
import torch.utils.data as data
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.data.utils import decode_img
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from PIL import Image

__all__ = ["Mot17Dataset", "Mot17Packer", "Mot17FromImage"]

logger = logging.getLogger(__name__)


def encode_hook(obj):
    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 3:
            img_bytes = cv2.imencode(".png", obj)[1]
            obj = np.asarray(img_bytes).astype(np.uint8).tobytes()
        elif len(obj.shape) == 2:
            obj = obj.tobytes()
        elif len(obj.shape) == 1:
            obj = obj.tobytes()
    return obj


def _period_index_for_epoch(epoch: int, sampler_steps: List[int]) -> int:
    """Map training epoch to the curriculum period index."""
    period = 0
    for step_i, step in enumerate(sampler_steps):
        if epoch >= step:
            period = step_i + 1
    return period


def _sample_frame_span(
    start_idx: int,
    num_frames: int,
    sample_mode: str,
    sample_interval: int,
):
    """Build ``(start, end_exclusive, interval)`` for a temporal clip."""
    allowed = ("fixed_interval", "random_interval")
    if sample_mode not in allowed:
        raise AssertionError(f"invalid sample mode: {sample_mode}")
    if sample_mode == "fixed_interval":
        interval = sample_interval
    else:
        interval = int(np.random.randint(1, sample_interval + 1))
    end = start_idx + (num_frames - 1) * interval + 1
    return start_idx, end, interval


def _norm_cxcywh_to_pixel_xyxy(
    labels_cxcywh: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Keep class/id cols; map normalized cxcywh (cols 2-5) to pixel xyxy."""
    converted = labels_cxcywh.copy()
    cx = labels_cxcywh[:, 2]
    cy = labels_cxcywh[:, 3]
    box_w = labels_cxcywh[:, 4]
    box_h = labels_cxcywh[:, 5]
    converted[:, 2] = width * (cx - box_w * 0.5)
    converted[:, 3] = height * (cy - box_h * 0.5)
    converted[:, 4] = width * (cx + box_w * 0.5)
    converted[:, 5] = height * (cy + box_h * 0.5)
    return converted


def _parse_seqinfo(seqinfo_path: str) -> dict:
    """Parse a MOT17 ``seqinfo.ini`` for the keys the packer needs.

    Returns ``{"seq_length", "im_width", "im_height"}`` as ints. Mirrors the
    ad-hoc string slicing in HAT ``gen_mot_data.py:44-58`` but is a little
    more robust (falls back to a key=value scan if the ini ordering changes).
    """
    with open(seqinfo_path, "r") as f:
        text = f.read()
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("["):
            continue
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip()
    return {
        "seq_length": int(fields["seqLength"]),
        "im_width": int(fields["imWidth"]),
        "im_height": int(fields["imHeight"]),
    }


def _process_mot17_seq(
    seq_dir: str,
    seq_name: str,
    split_name: str,
    tid_state: dict,
) -> dict:
    """Build the in-memory label/gt view for one MOT17-SDP sequence.

    This merges HAT ``gen_mot_data.py`` (``split_half_data``) into the packer:
    the official MOT17 ``train`` sequences are split in half — ``train`` split
    keeps the first half of frames, ``test`` split keeps the second half with
    frame ids renumbered to start at 1 — and ``labels_with_ids``-style rows
    are computed on the fly (no intermediate files are written).

    Args:
        seq_dir: path to ``<root>/train/<seq>`` (gt lives only under train/).
        seq_name: the sequence directory name (e.g. ``MOT17-02-SDP``).
        split_name: ``"train"`` or ``"test"`` (both halves come from the
            gt-bearing ``train/`` sequences, since the official ``test/``
            split has no public gt).
        tid_state: mutable dict shared across sequences within one split,
            carrying ``{"curr": int, "last": int}`` for the cumulative
            track-id remap (mirrors HAT's ``tid_train_curr``/``tid_train_last``
            that are initialized once *outside* the sequence loop and never
            reset per sequence).

    Returns:
        A dict with:
        - ``"images"``: sorted list of absolute image paths for this half.
        - ``"frame_labels"``: ``{renumbered_frame_id: np.ndarray of shape
          (n, 6)}`` with rows ``[class=0, tid_remapped, cx_n, cy_n, w_n, h_n]``
          (only ``mark==1 AND label==1`` rows, matching HAT's filter).
        - ``"gt_lines"``: list of MOTChallenge ``gt.txt`` lines
          (``fid,tid,x,y,w,h,mark,label,sss``) for this half, with ``fid``
          renumbered for the test split (used to write ``test_gt/``).
    """
    seqinfo = _parse_seqinfo(os.path.join(seq_dir, "seqinfo.ini"))
    seq_length = seqinfo["seq_length"]
    im_width = seqinfo["im_width"]
    im_height = seqinfo["im_height"]
    half = seq_length // 2

    images = sorted(glob.glob(os.path.join(seq_dir, "img1", "*.jpg")))
    if split_name == "train":
        seq_images = images[:half]
        frame_offset = 0  # train frames keep original 1..half numbering
    else:
        seq_images = images[half:]
        frame_offset = half  # test frames renumbered: fid - half (-> 1..M)

    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    gt = np.loadtxt(gt_path, dtype=np.float64, delimiter=",").reshape(-1, 9)

    frame_labels = {}
    gt_lines = []
    for row in gt:
        fid, tid, x, y, w, h, mark, label, sss = row
        frame = int(fid)
        in_train_half = frame <= half
        in_test_half = frame > half
        if split_name == "train" and not in_train_half:
            continue
        if split_name == "test" and not in_test_half:
            continue
        new_frame = frame - frame_offset
        gt_lines.append(
            "{},{},{},{},{},{},{},{},{}\n".format(
                new_frame, tid, x, y, w, h, mark, label, sss
            )
        )
        # Mirrors HAT: only mark==1 AND label==1 rows get labels_with_ids.
        if int(mark) == 0 or int(label) != 1:
            continue
        # Cumulative track-id remap, NOT reset per sequence (HAT semantics).
        if int(tid) != tid_state["last"]:
            tid_state["curr"] += 1
            tid_state["last"] = int(tid)
        # bbox top-left -> center, then normalize by image size.
        # Round-trip through "%.6f" to match HAT ``gen_mot_data.py`` exactly:
        # HAT writes labels_with_ids with ``"0 {:d} {:.6f} ...{:.6f}\\n"``,
        # then the legacy packer reloads them as float32 via ``np.loadtxt``.
        # That textual 6-decimal quantization (not a plain float64->float32
        # cast) is what fixes the label values, so reproducing it here keeps
        # the packed lmdb labels byte-identical to the legacy two-step output.
        cx = float("%.6f" % ((x + w / 2.0) / im_width))
        cy = float("%.6f" % ((y + h / 2.0) / im_height))
        wn = float("%.6f" % (w / im_width))
        hn = float("%.6f" % (h / im_height))
        label_row = np.array(
            [0, tid_state["curr"], cx, cy, wn, hn], dtype=np.float32
        )
        frame_labels.setdefault(new_frame, []).append(label_row)

    # Stack per-frame label arrays for fast lookup in _pre_single_frame.
    stacked_frame_labels = {
        f: np.stack(rows, axis=0) for f, rows in frame_labels.items()
    }
    return {
        "images": seq_images,
        "frame_labels": stacked_frame_labels,
        "gt_lines": gt_lines,
    }


def decode_hook(obj):
    def _decode_bytes(obj):
        if isinstance(obj, bytes):
            obj = obj.decode("utf-8")
        return obj

    new_obj = {}
    for k, v in obj.items():
        k = _decode_bytes(k)

        if k == "img":
            v = decode_img(v, iscolor=cv2.IMREAD_COLOR)
        elif k == "gt_bboxes":
            v = np.array(
                np.frombuffer(v, dtype=np.float32).reshape(-1, 4)
            ).astype(np.float32)
        elif k == "gt_classes":
            v = np.array(np.frombuffer(v, dtype=np.float32)).astype(np.float32)
        elif k == "gt_ids":
            v = np.array(np.frombuffer(v, dtype=np.float32)).astype(np.float32)
        else:
            v = _decode_bytes(v)
        new_obj[k] = v
    return new_obj


@OBJECT_REGISTRY.register
class Mot17Dataset(data.Dataset):  # noqa: D205,D400
    """
    Mot17Dataset provides the method of reading Mot17 data
    from target pack type.

    Args:
        data_path: The path of packed file.
        sampler_lengths: The length of the sequence data.
        sample_mode: The sampling mode,
            only support 'fixed_interval' or 'random_interval'.
        sample_interval: The sampling interval,
            if sample_mode is 'random_interval',
            randomly select from [1, sample_interval].
        sampler_steps: Sequence length changes according to the epoch.
        transforms: Transfroms of data before using.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
        self,
        data_path: str,
        sampler_lengths: List[int] = (1,),
        sample_mode: str = "fixed_interval",
        sample_interval: int = 10,
        sampler_steps: List[int] = None,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        to_rgb: bool = True,
    ):

        self.data_path = data_path
        self.transforms = transforms
        self.to_rgb = to_rgb

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.lmdb_samples = self.pack_file.get_keys()
        self.sampler_lengths = sampler_lengths
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.sampler_steps = sampler_steps
        self.num_frames_per_batch = max(self.sampler_lengths)

        self.num_samples = (
            len(self.lmdb_samples)
            - (self.num_frames_per_batch - 1) * self.sample_interval
        )

        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Curriculum: clip length grows across ``sampler_steps`` epochs.
            if len(self.sampler_lengths) == 0:
                raise AssertionError("sampler_lengths must be non-empty")
            if len(self.sampler_lengths) != len(self.sampler_steps) + 1:
                raise AssertionError(
                    "len(sampler_lengths) must equal len(sampler_steps) + 1"
                )
            for prev, nxt in zip(self.sampler_steps, self.sampler_steps[1:]):
                if prev >= nxt:
                    raise AssertionError(
                        "sampler_steps must be strictly increasing"
                    )
            self.num_samples = (
                len(self.lmdb_samples)
                - (self.num_frames_per_batch - 1) * self.sample_interval
            )
            self.period_idx = 0
            self.num_frames_per_batch = self.sampler_lengths[0]
            self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if not self.sampler_steps:
            return
        self.period_idx = _period_index_for_epoch(epoch, self.sampler_steps)
        logger.info(
            "set epoch: epoch %s period_idx=%s", epoch, self.period_idx
        )
        self.num_frames_per_batch = self.sampler_lengths[self.period_idx]

    def _get_sample_range(self, start_idx):
        return _sample_frame_span(
            start_idx,
            self.num_frames_per_batch,
            self.sample_mode,
            self.sample_interval,
        )

    def pre_continuous_frames(self, start, end, interval=1):
        frame_data_list = []
        for i in range(start, end, interval):
            single_data = self._pre_single_frame(i)
            frame_data_list.append(single_data)
        data_seq = {"frame_data_list": frame_data_list}
        return data_seq

    def _pre_single_frame(self, idx: int):
        raw_data = self.pack_file.read(self.lmdb_samples[idx])
        data = msgpack.unpackb(raw_data, object_hook=decode_hook, raw=True)

        data["img_shape"] = data["img"].shape[0:2]
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        data["seq_name"] = data["img_name"].split("/")[-3]
        if not self.to_rgb:
            cv2.cvtColor(data["img"], cv2.COLOR_RGB2BGR, data["img"])
            data["color_space"] = "bgr"

        data["ori_img"] = data["img"].copy()
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        data_seq = self.pre_continuous_frames(
            sample_start, sample_end, sample_interval
        )

        if self.transforms is not None:
            data_seq = self.transforms(data_seq)
        data_seq["frame_length"] = len(data_seq["frame_data_list"])

        return data_seq


@OBJECT_REGISTRY.register
class Mot17Packer(Packer):  # noqa: D205,D400
    """
    Mot17Packer is used for converting MOT17 dataset
    to target DataType format.

    Args:
        src_data_dir: The dir of original mot17 data.
        target_data_dir: Path for packed file.
        split_name: Split name of data, must be train or test.
        num_workers: Num workers for reading data using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        assert split_name in [
            "train",
            "test",
        ], "split_name must be one of train and test."
        # ``src_data_dir`` is the MOT17 root containing ``train/`` and
        # ``test/`` (official layout). Official ``test/`` has no public gt, so
        # both halves (train/test split) are produced from the gt-bearing
        # ``train/*SDP`` sequences inside ``Mot17FromImage`` (no intermediate
        # ``split_data``/``labels_with_ids`` files are written).
        if not os.path.isdir(os.path.join(src_data_dir, "train")):
            raise FileNotFoundError(
                "Expected {} to exist. --src-data-dir should point to the "
                "MOT17 root containing train/ and test/.".format(
                    os.path.join(src_data_dir, "train")
                )
            )

        self.dataset = Mot17FromImage(
            data_path=src_data_dir,
            sampler_lengths=[1],
            sample_mode="fixed_interval",
            sample_interval=1,
            split_name=split_name,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(Mot17Packer, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )
        self.pop_keys = ["img_shape", "layout", "color_space", "seq_name"]

        # For test split: write renumbered motchallenge-format gt.txt files
        # alongside the packed lmdb so MotMetric evaluation can find
        # ``{mot17_root}/test_gt/{seq}/gt/gt.txt``. motmetrics only reads the
        # raw ``gt.txt`` text format (not lmdb), so gt is written as text.
        # The frame ids are renumbered to start at 1 per sequence (matching
        # the MOTR predictor's per-seq ``seq_frame_id`` counter), and the gt
        # lines are generated in memory by ``Mot17FromImage`` from the
        # original ``train/<seq>/gt/gt.txt`` (second half of each sequence).
        # Note: ``target_data_dir`` here is the lmdb dir (e.g.
        # ``.../mot17/test_lmdb``) as passed by mot17_packer.py; the gt root
        # is its sibling, so use ``dirname`` to go up one level.
        if split_name == "test":
            gt_target_root = os.path.join(
                os.path.dirname(target_data_dir.rstrip(os.sep)),
                "test_gt",
            )
            os.makedirs(gt_target_root, exist_ok=True)
            for seq in self.dataset.seqs:
                dst_gt_dir = os.path.join(gt_target_root, seq, "gt")
                os.makedirs(dst_gt_dir, exist_ok=True)
                gt_lines = self.dataset.gt_lines.get(seq, [])
                dst_gt_path = os.path.join(dst_gt_dir, "gt.txt")
                if os.path.isdir(dst_gt_dir):
                    # remove any stale gt.txt from a previous run
                    stale = os.path.join(dst_gt_dir, "gt.txt")
                    if os.path.exists(stale):
                        os.remove(stale)
                with open(dst_gt_path, "w") as f:
                    f.writelines(gt_lines)
            logger.info("Wrote renumbered test gt/ to %s", gt_target_root)

    def pack_data(self, idx):
        data = self.dataset[idx]
        data_pack = data["frame_data_list"][0]

        for pop_key in self.pop_keys:
            data_pack.pop(pop_key)

        return msgpack.packb(data_pack, default=encode_hook)


@OBJECT_REGISTRY.register
class Mot17FromImage(data.Dataset):
    """Mot17FromImage which gets img data and gt from the data_path.

    Args:
        data_path: The dir of mot17 data.
        sampler_lengths: The length of the sequence data.
        sample_mode: The sampling mode,
            only support 'fixed_interval' or 'random_interval'.
        sample_interval: The sampling interval,
            if sample_mode is 'random_interval',
            randomly select from [1, sample_interval].
        sampler_steps: Sequence length changes according to the epoch.
        transforms: List of transform.
        to_rgb: Whether to convert to `rgb` color_space.
    """

    def __init__(
        self,
        data_path: str,
        sampler_lengths: List[int] = (1,),
        sample_mode: str = "fixed_interval",
        sample_interval: int = 10,
        sampler_steps: List[int] = None,
        transforms: Optional[List] = None,
        to_rgb: bool = True,
        split_name: str = "train",
    ):
        # ``data_path`` is the MOT17 root containing ``train/``. Only the SDP
        # detector sequences are used (matches HAT ``gen_mot_data.py``).
        # ``self.data_path`` is set to the ``train/`` subdir so that the
        # ``img_name = img_path[len(self.data_path)+1:]`` slice in
        # ``_pre_single_frame`` yields ``MOT17-xx-SDP/img1/....jpg`` (without
        # a ``train/`` prefix), matching the legacy HAT packer's img_name
        # (which used ``data_path = os.path.join(src_data_dir, "train")``).
        train_root = os.path.join(data_path, "train")
        if not os.path.isdir(train_root):
            raise FileNotFoundError(
                "Expected {} to exist. --src-data-dir should point to the "
                "MOT17 root containing train/ and test/.".format(train_root)
            )
        self.data_path = train_root
        self.sampler_lengths = sampler_lengths
        self.sample_mode = sample_mode
        self.sample_interval = sample_interval
        self.sampler_steps = sampler_steps
        self.transforms = transforms
        self.to_rgb = to_rgb
        # Official MOT17 has gt only under ``train/``; both halves (train /
        # test split) come from the gt-bearing ``train/*SDP`` sequences.
        assert split_name in ["train", "test"]
        self.split_name = split_name

        self.seqs = [
            s
            for s in sorted(os.listdir(train_root))
            if s.endswith("SDP") and os.path.isdir(os.path.join(train_root, s))
        ]

        # Cumulative track-id remap state, shared across all sequences within
        # this split (NOT reset per sequence) — mirrors HAT ``gen_mot_data``.
        tid_state = {"curr": 0, "last": -1}

        self.img_files = []
        # ``self.frame_labels[seq_name]`` maps renumbered-frame-id -> labels
        # array of shape (n, 6): [class=0, tid, cx_n, cy_n, w_n, h_n].
        self.frame_labels = {}
        # ``self.gt_lines[seq_name]`` holds the renumbered motchallenge gt.txt
        # lines for the test split (used by Mot17Packer to write test_gt/).
        self.gt_lines = {}
        # ``self.seq_frames[seq_name]`` lists the renumbered frame id for each
        # packed image of that sequence (1-based), so _pre_single_frame can
        # recover the (seq, frame) label key from a flat image index.
        self.seq_frames = {}
        self._seq_index = []  # parallel to self.img_files: seq name per image
        self._img_frame_ids = []  # parallel to self.img_files: renumbered fid

        for seq in self.seqs:
            seq_dir = os.path.join(train_root, seq)
            result = _process_mot17_seq(
                seq_dir=seq_dir,
                seq_name=seq,
                split_name=split_name,
                tid_state=tid_state,
            )
            seq_images = result["images"]
            self.frame_labels[seq] = result["frame_labels"]
            self.gt_lines[seq] = result["gt_lines"]
            # Image filenames are zero-padded 1-based frame ids, so the
            # renumbered frame id of each image is its index within the half
            # plus one (train: 1..half; test: 1..(seq_length-half)).
            seq_frame_ids = [i + 1 for i in range(len(seq_images))]
            self.seq_frames[seq] = seq_frame_ids
            self.img_files.extend(seq_images)
            self._seq_index.extend([seq] * len(seq_images))
            self._img_frame_ids.extend(seq_frame_ids)

        self.video_dict = {}
        self._register_videos()

        self.num_frames_per_batch = max(self.sampler_lengths)
        self.num_samples = (
            len(self.img_files)
            - (self.num_frames_per_batch - 1) * self.sample_interval
        )

        if self.sampler_steps is not None and len(self.sampler_steps) > 0:
            # Enable sampling length adjustment.
            assert len(self.sampler_lengths) > 0
            assert len(self.sampler_lengths) == len(self.sampler_steps) + 1
            for i in range(len(self.sampler_steps) - 1):
                assert self.sampler_steps[i] < self.sampler_steps[i + 1]
            self.num_samples = (
                len(self.img_files)
                - (self.sampler_lengths[-1] - 1) * self.sample_interval
            )
            self.period_idx = 0
            self.num_frames_per_batch = self.sampler_lengths[0]
            self.current_epoch = 0

    def __len__(self):
        return self.num_samples

    def _register_videos(self):
        for seq in self.seqs:
            if seq not in self.video_dict:
                logger.info(
                    "register {}-th video: {} ".format(
                        len(self.video_dict) + 1, seq
                    )
                )
                self.video_dict[seq] = len(self.video_dict)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if not self.sampler_steps:
            return
        self.period_idx = _period_index_for_epoch(epoch, self.sampler_steps)
        logger.info(
            "set epoch: epoch %s period_idx=%s", epoch, self.period_idx
        )
        self.num_frames_per_batch = self.sampler_lengths[self.period_idx]

    def _get_sample_range(self, start_idx):
        return _sample_frame_span(
            start_idx,
            self.num_frames_per_batch,
            self.sample_mode,
            self.sample_interval,
        )

    def pre_continuous_frames(self, start, end, interval=1):
        frame_data_list = []
        for i in range(start, end, interval):
            single_data = self._pre_single_frame(i)
            frame_data_list.append(single_data)
        data_seq = {"frame_data_list": frame_data_list}
        return data_seq

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        seq = self._seq_index[idx]
        frame_id = self._img_frame_ids[idx]
        img = Image.open(img_path)
        single_data = {}
        w, h = img.size
        if w <= 0 or h <= 0:
            raise AssertionError(
                "invalid image {} with shape {} {}".format(img_path, w, h)
            )

        # In-memory labels (no labels_with_ids files on disk). Rows are
        # normalized cxcywh [class=0, tid, cx_n, cy_n, w_n, h_n].
        raw_labels = self.frame_labels.get(seq, {}).get(frame_id)
        if raw_labels is None or len(raw_labels) == 0:
            labels = np.zeros((0, 6), dtype=np.float32)
        else:
            labels = _norm_cxcywh_to_pixel_xyxy(raw_labels, w, h)

        video_name = seq

        img_name = img_path[len(self.data_path) + 1 :]
        obj_idx_offset = self.video_dict[video_name] * 1000000

        single_data["img"] = np.array(img)
        single_data["gt_bboxes"] = []
        single_data["gt_classes"] = []
        single_data["gt_ids"] = []
        single_data["image_id"] = idx
        single_data["img_shape"] = (h, w)
        single_data["color_space"] = "rgb"
        single_data["layout"] = "hwc"
        single_data["img_name"] = img_name
        single_data["seq_name"] = img_name.split("/")[-3]

        for label in labels:
            single_data["gt_bboxes"].append(label[2:6].tolist())
            single_data["gt_classes"].append(label[0])
            obj_id = label[1] + obj_idx_offset if label[1] >= 0 else label[1]
            single_data["gt_ids"].append(obj_id)  # relative id

        if len(single_data["gt_bboxes"]) > 0:
            single_data["gt_bboxes"] = np.array(
                single_data["gt_bboxes"], dtype=np.float32
            )
            single_data["gt_classes"] = np.array(
                single_data["gt_classes"], dtype=np.float32
            )
            single_data["gt_ids"] = np.array(
                single_data["gt_ids"], dtype=np.float32
            )
        else:
            single_data["gt_bboxes"] = np.zeros((0, 4), dtype=np.float32)
            single_data["gt_classes"] = np.zeros((0, 1), dtype=np.float32)
            single_data["gt_ids"] = np.zeros((0, 1), dtype=np.float32)

        if not self.to_rgb:
            cv2.cvtColor(
                single_data["img"], cv2.COLOR_RGB2BGR, single_data["img"]
            )
            single_data["color_space"] = "bgr"

        return single_data

    def __getitem__(self, idx: int):
        sample_start, sample_end, sample_interval = self._get_sample_range(idx)
        data_seq = self.pre_continuous_frames(
            sample_start, sample_end, sample_interval
        )
        data_seq["frame_length"] = len(data_seq["frame_data_list"])
        if self.transforms is not None:
            data_seq = self.transforms(data_seq)
        return data_seq
