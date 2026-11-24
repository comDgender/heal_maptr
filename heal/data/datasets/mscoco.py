# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
from typing import Any, Callable, List, Optional

import cv2
import msgpack
import numpy as np
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.data.utils import decode_img
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from hat_engine.utils.package_helper import require_packages
from PIL import Image
from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

try:
    from torchvision.datasets.vision import VisionDataset
except ImportError:
    VisionDataset = object

__all__ = [
    "Coco",
    "COCO_CLASS_NAMES",
    "CocoDetection",
    "CocoDetectionPacker",
    "CocoFromImage",
]

_COCO_VOCAB_SEGMENTS = [
    "person;bicycle;car;motorcycle;airplane;bus;train;truck;boat",
    "traffic light;fire hydrant;stop sign;parking meter;bench",
    "bird;cat;dog;horse;sheep;cow;elephant;bear;zebra;giraffe",
    "backpack;umbrella;handbag;tie;suitcase",
    "frisbee;skis;snowboard;sports ball;kite",
    "baseball bat;baseball glove;skateboard;surfboard;tennis racket",
    "bottle;wine glass;cup;fork;knife;spoon;bowl",
    "banana;apple;sandwich;orange;broccoli;carrot;hot dog;pizza;donut;cake",
    "chair;couch;potted plant;bed;dining table;toilet",
    "tv;laptop;mouse;remote;keyboard;cell phone",
    "microwave;oven;toaster;sink;refrigerator",
    "book;clock;vase;scissors;teddy bear;hair drier;toothbrush",
]


def _tokens_from_vocab_segment(segment: str) -> tuple:
    return tuple(
        token.strip() for token in segment.split(";") if token.strip()
    )


def _flatten_vocab_segments(segments) -> List[str]:
    labels: List[str] = []
    for segment in segments:
        labels.extend(_tokens_from_vocab_segment(segment))
    return labels


_COCO_NAME_GROUPS = tuple(
    _tokens_from_vocab_segment(segment) for segment in _COCO_VOCAB_SEGMENTS
)
COCO_CLASS_NAMES = _flatten_vocab_segments(_COCO_VOCAB_SEGMENTS)

COCO_LABLE_TO_LABLE = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


@OBJECT_REGISTRY.register
class Coco(Dataset):
    """Coco dataset from packed lmdb."""

    def __init__(
        self,
        data_path: str,
        transforms: Optional[List] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.transforms = transforms
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)
        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        raw_data = self.pack_file.read(self.samples[item])
        raw_data = msgpack.unpackb(raw_data, raw=True)
        image_id = np.frombuffer(raw_data[:8], dtype=np.int64)
        num_bboxes = np.frombuffer(raw_data[8:9], dtype=np.uint8)[0]
        label = np.frombuffer(
            raw_data[9 : 9 + num_bboxes * 5 * 8], dtype=np.float64
        ).reshape((num_bboxes, 5))
        image_data = raw_data[9 + num_bboxes * 5 * 8 :]
        image = decode_img(image_data, iscolor=cv2.IMREAD_COLOR)
        data = {
            "img": image,
            "ori_img": image,
            "gt_bboxes": label[:, 0:4],
            "gt_classes": label[:, 4],
            "img_id": image_id,
            "img_name": "%012d.jpg" % int(image_id.flat[0]),
            "layout": "hwc",
            "img_shape": image.shape[0:2],
            "color_space": "rgb",
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return data


class CocoDetection(VisionDataset):
    """Coco Detection Dataset for packing."""

    @require_packages("torchvision", "pycocotools")
    def __init__(
        self,
        root: str,
        annFile: str,
        num_classes: Optional[int] = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        # Class count is read by the packer path; default to COCO-80.
        self.num_classes = 80 if num_classes is None else num_classes
        super().__init__(root, transforms, transform, target_transform)

        api = COCO(annFile)
        self.coco = api
        # Deterministic index order over image ids.
        self.ids = sorted(api.imgs)

    def _load_image(self, id):
        meta = self.coco.loadImgs(ids=[id])[0]
        img_path = os.path.join(self.root, str(meta["file_name"]))
        return Image.open(img_path).convert("RGB")

    def _load_target(self, id):
        ann_ids = self.coco.getAnnIds(imgIds=id, iscrowd=False)
        return self.coco.loadAnns(ann_ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        label = np.zeros((0, 5))
        if len(target) == 0:
            print("image {} has no bounding boxes".format(image_id))
            return image, label, image_id
        for _, tar in enumerate(target):
            if tar["bbox"][2] < 1 or tar["bbox"][3] < 1:
                continue
            anno = np.zeros((1, 5))
            anno[0, :4] = tar["bbox"]
            if self.num_classes == 91:
                anno[0, 4] = tar["category_id"]
            else:
                anno[0, 4] = COCO_LABLE_TO_LABLE[tar["category_id"]]
            label = np.append(label, anno, axis=0)
        label[:, 2] = label[:, 0] + label[:, 2]
        label[:, 3] = label[:, 1] + label[:, 3]
        return image, label, image_id

    def __len__(self) -> int:
        return len(self.ids)


def transforms(image, target):
    return np.array(image), target


class CocoDetectionPacker(Packer):
    """Pack coco dataset to target format (lmdb, etc.)."""

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        num_classes: int = 80,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        root = os.path.join(src_data_dir, "{}2017".format(split_name))
        annFile = os.path.join(
            src_data_dir,
            "annotations",
            "instances_{}2017.json".format(split_name),
        )
        self.dataset = CocoDetection(
            root=root,
            annFile=annFile,
            num_classes=num_classes,
            transforms=transforms,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super(CocoDetectionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        datas = self.dataset[idx]
        image, label, image_id = datas
        image_id = np.asarray(image_id, dtype=np.int64).tobytes()
        num_bboxes = label.shape[0]
        num_bboxes_data = np.asarray(num_bboxes, dtype=np.uint8).tobytes()
        image = cv2.imencode(".jpg", image)[1].tobytes()
        label = np.asarray(label, dtype=np.float64).tobytes()
        return msgpack.packb(image_id + num_bboxes_data + label + image)


class _CocoDetectionBase(Dataset):
    """Base for CocoFromImage (from image dir + annFile)."""

    @require_packages("pycocotools")
    def __init__(
        self,
        root: str,
        annFile: str,
        num_classes: Optional[int] = 80,
        transforms: Optional[Any] = None,
    ):
        super(_CocoDetectionBase, self).__init__()
        self.root = root
        self.num_classes = num_classes
        self.transforms = transforms
        self.coco = COCO(annFile)
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")


@OBJECT_REGISTRY.register
class CocoFromImage(_CocoDetectionBase):
    """Coco from image dir + annFile; returns dict with img, gt_bboxes, etc."""

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id, iscrowd=False))

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        image = np.array(image)
        gt_bboxes = np.zeros((0, 4))
        gt_classes = np.zeros((0, 1))
        if len(target) > 0:
            for tar in target:
                gt_bboxes = np.append(
                    gt_bboxes, np.array([tar["bbox"]]), axis=0
                )
                gt_classes = np.append(
                    gt_classes,
                    np.array([[COCO_LABLE_TO_LABLE[tar["category_id"]]]]),
                    axis=0,
                )
        data = {
            "img": image,
            "ori_img": image,
            "gt_bboxes": gt_bboxes,
            "gt_classes": gt_classes,
            "img_id": np.array([image_id]),
            "img_shape": image.shape[0:2],
            "img_name": "%012d.jpg" % image_id,
            "layout": "hwc",
            "color_space": "rgb",
        }
        if self.transforms is not None:
            data = self.transforms(data)
        return data
