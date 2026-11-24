# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import datetime
import itertools
import json
import logging
import os
import sys
from io import StringIO
from os import path as osp
from typing import Dict, Optional

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None

from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
from hat_engine.utils.package_helper import require_packages

__all__ = ["COCODetectionMetric"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class COCODetectionMetric(EvalMetric):
    """Evaluation in COCO protocol (mAP, etc.)."""

    @require_packages("pycocotools")
    def __init__(
        self,
        ann_file: str,
        val_interval: int = 1,
        name: str = "COCOMeanAP",
        save_prefix: str = "./WORKSPACE/results",
        adas_eval_task: Optional[str] = None,
        use_time: bool = True,
        cleanup: bool = False,
        warn_without_compute: bool = False,
    ):
        super().__init__(name, warn_without_compute=warn_without_compute)
        self.cleanup = cleanup
        self.coco = COCO(ann_file)
        self._img_ids = sorted(self.coco.getImgIds())
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories = sorted(self.categories, key=lambda x: x["id"])
        self.iter = 0
        self.val_interval = val_interval
        self.save_prefix = save_prefix
        self.use_time = use_time
        self.adas_eval_task = adas_eval_task
        try:
            os.makedirs(osp.expanduser(self.save_prefix))
        except Exception:
            pass
        self._filename = None

    def _init_states(self):
        self._results = []
        self._names, self._values = ["mAP"], [0.0]

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self._results)
        return global_output

    def __del__(self):
        if self.cleanup and getattr(self, "_filename", None):
            try:
                os.remove(self._filename)
            except (IOError, TypeError):
                pass

    def reset(self):
        self._results = []
        self._names, self._values = ["mAP"], [0.0]

    @rank_zero_only
    def _get(self, predictions):
        def _get_thr_ind(coco_eval, thr):
            ind = np.where(
                (coco_eval.params.iouThrs > thr - 1e-5)
                & (coco_eval.params.iouThrs < thr + 1e-5)
            )[0][0]
            return ind

        if self.use_time:
            t = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        else:
            t = ""
        self._filename = osp.abspath(
            osp.join(osp.expanduser(self.save_prefix), t + ".json")
        )
        with open(self._filename, "w") as f:
            json.dump(predictions, f)

        pred = self.coco.loadRes(predictions)
        gt = self.coco
        coco_eval = COCOeval(gt, pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        precision = coco_eval.eval["precision"][
            ind_lo : (ind_hi + 1), :, :, 0, 2
        ]
        ap_default = np.mean(precision[precision > -1])
        self._names.append("\n~~~~ Summary metrics ~~~~\n")
        _stdout = sys.stdout
        sys.stdout = StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout
        self._values.append(str(coco_summary).strip())
        for cls_ind, cls_name in enumerate(self.categories):
            precision = coco_eval.eval["precision"][
                ind_lo : (ind_hi + 1), :, cls_ind, 0, 2
            ]
            ap = np.mean(precision[precision > -1])
            self._names.append("\n" + cls_name["name"])
            self._values.append("{:.1f}".format(100 * ap))
        self._names.append(
            "\n~~~~ MeanAP @ IoU=[{:.2f},{:.2f}] ~~~~\n".format(
                IoU_lo_thresh, IoU_hi_thresh
            )
        )
        self._values.append("{:.1f}".format(100 * ap_default))
        self._names.append("mAP")
        self._values.append(100 * ap_default)
        if self.adas_eval_task is not None:
            self.iter += self.val_interval
            self.save_adas_eval(save_iter=self.iter)

        log_info = ""
        for k, v in zip(self._names, self._values):
            if isinstance(v, (int, float)):
                log_info += "%s[%.4f] " % (k, v)
            else:
                log_info += "%s[%s] " % (str(k), str(v))
        logger.info(log_info)

    def get(self):
        predictions = self._gather()
        predictions = list(itertools.chain(*predictions))
        if len(predictions) == 0:
            logger.warning(
                "[COCODetectionMetric] Did not receive valid predictions."
            )
            return ["mAP"], [0.0]
        self._get(predictions)
        return self._names[-1], self._values[-1]

    def save_adas_eval(self, save_iter):
        adas_eval_results = []
        unique_image_id = []
        with open(self._filename, "r") as f:
            coco_results = json.load(f)
        for line in coco_results:
            if line["image_id"] not in unique_image_id:
                cur_result = {
                    "image_key": line["image_name"],
                    self.adas_eval_task: [
                        {
                            "bbox": line["bbox"],
                            "bbox_score": line["score"],
                        }
                    ],
                }
                adas_eval_results.append(cur_result)
                unique_image_id.append(line["image_id"])
            else:
                idx = unique_image_id.index(line["image_id"])
                adas_eval_results[idx][self.adas_eval_task].append(
                    {"bbox": line["bbox"], "bbox_score": line["score"]}
                )
        save_path = os.path.split(self._filename)[0]
        save_path = os.path.join(
            save_path, self.adas_eval_task + "_" + str(save_iter) + ".json"
        )
        with open(save_path, "w") as save_file:
            for line in adas_eval_results:
                save_file.write(json.dumps(line) + "\n")

    def update(self, output: Dict):
        dets = output["pred_bboxes"]
        for idx, det in enumerate(dets):
            det = det.cpu().numpy()
            pred_label = det[:, -1]
            pred_score = det[:, -2]
            pred_bbox = det[:, 0:4].copy()
            if "scale_factor" in output:
                pred_bbox = (
                    pred_bbox / output["scale_factor"][idx].cpu().numpy()
                )
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :].astype(np.float64)
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred].astype(np.float64)
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                category_id = self.coco.getCatIds()[label]
                bbox[2:4] -= bbox[:2]
                self._results.append(
                    {
                        "image_name": output["img_name"][idx],
                        "image_id": output["img_id"][idx][0].item(),
                        "category_id": category_id,
                        "bbox": bbox.tolist(),
                        "score": score,
                    }
                )
