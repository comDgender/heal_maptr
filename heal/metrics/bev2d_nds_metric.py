# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import get_dist_info
from prettytable import PrettyTable

__all__ = ["BEV2DNDSMetric"]


@OBJECT_REGISTRY.register
class BEV2DNDSMetric(EvalMetric):
    """BEV 2D nuScenes-style Detection Metric.

    Metrics:
        - mean_ap
        - mATE (translation error)
        - mASE (scale error)
        - mAOE (yaw error)
        - mAVE (vel error)
        - BEV2D-NDS
    """

    def __init__(
        self,
        class_names: List[str],
        dist_ths: Tuple[float] = (0.5, 1.0, 2.0, 4.0),
        dist_th_tp: float = 2.0,
        weights=None,
        name="BEV2D-NDS",
        verbose=False,
        period=None,
        with_vel=False,
        min_recall=0.1,
        min_precision=0.1,
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.weights = weights
        if self.weights is None:
            self.weights = dict(ap=0.5, ate=0.2, ase=0.15, aoe=0.15)  # noqa
            if with_vel:
                self.weights = dict(  # noqa
                    ap=0.5, ate=0.125, ase=0.125, aoe=0.125, ave=0.125  # noqa
                )
        self.verbose = verbose
        self.rank, self.world_size = get_dist_info()
        self.period = period
        self.with_vel = with_vel
        self.min_recall = min_recall
        self.min_precision = min_precision
        super(BEV2DNDSMetric, self).__init__(name)

    def _init_states(self):
        if self.with_vel:
            self.add_state(
                "evl_err", torch.zeros(self.num_classes), dist_reduce_fx="sum"
            )

        # temporary local storage for AP computation
        self.local_preds = [[] for _ in range(self.num_classes)]
        self.local_gts = [[] for _ in range(self.num_classes)]
        self.local_tp_data = [[] for _ in range(self.num_classes)]

    def reset(self):
        super().reset()
        self.local_preds = [[] for _ in range(self.num_classes)]
        self.local_gts = [[] for _ in range(self.num_classes)]
        self.local_tp_data = [[] for _ in range(self.num_classes)]

    @staticmethod
    def yaw_diff(x, y, period=torch.pi):
        # 无朝向信息目标（翻转180°等价）: period=pi
        # 朝向信息重要（如箭头，翻转180°语义不同）: period=2*pi
        if period is None:
            return torch.abs(x - y)
        diff = (x - y + period / 2) % period - period / 2
        if diff > torch.pi:
            diff = diff - (2 * torch.pi)  # shift (pi, 2*pi] to (-pi, 0]
        diff = torch.abs(diff)
        return diff

    @staticmethod
    def scale_error(pred, gt):
        pred_wl = pred[2:4]
        gt_wl = gt[2:4]
        assert all(pred_wl > 0), "Error: pred_bbox sizes must be >0."
        assert all(gt_wl > 0), "Error: gt_bbox sizes must be >0."
        inter = torch.minimum(pred_wl, gt_wl).prod()
        union = pred_wl.prod() + gt_wl.prod() - inter
        return 1.0 - inter / union

    def cummean(self, x):
        """Compute the cumulative mean up to each position in a NaN sensitive way.

        - If all values are NaN return an array of ones.
        - If some values are NaN, accumulate arrays discording those entries.
        """
        if sum(np.isnan(x)) == len(x):
            # Is all numbers in array are NaN's.
            return np.ones(
                len(x)
            )  # If all errors are NaN set to error to 1 for all operating points.  # noqa
        else:
            # Accumulate in a nan-aware manner.
            sum_vals = np.nancumsum(
                x.astype(float)
            )  # Cumulative sum ignoring nans.
            count_vals = np.cumsum(
                ~np.isnan(x)
            )  # Number of non-nans up to each position.
            return np.divide(
                sum_vals,
                count_vals,
                out=np.zeros_like(sum_vals),
                where=count_vals != 0,
            )

    def update(self, pred, gt):
        # pred: dict with keys 'boxes':[N,5 or 7], 'scores':[N], 'labels':[N]
        # gt: dict with keys 'boxes':[M,5 or 7], 'labels':[M]
        # has_gt = len(gt['boxes']) > 0
        for cls_id in range(self.num_classes):
            # if not has_gt:
            #     continue
            p_mask = pred["labels"] == cls_id
            g_mask = gt["labels"] == cls_id
            p_boxes = pred["boxes"][p_mask]
            p_scores = pred["scores"][p_mask]
            g_boxes = gt["boxes"][g_mask]

            # 记录GT数量
            self.local_gts[cls_id].append(len(g_boxes))

            if len(p_boxes) == 0:
                # 每个 dist_th 都是空
                self.local_preds[cls_id].append([[] for _ in self.dist_ths])
                continue
            if len(g_boxes) == 0:
                self.local_preds[cls_id].append(
                    [[(0, s.item()) for s in p_scores] for _ in self.dist_ths]
                )
                continue

            sort_inds = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[sort_inds]
            p_scores = p_scores[sort_inds]

            # 对每个dist_th单独生成TP/FP
            tp_fp_per_dt = []
            for dt in self.dist_ths:
                used_gt = torch.zeros(
                    len(g_boxes), dtype=torch.bool, device=g_boxes.device
                )
                tp_fp_list = []
                for i in range(len(p_boxes)):
                    dists = torch.norm(g_boxes[:, :2] - p_boxes[i, :2], dim=1)
                    dists[used_gt] = 1e9
                    min_dist, idx = torch.min(dists, dim=0)
                    if min_dist < dt:
                        tp_fp_list.append((1, p_scores[i].item()))
                        used_gt[idx] = True

                        # 只有dist_th_tp阈值累加TP error
                        if abs(dt - self.dist_th_tp) < 1e-6:
                            self.local_tp_data[cls_id].append(
                                {
                                    "trans_err": min_dist,
                                    "scale_err": self.scale_error(
                                        p_boxes[i], g_boxes[idx]
                                    ),
                                    "orient_err": self.yaw_diff(
                                        p_boxes[i][4],
                                        g_boxes[idx][4],
                                        self.period,
                                    ),
                                    "vel_err": (
                                        np.linalg.norm(
                                            p_boxes[i][5:7] - g_boxes[idx][5:7]
                                        )
                                        if self.with_vel
                                        else 0.0
                                    ),
                                    "conf": p_scores[i].item(),
                                }
                            )
                    else:
                        tp_fp_list.append((0, p_scores[i].item()))
                tp_fp_per_dt.append(tp_fp_list)
            self.local_preds[cls_id].append(tp_fp_per_dt)

    def compute_ap_101points(self, preds_list, gts_list):
        # preds_list: list of images, each image is list of TP/FP per dist_th
        # gts_list: list of images, GT counts
        flat_preds = [x for img in preds_list for x in img]
        num_gt = sum(gts_list)
        if num_gt == 0 or len(flat_preds) == 0:
            # 返回与正常情况相同格式的 conf：长度为101的全零数组
            return 0.0, np.zeros(101, dtype=np.float32)

        # flatten TP/FP/score
        flat_preds.sort(key=lambda x: -x[1])
        tp = torch.tensor([x[0] for x in flat_preds], dtype=torch.float32)
        conf = torch.tensor([x[1] for x in flat_preds], dtype=torch.float32)
        fp = 1 - tp

        tp = tp.detach().cpu().numpy().astype(np.float32)
        fp = fp.detach().cpu().numpy().astype(np.float32)
        conf = conf.detach().cpu().numpy().astype(np.float32)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / num_gt
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

        # 101点插值
        rec_interp = np.linspace(0, 1, 101)
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf, right=0)
        rec = rec_interp
        start = round(100 * self.min_recall) + 1
        prec = prec[start:]

        prec = prec - self.min_precision
        prec[prec < 0] = 0

        ap = float(np.mean(prec)) / (1.0 - self.min_precision)
        return ap, conf

    def calc_tp_metric(self, tp_data, metric_name, conf):
        if len(tp_data) == 0:
            return 1.0
        tp_data.sort(key=lambda x: -x["conf"])
        errors = np.array(
            [x[metric_name].item() for x in tp_data], dtype=np.float32
        )
        tp_confs = np.array([x["conf"] for x in tp_data], dtype=np.float32)

        # For each match_data, we first calculate the accumulated mean.
        tmp = self.cummean(errors)
        # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)  # noqa
        tmp_errors = np.interp(conf[::-1], tp_confs[::-1], tmp[::-1])[::-1]

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(conf)[0]
        if (
            len(non_zero) == 0
        ):  # If there are no matches, all the confidence values will be zero.
            last_ind = 0
        else:
            last_ind = non_zero[-1]

        first_ind = (
            round(100 * self.min_recall) + 1
        )  # +1 to exclude the error at min recall.
        if last_ind < first_ind:
            return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.  # noqa
        else:
            return float(
                np.mean(tmp_errors[first_ind : last_ind + 1])
            )  # +1 to include error at max recall.

    def compute(self):
        # DDP all_gather
        if dist.is_initialized():
            all_preds = [None] * self.world_size
            all_gts = [None] * self.world_size
            all_tp_data = [None] * self.world_size
            dist.all_gather_object(all_preds, self.local_preds)
            dist.all_gather_object(all_gts, self.local_gts)
            dist.all_gather_object(all_tp_data, self.local_tp_data)
        else:
            all_preds = [self.local_preds]
            all_gts = [self.local_gts]
            all_tp_data = [self.local_tp_data]

        # only rank0 computes AP
        if self.rank == 0:
            combined_preds = [[] for _ in range(self.num_classes)]
            combined_gts = [[] for _ in range(self.num_classes)]
            combined_tp_data = [[] for _ in range(self.num_classes)]
            for r in range(self.world_size):
                for c in range(self.num_classes):
                    combined_preds[c].extend(all_preds[r][c])
                    combined_gts[c].extend(all_gts[r][c])
                    combined_tp_data[c].extend(all_tp_data[r][c])

            print("\n========== Calculation Metrics ==========")
            table = PrettyTable()
            table.field_names = ["Class"] + [
                f"AP@{dt:.1f}m" for dt in self.dist_ths
            ]  # noqa
            if self.with_vel:
                table.field_names += [
                    "MeanAP",
                    "ATE",
                    "ASE",
                    "AOE",
                    "AVE",
                    "#TP",
                ]
            else:
                table.field_names += ["MeanAP", "ATE", "ASE", "AOE", "#TP"]

            # compute mean AP over all thresholds
            # 存储每个类别的 conf，用于计算该类别的 TP 误差
            conf_per_class = [None] * self.num_classes
            ap_matrix = np.zeros(
                (self.num_classes, len(self.dist_ths)), dtype=np.float32
            )
            for c in range(self.num_classes):
                for d_i in range(len(self.dist_ths)):
                    # flatten每张图片第d_i个阈值的TP/FP
                    preds_flat = [
                        img_preds[d_i] for img_preds in combined_preds[c]
                    ]
                    ap_matrix[c, d_i], conf = self.compute_ap_101points(
                        preds_flat, combined_gts[c]
                    )
                    if abs(self.dist_ths[d_i] - self.dist_th_tp) < 1e-6:
                        conf_per_class[c] = conf
            mean_ap = float(ap_matrix.mean())

            ate_per_c, ase_per_c, aoe_per_c, ave_per_c = [], [], [], []
            for c, class_name in enumerate(self.class_names):
                tp_c = len(combined_tp_data[c])
                # 使用该类别自己的 conf
                conf_c = conf_per_class[c]
                if tp_c > 0 and conf_c is not None:
                    ate = self.calc_tp_metric(
                        combined_tp_data[c], "trans_err", conf_c
                    )
                    ate_per_c.append(ate)
                    ase = self.calc_tp_metric(
                        combined_tp_data[c], "scale_err", conf_c
                    )
                    ase_per_c.append(ase)
                    aoe = self.calc_tp_metric(
                        combined_tp_data[c], "orient_err", conf_c
                    )
                    aoe_per_c.append(aoe)
                    ave = (
                        self.calc_tp_metric(
                            combined_tp_data[c], "vel_err", conf_c
                        )
                        if self.with_vel
                        else None
                    )
                    ave_per_c.append(ave)
                else:
                    ate = ase = aoe = ave = 1.0
                    ate_per_c.append(1.0)
                    ase_per_c.append(1.0)
                    aoe_per_c.append(1.0)
                    ave_per_c.append(1.0)
                row = [class_name]
                row += [
                    f"{ap_matrix[c, d] * 100:.2f}"
                    for d in range(len(self.dist_ths))
                ]
                # row.append(f"{ap_matrix.mean() * 100:.2f}")
                row.append(f"{ap_matrix[c].mean() * 100:.2f}")
                if self.with_vel:
                    row.extend(
                        [
                            f"{ate:.3f}",
                            f"{ase:.3f}",
                            f"{aoe:.3f}",
                            f"{ave:.3f}",
                            int(tp_c),
                        ]
                    )
                else:
                    row.extend(
                        [f"{ate:.3f}", f"{ase:.3f}", f"{aoe:.3f}", int(tp_c)]
                    )
                table.add_row(row)
            print("\n========== BEV2D Per-Class Error Metrics ==========")
            print(table)

            mATE = float(np.nanmean(ate_per_c))
            mASE = float(np.nanmean(ase_per_c))
            mAOE = float(np.nanmean(aoe_per_c))
            if self.with_vel:
                mAVE = float(np.nanmean(ave_per_c))
                NDS = (
                    self.weights["ap"] * mean_ap
                    + self.weights["ate"] * max(0.0, 1 - mATE)
                    + self.weights["ase"] * max(0.0, 1 - mASE)
                    + self.weights["aoe"] * max(0.0, 1 - mAOE)
                    + self.weights["ave"] * max(0.0, 1 - mAVE)
                )
                if not self.verbose:
                    return NDS
                else:
                    res_dict = dict(  # noqa
                        mean_ap=mean_ap,
                        mATE=mATE,
                        mASE=mASE,
                        mAOE=mAOE,
                        mAVE=mAVE,
                        NDS=NDS,
                    )
                    return res_dict
            else:
                NDS = (
                    self.weights["ap"] * mean_ap
                    + self.weights["ate"] * max(0.0, 1 - mATE)
                    + self.weights["ase"] * max(0.0, 1 - mASE)
                    + self.weights["aoe"] * max(0.0, 1 - mAOE)
                )
                if not self.verbose:
                    return NDS
                else:
                    res_dict = dict(  # noqa
                        mean_ap=mean_ap,
                        mATE=mATE,
                        mASE=mASE,
                        mAOE=mAOE,
                        NDS=NDS,
                    )
                    return res_dict

        else:
            return 0.0  # other ranks return 0

    def get(self) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
        """Get current evaluation result.

        To skip the synchronization among devices, please override this method
        and calculate results without calling `self.compute()`.

        Returns:
           names: Name of the metrics.
           values: Value of the evaluations.
        """

        values = self.compute()
        if isinstance(values, dict):
            res_names = []
            res_values = []
            for name, value in values.items():
                res_names.append(name)
                res_values.append(value)
            return res_names, res_values
        else:
            return self.name, values
