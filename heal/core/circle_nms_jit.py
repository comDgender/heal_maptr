# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List

import numpy as np

from heal.core.numba_lazy import lazy_jit


@lazy_jit(nopython=True)
def circle_nms(dets: np.ndarray, thresh: float) -> List[int]:
    """Sort by NMS according to the score of box.

    Args:
        dets:(shape=[N, >=7]) the det results.
        thresh: score thresh.

    Returns:
        keep: the sorted index list.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    order = dets[:, 2].argsort()[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros(ndets, dtype=np.bool_)
    keep = []
    for rank in range(ndets):
        i = order[rank]
        if suppressed[i]:
            continue
        keep.append(i)
        for later in range(rank + 1, ndets):
            j = order[later]
            if suppressed[j]:
                continue
            dx = x1[i] - x1[j]
            dy = y1[i] - y1[j]
            if dx * dx + dy * dy <= thresh:
                suppressed[j] = True
    return keep
